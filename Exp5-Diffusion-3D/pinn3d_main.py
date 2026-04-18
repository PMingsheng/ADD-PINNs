import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import PIMOE3DModelConfig
from data_3d import sample_uniform_fit_3d, sample_xyz_interior
from model import PartitionPINN3D
from pinn3d_config import (
    PINN3DDataConfig,
    PINN3DEvalConfig,
    PINN3DModelConfig,
    PINN3DOutputConfig,
    TrainBlock,
    PINN3DTrainConfig,
)
from pinn3d_loss import compute_pinn3d_loss
from pinn3d_model import SinglePINN3D
from problem_3d import alpha_piecewise, beta_piecewise, exact_solution, grad_beta_piecewise, source_term_piecewise
from utils import set_seed


C1 = (0.4, 0.5, 0.5)


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_loss_csv(loss_records: List[List[float]], save_path: Path) -> None:
    if not loss_records:
        return
    arr = np.asarray(loss_records, dtype=np.float64)
    np.savetxt(
        save_path,
        arr,
        delimiter=",",
        header="epoch,total,data,pde,total_raw,data_raw,pde_raw",
        comments="",
    )


def plot_loss(loss_records: List[List[float]], save_path: Path) -> None:
    if not loss_records:
        return

    arr = np.asarray(loss_records, dtype=np.float64)
    epoch = arr[:, 0]
    total = arr[:, 1]
    data = arr[:, 2]
    pde = arr[:, 3]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epoch, total, label="total")
    plt.plot(epoch, data, label="data")
    plt.plot(epoch, pde, label="pde")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("3D PINN Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def _prediction_chunks(model: SinglePINN3D, xyz: torch.Tensor, batch_size: int) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for st in range(0, xyz.shape[0], batch_size):
            ed = min(st + batch_size, xyz.shape[0])
            chunks.append(model(xyz[st:ed]))
    return torch.cat(chunks, dim=0)


def _f_pred_chunks(model: SinglePINN3D, xyz: torch.Tensor, batch_size: int) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for st in range(0, xyz.shape[0], batch_size):
        ed = min(st + batch_size, xyz.shape[0])
        xyz_chunk = xyz[st:ed].detach().clone().requires_grad_(True)

        u = model(xyz_chunk)
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=xyz_chunk,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]

        d2u_dx2 = torch.autograd.grad(
            outputs=grad_u[:, 0:1],
            inputs=xyz_chunk,
            grad_outputs=torch.ones_like(grad_u[:, 0:1]),
            create_graph=False,
            retain_graph=True,
        )[0][:, 0:1]
        d2u_dy2 = torch.autograd.grad(
            outputs=grad_u[:, 1:2],
            inputs=xyz_chunk,
            grad_outputs=torch.ones_like(grad_u[:, 1:2]),
            create_graph=False,
            retain_graph=True,
        )[0][:, 1:2]
        d2u_dz2 = torch.autograd.grad(
            outputs=grad_u[:, 2:3],
            inputs=xyz_chunk,
            grad_outputs=torch.ones_like(grad_u[:, 2:3]),
            create_graph=False,
        )[0][:, 2:3]

        lap_u = d2u_dx2 + d2u_dy2 + d2u_dz2
        beta = beta_piecewise(xyz_chunk)
        grad_beta = grad_beta_piecewise(xyz_chunk)
        alpha = alpha_piecewise(xyz_chunk)

        div_beta_grad_u = beta * lap_u + (grad_beta * grad_u).sum(dim=1, keepdim=True)
        f_pred = -div_beta_grad_u + alpha * u
        chunks.append(f_pred.detach())
    return torch.cat(chunks, dim=0)


def _rar_select_topk_by_pde_residual(
    model: SinglePINN3D,
    *,
    device: torch.device,
    interior_margin: float,
    sample_batch_size: int,
    candidate_points: int,
    topk: int,
    eval_batch_size: int,
) -> Dict[str, object]:
    if candidate_points <= 0 or topk <= 0:
        return {
            "new_points": torch.empty((0, 3), device=device),
            "selected_mean_res2": 0.0,
            "selected_max_res2": 0.0,
            "pool_mean_res2": 0.0,
            "pool_max_res2": 0.0,
            "pool_size": 0,
            "selected_size": 0,
        }

    xyz_cand = sample_xyz_interior(
        candidate_points,
        device=device,
        margin=interior_margin,
        batch_size=sample_batch_size,
    )

    was_training = model.training
    model.eval()

    with torch.no_grad():
        f_true = source_term_piecewise(xyz_cand)
    f_pred = _f_pred_chunks(model, xyz_cand, max(1, eval_batch_size))
    res2 = (f_pred - f_true).pow(2).reshape(-1)

    k = min(int(topk), int(res2.numel()))
    if k <= 0:
        new_pts = torch.empty((0, 3), device=device)
        sel_mean = 0.0
        sel_max = 0.0
    else:
        topk_vals, topk_idx = torch.topk(res2, k=k, largest=True)
        new_pts = xyz_cand[topk_idx].detach()
        sel_mean = float(topk_vals.mean().item())
        sel_max = float(topk_vals.max().item())

    pool_mean = float(res2.mean().item()) if res2.numel() > 0 else 0.0
    pool_max = float(res2.max().item()) if res2.numel() > 0 else 0.0

    if was_training:
        model.train()

    return {
        "new_points": new_pts,
        "selected_mean_res2": sel_mean,
        "selected_max_res2": sel_max,
        "pool_mean_res2": pool_mean,
        "pool_max_res2": pool_max,
        "pool_size": int(res2.numel()),
        "selected_size": int(k),
    }


def evaluate_fields(
    model: SinglePINN3D,
    eval_cfg: PINN3DEvalConfig,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    x0, x1, y0, y1, z0, z1 = eval_cfg.bbox
    n = eval_cfg.n_grid

    xg = torch.linspace(x0, x1, n, device=device)
    yg = torch.linspace(y0, y1, n, device=device)
    zg = torch.linspace(z0, z1, n, device=device)
    Xg, Yg, Zg = torch.meshgrid(xg, yg, zg, indexing="ij")
    xyz_grid = torch.stack([Xg, Yg, Zg], dim=-1).reshape(-1, 3)

    was_training = model.training
    model.eval()

    batch_size = max(1, int(eval_cfg.eval_batch_size))
    u_pred = _prediction_chunks(model, xyz_grid, batch_size)
    f_pred = _f_pred_chunks(model, xyz_grid, batch_size)

    with torch.no_grad():
        u_true = exact_solution(xyz_grid)
        f_true = source_term_piecewise(xyz_grid)

    u_residual = u_pred - u_true
    f_residual = f_pred - f_true
    u_l2 = torch.linalg.vector_norm(u_residual.reshape(-1)).item()
    f_l2 = torch.linalg.vector_norm(f_residual.reshape(-1)).item()
    u_rel_l2 = u_l2 / (torch.linalg.vector_norm(u_true.reshape(-1)).item() + 1e-12)
    f_rel_l2 = f_l2 / (torch.linalg.vector_norm(f_true.reshape(-1)).item() + 1e-12)

    if was_training:
        model.train()

    return {
        "x": xg.detach().cpu().numpy(),
        "y": yg.detach().cpu().numpy(),
        "z": zg.detach().cpu().numpy(),
        "u_pred": u_pred.detach().cpu().numpy().reshape(n, n, n),
        "u_true": u_true.detach().cpu().numpy().reshape(n, n, n),
        "u_residual": u_residual.detach().cpu().numpy().reshape(n, n, n),
        "f_pred": f_pred.detach().cpu().numpy().reshape(n, n, n),
        "f_true": f_true.detach().cpu().numpy().reshape(n, n, n),
        "f_residual": f_residual.detach().cpu().numpy().reshape(n, n, n),
        "bbox": np.asarray(eval_cfg.bbox, dtype=np.float64),
        "u_mse": np.asarray([u_residual.pow(2).mean().item()], dtype=np.float64),
        "f_mse": np.asarray([f_residual.pow(2).mean().item()], dtype=np.float64),
        "u_l2": np.asarray([u_l2], dtype=np.float64),
        "f_l2": np.asarray([f_l2], dtype=np.float64),
        "u_rel_l2": np.asarray([u_rel_l2], dtype=np.float64),
        "f_rel_l2": np.asarray([f_rel_l2], dtype=np.float64),
    }


def evaluate_label_u_l2(
    model: SinglePINN3D,
    xyz_fit: torch.Tensor,
    u_fit: torch.Tensor,
    *,
    eval_batch_size: int,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    u_pred_fit = _prediction_chunks(model, xyz_fit, max(1, int(eval_batch_size)))
    if was_training:
        model.train()

    u_res_fit = u_pred_fit - u_fit
    u_fit_l2 = torch.linalg.vector_norm(u_res_fit.reshape(-1)).item()
    u_fit_rel_l2 = u_fit_l2 / (torch.linalg.vector_norm(u_fit.reshape(-1)).item() + 1e-12)
    return {"u_fit_l2": float(u_fit_l2), "u_fit_rel_l2": float(u_fit_rel_l2)}


def save_residual_slices(fields: Dict[str, np.ndarray], save_path: Path) -> None:
    x = fields["x"]
    y = fields["y"]
    z = fields["z"]
    u_res = fields["u_residual"]

    mid = u_res.shape[0] // 2
    yz = u_res[mid, :, :]
    xz = u_res[:, mid, :]
    xy = u_res[:, :, mid]

    vmax = max(float(np.max(np.abs(yz))), float(np.max(np.abs(xz))), float(np.max(np.abs(xy))))
    if vmax < 1e-14:
        vmax = 1e-14

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), dpi=220)

    im0 = axes[0].imshow(
        yz.T,
        origin="lower",
        extent=[float(y.min()), float(y.max()), float(z.min()), float(z.max())],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    axes[0].set_title(f"u residual @ x={x[mid]:.3f}")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("z")

    axes[1].imshow(
        xz.T,
        origin="lower",
        extent=[float(x.min()), float(x.max()), float(z.min()), float(z.max())],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    axes[1].set_title(f"u residual @ y={y[mid]:.3f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")

    axes[2].imshow(
        xy.T,
        origin="lower",
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    axes[2].set_title(f"u residual @ z={z[mid]:.3f}")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.86)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_uf_snapshot(fields: Dict[str, np.ndarray], snapshot_dir: Path, epoch: int) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        snapshot_dir / "u_f_fields.npz",
        epoch=np.asarray([epoch], dtype=np.int64),
        x=fields["x"],
        y=fields["y"],
        z=fields["z"],
        u_true=fields["u_true"],
        u_pred=fields["u_pred"],
        u_residual=fields["u_residual"],
        f_true=fields["f_true"],
        f_pred=fields["f_pred"],
        f_residual=fields["f_residual"],
    )


def _nearest_index(arr: np.ndarray, value: float) -> int:
    return int(np.abs(arr - float(value)).argmin())


def _extract_plane(field: np.ndarray, axis: str, idx: int):
    if axis == "x":
        return field[idx, :, :].T
    if axis == "y":
        return field[:, idx, :].T
    if axis == "z":
        return field[:, :, idx].T
    raise ValueError(f"unknown axis: {axis}")


def _plane_extent(axis: str, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    if axis == "x":
        return [float(y.min()), float(y.max()), float(z.min()), float(z.max())], "y", "z"
    if axis == "y":
        return [float(x.min()), float(x.max()), float(z.min()), float(z.max())], "x", "z"
    if axis == "z":
        return [float(x.min()), float(x.max()), float(y.min()), float(y.max())], "x", "y"
    raise ValueError(f"unknown axis: {axis}")


def _plane_specs(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    specs = [
        ("x-mid", "x", len(x) // 2, float(x[len(x) // 2])),
        ("x-c1", "x", _nearest_index(x, C1[0]), float(x[_nearest_index(x, C1[0])])),
        ("y-mid", "y", len(y) // 2, float(y[len(y) // 2])),
        ("y-c1", "y", _nearest_index(y, C1[1]), float(y[_nearest_index(y, C1[1])])),
        ("z-mid", "z", len(z) // 2, float(z[len(z) // 2])),
        ("z-c1", "z", _nearest_index(z, C1[2]), float(z[_nearest_index(z, C1[2])])),
    ]
    return specs


def save_true_pred_residual_multiplane(
    true_field: np.ndarray,
    pred_field: np.ndarray,
    residual_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    title_prefix: str,
    save_path: Path,
) -> None:
    specs = _plane_specs(x, y, z)
    n_rows = len(specs)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12.6, 2.4 * n_rows), dpi=220, squeeze=False)

    tp_min = float(min(true_field.min(), pred_field.min()))
    tp_max = float(max(true_field.max(), pred_field.max()))
    r_abs = float(np.max(np.abs(residual_field)))
    if r_abs < 1e-14:
        r_abs = 1e-14

    for row, (_, axis, idx, coord) in enumerate(specs):
        true_2d = _extract_plane(true_field, axis, idx)
        pred_2d = _extract_plane(pred_field, axis, idx)
        res_2d = _extract_plane(residual_field, axis, idx)
        extent, xlab, ylab = _plane_extent(axis, x, y, z)

        im0 = axes[row, 0].imshow(
            true_2d,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=tp_min,
            vmax=tp_max,
            aspect="auto",
        )
        axes[row, 0].set_title(f"true | {axis}={coord:.3f}")
        axes[row, 0].set_xlabel(xlab)
        axes[row, 0].set_ylabel(ylab)
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(
            pred_2d,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=tp_min,
            vmax=tp_max,
            aspect="auto",
        )
        axes[row, 1].set_title(f"pred | {axis}={coord:.3f}")
        axes[row, 1].set_xlabel(xlab)
        axes[row, 1].set_ylabel(ylab)
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        im2 = axes[row, 2].imshow(
            res_2d,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-r_abs,
            vmax=r_abs,
            aspect="auto",
        )
        axes[row, 2].set_title(f"residual | {axis}={coord:.3f}")
        axes[row, 2].set_xlabel(xlab)
        axes[row, 2].set_ylabel(ylab)
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    fig.suptitle(title_prefix)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_periodic_visuals(fields: Dict[str, np.ndarray], output_dir: Path, epoch: int, out_cfg: PINN3DOutputConfig) -> None:
    u_dir = output_dir / out_cfg.u_heatmap_dir
    f_dir = output_dir / out_cfg.f_heatmap_dir
    u_dir.mkdir(parents=True, exist_ok=True)
    f_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{epoch:08d}"
    u_png = u_dir / f"u_heatmap_{tag}.png"
    u_npz = u_dir / f"u_heatmap_{tag}.npz"
    f_png = f_dir / f"f_heatmap_{tag}.png"
    f_npz = f_dir / f"f_heatmap_{tag}.npz"

    save_true_pred_residual_multiplane(
        fields["u_true"],
        fields["u_pred"],
        fields["u_residual"],
        fields["x"],
        fields["y"],
        fields["z"],
        title_prefix=f"u true/pred/residual @ epoch {epoch}",
        save_path=u_png,
    )
    save_true_pred_residual_multiplane(
        fields["f_true"],
        fields["f_pred"],
        fields["f_residual"],
        fields["x"],
        fields["y"],
        fields["z"],
        title_prefix=f"f true/pred/residual @ epoch {epoch}",
        save_path=f_png,
    )

    np.savez_compressed(
        u_npz,
        epoch=np.asarray([epoch], dtype=np.int64),
        x=fields["x"],
        y=fields["y"],
        z=fields["z"],
        u_true=fields["u_true"],
        u_pred=fields["u_pred"],
        u_residual=fields["u_residual"],
    )
    np.savez_compressed(
        f_npz,
        epoch=np.asarray([epoch], dtype=np.int64),
        x=fields["x"],
        y=fields["y"],
        z=fields["z"],
        f_true=fields["f_true"],
        f_pred=fields["f_pred"],
        f_residual=fields["f_residual"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D PINN for single-sphere (c1) interface problem")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--eval-grid", type=int, default=None)
    return parser.parse_args()


def _configure_output_dirs(project_root: Path, out_cfg: PINN3DOutputConfig):
    output_root = out_cfg.resolve_output_dir(project_root)
    viz_dir = output_root / "viz"
    data_output_dir = output_root / "data_output"
    viz_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir.mkdir(parents=True, exist_ok=True)
    return output_root, viz_dir, data_output_dir


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    data_cfg = PINN3DDataConfig()
    model_cfg = PINN3DModelConfig()
    train_cfg = PINN3DTrainConfig()
    eval_cfg = PINN3DEvalConfig()
    out_cfg = PINN3DOutputConfig()

    if args.eval_grid is not None:
        eval_cfg.n_grid = int(args.eval_grid)
    if args.output_dir is not None:
        out_cfg.output_dir = str(args.output_dir)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir, viz_dir, data_output_dir = _configure_output_dirs(project_root, out_cfg)
    print(f"[Output] writing results to: {out_dir}")

    set_seed(train_cfg.seed)

    xyz_fit, u_fit = sample_uniform_fit_3d(
        nx=data_cfg.nx,
        ny=data_cfg.ny,
        nz=data_cfg.nz,
        device=device,
        drop_boundary=True,
    )
    xyz_int = sample_xyz_interior(
        train_cfg.interior_points,
        device=device,
        margin=train_cfg.interior_margin,
        batch_size=train_cfg.sample_batch_size,
    )

    blocks = list(train_cfg.blocks)
    if not blocks:
        raise ValueError("train_cfg.blocks cannot be empty")
    if args.epochs is not None:
        blocks = [TrainBlock(epochs=int(args.epochs), lr=blocks[0].lr)]
    total_epochs = int(sum(max(int(b.epochs), 0) for b in blocks))
    if total_epochs <= 0:
        raise ValueError("total epochs must be positive")

    model = SinglePINN3D(width=model_cfg.width, depth=model_cfg.depth).to(device)
    pimoe_cfg = PIMOE3DModelConfig()
    pimoe_params = _count_parameters(PartitionPINN3D(width=pimoe_cfg.width, depth=pimoe_cfg.depth))
    pinn_params = _count_parameters(model)
    print(f"[Model] ADD-PINNs3D params={pimoe_params:,d}, PINN3D params={pinn_params:,d}")
    print(
        f"[Budget] total_epochs={total_epochs:,d}, "
        f"fit_points={xyz_fit.shape[0]:,d}, interior_points={xyz_int.shape[0]:,d}, "
        f"rar={train_cfg.rar_topk:,d}/{train_cfg.rar_candidate_points:,d} every {train_cfg.rar_every}"
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(blocks[0].lr))
    snapshot_root = out_dir / out_cfg.snapshot_dir
    snapshot_root.mkdir(parents=True, exist_ok=True)

    loss_records: List[List[float]] = []
    last_eval_fields: Dict[str, np.ndarray] = {}
    last_eval_epoch = -1
    block_idx = 0
    epochs_in_block = 0
    for ep in range(1, total_epochs + 1):
        if epochs_in_block >= blocks[block_idx].epochs and block_idx + 1 < len(blocks):
            block_idx += 1
            epochs_in_block = 0
            new_lr = float(blocks[block_idx].lr)
            for group in opt.param_groups:
                group["lr"] = new_lr
            print(f"[LR] switch to block {block_idx + 1}/{len(blocks)} lr={new_lr:.3e} at epoch {ep}")

        total, weighted, raw = compute_pinn3d_loss(
            model,
            xyz_int,
            xyz_fit=xyz_fit,
            u_fit=u_fit,
            lam_data=train_cfg.lam_data,
            lam_pde=train_cfg.lam_pde,
        )

        opt.zero_grad()
        total.backward()
        opt.step()

        if ep % train_cfg.print_every == 0 or ep == 1:
            print(
                f"E{ep:>6d} total={weighted['total'].item():.3e} "
                f"data={weighted['data'].item():.3e} pde={weighted['pde'].item():.3e}"
            )

        if ep % train_cfg.record_every == 0 or ep == total_epochs:
            loss_records.append(
                [
                    float(ep),
                    float(weighted["total"].detach().cpu()),
                    float(weighted["data"].detach().cpu()),
                    float(weighted["pde"].detach().cpu()),
                    float(raw["total"].detach().cpu()),
                    float(raw["data"].detach().cpu()),
                    float(raw["pde"].detach().cpu()),
                ]
            )

        if out_cfg.snapshot_every > 0 and ep % out_cfg.snapshot_every == 0:
            fields_ep = evaluate_fields(model, eval_cfg, device)
            fit_l2_ep = evaluate_label_u_l2(
                model,
                xyz_fit,
                u_fit,
                eval_batch_size=eval_cfg.eval_batch_size,
            )
            snap_dir = snapshot_root / f"epoch_{ep:07d}"
            save_uf_snapshot(fields_ep, snap_dir, ep)
            save_periodic_visuals(fields_ep, out_dir, ep, out_cfg)
            last_eval_fields = fields_ep
            last_eval_epoch = ep
            print(f"[Save] u/f snapshot: {snap_dir / 'u_f_fields.npz'}")
            print(
                f"[L2] epoch {ep:>6d} | "
                f"u_l2={fields_ep['u_l2'][0]:.3e} (rel={fields_ep['u_rel_l2'][0]:.3e})  "
                f"f_l2={fields_ep['f_l2'][0]:.3e} (rel={fields_ep['f_rel_l2'][0]:.3e})"
            )
            print(
                f"[L2-fit] epoch {ep:>6d} | "
                f"u_l2={fit_l2_ep['u_fit_l2']:.3e} (rel={fit_l2_ep['u_fit_rel_l2']:.3e})"
            )

        if (
            train_cfg.rar_every > 0
            and ep >= train_cfg.rar_start_epoch
            and ep % train_cfg.rar_every == 0
            and train_cfg.rar_topk > 0
            and train_cfg.rar_candidate_points > 0
        ):
            rar = _rar_select_topk_by_pde_residual(
                model,
                device=device,
                interior_margin=train_cfg.interior_margin,
                sample_batch_size=train_cfg.sample_batch_size,
                candidate_points=train_cfg.rar_candidate_points,
                topk=train_cfg.rar_topk,
                eval_batch_size=eval_cfg.eval_batch_size,
            )
            new_pts = rar["new_points"]
            if new_pts.shape[0] > 0:
                xyz_int = torch.cat([xyz_int, new_pts], dim=0)
                print(
                    f"[RAR] epoch {ep:>6d} added {new_pts.shape[0]} points "
                    f"(pool={rar['pool_size']}, total_int={xyz_int.shape[0]}), "
                    f"res2 pool mean/max={rar['pool_mean_res2']:.3e}/{rar['pool_max_res2']:.3e}, "
                    f"selected mean/max={rar['selected_mean_res2']:.3e}/{rar['selected_max_res2']:.3e}"
                )

        epochs_in_block += 1

    loss_csv = out_dir / out_cfg.loss_csv_name
    write_loss_csv(loss_records, loss_csv)
    plot_loss(loss_records, viz_dir / out_cfg.loss_png_name)

    if last_eval_epoch != total_epochs:
        fields = evaluate_fields(model, eval_cfg, device)
        final_snap_dir = snapshot_root / f"epoch_{total_epochs:07d}"
        save_uf_snapshot(fields, final_snap_dir, total_epochs)
        save_periodic_visuals(fields, out_dir, total_epochs, out_cfg)
        last_eval_fields = fields
        last_eval_epoch = total_epochs
    else:
        fields = last_eval_fields
    fit_l2_final = evaluate_label_u_l2(
        model,
        xyz_fit,
        u_fit,
        eval_batch_size=eval_cfg.eval_batch_size,
    )
    fields["u_fit_l2"] = np.asarray([fit_l2_final["u_fit_l2"]], dtype=np.float64)
    fields["u_fit_rel_l2"] = np.asarray([fit_l2_final["u_fit_rel_l2"]], dtype=np.float64)
    field_path = data_output_dir / out_cfg.field_npz_name
    np.savez_compressed(field_path, **fields)
    save_residual_slices(fields, viz_dir / out_cfg.slices_png_name)

    model_path = out_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    if loss_csv.exists():
        shutil.copy2(loss_csv, data_output_dir / out_cfg.loss_csv_name)

    print(f"[Done] loss csv: {loss_csv}")
    print(f"[Done] final field npz: {field_path}")
    print(f"[Done] model weights: {model_path}")
    print(f"[Metric] u_mse={fields['u_mse'][0]:.3e}, f_mse={fields['f_mse'][0]:.3e}")
    print(
        f"[Metric] u_l2={fields['u_l2'][0]:.3e} (rel={fields['u_rel_l2'][0]:.3e}), "
        f"f_l2={fields['f_l2'][0]:.3e} (rel={fields['f_rel_l2'][0]:.3e})"
    )
    print(
        f"[Metric-fit] u_l2={fields['u_fit_l2'][0]:.3e} (rel={fields['u_fit_rel_l2'][0]:.3e})"
    )


if __name__ == "__main__":
    main()
