import argparse
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from data_3d import sample_uniform_fit_3d, sample_xyz_interior
from config import (
    PIMOE3DDataConfig,
    PIMOE3DEvalConfig,
    PIMOE3DModelConfig,
    PIMOE3DOutputConfig,
    PIMOE3DRuntimeConfig,
    TrainBlock,
    PIMOE3DTrainConfig,
)
from level_set_3d import evolve_phi_by_residual, predict_phi_next_by_residual
from loss import compute_pimoe3d_loss
from model import PartitionPINN3D
from plot_uf_slice_with_phi import save_uf_slice_with_phi_plot
from problem_3d import (
    ALPHA_INSIDE,
    ALPHA_OUTSIDE,
    BETA_INSIDE,
    C1,
    beta_outside,
    exact_solution,
    grad_beta_outside,
    phi_signed_c1_sphere,
    source_region_inside,
    source_region_outside,
    source_term_piecewise,
)
from utils import set_seed


def write_loss_csv(loss_records: List[List[float]], save_path: Path) -> None:
    if not loss_records:
        return
    arr = np.asarray(loss_records, dtype=np.float64)
    np.savetxt(
        save_path,
        arr,
        delimiter=",",
        header=(
            "epoch,total,data,pde,interface,eik,volume,surface,"
            "total_raw,data_raw,pde_raw,interface_raw,eik_raw,volume_raw,surface_raw"
        ),
        comments="",
    )


def plot_loss(loss_records: List[List[float]], save_path: Path) -> None:
    if not loss_records:
        return

    arr = np.asarray(loss_records, dtype=np.float64)
    ep = arr[:, 0]

    plt.figure(figsize=(7.4, 4.8))
    labels = ["total", "data", "pde", "interface", "eik", "volume", "surface"]
    for i, name in enumerate(labels, start=1):
        plt.plot(ep, arr[:, i], label=name)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("ADD-PINNs 3D Training Loss")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def _format_duration(seconds: float) -> str:
    sec = int(max(seconds, 0.0))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _capture_rng_state() -> Dict[str, object]:
    state: Dict[str, object] = {
        "torch": torch.get_rng_state().detach().cpu(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [s.detach().cpu() for s in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state: Dict[str, object]) -> None:
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8).cpu())
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = [torch.as_tensor(s, dtype=torch.uint8).cpu() for s in state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def _resolve_block_position(epoch_done: int, blocks: List[TrainBlock]) -> Tuple[int, int]:
    remaining = int(epoch_done)
    for idx, block in enumerate(blocks):
        block_epochs = max(int(block.epochs), 0)
        if remaining <= block_epochs:
            return idx, remaining
        remaining -= block_epochs
    last_idx = max(len(blocks) - 1, 0)
    return last_idx, max(int(blocks[last_idx].epochs), 0)


def _set_optimizer_lrs(
    *,
    block_idx: int,
    blocks: List[TrainBlock],
    train_cfg: PIMOE3DTrainConfig,
    opt,
    opt_main_phi,
    opt_phi,
) -> Tuple[float, float]:
    lr = float(blocks[block_idx].lr)
    phi_lr = float(train_cfg.phi_lr) if train_cfg.phi_lr is not None else float(train_cfg.phi_lr_scale * lr)
    for optimizer, new_lr in ((opt, lr), (opt_main_phi, lr), (opt_phi, phi_lr)):
        for group in optimizer.param_groups:
            group["lr"] = new_lr
    return lr, phi_lr


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model,
    opt,
    opt_main_phi,
    opt_phi,
    xyz_int: torch.Tensor,
    records: List[List[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "opt_main_phi": opt_main_phi.state_dict(),
        "opt_phi": opt_phi.state_dict(),
        "xyz_int": xyz_int.detach().cpu(),
        "records": [list(row) for row in records],
        "rng_state": _capture_rng_state(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    device: torch.device,
    model,
    opt,
    opt_main_phi,
    opt_phi,
) -> Dict[str, object]:
    data = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(data["model"])
    opt.load_state_dict(data["opt"])
    opt_main_phi.load_state_dict(data["opt_main_phi"])
    opt_phi.load_state_dict(data["opt_phi"])
    xyz_int = torch.as_tensor(data["xyz_int"], dtype=torch.float32, device=device)
    records = [list(row) for row in data.get("records", [])]
    _restore_rng_state(data.get("rng_state", {}))
    return {
        "epoch": int(data["epoch"]),
        "xyz_int": xyz_int,
        "records": records,
    }


def _mask_u(phi: torch.Tensor, u1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
    return torch.where(phi >= 0.0, u1, u2)


def _grad_and_lap(u: torch.Tensor, xyz: torch.Tensor, *, retain_graph: bool):
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=xyz,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2u_dx2 = torch.autograd.grad(
        outputs=grad_u[:, 0:1],
        inputs=xyz,
        grad_outputs=torch.ones_like(grad_u[:, 0:1]),
        create_graph=False,
        retain_graph=True,
    )[0][:, 0:1]
    d2u_dy2 = torch.autograd.grad(
        outputs=grad_u[:, 1:2],
        inputs=xyz,
        grad_outputs=torch.ones_like(grad_u[:, 1:2]),
        create_graph=False,
        retain_graph=True,
    )[0][:, 1:2]
    d2u_dz2 = torch.autograd.grad(
        outputs=grad_u[:, 2:3],
        inputs=xyz,
        grad_outputs=torch.ones_like(grad_u[:, 2:3]),
        create_graph=False,
        retain_graph=retain_graph,
    )[0][:, 2:3]
    lap_u = d2u_dx2 + d2u_dy2 + d2u_dz2
    return grad_u, lap_u


def _predict_chunks(model: PartitionPINN3D, xyz: torch.Tensor, batch_size: int):
    phi_chunks = []
    u1_chunks = []
    u2_chunks = []
    with torch.no_grad():
        for st in range(0, xyz.shape[0], batch_size):
            ed = min(st + batch_size, xyz.shape[0])
            phi, u1, u2 = model(xyz[st:ed])
            phi_chunks.append(phi)
            u1_chunks.append(u1)
            u2_chunks.append(u2)
    phi_all = torch.cat(phi_chunks, dim=0)
    u1_all = torch.cat(u1_chunks, dim=0)
    u2_all = torch.cat(u2_chunks, dim=0)
    u_masked = _mask_u(phi_all, u1_all, u2_all)
    return phi_all, u1_all, u2_all, u_masked


def _f_pred_chunks(model: PartitionPINN3D, xyz: torch.Tensor, batch_size: int) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for st in range(0, xyz.shape[0], batch_size):
        ed = min(st + batch_size, xyz.shape[0])
        xyz_chunk = xyz[st:ed].detach().clone().requires_grad_(True)

        phi, u1, u2 = model(xyz_chunk)
        grad_u1, lap_u1 = _grad_and_lap(u1, xyz_chunk, retain_graph=True)
        grad_u2, lap_u2 = _grad_and_lap(u2, xyz_chunk, retain_graph=False)

        f_in = -BETA_INSIDE * lap_u1 + ALPHA_INSIDE * u1

        beta_out = beta_outside(xyz_chunk)
        grad_beta_out = grad_beta_outside(xyz_chunk)
        div_beta_grad_u2 = beta_out * lap_u2 + (grad_beta_out * grad_u2).sum(dim=1, keepdim=True)
        f_out = -div_beta_grad_u2 + ALPHA_OUTSIDE * u2

        f_pred = torch.where(phi >= 0.0, f_in, f_out)
        chunks.append(f_pred.detach())
    return torch.cat(chunks, dim=0)


def _rar_select_topk_by_pde_residual(
    model: PartitionPINN3D,
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


def evaluate_fields(model: PartitionPINN3D, eval_cfg: PIMOE3DEvalConfig, device: torch.device) -> Dict[str, np.ndarray]:
    n = eval_cfg.n_grid
    x = torch.linspace(0.0, 1.0, n, device=device)
    y = torch.linspace(0.0, 1.0, n, device=device)
    z = torch.linspace(0.0, 1.0, n, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    xyz = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)

    was_training = model.training
    model.eval()

    batch_size = max(1, eval_cfg.eval_batch_size)
    phi_pred, u1_pred, u2_pred, u_pred = _predict_chunks(model, xyz, batch_size)
    f_pred = _f_pred_chunks(model, xyz, batch_size)

    with torch.no_grad():
        u_true = exact_solution(xyz)
        phi_true = phi_signed_c1_sphere(xyz)
        f_true_in = source_region_inside(xyz)
        f_true_out = source_region_outside(xyz)
        f_true = torch.where(phi_pred >= 0.0, f_true_in, f_true_out)

    if was_training:
        model.train()

    u_res = u_pred - u_true
    phi_res = phi_pred - phi_true
    f_res = f_pred - f_true
    u_l2 = torch.linalg.vector_norm(u_res.reshape(-1)).item()
    f_l2 = torch.linalg.vector_norm(f_res.reshape(-1)).item()
    u_rel_l2 = u_l2 / (torch.linalg.vector_norm(u_true.reshape(-1)).item() + 1e-12)
    f_rel_l2 = f_l2 / (torch.linalg.vector_norm(f_true.reshape(-1)).item() + 1e-12)

    return {
        "x": x.detach().cpu().numpy(),
        "y": y.detach().cpu().numpy(),
        "z": z.detach().cpu().numpy(),
        "phi_pred": phi_pred.detach().cpu().numpy().reshape(n, n, n),
        "phi_true": phi_true.detach().cpu().numpy().reshape(n, n, n),
        "phi_residual": phi_res.detach().cpu().numpy().reshape(n, n, n),
        "u1_pred": u1_pred.detach().cpu().numpy().reshape(n, n, n),
        "u2_pred": u2_pred.detach().cpu().numpy().reshape(n, n, n),
        "u_pred": u_pred.detach().cpu().numpy().reshape(n, n, n),
        "u_true": u_true.detach().cpu().numpy().reshape(n, n, n),
        "u_residual": u_res.detach().cpu().numpy().reshape(n, n, n),
        "f_pred": f_pred.detach().cpu().numpy().reshape(n, n, n),
        "f_true": f_true.detach().cpu().numpy().reshape(n, n, n),
        "f_residual": f_res.detach().cpu().numpy().reshape(n, n, n),
        "u_mse": np.asarray([u_res.pow(2).mean().item()], dtype=np.float64),
        "phi_mse": np.asarray([phi_res.pow(2).mean().item()], dtype=np.float64),
        "f_mse": np.asarray([f_res.pow(2).mean().item()], dtype=np.float64),
        "u_l2": np.asarray([u_l2], dtype=np.float64),
        "f_l2": np.asarray([f_l2], dtype=np.float64),
        "u_rel_l2": np.asarray([u_rel_l2], dtype=np.float64),
        "f_rel_l2": np.asarray([f_rel_l2], dtype=np.float64),
    }


def evaluate_label_u_l2(
    model: PartitionPINN3D,
    xyz_fit: torch.Tensor,
    u_fit: torch.Tensor,
    *,
    eval_batch_size: int,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    _, _, _, u_pred_fit = _predict_chunks(model, xyz_fit, max(1, int(eval_batch_size)))
    if was_training:
        model.train()

    u_res_fit = u_pred_fit - u_fit
    u_fit_l2 = torch.linalg.vector_norm(u_res_fit.reshape(-1)).item()
    u_fit_rel_l2 = u_fit_l2 / (torch.linalg.vector_norm(u_fit.reshape(-1)).item() + 1e-12)
    return {"u_fit_l2": float(u_fit_l2), "u_fit_rel_l2": float(u_fit_rel_l2)}


def _save_three_slices(field: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, title_prefix: str, save_path: Path, cmap: str = "RdBu_r") -> None:
    mid = field.shape[0] // 2
    yz = field[mid, :, :]
    xz = field[:, mid, :]
    xy = field[:, :, mid]

    vmax = max(float(np.max(np.abs(yz))), float(np.max(np.abs(xz))), float(np.max(np.abs(xy))))
    if vmax < 1e-14:
        vmax = 1e-14

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), dpi=220)

    im = axes[0].imshow(
        yz.T,
        origin="lower",
        extent=[float(y.min()), float(y.max()), float(z.min()), float(z.max())],
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    axes[0].set_title(f"{title_prefix} @ x={x[mid]:.3f}")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("z")

    axes[1].imshow(
        xz.T,
        origin="lower",
        extent=[float(x.min()), float(x.max()), float(z.min()), float(z.max())],
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    axes[1].set_title(f"{title_prefix} @ y={y[mid]:.3f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")

    axes[2].imshow(
        xy.T,
        origin="lower",
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    axes[2].set_title(f"{title_prefix} @ z={z[mid]:.3f}")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def save_uf_snapshot(fields: Dict[str, np.ndarray], snapshot_dir: Path, epoch: int) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        snapshot_dir / "u_f_fields.npz",
        epoch=np.asarray([epoch], dtype=np.int64),
        x=fields["x"],
        y=fields["y"],
        z=fields["z"],
        u1_pred=fields["u1_pred"],
        u2_pred=fields["u2_pred"],
        u_true=fields["u_true"],
        u_pred=fields["u_pred"],
        u_residual=fields["u_residual"],
        f_true=fields["f_true"],
        f_pred=fields["f_pred"],
        f_residual=fields["f_residual"],
        phi_true=fields["phi_true"],
        phi_pred=fields["phi_pred"],
        phi_residual=fields["phi_residual"],
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


def _plane_coords(axis: str, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    if axis == "x":
        return y, z
    if axis == "y":
        return x, z
    if axis == "z":
        return x, y
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
    draw_zero_contour: bool = False,
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
        cx, cy = _plane_coords(axis, x, y, z)
        Xc, Yc = np.meshgrid(cx, cy, indexing="xy")

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
        if draw_zero_contour and true_2d.min() <= 0.0 <= true_2d.max():
            axes[row, 0].contour(
                Xc,
                Yc,
                true_2d,
                levels=[0.0],
                colors="black",
                linewidths=1.2,
                linestyles="-",
            )

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
        if draw_zero_contour:
            # Overlay true phi=0 on the pred panel for direct interface comparison.
            if true_2d.min() <= 0.0 <= true_2d.max():
                axes[row, 1].contour(
                    Xc,
                    Yc,
                    true_2d,
                    levels=[0.0],
                    colors="black",
                    linewidths=1.2,
                    linestyles="-",
                )
            if pred_2d.min() <= 0.0 <= pred_2d.max():
                axes[row, 1].contour(
                    Xc,
                    Yc,
                    pred_2d,
                    levels=[0.0],
                    colors="yellow",
                    linewidths=1.2,
                    linestyles="--",
                )

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
        if draw_zero_contour:
            if true_2d.min() <= 0.0 <= true_2d.max():
                axes[row, 2].contour(
                    Xc,
                    Yc,
                    true_2d,
                    levels=[0.0],
                    colors="black",
                    linewidths=1.0,
                    linestyles="-",
                )
            if pred_2d.min() <= 0.0 <= pred_2d.max():
                axes[row, 2].contour(
                    Xc,
                    Yc,
                    pred_2d,
                    levels=[0.0],
                    colors="yellow",
                    linewidths=1.0,
                    linestyles="--",
                )

    fig.suptitle(title_prefix)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def save_phi_pred_multiplane(
    phi_true: np.ndarray,
    phi_pred: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    title_prefix: str,
    save_path: Path,
) -> None:
    # Use x-only slices that cut through the sphere interior, so each panel shows a circle-like cross section.
    x_count = 6
    x_inner_ratio = 0.80
    radius_est = float(np.sqrt(max(0.0, float(np.max(phi_true)))))
    x_left = float(C1[0] - x_inner_ratio * radius_est)
    x_right = float(C1[0] + x_inner_ratio * radius_est)
    x_targets = np.linspace(x_left, x_right, x_count, dtype=np.float64)

    idx_list: List[int] = []
    for xv in x_targets:
        idx = _nearest_index(x, float(xv))
        if idx not in idx_list:
            idx_list.append(idx)

    # Fallback when nearest indices collapse on coarse grids: fill with other x-planes
    # that still intersect the sphere (enough positive cells in true phi mask).
    if len(idx_list) < x_count:
        candidate: List[int] = []
        for i in range(len(x)):
            pos_cells = int((_extract_plane(phi_true, "x", i) > 0.0).sum())
            if pos_cells >= 8:
                candidate.append(i)
        candidate.sort(key=lambda i: abs(float(x[i]) - float(C1[0])))
        for i in candidate:
            if i not in idx_list:
                idx_list.append(i)
            if len(idx_list) >= x_count:
                break

    if len(idx_list) < x_count:
        all_idx = list(np.argsort(np.abs(x.astype(np.float64) - float(C1[0]))))
        for i in all_idx:
            if int(i) not in idx_list:
                idx_list.append(int(i))
            if len(idx_list) >= x_count:
                break

    idx_list = idx_list[:x_count]
    specs = [(f"x-{k+1}", "x", idx, float(x[idx])) for k, idx in enumerate(idx_list)]
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.8, 6.8), dpi=220, squeeze=False)
    vmin = float(phi_pred.min())
    vmax = float(phi_pred.max())
    if abs(vmax - vmin) < 1e-14:
        vmax = vmin + 1e-14

    def _iou_from_phi(phi_true_2d: np.ndarray, phi_pred_2d: np.ndarray) -> float:
        mask_true = phi_true_2d > 0.0
        mask_pred = phi_pred_2d > 0.0
        inter = np.logical_and(mask_true, mask_pred).sum(dtype=np.int64)
        union = np.logical_or(mask_true, mask_pred).sum(dtype=np.int64)
        if union == 0:
            return 1.0
        return float(inter / union)

    im = None
    iou_list: List[float] = []
    for i, (_, axis, idx, coord) in enumerate(specs):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r, c]
        true_2d = _extract_plane(phi_true, axis, idx)
        pred_2d = _extract_plane(phi_pred, axis, idx)
        extent, xlab, ylab = _plane_extent(axis, x, y, z)
        cx, cy = _plane_coords(axis, x, y, z)
        Xc, Yc = np.meshgrid(cx, cy, indexing="xy")

        im = ax.imshow(
            pred_2d,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        if true_2d.min() <= 0.0 <= true_2d.max():
            ax.contour(Xc, Yc, true_2d, levels=[0.0], colors="black", linewidths=1.2, linestyles="-")
        if pred_2d.min() <= 0.0 <= pred_2d.max():
            ax.contour(Xc, Yc, pred_2d, levels=[0.0], colors="yellow", linewidths=1.2, linestyles="--")

        iou = _iou_from_phi(true_2d, pred_2d)
        iou_list.append(iou)
        ax.text(
            0.02,
            0.98,
            f"IoU={iou:.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
        )
        ax.set_title(f"pred | {axis}={coord:.3f}")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    # Reserve a dedicated right-side colorbar axis to avoid overlap with subplots.
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.08, top=0.90, wspace=0.25, hspace=0.28)
    cax = fig.add_axes([0.92, 0.16, 0.018, 0.68])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=7)
    iou_mean = float(np.mean(iou_list)) if iou_list else 0.0
    iou_min = float(np.min(iou_list)) if iou_list else 0.0
    iou_vol = _iou_from_phi(phi_true, phi_pred)
    xcuts_txt = ", ".join([f"{coord:.3f}" for _, _, _, coord in specs])
    fig.suptitle(
        f"{title_prefix} | x-cuts=[{xcuts_txt}] | "
        f"IoU(slice mean/min)={iou_mean:.4f}/{iou_min:.4f}, IoU(vol)={iou_vol:.4f}",
        y=0.98,
    )
    fig.savefig(save_path)
    plt.close(fig)


def _save_scatter_like_flower(
    *,
    res_map: np.ndarray,
    phi_true_map: np.ndarray,
    phi_pred_map: np.ndarray,
    phi_next_map: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    save_png: Path,
    save_npz: Path,
) -> None:
    Yg, Zg = np.meshgrid(y, z, indexing="xy")
    yz_points = np.stack([Yg.ravel(), Zg.ravel()], axis=1).astype(np.float32)
    res_val = res_map.reshape(-1).astype(np.float32)

    vmax_lin = float(np.percentile(res_val, 99.0))
    if vmax_lin <= 0:
        vmax_lin = float(res_val.max()) if res_val.size > 0 else 1.0
    if vmax_lin <= 0:
        vmax_lin = 1.0
    res_clip = np.clip(res_val, 0.0, vmax_lin)

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, ax = plt.subplots(dpi=300, figsize=(3.8, 2.5))

    sc = ax.scatter(
        yz_points[:, 0],
        yz_points[:, 1],
        c=res_clip,
        s=8,
        marker="s",
        linewidths=0.0,
        cmap="cividis",
        vmin=0.0,
        vmax=vmax_lin,
    )

    if phi_true_map.min() <= 0.0 <= phi_true_map.max():
        ax.contour(Yg, Zg, phi_true_map, levels=[0.0], colors="black", linewidths=1.0, linestyles="-")
    if phi_pred_map.min() <= 0.0 <= phi_pred_map.max():
        ax.contour(Yg, Zg, phi_pred_map, levels=[0.0], colors="red", linewidths=1.0, linestyles="--")
    if phi_next_map.min() <= 0.0 <= phi_next_map.max():
        ax.contour(Yg, Zg, phi_next_map, levels=[0.0], colors="lime", linewidths=0.8, linestyles="dashdot")

    ax.set_xlim(float(y.min()), float(y.max()))
    ax.set_ylim(float(z.min()), float(z.max()))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_title(title)

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.0, ls="-", label="Exact"),
        Line2D([0], [0], color="red", lw=1.0, ls="--", label="Current"),
        Line2D([0], [0], color="lime", lw=1.0, ls="-.", label="Next"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        framealpha=0.8,
        facecolor="white",
        edgecolor="none",
        fontsize=6,
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=1.0)
    cbar.mappable.set_clim(0.0, vmax_lin)
    sf = ScalarFormatter(useMathText=True)
    sf.set_powerlimits((-2, 2))
    sf.set_scientific(True)
    cbar.ax.yaxis.set_major_formatter(sf)

    fig.tight_layout()
    fig.savefig(save_png, dpi=300)
    plt.close(fig)

    np.savez_compressed(
        save_npz,
        XY_residual=yz_points,
        YZ_residual=yz_points,
        RES_val=res_val.astype(np.float32),
        phi_model=phi_pred_map.astype(np.float32),
        phi_true=phi_true_map.astype(np.float32),
        phi_next=phi_next_map.astype(np.float32),
        bbox=np.asarray([float(y.min()), float(y.max()), float(z.min()), float(z.max())], dtype=np.float32),
    )


def save_scatter_visuals(
    model: PartitionPINN3D,
    fields: Dict[str, np.ndarray],
    *,
    xyz_fit: torch.Tensor,
    u_fit: torch.Tensor,
    train_cfg: PIMOE3DTrainConfig,
    output_dir: Path,
    epoch: int,
    out_cfg: PIMOE3DOutputConfig,
    device: torch.device,
) -> None:
    scatter_dir = output_dir / out_cfg.viz_scatter_dir
    scatter_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{epoch:08d}"

    # Flower-style scatter grid: dense and independent from eval n_grid.
    n_scatter = max(16, int(out_cfg.scatter_n))
    y0, y1, z0, z1 = [float(v) for v in out_cfg.scatter_yz_bbox]
    y = np.linspace(y0, y1, n_scatter, dtype=np.float32)
    z = np.linspace(z0, z1, n_scatter, dtype=np.float32)
    x_fix = float(C1[0] if out_cfg.scatter_x_fixed is None else out_cfg.scatter_x_fixed)

    Y, Z = np.meshgrid(y, z, indexing="ij")
    X = np.full_like(Y, fill_value=x_fix, dtype=np.float32)
    xyz_plane = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    xyz_plane_t = torch.from_numpy(xyz_plane).to(device)

    was_training = model.training
    model.eval()
    try:
        batch_size = max(1, int(out_cfg.scatter_batch_size))
        phi_pred_plane, _, _, u_pred_plane = _predict_chunks(model, xyz_plane_t, batch_size)
        f_pred_plane = _f_pred_chunks(model, xyz_plane_t, batch_size)
        with torch.no_grad():
            phi_true_plane = phi_signed_c1_sphere(xyz_plane_t)
            u_true_plane = exact_solution(xyz_plane_t)
            f_true_plane = source_term_piecewise(xyz_plane_t)

        phi_true_2d = phi_true_plane.detach().cpu().numpy().reshape(n_scatter, n_scatter).T
        phi_pred_2d = phi_pred_plane.detach().cpu().numpy().reshape(n_scatter, n_scatter).T
        pde_abs_2d = (
            (f_pred_plane - f_true_plane).abs().detach().cpu().numpy().reshape(n_scatter, n_scatter).T
        )
        data_abs_2d = (
            (u_pred_plane - u_true_plane).abs().detach().cpu().numpy().reshape(n_scatter, n_scatter).T
        )

        dz = float(z[1] - z[0]) if len(z) > 1 else 1.0
        dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
        g_z, g_y = np.gradient(pde_abs_2d, dz, dy)
        grad_2d = np.sqrt(g_z * g_z + g_y * g_y)

        kinds = [
            ("PDE", pde_abs_2d, "PDE"),
            ("GRAD", grad_2d, "GRAD"),
            # Keep flower naming: CV panel uses PDE residual coloring but CV(DATA) velocity for next contour.
            ("CV", pde_abs_2d, "CV"),
            # Keep DATA for explicit 3D data-residual diagnostics.
            ("DATA", data_abs_2d, "DATA"),
        ]
        for kind_name, res_2d, residual_type in kinds:
            pred = predict_phi_next_by_residual(
                model,
                xyz_plane_t,
                residual_type=residual_type,
                xyz_fit=xyz_fit,
                u_fit=u_fit,
                dt=train_cfg.phi_update_dt,
                band_eps=train_cfg.phi_update_band_eps,
                h=train_cfg.phi_update_radius,
                tau=train_cfg.phi_update_tau,
                clip_q=train_cfg.phi_update_clip_q,
            )
            phi_next_2d = pred["phi_next"].detach().cpu().numpy().reshape(n_scatter, n_scatter).T
            save_png = scatter_dir / f"scatter_{kind_name}_epoch_{tag}.png"
            save_npz = scatter_dir / f"scatter_{kind_name}_epoch_{tag}.npz"
            _save_scatter_like_flower(
                res_map=res_2d,
                phi_true_map=phi_true_2d,
                phi_pred_map=phi_pred_2d,
                phi_next_map=phi_next_2d,
                y=y,
                z=z,
                title=f"{kind_name} residual scatter @ x={x_fix:.3f}",
                save_png=save_png,
                save_npz=save_npz,
            )
    finally:
        if was_training:
            model.train()


def save_periodic_visuals(
    fields: Dict[str, np.ndarray],
    output_dir: Path,
    epoch: int,
    out_cfg: PIMOE3DOutputConfig,
    *,
    model: PartitionPINN3D,
    train_cfg: PIMOE3DTrainConfig,
    xyz_fit: torch.Tensor,
    u_fit: torch.Tensor,
    device: torch.device,
) -> None:
    u_dir = output_dir / out_cfg.u_heatmap_dir
    f_dir = output_dir / out_cfg.f_heatmap_dir
    phi_dir = output_dir / out_cfg.phi_heatmap_dir
    u_dir.mkdir(parents=True, exist_ok=True)
    f_dir.mkdir(parents=True, exist_ok=True)
    phi_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{epoch:08d}"
    u_png = u_dir / f"u_heatmap_{tag}.png"
    u_npz = u_dir / f"u_heatmap_{tag}.npz"
    f_png = f_dir / f"f_heatmap_{tag}.png"
    f_npz = f_dir / f"f_heatmap_{tag}.npz"
    phi_png = phi_dir / f"phi_heatmap_{tag}.png"
    phi_npz = phi_dir / f"phi_heatmap_{tag}.npz"

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
    save_phi_pred_multiplane(
        fields["phi_true"],
        fields["phi_pred"],
        fields["x"],
        fields["y"],
        fields["z"],
        title_prefix=f"phi pred(+true contour) @ epoch {epoch}",
        save_path=phi_png,
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
    np.savez_compressed(
        phi_npz,
        epoch=np.asarray([epoch], dtype=np.int64),
        x=fields["x"],
        y=fields["y"],
        z=fields["z"],
        phi_true=fields["phi_true"],
        phi_pred=fields["phi_pred"],
        phi_residual=fields["phi_residual"],
    )

    save_scatter_visuals(
        model,
        fields,
        xyz_fit=xyz_fit,
        u_fit=u_fit,
        train_cfg=train_cfg,
        output_dir=output_dir,
        epoch=epoch,
        out_cfg=out_cfg,
        device=device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADD-PINNs 3D solver for single-sphere (c1) interface problem")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-grid", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lam-eik", type=float, default=None)
    parser.add_argument("--lam-volume", type=float, default=None)
    parser.add_argument("--force-phi-trainable", action="store_true")
    parser.add_argument("--allow-phi-update-when-frozen", action="store_true")
    parser.add_argument("--phi-update-every", type=int, default=None)
    parser.add_argument("--phi-update-start-epoch", type=int, default=None)
    parser.add_argument("--phi-update-dt", type=float, default=None)
    parser.add_argument("--phi-update-inner-steps", type=int, default=None)
    parser.add_argument("--phi-update-stop-tol", type=float, default=None)
    parser.add_argument("--phi-update-residual-type", type=str, default=None)
    parser.add_argument("--phi-update-band-eps", type=float, default=None)
    parser.add_argument("--phi-update-radius", type=float, default=None)
    parser.add_argument("--phi-update-tau", type=float, default=None)
    parser.add_argument("--phi-update-clip-q", type=float, default=None)
    return parser.parse_args()


def _configure_output_dirs(project_root: Path, out_cfg: PIMOE3DOutputConfig):
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

    data_cfg = PIMOE3DDataConfig()
    model_cfg = PIMOE3DModelConfig()
    train_cfg = PIMOE3DTrainConfig()
    runtime_cfg = PIMOE3DRuntimeConfig()
    eval_cfg = PIMOE3DEvalConfig()
    out_cfg = PIMOE3DOutputConfig()

    if args.eval_grid is not None:
        eval_cfg.n_grid = int(args.eval_grid)
    if args.output_dir is not None:
        out_cfg.output_dir = str(args.output_dir)
    if args.lam_eik is not None:
        train_cfg.lam_weights["eik"] = float(args.lam_eik)
    if args.lam_volume is not None:
        train_cfg.lam_weights["volume"] = float(args.lam_volume)
    if args.phi_update_every is not None:
        train_cfg.phi_update_every = int(args.phi_update_every)
    if args.phi_update_start_epoch is not None:
        train_cfg.phi_update_start_epoch = int(args.phi_update_start_epoch)
    if args.phi_update_dt is not None:
        train_cfg.phi_update_dt = float(args.phi_update_dt)
    if args.phi_update_inner_steps is not None:
        train_cfg.phi_update_inner_steps = int(args.phi_update_inner_steps)
    if args.phi_update_stop_tol is not None:
        train_cfg.phi_update_stop_tol = float(args.phi_update_stop_tol)
    if args.phi_update_residual_type is not None:
        train_cfg.phi_update_residual_type = str(args.phi_update_residual_type)
    if args.phi_update_band_eps is not None:
        train_cfg.phi_update_band_eps = float(args.phi_update_band_eps)
    if args.phi_update_radius is not None:
        train_cfg.phi_update_radius = float(args.phi_update_radius)
    if args.phi_update_tau is not None:
        train_cfg.phi_update_tau = float(args.phi_update_tau)
    if args.phi_update_clip_q is not None:
        train_cfg.phi_update_clip_q = float(args.phi_update_clip_q)

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

    model = PartitionPINN3D(width=model_cfg.width, depth=model_cfg.depth).to(device)
    init_lr = float(blocks[0].lr)
    main_params = list(model.net_1.parameters()) + list(model.net_2.parameters())
    phi_params = list(model.phi.parameters())
    opt = torch.optim.Adam(main_params, lr=init_lr)
    opt_main_phi = torch.optim.Adam(phi_params, lr=init_lr)
    init_phi_lr = float(train_cfg.phi_lr) if train_cfg.phi_lr is not None else float(train_cfg.phi_lr_scale * init_lr)
    opt_phi = torch.optim.Adam(phi_params, lr=init_phi_lr)
    snapshot_root = out_dir / out_cfg.snapshot_dir
    snapshot_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = out_dir / out_cfg.checkpoint_dir
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    uf_slice_phi_root = out_dir / "uf_slice_with_phi"
    uf_slice_phi_root.mkdir(parents=True, exist_ok=True)

    records: List[List[float]] = []
    last_eval_fields: Dict[str, np.ndarray] = {}
    last_eval_epoch = -1
    start_epoch = 0
    block_idx = 0
    epochs_in_block = 0
    resume_path_str = args.resume if args.resume is not None else runtime_cfg.resume_checkpoint
    if resume_path_str:
        resume_path = Path(resume_path_str).expanduser()
        if not resume_path.is_absolute():
            resume_path = (project_root / resume_path).resolve()
        ckpt = load_checkpoint(
            resume_path,
            device=device,
            model=model,
            opt=opt,
            opt_main_phi=opt_main_phi,
            opt_phi=opt_phi,
        )
        start_epoch = int(ckpt["epoch"])
        xyz_int = ckpt["xyz_int"]
        records = ckpt["records"]
        block_idx, epochs_in_block = _resolve_block_position(start_epoch, blocks)
        current_lr, current_phi_lr = _set_optimizer_lrs(
            block_idx=block_idx,
            blocks=blocks,
            train_cfg=train_cfg,
            opt=opt,
            opt_main_phi=opt_main_phi,
            opt_phi=opt_phi,
        )
        print(
            f"[Resume] checkpoint={resume_path} epoch={start_epoch} "
            f"block={block_idx + 1}/{len(blocks)} lr={current_lr:.3e} "
            f"phi_lr={current_phi_lr:.3e} xyz_int={tuple(xyz_int.shape)}"
        )
    if start_epoch >= total_epochs:
        raise ValueError(
            f"resume epoch {start_epoch} already reaches/exceeds total_epochs {total_epochs}. "
            "Increase blocks/epochs before resuming."
        )
    train_start_time = time.perf_counter()
    for ep in range(start_epoch + 1, total_epochs + 1):
        if epochs_in_block >= blocks[block_idx].epochs and block_idx + 1 < len(blocks):
            block_idx += 1
            epochs_in_block = 0
            new_lr, new_phi_lr = _set_optimizer_lrs(
                block_idx=block_idx,
                blocks=blocks,
                train_cfg=train_cfg,
                opt=opt,
                opt_main_phi=opt_main_phi,
                opt_phi=opt_phi,
            )
            print(
                f"[LR] switch to block {block_idx + 1}/{len(blocks)} "
                f"lr={new_lr:.3e}, phi_lr={new_phi_lr:.3e} at epoch {ep}"
            )

        phi_trainable = runtime_cfg.force_phi_trainable or args.force_phi_trainable or (((ep - 1) // 5000) % 2 == 0)

        total, weighted, raw = compute_pimoe3d_loss(
            model,
            xyz_int,
            xyz_fit=xyz_fit,
            u_fit=u_fit,
            lam=train_cfg.lam_weights,
            phi_trainable=phi_trainable,
        )

        opt.zero_grad(set_to_none=True)
        opt_main_phi.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        if phi_trainable:
            opt_main_phi.step()

        if ep % train_cfg.print_every == 0 or ep == 1:
            phi_mode = "on" if phi_trainable else "off"
            elapsed_s = time.perf_counter() - train_start_time
            avg_epoch_s = elapsed_s / float(ep)
            eta_s = avg_epoch_s * float(total_epochs - ep)
            print(
                f"E{ep:>6d} total={weighted['total'].item():.3e} "
                f"data={weighted['data'].item():.3e} pde={weighted['pde'].item():.3e} "
                f"if={weighted['interface'].item():.3e} eik={weighted['eik'].item():.3e} "
                f"vol={weighted['volume'].item():.3e} surf={weighted['surface'].item():.3e} "
                f"phi_update={phi_mode} "
                f"elapsed={_format_duration(elapsed_s)} "
                f"avg_epoch={avg_epoch_s:.3f}s eta={_format_duration(eta_s)}"
            )

        if ep % train_cfg.record_every == 0 or ep == total_epochs:
            records.append(
                [
                    float(ep),
                    float(weighted["total"].detach().cpu()),
                    float(weighted["data"].detach().cpu()),
                    float(weighted["pde"].detach().cpu()),
                    float(weighted["interface"].detach().cpu()),
                    float(weighted["eik"].detach().cpu()),
                    float(weighted["volume"].detach().cpu()),
                    float(weighted["surface"].detach().cpu()),
                    float(raw["total"].detach().cpu()),
                    float(raw["data"].detach().cpu()),
                    float(raw["pde"].detach().cpu()),
                    float(raw["interface"].detach().cpu()),
                    float(raw["eik"].detach().cpu()),
                    float(raw["volume"].detach().cpu()),
                    float(raw["surface"].detach().cpu()),
                ]
            )

        need_regular_snapshot = out_cfg.snapshot_every > 0 and ep % out_cfg.snapshot_every == 0
        need_uf_slice_plot = (
            getattr(out_cfg, "slice_snapshot_every", 0) > 0
            and ep % int(getattr(out_cfg, "slice_snapshot_every", 0)) == 0
        )
        if need_regular_snapshot or need_uf_slice_plot:
            fields_ep = evaluate_fields(model, eval_cfg, device)
            fit_l2_ep = evaluate_label_u_l2(
                model,
                xyz_fit,
                u_fit,
                eval_batch_size=eval_cfg.eval_batch_size,
            )
            snap_dir = snapshot_root / f"epoch_{ep:07d}"
            save_uf_snapshot(fields_ep, snap_dir, ep)
            if need_regular_snapshot:
                save_periodic_visuals(
                    fields_ep,
                    out_dir,
                    ep,
                    out_cfg,
                    model=model,
                    train_cfg=train_cfg,
                    xyz_fit=xyz_fit,
                    u_fit=u_fit,
                    device=device,
                )
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
            if need_uf_slice_plot:
                uf_slice_png = uf_slice_phi_root / f"uf_phi_line_epoch_{ep:07d}.png"
                save_uf_slice_with_phi_plot(snap_dir / "u_f_fields.npz", save_path=uf_slice_png)
                print(f"[Save] uf-slice-phi: {uf_slice_png}")

        if train_cfg.checkpoint_every > 0 and ep % int(train_cfg.checkpoint_every) == 0:
            ckpt_path = checkpoint_root / f"checkpoint_{ep:08d}.pt"
            save_checkpoint(
                ckpt_path,
                epoch=ep,
                model=model,
                opt=opt,
                opt_main_phi=opt_main_phi,
                opt_phi=opt_phi,
                xyz_int=xyz_int,
                records=records,
            )
            print(f"[Checkpoint] saved: {ckpt_path}")

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

        if (
            (phi_trainable or runtime_cfg.allow_phi_update_when_frozen or args.allow_phi_update_when_frozen)
            and train_cfg.phi_update_every > 0
            and ep >= train_cfg.phi_update_start_epoch
            and ep % train_cfg.phi_update_every == 0
        ):
            phi_upd = evolve_phi_by_residual(
                model,
                xyz_int,
                opt_phi,
                residual_type=train_cfg.phi_update_residual_type,
                xyz_fit=xyz_fit,
                u_fit=u_fit,
                dt=train_cfg.phi_update_dt,
                n_inner=train_cfg.phi_update_inner_steps,
                stop_tol=train_cfg.phi_update_stop_tol,
                band_eps=train_cfg.phi_update_band_eps,
                h=train_cfg.phi_update_radius,
                tau=train_cfg.phi_update_tau,
                clip_q=train_cfg.phi_update_clip_q,
            )
            print(
                f"[PhiUpdate] epoch {ep:>6d} "
                f"type={phi_upd['type']}, "
                f"band={int(phi_upd['band_count'])}, "
                f"|Vn|max={phi_upd['vn_max']:.3e}, "
                f"res(mean/max)={phi_upd['r_mean']:.3e}/{phi_upd['r_max']:.3e}, "
                f"fit_loss={phi_upd['phi_fit_loss']:.3e}"
            )

        epochs_in_block += 1

    loss_csv = out_dir / out_cfg.loss_csv_name
    write_loss_csv(records, loss_csv)
    plot_loss(records, viz_dir / out_cfg.loss_png_name)

    if last_eval_epoch != total_epochs:
        fields = evaluate_fields(model, eval_cfg, device)
        final_snap_dir = snapshot_root / f"epoch_{total_epochs:07d}"
        save_uf_snapshot(fields, final_snap_dir, total_epochs)
        save_periodic_visuals(
            fields,
            out_dir,
            total_epochs,
            out_cfg,
            model=model,
            train_cfg=train_cfg,
            xyz_fit=xyz_fit,
            u_fit=u_fit,
            device=device,
        )
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
    np.savez_compressed(data_output_dir / out_cfg.field_npz_name, **fields)
    _save_three_slices(
        fields["u_residual"],
        fields["x"],
        fields["y"],
        fields["z"],
        "u residual",
        viz_dir / out_cfg.u_slices_png_name,
    )
    _save_three_slices(
        fields["phi_pred"] - fields["phi_true"],
        fields["x"],
        fields["y"],
        fields["z"],
        "phi residual",
        viz_dir / out_cfg.phi_slices_png_name,
    )

    torch.save(model.state_dict(), out_dir / "model.pt")
    final_ckpt_path = checkpoint_root / f"checkpoint_{total_epochs:08d}.pt"
    save_checkpoint(
        final_ckpt_path,
        epoch=total_epochs,
        model=model,
        opt=opt,
        opt_main_phi=opt_main_phi,
        opt_phi=opt_phi,
        xyz_int=xyz_int,
        records=records,
    )
    if loss_csv.exists():
        shutil.copy2(loss_csv, data_output_dir / out_cfg.loss_csv_name)

    total_train_time_s = time.perf_counter() - train_start_time
    print(f"[Time] train_elapsed={_format_duration(total_train_time_s)} ({total_train_time_s:.2f}s)")
    print(f"[Done] loss csv: {loss_csv}")
    print(f"[Done] final field npz: {data_output_dir / out_cfg.field_npz_name}")
    print(f"[Done] final checkpoint: {final_ckpt_path}")
    print(f"[Metric] u_mse={fields['u_mse'][0]:.3e}, f_mse={fields['f_mse'][0]:.3e}, phi_mse={fields['phi_mse'][0]:.3e}")
    print(
        f"[Metric] u_l2={fields['u_l2'][0]:.3e} (rel={fields['u_rel_l2'][0]:.3e}), "
        f"f_l2={fields['f_l2'][0]:.3e} (rel={fields['f_rel_l2'][0]:.3e})"
    )
    print(
        f"[Metric-fit] u_l2={fields['u_fit_l2'][0]:.3e} (rel={fields['u_fit_rel_l2'][0]:.3e})"
    )


if __name__ == "__main__":
    main()
