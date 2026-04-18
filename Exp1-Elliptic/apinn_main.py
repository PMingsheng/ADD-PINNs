import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from apinn_config import (
    APINNDataConfig,
    APINNEvalConfig,
    APINNModelConfig,
    APINNOutputConfig,
    APINNTrainConfig,
)
from apinn_loss import compute_apinn_loss, laplacian as apinn_laplacian
from apinn_model import APINN
from data import load_uniform_grid_fit, sample_xy_no_corners
from pinn_loss import piecewise_source_term
from problem import exact_solution, phi_signed_flower
from utils import set_seed


def write_loss_csv(loss_records: List[List[float]], save_path: Path) -> None:
    if not loss_records:
        return
    arr = np.asarray(loss_records, dtype=np.float64)
    np.savetxt(
        save_path,
        arr,
        delimiter=",",
        header="epoch,total,data,pde",
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
    plt.title("APINN Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def _prediction_chunks(
    model: APINN,
    xy: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    u_chunks: List[torch.Tensor] = []
    w_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for st in range(0, xy.shape[0], batch_size):
            ed = min(st + batch_size, xy.shape[0])
            pred = model(xy[st:ed])
            u_chunks.append(pred["u"])
            w_chunks.append(pred["weights"])
    return torch.cat(u_chunks, dim=0), torch.cat(w_chunks, dim=0)


def _f_pred_chunks(
    model: APINN,
    xy: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for st in range(0, xy.shape[0], batch_size):
        ed = min(st + batch_size, xy.shape[0])
        xy_chunk = xy[st:ed].detach().clone().requires_grad_(True)
        pred = model(xy_chunk)
        u_chunk = pred["u"]

        grad_u = torch.autograd.grad(
            outputs=u_chunk,
            inputs=xy_chunk,
            grad_outputs=torch.ones_like(u_chunk),
            create_graph=True,
        )[0]
        d2u_dx2 = torch.autograd.grad(
            outputs=grad_u[:, 0:1],
            inputs=xy_chunk,
            grad_outputs=torch.ones_like(grad_u[:, 0:1]),
            create_graph=False,
            retain_graph=True,
        )[0][:, 0:1]
        d2u_dy2 = torch.autograd.grad(
            outputs=grad_u[:, 1:2],
            inputs=xy_chunk,
            grad_outputs=torch.ones_like(grad_u[:, 1:2]),
            create_graph=False,
        )[0][:, 1:2]
        chunks.append((-(d2u_dx2 + d2u_dy2)).detach())
    return torch.cat(chunks, dim=0)


def _sample_candidate_points(
    n_cand: int,
    device: torch.device,
    *,
    corner_tol: float,
    batch_size: int,
    xlim,
    ylim,
) -> torch.Tensor:
    x0, x1 = xlim
    y0, y1 = ylim
    mx = corner_tol * (x1 - x0)
    my = corner_tol * (y1 - y0)
    pts: List[torch.Tensor] = []
    total = 0
    while total < n_cand:
        xr = torch.rand(batch_size, 1, device=device) * (x1 - x0) + x0
        yr = torch.rand(batch_size, 1, device=device) * (y1 - y0) + y0
        xy_batch = torch.cat([xr, yr], dim=1)
        mask = (
            (xy_batch[:, 0] > x0 + mx)
            & (xy_batch[:, 0] < x1 - mx)
            & (xy_batch[:, 1] > y0 + my)
            & (xy_batch[:, 1] < y1 - my)
        )
        xy_valid = xy_batch[mask]
        pts.append(xy_valid)
        total += xy_valid.shape[0]
    return torch.cat(pts, dim=0)[:n_cand]


def _rar_refine_apinn(
    model: APINN,
    xy_int: torch.Tensor,
    *,
    n_cand: int,
    n_new: int,
    corner_tol: float,
    batch_size: int,
    xlim,
    ylim,
) -> torch.Tensor:
    xy_cand = _sample_candidate_points(
        n_cand,
        xy_int.device,
        corner_tol=corner_tol,
        batch_size=batch_size,
        xlim=xlim,
        ylim=ylim,
    ).detach().clone().requires_grad_(True)

    pred = model(xy_cand)
    u_cand = pred["u"]
    f_cand = piecewise_source_term(xy_cand).detach()
    pde_res = apinn_laplacian(u_cand, xy_cand) + f_cand

    k = max(1, min(int(n_new), pde_res.numel()))
    topk_idx = torch.topk(pde_res.abs().reshape(-1), k)[1]
    xy_new = xy_cand[topk_idx].detach()
    return torch.cat([xy_int, xy_new], dim=0)


def evaluate_fields(
    model: APINN,
    eval_cfg: APINNEvalConfig,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    x0, x1, y0, y1 = eval_cfg.bbox
    n = eval_cfg.n_grid

    xg = torch.linspace(x0, x1, n, device=device)
    yg = torch.linspace(y0, y1, n, device=device)
    Xg, Yg = torch.meshgrid(xg, yg, indexing="ij")
    xy_grid = torch.stack([Xg, Yg], dim=-1).reshape(-1, 2)

    was_training = model.training
    model.eval()
    batch_size = max(1, int(eval_cfg.eval_batch_size))

    u_pred, weights = _prediction_chunks(model, xy_grid, batch_size)
    with torch.no_grad():
        u_true = exact_solution(xy_grid)
        f_true = piecewise_source_term(xy_grid)
    f_pred = _f_pred_chunks(model, xy_grid, batch_size)

    u_residual = u_pred - u_true
    f_residual = f_pred - f_true

    g1 = weights[:, 0:1]
    g2 = weights[:, 1:2]
    gdiff = g1 - g2

    u_mse = (u_residual.pow(2).mean()).item()
    f_mse = (f_residual.pow(2).mean()).item()

    if was_training:
        model.train()

    return {
        "x": xg.detach().cpu().numpy(),
        "y": yg.detach().cpu().numpy(),
        "u_pred": u_pred.detach().cpu().numpy().reshape(n, n),
        "u_true": u_true.detach().cpu().numpy().reshape(n, n),
        "u_residual": u_residual.detach().cpu().numpy().reshape(n, n),
        "f_pred": f_pred.detach().cpu().numpy().reshape(n, n),
        "f_true": f_true.detach().cpu().numpy().reshape(n, n),
        "f_residual": f_residual.detach().cpu().numpy().reshape(n, n),
        "g1": g1.detach().cpu().numpy().reshape(n, n),
        "g2": g2.detach().cpu().numpy().reshape(n, n),
        "gdiff": gdiff.detach().cpu().numpy().reshape(n, n),
        "bbox": np.asarray(eval_cfg.bbox, dtype=np.float64),
        "u_mse": np.asarray([u_mse], dtype=np.float64),
        "f_mse": np.asarray([f_mse], dtype=np.float64),
    }


def save_fields_npz(fields: Dict[str, np.ndarray], save_path: Path, epoch: int) -> None:
    np.savez_compressed(save_path, epoch=np.asarray([epoch], dtype=np.int64), **fields)


def save_heatmap(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    save_path: Path,
    *,
    cmap: str = "viridis",
    is_residual: bool,
) -> None:
    plt.figure(figsize=(6.2, 5.0))
    vmin = None
    vmax = None
    if is_residual:
        vmax = float(np.max(np.abs(field)))
        if vmax < 1e-14:
            vmax = 1e-14
        vmin = -vmax
        cmap = "RdBu_r"

    im = plt.imshow(
        field.T,
        origin="lower",
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def save_gate_partition_plot(
    gdiff: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    save_path: Path,
    epoch: int,
) -> None:
    Xg, Yg = np.meshgrid(x, y, indexing="ij")
    xy_np = np.stack([Xg.reshape(-1), Yg.reshape(-1)], axis=1)
    xy_t = torch.from_numpy(xy_np.astype(np.float32))
    with torch.no_grad():
        phi_true = phi_signed_flower(xy_t).cpu().numpy().reshape(gdiff.shape)

    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=220)
    cs = ax.contourf(Xg, Yg, gdiff, levels=60, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.contour(Xg, Yg, phi_true, levels=[0.0], colors="black", linewidths=1.2, linestyles="-")
    ax.contour(Xg, Yg, gdiff, levels=[0.0], colors="red", linewidths=1.2, linestyles="--")
    ax.legend([
        Line2D([0], [0], color="black", lw=1.2, ls="-"),
        Line2D([0], [0], color="red", lw=1.2, ls="--"),
    ], ["exact interface", "g1=g2"], loc="lower right")
    ax.set_title(f"Gate partition (epoch {epoch})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(cs, ax=ax, shrink=0.88)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_snapshot(
    model: APINN,
    eval_cfg: APINNEvalConfig,
    save_dir: Path,
    device: torch.device,
    epoch: int,
) -> Dict[str, np.ndarray]:
    save_dir.mkdir(parents=True, exist_ok=True)
    fields = evaluate_fields(model, eval_cfg, device)
    save_fields_npz(fields, save_dir / "fields.npz", epoch)

    x = fields["x"]
    y = fields["y"]

    save_heatmap(fields["u_true"], x, y, f"u true (epoch {epoch})", save_dir / "u_true.png", is_residual=False)
    save_heatmap(fields["u_pred"], x, y, f"u pred (epoch {epoch})", save_dir / "u_pred.png", is_residual=False)
    save_heatmap(
        fields["u_residual"], x, y, f"u residual (epoch {epoch})", save_dir / "u_residual.png", is_residual=True
    )

    save_heatmap(fields["f_true"], x, y, f"f true (epoch {epoch})", save_dir / "f_true.png", cmap="coolwarm", is_residual=False)
    save_heatmap(fields["f_pred"], x, y, f"f pred (epoch {epoch})", save_dir / "f_pred.png", cmap="coolwarm", is_residual=False)
    save_heatmap(
        fields["f_residual"], x, y, f"f residual (epoch {epoch})", save_dir / "f_residual.png", is_residual=True
    )

    save_heatmap(fields["g1"], x, y, f"gate g1 (epoch {epoch})", save_dir / "gate_g1.png", cmap="plasma", is_residual=False)
    save_heatmap(fields["g2"], x, y, f"gate g2 (epoch {epoch})", save_dir / "gate_g2.png", cmap="plasma", is_residual=False)
    save_heatmap(fields["gdiff"], x, y, f"gate g1-g2 (epoch {epoch})", save_dir / "gate_diff.png", is_residual=True)
    save_gate_partition_plot(fields["gdiff"], x, y, save_dir / "gate_partition.png", epoch)

    return fields


def pretrain_gate(
    model: APINN,
    xy_train: torch.Tensor,
    epochs: int,
    lr: float,
    use_hard_target: bool,
    print_every: int = 200,
) -> None:
    if epochs <= 0:
        return

    model.train()
    opt_gate = torch.optim.Adam(model.gate.parameters(), lr=lr)

    with torch.no_grad():
        phi0 = phi_signed_flower(xy_train)
        if use_hard_target:
            target_g1 = (phi0 < 0.0).float()
        else:
            target_g1 = torch.sigmoid(-phi0 / max(model.gate_tau, 1e-6))

    for ep in range(1, epochs + 1):
        weights, _ = model.gate_weights(xy_train)
        pred_g1 = weights[:, 0:1]
        loss_gate = ((pred_g1 - target_g1) ** 2).mean()

        opt_gate.zero_grad()
        loss_gate.backward()
        opt_gate.step()

        if ep % print_every == 0:
            print(f"[Gate pretrain] E{ep:>6d} mse={loss_gate.item():.3e}")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    data_cfg = APINNDataConfig()
    model_cfg = APINNModelConfig()
    train_cfg = APINNTrainConfig()
    eval_cfg = APINNEvalConfig()
    out_cfg = APINNOutputConfig()

    output_dir = out_cfg.resolve_output_dir(project_root)
    print(f"[Output] writing results to: {output_dir}")

    set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = (project_root / data_cfg.ttxt_filename).resolve()
    xy_fit, u_fit = load_uniform_grid_fit(
        nx=data_cfg.nx,
        ny=data_cfg.ny,
        use_synthetic=data_cfg.use_synthetic,
        synthetic_n_side=data_cfg.synthetic_n_side,
        ttxt_filename=str(data_path),
        device=device,
        circles=list(data_cfg.circles),
        annuli=list(data_cfg.annuli),
        dense_factor=data_cfg.dense_factor,
        drop_boundary=data_cfg.drop_boundary,
        xlim=data_cfg.xlim,
        ylim=data_cfg.ylim,
        tol=data_cfg.tol,
    )

    xy_int = sample_xy_no_corners(
        train_cfg.interior_points,
        device=device,
        corner_tol=train_cfg.corner_tol,
        batch_size=train_cfg.sample_batch_size,
        xlim=train_cfg.xy_int_xlim,
        ylim=train_cfg.xy_int_ylim,
    )

    model = APINN(
        n_experts=model_cfg.n_experts,
        shared_width=model_cfg.shared_width,
        shared_depth=model_cfg.shared_depth,
        shared_dim=model_cfg.shared_dim,
        expert_width=model_cfg.expert_width,
        expert_depth=model_cfg.expert_depth,
        gate_width=model_cfg.gate_width,
        gate_hidden_layers=model_cfg.gate_hidden_layers,
        gate_tau=model_cfg.gate_tau,
    ).to(device)

    snapshot_root = output_dir / out_cfg.snapshot_dir
    snapshot_root.mkdir(parents=True, exist_ok=True)

    # Save gate initialization before any optimization.
    save_snapshot(model, eval_cfg, snapshot_root / "epoch_0000000", device, 0)

    pretrain_gate(
        model,
        xy_int,
        epochs=train_cfg.gate_pretrain_epochs,
        lr=train_cfg.gate_pretrain_lr,
        use_hard_target=train_cfg.gate_pretrain_use_hard_target,
        print_every=max(1, train_cfg.print_every // 2),
    )
    if train_cfg.gate_pretrain_epochs > 0:
        save_snapshot(model, eval_cfg, snapshot_root / "gate_pretrain_done", device, 0)

    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    loss_records: List[List[float]] = []
    last_eval_epoch = -1
    last_eval_fields: Dict[str, np.ndarray] = {}

    for ep in range(1, train_cfg.epochs + 1):
        total, losses = compute_apinn_loss(
            model,
            xy_int,
            xy_fit=xy_fit,
            u_fit=u_fit,
            lam_data=train_cfg.lam_data,
            lam_pde=train_cfg.lam_pde,
        )

        opt.zero_grad()
        total.backward()
        opt.step()

        if train_cfg.rar_every > 0 and ep % train_cfg.rar_every == 0:
            xy_int = _rar_refine_apinn(
                model,
                xy_int,
                n_cand=train_cfg.rar_n_cand,
                n_new=train_cfg.rar_n_new,
                corner_tol=train_cfg.rar_corner_tol,
                batch_size=train_cfg.rar_batch_size,
                xlim=train_cfg.xy_int_xlim,
                ylim=train_cfg.xy_int_ylim,
            )
            print(f"[RAR] epoch {ep:>6d} | new training points -> {len(xy_int):,}")

        if ep % train_cfg.print_every == 0:
            print(
                f"E{ep:>6d} total={losses['total'].item():.3e} "
                f"data={losses['data'].item():.3e} pde={losses['pde'].item():.3e}"
            )

        if ep % train_cfg.record_every == 0:
            loss_records.append(
                [
                    float(ep),
                    float(losses["total"].detach().cpu()),
                    float(losses["data"].detach().cpu()),
                    float(losses["pde"].detach().cpu()),
                ]
            )

        if out_cfg.snapshot_every > 0 and ep % out_cfg.snapshot_every == 0:
            snapshot_dir = snapshot_root / f"epoch_{ep:07d}"
            last_eval_fields = save_snapshot(model, eval_cfg, snapshot_dir, device, ep)
            last_eval_epoch = ep
            print(f"[Save] snapshot saved: {snapshot_dir}")

    loss_csv = output_dir / out_cfg.loss_csv_name
    write_loss_csv(loss_records, loss_csv)
    plot_loss(loss_records, output_dir / out_cfg.loss_png_name)

    field_path = output_dir / out_cfg.field_npz_name
    if last_eval_epoch != train_cfg.epochs:
        final_snapshot_dir = snapshot_root / f"epoch_{train_cfg.epochs:07d}"
        last_eval_fields = save_snapshot(model, eval_cfg, final_snapshot_dir, device, train_cfg.epochs)
        last_eval_epoch = train_cfg.epochs
        print(f"[Save] snapshot saved: {final_snapshot_dir}")
    save_fields_npz(last_eval_fields, field_path, last_eval_epoch)

    print(f"[Done] loss csv: {loss_csv}")
    print(f"[Done] final field npz: {field_path}")


if __name__ == "__main__":
    main()
