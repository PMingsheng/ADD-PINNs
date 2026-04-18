from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from apinn_config import APINNDataConfig, APINNEvalConfig, APINNModelConfig, APINNOutputConfig, APINNTrainConfig
from apinn_loss import compute_apinn_loss
from apinn_model import APINN
from data import load_uniform_grid_fit, sample_xy_no_corners
from pinn_loss import piecewise_pde_residual
from problem import load_full_reference, phi_signed_circle
from utils import set_seed


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_loss_csv(loss_records: List[List[float]], save_path: Path) -> None:
    if not loss_records:
        return
    arr = np.asarray(loss_records, dtype=np.float64)
    np.savetxt(
        save_path,
        arr,
        delimiter=",",
        header="epoch,total_weighted,raw_data,raw_pde,raw_interface,f1,f2",
        comments="",
    )


def plot_loss(loss_records: List[List[float]], save_path: Path) -> None:
    if not loss_records:
        return
    arr = np.asarray(loss_records, dtype=np.float64)
    floor = 1e-16
    plt.figure(figsize=(7, 4.5))
    plt.plot(arr[:, 0], np.clip(arr[:, 1], floor, None), label="total")
    plt.plot(arr[:, 0], np.clip(arr[:, 2], floor, None), label="data")
    plt.plot(arr[:, 0], np.clip(arr[:, 3], floor, None), label="pde")
    plt.plot(arr[:, 0], np.clip(arr[:, 4], floor, None), label="interface")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Poisson APINN Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def save_field_plot(fields: Dict[str, np.ndarray], save_path: Path, *, title_prefix: str) -> None:
    x = fields["x"]
    y = fields["y"]
    xx, yy = np.meshgrid(x, y, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    im0 = axes[0].contourf(xx, yy, fields["T_pred"], levels=60, cmap="viridis")
    axes[0].contour(xx, yy, fields["phi_true"], levels=[0.0], colors="white", linewidths=1.0)
    axes[0].set_title(f"{title_prefix} Temperature")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0], shrink=0.9)

    im1 = axes[1].contourf(xx, yy, fields["pde_residual"], levels=60, cmap="magma")
    axes[1].contour(xx, yy, fields["phi_true"], levels=[0.0], colors="white", linewidths=1.0)
    axes[1].set_title(f"{title_prefix} PDE Residual")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1], shrink=0.9)

    im2 = axes[2].contourf(xx, yy, fields["gate_g1"], levels=60, cmap="coolwarm", vmin=0.0, vmax=1.0)
    axes[2].contour(xx, yy, fields["phi_true"], levels=[0.0], colors="black", linewidths=0.9)
    axes[2].set_title(f"{title_prefix} Gate g1")
    axes[2].set_aspect("equal")
    fig.colorbar(im2, ax=axes[2], shrink=0.9)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _load_training_data(data_cfg: APINNDataConfig, device: torch.device):
    sampling_cfg = data_cfg.sampling_configs.get(data_cfg.sampling_mode, data_cfg.sampling_configs["roi-off"])
    return load_uniform_grid_fit(
        nx=int(sampling_cfg["nx"]),
        ny=int(sampling_cfg["ny"]),
        ttxt_filename=str(Path(__file__).resolve().parent / data_cfg.txt_filename),
        device=device,
        circles=[data_cfg.circle],
        dense_factor=float(sampling_cfg["dense_factor"]),
        drop_boundary=bool(sampling_cfg["drop_boundary"]),
        xlim=tuple(float(v) for v in sampling_cfg["xlim"]),
        ylim=tuple(float(v) for v in sampling_cfg["ylim"]),
        tol=float(sampling_cfg["tol"]),
        target_total=sampling_cfg.get("target_total"),
    )


def _rar_refine_apinn(model: APINN, xy_int: torch.Tensor, *, n_cand: int, n_new: int, circle):
    xy_cand = sample_xy_no_corners(n_cand, device=xy_int.device).detach().clone().requires_grad_(True)
    pred = model(xy_cand)
    f1, f2 = model.get_f_scaled()
    r = piecewise_pde_residual(pred["T"], xy_cand, f1=f1, f2=f2, circle=circle)
    score = r.square().reshape(-1)
    k = max(1, min(int(n_new), score.numel()))
    idx = torch.topk(score, k)[1]
    return torch.cat([xy_int, xy_cand[idx].detach()], dim=0)


def evaluate_fields(
    model: APINN,
    eval_cfg: APINNEvalConfig,
    device: torch.device,
    *,
    circle,
    xy_full: torch.Tensor,
    t_full: torch.Tensor,
) -> Dict[str, np.ndarray]:
    x0, x1, y0, y1 = eval_cfg.bbox
    n = eval_cfg.n_grid

    xg = torch.linspace(x0, x1, n, device=device)
    yg = torch.linspace(y0, y1, n, device=device)
    xg_mesh, yg_mesh = torch.meshgrid(xg, yg, indexing="ij")
    xy_grid = torch.stack([xg_mesh, yg_mesh], dim=-1).reshape(-1, 2)

    was_training = model.training
    model.eval()
    batch_size = max(1, int(eval_cfg.eval_batch_size))

    t_chunks = []
    w_chunks = []
    pde_chunks = []
    phi_true_chunks = []
    for st in range(0, xy_grid.shape[0], batch_size):
        ed = min(st + batch_size, xy_grid.shape[0])
        xy_chunk = xy_grid[st:ed].detach().clone().requires_grad_(True)
        pred = model(xy_chunk)
        f1, f2 = model.get_f_scaled()
        r = piecewise_pde_residual(pred["T"], xy_chunk, f1=f1, f2=f2, circle=circle)
        t_chunks.append(pred["T"].detach())
        w_chunks.append(pred["weights"].detach())
        pde_chunks.append(torch.abs(r).detach())
        phi_true_chunks.append(phi_signed_circle(xy_chunk.detach(), circle=circle))

    with torch.no_grad():
        pred_full = model(xy_full)
        t_full_pred = pred_full["T"]

    if was_training:
        model.train()

    t_grid = torch.cat(t_chunks, dim=0)
    w_grid = torch.cat(w_chunks, dim=0)
    pde_grid = torch.cat(pde_chunks, dim=0)
    phi_true = torch.cat(phi_true_chunks, dim=0)
    t_mse = ((t_full_pred - t_full) ** 2).mean().item()
    f1, f2 = model.get_f_scaled()

    return {
        "x": xg.detach().cpu().numpy(),
        "y": yg.detach().cpu().numpy(),
        "T_pred": t_grid.cpu().numpy().reshape(n, n),
        "pde_residual": pde_grid.cpu().numpy().reshape(n, n),
        "gate_g1": w_grid[:, 0:1].cpu().numpy().reshape(n, n),
        "gate_g2": w_grid[:, 1:2].cpu().numpy().reshape(n, n),
        "gate_diff": (w_grid[:, 0:1] - w_grid[:, 1:2]).cpu().numpy().reshape(n, n),
        "phi_true": phi_true.detach().cpu().numpy().reshape(n, n),
        "bbox": np.asarray(eval_cfg.bbox, dtype=np.float64),
        "t_mse_full": np.asarray([t_mse], dtype=np.float64),
        "f1": np.asarray([float(f1.detach().cpu())], dtype=np.float64),
        "f2": np.asarray([float(f2.detach().cpu())], dtype=np.float64),
    }


def save_fields_npz(fields: Dict[str, np.ndarray], save_path: Path, epoch: int) -> None:
    np.savez_compressed(save_path, epoch=np.asarray([epoch], dtype=np.int64), **fields)


def pretrain_gate(
    model: APINN,
    xy_train: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    use_hard_target: bool,
    circle,
    print_every: int,
) -> None:
    if epochs <= 0:
        return

    opt_gate = torch.optim.Adam(model.gate.parameters(), lr=lr)
    with torch.no_grad():
        phi = phi_signed_circle(xy_train, circle=circle)
        if use_hard_target:
            target_g1 = (phi >= 0.0).float()
        else:
            target_g1 = torch.sigmoid(phi / max(model.gate_tau, 1e-6))

    for ep in range(1, epochs + 1):
        weights, _ = model.gate_weights(xy_train)
        loss_gate = ((weights[:, 0:1] - target_g1) ** 2).mean()

        opt_gate.zero_grad()
        loss_gate.backward()
        opt_gate.step()

        if ep % max(1, print_every) == 0:
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
    snapshot_root = output_dir / out_cfg.snapshot_dir
    snapshot_root.mkdir(parents=True, exist_ok=True)

    set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xy_fit, t_fit = _load_training_data(data_cfg, device)
    xy_full, t_full = load_full_reference(project_root / data_cfg.txt_filename, device)
    xy_int = sample_xy_no_corners(train_cfg.interior_points, device=device)

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
        f1_init=model_cfg.f1_init,
        f2_init=model_cfg.f2_init,
        learn_f1=model_cfg.learn_f1,
        learn_f2=model_cfg.learn_f2,
        f_scale=model_cfg.f_scale,
    ).to(device)

    print(
        f"[APINN] trainable_params={count_trainable_parameters(model)} "
        f"sampling_mode={data_cfg.sampling_mode} xy_fit={len(xy_fit)}"
    )

    pretrain_gate(
        model,
        xy_int,
        epochs=train_cfg.gate_pretrain_epochs,
        lr=train_cfg.gate_pretrain_lr,
        use_hard_target=train_cfg.gate_pretrain_use_hard_target,
        circle=data_cfg.circle,
        print_every=max(1, train_cfg.print_every // 2),
    )

    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    loss_records: List[List[float]] = []
    last_fields: Dict[str, np.ndarray] = {}
    last_epoch = 0

    for ep in range(1, train_cfg.epochs + 1):
        total, losses = compute_apinn_loss(
            model,
            xy_int,
            xy_fit=xy_fit,
            T_fit=t_fit,
            lam_data=train_cfg.lam_data,
            lam_pde=train_cfg.lam_pde,
            lam_interface=train_cfg.lam_interface,
            interface_points=train_cfg.interface_points,
            circle=data_cfg.circle,
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
                circle=data_cfg.circle,
            )
            print(f"[RAR] epoch {ep:>6d} | interior points -> {len(xy_int):,}")

        if ep % train_cfg.print_every == 0:
            f1, f2 = model.get_f_scaled()
            print(
                f"E{ep:>6d} total={losses['total'].item():.3e} "
                f"data={losses['data'].item():.3e} "
                f"pde={losses['pde'].item():.3e} "
                f"if={losses['interface'].item():.3e} "
                f"f1={float(f1.detach().cpu()):.4f} "
                f"f2={float(f2.detach().cpu()):.4f}"
            )

        if ep % train_cfg.record_every == 0:
            f1, f2 = model.get_f_scaled()
            loss_records.append(
                [
                    float(ep),
                    float(losses["total"].detach().cpu()),
                    float(losses["data"].detach().cpu()),
                    float(losses["pde"].detach().cpu()),
                    float(losses["interface"].detach().cpu()),
                    float(f1.detach().cpu()),
                    float(f2.detach().cpu()),
                ]
            )

        if out_cfg.snapshot_every > 0 and ep % out_cfg.snapshot_every == 0:
            last_fields = evaluate_fields(
                model,
                eval_cfg,
                device,
                circle=data_cfg.circle,
                xy_full=xy_full,
                t_full=t_full,
            )
            save_fields_npz(last_fields, snapshot_root / f"epoch_{ep:07d}.npz", ep)
            last_epoch = ep

    if last_epoch != train_cfg.epochs:
        last_fields = evaluate_fields(
            model,
            eval_cfg,
            device,
            circle=data_cfg.circle,
            xy_full=xy_full,
            t_full=t_full,
        )
        save_fields_npz(last_fields, snapshot_root / f"epoch_{train_cfg.epochs:07d}.npz", train_cfg.epochs)
        last_epoch = train_cfg.epochs

    write_loss_csv(loss_records, output_dir / out_cfg.loss_csv_name)
    plot_loss(loss_records, output_dir / out_cfg.loss_png_name)
    save_fields_npz(last_fields, output_dir / out_cfg.field_npz_name, last_epoch)
    save_field_plot(last_fields, output_dir / out_cfg.field_png_name, title_prefix="APINN")


if __name__ == "__main__":
    main()
