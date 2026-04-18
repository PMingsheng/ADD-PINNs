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
from data import (
    load_ellipse_uv_eps_fit,
    load_ellipse_uv_eps_fit_downsample,
    load_ellipse_uv_eps_fit_rect_dense,
    sample_xy_uniform,
)
from plot_u_slice_with_phi import save_u_slice_with_phi_plot_from_fields
from pinn_loss import piecewise_pde_residual, predict_strain
from problem import load_full_reference, phi_signed_ellipse
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
        header="epoch,total_weighted,raw_data_u,raw_data_eps,raw_pde,raw_interface,E_out,E_in",
        comments="",
    )


def plot_loss(loss_records: List[List[float]], save_path: Path) -> None:
    if not loss_records:
        return
    arr = np.asarray(loss_records, dtype=np.float64)
    plt.figure(figsize=(7, 4.5))
    plt.plot(arr[:, 0], arr[:, 1], label="total")
    plt.plot(arr[:, 0], arr[:, 2], label="data_u")
    plt.plot(arr[:, 0], arr[:, 3], label="data_eps")
    plt.plot(arr[:, 0], arr[:, 4], label="pde")
    plt.plot(arr[:, 0], arr[:, 5], label="interface")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Ellipse APINN Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def _load_training_data(data_cfg: APINNDataConfig, device: torch.device):
    if data_cfg.sampling_mode == "roi-on":
        cfg = data_cfg.roi_rect_config
        return load_ellipse_uv_eps_fit_rect_dense(
            nx=cfg["nx"],
            ny=cfg["ny"],
            txt_filename=str(Path(__file__).resolve().parent / data_cfg.txt_filename),
            device=device,
            ellipse=data_cfg.ellipse,
            tau_for_strain=cfg["tau_for_strain"],
            dense_factor=cfg["dense_factor"],
            rect_corners=cfg["rect_corners"],
            target_total=cfg["target_total"],
            random_state=cfg["random_state"],
        )
    if data_cfg.sampling_mode == "roi-rect":
        cfg = data_cfg.roi_rect_config
        return load_ellipse_uv_eps_fit_rect_dense(
            nx=cfg["nx"],
            ny=cfg["ny"],
            txt_filename=str(Path(__file__).resolve().parent / data_cfg.txt_filename),
            device=device,
            ellipse=data_cfg.ellipse,
            tau_for_strain=cfg["tau_for_strain"],
            dense_factor=cfg["dense_factor"],
            rect_corners=cfg["rect_corners"],
            target_total=cfg["target_total"],
            random_state=cfg["random_state"],
        )

    cfg = data_cfg.roi_off_config
    return load_ellipse_uv_eps_fit(
        nx=cfg["nx"],
        ny=cfg["ny"],
        txt_filename=str(Path(__file__).resolve().parent / data_cfg.txt_filename),
        device=device,
        ellipse=data_cfg.ellipse,
        tau_for_strain=cfg["tau_for_strain"],
        use_dense=cfg["use_dense"],
        dense_factor=cfg["dense_factor"],
        rect_corners=cfg["rect_corners"],
        xlim=cfg.get("xlim"),
        ylim=cfg.get("ylim"),
    )


def _rar_refine_apinn(model: APINN, xy_int: torch.Tensor, *, n_cand: int, n_new: int, nu: float, ellipse, xlim, ylim):
    xy_cand = sample_xy_uniform(n_cand, device=xy_int.device, xlim=xlim, ylim=ylim).detach().clone().requires_grad_(True)
    pred = model(xy_cand)
    E_out, E_in = model.get_E_scaled()
    Rx, Ry = piecewise_pde_residual(pred["u"], xy_cand, E_out=E_out, E_in=E_in, nu=nu, ellipse=ellipse)
    score = (Rx.square() + Ry.square()).reshape(-1)
    k = max(1, min(int(n_new), score.numel()))
    idx = torch.topk(score, k)[1]
    return torch.cat([xy_int, xy_cand[idx].detach()], dim=0)


def evaluate_fields(
    model: APINN,
    eval_cfg: APINNEvalConfig,
    device: torch.device,
    *,
    ellipse,
    nu: float,
    xy_full: torch.Tensor,
    u_full: torch.Tensor,
    eps_full: torch.Tensor,
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

    u_chunks = []
    w_chunks = []
    pde_chunks = []
    phi_true_chunks = []
    for st in range(0, xy_grid.shape[0], batch_size):
        ed = min(st + batch_size, xy_grid.shape[0])
        xy_chunk = xy_grid[st:ed].detach().clone().requires_grad_(True)
        pred = model(xy_chunk)
        E_out, E_in = model.get_E_scaled()
        Rx, Ry = piecewise_pde_residual(pred["u"], xy_chunk, E_out=E_out, E_in=E_in, nu=nu, ellipse=ellipse)
        u_chunks.append(pred["u"].detach())
        w_chunks.append(pred["weights"].detach())
        pde_chunks.append(torch.sqrt(Rx.square() + Ry.square()).detach())
        phi_true_chunks.append(phi_signed_ellipse(xy_chunk.detach(), ellipse=ellipse))

    with torch.no_grad():
        pred_full = model(xy_full)
        U_full_pred = pred_full["u"]
    xy_full_req = xy_full.detach().clone().requires_grad_(True)
    pred_full_eps = model(xy_full_req)
    eps_full_pred = predict_strain(pred_full_eps["u"], xy_full_req).detach()

    if was_training:
        model.train()

    u_grid = torch.cat(u_chunks, dim=0)
    w_grid = torch.cat(w_chunks, dim=0)
    pde_grid = torch.cat(pde_chunks, dim=0)
    phi_true = torch.cat(phi_true_chunks, dim=0)
    u_mse = ((U_full_pred - u_full) ** 2).mean().item()
    eps_mse = ((eps_full_pred - eps_full) ** 2).mean().item()

    return {
        "x": xg.detach().cpu().numpy(),
        "y": yg.detach().cpu().numpy(),
        "ux_pred": u_grid[:, 0:1].cpu().numpy().reshape(n, n),
        "uy_pred": u_grid[:, 1:2].cpu().numpy().reshape(n, n),
        "pde_residual": pde_grid.cpu().numpy().reshape(n, n),
        "gate_g1": w_grid[:, 0:1].cpu().numpy().reshape(n, n),
        "gate_g2": w_grid[:, 1:2].cpu().numpy().reshape(n, n),
        "gate_diff": (w_grid[:, 0:1] - w_grid[:, 1:2]).cpu().numpy().reshape(n, n),
        "phi_true": phi_true.detach().cpu().numpy().reshape(n, n),
        "bbox": np.asarray(eval_cfg.bbox, dtype=np.float64),
        "u_mse_full": np.asarray([u_mse], dtype=np.float64),
        "eps_mse_full": np.asarray([eps_mse], dtype=np.float64),
        "E_out": np.asarray([float(model.get_E_scaled()[0].detach().cpu())], dtype=np.float64),
        "E_in": np.asarray([float(model.get_E_scaled()[1].detach().cpu())], dtype=np.float64),
    }


def save_fields_npz(fields: Dict[str, np.ndarray], save_path: Path, epoch: int) -> None:
    np.savez_compressed(save_path, epoch=np.asarray([epoch], dtype=np.int64), **fields)


def save_slice_snapshot(
    fields: Dict[str, np.ndarray],
    save_dir: Path,
    *,
    epoch: int,
    txt_filename: str,
    ellipse,
) -> None:
    save_u_slice_with_phi_plot_from_fields(
        x_axis=fields["x"],
        y_axis=fields["y"],
        ux_pred_map=fields["ux_pred"],
        uy_pred_map=fields["uy_pred"],
        phi_map=fields.get("phi_true"),
        ellipse=ellipse,
        txt_filename=txt_filename,
        save_path=save_dir / f"u_slice_with_phi_{epoch:08d}.png",
        save_npz_path=save_dir / f"u_slice_with_phi_{epoch:08d}.npz",
        epoch=epoch,
        bbox=tuple(float(v) for v in fields["bbox"]),
        title_prefix="APINN",
    )


def pretrain_gate(
    model: APINN,
    xy_train: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    use_hard_target: bool,
    ellipse,
    print_every: int,
) -> None:
    if epochs <= 0:
        return

    opt_gate = torch.optim.Adam(model.gate.parameters(), lr=lr)
    with torch.no_grad():
        phi = phi_signed_ellipse(xy_train, ellipse=ellipse)
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
    slice_root = output_dir / "u_slice_with_phi"
    slice_root.mkdir(parents=True, exist_ok=True)

    set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xy_u, U_fit, xy_eps, E_fit, *_ = _load_training_data(data_cfg, device)
    xy_full, u_full, eps_full = load_full_reference(project_root / data_cfg.txt_filename, device)
    xy_int = sample_xy_uniform(
        train_cfg.interior_points,
        device=device,
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
        eps0=train_cfg.eps0,
        nu=train_cfg.nu,
        E_out_init=model_cfg.E_out_init,
        E_in_init=model_cfg.E_in_init,
        learn_E_out=model_cfg.learn_E_out,
        learn_E_in=model_cfg.learn_E_in,
        E_scale=model_cfg.E_scale,
    ).to(device)

    print(
        f"[APINN] trainable_params={count_trainable_parameters(model)} "
        f"sampling_mode={data_cfg.sampling_mode} xy_u={len(xy_u)} xy_eps={len(xy_eps)}"
    )

    pretrain_gate(
        model,
        xy_int,
        epochs=train_cfg.gate_pretrain_epochs,
        lr=train_cfg.gate_pretrain_lr,
        use_hard_target=train_cfg.gate_pretrain_use_hard_target,
        ellipse=data_cfg.ellipse,
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
            xy_u=xy_u,
            U_fit=U_fit,
            xy_eps=xy_eps,
            E_fit=E_fit,
            lam_data_u=train_cfg.lam_data_u,
            lam_data_eps=train_cfg.lam_data_eps,
            lam_pde=train_cfg.lam_pde,
            lam_interface=train_cfg.lam_interface,
            interface_points=train_cfg.interface_points,
            nu=train_cfg.nu,
            ellipse=data_cfg.ellipse,
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
                nu=train_cfg.nu,
                ellipse=data_cfg.ellipse,
                xlim=train_cfg.xy_int_xlim,
                ylim=train_cfg.xy_int_ylim,
            )
            print(f"[RAR] epoch {ep:>6d} | interior points -> {len(xy_int):,}")

        if ep % train_cfg.print_every == 0:
            E_out, E_in = model.get_E_scaled()
            print(
                f"E{ep:>6d} total={losses['total'].item():.3e} "
                f"u={losses['data_u'].item():.3e} "
                f"eps={losses['data_eps'].item():.3e} "
                f"pde={losses['pde'].item():.3e} "
                f"if={losses['interface'].item():.3e} "
                f"E_out={float(E_out.detach().cpu()):.4f} "
                f"E_in={float(E_in.detach().cpu()):.4f}"
            )

        if ep % train_cfg.record_every == 0:
            E_out, E_in = model.get_E_scaled()
            loss_records.append(
                [
                    float(ep),
                    float(losses["total"].detach().cpu()),
                    float(losses["data_u"].detach().cpu()),
                    float(losses["data_eps"].detach().cpu()),
                    float(losses["pde"].detach().cpu()),
                    float(losses["interface"].detach().cpu()),
                    float(E_out.detach().cpu()),
                    float(E_in.detach().cpu()),
                ]
            )

        if out_cfg.snapshot_every > 0 and ep % out_cfg.snapshot_every == 0:
            last_fields = evaluate_fields(
                model,
                eval_cfg,
                device,
                ellipse=data_cfg.ellipse,
                nu=train_cfg.nu,
                xy_full=xy_full,
                u_full=u_full,
                eps_full=eps_full,
            )
            save_fields_npz(last_fields, snapshot_root / f"epoch_{ep:07d}.npz", ep)
            save_slice_snapshot(last_fields, slice_root, epoch=ep, txt_filename=str(project_root / data_cfg.txt_filename), ellipse=data_cfg.ellipse)
            last_epoch = ep

    if last_epoch != train_cfg.epochs:
        last_fields = evaluate_fields(
            model,
            eval_cfg,
            device,
            ellipse=data_cfg.ellipse,
            nu=train_cfg.nu,
            xy_full=xy_full,
            u_full=u_full,
            eps_full=eps_full,
        )
        save_fields_npz(last_fields, snapshot_root / f"epoch_{train_cfg.epochs:07d}.npz", train_cfg.epochs)
        save_slice_snapshot(
            last_fields,
            slice_root,
            epoch=train_cfg.epochs,
            txt_filename=str(project_root / data_cfg.txt_filename),
            ellipse=data_cfg.ellipse,
        )
        last_epoch = train_cfg.epochs

    write_loss_csv(loss_records, output_dir / out_cfg.loss_csv_name)
    plot_loss(loss_records, output_dir / out_cfg.loss_png_name)
    save_fields_npz(last_fields, output_dir / out_cfg.field_npz_name, last_epoch)
    save_u_slice_with_phi_plot_from_fields(
        x_axis=last_fields["x"],
        y_axis=last_fields["y"],
        ux_pred_map=last_fields["ux_pred"],
        uy_pred_map=last_fields["uy_pred"],
        phi_map=last_fields.get("phi_true"),
        ellipse=data_cfg.ellipse,
        txt_filename=str(project_root / data_cfg.txt_filename),
        save_path=output_dir / "u_slice_with_phi_final.png",
        save_npz_path=output_dir / "u_slice_with_phi_final.npz",
        epoch=last_epoch,
        bbox=tuple(float(v) for v in last_fields["bbox"]),
        title_prefix="APINN",
    )


if __name__ == "__main__":
    main()
