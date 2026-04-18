"""
Main entry for APINN beam solver.
Follows APINN structure (shared + experts + gate) and keeps training schedule,
sampling, and output level aligned with current ADD-PINNs beam workflow.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from beam_apinn_loss import compute_apinn_beam_loss
from beam_apinn_model import APINNBeam, count_trainable_params
from config import (
    CORNER_TOL,
    DATA_RANGES,
    DENSE_FACTOR,
    DEVICE,
    EI_REAL,
    EPOCHS,
    H,
    LABEL_COLUMN,
    LOSS_WEIGHTS,
    N_COLLOCATION,
    NX_GRID,
    PLOT_INTERVAL,
    SAVE_EPOCH_CUTOFF,
    SAVE_INTERVAL_EARLY,
    SAVE_INTERVAL_LATE,
    TRUE_JUMPS,
)
from data import load_uniform_grid_fit, sample_xy_no_corners
from model import PartitionPINN
from utils import EI_true, set_seed


def _as_float(t: torch.Tensor) -> float:
    return float(t.detach().cpu().item())


def _format_suffix(suffix) -> str:
    if isinstance(suffix, str):
        return suffix
    return f"{int(suffix):08d}"


def _should_save_snapshot(
    current_epoch: int,
    cutoff: int,
    interval_early: int,
    interval_late: int,
) -> bool:
    if current_epoch <= cutoff:
        return interval_early > 0 and current_epoch % interval_early == 0
    return interval_late > 0 and current_epoch % interval_late == 0


def _load_label_arrays(label_path: str) -> Dict[str, np.ndarray]:
    # usecols=(x, displacement, theta, M, V, strain)
    lab = np.loadtxt(label_path, usecols=(0, 2, 3, 4, 5, 6), comments="%")
    x = lab[:, 0].astype(np.float32)
    u_lab = lab[:, 1].astype(np.float32)
    theta_lab = lab[:, 2].astype(np.float32)
    M_lab = (lab[:, 3] / EI_REAL).astype(np.float32)
    V_lab = (-lab[:, 4] / EI_REAL).astype(np.float32)
    eps_lab = lab[:, 5].astype(np.float32)
    kappa_lab = (eps_lab / (H / 2.0)).astype(np.float32)
    return {
        "x": x,
        "u_lab": u_lab,
        "theta_lab": theta_lab,
        "kappa_lab": kappa_lab,
        "M_lab": M_lab,
        "V_lab": V_lab,
        "eps_lab": eps_lab,
    }


def _phi_proxy_from_gate(pred: Dict[str, torch.Tensor]) -> torch.Tensor:
    gate_logits = pred["gate_logits"]
    if gate_logits.shape[1] >= 2:
        return gate_logits[:, 0:1] - gate_logits[:, 1:2]
    # fallback for uncommon single-gate setup
    return gate_logits


def save_apinn_snapshot(
    model: APINNBeam,
    *,
    label_path: str,
    out_dir: str,
    suffix,
    plot_panels: bool = True,
    true_jumps: Tuple[float, float] = TRUE_JUMPS,
    epoch: int = -1,
) -> Tuple[str, str]:
    device = next(model.parameters()).device
    arrays = _load_label_arrays(label_path)

    x_np = arrays["x"]
    x = torch.from_numpy(x_np).to(device).unsqueeze(1).requires_grad_(True)
    ei_true_t = EI_true(x.detach().cpu()).to(device) / EI_REAL

    was_training = model.training
    model.eval()

    pred = model(x)
    u = pred["u"]
    EI = pred["EI"]
    phi_proxy = _phi_proxy_from_gate(pred)

    theta_NN = torch.autograd.grad(
        u, x, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    kappa_NN = torch.autograd.grad(
        theta_NN, x, torch.ones_like(theta_NN), create_graph=True, retain_graph=True
    )[0]
    M_NN = EI * kappa_NN
    V_NN = torch.autograd.grad(
        M_NN, x, torch.ones_like(M_NN), create_graph=True, retain_graph=True
    )[0]
    Q_NN = torch.autograd.grad(
        V_NN, x, torch.ones_like(V_NN), create_graph=False, retain_graph=True
    )[0]
    # In u+EI APINN, AUTO and NN kinematic/mechanics traces are identical by construction.
    theta_auto = theta_NN
    kappa_auto = kappa_NN
    M_auto = M_NN
    V_auto = V_NN

    if was_training:
        model.train()

    to_np = lambda t: t.detach().cpu().squeeze().numpy().astype(np.float32)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    suffix_str = _format_suffix(suffix)
    phi_png = str(out_path / f"phi_{suffix_str}.png")
    phi_npz = str(out_path / f"phi_{suffix_str}.npz")

    plt.figure(figsize=(3.35, 2.3), dpi=200)
    plt.plot(x_np, to_np(phi_proxy), "b-", lw=0.8, label="gate logit diff")
    for vx in (true_jumps or []):
        plt.axvline(vx, color="r", ls=":", lw=1.0)
    plt.axhline(0, color="r", ls=":", lw=1.0)
    plt.xlabel("x")
    plt.ylabel("phi_proxy")
    plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(phi_png, bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close()

    np.savez_compressed(
        phi_npz,
        epoch=np.asarray([int(epoch)], dtype=np.int64),
        x=x_np.astype(np.float32),
        phi=to_np(phi_proxy),
        EI_pred=to_np(EI),
        EI_true=to_np(ei_true_t),
        u_NN=to_np(u),
        theta_NN=to_np(theta_NN),
        kappa_NN=to_np(kappa_NN),
        M_NN=to_np(M_NN),
        V_NN=to_np(V_NN),
        Q_NN=to_np(Q_NN),
        theta_AUTO=to_np(theta_auto),
        kappa_AUTO=to_np(kappa_auto),
        M_AUTO=to_np(M_auto),
        V_AUTO=to_np(V_auto),
        gate_logits=to_np(pred["gate_logits"]),
        gate_weights=to_np(pred["weights"]),
        u_lab=arrays["u_lab"],
        theta_lab=arrays["theta_lab"],
        kappa_lab=arrays["kappa_lab"],
        M_lab=arrays["M_lab"],
        V_lab=arrays["V_lab"],
        eps_lab=arrays["eps_lab"],
    )

    if plot_panels:
        panels = [
            ("u", to_np(u), None, arrays["u_lab"], "u"),
            ("theta=du/dx", to_np(theta_NN), to_np(theta_auto), arrays["theta_lab"], "theta"),
            ("kappa=d2u/dx2", to_np(kappa_NN), to_np(kappa_auto), arrays["kappa_lab"], "kappa"),
            ("M", to_np(M_NN), to_np(M_auto), arrays["M_lab"], "M"),
            ("V", to_np(V_NN), to_np(V_auto), arrays["V_lab"], "V"),
            ("Q", to_np(Q_NN), to_np(Q_NN), None, "Q"),
            ("EI", to_np(EI), None, to_np(ei_true_t), "EI"),
            ("phi_proxy", to_np(phi_proxy), None, None, "phi_proxy"),
        ]
        nrow = (len(panels) + 2) // 3
        fig, axs = plt.subplots(nrow, 3, figsize=(15, 4 * nrow))
        axs = np.atleast_1d(axs).ravel()
        for ax, (ttl, y_nn, y_auto, y_lab, ylabel) in zip(axs, panels):
            ax.plot(x_np, y_nn, "b-", lw=0.9, label="NN")
            if y_auto is not None:
                ax.plot(x_np, y_auto, "g--", lw=1.0, label="auto-diff")
            if y_lab is not None:
                ax.scatter(x_np, y_lab, c="r", marker="^", s=8, edgecolors="k", label="label")
            for vx in (true_jumps or []):
                ax.axvline(vx, color="r", ls=":", lw=1.0)
            if ttl not in ("EI", "Q"):
                ax.axhline(0, color="k", ls=":", lw=0.8, alpha=0.6)
            ax.set_xlabel("x")
            ax.set_ylabel(ylabel)
            ax.set_title(ttl)
            ax.grid(True, ls="--", alpha=0.3)
            ax.legend(fontsize=9)

        for ax in axs[len(panels) :]:
            ax.axis("off")

        plt.tight_layout()
        panel_png = str(out_path / f"panels_{suffix_str}.png")
        plt.savefig(panel_png, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    else:
        panel_png = ""

    print(f"[SAVE] phi-curve  -> {phi_png}")
    print(f"[SAVE] phi-data   -> {phi_npz}")
    if panel_png:
        print(f"[SAVE] panels     -> {panel_png}")
    return phi_png, phi_npz


def stage_train(
    model: APINNBeam,
    *,
    epochs: int,
    lr: float,
    lam: Dict[str, float],
    xy_fit: torch.Tensor,
    label_fit: torch.Tensor,
    label_fit_disp: torch.Tensor,
    xy_int_const: torch.Tensor,
    epoch_offset: int,
    loss_rows: List[List[float]],
    save_snapshots: bool,
    snapshot_out_dir: str,
    snapshot_label_path: str,
    snapshot_interval_early: int = SAVE_INTERVAL_EARLY,
    snapshot_interval_late: int = SAVE_INTERVAL_LATE,
    snapshot_epoch_cutoff: int = SAVE_EPOCH_CUTOFF,
    snapshot_plot_panels: bool = False,
) -> int:
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if save_snapshots or epochs == 0:
        save_apinn_snapshot(
            model,
            label_path=snapshot_label_path,
            out_dir=snapshot_out_dir,
            suffix=epoch_offset,
            plot_panels=snapshot_plot_panels,
            epoch=epoch_offset,
        )

    for ep in range(1, epochs + 1):
        current_epoch = epoch_offset + ep
        total, d = compute_apinn_beam_loss(
            model,
            xy_int_const,
            x_fit=xy_fit,
            strain_fit=label_fit,
            u_fit_disp=label_fit_disp,
            lam=lam,
        )
        opt.zero_grad()
        total.backward()
        opt.step()

        if save_snapshots and _should_save_snapshot(
            current_epoch,
            snapshot_epoch_cutoff,
            snapshot_interval_early,
            snapshot_interval_late,
        ):
            save_apinn_snapshot(
                model,
                label_path=snapshot_label_path,
                out_dir=snapshot_out_dir,
                suffix=current_epoch,
                plot_panels=snapshot_plot_panels,
                epoch=current_epoch,
            )

        if current_epoch % PLOT_INTERVAL == 0:
            loss_rows.append(
                [
                    float(current_epoch),
                    _as_float(total),
                    _as_float(d["data"]),
                    _as_float(d["fai"]),
                    _as_float(d["dfai"]),
                    0.0,  # weight
                    _as_float(d["M"]),
                    _as_float(d["V"]),
                    _as_float(d["Q"]),
                    _as_float(d["dEI"]),
                    0.0,  # interface
                    0.0,  # eik
                    0.0,  # area
                ]
            )
            print(
                f"E{current_epoch:>6d} total={_as_float(total):.3e} "
                f"fit={_as_float(d['data_w']):.3e} "
                f"du/dx={_as_float(d['fai_w']):.3e} "
                f"dtheta/dx={_as_float(d['dfai_w']):.3e} "
                f"M={_as_float(d['M_w']):.3e} "
                f"V={_as_float(d['V_w']):.3e} "
                f"Q={_as_float(d['Q_w']):.3e} "
                f"dEI={_as_float(d['dEI_w']):.3e}"
            )
    return epoch_offset + epochs


def main() -> APINNBeam:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    print(f"Using device: {DEVICE}")
    set_seed()

    # APINN architecture chosen to match ADD-PINNs parameter scale closely.
    model = APINNBeam(
        n_experts=2,
        shared_width=112,
        shared_depth=3,
        shared_dim=48,
        expert_width=112,
        expert_depth=3,
        gate_width=16,
        gate_depth=3,
    ).to(DEVICE)

    pimoe_params = count_trainable_params(PartitionPINN().to(DEVICE))
    apinn_total = count_trainable_params(model)
    apinn_shared = count_trainable_params(model.shared)
    apinn_experts = sum(count_trainable_params(m) for m in model.experts)
    apinn_gate = count_trainable_params(model.gate)
    print(
        "Parameter count | "
        f"ADD-PINNs={pimoe_params}, APINN={apinn_total} "
        f"(shared={apinn_shared}, experts={apinn_experts}, gate={apinn_gate})"
    )

    data_file = project_root / "Beam.txt"
    data_file_str = str(data_file)

    xy_fit, label_fit = load_uniform_grid_fit(
        nx=NX_GRID,
        filename=data_file_str,
        device=DEVICE,
        ranges=DATA_RANGES,
        dense_factor=DENSE_FACTOR,
        label_column=LABEL_COLUMN,
    )
    _, label_fit_disp = load_uniform_grid_fit(
        nx=NX_GRID,
        filename=data_file_str,
        device=DEVICE,
        ranges=DATA_RANGES,
        dense_factor=DENSE_FACTOR,
        label_column=2,
    )
    xy_int_const = sample_xy_no_corners(N_COLLOCATION, device=DEVICE, corner_tol=CORNER_TOL)
    print(f"Generated {len(xy_int_const)} collocation points")

    base_weights = LOSS_WEIGHTS.copy()
    high_weights = LOSS_WEIGHTS.copy()
    high_weights["dEI"] = 100

    loss_rows: List[List[float]] = []
    epoch_offset = 0
    viz_dir = str(project_root / "beam_bechmark_viz_apinn")

    print("\n=== APINN stage 1 ===")
    epoch_offset = stage_train(
        model,
        epochs=2 * EPOCHS,
        lr=1e-4,
        lam=base_weights,
        xy_fit=xy_fit,
        label_fit=label_fit,
        label_fit_disp=label_fit_disp,
        xy_int_const=xy_int_const,
        epoch_offset=epoch_offset,
        loss_rows=loss_rows,
        save_snapshots=True,
        snapshot_out_dir=viz_dir,
        snapshot_label_path=data_file_str,
        snapshot_plot_panels=False,
    )
    save_apinn_snapshot(
        model,
        label_path=data_file_str,
        out_dir=viz_dir,
        suffix=epoch_offset,
        plot_panels=True,
        epoch=epoch_offset,
    )

    print("\n=== APINN stage 2 ===")
    epoch_offset = stage_train(
        model,
        epochs=4 * EPOCHS,
        lr=1e-5,
        lam=high_weights,
        xy_fit=xy_fit,
        label_fit=label_fit,
        label_fit_disp=label_fit_disp,
        xy_int_const=xy_int_const,
        epoch_offset=epoch_offset,
        loss_rows=loss_rows,
        save_snapshots=True,
        snapshot_out_dir=viz_dir,
        snapshot_label_path=data_file_str,
        snapshot_plot_panels=False,
    )
    save_apinn_snapshot(
        model,
        label_path=data_file_str,
        out_dir=viz_dir,
        suffix=epoch_offset,
        plot_panels=True,
        epoch=epoch_offset,
    )

    out_dir = project_root / "data_output_apinn"
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_path = out_dir / "loss_list_global.csv"
    header = "epoch,total,data,fai,dfai,weight,M,V,Q,dEI,interface,eik,area"
    np.savetxt(
        loss_path,
        np.asarray(loss_rows, dtype=np.float64),
        delimiter=",",
        header=header,
        comments="",
    )

    save_apinn_snapshot(
        model,
        label_path=data_file_str,
        out_dir=str(out_dir),
        suffix="final",
        plot_panels=True,
        epoch=epoch_offset,
    )

    print(f"[DONE] APINN loss csv: {loss_path}")
    print(f"[DONE] APINN final outputs dir: {out_dir}")
    return model


if __name__ == "__main__":
    main()
