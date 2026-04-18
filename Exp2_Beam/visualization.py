"""
Visualization functions for LS-PINN Beam project.
"""
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

from config import EI_REAL, TRUE_JUMPS, E_BG, E_DEF_1, E_DEF_2, I, EIKONAL_DELTA_EPS
from utils import heaviside_derivative


def _phi_zero_roots(x_vals: np.ndarray, phi_vals: np.ndarray, tol: float = 1e-12) -> List[float]:
    x = np.asarray(x_vals, dtype=np.float64).reshape(-1)
    phi = np.asarray(phi_vals, dtype=np.float64).reshape(-1)
    if x.size == 0 or phi.size == 0 or x.size != phi.size:
        return []

    idx = np.argsort(x)
    x = x[idx]
    phi = phi[idx]

    roots: List[float] = []

    zero_idx = np.where(np.abs(phi) <= tol)[0]
    roots.extend(float(x[i]) for i in zero_idx)

    for i in range(len(x) - 1):
        fl, fr = phi[i], phi[i + 1]
        if fl * fr < 0:
            xl, xr = x[i], x[i + 1]
            root = xl + abs(fl) / (abs(fl) + abs(fr) + tol) * (xr - xl)
            roots.append(float(root))

    if not roots:
        return []

    roots = sorted(roots)
    merged = [roots[0]]
    for r in roots[1:]:
        if abs(r - merged[-1]) > 1e-8:
            merged.append(r)
    return merged


def plot_sampling_points(
    xy_fit: torch.Tensor,
    label_fit: torch.Tensor,
    *,
    ranges: Optional[List[Tuple[float, float]]] = None,
    title: str = "Sampling points",
    show: bool = True,
) -> None:
    """
    Plot sampling points with label values.
    
    Args:
        xy_fit: Coordinate tensor [N, 1]
        label_fit: Label tensor [N, 1]
        ranges: Local refinement regions
        title: Plot title
    """
    xy_np = xy_fit.detach().cpu().numpy()
    label_np = label_fit.detach().cpu().numpy()

    plt.figure(figsize=(5, 5))

    x_vals = xy_np[:, 0]
    plt.plot(x_vals, label_np.flatten(), color='b', label="Label curve", zorder=2)
    plt.scatter(x_vals, label_np.flatten(), color='r', s=50, label="Sampling points", zorder=5)

    if ranges:
        for (x_start, x_end) in ranges:
            plt.axvline(x=x_start, color='g', linestyle='--', lw=1, label=f"Range ({x_start}, {x_end})")
            plt.axvline(x=x_end, color='g', linestyle='--', lw=1)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("Label Data (Displacement or Strain)")
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


def plot_phi_and_phi_tar(
    x: torch.Tensor,
    phi: torch.Tensor,
    phi_tar: torch.Tensor,
    *,
    show: bool = True,
) -> None:
    """
    Plot current phi and target phi side by side.
    
    Args:
        x: Position tensor
        phi: Current phi values
        phi_tar: Target phi values
    """
    x_np = x.detach().cpu().numpy()
    phi_np = phi.detach().cpu().numpy().flatten()
    phi_tar_np = phi_tar.detach().cpu().numpy().flatten()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(x_np, phi_np, c=phi_np, cmap='coolwarm', s=20)
    plt.colorbar(label="phi value")
    plt.title("phi (current)")
    plt.xlabel("x")
    plt.ylabel("phi")
    plt.axhline(0, color='black', lw=1, linestyle='--')

    plt.subplot(1, 2, 2)
    plt.scatter(x_np, phi_tar_np, c=phi_tar_np, cmap='coolwarm', s=20)
    plt.colorbar(label="phi_tar value")
    plt.title("phi_tar (target)")
    plt.xlabel("x")
    plt.ylabel("phi_tar")
    plt.axhline(0, color='red', lw=1, linestyle='--')
    
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()


def plot_loss_curves(
    loss_terms_hist: list,
    *,
    epochs_axis: Optional[list] = None,
    show: bool = True,
) -> None:
    """
    Plot loss term curves over training.
    
    Args:
        loss_terms_hist: List of loss term dictionaries
    """
    if not loss_terms_hist:
        return
        
    plt.figure()
    if epochs_axis is None:
        epochs_axis = range(1, len(loss_terms_hist) + 1)
    for name in loss_terms_hist[0].keys():
        plt.semilogy(epochs_axis, [entry[name] for entry in loss_terms_hist], label=name)
    plt.title("Loss terms")
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.legend()
    if show:
        plt.show()
    else:
        plt.close()


def save_beam_phi_snapshot(
    model,
    *,
    label_path: str = "Beam.txt",
    out_dir: str = "beam_bechmark_viz",
    suffix: Optional[int] = None,
    plot_panels: bool = True,
    true_jumps: Tuple[float, float] = TRUE_JUMPS,
    EI_real: float = EI_REAL,
    epoch_offset_global: int = 0,
) -> Tuple[str, str]:
    """
    Save phi snapshot and multi-panel visualization.
    
    Args:
        model: Neural network model
        label_path: Path to label data file
        out_dir: Output directory
        suffix: File suffix (uses epoch or timestamp if None)
        plot_panels: Whether to generate multi-panel plot
        true_jumps: True stiffness jump locations
        EI_real: Reference EI value
        epoch_offset_global: Current epoch count
        
    Returns:
        phi_png: Path to phi plot
        phi_npz: Path to phi data
    """
    device = next(model.parameters()).device

    # Load labels if available
    x_lab = theta_lab = M_lab = V_lab = eps_lab = u_lab = None
    if label_path is not None and os.path.exists(label_path):
        lab = np.loadtxt(label_path, usecols=(0, 2, 3, 4, 5, 6), comments='%')
        x_lab = torch.from_numpy(lab[:, 0]).float()
        theta_lab = torch.from_numpy(lab[:, 2]).float()
        M_lab = torch.from_numpy(lab[:, 3] / EI_real).float()
        V_lab = torch.from_numpy(-lab[:, 4] / EI_real).float()
        eps_lab = torch.from_numpy(lab[:, 5]).float()
        u_lab = torch.from_numpy(lab[:, 1]).float()

    def EI_true(x):
        return torch.where(
            x < 0.3,
            torch.full_like(x, E_BG * I),
            torch.where(x < 0.6, torch.full_like(x, E_DEF_1 * I), torch.full_like(x, E_DEF_2 * I))
        )

    # Forward pass
    if x_lab is not None:
        x = x_lab.to(device).unsqueeze(1).requires_grad_(True)
    else:
        x = torch.linspace(0, 1, 1024, device=device).unsqueeze(1).requires_grad_(True)

    with torch.no_grad():
        EI_true_np = EI_true(x.detach().cpu()).numpy() / EI_real

    out = model(x)
    phi, u1, u2 = out[:3]
    theta1, theta2 = out[3:5]
    kappa1, kappa2 = out[5:7]
    M1, M2 = out[7:9]
    V1, V2 = out[9:11]
    EI1, EI2 = out[11:13]

    mask_pos = phi >= 0

    def pick(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        return torch.where(mask_pos, t1, t2)

    # Autograd derivatives
    du1_dx = torch.autograd.grad(u1, x, torch.ones_like(u1), create_graph=True)[0]
    du2_dx = torch.autograd.grad(u2, x, torch.ones_like(u2), create_graph=True)[0]
    d2u1_dx2 = torch.autograd.grad(theta1, x, torch.ones_like(theta1), create_graph=True)[0]
    d2u2_dx2 = torch.autograd.grad(theta2, x, torch.ones_like(theta2), create_graph=True)[0]
    dkappa1 = torch.autograd.grad(kappa1, x, torch.ones_like(kappa1), create_graph=True)[0]
    dkappa2 = torch.autograd.grad(kappa2, x, torch.ones_like(kappa2), create_graph=True)[0]

    Q_dNN1 = torch.autograd.grad(V1, x, torch.ones_like(V1), create_graph=True)[0]
    Q_dNN2 = torch.autograd.grad(V2, x, torch.ones_like(V2), create_graph=True)[0]

    to_np = lambda t: t.detach().cpu().squeeze().numpy()
    x_np = to_np(x)

    u_NN = pick(u1, u2)
    theta_NN = pick(theta1, theta2)
    kappa_NN = pick(kappa1, kappa2)
    dkappa_NN = pick(dkappa1, dkappa2)
    M_NN = pick(M1, M2)
    V_NN = pick(V1, V2)
    EI_NN = pick(EI1, EI2)
    Q_dNN = pick(Q_dNN1, Q_dNN2)

    theta_AUTO = pick(du1_dx, du2_dx)
    kappa_AUTO = pick(d2u1_dx2, d2u2_dx2)
    dkappa_AUTO = pick(dkappa1, dkappa2)
    M1_AUTO = EI1 * kappa_NN
    M2_AUTO = EI2 * kappa_NN
    M_AUTO = pick(M1_AUTO, M2_AUTO)
    V1_AUTO = torch.autograd.grad(M1, x, torch.ones_like(M1), create_graph=True)[0]
    V2_AUTO = torch.autograd.grad(M2, x, torch.ones_like(M2), create_graph=True)[0]
    V_AUTO = pick(V1_AUTO, V2_AUTO)

    # File suffix
    if suffix is None:
        try:
            suffix = int(epoch_offset_global)
        except Exception:
            suffix = int(time.time())
    if isinstance(suffix, str):
        suffix_str = suffix
    else:
        suffix_str = f"{int(suffix):08d}"

    os.makedirs(out_dir, exist_ok=True)

    # Save phi plot
    phi_png = os.path.join(out_dir, f"phi_{suffix_str}.png")
    phi_npz = os.path.join(out_dir, f"phi_{suffix_str}.npz")
    phi_np = to_np(phi)
    delta_phi_np = to_np(heaviside_derivative(phi, epsilon=EIKONAL_DELTA_EPS))
    
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 10,
    })
    fig, ax_phi = plt.subplots(figsize=(3.8, 2.5), dpi=200)
    line_phi = ax_phi.plot(x_np, phi_np, 'b-', lw=0.8, label=r'$\phi(x)$')[0]
    for vx in (true_jumps or []):
        ax_phi.axvline(vx, color='r', ls=':', lw=1.0)
    ax_phi.axhline(0, color='r', ls=':', lw=1.0)
    ax_phi.set_xlabel("x")
    ax_phi.set_ylabel(r'$\phi$', color='b')
    ax_phi.tick_params(axis='y', labelcolor='b')
    ax_phi.grid(True, ls='--', alpha=0.3)

    ax_delta = ax_phi.twinx()
    line_delta = ax_delta.plot(
        x_np, delta_phi_np, color='darkorange', lw=0.8, label=r"$H'_{\epsilon}(\phi)$"
    )[0]
    ax_delta.set_ylabel(r"$H'_{\epsilon}(\phi)$", color='darkorange')
    ax_delta.tick_params(axis='y', labelcolor='darkorange')

    ax_phi.legend([line_phi, line_delta], [line_phi.get_label(), line_delta.get_label()], loc="upper right")
    fig.tight_layout()
    fig.savefig(phi_png, bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)

    def _np_or_empty(t: Optional[torch.Tensor]) -> np.ndarray:
        if t is None:
            return np.array([], dtype=np.float32)
        return to_np(t).astype(np.float32)

    rel_err_eps = 1e-12
    u_nn_np = to_np(u_NN)
    theta_nn_np = to_np(theta_NN)
    kappa_nn_np = to_np(kappa_NN)
    M_nn_np = to_np(M_NN)
    V_nn_np = to_np(V_NN)
    EI_nn_np = to_np(EI_NN)
    theta_auto_np = to_np(theta_AUTO)
    kappa_auto_np = to_np(kappa_AUTO)
    M_auto_np = to_np(M_AUTO)
    V_auto_np = to_np(V_AUTO)
    q_nn_np = to_np(Q_dNN)
    u_lab_np = _np_or_empty(u_lab)
    theta_lab_np = _np_or_empty(theta_lab)
    kappa_lab_np = _np_or_empty(eps_lab / (0.05 / 2) if eps_lab is not None else None)
    M_lab_np = _np_or_empty(M_lab)
    V_lab_np = _np_or_empty(V_lab)

    def _relative_error(y_pred: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
        denom = np.maximum(np.abs(y_ref), rel_err_eps)
        return np.abs(y_pred - y_ref) / denom

    def _relative_l2_error(y_pred: np.ndarray, y_ref: np.ndarray) -> float:
        ref_norm = np.linalg.norm(y_ref.ravel())
        denom = max(ref_norm, rel_err_eps)
        return float(np.linalg.norm((y_pred - y_ref).ravel()) / denom)

    rel_err_u = _relative_error(u_nn_np, u_lab_np) if u_lab_np.size else np.array([], dtype=np.float32)
    rel_err_theta = _relative_error(theta_nn_np, theta_lab_np) if theta_lab_np.size else np.array([], dtype=np.float32)
    rel_err_kappa = _relative_error(kappa_nn_np, kappa_lab_np) if kappa_lab_np.size else np.array([], dtype=np.float32)
    rel_err_M = _relative_error(M_nn_np, M_lab_np) if M_lab_np.size else np.array([], dtype=np.float32)
    rel_err_V = _relative_error(V_nn_np, V_lab_np) if V_lab_np.size else np.array([], dtype=np.float32)
    rel_err_EI = _relative_error(EI_nn_np, EI_true_np.squeeze())
    rel_l2_u = _relative_l2_error(u_nn_np, u_lab_np) if u_lab_np.size else None
    rel_l2_theta = _relative_l2_error(theta_nn_np, theta_lab_np) if theta_lab_np.size else None
    rel_l2_kappa = _relative_l2_error(kappa_nn_np, kappa_lab_np) if kappa_lab_np.size else None
    rel_l2_M = _relative_l2_error(M_nn_np, M_lab_np) if M_lab_np.size else None
    rel_l2_V = _relative_l2_error(V_nn_np, V_lab_np) if V_lab_np.size else None
    rel_l2_EI = _relative_l2_error(EI_nn_np, EI_true_np.squeeze())

    rel_err_cumulative = np.zeros_like(x_np, dtype=np.float32)
    for rel_curve in (rel_err_u, rel_err_theta, rel_err_kappa, rel_err_M, rel_err_V, rel_err_EI):
        if rel_curve.size:
            rel_err_cumulative += rel_curve.astype(np.float32)
    phi_zero_roots = _phi_zero_roots(x_np, phi_np)

    np.savez_compressed(
        phi_npz,
        x=x_np.astype(np.float32),
        phi=phi_np.astype(np.float32),
        delta_phi=delta_phi_np.astype(np.float32),
        EI_pred=to_np(EI_NN).astype(np.float32),
        EI_true=EI_true_np.astype(np.float32),
        u_NN=to_np(u_NN).astype(np.float32),
        theta_NN=to_np(theta_NN).astype(np.float32),
        kappa_NN=to_np(kappa_NN).astype(np.float32),
        M_NN=to_np(M_NN).astype(np.float32),
        V_NN=to_np(V_NN).astype(np.float32),
        Q_NN=to_np(Q_dNN).astype(np.float32),
        theta_AUTO=to_np(theta_AUTO).astype(np.float32),
        kappa_AUTO=to_np(kappa_AUTO).astype(np.float32),
        M_AUTO=to_np(M_AUTO).astype(np.float32),
        V_AUTO=to_np(V_AUTO).astype(np.float32),
        u_lab=u_lab_np,
        theta_lab=theta_lab_np,
        kappa_lab=kappa_lab_np,
        M_lab=M_lab_np,
        V_lab=V_lab_np,
        eps_lab=_np_or_empty(eps_lab),
        rel_err_u=rel_err_u.astype(np.float32),
        rel_err_theta=rel_err_theta.astype(np.float32),
        rel_err_kappa=rel_err_kappa.astype(np.float32),
        rel_err_M=rel_err_M.astype(np.float32),
        rel_err_V=rel_err_V.astype(np.float32),
        rel_err_EI=rel_err_EI.astype(np.float32),
        rel_err_cumulative=rel_err_cumulative.astype(np.float32),
        rel_l2_u=np.array(np.nan if rel_l2_u is None else rel_l2_u, dtype=np.float32),
        rel_l2_theta=np.array(np.nan if rel_l2_theta is None else rel_l2_theta, dtype=np.float32),
        rel_l2_kappa=np.array(np.nan if rel_l2_kappa is None else rel_l2_kappa, dtype=np.float32),
        rel_l2_M=np.array(np.nan if rel_l2_M is None else rel_l2_M, dtype=np.float32),
        rel_l2_V=np.array(np.nan if rel_l2_V is None else rel_l2_V, dtype=np.float32),
        rel_l2_EI=np.array(rel_l2_EI, dtype=np.float32),
    )

    # Multi-panel plot
    panel_png = None
    if plot_panels:
        panels = [
            ("u", u_nn_np, None, u_lab_np if u_lab_np.size else None, "u", rel_l2_u),
            ("theta=du/dx", theta_nn_np, theta_auto_np, theta_lab_np if theta_lab_np.size else None, "theta", rel_l2_theta),
            ("kappa=d2u/dx2", kappa_nn_np, kappa_auto_np, kappa_lab_np if kappa_lab_np.size else None, "kappa", rel_l2_kappa),
            ("M", M_nn_np, M_auto_np, M_lab_np if M_lab_np.size else None, "M", rel_l2_M),
            ("V", V_nn_np, V_auto_np, V_lab_np if V_lab_np.size else None, "V", rel_l2_V),
            ("Q", q_nn_np, q_nn_np, None, "Q", None),
            ("EI", EI_nn_np, None, EI_true_np.squeeze(), "EI", rel_l2_EI),
            ("phi", phi_np, None, None, "phi", None),
        ]
        nrow = (len(panels) + 2) // 3
        fig, axs = plt.subplots(nrow, 3, figsize=(15, 4 * nrow))
        axs = axs.ravel()
        
        for ax, (ttl, y_nn, y_auto, y_lab, ylabel, rel_l2) in zip(axs, panels):
            ax.plot(x_np, y_nn, 'b-', lw=0.9, label="NN")
            if y_auto is not None:
                ax.plot(x_np, y_auto, 'g--', lw=1.0, label="auto-diff")
            if y_lab is not None:
                ax.scatter(
                    x_lab.numpy() if x_lab is not None else x_np,
                    y_lab, c='r', marker='^', s=8, edgecolors='k', label="label"
                )
            for vx in (true_jumps or []):
                ax.axvline(vx, color='r', ls=':', lw=1.0)
            if ttl != "EI" and ttl != "Q":
                ax.axhline(0, color='k', ls=':', lw=0.8, alpha=0.6)
            if ttl == "V":
                ax.set_ylim(min(y_nn.min(), -0.5), max(y_nn.max(), 0.5))
            ax.set_xlabel("x")
            ax.set_ylabel(ylabel)
            ax.set_title(ttl)
            ax.grid(True, ls='--', alpha=0.3)
            if rel_l2 is not None:
                ax.text(
                    0.02, 0.98, f"L2 rel={rel_l2:.1e}",
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="0.7"),
                )
            ax.legend(fontsize=11)
        
        for ax in axs[len(panels):]:
            ax.clear()
            ax.set_title("phi=0 roots")
            ax.axis("off")
            if phi_zero_roots:
                root_lines = [
                    ", ".join(f"{r:.5f}" for r in phi_zero_roots[i:i + 4])
                    for i in range(0, len(phi_zero_roots), 4)
                ]
                root_text = "\n".join(root_lines)
            else:
                root_text = "No root"
            ax.text(
                0.03, 0.97, root_text,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=12,
                family="monospace",
            )
        
        plt.tight_layout()
        panel_png = os.path.join(out_dir, f"panels_{suffix_str}.png")
        plt.savefig(panel_png, bbox_inches="tight", pad_inches=0.02)
        plt.close()

    print(f"[SAVE] phi-curve  -> {phi_png}")
    print(f"[SAVE] phi-data   -> {phi_npz}")
    if panel_png:
        print(f"[SAVE] panels     -> {panel_png}")

    return phi_png, phi_npz
