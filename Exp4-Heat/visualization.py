from typing import Iterable, Optional, Tuple

import copy
import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from matplotlib.ticker import MaxNLocator, FuncFormatter, ScalarFormatter

import config
from pde import div_kgrad
from utils import masked_partition_value


def local_velocity_viz_cpu(
    model_cpu,
    xy: torch.Tensor,
    get_f1_f2,
    *,
    band_eps: float = 2e-2,
    h: float = 3e-2,
    tau: float = 1e-6,
    eps: float = 1e-8,
    clip_q: float = 0.99,
):
    device_here = torch.device("cpu")
    xy = xy.detach().clone().to(device_here).requires_grad_(True)

    phi, T1, T2 = model_cpu(xy)

    f1, f2 = get_f1_f2()
    f1 = f1.detach().to(device_here)
    f2 = f2.detach().to(device_here)

    R1 = div_kgrad(T1, f1, xy)
    R2 = div_kgrad(T2, f2, xy)
    r_val = masked_partition_value(phi, R1, R2).detach().abs()

    band_mask = (phi.detach().abs() < band_eps).squeeze()
    if band_mask.sum() == 0:
        return torch.zeros_like(phi)

    xy_band = xy[band_mask]
    phi_band = phi.detach()[band_mask]
    dmat = torch.cdist(xy_band.detach(), xy.detach(), p=2)
    knn = dmat < h

    pos_side = (phi.detach() > 0).T
    neg_side = ~pos_side

    g_pos_mat = (knn & pos_side).float()
    g_neg_mat = (knn & neg_side).float()

    r_pos = g_pos_mat @ r_val / g_pos_mat.sum(1, keepdim=True).clamp_min(1)
    r_neg = g_neg_mat @ r_val / g_neg_mat.sum(1, keepdim=True).clamp_min(1)

    delta = r_neg - r_pos
    if delta.numel() == 0:
        return torch.zeros_like(phi)

    scale = torch.quantile(delta.abs(), clip_q) + eps
    vel_band = torch.tanh(delta / scale)
    vel_band = vel_band * (1 / np.pi) * band_eps / (phi_band**2 + band_eps**2 + eps)

    Vn_full = torch.zeros_like(phi.detach())
    Vn_full[band_mask] = vel_band.detach()
    return Vn_full


def local_velocity_grad_viz_cpu(
    model_cpu,
    xy: torch.Tensor,
    get_f1_f2,
    *,
    band_eps: float = 0.02,
    h: float = 0.03,
    tau: float = 1e-4,
    eps: float = 1e-8,
):
    device_here = torch.device("cpu")
    xy = xy.detach().clone().to(device_here).requires_grad_(True)

    phi, T1, T2 = model_cpu(xy)

    f1, f2 = get_f1_f2()
    f1 = f1.detach().to(device_here)
    f2 = f2.detach().to(device_here)

    R1 = div_kgrad(T1, f1, xy, keep_graph=True)
    R2 = div_kgrad(T2, f2, xy, keep_graph=True)

    def grad_norm_cpu(scalar_field, pts):
        g = torch.autograd.grad(
            scalar_field,
            pts,
            grad_outputs=torch.ones_like(scalar_field),
            create_graph=False,
            retain_graph=True,
        )[0]
        return g.norm(dim=1, keepdim=True)

    gR1 = grad_norm_cpu(R1, xy)
    gR2 = grad_norm_cpu(R2, xy)

    w_pos = torch.relu(phi)
    w_neg = torch.relu(-phi)
    g_val = (w_pos * gR1 + w_neg * gR2).detach()
    phi_det = phi.detach()

    band_mask = (phi_det.abs() < band_eps).squeeze()
    if not band_mask.any():
        return torch.zeros_like(phi_det)

    xy_band = xy[band_mask]
    phi_band = phi_det[band_mask]

    dmat = torch.cdist(xy_band.detach(), xy.detach(), p=2)
    knn = dmat < h
    pos_side = (phi_det > 0).T
    neg_side = ~pos_side

    m_pos = (knn & pos_side).float()
    m_neg = (knn & neg_side).float()

    g_pos = m_pos @ g_val / m_pos.sum(1, keepdim=True).clamp_min(1)
    g_neg = m_neg @ g_val / m_neg.sum(1, keepdim=True).clamp_min(1)

    delta = g_neg - g_pos
    delta = delta / (delta.abs().mean() + eps)
    vel_band = delta * (delta.abs() / (delta.abs() + tau))
    vel_band = vel_band * (1 / np.pi) * band_eps / (phi_band.pow(2) + band_eps**2 + eps)

    scale = torch.quantile(vel_band.abs(), 0.95) + eps
    vel_band = torch.tanh(vel_band / scale)

    Vn_full = torch.zeros_like(phi_det)
    Vn_full[band_mask] = vel_band.detach()
    return Vn_full


def local_velocity_CV_full_viz_cpu(
    model_cpu,
    xy: torch.Tensor,
    get_f1_f2,
    *,
    band_eps: float = 2e-2,
    h: float = 3e-2,
    eps: float = 1e-8,
    tau: float = 1e-8,
    clip_q: float = 0.99,
    fallback_circles: Optional[Iterable[Tuple[float, float, float]]] = None,
):
    device_here = torch.device("cpu")
    xy = xy.detach().clone().to(device_here).requires_grad_(True)

    phi, T1, T2 = model_cpu(xy)

    f1, f2 = get_f1_f2()
    f1 = f1.detach().to(device_here)
    f2 = f2.detach().to(device_here)

    R1 = div_kgrad(T1, f1, xy)
    R2 = div_kgrad(T2, f2, xy)
    R_abs = masked_partition_value(phi, R1, R2).abs().detach()

    band_mask = (phi.detach().abs() < band_eps).squeeze()
    Fai_empty = False
    if not band_mask.any() and fallback_circles:
        cx, cy, r0 = list(fallback_circles)[0]
        dist = ((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2).sqrt()
        band_mask = ((dist - r0).abs() < band_eps)
    if not band_mask.any():
        return torch.zeros_like(phi), True

    R_b = R_abs[band_mask]
    phi_b = phi[band_mask]

    inside_mask = (phi.detach() > 0).squeeze()
    C1 = R_abs[inside_mask].mean() if inside_mask.any() else torch.tensor(
        0.0, device=device_here
    )
    C2 = R_abs[~inside_mask].mean() if (~inside_mask).any() else torch.tensor(
        0.0, device=device_here
    )

    CV = (R_b - C1) ** 2 - (R_b - C2) ** 2
    delta = (band_eps / np.pi) / (phi_b.detach() ** 2 + band_eps**2 + eps)
    Vn_b = delta * CV

    s0 = torch.quantile(Vn_b.abs(), clip_q) + eps
    Vn_b = torch.tanh(Vn_b / s0)

    phi_max = phi.abs().max()
    vn_trim = torch.quantile(Vn_b.abs(), clip_q) + tau
    scale = phi_max / vn_trim
    Vn_b = Vn_b * scale * 2.0

    Vn_full = torch.zeros_like(phi)
    Vn_full[band_mask] = Vn_b.detach()
    return Vn_full, Fai_empty


def _compute_phi_next_on_grid_heat_cpu(
    model_cpu,
    XY_grid_np: np.ndarray,
    n: int,
    vel_type: str,
    dt: float,
    band_eps_vel: float,
    h_vel: float,
    tau_vel: float,
    clip_q_vel: float,
    xy_fit_local: Optional[torch.Tensor],
    T_fit_local: Optional[torch.Tensor],
    get_f1_f2,
    fallback_circles: Optional[Iterable[Tuple[float, float, float]]] = None,
):
    device_cpu = torch.device("cpu")
    XY_t = torch.from_numpy(XY_grid_np.astype(np.float32)).to(device_cpu)
    XY_t.requires_grad_(True)

    phi_now, T1_now, T2_now = model_cpu(XY_t)

    vel_name = vel_type.upper()
    Fai_empty = False

    if vel_name == "PDE":
        Vn_full = local_velocity_viz_cpu(
            model_cpu,
            XY_t,
            get_f1_f2,
            band_eps=band_eps_vel,
            h=h_vel,
            tau=tau_vel,
            clip_q=clip_q_vel,
        )
    elif vel_name == "GRAD":
        Vn_full = local_velocity_grad_viz_cpu(
            model_cpu,
            XY_t,
            get_f1_f2,
            band_eps=band_eps_vel,
            h=h_vel,
            tau=tau_vel,
        )
    elif vel_name == "CV":
        Vn_full, Fai_empty = local_velocity_CV_full_viz_cpu(
            model_cpu,
            XY_t,
            get_f1_f2,
            band_eps=band_eps_vel,
            h=h_vel,
            tau=tau_vel,
            clip_q=clip_q_vel,
            fallback_circles=fallback_circles,
        )
    elif vel_name == "DATA":
        Vn_full = local_velocity_viz_cpu(
            model_cpu,
            XY_t,
            get_f1_f2,
            band_eps=band_eps_vel,
            h=h_vel,
            tau=tau_vel,
            clip_q=clip_q_vel,
        )
    else:
        raise ValueError(f"Unknown vel_type '{vel_type}'")

    grad_phi = torch.autograd.grad(
        phi_now,
        XY_t,
        grad_outputs=torch.ones_like(phi_now),
        create_graph=False,
        retain_graph=False,
    )[0]
    grad_norm = grad_phi.norm(dim=1, keepdim=True).clamp_min(1e-6)

    if Fai_empty:
        phi_next = phi_now + dt * Vn_full
    else:
        phi_next = phi_now + dt * Vn_full * grad_norm

    phi_now_np = phi_now.detach().cpu().numpy().reshape(n, n).astype(np.float32)
    phi_next_np = phi_next.detach().cpu().numpy().reshape(n, n).astype(np.float32)

    del XY_t, phi_now, T1_now, T2_now, Vn_full, grad_phi, grad_norm, phi_next
    return phi_now_np, phi_next_np


def plot_residual_scatter_heat(
    model,
    *,
    kind: str = "pde",
    xy_fit: Optional[torch.Tensor] = None,
    T_fit: Optional[torch.Tensor] = None,
    n: int = 200,
    bbox: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    batch_size: int = 4096,
    cross_params: Tuple[float, float, float, float] = (0.40, 0.60, 0.05, 0.15),
    cmap: str = "cividis",
    savepath: Optional[str] = None,
    show: bool = False,
    show_next: bool = True,
    vel_type_for_next: str = "PDE",
    dt_next: float = 1e-2,
    band_eps_vel: float = 0.02,
    h_vel: float = 0.03,
    tau_vel: float = 1e-4,
    clip_q_vel: float = 0.99,
    get_f1_f2=None,
    fallback_circles: Optional[Iterable[Tuple[float, float, float]]] = None,
    save_npz_path: Optional[str] = None,
):
    device_cpu = torch.device("cpu")
    model_cpu = copy.deepcopy(model).to(device_cpu).eval()

    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, n, dtype=np.float32)
    ys = np.linspace(ymin, ymax, n, dtype=np.float32)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    XY_grid = np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32)

    phi_model_flat = np.zeros(XY_grid.shape[0], dtype=np.float32)
    for i0 in range(0, XY_grid.shape[0], batch_size):
        i1 = min(XY_grid.shape[0], i0 + batch_size)
        XY_b = torch.from_numpy(XY_grid[i0:i1]).to(device_cpu)
        with torch.no_grad():
            phi_b, T1_b, T2_b = model_cpu(XY_b)
        phi_model_flat[i0:i1] = phi_b.detach().cpu().numpy().reshape(-1).astype(np.float32)
    phi_model_map = phi_model_flat.reshape(n, n)

    xc, yc, w_half, l_half = cross_params
    h_mask = (np.abs(Xg - xc) <= w_half) & (np.abs(Yg - yc) <= l_half)
    v_mask = (np.abs(Yg - yc) <= w_half) & (np.abs(Xg - xc) <= l_half)
    cross_mask = (h_mask | v_mask).astype(np.float32)
    phi_true_map = cross_mask

    if show_next:
        _, phi_next_map = _compute_phi_next_on_grid_heat_cpu(
            model_cpu,
            XY_grid,
            n,
            vel_type_for_next,
            dt_next,
            band_eps_vel,
            h_vel,
            tau_vel,
            clip_q_vel,
            xy_fit,
            T_fit,
            get_f1_f2,
            fallback_circles=fallback_circles,
        )
    else:
        phi_next_map = None

    f1_val, f2_val = get_f1_f2()
    f1_val = f1_val.detach().to(device_cpu)
    f2_val = f2_val.detach().to(device_cpu)

    if kind == "data":
        if xy_fit is None or T_fit is None:
            raise ValueError("kind='data' requires xy_fit and T_fit")

        XY_lab_np = xy_fit.detach().cpu().numpy().astype(np.float32)
        T_lab_np = T_fit.detach().cpu().numpy().reshape(-1, 1).astype(np.float32)
        XY_lab_t = torch.from_numpy(XY_lab_np).to(device_cpu)

        with torch.no_grad():
            phi_b, T1_b, T2_b = model_cpu(XY_lab_t)
            T_pred = masked_partition_value(phi_b, T1_b, T2_b)

        diff = (T_pred.cpu().numpy() - T_lab_np)
        RES_val = np.abs(diff).reshape(-1).astype(np.float32)
        XY_residual = XY_lab_np

    elif kind in ("pde", "grad"):
        R_soft_flat = np.zeros(XY_grid.shape[0], dtype=np.float32)
        for i0 in range(0, XY_grid.shape[0], batch_size):
            i1 = min(XY_grid.shape[0], i0 + batch_size)
            XY_b = torch.from_numpy(XY_grid[i0:i1]).to(device_cpu)
            XY_b_req = XY_b.clone().requires_grad_(True)

            phi_b, T1_b, T2_b = model_cpu(XY_b_req)

            R1_b = div_kgrad(T1_b, f1_val, XY_b_req)
            R2_b = div_kgrad(T2_b, f2_val, XY_b_req)

            Rmag1 = torch.abs(R1_b)
            Rmag2 = torch.abs(R2_b)

            R_soft = masked_partition_value(phi_b, Rmag1, Rmag2)

            R_soft_flat[i0:i1] = (
                R_soft.detach().cpu().numpy().reshape(-1).astype(np.float32)
            )

        if kind == "pde":
            RES_val = R_soft_flat
            XY_residual = XY_grid.astype(np.float32)
        else:
            R_grid = R_soft_flat.reshape(n, n)
            dx = (xmax - xmin) / (n - 1)
            dy = (ymax - ymin) / (n - 1)
            gx = np.zeros_like(R_grid, dtype=np.float32)
            gy = np.zeros_like(R_grid, dtype=np.float32)

            gx[1:-1, :] = (R_grid[2:, :] - R_grid[:-2, :]) / (2 * dx)
            gy[:, 1:-1] = (R_grid[:, 2:] - R_grid[:, :-2]) / (2 * dy)

            grad_norm = np.sqrt(gx * gx + gy * gy).astype(np.float32)
            RES_val = grad_norm.ravel()
            XY_residual = XY_grid.astype(np.float32)
    else:
        raise ValueError(f"unknown kind '{kind}'")

    RES_plot = RES_val.copy()
    vmax_lin = np.percentile(RES_plot, 99.0)
    if vmax_lin <= 0:
        vmax_lin = RES_plot.max() if RES_plot.max() > 0 else 1.0
    RES_clip = np.clip(RES_plot, 0.0, vmax_lin)

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    fig, ax = plt.subplots(dpi=300, figsize=(3.8, 2.5))

    sc = ax.scatter(
        XY_residual[:, 0],
        XY_residual[:, 1],
        c=RES_clip,
        s=(40 if kind == "data" else 8),
        marker=("o" if kind == "data" else "s"),
        linewidths=0.0,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax_lin,
    )

    ax.contour(
        Xg,
        Yg,
        phi_true_map,
        levels=[0.5],
        colors="black",
        linewidths=1.0,
        linestyles="-",
    )

    ax.contour(
        Xg,
        Yg,
        phi_model_map,
        levels=[0.0],
        colors="red",
        linewidths=1.0,
        linestyles="--",
    )

    if show_next and (phi_next_map is not None):
        ax.contour(
            Xg,
            Yg,
            phi_next_map,
            levels=[0.0],
            colors="lime",
            linewidths=0.7,
            linestyles="dashdot",
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.locator_params(axis="x", nbins=4)
    ax.locator_params(axis="y", nbins=4)

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.0, ls="-", label="Exact"),
        Line2D([0], [0], color="red", lw=1.0, ls="--", label="Current"),
    ]
    if show_next and (phi_next_map is not None):
        legend_handles.append(
            Line2D([0], [0], color="lime", lw=1.0, ls="-.", label="Next")
        )

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
    cbar.ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))
    offset_txt = cbar.ax.yaxis.get_offset_text()
    offset_txt.set_x(2.4)
    offset_txt.set_va("bottom")
    offset_txt.set_fontsize(7)

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        print("[SAVE]", savepath)
    if save_npz_path is not None:
        np.savez_compressed(
            save_npz_path,
            XY_residual=XY_residual,
            RES_val=RES_val.astype(np.float32),
            phi_model=phi_model_map.astype(np.float32),
            phi_true=phi_true_map.astype(np.float32),
            phi_next=None if phi_next_map is None else phi_next_map.astype(np.float32),
            bbox=np.asarray([xmin, xmax, ymin, ymax], dtype=np.float32),
        )
        print("[SAVE]", save_npz_path)

    if show:
        plt.show()
    else:
        plt.close(fig)
    del model_cpu

    return {
        "XY_residual": XY_residual,
        "RES_val": RES_val,
        "phi_model": phi_model_map,
        "phi_next": phi_next_map,
        "phi_true": phi_true_map,
    }


def plot_E_from_lossfile(
    filename: str,
    names: Tuple[str, str] = ("E1", "E2"),
    true_vals: Optional[Tuple[float, float]] = None,
    smooth: Optional[int] = None,
    savepath: Optional[str] = None,
    show: bool = False,
    add_pred_markers: bool = True,
    marker_every: str = "auto",
    marker_epochs: Optional[Iterable[int]] = None,
    pred_markers: Tuple[str, ...] = ("o", "^", "s", "D"),
    markersize: float = 4.0,
    marker_lw: float = 0.9,
    true_linestyles: Tuple = ("--", "-.", ":", (0, (3, 1, 1, 1))),
    true_linewidth: float = 1.0,
):
    with open(filename, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    has_header = any(ch.isalpha() for ch in first_line)
    if has_header:
        data = np.genfromtxt(filename, delimiter=",", skip_header=1)
    else:
        data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    epochs = data[:, 0].astype(int)
    E1 = data[:, -2]
    E2 = data[:, -1]
    E_vals = np.stack([E1, E2], axis=1)
    N, C = E_vals.shape

    def moving_avg(x, y, w):
        if not w or w <= 1:
            return x, y
        ker = np.ones(w, dtype=float) / float(w)
        y_s = np.convolve(y, ker, mode="valid")
        x_s = x[w - 1 :]
        return x_s, y_s

    plt.rcParams.update(
        {"font.size": 7, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7}
    )
    fig, ax = plt.subplots(figsize=(3.25, 2.3), dpi=400)

    colors = ["C0", "C1", "C2", "C3"]

    for i in range(C):
        color = colors[i % len(colors)]
        marker = pred_markers[i % len(pred_markers)]
        x_plot, y_plot = moving_avg(epochs, E_vals[:, i], smooth)

        me = None
        if add_pred_markers:
            if marker_every == "auto":
                target = 8
                step = max(len(x_plot) // max(target, 1), 1)
                me = slice(0, None, step)
            elif marker_every == "epochs":
                if marker_epochs:
                    xs = np.asarray(marker_epochs, dtype=float)
                    xs = np.clip(xs, x_plot.min(), x_plot.max())
                    idxs = np.searchsorted(x_plot, xs)
                    idxs = np.clip(idxs, 0, len(x_plot) - 1)
                    me = sorted(set(idxs.tolist()))
            else:
                me = marker_every

        ax.plot(
            x_plot,
            y_plot,
            color=color,
            lw=0.9,
            marker=(marker if add_pred_markers else None),
            markevery=me,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=marker_lw,
            markersize=markersize,
            label=f"{names[i]}",
        )

        if true_vals is not None and i < len(true_vals) and true_vals[i] is not None:
            true_y = float(true_vals[i])
            ls = true_linestyles[i % len(true_linestyles)]
            ax.axhline(true_y, color=color, lw=true_linewidth, ls=ls, label="Exact")

    ax.set_ylim(0.02, 14.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True, ls="--", alpha=0.4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.legend(frameon=False, fontsize=7, ncols=2, loc="lower right")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=400, bbox_inches="tight")
        print("Saved:", savepath)
    if show:
        plt.show()
    else:
        plt.close(fig)

    print("Final values:", {names[i]: float(E_vals[-1, i]) for i in range(C)})


def plot_loss_components_from_records(
    loss_records: Iterable[Iterable[float]],
    savepath: Optional[str] = None,
    show: bool = False,
):
    block = np.asarray(loss_records, float)
    if block.size == 0:
        print("No loss records to plot.")
        return

    x = block[:, 0].astype(int)
    col_map = dict(total=1, data=2, pde=3, bc=4, interface=5, eik=6, area=7)
    cols_to_plot = [
        col_map["data"],
        col_map["pde"],
        col_map["interface"],
        col_map["eik"],
        col_map["area"],
        col_map["total"],
    ]
    labels = ["Data", "PDE", "Interface", "Eikonal", "Area", "Total"]

    arr = np.sqrt(block[:, cols_to_plot])

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )

    BW = False
    linestyles = ["-", "--", "-.", "-", (0, (5, 1, 1, 1)), (0, (3, 1, 3, 1, 1, 1))]
    markers = ["o", "s", "^", "D", "x", "+"]
    markevery = max(1, len(x) // 10)
    ms = 3

    fig, ax = plt.subplots(dpi=500, figsize=(3.35, 2.3))

    for j in range(arr.shape[1]):
        y = arr[:, j].astype(float).copy()
        m = np.isfinite(y) & (y > 0)
        y[~m] = np.nan

        ls = linestyles[j % len(linestyles)]
        mk = markers[j % len(markers)]
        color = "black" if BW else None

        kw = dict(
            linestyle=ls,
            marker=mk,
            markevery=markevery,
            linewidth=0.6,
            markersize=ms,
            color=color,
            alpha=0.7,
        )
        if mk in {"o", "s", "^", "D"}:
            kw.update(markerfacecolor="none")
        ax.semilogy(x, y, label=labels[j], **kw)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    def sci_fmt(xv, pos):
        if xv == 0:
            return "0"
        p = int(math.log10(abs(xv)))
        base = xv / (10**p)
        if abs(base - round(base)) < 1e-6:
            base = int(round(base))
        return rf"{base}$\times 10^{{{p}}}$"

    ax.xaxis.set_major_formatter(FuncFormatter(sci_fmt))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", ls="--", alpha=0.35, linewidth=0.6)

    plt.legend(
        ncol=3,
        frameon=False,
        handlelength=2.0,
        handletextpad=0.4,
        columnspacing=0.8,
        bbox_to_anchor=(0.2, 0.75),
        fontsize=7,
    )

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0.02, transparent=True, dpi=400)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_phi_compare_with_cross_and_circle(
    model,
    *,
    cross: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 1.2, 1.2, 0.2, 0.2),
    circle_roi: Optional[Tuple[float, float, float]] = None,
    bbox: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    n: int = 200,
    batch_size: int = 4096,
    band_eps: float = 0.05,
    xy_label: Optional[torch.Tensor] = None,
    savepath: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
):
    def sdf_rect(xx, yy, xc, yc, a, b):
        qx = np.abs(xx - xc) - a
        qy = np.abs(yy - yc) - b
        qx_pos = np.maximum(qx, 0.0)
        qy_pos = np.maximum(qy, 0.0)
        outside = np.hypot(qx_pos, qy_pos)
        inside = np.maximum(qx, qy)
        return outside + np.minimum(inside, 0.0)

    def sdf_cross(xx, yy, xc, yc, Lx, Ly, wh, wv):
        phi_h = sdf_rect(xx, yy, xc, yc, a=Lx / 2.0, b=wh / 2.0)
        phi_v = sdf_rect(xx, yy, xc, yc, a=wv / 2.0, b=Ly / 2.0)
        return np.minimum(phi_h, phi_v)

    def take_phi(out):
        if torch.is_tensor(out):
            return out
        if isinstance(out, (list, tuple)):
            for v in out:
                if torch.is_tensor(v):
                    return v
        if isinstance(out, dict):
            if "phi" in out and torch.is_tensor(out["phi"]):
                return out["phi"]
            for v in out.values():
                if torch.is_tensor(v):
                    return v
        return None

    device = torch.device("cpu")
    dtype = torch.float32
    model_cpu = copy.deepcopy(model).to(device).eval()

    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, n, dtype=np.float32)
    ys = np.linspace(ymin, ymax, n, dtype=np.float32)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    XY_grid = np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32)

    phi_pred_flat = np.zeros(XY_grid.shape[0], dtype=np.float32)
    for i0 in range(0, XY_grid.shape[0], batch_size):
        i1 = min(XY_grid.shape[0], i0 + batch_size)
        XY_b = torch.from_numpy(XY_grid[i0:i1]).to(device=device, dtype=dtype)
        with torch.no_grad():
            out = model_cpu(XY_b)
            phi_b = take_phi(out)
        phi_pred_flat[i0:i1] = phi_b.detach().cpu().numpy().reshape(-1).astype(np.float32)
    phi_pred_map = phi_pred_flat.reshape(n, n)

    xc, yc, Lx, Ly, wh, wv = cross
    phi_true_map = sdf_cross(Xg, Yg, xc, yc, Lx, Ly, wh, wv).astype(np.float32)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig, ax = plt.subplots(dpi=dpi, figsize=(3.35 * 0.9, 3.3 * 0.9))

    phi_abs_map = np.abs(phi_pred_map)
    band_levels = [0.0, float(band_eps), phi_abs_map.max() + 1.0]
    band_colors = [(0.5, 0.5, 0.5, 0.30), (0, 0, 0, 0)]
    ax.contourf(Xg, Yg, phi_abs_map, levels=band_levels, colors=band_colors, antialiased=False, zorder=0)

    ax.contour(Xg, Yg, phi_true_map, levels=[0.0], colors="black", linewidths=1.0, linestyles="-", zorder=2)
    ax.contour(Xg, Yg, phi_pred_map, levels=[0.0], colors="red", linewidths=1.0, linestyles="--", zorder=3)

    label_handle = None
    if xy_label is not None:
        xy_lab_np = (
            xy_label.detach().cpu().numpy()
            if isinstance(xy_label, torch.Tensor)
            else np.asarray(xy_label, dtype=np.float32)
        )
        ax.scatter(
            xy_lab_np[:, 0],
            xy_lab_np[:, 1],
            s=6,
            facecolors="white",
            edgecolors="green",
            linewidths=0.7,
            alpha=0.6,
            marker="o",
            zorder=4,
        )
        label_handle = Line2D(
            [0],
            [0],
            marker="o",
            markersize=3,
            markerfacecolor="white",
            markeredgecolor="green",
            markeredgewidth=0.7,
            linestyle="None",
            alpha=0.6,
            label="Label points",
        )

    circle_handle = None
    if circle_roi is not None:
        cx, cy, r = circle_roi
        circ = Circle((cx, cy), r, facecolor="none", edgecolor="blue", linestyle="dashdot", linewidth=1.0, zorder=5)
        ax.add_patch(circ)
        circle_handle = Line2D([0], [0], color="blue", lw=1.0, ls="-.", label="Predicted ROI")

    legend_handles = [
        Patch(facecolor=(0.5, 0.5, 0.5, 0.3), edgecolor="none", label="Narrow band"),
        Line2D([0], [0], color="black", lw=1.0, ls="-", label="Exact"),
        Line2D([0], [0], color="red", lw=1.0, ls="--", label="Prediction"),
    ]
    if label_handle is not None:
        legend_handles.append(label_handle)
    if circle_handle is not None:
        legend_handles.append(circle_handle)

    ax.legend(
        handles=legend_handles,
        loc="lower right",
        framealpha=0.8,
        edgecolor="none",
        ncols=2,
        fontsize=8,
        handlelength=0.9,
        columnspacing=0.65,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        print("[SAVE]", savepath)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"Xg": Xg, "Yg": Yg, "phi_true_map": phi_true_map, "phi_pred_map": phi_pred_map}


def plot_phi_heatmap(
    model,
    *,
    bbox: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    n: int = 200,
    batch_size: int = 4096,
    savepath: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
):
    def sdf_rect(xx, yy, xc, yc, a, b):
        qx = np.abs(xx - xc) - a
        qy = np.abs(yy - yc) - b
        qx_pos = np.maximum(qx, 0.0)
        qy_pos = np.maximum(qy, 0.0)
        outside = np.hypot(qx_pos, qy_pos)
        inside = np.maximum(qx, qy)
        return outside + np.minimum(inside, 0.0)

    def sdf_cross(xx, yy, xc, yc, Lx, Ly, wh, wv):
        phi_h = sdf_rect(xx, yy, xc, yc, a=Lx / 2.0, b=wh / 2.0)
        phi_v = sdf_rect(xx, yy, xc, yc, a=wv / 2.0, b=Ly / 2.0)
        return np.minimum(phi_h, phi_v)

    def take_phi(out):
        if torch.is_tensor(out):
            return out
        if isinstance(out, (list, tuple)):
            for v in out:
                if torch.is_tensor(v):
                    return v
        if isinstance(out, dict):
            if "phi" in out and torch.is_tensor(out["phi"]):
                return out["phi"]
            for v in out.values():
                if torch.is_tensor(v):
                    return v
        return None

    device = torch.device("cpu")
    dtype = torch.float32
    model_cpu = copy.deepcopy(model).to(device).eval()

    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, n, dtype=np.float32)
    ys = np.linspace(ymin, ymax, n, dtype=np.float32)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    XY_grid = np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32)

    phi_flat = np.zeros(XY_grid.shape[0], dtype=np.float32)
    for i0 in range(0, XY_grid.shape[0], batch_size):
        i1 = min(XY_grid.shape[0], i0 + batch_size)
        XY_b = torch.from_numpy(XY_grid[i0:i1]).to(device=device, dtype=dtype)
        with torch.no_grad():
            out = model_cpu(XY_b)
            phi_b = take_phi(out)
        phi_flat[i0:i1] = phi_b.detach().cpu().numpy().reshape(-1).astype(np.float32)

    phi_map = phi_flat.reshape(n, n)
    xc, yc, Lx, Ly, wh, wv = config.VisualizationConfig().phi_compare_cross
    phi_true_map = sdf_cross(Xg, Yg, xc, yc, Lx, Ly, wh, wv).astype(np.float32)

    fig, ax = plt.subplots(dpi=dpi, figsize=(3.6, 3.2))
    cs = ax.contourf(Xg, Yg, phi_map, levels=60, cmap="coolwarm", alpha=0.9)
    ax.contour(Xg, Yg, phi_true_map, levels=[0.0], colors="white", linewidths=0.9, linestyles="-")
    ax.contour(Xg, Yg, phi_map, levels=[0.0], colors="black", linewidths=0.8, linestyles="--")
    fig.colorbar(cs, ax=ax, shrink=0.9)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(
        handles=[
            Line2D([0], [0], color="white", lw=0.9, ls="-", label="True $\\phi=0$"),
            Line2D([0], [0], color="black", lw=0.8, ls="--", label="Pred $\\phi=0$"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=7,
    )
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        print("[SAVE]", savepath)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"Xg": Xg, "Yg": Yg, "phi_map": phi_map}
