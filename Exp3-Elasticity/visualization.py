import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LogNorm

from level_set import (
    local_velocity,
    local_velocity_fit,
    local_velocity_grad,
    local_velocity_CV_full,
)
from pde import lame_from_E, div_sigma_batch


def plot_sampling_points_ellipse_downsample(
    xy_basic_plot,
    xy_dense_extra_plot,
    ellipse,
    rect_info=None,
    title="Rotated Dense Box Sampling",
    savepath=None,
    show=True,
):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xy_basic_plot[:, 0], xy_basic_plot[:, 1], s=12, c="tab:blue", label="Base grid")
    if xy_dense_extra_plot is not None and len(xy_dense_extra_plot) > 0:
        ax.scatter(
            xy_dense_extra_plot[:, 0],
            xy_dense_extra_plot[:, 1],
            s=18,
            c="tab:orange",
            label="Dense extra",
        )

    xc, yc, A, B, Gamma = ellipse
    t = np.linspace(0.0, 2 * np.pi, 400, endpoint=False)
    c, s = np.cos(Gamma), np.sin(Gamma)
    r = np.array([[c, -s], [s, c]])
    pts_local = np.stack([A * np.cos(t), B * np.sin(t)], axis=1)
    pts_world = pts_local @ r.T + np.array([xc, yc])[None, :]
    ax.plot(pts_world[:, 0], pts_world[:, 1], color="black", linewidth=1.2, label="Ellipse")

    if rect_info is not None:
        cx, cy = rect_info["center"]
        width = rect_info["width"]
        height = rect_info["height"]
        angle_deg = rect_info["angle"]
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        r = np.array([[c, -s], [s, c]])
        w = width / 2.0
        h = height / 2.0
        corners_local = np.array([
            [-w, -h],
            [w, -h],
            [w, h],
            [-w, h],
            [-w, -h],
        ])
        corners_world = corners_local @ r.T + np.array([cx, cy])
        ax.plot(corners_world[:, 0], corners_world[:, 1], color="limegreen", linestyle="-.", linewidth=1.5, label="Dense box")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        print("[SAVE]", savepath)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _compute_phi_next_on_grid(
    model_cpu,
    xy_grid,
    n,
    *,
    vel_type="pde",
    dt=1e-2,
    band_eps_vel=0.02,
    h_vel=0.03,
    tau_vel=1e-4,
    clip_q_vel=0.99,
    xy_u=None,
    U_fit=None,
    nu=0.30,
):
    device = next(model_cpu.parameters()).device
    xy = torch.from_numpy(xy_grid).to(device=device, dtype=torch.float32)
    xy_req = xy.clone().requires_grad_(True)

    vel_type = vel_type.lower()
    Fai_empty = False
    if vel_type == "pde":
        Vn = local_velocity(model_cpu, xy_req, band_eps=band_eps_vel, h=h_vel, tau=tau_vel, clip_q=clip_q_vel, nu=nu)
    elif vel_type == "data":
        if xy_u is None or U_fit is None:
            raise ValueError("vel_type='data' requires xy_u and U_fit")
        xy_u_cpu = xy_u.detach().cpu()
        U_fit_cpu = U_fit.detach().cpu()
        Vn = local_velocity_fit(model_cpu, xy_req, xy_u_cpu, U_fit_cpu, band_eps=band_eps_vel, h=h_vel, tau=tau_vel)
    elif vel_type == "grad":
        Vn = local_velocity_grad(model_cpu, xy_req, band_eps_vel, h_vel, tau_vel, nu=nu)
    elif vel_type == "cv":
        Vn, Fai_empty = local_velocity_CV_full(model_cpu, xy_req, band_eps=band_eps_vel, h=h_vel, tau=tau_vel, nu=nu)
    else:
        raise ValueError(f"Unknown vel_type '{vel_type}'")

    Vn = Vn.detach()

    phi_now = model_cpu.phi(xy_req)
    if not Fai_empty:
        grad_phi = torch.autograd.grad(phi_now, xy_req, torch.ones_like(phi_now), create_graph=True)[0]
        norm_g = grad_phi.norm(dim=1, keepdim=True).clamp_min(1e-6)
        phi_next = phi_now + dt * Vn * norm_g
    else:
        phi_next = phi_now + dt * Vn

    phi_now_map = phi_now.detach().cpu().numpy().reshape(n, n)
    phi_next_map = phi_next.detach().cpu().numpy().reshape(n, n)

    return phi_now_map, phi_next_map


def plot_residual_scatter(
    model,
    kind="pde",
    nu=0.30,
    xy_u=None,
    U_fit=None,
    n=200,
    bbox=(-1, 1, -1, 1),
    batch_size=4096,
    ellipse=(0.05, 0.10, 0.35, 0.15, np.deg2rad(-30)),
    cmap="cividis",
    use_log_norm=False,
    savepath=None,
    show_next=True,
    vel_type_for_next="pde",
    dt_next=1e-2,
    band_eps_vel=0.02,
    h_vel=0.03,
    tau_vel=1e-4,
    clip_q_vel=0.99,
    save_npz_path=None,
    show=True,
):
    device = torch.device("cpu")
    dtype = torch.float32
    model_cpu = copy.deepcopy(model).to(device).eval()

    if hasattr(model_cpu, "get_E_scaled"):
        E_1_scaled, E_2_scaled = model_cpu.get_E_scaled()
    else:
        E_1_scaled, E_2_scaled = model_cpu.E_1, model_cpu.E_2
    lam_1, mu_1 = lame_from_E(E_1_scaled, nu)
    lam_2, mu_2 = lame_from_E(E_2_scaled, nu)

    xc, yc, A, B, Gamma = ellipse
    cG = np.cos(Gamma)
    sG = np.sin(Gamma)

    def phi_true_eval(xx, yy):
        dx = xx - xc
        dy = yy - yc
        xp = cG * dx + sG * dy
        yp = -sG * dx + cG * dy
        return (xp / A) ** 2 + (yp / B) ** 2 - 1.0

    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, n, dtype=np.float32)
    ys = np.linspace(ymin, ymax, n, dtype=np.float32)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    XY_grid = np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32)

    phi_model_flat = np.zeros(XY_grid.shape[0], dtype=np.float32)
    for i0 in range(0, XY_grid.shape[0], batch_size):
        i1 = min(XY_grid.shape[0], i0 + batch_size)
        XY_b = torch.from_numpy(XY_grid[i0:i1]).to(device=device, dtype=dtype)
        with torch.no_grad():
            phi_b, _, _, _, _ = model_cpu(XY_b)
        phi_model_flat[i0:i1] = phi_b.detach().cpu().numpy().reshape(-1).astype(np.float32)

    phi_model_map = phi_model_flat.reshape(n, n)
    phi_true_map = phi_true_eval(XY_grid[:, 0], XY_grid[:, 1]).astype(np.float32).reshape(n, n)

    if show_next:
        _, phi_next_map = _compute_phi_next_on_grid(
            model_cpu,
            XY_grid,
            n,
            vel_type=vel_type_for_next,
            dt=dt_next,
            band_eps_vel=band_eps_vel,
            h_vel=h_vel,
            tau_vel=tau_vel,
            clip_q_vel=clip_q_vel,
            xy_u=xy_u,
            U_fit=U_fit,
            nu=nu,
        )
    else:
        phi_next_map = None

    if kind == "data":
        if xy_u is None or U_fit is None:
            raise ValueError("kind='data' requires xy_u and U_fit")

        XY_lab = xy_u.detach().cpu().to(dtype).numpy()
        U_lab = U_fit.detach().cpu().to(dtype).numpy()
        XY_lab_t = torch.from_numpy(XY_lab).to(device=device, dtype=dtype)

        with torch.no_grad():
            phi_b, ux1_b, uy1_b, ux2_b, uy2_b = model_cpu(XY_lab_t)
            w1 = torch.relu(phi_b)
            w2 = torch.relu(-phi_b)
            denom = (w1 + w2 + 1e-12)
            ux_pred = (w1 * ux1_b + w2 * ux2_b) / denom
            uy_pred = (w1 * uy1_b + w2 * uy2_b) / denom
            U_pred = torch.cat([ux_pred, uy_pred], dim=1)

        diff = U_pred.detach().cpu().numpy() - U_lab
        RES_val = np.sqrt((diff ** 2).sum(axis=1)).astype(np.float32)
        XY_residual = XY_lab

    elif kind in ("pde", "grad"):
        R_soft_flat = np.zeros(XY_grid.shape[0], dtype=np.float32)

        for i0 in range(0, XY_grid.shape[0], batch_size):
            i1 = min(XY_grid.shape[0], i0 + batch_size)
            XY_b = torch.from_numpy(XY_grid[i0:i1]).to(device=device, dtype=dtype)
            XY_b_req = XY_b.clone().requires_grad_(True)

            phi_b, ux1_b, uy1_b, ux2_b, uy2_b = model_cpu(XY_b_req)
            Rx1, Ry1 = div_sigma_batch(XY_b_req, ux1_b, uy1_b, lam_1, mu_1)
            Rx2, Ry2 = div_sigma_batch(XY_b_req, ux2_b, uy2_b, lam_2, mu_2)

            Rmag1 = torch.sqrt(Rx1 ** 2 + Ry1 ** 2)
            Rmag2 = torch.sqrt(Rx2 ** 2 + Ry2 ** 2)

            w1 = torch.relu(phi_b)
            w2 = torch.relu(-phi_b)
            denom = (w1 + w2 + 1e-12)
            R_soft = (w1 * Rmag1 + w2 * Rmag2) / denom

            R_soft_flat[i0:i1] = R_soft.detach().cpu().numpy().reshape(-1).astype(np.float32)

        if kind == "pde":
            RES_val = R_soft_flat
            XY_residual = XY_grid
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
            XY_residual = XY_grid
    else:
        raise ValueError(f"unknown kind '{kind}'")

    RES_plot = RES_val.copy()
    vmax_lin = np.percentile(RES_plot, 99.0)
    if vmax_lin <= 0:
        vmax_lin = RES_plot.max() if RES_plot.max() > 0 else 1.0
    RES_clip = np.clip(RES_plot, 0.0, vmax_lin)

    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    fig, ax = plt.subplots(dpi=300, figsize=(3.8, 2.5))

    norm = LogNorm(vmin=max(1e-12, RES_clip.min()), vmax=vmax_lin) if use_log_norm else None

    sc = ax.scatter(
        XY_residual[:, 0],
        XY_residual[:, 1],
        c=RES_clip,
        s=(40 if kind == "data" else 8),
        marker=("o" if kind == "data" else "s"),
        linewidths=0.0,
        cmap=cmap,
        norm=norm,
        vmin=0.0 if not use_log_norm else None,
        vmax=vmax_lin if not use_log_norm else None,
    )

    ax.contour(Xg, Yg, phi_true_map, levels=[0.0], colors="black", linewidths=1.0, linestyles="-")
    ax.contour(Xg, Yg, phi_model_map, levels=[0.0], colors="red", linewidths=1.0, linestyles="--")
    if show_next and (phi_next_map is not None):
        ax.contour(Xg, Yg, phi_next_map, levels=[0.0], colors="lime", linewidths=0.7, linestyles="dashdot")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.0, ls="-", label="Exact"),
        Line2D([0], [0], color="red", lw=1.0, ls="--", label="Current"),
    ]
    if show_next and (phi_next_map is not None):
        legend_handles.append(Line2D([0], [0], color="lime", lw=1.0, ls="-.", label="Next"))

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
    if not use_log_norm:
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

    return {
        "XY_residual": XY_residual,
        "RES_val": RES_val,
        "phi_model": phi_model_map,
        "phi_next": phi_next_map,
        "phi_true": phi_true_map,
    }


def plot_phi_compare(model, ellipse, n=200, bbox=(-1.0, 1.0, -1.0, 1.0), savepath=None, show=True):
    device = next(model.parameters()).device
    xmin, xmax, ymin, ymax = bbox
    xg = torch.linspace(xmin, xmax, n, device=device)
    yg = torch.linspace(ymin, ymax, n, device=device)
    Xg, Yg = torch.meshgrid(xg, yg, indexing="ij")
    xy_grid = torch.stack([Xg, Yg], dim=-1).reshape(-1, 2)

    with torch.no_grad():
        phi_pred = model.phi(xy_grid).reshape(n, n).cpu().numpy()

    xc, yc, A, B, Gamma = ellipse
    cG = np.cos(Gamma)
    sG = np.sin(Gamma)
    Xn = Xg.cpu().numpy()
    Yn = Yg.cpu().numpy()
    dx = Xn - xc
    dy = Yn - yc
    xp = cG * dx + sG * dy
    yp = -sG * dx + cG * dy
    phi_true = (xp / A) ** 2 + (yp / B) ** 2 - 1.0

    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    ax.contour(Xn, Yn, phi_true, levels=[0.0], colors="black", linewidths=1.0)
    ax.contour(Xn, Yn, phi_pred, levels=[0.0], colors="red", linewidths=1.0, linestyles="--")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Phi: true vs predicted")
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", lw=1.0, label="Exact"),
            Line2D([0], [0], color="red", lw=1.0, ls="--", label="Predicted"),
        ],
        loc="lower left",
        fontsize=7,
    )
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        print("[SAVE]", savepath)
    if show:
        plt.show()
    else:
        plt.close(fig)
