import os
import numpy as np
import torch
from scipy.spatial import cKDTree


def phi_np(xy, xc, yc, A, B, Gamma):
    x, y = xy[..., 0], xy[..., 1]
    dx, dy = x - xc, y - yc
    c, s = np.cos(Gamma), np.sin(Gamma)
    xp = c * dx + s * dy
    yp = -s * dx + c * dy
    return (xp / A) ** 2 + (yp / B) ** 2 - 1.0


def ellipse_aabb(xc, yc, A, B, Gamma, pad=0.05, n_sample=400):
    t = np.linspace(0.0, 2 * np.pi, n_sample, endpoint=False)
    c, s = np.cos(Gamma), np.sin(Gamma)
    r = np.array([[c, -s], [s, c]])
    pts_local = np.stack([A * np.cos(t), B * np.sin(t)], axis=1)
    pts_world = pts_local @ r.T + np.array([xc, yc])[None, :]
    xmin = pts_world[:, 0].min() - pad
    xmax = pts_world[:, 0].max() + pad
    ymin = pts_world[:, 1].min() - pad
    ymax = pts_world[:, 1].max() + pad
    return xmin, xmax, ymin, ymax


def load_ellipse_uv_eps_fit_downsample(
    nx=40,
    ny=40,
    *,
    txt_filename="Ellipse.txt",
    device="cpu",
    ellipse=(0.05, 0.10, 0.35, 0.15, np.deg2rad(-30.0)),
    tau_for_strain=0.02,
    use_dense=True,
    dense_factor=0.5,
    rect_params=((0.05, 0.10), 0.9, 0.5, -30.0),
    rect_corners=None,
    xlim=None,
    ylim=None,
    target_total=None,
    random_state=0,
):
    if not os.path.exists(txt_filename):
        print(f"Warning: {txt_filename} not found. Generating dummy data for demo.")
        dummy_x = np.linspace(-1, 1, 100)
        gx, gy = np.meshgrid(dummy_x, dummy_x)
        xy_full = np.stack([gx.ravel(), gy.ravel()], axis=1)
        uv_full = np.zeros_like(xy_full)
        eps_full = np.zeros((xy_full.shape[0], 3))
    else:
        with open(txt_filename, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln for ln in f if not ln.lstrip().startswith("%")]
        arr = np.loadtxt(lines)
        if arr.ndim != 2 or arr.shape[1] < 7:
            raise ValueError("Expected at least 7 columns: x y u v eXX eYY eXY")
        xy_full = arr[:, 0:2]
        uv_full = arr[:, 2:4]
        eps_full = arr[:, 4:7]

    all_nodes = xy_full
    tree = cKDTree(all_nodes)

    if xlim is not None and ylim is not None:
        xmin0, xmax0 = float(xlim[0]), float(xlim[1])
        ymin0, ymax0 = float(ylim[0]), float(ylim[1])
    else:
        xmin0, ymin0 = xy_full.min(axis=0)
        xmax0, ymax0 = xy_full.max(axis=0)
    if rect_corners is not None and xlim is None and ylim is None:
        (x0, y0), (x1, y1) = rect_corners
        xmin0, xmax0 = (x0, x1) if x0 <= x1 else (x1, x0)
        ymin0, ymax0 = (y0, y1) if y0 <= y1 else (y1, y0)

    x_tar = np.linspace(xmin0, xmax0, nx, dtype=np.float64)
    y_tar = np.linspace(ymin0, ymax0, ny, dtype=np.float64)

    xg, yg = np.meshgrid(x_tar, y_tar, indexing="ij")
    pts_main = np.stack([xg.ravel(), yg.ravel()], axis=1)

    _, idxs_main = tree.query(pts_main, k=1)
    idxs_basic_unique = np.unique(idxs_main)

    rect_info = None
    idxs_extra_only = np.array([], dtype=int)
    idxs_extra_keep = None

    if use_dense:
        if rect_corners is not None:
            (x0, y0), (x1, y1) = rect_corners
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            width = abs(x1 - x0)
            height = abs(y1 - y0)
            angle_deg = 0.0
        else:
            (cx, cy), width, height, angle_deg = rect_params
        angle_rad = np.deg2rad(angle_deg)
        rect_info = {
            "center": (cx, cy),
            "width": width,
            "height": height,
            "angle": angle_deg,
        }

        w_half, h_half = width / 2.0, height / 2.0
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        r = np.array([[c, -s], [s, c]])

        dx_global = (xmax0 - xmin0) / (nx - 1) if nx > 1 else 1.0
        target_dx = dx_global * dense_factor

        dense_nx = max(2, int(np.ceil(width / target_dx)) + 1)
        dense_ny = max(2, int(np.ceil(height / target_dx)) + 1)

        x_local = np.linspace(-w_half, w_half, dense_nx + 2)[1:-1]
        y_local = np.linspace(-h_half, h_half, dense_ny + 2)[1:-1]

        x_loc, y_loc = np.meshgrid(x_local, y_local, indexing="xy")
        pts_local = np.stack([x_loc.ravel(), y_loc.ravel()], axis=1)

        pts_dense_world = pts_local @ r.T + np.array([cx, cy])

        _, idxs_rect = tree.query(pts_dense_world, k=1)
        idxs_rect_unique = np.unique(idxs_rect)

        idxs_extra_only = np.setdiff1d(
            idxs_rect_unique,
            idxs_basic_unique,
            assume_unique=True,
        )
    elif rect_corners is not None:
        (x0, y0), (x1, y1) = rect_corners
        rect_info = {
            "center": (0.5 * (x0 + x1), 0.5 * (y0 + y1)),
            "width": abs(x1 - x0),
            "height": abs(y1 - y0),
            "angle": 0.0,
        }

    if idxs_extra_keep is None:
        idxs_extra_keep = idxs_extra_only.copy()

    if target_total is not None and use_dense:
        nb = idxs_basic_unique.size
        ne = idxs_extra_only.size
        total_now = nb + ne
        if total_now > target_total:
            max_extra_allowed = max(target_total - nb, 0)
            if max_extra_allowed <= 0:
                idxs_extra_keep = np.array([], dtype=int)
            elif ne > max_extra_allowed:
                xy_extra = all_nodes[idxs_extra_only]
                n_keep = max_extra_allowed
                n_total = idxs_extra_only.size
                if n_keep <= 0:
                    idxs_extra_keep = np.array([], dtype=int)
                elif n_keep >= n_total:
                    idxs_extra_keep = idxs_extra_only.copy()
                else:
                    local = (xy_extra - np.array([cx, cy])) @ r
                    x_vals = local[:, 0]
                    if n_keep == 1:
                        keep_idx = [int(np.argmax(np.abs(x_vals)))]
                    else:
                        order = np.argsort(x_vals)
                        pick = np.linspace(0, n_total - 1, num=n_keep).astype(int)
                        keep_idx = order[pick]
                    idxs_extra_keep = idxs_extra_only[keep_idx]

    idxs_extra_keep = np.array(sorted(np.unique(idxs_extra_keep)), dtype=int)

    idxs_all_final = np.union1d(idxs_basic_unique, idxs_extra_keep)

    xy_u_np = all_nodes[idxs_all_final]
    u_np = uv_full[idxs_all_final]
    e_np = eps_full[idxs_all_final]

    xy_basic_plot = all_nodes[idxs_basic_unique]
    xy_dense_extra_plot = all_nodes[idxs_extra_keep]

    xc, yc, A, B, Gamma = ellipse
    cG, sG = np.cos(Gamma), np.sin(Gamma)

    def phi_np_local(xy):
        xx = xy[..., 0]
        yy = xy[..., 1]
        dx = xx - xc
        dy = yy - yc
        xp = cG * dx + sG * dy
        yp = -sG * dx + cG * dy
        return (xp / A) ** 2 + (yp / B) ** 2 - 1.0

    if tau_for_strain is not None:
        phi_vals = phi_np_local(xy_u_np)
        mask_far = np.abs(phi_vals) > float(tau_for_strain)
        xy_eps_np = xy_u_np[mask_far]
        e_np_eps = e_np[mask_far]
    else:
        xy_eps_np = xy_u_np
        e_np_eps = e_np

    xy_u = torch.tensor(xy_u_np, dtype=torch.float32, device=device)
    u_fit = torch.tensor(u_np, dtype=torch.float32, device=device)
    xy_eps = torch.tensor(xy_eps_np, dtype=torch.float32, device=device)
    e_fit = torch.tensor(e_np_eps, dtype=torch.float32, device=device)

    return (
        xy_u,
        u_fit,
        xy_eps,
        e_fit,
        rect_info,
        xy_basic_plot,
        xy_dense_extra_plot,
    )


def load_ellipse_uv_eps_fit(
    nx=10,
    ny=10,
    *,
    txt_filename="Ellipse.txt",
    device="cpu",
    ellipse=(0.05, 0.10, 0.35, 0.15, np.deg2rad(-30.0)),
    tau_for_strain=0.001,
    use_dense=False,
    dense_factor=1.0,
    rect_corners=None,
    xlim=None,
    ylim=None,
):
    return load_ellipse_uv_eps_fit_downsample(
        nx=nx,
        ny=ny,
        txt_filename=txt_filename,
        device=device,
        ellipse=ellipse,
        tau_for_strain=tau_for_strain,
        use_dense=use_dense,
        dense_factor=dense_factor,
        rect_corners=rect_corners,
        xlim=xlim,
        ylim=ylim,
        target_total=None,
        random_state=0,
    )


def load_ellipse_uv_eps_fit_rect_dense(
    nx=7,
    ny=7,
    *,
    txt_filename="Ellipse.txt",
    device="cpu",
    ellipse=(0.05, 0.10, 0.35, 0.15, np.deg2rad(-30.0)),
    tau_for_strain=0.001,
    dense_factor=0.7,
    rect_corners=((-0.5, -0.2), (0.7, 0.4)),
    target_total=100,
    random_state=2,
):
    (x0, y0), (x1, y1) = rect_corners
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    rect_params = ((cx, cy), width, height, 0.0)
    return load_ellipse_uv_eps_fit_downsample(
        nx=nx,
        ny=ny,
        txt_filename=txt_filename,
        device=device,
        ellipse=ellipse,
        tau_for_strain=tau_for_strain,
        use_dense=True,
        dense_factor=dense_factor,
        rect_params=rect_params,
        rect_corners=None,
        target_total=target_total,
        random_state=random_state,
    )


def sample_xy_uniform(n, device, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)):
    x = torch.rand(n, 1, device=device) * (xlim[1] - xlim[0]) + xlim[0]
    y = torch.rand(n, 1, device=device) * (ylim[1] - ylim[0]) + ylim[0]
    return torch.cat([x, y], dim=1)
