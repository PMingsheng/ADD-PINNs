from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

from problem import flower_interface_curve, generate_full_field


def _load_full_field(
    *,
    use_synthetic: bool,
    synthetic_n_side: int,
    ttxt_filename: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if use_synthetic:
        return generate_full_field(n_side=synthetic_n_side)

    xy_full = np.loadtxt(ttxt_filename, usecols=[0, 1], comments="%")
    u_full = np.loadtxt(ttxt_filename, usecols=[2], comments="%").reshape(-1, 1)
    return xy_full, u_full


def load_uniform_grid_fit(
    nx: int = 6,
    ny: int = 6,
    *,
    use_synthetic: bool = True,
    synthetic_n_side: int = 201,
    ttxt_filename: str = "Possion.txt",
    device: str = "cpu",
    circles: Optional[Iterable[Tuple[float, float, float]]] = None,
    annuli: Optional[Iterable[Tuple[float, float, float, float]]] = None,
    dense_factor: float = 0.5,
    drop_boundary: bool = True,
    xlim: Tuple[float, float] = (-1.0, 1.0),
    ylim: Tuple[float, float] = (-1.0, 1.0),
    tol: float = 1e-12,
    target_total: Optional[int] = None,
):
    xy_full, u_full = _load_full_field(
        use_synthetic=use_synthetic,
        synthetic_n_side=synthetic_n_side,
        ttxt_filename=ttxt_filename,
    )

    x_vec = np.unique(xy_full[:, 0])
    y_vec = np.unique(xy_full[:, 1])

    all_nodes = xy_full
    tree = cKDTree(all_nodes)

    x_tar = np.linspace(x_vec[1], x_vec[-2], nx)
    y_tar = np.linspace(y_vec[1], y_vec[-2], ny)
    Xg, Yg = np.meshgrid(x_tar, y_tar, indexing="ij")
    ideal_points = np.stack([Xg.flatten(), Yg.flatten()], axis=1)
    _, idxs = tree.query(ideal_points, k=1)
    idxs_basic = np.unique(idxs)
    idxs_dense = np.array([], dtype=int)

    if circles or annuli:
        dense_nx = max(2, int(nx / dense_factor))
        dense_ny = max(2, int(ny / dense_factor))
        x_dense = np.linspace(x_vec[1], x_vec[-2], dense_nx)
        y_dense = np.linspace(y_vec[1], y_vec[-2], dense_ny)
        Xd, Yd = np.meshgrid(x_dense, y_dense, indexing="ij")
        pts_dense = np.stack([Xd.flatten(), Yd.flatten()], axis=1)

        if circles:
            for (cx, cy, r) in circles:
                mask = ((pts_dense[:, 0] - cx) ** 2 + (pts_dense[:, 1] - cy) ** 2) <= r**2
                pts_in = pts_dense[mask]
                if len(pts_in) > 0:
                    idxs_in = tree.query(pts_in, k=1)[1]
                    idxs_dense = np.unique(np.concatenate([idxs_dense, idxs_in]))

        if annuli:
            for (cx, cy, r_in, r_out) in annuli:
                rin = min(r_in, r_out)
                rout = max(r_in, r_out)
                dist2 = (pts_dense[:, 0] - cx) ** 2 + (pts_dense[:, 1] - cy) ** 2
                mask = (dist2 >= rin**2) & (dist2 <= rout**2)
                pts_in = pts_dense[mask]
                if len(pts_in) > 0:
                    idxs_in = tree.query(pts_in, k=1)[1]
                    idxs_dense = np.unique(np.concatenate([idxs_dense, idxs_in]))

    if idxs_dense.size > 0:
        idxs_dense = np.setdiff1d(idxs_dense, idxs_basic, assume_unique=False)

    def _filter_boundary(idxs_local: np.ndarray) -> np.ndarray:
        if idxs_local.size == 0:
            return idxs_local
        xy = all_nodes[idxs_local]
        mask_inner = (
            (xy[:, 0] > xlim[0] + tol)
            & (xy[:, 0] < xlim[1] - tol)
            & (xy[:, 1] > ylim[0] + tol)
            & (xy[:, 1] < ylim[1] - tol)
        )
        return idxs_local[mask_inner]

    if drop_boundary:
        before = idxs_basic.size + idxs_dense.size
        idxs_basic = _filter_boundary(idxs_basic)
        idxs_dense = _filter_boundary(idxs_dense)
        after = idxs_basic.size + idxs_dense.size
        print(f"[filter] removed boundary points: {before - after} (kept {after})")

    if target_total is not None:
        nb = idxs_basic.size
        ne = idxs_dense.size
        total_now = nb + ne
        if total_now > target_total:
            if nb >= target_total:
                if nb > target_total:
                    idxs_basic = np.random.choice(idxs_basic, size=target_total, replace=False)
                    idxs_basic = np.array(sorted(idxs_basic), dtype=int)
                idxs_dense = np.array([], dtype=int)
                print(
                    f"[downsample] base points exceed target_total; "
                    f"kept {idxs_basic.size}, dropped all dense"
                )
            else:
                max_dense_keep = max(target_total - nb, 0)
                if max_dense_keep <= 0:
                    idxs_dense = np.array([], dtype=int)
                elif ne > max_dense_keep:
                    idxs_dense = np.random.choice(idxs_dense, size=max_dense_keep, replace=False)
                idxs_dense = np.array(sorted(idxs_dense), dtype=int)
                print(
                    f"[downsample] dense extra keep {idxs_dense.size}/{ne} "
                    f"(target_total={target_total})"
                )

    idxs_all = np.union1d(idxs_basic, idxs_dense)
    xy_fit_np = all_nodes[idxs_all]
    u_fit_np = u_full[idxs_all].reshape(-1)

    xy_fit = torch.tensor(xy_fit_np, dtype=torch.float32, device=device)
    u_fit = torch.tensor(u_fit_np[:, None], dtype=torch.float32, device=device)

    print(
        f"[load_uniform_grid_fit] {nx}x{ny} grid + dense ROI -> {xy_fit.shape[0]} points"
    )
    return xy_fit, u_fit


def plot_sampling_points(
    xy_fit: torch.Tensor,
    *,
    circles: Optional[Iterable[Tuple[float, float, float]]] = None,
    annuli: Optional[Iterable[Tuple[float, float, float, float]]] = None,
    show_circles: bool = False,
    show_annuli: bool = False,
    show_flower_interface: bool = True,
    title: str = "Sampling points",
    savepath: Optional[str] = None,
    show: bool = False,
):
    xy_np = xy_fit.detach().cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.scatter(xy_np[:, 0], xy_np[:, 1], s=14, color="tab:blue", alpha=0.8)

    if show_flower_interface:
        xf, yf = flower_interface_curve(n_theta=1000)
        plt.plot(xf, yf, color="black", lw=1.6, ls="-", label="Interface")

    if show_circles and circles:
        theta = np.linspace(0, 2 * np.pi, 400)
        for (cx, cy, r) in circles:
            plt.plot(
                cx + r * np.cos(theta),
                cy + r * np.sin(theta),
                color="tab:red",
                ls="--",
                lw=1.2,
                alpha=0.8,
            )

    if show_annuli and annuli:
        theta = np.linspace(0, 2 * np.pi, 400)
        for (cx, cy, r_in, r_out) in annuli:
            rin = min(r_in, r_out)
            rout = max(r_in, r_out)
            plt.plot(
                cx + rin * np.cos(theta),
                cy + rin * np.sin(theta),
                color="tab:green",
                ls="--",
                lw=1.2,
                alpha=0.9,
            )
            plt.plot(
                cx + rout * np.cos(theta),
                cy + rout * np.sin(theta),
                color="tab:green",
                ls="--",
                lw=1.2,
                alpha=0.9,
            )

    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def sample_xy_no_corners(
    n: int,
    device: torch.device,
    *,
    corner_tol: float = 0.0,
    batch_size: int = 10000,
    xlim: Tuple[float, float] = (-1.0, 1.0),
    ylim: Tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    x0, x1 = xlim
    y0, y1 = ylim
    mx = corner_tol * (x1 - x0)
    my = corner_tol * (y1 - y0)

    xy_list = []
    total = 0
    while total < n:
        x = torch.rand(batch_size, 1, device=device) * (x1 - x0) + x0
        y = torch.rand(batch_size, 1, device=device) * (y1 - y0) + y0
        xy_batch = torch.cat([x, y], dim=1)
        mask = (
            (xy_batch[:, 0] > x0 + mx)
            & (xy_batch[:, 0] < x1 - mx)
            & (xy_batch[:, 1] > y0 + my)
            & (xy_batch[:, 1] < y1 - my)
        )
        xy_valid = xy_batch[mask]
        xy_list.append(xy_valid)
        total += xy_valid.shape[0]

    xy_all = torch.cat(xy_list, dim=0)[:n]
    return xy_all


def sample_boundary_points(
    n_per_edge: int,
    device: torch.device,
    *,
    xlim: Tuple[float, float] = (-1.0, 1.0),
    ylim: Tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    x0, x1 = xlim
    y0, y1 = ylim

    xs = torch.linspace(x0, x1, n_per_edge, device=device)
    ys = torch.linspace(y0, y1, n_per_edge, device=device)

    bottom = torch.stack([xs, torch.full_like(xs, y0)], dim=1)
    top = torch.stack([xs, torch.full_like(xs, y1)], dim=1)
    left = torch.stack([torch.full_like(ys, x0), ys], dim=1)
    right = torch.stack([torch.full_like(ys, x1), ys], dim=1)

    xy = torch.cat([bottom, top, left, right], dim=0)
    xy = torch.unique(xy, dim=0)
    return xy
