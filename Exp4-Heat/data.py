from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


def load_uniform_grid_fit(
    nx: int = 6,
    ny: int = 6,
    *,
    ttxt_filename: str = "Possion.txt",
    device: str = "cpu",
    circles: Optional[Iterable[Tuple[float, float, float]]] = None,
    dense_factor: float = 0.5,
    drop_boundary: bool = True,
    xlim: Tuple[float, float] = (0.0, 1.0),
    ylim: Tuple[float, float] = (0.0, 1.0),
    tol: float = 1e-12,
    target_total: Optional[int] = None,
):
    xy_full = np.loadtxt(ttxt_filename, usecols=[0, 1], comments="%")
    T_full = np.loadtxt(ttxt_filename, usecols=[2], comments="%")
    N_side = int(np.sqrt(len(xy_full)))
    X_full = xy_full[:, 0].reshape(N_side, N_side)
    Y_full = xy_full[:, 1].reshape(N_side, N_side)
    T_full = T_full.reshape(N_side, N_side)

    x_vec = np.unique(X_full)
    y_vec = np.unique(Y_full)

    all_nodes = np.stack([X_full.flatten(), Y_full.flatten()], axis=1)
    tree = cKDTree(all_nodes)

    x_left = x_vec[1] if drop_boundary else x_vec[0]
    x_right = x_vec[-2] if drop_boundary else x_vec[-1]
    y_bottom = y_vec[1] if drop_boundary else y_vec[0]
    y_top = y_vec[-2] if drop_boundary else y_vec[-1]

    x_tar = np.linspace(x_left, x_right, nx)
    y_tar = np.linspace(y_bottom, y_top, ny)
    Xg, Yg = np.meshgrid(x_tar, y_tar, indexing="ij")
    ideal_points = np.stack([Xg.flatten(), Yg.flatten()], axis=1)
    _, idxs = tree.query(ideal_points, k=1)
    idxs_basic = np.unique(idxs)
    idxs_dense = np.array([], dtype=int)

    if circles:
        for (cx, cy, r) in circles:
            dense_nx = int(nx / dense_factor)
            dense_ny = int(ny / dense_factor)
            x_dense = np.linspace(x_left, x_right, dense_nx)
            y_dense = np.linspace(y_bottom, y_top, dense_ny)
            Xd, Yd = np.meshgrid(x_dense, y_dense, indexing="ij")
            pts_dense = np.stack([Xd.flatten(), Yd.flatten()], axis=1)
            mask = ((pts_dense[:, 0] - cx) ** 2 + (pts_dense[:, 1] - cy) ** 2) <= r**2
            pts_in = pts_dense[mask]
            if len(pts_in) > 0:
                idxs_in = tree.query(pts_in, k=1)[1]
                idxs_dense = np.unique(np.concatenate([idxs_dense, idxs_in]))

    if idxs_dense.size > 0:
        idxs_dense = np.setdiff1d(idxs_dense, idxs_basic, assume_unique=False)

    def _filter_boundary(idxs: np.ndarray) -> np.ndarray:
        if idxs.size == 0:
            return idxs
        xy = all_nodes[idxs]
        mask_inner = (
            (xy[:, 0] > xlim[0] + tol)
            & (xy[:, 0] < xlim[1] - tol)
            & (xy[:, 1] > ylim[0] + tol)
            & (xy[:, 1] < ylim[1] - tol)
        )
        return idxs[mask_inner]

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
                    idxs_dense = np.random.choice(
                        idxs_dense, size=max_dense_keep, replace=False
                    )
                idxs_dense = np.array(sorted(idxs_dense), dtype=int)
                print(
                    f"[downsample] dense extra keep {idxs_dense.size}/{ne} "
                    f"(target_total={target_total})"
                )

    idxs_all = np.union1d(idxs_basic, idxs_dense)
    xy_fit_np = all_nodes[idxs_all]
    T_fit_np = T_full.flatten()[idxs_all]

    xy_fit = torch.tensor(xy_fit_np, dtype=torch.float32, device=device)
    T_fit = torch.tensor(T_fit_np[:, None], dtype=torch.float32, device=device)

    print(
        f"[load_uniform_grid_fit] {nx}x{ny} grid + dense defects -> {xy_fit.shape[0]} points"
    )
    return xy_fit, T_fit


def plot_sampling_points(
    xy_fit: torch.Tensor,
    *,
    circles: Optional[Iterable[Tuple[float, float, float]]] = None,
    title: str = "Sampling points",
    savepath: Optional[str] = None,
    show: bool = False,
):
    xy_np = xy_fit.detach().cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.scatter(xy_np[:, 0], xy_np[:, 1], s=14, color="tab:blue", alpha=0.8)

    if circles:
        theta = np.linspace(0, 2 * np.pi, 200)
        for (cx, cy, r) in circles:
            plt.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), "r--", lw=1.5)

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
    n: int, device: torch.device, corner_tol: float = 0.0, batch_size: int = 10000
) -> torch.Tensor:
    xy_list = []
    total_points = 0
    while total_points < n:
        xy_batch = torch.rand(batch_size, 2, device=device)
        mask = (
            (xy_batch[:, 0] > corner_tol)
            & (xy_batch[:, 0] < 1 - corner_tol)
            & (xy_batch[:, 1] > corner_tol)
            & (xy_batch[:, 1] < 1 - corner_tol)
        )
        xy_valid = xy_batch[mask]
        if xy_valid.numel() == 0:
            continue
        xy_list.append(xy_valid)
        total_points += int(xy_valid.shape[0])
    xy_all = torch.cat(xy_list, dim=0)[:n]
    return xy_all
