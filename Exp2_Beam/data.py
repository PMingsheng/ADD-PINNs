"""
Data loading and sampling functions for LS-PINN Beam project.
"""
import torch
import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional, List


def load_uniform_grid_fit(
    nx: int = 9,
    *,
    filename: str = "Beam.txt",
    device: str = "cpu",
    ranges: Optional[List[Tuple[float, float]]] = None,
    dense_factor: float = 0.5,
    label_column: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load training data with uniform grid and optional local refinement.
    
    Args:
        nx: Number of global sampling points
        filename: Path to data file
        device: Device to place tensors on
        ranges: List of (x_start, x_end) tuples for local refinement
        dense_factor: Refinement factor for local regions
        label_column: Column index for labels (2=displacement, 6=strain)
        
    Returns:
        xy_fit: Coordinates tensor [N, 1]
        label_fit: Label values tensor [N, 1]
    """
    # Load data: X, Y, displacement, rotation, moment, shear, strain
    data = np.loadtxt(filename, usecols=[0, 1, 2, 3, 4, 5, 6], comments='%')
    X_full = data[:, 0]
    disp_full = data[:, 2]
    strain_full = data[:, 6]

    # Select label based on column
    if label_column == 2:
        label_data = disp_full
    elif label_column == 6:
        label_data = strain_full
    else:
        label_data = disp_full

    # Create uniform grid
    x_vals = np.unique(X_full)
    all_nodes = np.stack([X_full.flatten()], axis=1)
    tree = cKDTree(all_nodes)

    # Generate global uniform grid
    x_tar = np.linspace(x_vals[1], x_vals[-2], nx)
    ideal_points = x_tar[:, None]
    _, idxs = tree.query(ideal_points, k=1)
    idxs_all = set(idxs.tolist())

    # Local refinement sampling
    if ranges:
        for (x_start, x_end) in ranges:
            dense_n = int(nx / dense_factor)
            x_dense = np.linspace(x_start, x_end, dense_n)
            pts_dense = x_dense[:, None]
            idxs_in = tree.query(pts_dense, k=1)[1]
            idxs_all.update(idxs_in.tolist())

    # Collect sampled points
    idxs_all = np.array(sorted(list(idxs_all)), dtype=int)
    xy_fit_np = all_nodes[idxs_all]
    label_fit_np = label_data.flatten()[idxs_all]

    # Convert to torch tensors
    xy_fit = torch.tensor(xy_fit_np, dtype=torch.float32, device=device)
    label_fit = torch.tensor(label_fit_np[:, None], dtype=torch.float32, device=device)

    print(f"[load_uniform_grid_fit] global {nx} grid + dense defects -> {xy_fit.shape[0]} points")
    return xy_fit.to(device), label_fit.to(device)


def sample_xy_no_corners(
    n: int,
    device: str,
    corner_tol: float = 0.001,
    batch_size: int = 1000
) -> torch.Tensor:
    """
    Sample random points excluding corner regions.
    
    Args:
        n: Number of points to sample
        device: Device to place tensor on
        corner_tol: Distance from corners to exclude
        batch_size: Batch size for sampling
        
    Returns:
        Tensor of sampled points [n, 1]
    """
    x_list = []
    while len(x_list) < n:
        x_batch = torch.rand(batch_size, 1, device=device)
        mask = (x_batch[:, 0] > corner_tol) & (x_batch[:, 0] < 1 - corner_tol)
        x_valid = x_batch[mask]
        x_list.append(x_valid)
    x_all = torch.cat(x_list, dim=0)[:n]
    return x_all
