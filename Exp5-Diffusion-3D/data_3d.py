from __future__ import annotations

from typing import Tuple

import torch

from problem_3d import exact_solution


def sample_uniform_fit_3d(
    nx: int,
    ny: int,
    nz: int,
    *,
    device: torch.device,
    drop_boundary: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("nx, ny, nz must be >= 2")

    if drop_boundary:
        xs = torch.linspace(0.0, 1.0, nx + 2, device=device)[1:-1]
        ys = torch.linspace(0.0, 1.0, ny + 2, device=device)[1:-1]
        zs = torch.linspace(0.0, 1.0, nz + 2, device=device)[1:-1]
    else:
        xs = torch.linspace(0.0, 1.0, nx, device=device)
        ys = torch.linspace(0.0, 1.0, ny, device=device)
        zs = torch.linspace(0.0, 1.0, nz, device=device)

    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    xyz = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
    with torch.no_grad():
        u = exact_solution(xyz)
    return xyz, u


def sample_xyz_interior(
    n: int,
    *,
    device: torch.device,
    margin: float = 0.0,
    batch_size: int = 10000,
) -> torch.Tensor:
    if n <= 0:
        raise ValueError("n must be positive")
    if margin < 0.0 or margin >= 0.5:
        raise ValueError("margin must satisfy 0 <= margin < 0.5")

    lo = margin
    hi = 1.0 - margin

    pts = []
    total = 0
    while total < n:
        xyz = torch.rand(batch_size, 3, device=device) * (hi - lo) + lo
        pts.append(xyz)
        total += xyz.shape[0]

    return torch.cat(pts, dim=0)[:n]
