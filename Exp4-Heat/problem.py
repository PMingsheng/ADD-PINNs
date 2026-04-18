from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

import config


CircleParams = Tuple[float, float, float]


def _resolve_circle(circle: Optional[CircleParams] = None) -> CircleParams:
    if circle is not None:
        return tuple(float(v) for v in circle)
    if not config.DataConfig().circles:
        raise ValueError("At least one circle is required in DataConfig.circles.")
    return tuple(float(v) for v in config.DataConfig().circles[0])


def phi_signed_circle(xy: torch.Tensor, circle: Optional[CircleParams] = None) -> torch.Tensor:
    cx, cy, r = _resolve_circle(circle)
    dx = xy[:, 0:1] - cx
    dy = xy[:, 1:2] - cy
    return torch.sqrt(dx.square() + dy.square() + 1e-12) - r


def piecewise_source(
    xy: torch.Tensor,
    f1: torch.Tensor | float,
    f2: torch.Tensor | float,
    circle: Optional[CircleParams] = None,
) -> torch.Tensor:
    phi = phi_signed_circle(xy, circle=circle)
    f1_t = torch.as_tensor(f1, dtype=xy.dtype, device=xy.device)
    f2_t = torch.as_tensor(f2, dtype=xy.dtype, device=xy.device)
    return torch.where(phi >= 0.0, f1_t, f2_t)


def boundary_temperature(xy: torch.Tensor) -> torch.Tensor:
    return torch.zeros((xy.shape[0], 1), dtype=xy.dtype, device=xy.device)


def sample_interface_points(
    n: int,
    device: torch.device,
    *,
    circle: Optional[CircleParams] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    cx, cy, r = _resolve_circle(circle)
    theta = torch.rand(n, 1, device=device, dtype=dtype) * (2.0 * np.pi)
    x = cx + r * torch.cos(theta)
    y = cy + r * torch.sin(theta)
    return torch.cat([x, y], dim=1)


def load_full_reference(
    txt_filename: str | Path,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    path = Path(txt_filename)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("%")]
    arr = np.loadtxt(lines)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Expected Possion.txt with at least 3 columns: x y T")

    xy = torch.tensor(arr[:, 0:2], dtype=torch.float32, device=device)
    t = torch.tensor(arr[:, 2:3], dtype=torch.float32, device=device)
    return xy, t
