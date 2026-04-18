from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

import config


EllipseParams = Tuple[float, float, float, float, float]


def _resolve_ellipse(ellipse: Optional[EllipseParams] = None) -> EllipseParams:
    if ellipse is None:
        return tuple(float(v) for v in config.ELLIPSE_PARAMS)
    return tuple(float(v) for v in ellipse)


def phi_signed_ellipse(xy: torch.Tensor, ellipse: Optional[EllipseParams] = None) -> torch.Tensor:
    xc, yc, a, b, gamma = _resolve_ellipse(ellipse)
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    dx = x - xc
    dy = y - yc
    cg = np.cos(gamma)
    sg = np.sin(gamma)

    xp = cg * dx + sg * dy
    yp = -sg * dx + cg * dy
    return (xp / a) ** 2 + (yp / b) ** 2 - 1.0


def boundary_displacement(
    xy: torch.Tensor,
    *,
    eps0: Optional[float] = None,
    nu: Optional[float] = None,
) -> torch.Tensor:
    eps0 = float(config.EPS0 if eps0 is None else eps0)
    nu = float(config.NU if nu is None else nu)

    x = xy[:, 0:1]
    y = xy[:, 1:2]
    ux = eps0 * x
    uy = -nu * eps0 * y
    return torch.cat([ux, uy], dim=1)


def piecewise_modulus(
    xy: torch.Tensor,
    E_out: torch.Tensor | float,
    E_in: torch.Tensor | float,
    ellipse: Optional[EllipseParams] = None,
) -> torch.Tensor:
    phi = phi_signed_ellipse(xy, ellipse=ellipse)
    E_out_t = torch.as_tensor(E_out, dtype=xy.dtype, device=xy.device)
    E_in_t = torch.as_tensor(E_in, dtype=xy.dtype, device=xy.device)
    return torch.where(phi >= 0.0, E_out_t, E_in_t)


def sample_interface_points(
    n: int,
    device: torch.device,
    *,
    ellipse: Optional[EllipseParams] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xc, yc, a, b, gamma = _resolve_ellipse(ellipse)
    theta = torch.rand(n, 1, device=device, dtype=dtype) * (2.0 * np.pi)

    ct = torch.cos(theta)
    st = torch.sin(theta)
    cg = torch.as_tensor(np.cos(gamma), device=device, dtype=dtype)
    sg = torch.as_tensor(np.sin(gamma), device=device, dtype=dtype)

    xp = a * ct
    yp = b * st

    x = xc + cg * xp - sg * yp
    y = yc + sg * xp + cg * yp
    xy = torch.cat([x, y], dim=1)

    grad_x = 2.0 * (cg * xp / (a * a) - sg * yp / (b * b))
    grad_y = 2.0 * (sg * xp / (a * a) + cg * yp / (b * b))
    normals = torch.cat([grad_x, grad_y], dim=1)
    normals = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return xy, normals


def load_full_reference(
    txt_filename: str | Path,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path = Path(txt_filename)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("%")]
    arr = np.loadtxt(lines)
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise ValueError("Expected Ellipse.txt with at least 7 columns: x y ux uy exx eyy exy")

    xy = torch.tensor(arr[:, 0:2], dtype=torch.float32, device=device)
    u = torch.tensor(arr[:, 2:4], dtype=torch.float32, device=device)
    eps = torch.tensor(arr[:, 4:7], dtype=torch.float32, device=device)
    return xy, u, eps
