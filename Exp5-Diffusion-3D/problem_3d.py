from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


# Domain: Omega = [0, 1]^3
# Interface: single sphere
#   Gamma = dB(c1, r)
#   c1 = (0.4, 0.5, 0.5), r = 0.1
#
# Region convention in this file:
#   phi >= 0 -> Omega_1 (inside sphere centered at c1)
#   phi <  0 -> Omega_2 (outside)

C1 = (0.4, 0.5, 0.5)
RADIUS = 0.1

BETA_INSIDE = 5.0

# Reference case requested by user (alpha1 = alpha2 = 0).
ALPHA_INSIDE = 0.0
ALPHA_OUTSIDE = 0.0

# Interface jump constants.
INTERFACE_BETA1 = 0.0
INTERFACE_BETA2 = 0.0


def _split_xyz(xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = xyz[:, 0:1]
    y = xyz[:, 1:2]
    z = xyz[:, 2:3]
    return x, y, z


def _phi_sphere(xyz: torch.Tensor, center: Tuple[float, float, float], radius: float) -> torch.Tensor:
    x, y, z = _split_xyz(xyz)
    cx, cy, cz = center
    d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return radius * radius - d2


def phi_signed_c1_sphere(xyz: torch.Tensor) -> torch.Tensor:
    return _phi_sphere(xyz, C1, RADIUS)


def phi_signed_two_spheres(xyz: torch.Tensor) -> torch.Tensor:
    # Backward-compatibility alias.
    return phi_signed_c1_sphere(xyz)


def inside_mask(xyz: torch.Tensor) -> torch.Tensor:
    return phi_signed_c1_sphere(xyz) >= 0.0


def beta_outside(xyz: torch.Tensor) -> torch.Tensor:
    x, y, _ = _split_xyz(xyz)
    return 2.0 + torch.cos(x + y)


def grad_beta_outside(xyz: torch.Tensor) -> torch.Tensor:
    x, y, _ = _split_xyz(xyz)
    s = -torch.sin(x + y)
    gz = torch.zeros_like(s)
    return torch.cat([s, s, gz], dim=1)


def beta_piecewise(xyz: torch.Tensor) -> torch.Tensor:
    mask = inside_mask(xyz)
    b_in = torch.full_like(mask, fill_value=BETA_INSIDE, dtype=xyz.dtype)
    b_out = beta_outside(xyz)
    return torch.where(mask, b_in, b_out)


def grad_beta_piecewise(xyz: torch.Tensor) -> torch.Tensor:
    mask = inside_mask(xyz).expand(-1, 3)
    g_out = grad_beta_outside(xyz)
    g_in = torch.zeros_like(g_out)
    return torch.where(mask, g_in, g_out)


def alpha_piecewise(xyz: torch.Tensor) -> torch.Tensor:
    mask = inside_mask(xyz)
    a_in = torch.full_like(mask, fill_value=ALPHA_INSIDE, dtype=xyz.dtype)
    a_out = torch.full_like(mask, fill_value=ALPHA_OUTSIDE, dtype=xyz.dtype)
    return torch.where(mask, a_in, a_out)


def u_inside(xyz: torch.Tensor) -> torch.Tensor:
    x, y, z = _split_xyz(xyz)
    return torch.exp(x * y * z)


def grad_u_inside(xyz: torch.Tensor) -> torch.Tensor:
    x, y, z = _split_xyz(xyz)
    e = torch.exp(x * y * z)
    return torch.cat([y * z * e, x * z * e, x * y * e], dim=1)


def lap_u_inside(xyz: torch.Tensor) -> torch.Tensor:
    x, y, z = _split_xyz(xyz)
    e = torch.exp(x * y * z)
    return e * (y * y * z * z + x * x * z * z + x * x * y * y)


def u_outside(xyz: torch.Tensor) -> torch.Tensor:
    x, y, z = _split_xyz(xyz)
    return torch.sin(x + y + z)


def grad_u_outside(xyz: torch.Tensor) -> torch.Tensor:
    x, y, z = _split_xyz(xyz)
    c = torch.cos(x + y + z)
    return torch.cat([c, c, c], dim=1)


def lap_u_outside(xyz: torch.Tensor) -> torch.Tensor:
    x, y, z = _split_xyz(xyz)
    s = torch.sin(x + y + z)
    return -3.0 * s


def exact_solution(xyz: torch.Tensor) -> torch.Tensor:
    mask = inside_mask(xyz)
    return torch.where(mask, u_inside(xyz), u_outside(xyz))


def source_region_inside(xyz: torch.Tensor) -> torch.Tensor:
    # f = -div(beta grad u) + alpha u
    #   = -BETA_INSIDE * Delta(u_inside) + ALPHA_INSIDE * u_inside
    u = u_inside(xyz)
    lap_u = lap_u_inside(xyz)
    return -BETA_INSIDE * lap_u + ALPHA_INSIDE * u


def source_region_outside(xyz: torch.Tensor) -> torch.Tensor:
    # beta = 2 + cos(x+y), grad(beta)=(-sin(x+y), -sin(x+y), 0)
    # u = sin(x+y+z)
    # div(beta grad u) = beta*Delta(u) + grad(beta)·grad(u)
    # f = -div(beta grad u) + alpha u
    x, y, z = _split_xyz(xyz)
    beta = 2.0 + torch.cos(x + y)
    sin_sum = torch.sin(x + y + z)
    cos_sum = torch.cos(x + y + z)
    sin_xy = torch.sin(x + y)

    div_beta_grad_u = -3.0 * beta * sin_sum - 2.0 * sin_xy * cos_sum
    u = sin_sum
    return -div_beta_grad_u + ALPHA_OUTSIDE * u


def source_term_piecewise(xyz: torch.Tensor) -> torch.Tensor:
    mask = inside_mask(xyz)
    f_in = source_region_inside(xyz)
    f_out = source_region_outside(xyz)
    return torch.where(mask, f_in, f_out)


def boundary_g(xyz: torch.Tensor) -> torch.Tensor:
    # On [0,1]^3 boundary, points are outside the small sphere.
    return u_outside(xyz)


def interface_beta1(xyz: torch.Tensor) -> torch.Tensor:
    return torch.full((xyz.shape[0], 1), INTERFACE_BETA1, dtype=xyz.dtype, device=xyz.device)


def interface_beta2(xyz: torch.Tensor) -> torch.Tensor:
    return torch.full((xyz.shape[0], 1), INTERFACE_BETA2, dtype=xyz.dtype, device=xyz.device)


def generate_full_field(n_side: int = 41) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, 1.0, n_side, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, n_side, dtype=np.float64)
    zs = np.linspace(0.0, 1.0, n_side, dtype=np.float64)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    phi1 = RADIUS * RADIUS - ((X - C1[0]) ** 2 + (Y - C1[1]) ** 2 + (Z - C1[2]) ** 2)
    mask = phi1 >= 0.0

    u1 = np.exp(X * Y * Z)
    u2 = np.sin(X + Y + Z)
    u = np.where(mask, u1, u2)

    xyz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    uu = u.reshape(-1, 1)
    return xyz.astype(np.float64), uu.astype(np.float64)
