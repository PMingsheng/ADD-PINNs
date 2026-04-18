from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


DEFAULT_EPS = 1e-8
FLOWER_N = 0.0


# Region convention in this project:
#   phi <  0  -> region-1 (outside flower):  u_outside
#   phi >= 0  -> region-2 (inside flower): u_inside


def _split_xy(xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    rho = x * x + y * y
    # Follow the uploaded formulation: theta = atan2(x, y).
    theta = torch.atan2(x, y)
    return x, y, rho, theta


def flower_radius(theta: torch.Tensor) -> torch.Tensor:
    return 0.4 + 0.1 * torch.sin(FLOWER_N * theta)


def phi_signed_flower(xy: torch.Tensor) -> torch.Tensor:
    _, _, rho, theta = _split_xy(xy)
    r = flower_radius(theta)
    return r * r - rho


def phi_flower(xy: torch.Tensor) -> torch.Tensor:
    return torch.tanh(phi_signed_flower(xy))


def u_inside(xy: torch.Tensor) -> torch.Tensor:
    _, _, rho, _ = _split_xy(xy)
    return torch.exp(rho)


def _a_outside(rho: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    # A(rho) = 0.1 * rho^2 - 0.01 * log(2 * sqrt(rho))
    # eps is used only for numerical safety near rho=0.
    r_safe = torch.sqrt(rho + eps)
    return 0.1 * rho * rho - 0.01 * torch.log(2.0 * r_safe + eps)


def u_outside(xy: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    _, _, rho, _ = _split_xy(xy)
    A = _a_outside(rho, eps=eps)
    return torch.sqrt(torch.clamp(A, min=eps))


def exact_solution(xy: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    phi_s = phi_signed_flower(xy)
    u_out = u_outside(xy, eps=eps)
    u_in = u_inside(xy)
    return torch.where(phi_s < 0.0, u_out, u_in)


def f_region_outside(xy: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    # f = -Delta(u) for u = u_outside in region-1.
    _, _, rho, _ = _split_xy(xy)

    u = u_outside(xy, eps=eps)
    rho_safe = rho + eps

    # A(rho), A'(rho), A''(rho)
    A_rho = 0.2 * rho - 0.005 / rho_safe
    A_rho2 = 0.2 + 0.005 / (rho_safe * rho_safe)

    u_rho = A_rho / (2.0 * (u + eps))
    u_rho2 = A_rho2 / (2.0 * (u + eps)) - (A_rho * A_rho) / (4.0 * (u + eps) ** 3)

    lap_u = 4.0 * rho * u_rho2 + 4.0 * u_rho
    return -lap_u


def f_region_inside(xy: torch.Tensor) -> torch.Tensor:
    # f = -Delta(u) for u = exp(x^2 + y^2) in region-2.
    _, _, rho, _ = _split_xy(xy)
    e = torch.exp(rho)
    return -4.0 * (rho + 1.0) * e


def grad_u_outside(xy: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    x, y, rho, _ = _split_xy(xy)
    u = u_outside(xy, eps=eps)
    rho_safe = rho + eps
    A_rho = 0.2 * rho - 0.005 / rho_safe
    coef = A_rho / (u + eps)
    return torch.cat([x * coef, y * coef], dim=1)


def grad_u_inside(xy: torch.Tensor) -> torch.Tensor:
    x, y, rho, _ = _split_xy(xy)
    e = torch.exp(rho)
    return torch.cat([2.0 * x * e, 2.0 * y * e], dim=1)


def jump_w(xy: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    # [u] = u1 - u2, with region-1 = outside, region-2 = inside.
    return u_outside(xy, eps=eps) - u_inside(xy)


def jump_v(xy: torch.Tensor, n_hat: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    # [du/dn] = grad(u1)·n - grad(u2)·n
    g1 = grad_u_outside(xy, eps=eps)
    g2 = grad_u_inside(xy)
    dn1 = (g1 * n_hat).sum(dim=1, keepdim=True)
    dn2 = (g2 * n_hat).sum(dim=1, keepdim=True)
    return dn1 - dn2


def boundary_g(xy: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    # Dirichlet data on ∂Omega comes from the outside branch.
    return u_outside(xy, eps=eps)


def flower_interface_curve(n_theta: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    r = 0.4 + 0.1 * np.sin(FLOWER_N * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x.astype(np.float64), y.astype(np.float64)


def generate_full_field(n_side: int = 201, *, eps: float = DEFAULT_EPS) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-1.0, 1.0, n_side, dtype=np.float64)
    ys = np.linspace(-1.0, 1.0, n_side, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    rho = X * X + Y * Y
    theta = np.arctan2(X, Y)
    r = 0.4 + 0.1 * np.sin(FLOWER_N * theta)
    phi_s = r * r - rho 

    u_in = np.exp(rho)
    A = 0.1 * rho * rho - 0.01 * np.log(2.0 * np.sqrt(rho + eps) + eps)
    u_out = np.sqrt(np.clip(A, eps, None))
    
    u = np.where(phi_s < 0.0, u_out, u_in)

    xy = np.stack([X.ravel(), Y.ravel()], axis=1)
    uu = u.reshape(-1, 1)
    return xy.astype(np.float64), uu.astype(np.float64)
