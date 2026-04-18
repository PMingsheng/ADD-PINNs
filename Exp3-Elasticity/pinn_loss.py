from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from pde import div_sigma_from_u, lame_from_E, strain_from_u, stress_from_u
from problem import piecewise_modulus, sample_interface_points


def predict_strain(U: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    exx, eyy, exy = strain_from_u(U, xy)
    return torch.cat([exx, eyy, exy], dim=1)


def piecewise_pde_residual(
    U: torch.Tensor,
    xy: torch.Tensor,
    *,
    E_out: torch.Tensor | float,
    E_in: torch.Tensor | float,
    nu: float,
    ellipse=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    E_field = piecewise_modulus(xy, E_out=E_out, E_in=E_in, ellipse=ellipse)
    lam_field, mu_field = lame_from_E(E_field, nu)
    return div_sigma_from_u(U, xy, lam_field, mu_field, keep_graph=True)


def interface_traction_jump(
    U: torch.Tensor,
    xy: torch.Tensor,
    normals: torch.Tensor,
    *,
    E_out: torch.Tensor | float,
    E_in: torch.Tensor | float,
    nu: float,
) -> torch.Tensor:
    lam_out, mu_out = lame_from_E(torch.as_tensor(E_out, dtype=xy.dtype, device=xy.device), nu)
    lam_in, mu_in = lame_from_E(torch.as_tensor(E_in, dtype=xy.dtype, device=xy.device), nu)

    sxx_out, syy_out, sxy_out = stress_from_u(U, xy, lam_out, mu_out)
    sxx_in, syy_in, sxy_in = stress_from_u(U, xy, lam_in, mu_in)

    nx = normals[:, 0:1]
    ny = normals[:, 1:2]
    tx_out = sxx_out * nx + sxy_out * ny
    ty_out = sxy_out * nx + syy_out * ny
    tx_in = sxx_in * nx + sxy_in * ny
    ty_in = sxy_in * nx + syy_in * ny
    return torch.cat([tx_out - tx_in, ty_out - ty_in], dim=1)


def compute_pinn_loss(
    model,
    xy_int: torch.Tensor,
    *,
    xy_u: Optional[torch.Tensor] = None,
    U_fit: Optional[torch.Tensor] = None,
    xy_eps: Optional[torch.Tensor] = None,
    E_fit: Optional[torch.Tensor] = None,
    lam_data_u: float = 1.0,
    lam_data_eps: float = 1.0,
    lam_pde: float = 1.0,
    lam_interface: float = 0.0,
    interface_points: int = 0,
    nu: float = 0.30,
    ellipse=None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = xy_int.device
    E_out, E_in = model.get_E_scaled()

    xy_int_req = xy_int.detach().clone().requires_grad_(True)
    U_int = model(xy_int_req)
    Rx, Ry = piecewise_pde_residual(U_int, xy_int_req, E_out=E_out, E_in=E_in, nu=nu, ellipse=ellipse)
    loss_pde = (Rx.square() + Ry.square()).mean()

    loss_data_u = torch.tensor(0.0, device=device)
    if xy_u is not None and U_fit is not None:
        U_pred = model(xy_u)
        loss_data_u = ((U_pred - U_fit) ** 2).mean()

    loss_data_eps = torch.tensor(0.0, device=device)
    if xy_eps is not None and E_fit is not None:
        xy_eps_req = xy_eps.detach().clone().requires_grad_(True)
        U_eps = model(xy_eps_req)
        eps_pred = predict_strain(U_eps, xy_eps_req)
        loss_data_eps = ((eps_pred - E_fit) ** 2).mean()

    loss_interface = torch.tensor(0.0, device=device)
    if lam_interface > 0.0 and interface_points > 0:
        xy_if, normals = sample_interface_points(interface_points, device, ellipse=ellipse, dtype=xy_int.dtype)
        xy_if_req = xy_if.detach().clone().requires_grad_(True)
        U_if = model(xy_if_req)
        traction_jump = interface_traction_jump(
            U_if,
            xy_if_req,
            normals,
            E_out=E_out,
            E_in=E_in,
            nu=nu,
        )
        loss_interface = traction_jump.square().mean()

    total = (
        lam_data_u * loss_data_u
        + lam_data_eps * loss_data_eps
        + lam_pde * loss_pde
        + lam_interface * loss_interface
    )
    return total, {
        "total": total,
        "data_u": loss_data_u,
        "data_eps": loss_data_eps,
        "pde": loss_pde,
        "interface": loss_interface,
    }
