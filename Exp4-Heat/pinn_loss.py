from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from pde import div_kgrad
from problem import piecewise_source


def predict_grad(t: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(
        t,
        xy,
        grad_outputs=torch.ones_like(t),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad


def piecewise_pde_residual(
    t: torch.Tensor,
    xy: torch.Tensor,
    *,
    f1: torch.Tensor | float,
    f2: torch.Tensor | float,
    circle=None,
) -> torch.Tensor:
    f_field = piecewise_source(xy, f1=f1, f2=f2, circle=circle)
    return div_kgrad(t, f_field, xy, keep_graph=True)


def compute_pinn_loss(
    model,
    xy_int: torch.Tensor,
    *,
    xy_fit: Optional[torch.Tensor] = None,
    T_fit: Optional[torch.Tensor] = None,
    lam_data: float = 1.0,
    lam_pde: float = 1.0,
    lam_interface: float = 0.0,
    interface_points: int = 0,
    circle=None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    del lam_interface, interface_points

    device = xy_int.device
    f1, f2 = model.get_f_scaled()

    xy_int_req = xy_int.detach().clone().requires_grad_(True)
    t_int = model(xy_int_req)
    r = piecewise_pde_residual(t_int, xy_int_req, f1=f1, f2=f2, circle=circle)
    loss_pde = r.square().mean()

    loss_data = torch.tensor(0.0, device=device)
    if xy_fit is not None and T_fit is not None:
        t_fit_pred = model(xy_fit)
        loss_data = ((t_fit_pred - T_fit) ** 2).mean()

    loss_interface = torch.tensor(0.0, device=device)

    total = lam_data * loss_data + lam_pde * loss_pde
    return total, {
        "total": total,
        "data": loss_data,
        "pde": loss_pde,
        "interface": loss_interface,
    }
