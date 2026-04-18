from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from pinn_loss import piecewise_pde_residual, predict_grad
from problem import sample_interface_points


def expert_interface_jump(
    t1: torch.Tensor,
    t2: torch.Tensor,
    xy: torch.Tensor,
) -> torch.Tensor:
    grad1 = predict_grad(t1, xy)
    grad2 = predict_grad(t2, xy)
    return torch.cat([t1 - t2, grad1 - grad2], dim=1)


def compute_apinn_loss(
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
    device = xy_int.device
    f1, f2 = model.get_f_scaled()

    xy_int_req = xy_int.detach().clone().requires_grad_(True)
    pred_int = model(xy_int_req)
    r = piecewise_pde_residual(pred_int["T"], xy_int_req, f1=f1, f2=f2, circle=circle)
    loss_pde = r.square().mean()

    loss_data = torch.tensor(0.0, device=device)
    if xy_fit is not None and T_fit is not None:
        pred_fit = model(xy_fit)
        loss_data = ((pred_fit["T"] - T_fit) ** 2).mean()

    loss_interface = torch.tensor(0.0, device=device)
    if lam_interface > 0.0 and interface_points > 0:
        xy_if = sample_interface_points(interface_points, xy_int.device, circle=circle, dtype=xy_int.dtype)
        xy_if_req = xy_if.detach().clone().requires_grad_(True)
        pred_if = model(xy_if_req)
        t1 = pred_if["T_experts"][:, 0, :]
        t2 = pred_if["T_experts"][:, 1, :]
        jump = expert_interface_jump(t1, t2, xy_if_req)
        loss_interface = jump.square().mean()

    total = lam_data * loss_data + lam_pde * loss_pde + lam_interface * loss_interface
    return total, {
        "total": total,
        "data": loss_data,
        "pde": loss_pde,
        "interface": loss_interface,
    }
