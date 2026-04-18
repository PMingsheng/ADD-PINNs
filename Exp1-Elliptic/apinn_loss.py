from typing import Dict, Optional, Tuple

import torch

from pinn_loss import piecewise_source_term


def laplacian(u: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=xy,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2u_dx2 = torch.autograd.grad(
        outputs=grad_u[:, 0:1],
        inputs=xy,
        grad_outputs=torch.ones_like(grad_u[:, 0:1]),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]

    d2u_dy2 = torch.autograd.grad(
        outputs=grad_u[:, 1:2],
        inputs=xy,
        grad_outputs=torch.ones_like(grad_u[:, 1:2]),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1:2]

    return d2u_dx2 + d2u_dy2


def compute_apinn_loss(
    model,
    xy_int: torch.Tensor,
    *,
    xy_fit: Optional[torch.Tensor] = None,
    u_fit: Optional[torch.Tensor] = None,
    lam_data: float = 1.0,
    lam_pde: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    xy_int_req = xy_int.detach().clone().requires_grad_(True)
    pred_int = model(xy_int_req)
    u_int = pred_int["u"]

    f_int = piecewise_source_term(xy_int_req).detach()
    pde_res = laplacian(u_int, xy_int_req) + f_int
    loss_pde = (pde_res ** 2).mean()

    loss_data = torch.tensor(0.0, device=xy_int.device)
    if xy_fit is not None and u_fit is not None:
        pred_fit = model(xy_fit)
        loss_data = ((pred_fit["u"] - u_fit) ** 2).mean()

    total = lam_data * loss_data + lam_pde * loss_pde
    loss_dict = {
        "total": total,
        "data": loss_data,
        "pde": loss_pde,
    }
    return total, loss_dict
