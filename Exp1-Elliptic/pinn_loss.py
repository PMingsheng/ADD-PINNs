from typing import Dict, Optional, Tuple

import torch

from problem import f_region_inside, f_region_outside, phi_signed_flower


def piecewise_source_term(xy: torch.Tensor) -> torch.Tensor:
    """f(x, y) selected by the known flower interface."""
    phi = phi_signed_flower(xy)
    f_in = f_region_inside(xy)
    f_out = f_region_outside(xy)
    return torch.where(phi >= 0.0, f_in, f_out)


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


def compute_pinn_loss(
    model,
    xy_int: torch.Tensor,
    *,
    xy_fit: Optional[torch.Tensor] = None,
    u_fit: Optional[torch.Tensor] = None,
    lam_data: float = 1.0,
    lam_pde: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    xy_int_req = xy_int.detach().clone().requires_grad_(True)
    u_int = model(xy_int_req)

    f_int = piecewise_source_term(xy_int_req).detach()
    pde_res = laplacian(u_int, xy_int_req) + f_int
    loss_pde = (pde_res ** 2).mean()

    loss_data = torch.tensor(0.0, device=xy_int.device)
    if xy_fit is not None and u_fit is not None:
        u_pred = model(xy_fit)
        loss_data = ((u_pred - u_fit) ** 2).mean()

    total = lam_data * loss_data + lam_pde * loss_pde
    loss_dict = {
        "total": total,
        "data": loss_data,
        "pde": loss_pde,
    }
    return total, loss_dict
