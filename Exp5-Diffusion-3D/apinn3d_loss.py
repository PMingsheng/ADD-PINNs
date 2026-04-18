from typing import Dict, Optional, Tuple

import torch

from problem_3d import alpha_piecewise, beta_piecewise, grad_beta_piecewise, source_term_piecewise


def _grad_and_laplacian(u: torch.Tensor, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=xyz,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2u_dx2 = torch.autograd.grad(
        outputs=grad_u[:, 0:1],
        inputs=xyz,
        grad_outputs=torch.ones_like(grad_u[:, 0:1]),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]
    d2u_dy2 = torch.autograd.grad(
        outputs=grad_u[:, 1:2],
        inputs=xyz,
        grad_outputs=torch.ones_like(grad_u[:, 1:2]),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1:2]
    d2u_dz2 = torch.autograd.grad(
        outputs=grad_u[:, 2:3],
        inputs=xyz,
        grad_outputs=torch.ones_like(grad_u[:, 2:3]),
        create_graph=True,
        retain_graph=True,
    )[0][:, 2:3]

    lap_u = d2u_dx2 + d2u_dy2 + d2u_dz2
    return grad_u, lap_u


def compute_apinn3d_loss(
    model,
    xyz_int: torch.Tensor,
    *,
    xyz_fit: Optional[torch.Tensor] = None,
    u_fit: Optional[torch.Tensor] = None,
    lam_data: float = 1.0,
    lam_pde: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    xyz_req = xyz_int.detach().clone().requires_grad_(True)
    pred_int = model(xyz_req)
    u_int = pred_int["u"]

    grad_u, lap_u = _grad_and_laplacian(u_int, xyz_req)
    beta = beta_piecewise(xyz_req)
    grad_beta = grad_beta_piecewise(xyz_req)
    alpha = alpha_piecewise(xyz_req)
    f = source_term_piecewise(xyz_req).detach()

    div_beta_grad_u = beta * lap_u + (grad_beta * grad_u).sum(dim=1, keepdim=True)
    pde_res = -div_beta_grad_u + alpha * u_int - f
    loss_pde = (pde_res ** 2).mean()

    loss_data = torch.tensor(0.0, device=xyz_int.device)
    if xyz_fit is not None and u_fit is not None:
        pred_fit = model(xyz_fit)
        loss_data = ((pred_fit["u"] - u_fit) ** 2).mean()

    total = lam_data * loss_data + lam_pde * loss_pde
    weighted = {
        "total": total,
        "data": lam_data * loss_data,
        "pde": lam_pde * loss_pde,
    }
    raw = {
        "total": loss_data + loss_pde,
        "data": loss_data,
        "pde": loss_pde,
    }
    return total, weighted, raw
