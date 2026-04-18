from typing import Dict, Optional, Tuple

import torch

from level_set_3d import dirac_smooth, heaviside
from problem_3d import (
    ALPHA_INSIDE,
    ALPHA_OUTSIDE,
    BETA_INSIDE,
    beta_outside,
    grad_beta_outside,
    interface_beta1,
    interface_beta2,
    source_region_inside,
    source_region_outside,
)


def _grad_lap(u: torch.Tensor, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=xyz,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2x = torch.autograd.grad(
        outputs=grad_u[:, 0:1],
        inputs=xyz,
        grad_outputs=torch.ones_like(grad_u[:, 0:1]),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]
    d2y = torch.autograd.grad(
        outputs=grad_u[:, 1:2],
        inputs=xyz,
        grad_outputs=torch.ones_like(grad_u[:, 1:2]),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1:2]
    d2z = torch.autograd.grad(
        outputs=grad_u[:, 2:3],
        inputs=xyz,
        grad_outputs=torch.ones_like(grad_u[:, 2:3]),
        create_graph=True,
        retain_graph=True,
    )[0][:, 2:3]
    return grad_u, d2x + d2y + d2z


def _target_volume_fraction() -> float:
    # Omega volume is 1.0. Omega_1 is one sphere with radius r=0.1.
    r = 0.1
    return float((4.0 / 3.0) * torch.pi * (r ** 3))


def compute_pimoe3d_loss(
    model,
    xyz_int: torch.Tensor,
    *,
    xyz_fit: Optional[torch.Tensor] = None,
    u_fit: Optional[torch.Tensor] = None,
    lam: Optional[dict] = None,
    interface_band: float = 0.01,
    phi_trainable: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if lam is None:
        raise ValueError("lam must be provided")

    xyz_req = xyz_int.detach().clone().requires_grad_(True)
    phi, u1, u2 = model(xyz_req)

    w1 = torch.relu(phi)
    w2 = torch.relu(-phi)
    m1 = (phi > 0.0).to(phi.dtype)
    m2 = (phi < 0.0).to(phi.dtype)

    loss_data = torch.tensor(0.0, device=xyz_req.device)
    if xyz_fit is not None and u_fit is not None:
        phi_f, u1f, u2f = model(xyz_fit)
        w1f = torch.relu(phi_f)
        w2f = torch.relu(-phi_f)
        m1f = (phi_f > 0.0).to(phi_f.dtype)
        m2f = (phi_f < 0.0).to(phi_f.dtype)
        loss_data_relu = (w1f * (u1f - u_fit) ** 2 + w2f * (u2f - u_fit) ** 2).mean()
        loss_data_norm = (m1f * (u1f - u_fit) ** 2 + m2f * (u2f - u_fit) ** 2).mean()
        loss_data = loss_data_norm + loss_data_relu

    grad_u1, lap_u1 = _grad_lap(u1, xyz_req)
    grad_u2, lap_u2 = _grad_lap(u2, xyz_req)

    f1 = source_region_inside(xyz_req).detach()
    f2 = source_region_outside(xyz_req).detach()

    r1 = -BETA_INSIDE * lap_u1 + ALPHA_INSIDE * u1 - f1

    beta2 = beta_outside(xyz_req)
    gbeta2 = grad_beta_outside(xyz_req)
    div_beta_grad_u2 = beta2 * lap_u2 + (gbeta2 * grad_u2).sum(dim=1, keepdim=True)
    r2 = -div_beta_grad_u2 + ALPHA_OUTSIDE * u2 - f2

    pde_relu_pos = w1 * r1.pow(2)
    pde_relu_neg = w2 * r2.pow(2)
    # scale1 =10
    # scale2 =1
    # if not phi_trainable:
    #     scale2 = 10
    #     scale1 = 100
    #     # In phi-frozen phase, rebalance PDE-ReLU loss by side-weight ratio.
    #     # Amplify the side with smaller total gate weight.
    #     sum_pos = w1.detach().sum()
    #     sum_neg = w2.detach().sum()
    #     eps = torch.as_tensor(1e-12, device=xyz_req.device, dtype=xyz_req.dtype)
    #     if sum_pos < sum_neg:
    #         scale = (sum_neg + eps) / (sum_pos + eps) * 1000
    #         pde_relu_pos = pde_relu_pos * scale
    #     elif sum_neg < sum_pos:
    #         scale = (sum_pos + eps) / (sum_neg + eps) * 1000
    #         pde_relu_neg = pde_relu_neg * scale
    loss_pde_relu = (pde_relu_pos + pde_relu_neg).mean()
    loss_pde_norm = (m1 * r1.pow(2) + m2 * r2.pow(2)).mean()
    loss_pde = loss_pde_relu + loss_pde_norm
    loss_if = torch.tensor(0.0, device=xyz_req.device)
    if lam.get("interface", 0.0) > 0.0:
        band = (phi.detach().abs() < interface_band).squeeze()
        if band.any():
            gphi = torch.autograd.grad(
                outputs=phi,
                inputs=xyz_req,
                grad_outputs=torch.ones_like(phi),
                create_graph=True,
            )[0]
            n_hat = gphi[band] / gphi[band].norm(dim=1, keepdim=True).clamp_min(1e-8)

            xyz_b = xyz_req[band]
            u_jump = u1[band] - u2[band]
            dn1 = (grad_u1[band] * n_hat).sum(dim=1, keepdim=True)
            dn2 = (grad_u2[band] * n_hat).sum(dim=1, keepdim=True)
            flux_jump = BETA_INSIDE * dn1 - beta_outside(xyz_b) * dn2

            w_tar = interface_beta1(xyz_b).detach()
            v_tar = interface_beta2(xyz_b).detach()
            loss_if = (u_jump - w_tar).pow(2).mean() + (flux_jump - v_tar).pow(2).mean()

    gphi_all = torch.autograd.grad(
        outputs=phi,
        inputs=xyz_req,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
    )[0]
    eik_res = (gphi_all.norm(dim=1, keepdim=True) - 1.0).pow(2)
    loss_eik = (dirac_smooth(phi, epsilon=0.05).detach() * eik_res).mean()
    # loss_eik = (eik_res).mean()

    vol_pred = heaviside(phi).mean()
    vol_tar = _target_volume_fraction()
    vol_tar_t = torch.as_tensor(vol_tar, device=xyz_req.device, dtype=xyz_req.dtype)
    loss_volume = (vol_pred - vol_tar_t).pow(2)

    loss_surface = (dirac_smooth(phi, epsilon=0.05) * gphi_all.norm(dim=1, keepdim=True)).mean()

    total = (
        lam["data"] * loss_data
        + lam["pde"] * loss_pde
        + lam.get("interface", 0.0) * loss_if
        + lam.get("eik", 0.0) * loss_eik
        + lam.get("volume", 0.0) * loss_volume
        + lam.get("surface", 0.0) * loss_surface
    )

    weighted = {
        "total": total,
        "data": lam["data"] * loss_data,
        "pde": lam["pde"] * loss_pde,
        "interface": lam.get("interface", 0.0) * loss_if,
        "eik": lam.get("eik", 0.0) * loss_eik,
        "volume": lam.get("volume", 0.0) * loss_volume,
        "surface": lam.get("surface", 0.0) * loss_surface,
    }
    raw = {
        "total": loss_data + loss_pde + loss_if + loss_eik + loss_volume + loss_surface,
        "data": loss_data,
        "pde": loss_pde,
        "interface": loss_if,
        "eik": loss_eik,
        "volume": loss_volume,
        "surface": loss_surface,
    }
    return total, weighted, raw


# Generic alias for base ADD-PINNs pipeline.
compute_loss = compute_pimoe3d_loss
