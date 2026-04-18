from typing import Optional

import torch

from level_set import dirac_smooth, heaviside
from problem import (
    boundary_g,
    f_region_outside,
    f_region_inside,
    jump_v,
    jump_w,
)


def compute_loss(
    model,
    xy_int: torch.Tensor,
    *,
    xy_fit: Optional[torch.Tensor] = None,
    u_fit: Optional[torch.Tensor] = None,
    xy_bnd: Optional[torch.Tensor] = None,
    target_area: Optional[float] = None,
    lam: Optional[dict] = None,
    interface_band: float = 0.01,
):
    if lam is None:
        raise ValueError("lam must be provided")

    xy_int = xy_int.requires_grad_(True)
    phi, u1, u2 = model(xy_int)

    w1 = torch.relu(phi)
    w2 = torch.relu(-phi)
    w1_normalized = (phi > 0).to(phi.dtype)
    w2_normalized = (phi < 0).to(phi.dtype)

    loss_data = torch.tensor(0.0, device=xy_int.device)
    if xy_fit is not None and u_fit is not None:
        phi_f, u1f, u2f = model(xy_fit)
        w1f = torch.relu(phi_f)
        w2f = torch.relu(-phi_f)
        w1f_normalized = (phi_f > 0).to(phi_f.dtype)
        w2f_normalized = (phi_f < 0).to(phi_f.dtype)
        loss_data = (
            w1f_normalized * (u1f - u_fit) ** 2
            + w2f_normalized * (u2f - u_fit) ** 2
        ).mean() +\
        ((w1f * (u1f - u_fit) ** 2 + w2f * (u2f - u_fit) ** 2)).mean()*100

    # Current sign convention:
    #   phi > 0 : inside flower   -> branch u1
    #   phi < 0 : outside domain  -> branch u2
    f1 = f_region_inside(xy_int).detach()
    f2 = f_region_outside(xy_int).detach()

    gradu1 = torch.autograd.grad(u1, xy_int, torch.ones_like(u1), create_graph=True)[0]
    gradu2 = torch.autograd.grad(u2, xy_int, torch.ones_like(u2), create_graph=True)[0]

    d2u1_xx = torch.autograd.grad(
        gradu1[:, 0:1], xy_int, torch.ones_like(gradu1[:, 0:1]), create_graph=True
    )[0][:, 0:1]
    d2u1_yy = torch.autograd.grad(
        gradu1[:, 1:2], xy_int, torch.ones_like(gradu1[:, 1:2]), create_graph=True
    )[0][:, 1:2]
    R1 = d2u1_xx + d2u1_yy + f1

    d2u2_xx = torch.autograd.grad(
        gradu2[:, 0:1], xy_int, torch.ones_like(gradu2[:, 0:1]), create_graph=True
    )[0][:, 0:1]
    d2u2_yy = torch.autograd.grad(
        gradu2[:, 1:2], xy_int, torch.ones_like(gradu2[:, 1:2]), create_graph=True
    )[0][:, 1:2]
    R2 = d2u2_xx + d2u2_yy + f2

    loss_pde = (w1 * R1**2 + w2 * R2**2).mean()*0.001 + (
        w1_normalized.detach() * R1**2
        + w2_normalized.detach() * R2**2
    ).mean()
    # loss_pde = torch.tensor(0.0, device=xy_int.device)

    loss_bc = torch.tensor(0.0, device=xy_int.device)
    # if xy_bnd is not None and xy_bnd.numel() > 0:
    #     _, _, u2b = model(xy_bnd)
    #     gb = boundary_g(xy_bnd)
    #     loss_bc = torch.mean((u2b - gb) ** 2)

    # band = (phi.detach().abs() < interface_band).squeeze()
    # if band.any():
    #     gphi = torch.autograd.grad(phi, xy_int, torch.ones_like(phi), create_graph=True)[0]
    #     n_hat = gphi[band] / gphi[band].norm(dim=1, keepdim=True).clamp_min(1e-8)

    #     xy_band = xy_int[band]
    #     w_tar = jump_w(xy_band).detach()
    #     v_tar = jump_v(xy_band, n_hat.detach()).detach()

    #     u_jump = u1[band] - u2[band]
    #     flux_jump = ((gradu1[band] - gradu2[band]) * n_hat).sum(dim=1, keepdim=True)

    #     loss_ujump = (u_jump - w_tar).pow(2).mean()
    #     loss_flux = (flux_jump - v_tar).pow(2).mean()
    #     loss_if = loss_ujump + loss_flux
    # else:
    #     loss_if = torch.tensor(0.0, device=xy_int.device)
    loss_if = torch.tensor(0.0, device=xy_int.device)

    gphi_all = torch.autograd.grad(phi, xy_int, torch.ones_like(phi), create_graph=True)[0]
    eik_res = (gphi_all.norm(dim=1, keepdim=True) - 1.0).pow(2)
    eps_band = 0.05
    delta = dirac_smooth(phi, epsilon=eps_band)
    
    loss_eik = (delta.detach() * eik_res).mean()

    H = heaviside(phi)
    area_pred = H.mean()
    if target_area is None:
        loss_area = area_pred
        # min_area = 0.05
        # k = 100.0
        # loss_area = torch.exp(-k * (area_pred - min_area))
    else:
        loss_area = (area_pred - target_area) ** 2

    delta = dirac_smooth(phi)
    gphi_norm = gphi_all.norm(dim=1, keepdim=True)
    loss_perimeter = (delta * gphi_norm).mean()

    lam_perimeter = lam.get("perimeter", 0.0)

    total = (
        lam["data"] * loss_data
        + lam["pde"] * loss_pde
        + lam["bc"] * loss_bc
        + lam["interface"] * loss_if
        + lam["eik"] * loss_eik
        + lam["area"] * loss_area
        + lam_perimeter * loss_perimeter
    )

    weighted = dict(
        data = lam["data"] * loss_data,
        pde = lam["pde"] * loss_pde,
        bc = lam["bc"] * loss_bc,
        interface = lam["interface"] * loss_if,
        eik = lam["eik"] * loss_eik,
        area = lam["area"] * loss_area,
        perimeter = lam_perimeter * loss_perimeter,
        total = total,
    )
    unweighted_total = (
        loss_data + loss_pde + loss_bc + loss_if + loss_eik + loss_area + loss_perimeter
    )
    unweighted = dict(
        data=loss_data,
        pde=loss_pde,
        bc=loss_bc,
        interface=loss_if,
        eik=loss_eik,
        area=loss_area,
        perimeter=loss_perimeter,
        total=unweighted_total,
    )
    return total, weighted, unweighted
