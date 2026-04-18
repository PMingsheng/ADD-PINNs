from typing import Callable, Optional

import torch

from level_set import dirac_smooth, heaviside
from utils import activation_masks


def compute_loss(
    model,
    xy_int: torch.Tensor,
    xy_fit: Optional[torch.Tensor] = None,
    T_fit: Optional[torch.Tensor] = None,
    target_area: Optional[float] = None,
    lam: Optional[dict] = None,
    get_f1_f2: Optional[Callable[[], tuple]] = None,
    eps_eik: float = 0.001,
):
    if get_f1_f2 is None:
        raise ValueError("get_f1_f2 must be provided")

    xy_int.requires_grad_(True)
    phi, T1, T2 = model(xy_int)

    w1 = torch.relu(phi)
    w2 = torch.relu(-phi)
    mask1, mask2 = activation_masks(phi)

    loss_data = 0.0
    if xy_fit is not None:
        phi_f, T1f, T2f = model(xy_fit)
        w1f = torch.relu(phi_f)
        w2f = torch.relu(-phi_f)
        mask1f, mask2f = activation_masks(phi_f)
        loss_data = (
            mask1f * (T1f - T_fit) ** 2
            + mask2f * (T2f - T_fit) ** 2
        ).mean()
        # +(
        #     w1f * (T1f - T_fit) ** 2
        #     + w2f * (T2f - T_fit) ** 2
        # ).mean()

    f1, f2 = get_f1_f2()

    gradT1 = torch.autograd.grad(T1, xy_int, torch.ones_like(T1), create_graph=True)[0]
    gradT2 = torch.autograd.grad(T2, xy_int, torch.ones_like(T2), create_graph=True)[0]

    d2T1_xx = torch.autograd.grad(
        gradT1[:, 0:1], xy_int, torch.ones_like(gradT1[:, 0:1]), create_graph=True
    )[0][:, 0:1]
    d2T1_yy = torch.autograd.grad(
        gradT1[:, 1:2], xy_int, torch.ones_like(gradT1[:, 1:2]), create_graph=True
    )[0][:, 1:2]
    R1 = d2T1_xx + d2T1_yy + f1

    d2T2_xx = torch.autograd.grad(
        gradT2[:, 0:1], xy_int, torch.ones_like(gradT2[:, 0:1]), create_graph=True
    )[0][:, 0:1]
    d2T2_yy = torch.autograd.grad(
        gradT2[:, 1:2], xy_int, torch.ones_like(gradT2[:, 1:2]), create_graph=True
    )[0][:, 1:2]
    R2 = d2T2_xx + d2T2_yy + f2

    loss_pde = (w1 * R1**2 + w2 * R2**2).mean() + (mask1 * R1**2 + mask2 * R2**2).mean()*1e-3

    loss_bc = torch.tensor([0.0], device=xy_int.device)

    band = (phi.detach().abs() < 0.001).squeeze()
    if band.any():
        loss_fluxGrad = (gradT1[band] - gradT2[band]).pow(2).mean()
        loss_Tjump = (T1[band] - T2[band]).pow(2).mean()
        loss_if = loss_Tjump + loss_fluxGrad
    else:
        loss_if = torch.tensor(0.0, device=xy_int.device)

    gphi = torch.autograd.grad(phi, xy_int, torch.ones_like(phi), create_graph=True)[0]
    eik_res = (gphi.norm(dim=1, keepdim=True) - 1.0).pow(2)
    w = dirac_smooth(phi.detach(), epsilon=eps_eik)
    loss_eik = (w * eik_res).sum() / (w.sum() + 1e-12)

    H = heaviside(phi)
    area_pred = H.mean()
    loss_area = area_pred if target_area is None else (area_pred - target_area) ** 2

    total = (
        lam["data"] * loss_data
        + lam["pde"] * loss_pde
        + lam["bc"] * loss_bc
        + lam["interface"] * loss_if
        + lam["eik"] * loss_eik
        + lam["area"] * loss_area
    )
    core_loss = (
        lam["data"] * loss_data
        + lam["pde"] * loss_pde
        + lam["bc"] * loss_bc
        + lam["interface"] * loss_if
    )

    weighted = dict(
        data=lam["data"] * loss_data,
        pde=lam["pde"] * loss_pde,
        bc=lam["bc"] * loss_bc,
        interface=lam["interface"] * loss_if,
        eik=lam["eik"] * loss_eik,
        area=lam["area"] * loss_area,
        total=total,
    )
    return total, weighted, core_loss
