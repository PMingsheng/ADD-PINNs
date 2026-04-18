"""
Loss for reduced-order beam PINN.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from config import H, LOSS_WEIGHTS
from pde import euler_beam_pde


def _dEI_regularization_near_m0(
    x: torch.Tensor,
    M: torch.Tensor,
    EI: torch.Tensor,
    *,
    band: float = 0.025,
    eps: float = 1e-8,
) -> torch.Tensor:
    dEI = torch.autograd.grad(
        EI,
        x,
        grad_outputs=torch.ones_like(EI),
        create_graph=True,
        retain_graph=True,
    )[0]
    dEI_norm2 = dEI.pow(2)

    x_flat = x.view(-1)
    m_flat = M.detach().view(-1)
    idx_sort = torch.argsort(x_flat)
    x_sorted = x_flat[idx_sort]
    m_sorted = m_flat[idx_sort]

    sign = torch.sign(m_sorted)
    sign[sign == 0] = 1
    change_pos = (sign[:-1] * sign[1:] < 0).nonzero().flatten()

    roots = []
    for idx in change_pos.tolist():
        xl, xr = x_sorted[idx], x_sorted[idx + 1]
        fl, fr = m_sorted[idx], m_sorted[idx + 1]
        root = xl + fl.abs() / (fl.abs() + fr.abs() + eps) * (xr - xl)
        roots.append(root.item())

    if not roots:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    roots_t = torch.tensor(roots, device=x.device, dtype=x.dtype)
    dist = (x_flat[:, None] - roots_t[None, :]).abs()
    mask = (dist.min(dim=1)[0] <= band).float().view(-1, 1)
    return (dEI_norm2 * mask).sum() / (mask.sum() + 1e-12)


def compute_reduced_loss(
    model: nn.Module,
    x_int: torch.Tensor,
    *,
    x_fit: Optional[torch.Tensor] = None,
    strain_fit: Optional[torch.Tensor] = None,
    u_fit_disp: Optional[torch.Tensor] = None,
    lam: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if lam is None:
        lam = LOSS_WEIGHTS.copy()

    device = x_int.device
    x_int_req = x_int.clone().requires_grad_(True)

    u, theta, kappa, M, V, EI = model(x_int_req)
    R_theta, R_kappa, R_M, R_V, R_Q = euler_beam_pde(
        x_int_req, u, theta, kappa, M, V, EI
    )

    loss_theta = (R_theta ** 2).mean()
    loss_kappa = (R_kappa ** 2).mean()
    loss_M = (R_M ** 2).mean()
    loss_V = (R_V ** 2).mean()
    loss_Q = (R_Q ** 2).mean()
    loss_dEI = _dEI_regularization_near_m0(x_int_req, M, EI)

    loss_data_strain = torch.zeros((), device=device)
    loss_data_disp = torch.zeros((), device=device)
    if x_fit is not None and strain_fit is not None:
        u_fit_pred, _, kappa_fit_pred, _, _, _ = model(x_fit)
        strain_pred = (H / 2.0) * kappa_fit_pred
        loss_data_strain = ((strain_pred - strain_fit) ** 2).mean()
        if u_fit_disp is not None:
            loss_data_disp = ((u_fit_pred - u_fit_disp) ** 2).mean()

    loss_data = loss_data_strain + loss_data_disp

    # Keep weighting convention aligned with original project LOSS_WEIGHTS.
    w_data_strain = float(lam.get("data_strain", lam.get("data", 0.0)))
    w_data_disp = float(lam.get("data_disp", lam.get("data", 0.0)))
    w_fai = float(lam.get("fai", 0.0))      # du/dx consistency term
    w_dfai = float(lam.get("dfai", 0.0))    # dtheta/dx consistency term
    w_M = float(lam.get("M", 0.0))
    w_V = float(lam.get("V", 0.0))
    w_Q = float(lam.get("Q", 0.0))
    w_dEI = float(lam.get("dEI", 0.0))

    data_strain_w = w_data_strain * loss_data_strain
    data_disp_w = w_data_disp * loss_data_disp
    fai_w = w_fai * loss_theta
    dfai_w = w_dfai * loss_kappa
    M_w = w_M * loss_M
    V_w = w_V * loss_V
    Q_w = w_Q * loss_Q
    dEI_w = w_dEI * loss_dEI

    total = data_strain_w + data_disp_w + fai_w + dfai_w + M_w + V_w + Q_w + dEI_w

    loss_dict = {
        "total": total,
        "data": loss_data,
        "data_strain": loss_data_strain,
        "data_disp": loss_data_disp,
        "fai": loss_theta,
        "dfai": loss_kappa,
        "M": loss_M,
        "V": loss_V,
        "Q": loss_Q,
        "dEI": loss_dEI,
        "data_strain_w": data_strain_w,
        "data_disp_w": data_disp_w,
        "data_w": data_strain_w + data_disp_w,
        "fai_w": fai_w,
        "dfai_w": dfai_w,
        "M_w": M_w,
        "V_w": V_w,
        "Q_w": Q_w,
        "dEI_w": dEI_w,
    }
    return total, loss_dict
