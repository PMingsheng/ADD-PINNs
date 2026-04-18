"""
Loss computation functions for LS-PINN Beam project.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Union, Optional

from config import H, INTERFACE_SAMPLE_EACH, INTERFACE_SAMPLE_RADIUS, MIN_ROOT_GAP, LOSS_WEIGHTS, EIKONAL_DELTA_EPS
from utils import heaviside, heaviside_derivative
from pde import euler_beam_pde


def interface_loss(
    *,
    model: nn.Module,
    x_int: torch.Tensor,
    phi: torch.Tensor,
    sample_each: int = INTERFACE_SAMPLE_EACH,
    sample_radius: float = INTERFACE_SAMPLE_RADIUS,
    min_root_gap: float = MIN_ROOT_GAP,
    verbose: bool = False,
    return_points: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[int, torch.Tensor]]]:
    """
    Compute interface loss ensuring continuity across phi=0.
    
    Args:
        model: Neural network model
        x_int: Interior points
        phi: Level-set values at x_int
        sample_each: Number of samples on each side of root
        sample_radius: Sampling radius around root
        min_root_gap: Minimum gap between detected roots
        verbose: Print debug info
        return_points: Whether to return sampled points
        
    Returns:
        loss_interface: Interface loss value
        pts_out: (optional) Dictionary of sampled points per root
    """
    device = x_int.device
    x_int = x_int.view(-1)
    phi = phi.detach().view(-1)
    eps = 1e-8

    # Detect all phi=0 roots
    idx_sort = torch.argsort(x_int)
    x_sorted = x_int[idx_sort]
    phi_sorted = phi[idx_sort]

    sign = torch.sign(phi_sorted)
    sign[sign == 0] = 1
    change_pos = (sign[:-1] * sign[1:] < 0).nonzero().flatten()

    roots = []
    for idx in change_pos.tolist():
        xl, xr = x_sorted[idx], x_sorted[idx + 1]
        fl, fr = phi_sorted[idx], phi_sorted[idx + 1]
        root = xl + fl.abs() / (fl.abs() + fr.abs() + eps) * (xr - xl)
        roots.append(root.item())

    # Merge close roots
    roots = torch.tensor(sorted(roots), device=device)
    if roots.numel():
        keep = [roots[0]]
        for r in roots[1:]:
            if (r - keep[-1]) > min_root_gap:
                keep.append(r)
        roots = torch.tensor(keep, device=device)

    if roots.numel() == 0:
        if verbose:
            print("[interface_loss] no root found.")
        return (
            (torch.tensor(0.0, device=device), {})
            if return_points
            else torch.tensor(0.0, device=device)
        )

    loss_interface = torch.tensor(0.0, device=device)
    pts_out: Dict[int, torch.Tensor] = {}

    # Loop over roots
    for i_root, x0 in enumerate(roots):
        left_offsets = -torch.rand(sample_each, device=device) * sample_radius
        right_offsets = torch.rand(sample_each, device=device) * sample_radius
        x_new = torch.cat([x0 + left_offsets, x0 + right_offsets]).unsqueeze(-1)
        x_new = x_new.clamp_(0.0, 1.0).requires_grad_(True)

        # Forward pass
        (
            phi_pred, u1, u2, fai1, fai2, Dfai1, Dfai2,
            M1, M2, V1, V2, EI1, EI2
        ) = model(x_new)

        # Autograd derivatives
        # fai_dNN1 = torch.autograd.grad(u1, x_new, torch.ones_like(u1), create_graph=True, retain_graph=True)[0]
        # fai_dNN2 = torch.autograd.grad(u2, x_new, torch.ones_like(u2), create_graph=True, retain_graph=True)[0]

        # V_dNN1 = torch.autograd.grad(M1, x_new, torch.ones_like(M1), create_graph=True, retain_graph=True)[0]
        # V_dNN2 = torch.autograd.grad(M2, x_new, torch.ones_like(M2), create_graph=True, retain_graph=True)[0]
        M_dNN1 = EI1 * Dfai1
        M_dNN2 = EI2 * Dfai2

        # Split points by geometry
        k = sample_each
        idx_geo_l = torch.arange(0, k, device=device)
        idx_geo_r = torch.arange(k, 2 * k, device=device)

        is_geo_l_pos = (phi_pred[idx_geo_l] >= 0).float().mean() > 0.5
        idx_pos, idx_neg = (idx_geo_l, idx_geo_r) if is_geo_l_pos else (idx_geo_r, idx_geo_l)
        sign_pos = phi_pred[idx_pos] >= 0
        sign_neg = phi_pred[idx_neg] >= 0

        def pick(t1, t2, s):
            return torch.where(s, t1, t2)

        u_l = pick(u1[idx_pos], u2[idx_pos], sign_pos)
        u_r = pick(u1[idx_neg], u2[idx_neg], sign_neg)
        fai_l = pick(fai1[idx_pos], fai2[idx_pos], sign_pos)
        fai_r = pick(fai1[idx_neg], fai2[idx_neg], sign_neg)
        # fai_dNN_l = pick(fai_dNN1[idx_pos], fai_dNN2[idx_pos], sign_pos)
        # fai_dNN_r = pick(fai_dNN1[idx_neg], fai_dNN2[idx_neg], sign_neg)
        M_l = pick(M1[idx_pos], M2[idx_pos], sign_pos)
        M_r = pick(M1[idx_neg], M2[idx_neg], sign_neg)
        M_dNN_l = pick(M_dNN1[idx_pos], M_dNN2[idx_pos], sign_pos)
        M_dNN_r = pick(M_dNN1[idx_neg], M_dNN2[idx_neg], sign_neg)
        V_l = pick(V1[idx_pos], V2[idx_pos], sign_pos)
        V_r = pick(V1[idx_neg], V2[idx_neg], sign_neg)
        # V_dNN_l = pick(V_dNN1[idx_pos], V_dNN2[idx_pos], sign_pos)
        # V_dNN_r = pick(V_dNN1[idx_neg], V_dNN2[idx_neg], sign_neg)

        eps_norm = 1e-12
        def norm_diff(a, b, scale):
            return ((a - b) / scale.detach()).pow(2).mean()

        scale_u = 0.5 * (u_l.abs().mean() + u_r.abs().mean()) + eps_norm
        scale_fai = 0.5 * (fai_l.abs().mean() + fai_r.abs().mean()) + eps_norm
        # scale_fai_dNN = 0.5 * (fai_dNN_l.abs().mean() + fai_dNN_r.abs().mean()) + eps_norm
        scale_M = 0.5 * (M_l.abs().mean() + M_r.abs().mean()) + eps_norm
        scale_M_dNN = 0.5 * (M_dNN_l.abs().mean() + M_dNN_r.abs().mean()) + eps_norm
        scale_V = 0.5 * (V_l.abs().mean() + V_r.abs().mean()) + eps_norm
        # scale_V_dNN = 0.5 * (V_dNN_l.abs().mean() + V_dNN_r.abs().mean()) + eps_norm

        loss_interface += (
            norm_diff(u_l, u_r, scale_u)
            + norm_diff(fai_l, fai_r, scale_fai)
            # + norm_diff(fai_dNN_l, fai_dNN_r, scale_fai_dNN)
            + 10*norm_diff(M_l, M_r, scale_M)
            # + 10*norm_diff(M_dNN_l, M_dNN_r, scale_M_dNN)
            + norm_diff(V_l, V_r, scale_V)
            # + norm_diff(V_dNN_l, V_dNN_r, scale_V_dNN)
        )
        if return_points:
            pts_out[i_root] = x_new.detach().view(-1)

        if verbose:
            print(
                f"[interface_loss] root#{i_root}  x0={x0:.5f}  "
                f"sampled {2 * sample_each} pts  radius={sample_radius}"
            )

    return (loss_interface, pts_out) if return_points else loss_interface


def compute_loss(
    model: nn.Module,
    x_int: torch.Tensor,
    x_fit: Optional[torch.Tensor] = None,
    strain_fit: Optional[torch.Tensor] = None,
    u_fit_disp: Optional[torch.Tensor] = None,
    target_area: Optional[float] = None,
    lam: Optional[Dict[str, float]] = None,
    xy_int_const: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """
    Compute total loss for training.
    
    Args:
        model: Neural network model
        x_int: Interior collocation points
        x_fit: Data fitting points
        strain_fit: Strain labels
        u_fit_disp: Displacement labels at x_fit (optional)
        target_area: Target area constraint
        lam: Loss weight dictionary
        xy_int_const: Constant interior points for interface loss
        
    Returns:
        total_loss: Total weighted loss
        loss_dict: Dictionary of individual loss terms
        core_loss: Core physics loss (excluding regularization)
    """
    if lam is None:
        lam = LOSS_WEIGHTS.copy()
    
    eps = 1e-12
    device = x_int.device

    # Forward pass
    x_int = x_int.clone().requires_grad_(True)
    phi, u1_NN, u2_NN, fai1_NN, fai2_NN, Dfai1_NN, Dfai2_NN, M1_NN, M2_NN, V1_NN, V2_NN, EI1, EI2 = model(x_int)
    
    dEI1 = torch.autograd.grad(EI1, x_int, grad_outputs=torch.ones_like(EI1), create_graph=True, retain_graph=True)[0]
    dEI2 = torch.autograd.grad(EI2, x_int, grad_outputs=torch.ones_like(EI2), create_graph=True, retain_graph=True)[0]

    w1 = torch.relu(phi)
    w2 = torch.relu(-phi)
    w1_mask = (phi >= 0).to(phi.dtype).detach()
    w2_mask = (phi < 0).to(phi.dtype).detach()

    # dEI regularization near M=0 (within +/-0.025 in x)
    m_eps = 1e-8
    m_band = 0.025
    m_blend = (w1 * M1_NN + w2 * M2_NN) / (w1 + w2 + eps)
    idx_sort = torch.argsort(x_int.view(-1))
    x_sorted = x_int.view(-1)[idx_sort]
    m_sorted = m_blend.view(-1)[idx_sort].detach()
    sign = torch.sign(m_sorted)
    sign[sign == 0] = 1
    change_pos = (sign[:-1] * sign[1:] < 0).nonzero().flatten()
    roots = []
    for idx in change_pos.tolist():
        xl, xr = x_sorted[idx], x_sorted[idx + 1]
        fl, fr = m_sorted[idx], m_sorted[idx + 1]
        root = xl + fl.abs() / (fl.abs() + fr.abs() + m_eps) * (xr - xl)
        roots.append(root.item())
    if roots:
        roots_t = torch.tensor(roots, device=device)
        dist = (x_int.view(-1, 1) - roots_t.view(1, -1)).abs()
        mask_global = (dist.min(dim=1)[0] <= m_band).float().view(-1, 1)
    else:
        mask_global = torch.zeros_like(x_int)

    dEI1_norm2 = dEI1.pow(2).sum(dim=1, keepdim=True)
    dEI2_norm2 = dEI2.pow(2).sum(dim=1, keepdim=True)
    loss_dEI1 = dEI1_norm2 * w1_mask
    loss_dEI2 = dEI2_norm2 * w2_mask
    loss_dEI = (loss_dEI1 + loss_dEI2).mean()

    lam_data_strain = lam.get('data_strain', lam.get('data', 0.0))
    lam_data_disp = lam.get('data_disp', lam.get('data', 0.0))

    # Data fitting loss
    loss_data = torch.tensor(0.0, device=device)
    loss_data_strain = torch.tensor(0.0, device=device)
    loss_data_disp = torch.tensor(0.0, device=device)
    if x_fit is not None and strain_fit is not None:
        x_fit_grad = x_fit.clone().detach().requires_grad_(True)
        phi_f, u1_NN_fit, u2_NN_fit, _, _, Dfai1_NN_fit, Dfai2_NN_fit, *_ = model(x_fit_grad)
        w1f, w2f = torch.relu(phi_f), torch.relu(-phi_f)
        w_1f = (phi_f >= 0).to(phi_f.dtype).detach()
        w_2f = (phi_f < 0).to(phi_f.dtype).detach()
        
        strain_pred1 = (H / 2) * Dfai1_NN_fit
        strain_pred2 = (H / 2) * Dfai2_NN_fit
        
        loss_data_strain = (
            (w1f * (strain_pred1 - strain_fit) ** 2)
            + (w2f * (strain_pred2 - strain_fit) ** 2)
        ).mean()*10 + (
            (w_1f * (strain_pred1 - strain_fit) ** 2)
            + (w_2f * (strain_pred2 - strain_fit) ** 2)
        ).mean()

        loss_data_disp = (
            (w_1f * (u1_NN_fit - u_fit_disp) ** 2)
            + (w_2f * (u2_NN_fit - u_fit_disp) ** 2)
        ).mean()
        loss_data = loss_data_strain + loss_data_disp

    # PDE residuals
    R_fai1, R_dfai1, R_M1, R_V1, R_Q1 = euler_beam_pde(x_int, u1_NN, fai1_NN, Dfai1_NN, M1_NN, V1_NN, EI1)
    R_fai2, R_dfai2, R_M2, R_V2, R_Q2 = euler_beam_pde(x_int, u2_NN, fai2_NN, Dfai2_NN, M2_NN, V2_NN, EI2)

    loss_fai = ((w1_mask * R_fai1**2) + (w2_mask * R_fai2**2)).mean()
    loss_dfai = ((w1_mask * R_dfai1**2) + (w2_mask * R_dfai2**2)).mean()
    # loss_dfai = ((w1 * R_dfai1**2) + (w2 * R_dfai2**2)).mean()
    loss_M = ((w1_mask * R_M1**2) + (w2_mask * R_M2**2)).mean()
    loss_V = ((w1_mask * R_V1**2) + (w2_mask * R_V2**2)).mean()
    loss_Q = ((w1_mask * R_Q1**2) + (w2_mask * R_Q2**2)).mean()

    # Interface loss
    x_for_interface = xy_int_const if xy_int_const is not None else x_int
    loss_if, pts = interface_loss(
        model=model,
        x_int=x_for_interface,
        phi=phi,
        sample_each=INTERFACE_SAMPLE_EACH,
        sample_radius=INTERFACE_SAMPLE_RADIUS,
        verbose=False,
        return_points=True,
    )

    mean_w = (w1 + w2).mean()
    loss_weight = -torch.log(0.1 * mean_w + eps)

    # Eikonal and area losses
    g_phi = torch.autograd.grad(phi, x_int, torch.ones_like(phi), create_graph=True)[0]
    eik_res = (g_phi.norm(2, 1, keepdim=True) - 1.0).pow(2)
    delta = heaviside_derivative(phi, epsilon=EIKONAL_DELTA_EPS)
    loss_eik = (delta.detach() * eik_res).mean()
    H_val = heaviside(phi)
    area_pred = H_val.mean()
    loss_area = area_pred if target_area is None else (area_pred - target_area) ** 2

    # Total loss
    core_loss = (
        lam_data_strain * loss_data_strain
        + lam_data_disp * loss_data_disp
        + lam['fai'] * loss_fai
        + lam['dfai'] * loss_dfai
        + lam['M'] * loss_M
        + lam['V'] * loss_V
        + lam['Q'] * loss_Q
        + lam['interface'] * loss_if
    )

    total_loss = (
        core_loss
        + lam['weight'] * loss_weight
        + lam['eik'] * loss_eik
        + lam['area'] * loss_area
        + lam['dEI'] * loss_dEI
    )

    loss_dict = {
        'data': loss_data,
        'data_strain': loss_data_strain,
        'data_disp': loss_data_disp,
        'fai': loss_fai,
        'dfai': loss_dfai,
        'weight': loss_weight,
        'M': loss_M,
        'V': loss_V,
        'Q': loss_Q,
        'interface': loss_if,
        'eik': loss_eik,
        'area': loss_area,
        'dEI': loss_dEI,
        'total': total_loss,
    }

    return total_loss, loss_dict, core_loss
