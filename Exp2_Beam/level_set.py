"""
Level-set evolution functions for LS-PINN Beam project.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

from config import BAND_EPS, H_KNN, TAU, DT, N_INNER
from pde import euler_beam_pde


def local_velocity(
    model: nn.Module,
    x: torch.Tensor,
    *,
    band_eps: float = BAND_EPS,
    h: float = H_KNN,
    eps: float = 1e-16,
    clip_q: float = 0.99,
    tau: float = TAU,
) -> torch.Tensor:
    """
    Compute level-set normal velocity based on PDE residual difference.
    
    Args:
        model: Neural network model
        x: Position coordinates
        band_eps: Narrow band width
        h: KNN radius
        eps: Small constant for numerical stability
        clip_q: Quantile for soft clipping
        tau: Small constant
        
    Returns:
        Vn: Normal velocity (zero outside narrow band)
    """
    x = x.detach().clone().requires_grad_(True)
    phi, u1_NN, u2_NN, fai1_NN, fai2_NN, Dfai1_NN, Dfai2_NN, M1_NN, M2_NN, V1_NN, V2_NN, EI1, EI2 = model(x)
    
    R_fai1, R_dfai1, R_M1, R_V1, R_Q1 = euler_beam_pde(x, u1_NN, fai1_NN, Dfai1_NN, M1_NN, V1_NN, EI1)
    R_fai2, R_dfai2, R_M2, R_V2, R_Q2 = euler_beam_pde(x, u2_NN, fai2_NN, Dfai2_NN, M2_NN, V2_NN, EI2)

    R1 = R_Q1.detach()
    R2 = R_Q2.detach()
    
    w1, w2 = torch.relu(phi), torch.relu(-phi)
    r_val = (w1 * R1 + w2 * R2) / (w1 + w2 + eps)

    # Narrow band
    band = (phi.abs() < band_eps).squeeze()
    if not band.any():
        return torch.zeros_like(phi)

    x_b = x[band]
    phi_b = phi[band].detach()

    # KNN: mean residuals on each side
    dmat = torch.cdist(x_b, x, p=2) < h
    pos = (phi.detach() > 0).T
    neg = ~pos

    r_pos = (dmat & pos).float() @ r_val / (dmat & pos).float().sum(1, keepdim=True).clamp_min(1)
    r_neg = (dmat & neg).float() @ r_val / (dmat & neg).float().sum(1, keepdim=True).clamp_min(1)

    delta = r_neg - r_pos

    # Normalize + soft clip
    scale = torch.quantile(delta.abs(), clip_q) + eps
    vel = torch.tanh(delta / scale)

    # Dirac delta regularization
    vel *= (band_eps / np.pi) / (phi_b**2 + band_eps**2 + eps)

    # Fill back
    Vn = torch.zeros_like(phi)
    Vn[band] = vel
    return Vn


def local_velocity_fit(
    model: nn.Module,
    x_band_all: torch.Tensor,
    x_fit: torch.Tensor,
    u_fit: torch.Tensor,
    *,
    band_eps: float = BAND_EPS,
    h: float = H_KNN,
    tau: float = TAU,
    eps: float = 1e-16,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Compute velocity based on strain label fitting error.
    
    Args:
        model: Neural network model
        x_band_all: All points for narrow band detection
        x_fit: Data fitting points
        u_fit: Strain labels
        band_eps: Narrow band width
        h: KNN radius
        tau: Small constant
        eps: Numerical stability constant
        verbose: Print debug info
        
    Returns:
        Vn: Normal velocity
    """
    x_all = x_band_all.reshape(-1, 1).detach()
    x_fit = x_fit.reshape(-1, 1).detach()
    u_fit = u_fit.reshape(-1)

    phi_all = model(x_all)[0].view(-1)
    band_mask = phi_all.abs() < band_eps
    
    if not band_mask.any():
        if verbose:
            print("[lv_fit] no band pts")
        return torch.zeros_like(phi_all).unsqueeze(1)

    x_band = x_all[band_mask]
    phi_band = phi_all[band_mask]

    dmat = torch.cdist(x_band, x_fit) < h

    x_fit_grad = x_fit.clone().detach().requires_grad_(True)
    phi_f, u1_NN, u2_NN, fai1_NN, fai2_NN, d2u1f, d2u2f, *_ = model(x_fit_grad)
    w1f, w2f = torch.relu(phi_f), torch.relu(-phi_f)

    strain_pred1 = d2u1f.view(-1) * h / 2
    strain_pred2 = d2u2f.view(-1) * h / 2

    res1 = (strain_pred1 - u_fit).abs().detach()
    res2 = (strain_pred2 - u_fit).abs().detach()

    is_pos = (phi_f.view(-1) > 0)

    mask_pos = dmat & is_pos.view(1, -1)
    mask_neg = dmat & (~is_pos).view(1, -1)

    w_pos = mask_pos.float()
    w_neg = mask_neg.float()

    r_pos = (w_pos @ res1.view(-1)) / w_pos.sum(dim=1).clamp_min(1.0)
    r_neg = (w_neg @ res2.view(-1)) / w_neg.sum(dim=1).clamp_min(1.0)

    delta = r_neg - r_pos

    scale = torch.quantile(delta.abs(), 0.95) + eps
    vel_b = torch.tanh(delta / scale)
    vel_b *= (band_eps / np.pi) / (phi_band**2 + band_eps**2 + eps)

    Vn = torch.zeros_like(phi_all)
    Vn[band_mask] = vel_b
    return Vn.unsqueeze(1)


def evolve_phi_local(
    model: nn.Module,
    xy: torch.Tensor,
    opt_phi: torch.optim.Optimizer,
    *,
    dt: float = DT,
    n_inner: int = N_INNER,
    stop_tol: float = 1e-6,
    band_eps: float = BAND_EPS,
    h: float = H_KNN,
    tau: float = TAU,
    typeVn: str = "PDE",
    plot_interval: int = 5,
    x_fit: Optional[torch.Tensor] = None,
    u_fit: Optional[torch.Tensor] = None,
) -> None:
    """
    Evolve level-set function using computed normal velocity.
    
    Args:
        model: Neural network model
        xy: Collocation points
        opt_phi: Optimizer for phi network
        dt: Time step
        n_inner: Number of inner iterations
        stop_tol: Convergence tolerance
        band_eps: Narrow band width
        h: KNN radius
        tau: Small constant
        typeVn: Type of velocity ('PDE', 'Grad', 'Data', 'CV', 'CV_fit')
        plot_interval: Plotting interval
        x_fit: Data fitting points
        u_fit: Strain labels
    """
    xy = xy.detach().clone().requires_grad_(True)

    Fai_empty = False
    if typeVn == 'PDE':
        Vn = local_velocity(model, xy, band_eps=band_eps, h=h, tau=tau)
    elif typeVn == 'Data':
        if x_fit is None or u_fit is None:
            raise ValueError("typeVn='Data' requires x_fit and u_fit")
        Vn = local_velocity_fit(model, xy, x_fit, u_fit, band_eps=band_eps, h=h, tau=tau)
    else:
        raise ValueError(f"Unknown typeVn '{typeVn}'")

    Vn = Vn.detach()

    # Euler extrapolation target
    phi = model.phi(xy)
    if not Fai_empty:
        grad_phi = torch.autograd.grad(phi, xy, torch.ones_like(phi), create_graph=True)[0]
        norm_g = grad_phi.norm(dim=1, keepdim=True).clamp_min(1e-6)
        phi_tar = phi + dt * Vn * norm_g
    else:
        phi_tar = phi + dt * Vn

    # Inner loop fitting
    for i in range(n_inner):
        loss_phi = torch.nn.functional.mse_loss(model.phi(xy), phi_tar.detach())
        opt_phi.zero_grad()
        loss_phi.backward()
        opt_phi.step()

        if loss_phi.item() < stop_tol:
            print(f"Converged (loss < {stop_tol}), stopping.")
            break
