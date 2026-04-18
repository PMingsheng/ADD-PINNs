"""
PDE residual functions for Euler-Bernoulli beam.
"""
import torch
from typing import Tuple

from config import EI_REAL


def euler_beam_pde(
    x: torch.Tensor,
    u_NN: torch.Tensor,
    fai_NN: torch.Tensor,
    Dfai_NN: torch.Tensor,
    M_NN: torch.Tensor,
    V_NN: torch.Tensor,
    EI: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Euler-Bernoulli beam PDE residuals.
    
    The governing equation is: d^2/dx^2(EI(x) * d^2u(x)/dx^2) = f(x)
    
    Args:
        x: Position coordinates
        u_NN: Network displacement output
        fai_NN: Network rotation output
        Dfai_NN: Network curvature output
        M_NN: Network moment output
        V_NN: Network shear output
        EI: Stiffness values
        
    Returns:
        R_fai: Rotation residual
        R_dfai: Curvature residual
        R_M: Moment residual
        R_V: Shear residual
        R_Q: Load residual
    """
    eps = 1e-8
    
    # Compute derivatives via autograd
    fai_dNN = torch.autograd.grad(
        u_NN, x, grad_outputs=torch.ones_like(u_NN),
        create_graph=True, retain_graph=True
    )[0]
    
    Dfai_dNN = torch.autograd.grad(
        fai_NN, x, grad_outputs=torch.ones_like(fai_NN),
        create_graph=True, retain_graph=True
    )[0]
    
    M_dNN = Dfai_NN * EI
    
    V_dNN = torch.autograd.grad(
        M_NN, x, grad_outputs=torch.ones_like(M_NN),
        create_graph=True, retain_graph=True
    )[0]
    
    Q = torch.autograd.grad(
        V_NN, x, grad_outputs=torch.ones_like(V_NN),
        create_graph=True, retain_graph=True
    )[0]
    
    Q_true = 1e3 / EI_REAL
    
    # Compute residuals
    R_fai = fai_NN - fai_dNN
    R_dfai = Dfai_NN - Dfai_dNN
    R_M = M_NN - M_dNN
    R_V = V_NN - V_dNN
    R_Q = Q - Q_true
    
    return R_fai, R_dfai, R_M, R_V, R_Q
