"""
Utility functions for LS-PINN Beam project.
"""
import torch
import numpy as np
import random

from config import E_BG, E_DEF_1, E_DEF_2, I


def set_seed(seed: int = 1234) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def heaviside(phi: torch.Tensor, epsilon: float = 0.005) -> torch.Tensor:
    """Smooth Heaviside function."""
    return 0.5 * (1 + (2 / torch.pi) * torch.atan(phi / epsilon))


def heaviside_derivative(phi: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
    """Derivative of smooth Heaviside function."""
    return (1 / (np.pi * epsilon)) * (1 / (1 + (phi / epsilon) ** 2))


def EI_true(x: torch.Tensor) -> torch.Tensor:
    """
    Ground truth stiffness function EI(x).
    
    Args:
        x: Position tensor
        
    Returns:
        EI values at each position
    """
    EI = torch.full_like(x, E_BG * I)
    EI[(x >= 0.3) & (x < 0.6)] = E_DEF_1 * I
    EI[(x >= 0.6) & (x <= 1.0)] = E_DEF_2 * I
    return EI
