"""
Standard PINN model for the 1D beam problem.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import L, VAR_DEPTH, VAR_WIDTH


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PINNBeam(nn.Module):
    """
    Standard single-network PINN for beam:
    - network outputs displacement u(x) and stiffness EI(x)
    - theta/kappa/M/V/Q are derived via autograd
    """

    def __init__(self, width: int = VAR_WIDTH, depth: int = VAR_DEPTH):
        super().__init__()
        self.variable = MLP(1, 2, width=width, depth=depth)

    def forward(self, x: torch.Tensor):
        var = self.variable(x)
        u, EI = var[:, 0:1], var[:, 1:2]

        u = u * x * (x - L) * 0.01
        EI = F.softplus(EI)
        return {
            "u": u,
            "EI": EI,
        }


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
