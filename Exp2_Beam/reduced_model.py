"""
Reduced-order PINN model for beam problem (single branch, no partition).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import L, VAR_WIDTH, VAR_DEPTH


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


class ReducedOrderPINN(nn.Module):
    """
    Single-branch reduced-order model with outputs:
    u, theta, kappa, M, V, EI
    """

    def __init__(self, width: int = VAR_WIDTH, depth: int = VAR_DEPTH):
        super().__init__()
        self.variable = MLP(1, 6, width=width, depth=depth)

    def forward(self, x: torch.Tensor):
        var = self.variable(x)
        u, theta, kappa, M, V, EI = (
            var[:, 0:1],
            var[:, 1:2],
            var[:, 2:3],
            var[:, 3:4],
            var[:, 4:5],
            var[:, 5:6],
        )

        # Keep same scaling/boundary lifting style as existing ADD-PINNs model.
        u = u * x * (x - L) * 0.01
        theta = theta * x * 0.01
        kappa = kappa * (x - L) * 0.01
        M = M * (x - L) * 0.1
        V = V * 0.1
        EI = F.softplus(EI)
        return u, theta, kappa, M, V, EI

