import torch
import torch.nn as nn

from problem_3d import boundary_g


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinglePINN3D(nn.Module):
    """Single-network PINN for Omega=[0,1]^3 with hard Dirichlet BC."""

    def __init__(self, width: int = 96, depth: int = 4):
        super().__init__()
        self.net = MLP(3, 1, width=width, depth=depth)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        x = xyz[:, 0:1]
        y = xyz[:, 1:2]
        z = xyz[:, 2:3]

        # Vanishes on all six faces of [0,1]^3.
        lift = x * (1.0 - x) * y * (1.0 - y) * z * (1.0 - z)

        g = boundary_g(xyz)
        return g + lift * self.net(xyz)
