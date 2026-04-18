import torch
import torch.nn as nn

from problem import boundary_g


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinglePINN(nn.Module):
    """Single-network PINN with hard Dirichlet boundary constraint."""

    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        self.net = MLP(2, 1, width=width, depth=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]

        # Equals zero on the whole boundary of [-1, 1]^2.
        boundary_lift = (x1 * x1 - 1.0) * (x2 * x2 - 1.0)

        g = boundary_g(x)
        u = g + boundary_lift * self.net(x)
        return u
