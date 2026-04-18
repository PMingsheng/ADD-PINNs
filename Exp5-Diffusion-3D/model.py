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


class PartitionPINN3D(nn.Module):
    """ADD-PINNs style model: one partition field + two sub-networks."""

    def __init__(self, width: int = 96, depth: int = 4):
        super().__init__()
        self.phi = MLP(3, 1, width=width, depth=depth)
        self.net_1 = MLP(3, 1, width=width, depth=depth)
        self.net_2 = MLP(3, 1, width=width, depth=depth)

    @staticmethod
    def boundary_lift(xyz: torch.Tensor) -> torch.Tensor:
        x = xyz[:, 0:1]
        y = xyz[:, 1:2]
        z = xyz[:, 2:3]
        return x * (1.0 - x) * y * (1.0 - y) * z * (1.0 - z)

    def forward(self, xyz: torch.Tensor):
        phi = self.phi(xyz)
        g = boundary_g(xyz)
        lift = self.boundary_lift(xyz)
        u1 = g + lift * self.net_1(xyz)
        u2 = g + lift * self.net_2(xyz)
        return phi, u1, u2


# Generic alias for base ADD-PINNs pipeline.
PartitionPINN = PartitionPINN3D
