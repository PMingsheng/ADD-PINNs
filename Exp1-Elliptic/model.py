import torch
import torch.nn as nn
import torch.nn.functional as F

from problem import boundary_g, phi_signed_flower


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 30, depth: int = 5):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConstrainedPhi(nn.Module):
    def __init__(self, width: int = 50, depth: int = 4):
        super().__init__()
        self.base = MLP(2, 1, width, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enforce phi=0 exactly on the flower interface.
        # phi_signed_flower(x)=0 on the interface; positive scale keeps sign convention.
        phi_raw = self.base(x)
        phi_scale = F.softplus(phi_raw) + 1e-3
        return phi_signed_flower(x) * phi_scale


class PartitionPINN(nn.Module):
    def __init__(self, width: int = 30, depth: int = 3):
        super().__init__()
        # self.phi = ConstrainedPhi(width, depth)
        self.phi = MLP(2, 1, width, depth)
        self.net_1 = MLP(2, 1, width, depth)
        self.net_2 = MLP(2, 1, width, depth)

    def forward(self, x: torch.Tensor):
        phi = self.phi(x)

        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        # Zero on the full boundary of [-1, 1]^2.
        bnd_lift = (x1 * x1 - 1.0) * (x2 * x2 - 1.0)

        g = boundary_g(x)
        u1 = self.net_1(x)
        u2 = g + bnd_lift * self.net_2(x)

        return phi, u1, u2
