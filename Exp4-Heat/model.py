import torch
import torch.nn as nn


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


class PartitionPINN(nn.Module):
    def __init__(self, width: int = 50, depth: int = 4):
        super().__init__()
        self.phi = MLP(2, 1, width, depth)
        self.net_1 = MLP(2, 1, width, depth)
        self.net_2 = MLP(2, 1, width, depth)

    def forward(self, x: torch.Tensor):
        phi = self.phi(x)
        T1 = self.net_1(x)
        T2raw = self.net_2(x)
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        T2 = T2raw * x1 * (x1 - 1) * x2 * (x2 - 1)
        return phi, T1, T2
