from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinglePINN(nn.Module):
    def __init__(
        self,
        *,
        width: int = 64,
        depth: int = 4,
        f1_init: float = 1.0,
        f2_init: float = 1.0,
        learn_f1: bool = True,
        learn_f2: bool = True,
        f_scale: float = 10.0,
    ):
        super().__init__()
        self.net = MLP(2, 1, width=width, depth=depth)
        self.f_scale = float(f_scale)
        self.f1 = nn.Parameter(torch.tensor(float(f1_init)), requires_grad=bool(learn_f1))
        self.f2 = nn.Parameter(torch.tensor(float(f2_init)), requires_grad=bool(learn_f2))

    def get_f_scaled(self):
        return self.f1 * self.f_scale, self.f2 * self.f_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        lift = x1 * (1.0 - x1) * x2 * (1.0 - x2)
        return lift * self.net(x)
