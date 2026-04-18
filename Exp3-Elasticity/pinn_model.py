from __future__ import annotations

import torch
import torch.nn as nn

from problem import boundary_displacement


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
        eps0: float = 3.0e-3,
        nu: float = 0.30,
        E_out_init: float = 1.0,
        E_in_init: float = 1.0,
        learn_E_out: bool = True,
        learn_E_in: bool = False,
        E_scale: float = 1.0,
    ):
        super().__init__()
        self.net = MLP(2, 2, width=width, depth=depth)
        self.eps0 = float(eps0)
        self.nu = float(nu)
        self.E_scale = float(E_scale)
        self.E_out = nn.Parameter(torch.tensor(float(E_out_init)), requires_grad=bool(learn_E_out))
        self.E_in = nn.Parameter(torch.tensor(float(E_in_init)), requires_grad=bool(learn_E_in))

    def get_E_scaled(self):
        return self.E_out * self.E_scale, self.E_in * self.E_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        lift = (x1 * x1 - 1.0) * (x2 * x2 - 1.0)
        u_bc = boundary_displacement(x, eps0=self.eps0, nu=self.nu)
        return u_bc + lift * self.net(x)
