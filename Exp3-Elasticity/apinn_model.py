from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from problem import boundary_displacement


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int):
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


class GateNet(nn.Module):
    def __init__(self, n_experts: int, width: int = 32, hidden_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(max(hidden_layers - 1, 0)):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, n_experts))
        self.net = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class APINN(nn.Module):
    def __init__(
        self,
        *,
        n_experts: int = 2,
        shared_width: int = 32,
        shared_depth: int = 2,
        shared_dim: int = 16,
        expert_width: int = 55,
        expert_depth: int = 4,
        gate_width: int = 32,
        gate_hidden_layers: int = 2,
        gate_tau: float = 0.05,
        eps0: float = 3.0e-3,
        nu: float = 0.30,
        E_out_init: float = 1.0,
        E_in_init: float = 1.0,
        learn_E_out: bool = True,
        learn_E_in: bool = False,
        E_scale: float = 1.0,
    ):
        super().__init__()
        if n_experts != 2:
            raise ValueError("Ellipse APINN currently expects exactly two experts.")

        self.gate_tau = float(gate_tau)
        self.eps0 = float(eps0)
        self.nu = float(nu)
        self.E_scale = float(E_scale)
        self.shared = MLP(2, shared_dim, width=shared_width, depth=shared_depth)
        self.experts = nn.ModuleList(
            [MLP(shared_dim, 2, width=expert_width, depth=expert_depth) for _ in range(n_experts)]
        )
        self.gate = GateNet(n_experts=n_experts, width=gate_width, hidden_layers=gate_hidden_layers)
        self.E_out = nn.Parameter(torch.tensor(float(E_out_init)), requires_grad=bool(learn_E_out))
        self.E_in = nn.Parameter(torch.tensor(float(E_in_init)), requires_grad=bool(learn_E_in))

    def get_E_scaled(self):
        return self.E_out * self.E_scale, self.E_in * self.E_scale

    def gate_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_logit = self.gate(x)
        weights = torch.softmax(gate_logit, dim=1)
        return weights, gate_logit

    def forward(self, x: torch.Tensor):
        weights, gate_logit = self.gate_weights(x)
        feat = self.shared(x)

        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        lift = (x1 * x1 - 1.0) * (x2 * x2 - 1.0)
        u_bc = boundary_displacement(x, eps0=self.eps0, nu=self.nu)

        u_experts = torch.stack([u_bc + lift * expert(feat) for expert in self.experts], dim=1)
        u_mix = (weights.unsqueeze(-1) * u_experts).sum(dim=1)

        return {
            "u": u_mix,
            "weights": weights,
            "u_experts": u_experts,
            "gate_logit": gate_logit,
        }
