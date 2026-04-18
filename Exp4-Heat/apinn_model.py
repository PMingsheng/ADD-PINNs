from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


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
        gate_tau: float = 0.02,
        f1_init: float = 1.0,
        f2_init: float = 1.0,
        learn_f1: bool = True,
        learn_f2: bool = True,
        f_scale: float = 10.0,
    ):
        super().__init__()
        if n_experts != 2:
            raise ValueError("Poisson APINN currently expects exactly two experts.")

        self.gate_tau = float(gate_tau)
        self.f_scale = float(f_scale)
        self.shared = MLP(2, shared_dim, width=shared_width, depth=shared_depth)
        self.experts = nn.ModuleList(
            [MLP(shared_dim, 1, width=expert_width, depth=expert_depth) for _ in range(n_experts)]
        )
        self.gate = GateNet(n_experts=n_experts, width=gate_width, hidden_layers=gate_hidden_layers)
        self.f1 = nn.Parameter(torch.tensor(float(f1_init)), requires_grad=bool(learn_f1))
        self.f2 = nn.Parameter(torch.tensor(float(f2_init)), requires_grad=bool(learn_f2))

    def get_f_scaled(self):
        return self.f1 * self.f_scale, self.f2 * self.f_scale

    def gate_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_logit = self.gate(x)
        weights = torch.softmax(gate_logit, dim=1)
        return weights, gate_logit

    def forward(self, x: torch.Tensor):
        weights, gate_logit = self.gate_weights(x)
        feat = self.shared(x)

        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        lift = x1 * (1.0 - x1) * x2 * (1.0 - x2)
        t_experts = torch.stack([lift * expert(feat) for expert in self.experts], dim=1)
        t_mix = (weights.unsqueeze(-1) * t_experts).sum(dim=1)

        return {
            "T": t_mix,
            "weights": weights,
            "T_experts": t_experts,
            "gate_logit": gate_logit,
        }
