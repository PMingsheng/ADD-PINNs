from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from problem import boundary_g


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
    """Gating MLP that maps input to simplex weights via softmax."""

    def __init__(self, n_experts: int, width: int = 32, hidden_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(max(hidden_layers - 1, 0)):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, n_experts))
        self.net = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        linears = [m for m in self.modules() if isinstance(m, nn.Linear)]
        for linear in linears:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class APINN(nn.Module):
    """APINN-style soft-gated model with shared trunk and two experts."""

    def __init__(
        self,
        *,
        n_experts: int = 2,
        shared_width: int = 32,
        shared_depth: int = 2,
        shared_dim: int = 16,
        expert_width: int = 64,
        expert_depth: int = 4,
        gate_width: int = 32,
        gate_hidden_layers: int = 2,
        gate_tau: float = 0.05,
    ):
        super().__init__()
        if n_experts != 2:
            raise ValueError("This APINN implementation currently requires n_experts=2")

        self.gate_tau = float(gate_tau)
        self.shared = MLP(2, shared_dim, width=shared_width, depth=shared_depth)
        self.experts = nn.ModuleList(
            [MLP(shared_dim, 1, width=expert_width, depth=expert_depth) for _ in range(n_experts)]
        )
        self.gate = GateNet(n_experts=n_experts, width=gate_width, hidden_layers=gate_hidden_layers)

    def gate_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_logit = self.gate(x)
        weights = torch.softmax(gate_logit, dim=1)
        return weights, gate_logit

    def forward(self, x: torch.Tensor):
        weights, gate_logit = self.gate_weights(x)

        feat = self.shared(x)

        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        bnd_lift = (x1 * x1 - 1.0) * (x2 * x2 - 1.0)
        g = boundary_g(x)

        u_experts = [g + bnd_lift * expert(feat) for expert in self.experts]
        u_experts = torch.cat(u_experts, dim=1)
        u_mix = (weights * u_experts).sum(dim=1, keepdim=True)

        return {
            "u": u_mix,
            "weights": weights,
            "u_experts": u_experts,
            "gate_logit": gate_logit,
        }
