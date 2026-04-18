"""
APINN model for beam problem.
Implements Eq.(10)-style structure: sum_i G_i(x) * E_i(h(x)).
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import L


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class APINNBeam(nn.Module):
    """
    APINN for 1D beam:
    - shared network h(x)
    - expert networks E_i(h(x)) producing only (u, EI)
    - gate network G(x) producing simplex weights
    """

    def __init__(
        self,
        *,
        n_experts: int = 2,
        shared_width: int = 112,
        shared_depth: int = 3,
        shared_dim: int = 48,
        expert_width: int = 112,
        expert_depth: int = 3,
        gate_width: int = 16,
        gate_depth: int = 3,
    ):
        super().__init__()
        if n_experts < 2:
            raise ValueError("n_experts must be >= 2 for APINN beam.")
        self.n_experts = int(n_experts)

        self.shared = MLP(1, shared_dim, width=shared_width, depth=shared_depth)
        self.experts = nn.ModuleList(
            [MLP(shared_dim, 2, width=expert_width, depth=expert_depth) for _ in range(n_experts)]
        )
        self.gate = MLP(1, n_experts, width=gate_width, depth=gate_depth)

    @staticmethod
    def _apply_scaling(x: torch.Tensor, var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u, EI = (
            var[:, 0:1],
            var[:, 1:2],
        )
        # Keep same scaling style as project baseline for learned fields.
        u = u * x * (x - L) * 0.01
        EI = F.softplus(EI)
        return u, EI

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.shared(x)
        gate_logits = self.gate(x)
        weights = torch.softmax(gate_logits, dim=1)  # partition of unity

        expert_outputs = []
        for expert in self.experts:
            raw = expert(feat)
            expert_outputs.append(self._apply_scaling(x, raw))

        # Mix each learned quantity with gate weights.
        mixed_components: List[torch.Tensor] = []
        w = weights.unsqueeze(-1)  # [N, E, 1]
        for k in range(2):
            comp_stack = torch.stack([eo[k] for eo in expert_outputs], dim=1)  # [N, E, 1]
            comp_mix = (w * comp_stack).sum(dim=1)  # [N, 1]
            mixed_components.append(comp_mix)

        u, EI = mixed_components
        return {
            "u": u,
            "EI": EI,
            "weights": weights,
            "gate_logits": gate_logits,
        }


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
