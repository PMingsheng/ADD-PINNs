import numpy as np
import torch
import torch.nn as nn

import config


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=30, depth=5):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PartitionPINN(nn.Module):
    def __init__(
        self,
        width=50,
        depth=4,
        *,
        xc=0.05,
        yc=0.10,
        A=0.35,
        B=0.15,
        Gamma=np.deg2rad(-30.0),
        E2_init=0.1,
        E1_init=0.1,
        learn_E2=False,
        learn_E1=False,
    ):
        super().__init__()
        self.net_1 = MLP(2, 2, width, depth)
        self.net_2 = MLP(2, 2, width, depth)
        self.phi = MLP(2, 1, width, depth)

        self.xc, self.yc = float(xc), float(yc)
        self.A, self.B = float(A), float(B)
        self.Gamma = float(Gamma)

        self.register_buffer("cG", torch.tensor(np.cos(self.Gamma), dtype=torch.float32))
        self.register_buffer("sG", torch.tensor(np.sin(self.Gamma), dtype=torch.float32))

        self.E_1 = nn.Parameter(torch.tensor(float(E1_init)), requires_grad=bool(learn_E1))
        self.E_2 = nn.Parameter(torch.tensor(float(E2_init)), requires_grad=bool(learn_E2))

    @torch.no_grad()
    def set_ellipse(self, xc, yc, A, B, Gamma):
        self.xc, self.yc = float(xc), float(yc)
        self.A, self.B = float(A), float(B)
        self.Gamma = float(Gamma)
        self.cG.copy_(torch.tensor(np.cos(self.Gamma), dtype=self.cG.dtype, device=self.cG.device))
        self.sG.copy_(torch.tensor(np.sin(self.Gamma), dtype=self.sG.dtype, device=self.sG.device))

    def forward(self, x):
        phi_pred = self.phi(x)
        uv_1 = self.net_1(x)
        uv_2 = self.net_2(x)
        ux_1, uy_1 = uv_1.split(1, dim=1)
        ux_2, uy_2 = uv_2.split(1, dim=1)
        scale = 0.001
        return (
            phi_pred,
            scale * ux_1,
            scale * uy_1,
            scale * ux_2,
            scale * uy_2,
        )

    def get_E_scaled(self):
        return self.E_1 * config.E_SCALE, self.E_2 * config.E_SCALE
