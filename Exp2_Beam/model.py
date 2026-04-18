"""
Neural network models for LS-PINN Beam project.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import L, PHI_WIDTH, PHI_DEPTH, VAR_WIDTH, VAR_DEPTH


class MLP(nn.Module):
    """Multi-layer perceptron with Tanh activation."""
    
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
    """
    Partition-based Physics-Informed Neural Network for beam problems.
    
    Outputs:
        phi: Level-set function
        u1, u2: Displacements for each partition
        fai1, fai2: Rotations for each partition
        Dfai1, Dfai2: Curvatures for each partition
        M1, M2: Moments for each partition
        V1, V2: Shear outputs for each partition
        EI1, EI2: Stiffness for each partition
    """
    
    def __init__(self, width: int = VAR_WIDTH, depth: int = VAR_DEPTH):
        super().__init__()
        self.phi = MLP(1, 1, PHI_WIDTH, PHI_DEPTH)
        self.variable = MLP(1, 12, width, 5)

    def forward(self, x: torch.Tensor):
        phi = self.phi(x)
        variable = self.variable(x)
        
        # Split outputs
        u1, u2 = variable[:, 0:1], variable[:, 1:2]
        fai1, fai2 = variable[:, 2:3], variable[:, 3:4]
        Dfai1, Dfai2 = variable[:, 4:5], variable[:, 5:6]
        M1, M2 = variable[:, 6:7], variable[:, 7:8]
        V1, V2 = variable[:, 8:9], variable[:, 9:10]
        EI1, EI2 = variable[:, 10:11], variable[:, 11:12]
        
        # Apply boundary conditions and scaling
        return (
            phi,
            u1 * x * (x - L) * 0.001,
            u2 * x * (x - L) * 0.001,
            fai1 * x * 0.01,
            fai2 * x * 0.01,
            Dfai1 * (x - L) * 0.01,
            Dfai2 * (x - L) * 0.01,
            M1 * (x - L) * 0.1,
            M2 * (x - L) * 0.1,
            V1 * 0.1,
            V2 * 0.1,
            F.softplus(EI1),
            F.softplus(EI2),
        )
