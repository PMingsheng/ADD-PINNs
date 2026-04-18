from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import config


DEFAULT_GATE_PRETRAIN_EPOCHS = 1000


def _budget_aligned_blocks(pretrain_epochs: int) -> List["TrainBlock"]:
    blocks = [TrainBlock(epochs=block.epochs, lr=block.lr) for block in config.PIMOE3DTrainConfig().blocks]
    remain = int(pretrain_epochs)

    for block in reversed(blocks):
        if remain <= 0:
            break
        take = min(block.epochs, remain)
        block.epochs -= take
        remain -= take

    if remain > 0:
        raise ValueError("gate pretrain epochs exceed the ADD-PINNs 3D training budget")

    return [block for block in blocks if block.epochs > 0]


@dataclass
class APINN3DDataConfig:
    nx: int = config.PIMOE3DDataConfig().nx
    ny: int = config.PIMOE3DDataConfig().ny
    nz: int = config.PIMOE3DDataConfig().nz


@dataclass
class APINN3DModelConfig:
    n_experts: int = 2

    # Parameter count is 85,252, which is almost the same as ADD-PINNs 3D (85,251).
    shared_width: int = 112
    shared_depth: int = 2
    shared_dim: int = 48

    expert_width: int = 96
    expert_depth: int = 4

    gate_width: int = 32
    gate_hidden_layers: int = 2
    gate_tau: float = 0.05


@dataclass
class TrainBlock:
    epochs: int
    lr: float


@dataclass
class APINN3DTrainConfig:
    seed: int = config.PIMOE3DTrainConfig().seed

    # Keep point sampling aligned with the ADD-PINNs 3D baseline.
    interior_points: int = config.PIMOE3DTrainConfig().interior_points
    interior_margin: float = config.PIMOE3DTrainConfig().interior_margin
    sample_batch_size: int = config.PIMOE3DTrainConfig().sample_batch_size

    # Keep data/PDE weights aligned with the ADD-PINNs baseline.
    lam_data: float = config.PIMOE3DTrainConfig().lam_weights["data"]
    lam_pde: float = config.PIMOE3DTrainConfig().lam_weights["pde"]

    # The optimization blocks plus gate pretraining sum to the same total budget as ADD-PINNs/PINN.
    blocks: List[TrainBlock] = field(default_factory=lambda: _budget_aligned_blocks(DEFAULT_GATE_PRETRAIN_EPOCHS))

    print_every: int = 500
    record_every: int = 50

    # Residual-based adaptive refinement (RAR), aligned with ADD-PINNs settings.
    rar_every: int = config.PIMOE3DTrainConfig().rar_every
    rar_candidate_points: int = config.PIMOE3DTrainConfig().rar_candidate_points
    rar_topk: int = config.PIMOE3DTrainConfig().rar_topk
    rar_start_epoch: int = config.PIMOE3DTrainConfig().rar_start_epoch

    # Gate pretraining using known interface sign.
    gate_pretrain_epochs: int = DEFAULT_GATE_PRETRAIN_EPOCHS
    gate_pretrain_lr: float = 5e-4
    gate_pretrain_use_hard_target: bool = True


@dataclass
class APINN3DEvalConfig:
    bbox: Tuple[float, float, float, float, float, float] = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    n_grid: int = 51
    eval_batch_size: int = 4096


@dataclass
class APINN3DOutputConfig:
    output_dir: str = "outputs_apinn3d_c1_sphere"
    loss_csv_name: str = "loss_list_global.csv"
    loss_png_name: str = "Loss.png"
    field_npz_name: str = "final_fields.npz"
    slices_png_name: str = "u_residual_slices.png"
    snapshot_every: int = 5000
    snapshot_dir: str = "snapshots"
    u_heatmap_dir: str = "u_heatmaps"
    f_heatmap_dir: str = "f_heatmaps"
    gate_heatmap_dir: str = "gate_heatmaps"

    def resolve_output_dir(self, project_root: Path) -> Path:
        out = project_root / self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        return out
