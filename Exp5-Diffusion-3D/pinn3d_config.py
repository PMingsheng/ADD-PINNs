from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import config


@dataclass
class PINN3DDataConfig:
    nx: int = config.PIMOE3DDataConfig().nx
    ny: int = config.PIMOE3DDataConfig().ny
    nz: int = config.PIMOE3DDataConfig().nz


@dataclass
class PINN3DModelConfig:
    # width=167, depth=4 gives 85,004 params, close to ADD-PINNs (85,251).
    width: int = 167
    depth: int = 4


@dataclass
class TrainBlock:
    epochs: int
    lr: float


@dataclass
class PINN3DTrainConfig:
    seed: int = config.PIMOE3DTrainConfig().seed

    # Keep point sampling aligned with the ADD-PINNs 3D baseline.
    interior_points: int = config.PIMOE3DTrainConfig().interior_points
    interior_margin: float = config.PIMOE3DTrainConfig().interior_margin
    sample_batch_size: int = config.PIMOE3DTrainConfig().sample_batch_size

    lam_data: float = config.PIMOE3DTrainConfig().lam_weights["data"]
    lam_pde: float = config.PIMOE3DTrainConfig().lam_weights["pde"]
    blocks: List[TrainBlock] = field(
        default_factory=lambda: [
            TrainBlock(epochs=block.epochs, lr=block.lr) for block in config.PIMOE3DTrainConfig().blocks
        ]
    )

    print_every: int = 500
    record_every: int = 50

    # Residual-based adaptive refinement (RAR), aligned with ADD-PINNs settings.
    rar_every: int = config.PIMOE3DTrainConfig().rar_every
    rar_candidate_points: int = config.PIMOE3DTrainConfig().rar_candidate_points
    rar_topk: int = config.PIMOE3DTrainConfig().rar_topk
    rar_start_epoch: int = config.PIMOE3DTrainConfig().rar_start_epoch


@dataclass
class PINN3DEvalConfig:
    bbox: Tuple[float, float, float, float, float, float] = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    n_grid: int = 51
    eval_batch_size: int = 4096


@dataclass
class PINN3DOutputConfig:
    output_dir: str = "outputs_pinn3d_c1_sphere"
    loss_csv_name: str = "loss_list_global.csv"
    loss_png_name: str = "Loss.png"
    field_npz_name: str = "final_fields.npz"
    slices_png_name: str = "u_residual_slices.png"
    snapshot_every: int = 5000
    snapshot_dir: str = "snapshots"
    u_heatmap_dir: str = "u_heatmaps"
    f_heatmap_dir: str = "f_heatmaps"

    def resolve_output_dir(self, project_root: Path) -> Path:
        out = project_root / self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        return out
