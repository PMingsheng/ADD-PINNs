from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PIMOE3DDataConfig:
    nx: int = 18
    ny: int = 18
    nz: int = 18


@dataclass
class PIMOE3DModelConfig:
    width: int = 96
    depth: int = 4


@dataclass
class TrainBlock:
    epochs: int
    lr: float


@dataclass
class PIMOE3DTrainConfig:
    seed: int = 1234

    interior_points: int = 80000
    interior_margin: float = 0.0
    sample_batch_size: int = 100000

    lam_weights: dict = field(
        default_factory=lambda: {
            "data": 1,
            "pde": 1e-4,
            "interface": 0.0,
            "eik": 1e-4,
            "volume": 1e-5,
            "surface": 0,
        }
    )
    blocks: List[TrainBlock] = field(
        default_factory=lambda: [
            TrainBlock(epochs=5000, lr=1e-3),
            TrainBlock(epochs=5000, lr=1e-3),
        ]
    )

    print_every: int = 500
    record_every: int = 50
    checkpoint_every: int = 1000

    # Residual-based adaptive refinement (RAR)
    rar_every: int = 2000
    rar_candidate_points: int = 20000
    rar_topk: int = 5000
    rar_start_epoch: int = 4000

    # Residual-driven phi update
    phi_lr: Optional[float] = None
    phi_lr_scale: float = 1e-3
    phi_update_every: int = 5000
    phi_update_start_epoch: int = 50000
    phi_update_dt: float = 1e-3
    phi_update_inner_steps: int = 10
    phi_update_stop_tol: float = 1e-6
    # One of: PDE, GRAD, DATA
    phi_update_residual_type: str = "PDE"
    phi_update_band_eps: float = 0.05
    phi_update_radius: float = 0.05
    phi_update_tau: float = 1
    phi_update_clip_q: float = 0.99


@dataclass
class PIMOE3DEvalConfig:
    n_grid: int = 51
    eval_batch_size: int = 4096


@dataclass
class PIMOE3DOutputConfig:
    output_dir: str = "outputs_add_pinns3d_c1_sphere"
    checkpoint_dir: str = "checkpoints"
    loss_csv_name: str = "loss_list_global.csv"
    loss_png_name: str = "Loss.png"
    field_npz_name: str = "final_fields.npz"
    u_slices_png_name: str = "u_residual_slices.png"
    phi_slices_png_name: str = "phi_compare_slices.png"
    snapshot_every: int = 1000
    slice_snapshot_every: int = 1000
    snapshot_dir: str = "snapshots"
    u_heatmap_dir: str = "u_heatmaps"
    f_heatmap_dir: str = "f_heatmaps"
    phi_heatmap_dir: str = "phi_heatmaps"
    viz_scatter_dir: str = "viz_scatter"
    # Flower-style scatter visualization config on a fixed x-slice.
    scatter_n: int = 200
    scatter_batch_size: int = 4096
    scatter_yz_bbox: tuple = (0.0, 1.0, 0.0, 1.0)
    # If None, use x = C1[0].
    scatter_x_fixed: Optional[float] = None

    def resolve_output_dir(self, project_root: Path) -> Path:
        out = project_root / self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        return out


@dataclass
class PIMOE3DRuntimeConfig:
    resume_checkpoint: str = ""
    force_phi_trainable: bool = False
    allow_phi_update_when_frozen: bool = False


# Generic aliases for base ADD-PINNs pipeline.
DataConfig = PIMOE3DDataConfig
ModelConfig = PIMOE3DModelConfig
TrainConfig = PIMOE3DTrainConfig
EvalConfig = PIMOE3DEvalConfig
OutputConfig = PIMOE3DOutputConfig
RuntimeConfig = PIMOE3DRuntimeConfig
