from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class APINNDataConfig:
    use_synthetic: bool = True
    synthetic_n_side: int = 201
    ttxt_filename: str = "Possion.txt"

    nx: int = 20
    ny: int = 20
    circles: Tuple[Tuple[float, float, float], ...] = ((0.0, 0.0, 0.55),)
    annuli: Tuple[Tuple[float, float, float, float], ...] = ()
    dense_factor: float = 1.0
    drop_boundary: bool = True
    xlim: Tuple[float, float] = (-1.0, 1.0)
    ylim: Tuple[float, float] = (-1.0, 1.0)
    tol: float = 0.02


@dataclass
class APINNModelConfig:
    n_experts: int = 2

    shared_width: int = 32
    shared_depth: int = 2
    shared_dim: int = 16

    expert_width: int = 55
    expert_depth: int = 4

    gate_width: int = 32
    gate_hidden_layers: int = 2
    gate_tau: float = 0.05


@dataclass
class APINNTrainConfig:
    seed: int = 1234
    lr: float = 1e-3
    epochs: int = 30000

    interior_points: int = 80
    corner_tol: float = 0.02
    sample_batch_size: int = 8000
    xy_int_xlim: Tuple[float, float] = (-1.0, 1.0)
    xy_int_ylim: Tuple[float, float] = (-1.0, 1.0)

    rar_every: int = 5000
    rar_n_cand: int = 4096
    rar_n_new: int = 128
    rar_corner_tol: float = 0.05
    rar_batch_size: int = 8192

    lam_data: float = 1e2
    lam_pde: float = 1.0

    gate_pretrain_epochs: int = 1000
    gate_pretrain_lr: float = 5e-4
    gate_pretrain_use_hard_target: bool = True

    print_every: int = 500
    record_every: int = 50


@dataclass
class APINNEvalConfig:
    bbox: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
    n_grid: int = 200
    eval_batch_size: int = 4096


@dataclass
class APINNOutputConfig:
    output_dir: str = "outputs_apinn"
    loss_csv_name: str = "loss.csv"
    loss_png_name: str = "loss.png"
    field_npz_name: str = "final_fields.npz"
    snapshot_every: int = 5000
    snapshot_dir: str = "snapshots"

    def resolve_output_dir(self, project_root: Path) -> Path:
        out = project_root / self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        return out
