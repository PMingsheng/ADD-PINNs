from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import config


@dataclass
class PINNDataConfig:
    txt_filename: str = config.DataConfig().ttxt_filename
    sampling_mode: str = "roi-off"
    circle: Tuple[float, float, float] = tuple(float(v) for v in config.DataConfig().circles[0])
    sampling_configs: Dict[str, Dict[str, object]] = field(
        default_factory=lambda: {key: dict(val) for key, val in config.SAMPLING_CONFIGS.items()}
    )


@dataclass
class PINNModelConfig:
    width: int = 139
    depth: int = 6
    f1_init: float = 1.0
    f2_init: float = 1.0
    learn_f1: bool = True
    learn_f2: bool = True
    f_scale: float = 10.0


@dataclass
class PINNTrainConfig:
    seed: int = config.TrainConfig().seed
    lr: float = 1e-3
    epochs: int = 80000
    interior_points: int = config.TrainConfig().interior_points
    rar_every: int = 5000
    rar_n_cand: int = 4096
    rar_n_new: int = 256
    lam_data: float = 1e3
    lam_pde: float = 1e-2
    lam_interface: float = 0.0
    interface_points: int = 512
    print_every: int = 500
    record_every: int = 50


@dataclass
class PINNEvalConfig:
    bbox: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    n_grid: int = 201
    eval_batch_size: int = 4096


@dataclass
class PINNOutputConfig:
    output_dir: str = "outputs_pinn_single"
    loss_csv_name: str = "loss.csv"
    loss_png_name: str = "loss.png"
    field_npz_name: str = "final_fields.npz"
    field_png_name: str = "final_fields.png"
    snapshot_every: int = 5000
    snapshot_dir: str = "snapshots"

    def resolve_output_dir(self, project_root: Path) -> Path:
        out = project_root / self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        return out
