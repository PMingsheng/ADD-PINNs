from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import config


@dataclass
class PINNDataConfig:
    txt_filename: str = "Ellipse.txt"
    sampling_mode: str = "roi-rect"
    ellipse: Tuple[float, float, float, float, float] = tuple(float(v) for v in config.ELLIPSE_PARAMS)
    data_config: Dict[str, object] = field(default_factory=lambda: dict(config.DATA_CONFIG))
    roi_off_config: Dict[str, object] = field(default_factory=lambda: dict(config.ROI_OFF_CONFIG))
    roi_rect_config: Dict[str, object] = field(default_factory=lambda: dict(config.ROI_RECT_CONFIG))


@dataclass
class PINNModelConfig:
    width: int = 87
    depth: int = 4
    E_out_init: float = 1.0
    E_in_init: float = 1.0
    learn_E_out: bool = True
    learn_E_in: bool = False
    E_scale: float = config.E_SCALE


@dataclass
class PINNTrainConfig:
    seed: int = config.SEED
    lr: float = 1e-3
    epochs: int = 60000
    interior_points: int = 8000
    xy_int_xlim: Tuple[float, float] = (-1.0, 1.0)
    xy_int_ylim: Tuple[float, float] = (-1.0, 1.0)
    rar_every: int = 5000
    rar_n_cand: int = 4096
    rar_n_new: int = 256
    interface_points: int = 512
    lam_data_u: float = 1e5
    lam_data_eps: float = 1e5
    lam_pde: float = 1e3
    lam_interface: float = 1e2
    nu: float = config.NU
    eps0: float = config.EPS0
    print_every: int = 500
    record_every: int = 50


@dataclass
class PINNEvalConfig:
    bbox: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
    n_grid: int = 201
    eval_batch_size: int = 4096


@dataclass
class PINNOutputConfig:
    output_dir: str = "outputs_pinn_single"
    loss_csv_name: str = "loss.csv"
    loss_png_name: str = "loss.png"
    field_npz_name: str = "final_fields.npz"
    snapshot_every: int = 5000
    snapshot_dir: str = "snapshots"

    def resolve_output_dir(self, project_root: Path) -> Path:
        out = project_root / self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        return out
