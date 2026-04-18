from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    ttxt_filename: str = "Possion.txt"
    nx: int = 8
    ny: int = 8
    circles: Tuple[Tuple[float, float, float], ...] = ((0.40, 0.60, 0.2),)
    dense_factor: float = 0.335
    drop_boundary: bool = True
    xlim: Tuple[float, float] = (0.0, 1.0)
    ylim: Tuple[float, float] = (0.0, 1.0)
    tol: float = 0.02


SAMPLING_CONFIGS = {
    "roi-on": dict(
        nx=7,
        ny=7,
        ttxt_filename="Possion.txt",
        dense_factor=0.25,
        drop_boundary=False,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        tol=0.02,
        target_total=100,
    ),
    "roi-off": dict(
        nx=12,
        ny=12,
        ttxt_filename="Possion.txt",
        dense_factor=1.0,
        drop_boundary=False,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        tol=0.02,
    ),
    "full-data": dict(
        nx=20,
        ny=20,
        ttxt_filename="Possion.txt",
        dense_factor=0.2,
        drop_boundary=False,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        tol=0.02,
    ),
}


@dataclass
class ModelConfig:
    width: int = 80
    depth: int = 6


@dataclass
class TrainBlock:
    epochs: int
    lr: float


@dataclass
class TrainConfig:
    seed: int = 1234
    interior_points: int = 8000
    corner_tol: float = 0.0
    sample_batch_size: int = 10000
    eps_eik: float = 0.01
    lam_weights: dict = field(
        default_factory=lambda: {
            "data": 1e3,
            "pde": 1e-2,
            "bc": 0.0,
            "interface": 1e-2,
            "eik": 1e-4,
            "area": 1e-4,
        }
    )
    blocks: List[TrainBlock] = field(
        default_factory=lambda: [
            TrainBlock(epochs=20000, lr=1e-3),
            TrainBlock(epochs=60000, lr=1e-4),
        ]
    )
    rar_every: int = 5000
    rar_n_cand: int = 4096
    rar_n_new: int = 128
    rar_band_eps: float = 0.02
    rar_corner_tol: float = 0.0
    rar_batch_size: int = 8192
    print_every: int = 500
    record_every: int = 50
    viz_every: int = 5000
    phi_snapshot_n: int = 200
    phi_snapshot_bbox: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)


@dataclass
class VisualizationConfig:
    loss_hist_file: str = "loss_hist_global.txt"
    loss_csv_file: str = "loss_list_global.csv"
    phi_snapshots_dir: str = "phi_snapshots"
    phi_heatmap_dir: str = "phi_heatmaps"
    slice_snapshots_dir: str = "T_slice_with_phi"
    slice_snapshot_every: int = 5000
    fig12_snapshots_dir: str = "Fig12_snapshots"
    phi_heatmap_every: int = 5000
    viz_scatter_dir: str = "viz_scatter"
    slice_percentile: float = 99.5
    slice_points: int = 401
    scatter_n: int = 200
    scatter_bbox: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    scatter_batch_size: int = 4096
    cross_params: Tuple[float, float, float, float] = (0.40, 0.60, 0.05, 0.15)
    f_plot_smooth: int = 5
    f_plot_save: str = "f.png"
    loss_plot_save: str = "Loss.png"
    phi_heatmap_save: str = "phi_heatmap.png"
    phi_heatmap_bbox: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    phi_heatmap_n: int = 250
    phi_heatmap_dpi: int = 300
    phi_compare_cross: Tuple[float, float, float, float, float, float] = (
        0.4,
        0.6,
        0.3,
        0.3,
        0.1,
        0.1,
    )
    phi_compare_circle: Tuple[float, float, float] = (0.4, 0.6, 0.2)
    phi_compare_bbox: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    phi_compare_n: int = 250
    phi_compare_band_eps: float = 0.02
    phi_compare_dpi: int = 400
    phi_compare_save: str = "phi_cross_circle.png"
