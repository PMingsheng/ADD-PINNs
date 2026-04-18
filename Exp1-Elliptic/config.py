from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Global static bandwidths for smoothed Heaviside/Dirac.
HEAVISIDE_BANDWIDTH: float = 0.05
DIRAC_BANDWIDTH: float = 0.05


@dataclass
class DataConfig:
    # If use_synthetic is True, data will be generated from the analytic flower problem.
    use_synthetic: bool = True
    synthetic_n_side: int = 201
    ttxt_filename: str = "Possion.txt"

    nx: int = 8
    ny: int = 8
    circles: Tuple[Tuple[float, float, float], ...] = ((0.0, 0.0, 0.55),)
    # Ring ROI densification: (cx, cy, r_inner, r_outer)
    annuli: Tuple[Tuple[float, float, float, float], ...] = ((0.0, 0.0, 0.25, 0.55),)
    dense_factor: float = 0.335
    drop_boundary: bool = True
    xlim: Tuple[float, float] = (-1.0, 1.0)
    ylim: Tuple[float, float] = (-1.0, 1.0)
    tol: float = 0.02


SAMPLING_CONFIGS = {
    "roi-on": dict(
        nx=7,
        ny=7,
        dense_factor=0.25,
        drop_boundary=True,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        tol=0.02,
        target_total=100,
    ),
    "roi-off": dict(
        nx=20,
        ny=20,
        dense_factor=1.0,
        drop_boundary=True,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        tol=0.02,
    ),
    "full-data": dict(
        nx=20,
        ny=20,
        dense_factor=0.2,
        drop_boundary=True,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        tol=0.02,
    ),
}


@dataclass
class ModelConfig:
    width: int = 50
    depth: int = 4


@dataclass
class TrainBlock:
    epochs: int
    lr: float
    # If None, phi optimizer uses lr.
    phi_lr: Optional[float] = None


@dataclass
class TrainConfig:
    seed: int = 1234
    interior_points: int = 4000
    corner_tol: float = 0.02
    sample_batch_size: int = 10000
    xy_int_xlim: Tuple[float, float] = (-1.0, 1.0)
    xy_int_ylim: Tuple[float, float] = (-1.0, 1.0)

    boundary_points_per_edge: int = 300

    lam_weights: dict = field(
        default_factory=lambda: {
            "data": 1,
            "pde": 1,
            "bc": 1,
            "interface": 0,
            "eik": 1e-4,
            "area": 1e-4,
            "perimeter": 1e-4,
        }
    )
    # Exponential decay for eik weight:
    # lam_eik(t) = lam_eik0 * exp(-k*t), with lam_eik(decay_steps) = decay_ratio * lam_eik0.
    eik_decay_enabled: bool = False
    eik_decay_steps: int = 5000
    eik_decay_ratio: float = 0.1
    blocks: List[TrainBlock] = field(
        default_factory=lambda: [
            TrainBlock(epochs=5000, lr=1e-3, phi_lr=1e-3),
            TrainBlock(epochs=25000, lr=1e-3, phi_lr=1e-3),
        ]
    )
    rar_every: int = 5000
    rar_n_cand: int = 4096
    rar_n_new: int = 128
    rar_band_eps: float = 0.02
    rar_corner_tol: float = 0.05
    rar_batch_size: int = 8192
    print_every: int = 500
    record_every: int = 50
    viz_every: int = 5000
    phi_snapshot_n: int = 200
    phi_snapshot_bbox: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)


@dataclass
class VisualizationConfig:
    loss_hist_file: str = "loss_hist_global.txt"
    loss_csv_file: str = "loss_list_global.csv"
    phi_snapshots_dir: str = "phi_snapshots"
    phi_heatmap_dir: str = "phi_heatmaps"
    phi_heatmap_every: int = 5000
    viz_scatter_dir: str = "viz_scatter"
    scatter_n: int = 200
    scatter_bbox: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
    scatter_batch_size: int = 4096
    f_plot_smooth: int = 5
    f_plot_save: str = "f.png"
    loss_plot_save: str = "Loss.png"
    phi_heatmap_save: str = "phi_heatmap.png"
    phi_heatmap_bbox: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
    phi_heatmap_n: int = 250
    phi_heatmap_dpi: int = 300
    u_heatmap_dir: str = "u_heatmaps"
    u_heatmap_every: int = 5000
    u_heatmap_bbox: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
    u_heatmap_n: int = 250
    u_heatmap_dpi: int = 300
    f_heatmap_dir: str = "f_heatmaps"
    f_heatmap_every: int = 5000
    f_heatmap_bbox: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
    f_heatmap_n: int = 250
    f_heatmap_dpi: int = 300
    uf_slice_with_phi_dir: str = "uf_slice_with_phi"
    uf_slice_with_phi_every: int = 5000
    uf_slice_line_axis: str = "x"
    uf_slice_line_mode: str = "value"
    uf_slice_line_value: float = 0.0
    uf_slice_line_index: int = 0

    phi_compare_bbox: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
    phi_compare_n: int = 250
    phi_compare_band_eps: float = 0.02
    phi_compare_dpi: int = 400
    phi_compare_save: str = "phi_flower_compare.png"
    # Keep these for backward compatibility with copied figure scripts.
    phi_compare_cross: Tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        1.0,
        1.0,
        0.2,
        0.2,
    )
    phi_compare_circle: Tuple[float, float, float] = (0.0, 0.0, 0.55)
