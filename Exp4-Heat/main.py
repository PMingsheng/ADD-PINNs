import os
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from config import DataConfig, ModelConfig, TrainConfig, VisualizationConfig, SAMPLING_CONFIGS
from data import load_uniform_grid_fit, plot_sampling_points, sample_xy_no_corners
from model import PartitionPINN
from plot_T_slice_with_phi import save_T_slice_with_phi_plot_from_fields
from train import TrainState, train_main, write_loss_history_csv
from utils import masked_partition_value, set_seed
from visualization import (
    plot_E_from_lossfile,
    plot_loss_components_from_records,
    plot_phi_compare_with_cross_and_circle,
    plot_phi_heatmap,
)


def save_data_output(model, state, *, train_cfg, viz_cfg, output_root: Path) -> None:
    data_output = output_root / "data_output"
    data_output.mkdir(parents=True, exist_ok=True)

    loss_csv = Path(viz_cfg.loss_csv_file)
    if loss_csv.exists():
        shutil.copy2(loss_csv, data_output / loss_csv.name)

    n_phi = train_cfg.phi_snapshot_n
    bbox = train_cfg.phi_snapshot_bbox
    device = next(model.parameters()).device

    xg = torch.linspace(bbox[0], bbox[1], n_phi, device=device)
    yg = torch.linspace(bbox[2], bbox[3], n_phi, device=device)
    Xg, Yg = torch.meshgrid(xg, yg, indexing="ij")
    xy_grid = torch.stack([Xg, Yg], dim=-1).reshape(-1, 2)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        phi, T1, T2 = model(xy_grid)
        T = masked_partition_value(phi, T1, T2)

    if was_training:
        model.train()

    f1, f2 = state.get_f1_f2()
    np.savez_compressed(
        data_output / "final_fields.npz",
        x=xg.detach().cpu().numpy(),
        y=yg.detach().cpu().numpy(),
        phi=phi.detach().cpu().numpy().reshape(n_phi, n_phi),
        T1=T1.detach().cpu().numpy().reshape(n_phi, n_phi),
        T2=T2.detach().cpu().numpy().reshape(n_phi, n_phi),
        T=T.detach().cpu().numpy().reshape(n_phi, n_phi),
        bbox=np.asarray(bbox, dtype=np.float64),
        f1=float(f1.detach().cpu()),
        f2=float(f2.detach().cpu()),
    )

    save_T_slice_with_phi_plot_from_fields(
        x_axis=xg.detach().cpu().numpy(),
        y_axis=yg.detach().cpu().numpy(),
        T_pred_map=T.detach().cpu().numpy().reshape(n_phi, n_phi),
        phi_map=phi.detach().cpu().numpy().reshape(n_phi, n_phi),
        txt_filename=str(Path(__file__).resolve().parent / DataConfig().ttxt_filename),
        save_path=data_output / "T_slice_with_phi_final.png",
        save_npz_path=data_output / "T_slice_with_phi_final.npz",
        epoch=int(state.epoch_offset_global),
        bbox=bbox,
        title_prefix="ADD-PINNs",
        percentile=viz_cfg.slice_percentile,
        max_points=viz_cfg.slice_points,
        eps_eik=train_cfg.eps_eik,
    )


def _configure_output_dirs(
    project_root: Path,
    sampling_tag: str,
    viz_cfg: VisualizationConfig,
) -> Tuple[Path, Path]:
    output_root = project_root / f"output_{sampling_tag}"
    output_root.mkdir(parents=True, exist_ok=True)
    viz_dir = output_root / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    viz_cfg.loss_csv_file = str(output_root / viz_cfg.loss_csv_file)
    viz_cfg.phi_snapshots_dir = str(output_root / viz_cfg.phi_snapshots_dir)
    viz_cfg.phi_heatmap_dir = str(output_root / viz_cfg.phi_heatmap_dir)
    viz_cfg.slice_snapshots_dir = str(output_root / viz_cfg.slice_snapshots_dir)
    viz_cfg.fig12_snapshots_dir = str(output_root / viz_cfg.fig12_snapshots_dir)
    viz_cfg.viz_scatter_dir = str(output_root / viz_cfg.viz_scatter_dir)
    viz_cfg.f_plot_save = str(viz_dir / Path(viz_cfg.f_plot_save).name)
    viz_cfg.loss_plot_save = str(viz_dir / Path(viz_cfg.loss_plot_save).name)
    viz_cfg.phi_heatmap_save = str(viz_dir / Path(viz_cfg.phi_heatmap_save).name)
    viz_cfg.phi_compare_save = str(viz_dir / Path(viz_cfg.phi_compare_save).name)
    return output_root, viz_dir


def main() -> None:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    viz_cfg = VisualizationConfig()

    sampling_mode = os.environ.get("PIMOE_SAMPLING_MODE", "roi-off").strip().lower()
    if sampling_mode not in SAMPLING_CONFIGS:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}. Choose from {tuple(SAMPLING_CONFIGS.keys())}.")
    sampling_cfg = SAMPLING_CONFIGS[sampling_mode]
    sampling_tag = sampling_mode.replace("-", "_")
    output_root, viz_dir = _configure_output_dirs(project_root, sampling_tag, viz_cfg)

    set_seed(train_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = (project_root / sampling_cfg.get("ttxt_filename", data_cfg.ttxt_filename)).resolve()
    xy_fit, T_fit = load_uniform_grid_fit(
        nx=sampling_cfg["nx"],
        ny=sampling_cfg["ny"],
        ttxt_filename=str(data_path),
        device=device,
        circles=list(data_cfg.circles),
        dense_factor=sampling_cfg["dense_factor"],
        drop_boundary=sampling_cfg["drop_boundary"],
        xlim=sampling_cfg["xlim"],
        ylim=sampling_cfg["ylim"],
        tol=sampling_cfg["tol"],
        target_total=sampling_cfg.get("target_total"),
    )

    plot_sampling_points(
        xy_fit,
        circles=data_cfg.circles,
        title="Uniform grid + dense defect",
        savepath=str(viz_dir / "sampling_points.png"),
        show=False,
    )

    model = PartitionPINN(width=model_cfg.width, depth=model_cfg.depth).to(device)
    f1_raw = nn.Parameter(torch.tensor(1.0, device=device))
    f2_raw = nn.Parameter(torch.tensor(1.0, device=device))
    state = TrainState(f1_raw=f1_raw, f2_raw=f2_raw)
    state.xy_int_const = sample_xy_no_corners(
        train_cfg.interior_points,
        device=device,
        corner_tol=train_cfg.corner_tol,
        batch_size=train_cfg.sample_batch_size,
    )

    opt = None
    opt_phi = None
    for block in train_cfg.blocks:
        model, opt, opt_phi, xy_train = train_main(
            model,
            state,
            block.epochs,
            xy_fit=xy_fit,
            T_fit=T_fit,
            opt=opt,
            opt_phi=opt_phi,
            lr=block.lr,
            lam_weights=train_cfg.lam_weights,
            train_cfg=train_cfg,
            viz_cfg=viz_cfg,
            fallback_circles=data_cfg.circles,
        )
        state.xy_int_const = xy_train

    write_loss_history_csv(state.loss_list_global_item, viz_cfg.loss_csv_file)

    plot_loss_components_from_records(
        state.loss_list_global_item,
        savepath=viz_cfg.loss_plot_save,
        show=False,
    )

    plot_E_from_lossfile(
        viz_cfg.loss_csv_file,
        names=("$f_1$", "$f_2$"),
        true_vals=(10, 5),
        smooth=viz_cfg.f_plot_smooth,
        savepath=viz_cfg.f_plot_save,
        show=False,
        add_pred_markers=True,
        marker_every="auto",
        pred_markers=("o", "^"),
        markersize=4.0,
    )

    plot_phi_compare_with_cross_and_circle(
        model,
        cross=viz_cfg.phi_compare_cross,
        circle_roi=viz_cfg.phi_compare_circle,
        bbox=viz_cfg.phi_compare_bbox,
        n=viz_cfg.phi_compare_n,
        band_eps=viz_cfg.phi_compare_band_eps,
        xy_label=xy_fit,
        savepath=viz_cfg.phi_compare_save,
        dpi=viz_cfg.phi_compare_dpi,
        show=False,
    )

    plot_phi_heatmap(
        model,
        bbox=viz_cfg.phi_heatmap_bbox,
        n=viz_cfg.phi_heatmap_n,
        savepath=viz_cfg.phi_heatmap_save,
        dpi=viz_cfg.phi_heatmap_dpi,
        show=False,
    )

    save_data_output(
        model,
        state,
        train_cfg=train_cfg,
        viz_cfg=viz_cfg,
        output_root=output_root,
    )


if __name__ == "__main__":
    main()
