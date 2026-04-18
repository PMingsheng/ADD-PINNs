import os
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from config import DataConfig, ModelConfig, TrainConfig, VisualizationConfig, SAMPLING_CONFIGS
from data import (
    load_uniform_grid_fit,
    plot_sampling_points,
    sample_boundary_points,
    sample_xy_no_corners,
)
from model import PartitionPINN
from train import TrainState, train_main, write_loss_history_csv
from utils import set_seed
from visualization import (
    plot_f_true_pred_residual_heatmap,
    plot_loss_components_from_records,
    plot_phi_compare_with_flower,
    plot_phi_heatmap,
    plot_u_true_pred_residual_heatmap,
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
        phi, u1, u2 = model(xy_grid)
        mask_pos = (phi >= 0).to(phi.dtype)
        mask_neg = 1.0 - mask_pos
        u = mask_pos * u1 + mask_neg * u2

    if was_training:
        model.train()

    u_maps = plot_u_true_pred_residual_heatmap(
        model,
        bbox=bbox,
        n=n_phi,
        savepath=None,
        show=False,
    )
    f_maps = plot_f_true_pred_residual_heatmap(
        model,
        bbox=bbox,
        n=n_phi,
        savepath=None,
        show=False,
    )

    np.savez_compressed(
        data_output / "final_fields.npz",
        x=xg.detach().cpu().numpy(),
        y=yg.detach().cpu().numpy(),
        phi=phi.detach().cpu().numpy().reshape(n_phi, n_phi),
        u1=u1.detach().cpu().numpy().reshape(n_phi, n_phi),
        u2=u2.detach().cpu().numpy().reshape(n_phi, n_phi),
        u=u.detach().cpu().numpy().reshape(n_phi, n_phi),
        u_true=u_maps["u_true_map"].astype(np.float32),
        u_pred=u_maps["u_pred_map"].astype(np.float32),
        u_residual=u_maps["u_res_map"].astype(np.float32),
        f_true=f_maps["f_true_map"].astype(np.float32),
        f_pred=f_maps["f_pred_map"].astype(np.float32),
        f_residual=f_maps["f_res_map"].astype(np.float32),
        bbox=np.asarray(bbox, dtype=np.float64),
    )


def _configure_output_dirs(
    project_root: Path,
    sampling_tag: str,
    viz_cfg: VisualizationConfig,
) -> Tuple[Path, Path]:
    output_root = project_root / "outputs_flower" / sampling_tag
    output_root.mkdir(parents=True, exist_ok=True)
    viz_dir = output_root / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    viz_cfg.loss_csv_file = str(output_root / viz_cfg.loss_csv_file)
    viz_cfg.phi_snapshots_dir = str(output_root / viz_cfg.phi_snapshots_dir)
    viz_cfg.phi_heatmap_dir = str(output_root / viz_cfg.phi_heatmap_dir)
    viz_cfg.u_heatmap_dir = str(output_root / viz_cfg.u_heatmap_dir)
    viz_cfg.f_heatmap_dir = str(output_root / viz_cfg.f_heatmap_dir)
    viz_cfg.uf_slice_with_phi_dir = str(output_root / viz_cfg.uf_slice_with_phi_dir)
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

    sampling_mode = "roi-off"  # "roi-on", "roi-off", or "full-data"
    sampling_cfg = SAMPLING_CONFIGS.get(sampling_mode, SAMPLING_CONFIGS["roi-on"])
    sampling_tag = sampling_mode.replace("-", "_")
    output_root, viz_dir = _configure_output_dirs(project_root, sampling_tag, viz_cfg)
    print(f"[Output] writing results to: {output_root}")

    set_seed(train_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = (project_root / data_cfg.ttxt_filename).resolve()
    xy_fit, u_fit = load_uniform_grid_fit(
        nx=sampling_cfg["nx"],
        ny=sampling_cfg["ny"],
        use_synthetic=data_cfg.use_synthetic,
        synthetic_n_side=data_cfg.synthetic_n_side,
        ttxt_filename=str(data_path),
        device=device,
        circles=list(data_cfg.circles),
        annuli=list(data_cfg.annuli),
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
        annuli=data_cfg.annuli,
        show_circles=False,
        show_annuli=True,
        show_flower_interface=True,
        title="Uniform grid + ring ROI densification (circular interface)",
        savepath=str(viz_dir / "sampling_points.png"),
        show=False,
    )

    model = PartitionPINN(width=model_cfg.width, depth=model_cfg.depth).to(device)
    state = TrainState()
    state.xy_int_const = sample_xy_no_corners(
        4000,
        device=device,
        corner_tol=train_cfg.corner_tol,
        batch_size=train_cfg.sample_batch_size,
        xlim=train_cfg.xy_int_xlim,
        ylim=train_cfg.xy_int_ylim,
    )
    state.xy_bnd_const = sample_boundary_points(
        train_cfg.boundary_points_per_edge,
        device=device,
        xlim=train_cfg.xy_int_xlim,
        ylim=train_cfg.xy_int_ylim,
    )

    opt = None
    opt_phi = None
    for block in train_cfg.blocks:
        model, opt, opt_phi, xy_train = train_main(
            model,
            state,
            block.epochs,
            xy_fit=xy_fit,
            u_fit=u_fit,
            opt=opt,
            opt_phi=opt_phi,
            lr=block.lr,
            phi_lr=block.phi_lr,
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

    plot_phi_compare_with_flower(
        model,
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
