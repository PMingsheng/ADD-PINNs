import os
import shutil
from pathlib import Path

import config
import numpy as np
import torch
from data import (
    load_ellipse_uv_eps_fit,
    load_ellipse_uv_eps_fit_downsample,
    load_ellipse_uv_eps_fit_rect_dense,
    sample_xy_uniform,
)
from model import PartitionPINN
from plot_u_slice_with_phi import save_u_slice_with_phi_plot_from_fields
from train import train_main
from utils import set_seed
from visualization import plot_sampling_points_ellipse_downsample, plot_residual_scatter, plot_phi_compare


def save_data_output(model, output_root: Path) -> None:
    data_output = output_root / "data_output"
    data_output.mkdir(parents=True, exist_ok=True)

    loss_csv = config.LOSS_CSV_PATH
    if loss_csv.exists():
        shutil.copy2(loss_csv, data_output / loss_csv.name)

    n_phi = config.TRAIN_CONFIG["phi_snapshot_n"]
    bbox = config.TRAIN_CONFIG["phi_snapshot_bbox"]
    device = next(model.parameters()).device

    xg = torch.linspace(bbox[0], bbox[1], n_phi, device=device)
    yg = torch.linspace(bbox[2], bbox[3], n_phi, device=device)
    Xg, Yg = torch.meshgrid(xg, yg, indexing="ij")
    xy_grid = torch.stack([Xg, Yg], dim=-1).reshape(-1, 2)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        phi, ux1, uy1, ux2, uy2 = model(xy_grid)
        w1 = torch.relu(phi)
        w2 = torch.relu(-phi)
        denom = w1 + w2 + 1e-12
        ux = (w1 * ux1 + w2 * ux2) / denom
        uy = (w1 * uy1 + w2 * uy2) / denom

    if was_training:
        model.train()

    if hasattr(model, "get_E_scaled"):
        E_1_scaled, E_2_scaled = model.get_E_scaled()
    else:
        E_1_scaled, E_2_scaled = model.E_1, model.E_2

    np.savez_compressed(
        data_output / "final_fields.npz",
        x=xg.detach().cpu().numpy(),
        y=yg.detach().cpu().numpy(),
        phi=phi.detach().cpu().numpy().reshape(n_phi, n_phi),
        ux1=ux1.detach().cpu().numpy().reshape(n_phi, n_phi),
        uy1=uy1.detach().cpu().numpy().reshape(n_phi, n_phi),
        ux2=ux2.detach().cpu().numpy().reshape(n_phi, n_phi),
        uy2=uy2.detach().cpu().numpy().reshape(n_phi, n_phi),
        ux=ux.detach().cpu().numpy().reshape(n_phi, n_phi),
        uy=uy.detach().cpu().numpy().reshape(n_phi, n_phi),
        bbox=np.asarray(bbox, dtype=np.float64),
        E1=float(E_1_scaled.detach().cpu()),
        E2=float(E_2_scaled.detach().cpu()),
    )

    save_u_slice_with_phi_plot_from_fields(
        x_axis=xg.detach().cpu().numpy(),
        y_axis=yg.detach().cpu().numpy(),
        ux_pred_map=ux.detach().cpu().numpy().reshape(n_phi, n_phi),
        uy_pred_map=uy.detach().cpu().numpy().reshape(n_phi, n_phi),
        phi_map=phi.detach().cpu().numpy().reshape(n_phi, n_phi),
        ellipse=config.ELLIPSE_PARAMS,
        txt_filename=config.DATA_FILE,
        save_path=data_output / "u_slice_with_phi_final.png",
        save_npz_path=data_output / "u_slice_with_phi_final.npz",
        epoch=-1,
        bbox=bbox,
        title_prefix="ADD-PINNs",
    )

def _configure_output_dirs(sampling_tag: str) -> Path:
    output_root = config.PROJECT_ROOT / f"output_{sampling_tag}"
    output_root.mkdir(parents=True, exist_ok=True)
    config.PHI_SNAPSHOT_DIR = output_root / "phi_snapshots"
    config.VIZ_SCATTER_DIR = output_root / "viz_scatter"
    config.DISP_STRAIN_SNAPSHOT_DIR = output_root / "disp_strain_snapshots"
    config.SLICE_SNAPSHOT_DIR = output_root / "u_slice_with_phi"
    config.LOSS_CSV_PATH = output_root / "loss_list_global.csv"
    config.E_HISTORY_PATH = output_root / "E_history.txt"
    return output_root


def main():
    set_seed(config.SEED)
    sampling_mode = "roi-off"  # "roi-on", "roi-rect", or "roi-off"
    sampling_tag = sampling_mode.replace("-", "_")
    output_root = _configure_output_dirs(sampling_tag)

    if sampling_mode == "roi-on":
        cfg = config.ROI_RECT_CONFIG
        (
            xy_u,
            U_fit,
            xy_eps,
            E_fit,
            rect_info,
            xy_basic_plot,
            xy_dense_extra_plot,
        ) = load_ellipse_uv_eps_fit_rect_dense(
            nx=cfg["nx"],
            ny=cfg["ny"],
            txt_filename=cfg["txt_filename"],
            device=config.DEVICE,
            ellipse=config.ELLIPSE_PARAMS,
            tau_for_strain=cfg["tau_for_strain"],
            dense_factor=cfg["dense_factor"],
            rect_corners=cfg["rect_corners"],
            target_total=cfg["target_total"],
            random_state=cfg["random_state"],
        )
    elif sampling_mode == "roi-rect":
        cfg = config.ROI_RECT_CONFIG
        (
            xy_u,
            U_fit,
            xy_eps,
            E_fit,
            rect_info,
            xy_basic_plot,
            xy_dense_extra_plot,
        ) = load_ellipse_uv_eps_fit_rect_dense(
            nx=cfg["nx"],
            ny=cfg["ny"],
            txt_filename=cfg["txt_filename"],
            device=config.DEVICE,
            ellipse=config.ELLIPSE_PARAMS,
            tau_for_strain=cfg["tau_for_strain"],
            dense_factor=cfg["dense_factor"],
            rect_corners=cfg["rect_corners"],
            target_total=cfg["target_total"],
            random_state=cfg["random_state"],
        )
    else:
        cfg = config.ROI_OFF_CONFIG
        (
            xy_u,
            U_fit,
            xy_eps,
            E_fit,
            rect_info,
            xy_basic_plot,
            xy_dense_extra_plot,
        ) = load_ellipse_uv_eps_fit(
            nx=cfg["nx"],
            ny=cfg["ny"],
            txt_filename=cfg["txt_filename"],
            device=config.DEVICE,
            ellipse=config.ELLIPSE_PARAMS,
            tau_for_strain=cfg["tau_for_strain"],
            use_dense=cfg["use_dense"],
            dense_factor=cfg["dense_factor"],
            rect_corners=cfg["rect_corners"],
            xlim=cfg.get("xlim"),
            ylim=cfg.get("ylim"),
        )

    viz_dir = output_root / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[Sampling] mode={sampling_mode}  xy_u={len(xy_u)}  xy_eps={len(xy_eps)}  "
        f"base={len(xy_basic_plot)}  dense_extra={len(xy_dense_extra_plot)}"
    )

    if sampling_mode == "roi-on":
        title = "Axis-Aligned Dense Box Sampling"
    elif sampling_mode == "roi-rect":
        title = "Axis-Aligned Dense Box Sampling"
    else:
        title = "Rectangle Grid Sampling"
    plot_sampling_points_ellipse_downsample(
        xy_basic_plot,
        xy_dense_extra_plot,
        ellipse=config.DATA_CONFIG["ellipse"],
        rect_info=rect_info,
        title=title,
        savepath=str(viz_dir / "sampling_points.png"),
        show=False,
    )

    model = PartitionPINN(
        width=config.MODEL_CONFIG["width"],
        depth=config.MODEL_CONFIG["depth"],
        E1_init=config.MODEL_CONFIG["E1_init"],
        E2_init=config.MODEL_CONFIG["E2_init"],
        learn_E1=config.MODEL_CONFIG["learn_E1"],
        learn_E2=config.MODEL_CONFIG["learn_E2"],
    ).to(config.DEVICE)

    xy_int_const = sample_xy_uniform(
        config.TRAIN_CONFIG["xy_int_n"],
        device=config.DEVICE,
        xlim=config.TRAIN_CONFIG["xy_int_xlim"],
        ylim=config.TRAIN_CONFIG["xy_int_ylim"],
    )

    model, _, _, _ = train_main(
        model,
        xy_int_const,
        xy_u=xy_u,
        U_fit=U_fit,
        xy_eps=xy_eps,
        E_fit=E_fit,
        epochs=30000,
        lr=config.TRAIN_CONFIG["lr"],
        phi_lr=config.TRAIN_CONFIG["phi_lr"],
        lam_weights=config.LAM_WEIGHTS,
        nu=config.NU,
        eps0=config.EPS0,
    )

    model, _, _, _ = train_main(
        model,
        xy_int_const,
        xy_u=xy_u,
        U_fit=U_fit,
        xy_eps=xy_eps,
        E_fit=E_fit,
        epochs=30000,
        lr=1e-4,
        phi_lr=1e-4,
        lam_weights=config.LAM_WEIGHTS,
        nu=config.NU,
        eps0=config.EPS0,
    )
    plot_phi_compare(
        model,
        config.ELLIPSE_PARAMS,
        savepath=str(viz_dir / "phi_compare.png"),
        show=False,
    )

    plot_residual_scatter(
        model,
        kind="pde",
        nu=config.NU,
        xy_u=xy_u,
        U_fit=U_fit,
        n=200,
        bbox=(-1, 1, -1, 1),
        ellipse=config.ELLIPSE_PARAMS,
        cmap="cividis",
        use_log_norm=False,
        show_next=True,
        vel_type_for_next="pde",
        dt_next=1e-3,
        band_eps_vel=0.01,
        h_vel=0.05,
        tau_vel=1e-12,
        clip_q_vel=0.99,
        savepath=str(viz_dir / "scatter_final.png"),
        save_npz_path=str(viz_dir / "scatter_final.npz"),
        show=False,
    )

    save_data_output(model, output_root)


if __name__ == "__main__":
    main()
