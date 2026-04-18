"""
Standalone sampling-point visualization.
"""
from pathlib import Path

import config
from data import (
    load_ellipse_uv_eps_fit,
    load_ellipse_uv_eps_fit_downsample,
    load_ellipse_uv_eps_fit_rect_dense,
)
from visualization import plot_sampling_points_ellipse_downsample

BASE_DIR = Path(__file__).resolve().parent

SAMPLING_MODE = "roi-rect"  # "roi-on", "roi-rect", or "roi-off"
SAVE_FIG = True
SHOW_FIG = True


def _get_save_path(sampling_mode: str):
    if not SAVE_FIG:
        return None
    sampling_tag = sampling_mode.replace("-", "_")
    viz_dir = BASE_DIR / f"output_{sampling_tag}" / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    return str(viz_dir / "sampling_points.png")


def main() -> None:
    sampling_mode = SAMPLING_MODE
    if sampling_mode == "roi-on":
        cfg = config.ROI_RECT_CONFIG
        (
            _xy_u,
            _U_fit,
            _xy_eps,
            _E_fit,
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
        title = "Axis-Aligned Dense Box Sampling"
        ellipse = config.ELLIPSE_PARAMS
    elif sampling_mode == "roi-rect":
        cfg = config.ROI_RECT_CONFIG
        (
            _xy_u,
            _U_fit,
            _xy_eps,
            _E_fit,
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
        title = "Axis-Aligned Dense Box Sampling"
        ellipse = config.ELLIPSE_PARAMS
    else:
        cfg = config.ROI_OFF_CONFIG
        (
            _xy_u,
            _U_fit,
            _xy_eps,
            _E_fit,
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
        title = "Rectangle Grid Sampling"
        ellipse = config.ELLIPSE_PARAMS

    plot_sampling_points_ellipse_downsample(
        xy_basic_plot,
        xy_dense_extra_plot,
        ellipse=ellipse,
        rect_info=rect_info,
        title=title,
        savepath=_get_save_path(sampling_mode),
        show=SHOW_FIG,
    )


if __name__ == "__main__":
    main()
