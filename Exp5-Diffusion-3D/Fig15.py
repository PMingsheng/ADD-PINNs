#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps, cm, colors, rcParams


rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["mathtext.fontset"] = "stixsans"
rcParams["font.size"] = 8
rcParams["axes.labelsize"] = 8
rcParams["axes.titlesize"] = 10
rcParams["xtick.labelsize"] = 7
rcParams["ytick.labelsize"] = 7
rcParams["axes.linewidth"] = 0.6


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_ROOT.parent / "Figure" / "Fig15.png"
DATASETS = {
    "Truth": {
        "path": PROJECT_ROOT / "outputs_add_pinns3d_c1_sphere" / "data_output" / "final_fields.npz",
        "u_field": "u_true",
        "f_field": "f_true",
    },
    "PINNs": {
        "path": PROJECT_ROOT / "outputs_pinn3d_c1_sphere" / "data_output" / "final_fields.npz",
        "u_field": "u_pred",
        "f_field": "f_pred",
    },
    "APINNs": {
        "path": PROJECT_ROOT / "outputs_apinn3d_c1_sphere" / "data_output" / "final_fields.npz",
        "u_field": "u_pred",
        "f_field": "f_pred",
    },
    "ADD-PINNs": {
        "path": PROJECT_ROOT / "outputs_add_pinns3d_c1_sphere" / "data_output" / "final_fields.npz",
        "u_field": "u_pred",
        "f_field": "f_pred",
    },
}


def _load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.abs(np.asarray(values, dtype=np.float64) - float(target)).argmin())


def _shared_norm(fields: list[np.ndarray]) -> colors.Normalize:
    finite_blocks = [field[np.isfinite(field)] for field in fields]
    finite_blocks = [block for block in finite_blocks if block.size > 0]
    if not finite_blocks:
        return colors.Normalize(vmin=0.0, vmax=1.0)
    all_values = np.concatenate(finite_blocks)
    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _style_axes(
    ax: plt.Axes,
    elev: float,
    azim: float,
    x_value: float,
    y_value: float,
    z_value: float,
) -> None:
    ax.set_xlim(0.0, x_value)
    ax.set_ylim(y_value, 1.0)
    ax.set_zlim(0.0, z_value)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([0.0, x_value])
    ax.set_yticks([y_value, 1.0])
    ax.set_zticks([0.0, z_value])
    ax.set_box_aspect((0.9, 0.9, 0.9))
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.tick_params(pad=-1, length=2, width=0.5)
    ax.text2D(0.26, 0.065, "x", transform=ax.transAxes, fontsize=8, ha="center", va="center")
    ax.text2D(0.87, 0.15, "y", transform=ax.transAxes, fontsize=8, ha="center", va="center")
    ax.text2D(0.965, 0.54, "z", transform=ax.transAxes, fontsize=8, ha="left", va="center")


def _draw_visible_cube(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    field: np.ndarray,
    norm: colors.Normalize,
    cmap: colors.Colormap,
    x_value: float,
    y_value: float,
    z_value: float,
) -> None:
    x_idx = _nearest_index(x, x_value)
    y_idx = _nearest_index(y, y_value)
    z_idx = _nearest_index(z, z_value)

    x_face = x[: x_idx + 1]
    y_face = y[y_idx:]
    z_face = z[: z_idx + 1]

    yz_values = np.asarray(field[x_idx, y_idx:, : z_idx + 1], dtype=np.float64)
    xz_values = np.asarray(field[: x_idx + 1, y_idx, : z_idx + 1], dtype=np.float64)
    xy_values = np.asarray(field[: x_idx + 1, y_idx:, z_idx], dtype=np.float64)

    y_grid, z_grid = np.meshgrid(y_face, z_face, indexing="ij")
    x_grid, z_grid_2 = np.meshgrid(x_face, z_face, indexing="ij")
    x_grid_2, y_grid_2 = np.meshgrid(x_face, y_face, indexing="ij")

    x_plane = np.full_like(y_grid, float(x[x_idx]), dtype=np.float64)
    y_plane = np.full_like(x_grid, float(y[y_idx]), dtype=np.float64)
    z_plane = np.full_like(x_grid_2, float(z[z_idx]), dtype=np.float64)

    yz_colors = cmap(norm(yz_values))
    xz_colors = cmap(norm(xz_values))
    xy_colors = cmap(norm(xy_values))

    ax.plot_surface(
        x_plane,
        y_grid,
        z_grid,
        facecolors=yz_colors,
        shade=False,
        linewidth=0.0,
        antialiased=False,
        alpha=1.0,
    )
    ax.plot_surface(
        x_grid,
        y_plane,
        z_grid_2,
        facecolors=xz_colors,
        shade=False,
        linewidth=0.0,
        antialiased=False,
        alpha=1.0,
    )
    ax.plot_surface(
        x_grid_2,
        y_grid_2,
        z_plane,
        facecolors=xy_colors,
        shade=False,
        linewidth=0.0,
        antialiased=False,
        alpha=1.0,
    )

    edge_color = (0.08, 0.08, 0.08, 0.78)
    lw = 0.8
    ax.plot([0.0, x[x_idx]], [y[y_idx], y[y_idx]], [0.0, 0.0], color=edge_color, linewidth=lw)
    ax.plot([0.0, x[x_idx]], [y[y_idx], y[y_idx]], [z[z_idx], z[z_idx]], color=edge_color, linewidth=lw)
    ax.plot([0.0, 0.0], [y[y_idx], y[-1]], [0.0, 0.0], color=edge_color, linewidth=lw)
    ax.plot([0.0, 0.0], [y[y_idx], y[-1]], [z[z_idx], z[z_idx]], color=edge_color, linewidth=lw)
    ax.plot([x[x_idx], x[x_idx]], [y[y_idx], y[-1]], [0.0, 0.0], color=edge_color, linewidth=lw)
    ax.plot([x[x_idx], x[x_idx]], [y[y_idx], y[-1]], [z[z_idx], z[z_idx]], color=edge_color, linewidth=lw)
    ax.plot([0.0, 0.0], [y[y_idx], y[y_idx]], [0.0, z[z_idx]], color=edge_color, linewidth=lw)
    ax.plot([x[x_idx], x[x_idx]], [y[y_idx], y[y_idx]], [0.0, z[z_idx]], color=edge_color, linewidth=lw)
    ax.plot([0.0, 0.0], [y[-1], y[-1]], [0.0, z[z_idx]], color=edge_color, linewidth=lw)
    ax.plot([x[x_idx], x[x_idx]], [y[-1], y[-1]], [0.0, z[z_idx]], color=edge_color, linewidth=lw)


def _shrink_axes(ax: plt.Axes, scale_x: float = 0.80, scale_y: float = 0.80) -> None:
    pos = ax.get_position()
    new_w = pos.width * float(scale_x)
    new_h = pos.height * float(scale_y)
    new_x0 = pos.x0 + 0.5 * (pos.width - new_w)
    new_y0 = pos.y0 + 0.5 * (pos.height - new_h)
    ax.set_position([new_x0, new_y0, new_w, new_h])


def _offset_axes(ax: plt.Axes, dy: float = 0.0) -> None:
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 + float(dy), pos.width, pos.height])


def _set_row_y0(axes: list[plt.Axes], y0: float) -> None:
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, float(y0), pos.width, pos.height])


def _row_bbox(axes: list[plt.Axes]) -> tuple[float, float, float, float]:
    x0 = min(ax.get_position().x0 for ax in axes)
    x1 = max(ax.get_position().x1 for ax in axes)
    y0 = min(ax.get_position().y0 for ax in axes)
    y1 = max(ax.get_position().y1 for ax in axes)
    return x0, x1, y0, y1


def save_fig15(
    *,
    out_path: Path,
    x_value: float = 0.5,
    y_value: float = 0.5,
    z_value: float = 0.5,
    elev: float = 24.0,
    azim: float = -58.0,
    cmap_name: str = "coolwarm",
    save_pdf: bool = True,
) -> Path:
    loaded = {}
    for name, spec in DATASETS.items():
        loaded[name] = _load_npz(Path(spec["path"]).expanduser().resolve())

    x = np.asarray(loaded["Truth"]["x"], dtype=np.float64).reshape(-1)
    y = np.asarray(loaded["Truth"]["y"], dtype=np.float64).reshape(-1)
    z = np.asarray(loaded["Truth"]["z"], dtype=np.float64).reshape(-1)

    u_fields = []
    f_fields = []
    for name, spec in DATASETS.items():
        u_key = str(spec["u_field"])
        f_key = str(spec["f_field"])
        if u_key not in loaded[name]:
            raise KeyError(f"Field '{u_key}' not found in {spec['path']}")
        if f_key not in loaded[name]:
            raise KeyError(f"Field '{f_key}' not found in {spec['path']}")
        u_fields.append(np.asarray(loaded[name][u_key], dtype=np.float64))
        f_fields.append(np.asarray(loaded[name][f_key], dtype=np.float64))

    u_norm = _shared_norm(u_fields)
    f_norm = _shared_norm(f_fields)
    cmap = colormaps.get_cmap(cmap_name)

    fig = plt.figure(figsize=(11.8, 6.45), dpi=320)
    axes = [
        fig.add_subplot(2, 4, idx + 1, projection="3d")
        for idx in range(8)
    ]
    fig.subplots_adjust(
        left=0.05,
        right=0.86,
        bottom=0.06,
        top=0.94,
        wspace=0.26,
        hspace=0.02,
    )

    for col, ((name, spec), u_field) in enumerate(zip(DATASETS.items(), u_fields)):
        ax = axes[col]
        _draw_visible_cube(
            ax=ax,
            x=x,
            y=y,
            z=z,
            field=u_field,
            norm=u_norm,
            cmap=cmap,
            x_value=x_value,
            y_value=y_value,
            z_value=z_value,
        )
        _style_axes(
            ax,
            elev=elev,
            azim=azim,
            x_value=x_value,
            y_value=y_value,
            z_value=z_value,
        )
        ax.set_title(name, pad=2)
        _shrink_axes(ax, scale_x=0.80, scale_y=0.80)

    for col, ((name, spec), f_field) in enumerate(zip(DATASETS.items(), f_fields)):
        ax = axes[4 + col]
        _draw_visible_cube(
            ax=ax,
            x=x,
            y=y,
            z=z,
            field=f_field,
            norm=f_norm,
            cmap=cmap,
            x_value=x_value,
            y_value=y_value,
            z_value=z_value,
        )
        _style_axes(
            ax,
            elev=elev,
            azim=azim,
            x_value=x_value,
            y_value=y_value,
            z_value=z_value,
        )
        _shrink_axes(ax, scale_x=0.80, scale_y=0.80)

    bot_h = min(ax.get_position().height for ax in axes[4:])
    row_gap = 0.036
    bottom_row_y0 = 0.11
    top_row_y0 = bottom_row_y0 + bot_h + row_gap
    _set_row_y0(axes[4:], bottom_row_y0)
    _set_row_y0(axes[:4], top_row_y0)

    top_x0, top_x1, top_y0, top_y1 = _row_bbox(axes[:4])
    bot_x0, bot_x1, bot_y0, bot_y1 = _row_bbox(axes[4:])
    cbar_pad = 0.028
    cbar_w = 0.011

    u_sm = cm.ScalarMappable(norm=u_norm, cmap=cmap)
    u_sm.set_array([])
    u_cbar = fig.colorbar(
        u_sm,
        cax=fig.add_axes([top_x1 + cbar_pad, top_y0, cbar_w, top_y1 - top_y0]),
    )
    u_cbar.set_label("u", fontsize=8)
    u_cbar.ax.tick_params(labelsize=7)

    f_sm = cm.ScalarMappable(norm=f_norm, cmap=cmap)
    f_sm.set_array([])
    f_cbar = fig.colorbar(
        f_sm,
        cax=fig.add_axes([bot_x1 + cbar_pad, bot_y0, cbar_w, bot_y1 - bot_y0]),
    )
    f_cbar.set_label("f", fontsize=8)
    f_cbar.ax.tick_params(labelsize=7)

    left_pad = 0.014
    fig.text(top_x0 - left_pad, 0.5 * (top_y0 + top_y1), "u", fontsize=10, rotation=90, va="center", ha="center")
    fig.text(bot_x0 - left_pad, 0.5 * (bot_y0 + bot_y1), "f", fontsize=10, rotation=90, va="center", ha="center")

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_path.with_suffix(".png")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot x=0.5 and y=0.5 slices for truth and model predictions."
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--x-value", type=float, default=0.5)
    parser.add_argument("--y-value", type=float, default=0.5)
    parser.add_argument("--z-value", type=float, default=0.5)
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    parser.add_argument("--cmap", type=str, default="coolwarm")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()

    out_path = save_fig15(
        out_path=Path(args.out),
        x_value=float(args.x_value),
        y_value=float(args.y_value),
        z_value=float(args.z_value),
        elev=float(args.elev),
        azim=float(args.azim),
        cmap_name=str(args.cmap),
        save_pdf=not bool(args.no_pdf),
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
