#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

from config import SAMPLING_CONFIGS, DataConfig, VisualizationConfig

# ==============================================================================
# Part 0: 全局绘图风格配置
# ==============================================================================
plt.rcParams.update({
    'font.size': 18,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.5,
    'font.family': 'sans-serif',
    'mathtext.fontset': 'stix',
    'font.weight': 'normal',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 8,
    'xtick.major.width': 1.5,
    'ytick.major.size': 8,
    'ytick.major.width': 1.5,
})

# ==============================================================================
# Part 1: 数据生成
# ==============================================================================

def load_uniform_grid_fit_downsample(
    nx=8,
    ny=8,
    *,
    ttxt_filename="Possion.txt",
    circles=None,
    dense_factor=0.5,
    drop_boundary=True,
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    tol=0.02,
):
    xy_full = np.loadtxt(ttxt_filename, usecols=[0, 1], comments="%")
    n_side = int(np.sqrt(len(xy_full)))
    x_full = xy_full[:, 0].reshape(n_side, n_side)
    y_full = xy_full[:, 1].reshape(n_side, n_side)

    x_vec = np.unique(x_full)
    y_vec = np.unique(y_full)

    all_nodes = np.stack([x_full.flatten(), y_full.flatten()], axis=1)
    tree = cKDTree(all_nodes)

    x_tar = np.linspace(x_vec[1], x_vec[-2], nx)
    y_tar = np.linspace(y_vec[1], y_vec[-2], ny)
    xg, yg = np.meshgrid(x_tar, y_tar, indexing="ij")
    ideal_points = np.stack([xg.flatten(), yg.flatten()], axis=1)
    idxs_basic = tree.query(ideal_points, k=1)[1]
    idxs_basic = np.unique(idxs_basic)
    idxs_dense = np.array([], dtype=int)

    if circles:
        for (cx, cy, r) in circles:
            dense_nx = max(2, int(nx / dense_factor))
            dense_ny = max(2, int(ny / dense_factor))
            x_dense = np.linspace(x_vec[1], x_vec[-2], dense_nx)
            y_dense = np.linspace(y_vec[1], y_vec[-2], dense_ny)
            xd, yd = np.meshgrid(x_dense, y_dense, indexing="ij")
            pts_dense = np.stack([xd.flatten(), yd.flatten()], axis=1)
            mask = ((pts_dense[:, 0] - cx) ** 2 + (pts_dense[:, 1] - cy) ** 2) <= r**2
            pts_in = pts_dense[mask]
            if pts_in.size > 0:
                idxs_in = tree.query(pts_in, k=1)[1]
                idxs_dense = np.unique(np.concatenate([idxs_dense, idxs_in]))

    if idxs_dense.size > 0:
        idxs_dense = np.setdiff1d(idxs_dense, idxs_basic, assume_unique=False)

    xy_basic = all_nodes[idxs_basic]
    xy_dense = all_nodes[idxs_dense] if idxs_dense.size > 0 else np.array([]).reshape(0, 2)

    if drop_boundary:
        mask_basic = (
            (xy_basic[:, 0] > xlim[0] + tol)
            & (xy_basic[:, 0] < xlim[1] - tol)
            & (xy_basic[:, 1] > ylim[0] + tol)
            & (xy_basic[:, 1] < ylim[1] - tol)
        )
        xy_basic = xy_basic[mask_basic]

        if xy_dense.size > 0:
            mask_dense = (
                (xy_dense[:, 0] > xlim[0] + tol)
                & (xy_dense[:, 0] < xlim[1] - tol)
                & (xy_dense[:, 1] > ylim[0] + tol)
                & (xy_dense[:, 1] < ylim[1] - tol)
            )
            xy_dense = xy_dense[mask_dense]

    return xy_basic, xy_dense


def sdf_rect(xx, yy, xc, yc, a, b):
    qx = np.abs(xx - xc) - a
    qy = np.abs(yy - yc) - b
    qx_pos = np.maximum(qx, 0.0)
    qy_pos = np.maximum(qy, 0.0)
    outside = np.hypot(qx_pos, qy_pos)
    inside = np.maximum(qx, qy)
    return outside + np.minimum(inside, 0.0)


def sdf_cross(xx, yy, xc, yc, lx, ly, wh, wv):
    phi_h = sdf_rect(xx, yy, xc, yc, a=lx / 2.0, b=wh / 2.0)
    phi_v = sdf_rect(xx, yy, xc, yc, a=wv / 2.0, b=ly / 2.0)
    return np.minimum(phi_h, phi_v)


# ==============================================================================
# Part 2: 绘图辅助函数
# ==============================================================================
XMIN, XMAX = 0.0, 1.0
YMIN, YMAX = 0.0, 1.0


def _reserve_same_layout_as_evolution(ax, *, reserve_title="Iteration 180000"):
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect('equal', adjustable='box')

    ax.set_title(reserve_title, pad=10)
    ax.title.set_alpha(0.0)

    ax.set_xticks([0.1, 0.5, 0.9])
    ax.set_yticks([0.1, 0.5, 0.9])
    ax.set_xlabel(r"$x_1$", fontsize=18)
    ax.set_ylabel(r"$x_2$", fontsize=18)

    ax.xaxis.label.set_alpha(0.0)
    ax.yaxis.label.set_alpha(0.0)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_alpha(0.0)

    ax.tick_params(bottom=False, left=False, top=False, right=False)


def _show_sampling_ticks(ax):
    ax.set_xlabel(r"$x_1$", fontsize=26)
    ax.set_ylabel(r"$x_2$", fontsize=26)
    ax.xaxis.label.set_alpha(1.0)
    ax.yaxis.label.set_alpha(1.0)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_alpha(1.0)
    ax.tick_params(bottom=True, left=True, top=True, right=True)


def _draw_cross_patch(ax, cross, *, facecolor="#f0e4d6", edgecolor="gray"):
    xc, yc, lx, ly, wh, wv = cross
    rect_h = patches.Rectangle(
        (xc - lx / 2.0, yc - wh / 2.0),
        lx,
        wh,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
        linestyle='--',
        zorder=1,
    )
    rect_v = patches.Rectangle(
        (xc - wv / 2.0, yc - ly / 2.0),
        wv,
        ly,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
        linestyle='--',
        zorder=1,
    )
    ax.add_patch(rect_h)
    ax.add_patch(rect_v)


def _draw_roi_circle(ax, circle, *, color="limegreen", linestyle="--", lw=2.0):
    cx, cy, r = circle
    circ = patches.Circle((cx, cy), r, fill=False, edgecolor=color, linestyle=linestyle, linewidth=lw)
    ax.add_patch(circ)


def draw_sampling_panel(
    ax,
    *,
    xy_basic,
    xy_dense,
    cross_params,
    circle_roi=None,
    show_roi=False,
    dense_style=None,
    basic_size=16,
):
    _reserve_same_layout_as_evolution(ax, reserve_title="Iteration 180000")
    _show_sampling_ticks(ax)
    ax.set_facecolor('#d0f0fd')

    _draw_cross_patch(ax, cross_params)

    if show_roi and circle_roi is not None:
        _draw_roi_circle(ax, circle_roi, color="limegreen", linestyle="--", lw=2.0)

    if xy_basic is not None and xy_basic.size > 0:
        ax.scatter(
            xy_basic[:, 0], xy_basic[:, 1],
            s=basic_size, c='lightgray', edgecolors='gray', linewidth=0.6,
            alpha=0.9, zorder=3
        )

    if xy_dense is not None and xy_dense.size > 0:
        style = dense_style or {}
        ax.scatter(
            xy_dense[:, 0], xy_dense[:, 1],
            s=style.get("s", 8),
            c=style.get("c", "#777777"),
            edgecolors=style.get("edgecolors", "white"),
            linewidth=style.get("linewidth", 0.4),
            alpha=style.get("alpha", 0.9),
            zorder=4,
        )

def plot_evolution_panel(ax, filepath, epoch_num, cross_params):
    if os.path.exists(filepath):
        data = np.load(filepath)
        phi = data['phi']

        x_grid, y_grid = data['x'], data['y']
        if x_grid.ndim == 1:
            X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
        else:
            X, Y = x_grid, y_grid
    else:
        # fallback grid
        xs = np.linspace(XMIN, XMAX, 200)
        ys = np.linspace(YMIN, YMAX, 200)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        phi = None

    phi_true = sdf_cross(X, Y, *cross_params)
    ax.contour(X, Y, phi_true, levels=[0], colors='blue', linewidths=2.5)

    if phi is not None:
        ax.contour(X, Y, phi, levels=[0], colors='red', linewidths=2.5)

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Iteration {epoch_num}", pad=10)

    ax.set_xticks([0.1, 0.5, 0.9])
    ax.set_yticks([0.1, 0.5, 0.9])
    ax.set_xlabel(r"$x_1$", fontsize=26)
    ax.set_ylabel(r"$x_2$", fontsize=26)
    ax.yaxis.set_label_coords(-0.19, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.12)


# ==============================================================================
# Part 3: 主逻辑
# ==============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(script_dir, "Possion.txt")

    output_dir = os.path.join(os.path.dirname(script_dir), "Figure")
    os.makedirs(output_dir, exist_ok=True)

    data_cfg = DataConfig()
    viz_cfg = VisualizationConfig()

    cross_params = viz_cfg.phi_compare_cross
    circle_roi = viz_cfg.phi_compare_circle

    sampling_modes = [
        ("roi-off", "Dense\nMeasurement", False),
        ("roi-on", "Targeted\nMeasurement", True),
        ("full-data", "Full-Field\nMeasurement", True),
    ]

    data_dirs = [
        os.path.join(script_dir, "output_roi_off", "phi_snapshots"),
        os.path.join(script_dir, "output_roi_on", "phi_snapshots"),
        os.path.join(script_dir, "output_full_data", "phi_snapshots"),
    ]

    sampling_points = []
    for mode, _, _ in sampling_modes:
        cfg = SAMPLING_CONFIGS[mode]
        xy_basic, xy_dense = load_uniform_grid_fit_downsample(
            nx=cfg["nx"],
            ny=cfg["ny"],
            ttxt_filename=txt_path,
            circles=list(data_cfg.circles),
            dense_factor=cfg["dense_factor"],
            drop_boundary=cfg["drop_boundary"],
            xlim=cfg["xlim"],
            ylim=cfg["ylim"],
            tol=cfg["tol"],
        )
        sampling_points.append((xy_basic, xy_dense))

    # 2) 绘图
    target_epochs = [40000,45000, 80000]
    nrows = len(data_dirs)

    fig = plt.figure(figsize=(22, 16), dpi=150)
    outer_gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 3.8], wspace=0.2)

    axes = np.empty((nrows, 4), dtype=object)
    for r in range(nrows):
        left_gs = outer_gs[0].subgridspec(nrows, 1, hspace=0.15)
        axes[r, 0] = fig.add_subplot(left_gs[r, 0])

        right_gs = outer_gs[1].subgridspec(nrows, 3, wspace=0.4, hspace=0.15)
        for c in range(3):
            axes[r, c + 1] = fig.add_subplot(right_gs[r, c])

    if nrows == 1:
        axes = axes.reshape(1, 4)

    for row_idx, (mode_info, data_dir) in enumerate(zip(sampling_modes, data_dirs)):
        row_axes = axes[row_idx]
        mode, row_label, show_roi = mode_info

        xy_basic, xy_dense = sampling_points[row_idx]
        dense_style = None
        basic_size = 16
        if row_idx in (1, 2):
            dense_style = {"s": 8, "c": "#666666", "edgecolors": "white", "linewidth": 0.4, "alpha": 0.9}
        if row_idx == 2 and dense_style is not None:
            dense_style = {**dense_style, "s": 2}

        draw_sampling_panel(
            row_axes[0],
            xy_basic=xy_basic,
            xy_dense=xy_dense,
            cross_params=cross_params,
            circle_roi=circle_roi,
            show_roi=show_roi,
            dense_style=dense_style,
            basic_size=basic_size,
        )
        row_axes[0].text(
            -0.37, 0.5, row_label,
            transform=row_axes[0].transAxes,
            rotation=90, va='center', ha='center',
            fontsize=20, clip_on=False
        )

        for i, epoch in enumerate(target_epochs):
            ax = row_axes[i + 1]
            filename = f"phi_epoch_{epoch:08d}.npz"
            filepath = os.path.join(data_dir, filename)
            plot_evolution_panel(ax, filepath, epoch, cross_params)

            if row_idx == 0 and epoch == target_epochs[-1]:
                _draw_roi_circle(ax, circle_roi, color="limegreen", linestyle="--", lw=2.0)
                ax.text(
                    circle_roi[0], circle_roi[1] + circle_roi[2] + 0.03, "ROI",
                    color="limegreen", fontsize=18,
                    ha='center', va='bottom', zorder=11
                )

            if row_idx == 0 and i == 0:
                custom_lines = [
                    Line2D([0], [0], color='blue', lw=2.5),
                    Line2D([0], [0], color='red', lw=2.5),
                ]
                ax.legend(custom_lines, ['True', 'Pred'], loc='lower right', frameon=False, fontsize=20)

    out_path = os.path.join(output_dir, "Fig10.png")
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "Fig10.pdf"), bbox_inches='tight')
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
