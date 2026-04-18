#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
import os
import matplotlib.gridspec as gridspec
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

def generate_simple_grid_row1(L_box=1.0, n_internal=10):
    grid_1d = np.linspace(-L_box, L_box, n_internal)
    gx, gy = np.meshgrid(grid_1d, grid_1d, indexing="xy")
    xy_basic = np.stack([gx.ravel(), gy.ravel()], axis=1)
    xy_extra = np.array([])
    return xy_basic, xy_extra


def load_ellipse_uv_eps_fit_downsample(
    nx=40, ny=40,
    *,
    txt_filename="Ellipse.txt",
    ellipse=(0.05, 0.10, 0.35, 0.15, np.deg2rad(-30.0)),
    tau_for_strain=0.02,
    use_dense=True,
    dense_factor=0.5,
    rect_params=((0.05, 0.10), 0.9, 0.5, -30.0),
    target_total=100,
    random_state=0,
    remove_boundary=False,
    L_box=1.0,
    **kwargs
):
    if not os.path.exists(txt_filename):
        dummy = np.linspace(-1, 1, 40)
        gx, gy = np.meshgrid(dummy, dummy)
        xy_full = np.stack([gx.ravel(), gy.ravel()], axis=1)
    else:
        with open(txt_filename, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln for ln in f if not ln.lstrip().startswith("%")]
        arr = np.loadtxt(lines)
        xy_full = arr[:, 0:2]

    all_nodes = xy_full
    tree = cKDTree(all_nodes)

    xmin0, ymin0 = xy_full.min(axis=0)
    xmax0, ymax0 = xy_full.max(axis=0)

    x_tar = np.linspace(xmin0, xmax0, nx, dtype=np.float64)
    y_tar = np.linspace(ymin0, ymax0, ny, dtype=np.float64)
    Xg, Yg = np.meshgrid(x_tar, y_tar, indexing="ij")
    pts_main = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    _, idxs_main = tree.query(pts_main, k=1)
    idxs_basic_unique = np.unique(idxs_main)

    rect_info_out = None
    idxs_extra_only = np.array([], dtype=int)

    if use_dense:
        (cx, cy), width, height, angle_deg = rect_params
        rect_info_out = {'center': (cx, cy), 'width': width, 'height': height, 'angle': angle_deg}
        angle_rad = np.deg2rad(angle_deg)

        dx_global = (xmax0 - xmin0) / (nx - 1) if nx > 1 else 1.0
        target_dx = dx_global * dense_factor
        dense_nx = max(2, int(np.ceil(width / target_dx)) + 1)
        dense_ny = max(2, int(np.ceil(height / target_dx)) + 1)

        w_half, h_half = width / 2.0, height / 2.0
        x_local = np.linspace(-w_half, w_half, dense_nx + 2)[1:-1]
        y_local = np.linspace(-h_half, h_half, dense_ny + 2)[1:-1]
        X_loc, Y_loc = np.meshgrid(x_local, y_local, indexing='xy')
        pts_local = np.stack([X_loc.ravel(), Y_loc.ravel()], axis=1)

        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        pts_dense_world = pts_local @ R.T + np.array([cx, cy])

        _, idxs_rect = tree.query(pts_dense_world, k=1)
        idxs_rect_unique = np.unique(idxs_rect)
        idxs_extra_only = np.setdiff1d(idxs_rect_unique, idxs_basic_unique, assume_unique=True)

    idxs_extra_keep = idxs_extra_only.copy()

    if target_total is not None:
        Nb = idxs_basic_unique.size
        Ne = idxs_extra_only.size
        total_now = Nb + Ne
        if total_now > target_total:
            max_extra_allowed = max(target_total - Nb, 0)
            rng = np.random.default_rng(random_state)
            if Ne > max_extra_allowed:
                idxs_extra_keep = rng.choice(idxs_extra_only, size=max_extra_allowed, replace=False)

    idxs_extra_keep = np.array(sorted(np.unique(idxs_extra_keep)), dtype=int)

    xy_basic_plot = all_nodes[idxs_basic_unique]
    xy_dense_extra_plot = all_nodes[idxs_extra_keep]

    if remove_boundary:
        tol = 1e-5
        if xy_basic_plot.size > 0:
            mask_basic = (np.abs(xy_basic_plot[:, 0]) < L_box - tol) & (np.abs(xy_basic_plot[:, 1]) < L_box - tol)
            xy_basic_plot = xy_basic_plot[mask_basic]
        if xy_dense_extra_plot.size > 0:
            mask_extra = (np.abs(xy_dense_extra_plot[:, 0]) < L_box - tol) & (np.abs(xy_dense_extra_plot[:, 1]) < L_box - tol)
            xy_dense_extra_plot = xy_dense_extra_plot[mask_extra]

    return xy_basic_plot, xy_dense_extra_plot, rect_info_out


def get_true_ellipse_contour():
    t = np.linspace(0, 2*np.pi, 200)
    a, b = 0.35, 0.15
    xc, yc = 0.05, 0.10
    theta = np.radians(-30)
    x, y = a * np.cos(t), b * np.sin(t)
    return x * np.cos(theta) - y * np.sin(theta) + xc, x * np.sin(theta) + y * np.cos(theta) + yc


# ==============================================================================
# Part 2: 绘图辅助函数
# ==============================================================================
# 统一所有子图的数据范围：固定为物理域 [-1, 1]
COMMON_LIMIT = 1.0


def _reserve_same_layout_as_evolution(ax, *, reserve_title="Iteration 20000"):
    ax.set_xlim(-COMMON_LIMIT, COMMON_LIMIT)
    ax.set_ylim(-COMMON_LIMIT, COMMON_LIMIT)
    ax.set_aspect('equal', adjustable='box')

    # 给 constrained_layout 预留与演化图一致的标题/标签/刻度空间
    ax.set_title(reserve_title, pad=10)
    ax.title.set_alpha(0.0)

    ax.set_xticks([-0.8, 0.0, 0.8])
    ax.set_yticks([-0.8, 0.0, 0.8])
    ax.set_xlabel(r"$x_1$", fontsize=18)
    ax.set_ylabel(r"$x_2$", fontsize=18)

    ax.xaxis.label.set_alpha(0.0)
    ax.yaxis.label.set_alpha(0.0)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_alpha(0.0)

    # 不显示刻度线
    ax.tick_params(bottom=False, left=False, top=False, right=False)


def _draw_boundary_arrows_axescoords(ax, n=6, offset=0.02, length=0.12):
    ps = np.linspace(0.05, 0.95, n)

    arrowprops = dict(
        arrowstyle='-|>',     # 典型“箭杆+实心箭头”
        lw=1.3,               # 箭杆粗细
        color='red',
        mutation_scale=18,    # 箭头大小
        shrinkA=0,
        shrinkB=0
    )

    # 上下
    for p in ps:
        ax.annotate(
            "", xy=(p, 1 + offset + length), xytext=(p, 1 + offset),
            xycoords=ax.transAxes, textcoords=ax.transAxes,
            arrowprops=arrowprops, annotation_clip=False, zorder=20
        )
        ax.annotate(
            "", xy=(p, -offset - length), xytext=(p, -offset),
            xycoords=ax.transAxes, textcoords=ax.transAxes,
            arrowprops=arrowprops, annotation_clip=False, zorder=20
        )

    # 左右
    for p in ps:
        ax.annotate(
            "", xy=(1 + offset + length, p), xytext=(1 + offset, p),
            xycoords=ax.transAxes, textcoords=ax.transAxes,
            arrowprops=arrowprops, annotation_clip=False, zorder=20
        )
        ax.annotate(
            "", xy=(-offset - length, p), xytext=(-offset, p),
            xycoords=ax.transAxes, textcoords=ax.transAxes,
            arrowprops=arrowprops, annotation_clip=False, zorder=20
        )


def draw_schematic(ax, *, xy_basic_plot=None, xy_dense_extra_plot=None, is_sparse_mode=False, rect_info=None):
    xc, yc = 0.05, 0.10
    axis_A, axis_B, angle = 0.35 * 2, 0.15 * 2, -30

    # 关键：固定数据范围 + 预留布局空间，让第一列和后面列等大对齐
    _reserve_same_layout_as_evolution(ax, reserve_title="Iteration 20000")

    # 背景色
    ax.set_facecolor('#d0f0fd')

    # 椭圆
    ellipse = patches.Ellipse(
        (xc, yc), axis_A, axis_B, angle=angle,
        edgecolor='gray', linestyle='--', linewidth=1.5,
        facecolor='#fce4d6', alpha=0.8, zorder=1
    )
    ax.add_patch(ellipse)

    # 箭头：轴坐标绘制到轴外，不改变矩形大小
    _draw_boundary_arrows_axescoords(ax, n=6, offset=0.02, length=0.10)

    # ROI 矩形
    if rect_info is not None:
        cx, cy = rect_info['center']
        w, h = rect_info['width'], rect_info['height']
        ang = rect_info['angle']
        ang_rad = np.deg2rad(ang)
        c, s = np.cos(ang_rad), np.sin(ang_rad)
        R = np.array([[c, -s], [s, c]])
        corners_local = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2], [-w/2, -h/2]])
        corners_world = corners_local @ R.T + np.array([cx, cy])
        ax.plot(corners_world[:, 0], corners_world[:, 1],
                color="limegreen", linestyle="-.", linewidth=2.0, zorder=2)

    # 采样点
    if xy_basic_plot is not None and xy_basic_plot.size > 0:
        ax.scatter(
            xy_basic_plot[:, 0], xy_basic_plot[:, 1],
            s=30, c='lightgray', edgecolors='gray', linewidth=0.8,
            alpha=0.9, zorder=3
        )

    if xy_dense_extra_plot is not None and xy_dense_extra_plot.size > 0:
        ax.scatter(
            xy_dense_extra_plot[:, 0], xy_dense_extra_plot[:, 1],
            s=16, c='#666666', edgecolors='white', linewidth=0.4,
            alpha=0.85, zorder=4
        )


def plot_evolution_panel(ax, filepath, epoch_num):
    true_x, true_y = get_true_ellipse_contour()
    ax.plot(true_x, true_y, color='blue', linewidth=2.5, linestyle='-', label='True')

    if os.path.exists(filepath):
        data = np.load(filepath)
        phi = data['phi'] if 'phi' in data else data['fai']

        if 'x' in data:
            x_grid, y_grid = data['x'], data['y']
            if x_grid.ndim == 1:
                X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
            else:
                X, Y = x_grid, y_grid
        else:
            n = int(np.sqrt(phi.size))
            x_grid = np.linspace(-1, 1, n)
            X, Y = np.meshgrid(x_grid, x_grid, indexing="ij")
            phi = phi.reshape(n, n)

        ax.contour(X, Y, phi, levels=[0], colors='red', linewidths=2.5)
    else:
        t = np.linspace(0, 2*np.pi, 100)
        r = 0.4 if epoch_num > 10000 else 0.2
        ax.plot(r*np.cos(t), r*np.sin(t), 'r-', linewidth=2.5)

    # 关键：所有子图范围一致
    ax.set_xlim(-COMMON_LIMIT, COMMON_LIMIT)
    ax.set_ylim(-COMMON_LIMIT, COMMON_LIMIT)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Iteration {epoch_num}", pad=10)

    ax.set_xticks([-0.8, 0, 0.8])
    ax.set_yticks([-0.8, 0, 0.8])
    ax.set_xlabel(r"$x_1$", fontsize=26)
    ax.set_ylabel(r"$x_2$", fontsize=26)
    ax.yaxis.set_label_coords(-0.19, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.1)

# ==============================================================================
# Part 3: 主逻辑
# ==============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(script_dir, "Ellipse.txt")
    output_dir = os.path.join(os.path.dirname(script_dir), "Figure")
    os.makedirs(output_dir, exist_ok=True)

    roi_rect_corners = ((-0.5, -0.2), (0.7, 0.4))
    (x0, y0), (x1, y1) = roi_rect_corners
    roi_params = {
        'center': (0.5 * (x0 + x1), 0.5 * (y0 + y1)),
        'width': abs(x1 - x0),
        'height': abs(y1 - y0),
        'angle': 0.0,
        'color': '#2ca02c',
        'lw': 2.5,
        'ls': '--',
        'alpha': 0.8
    }

    data_dirs = [
        os.path.join(script_dir, "output_roi_off", "phi_snapshots"),
        os.path.join(script_dir, "output_roi_on", "phi_snapshots"),
    ]

    ellipse_params = (0.05, 0.10, 0.35, 0.15, np.deg2rad(-30.0))

    # 1) 数据生成
    xy_basic_row1, xy_extra_row1 = generate_simple_grid_row1(L_box=1.0, n_internal=10)

    xy_basic_row2, xy_extra_row2, rect_info_row2 = load_ellipse_uv_eps_fit_downsample(
        nx=7, ny=7, txt_filename=txt_path, ellipse=ellipse_params,
        tau_for_strain=0.001, use_dense=True, dense_factor=0.3,
        rect_params=(roi_params['center'], roi_params['width'], roi_params['height'], roi_params['angle']),
        target_total=100, random_state=2, remove_boundary=False, L_box=1.0
    )

    # 2) 绘图
    target_epochs = [1000, 15000, 60000]
    nrows = len(data_dirs)

    # 1. 使用 Figure 初始化画布
    fig = plt.figure(figsize=(22, 12), dpi=150)

    # 2. 定义外层网格：1 行 2 列
    # width_ratios=[1, 3.5] 这里的 3.5 决定了右侧三个子图的总宽度
    # wspace=0.15 这里的数值专门用来控制【第一列】和【后面三列】之间的间距
    outer_gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 3.8], wspace=0.2)

    # 3. 创建 axes 存储阵列
    axes = np.empty((nrows, 4), dtype=object)

    # 4. 填充子图
    for r in range(nrows):
        # 左侧：示意图区域 (1列)
        # 如果 nrows > 1，这里需要进一步对 outer_gs[0] 进行 row 切分
        left_gs = outer_gs[0].subgridspec(nrows, 1, hspace=0.15)
        axes[r, 0] = fig.add_subplot(left_gs[r, 0])

        # 右侧：演化图区域 (3列)
        # wspace=0.4 是这三列演化图之间的独立间距，不会影响到第一列
        right_gs = outer_gs[1].subgridspec(nrows, 3, wspace=0.4, hspace=0.15)
        for c in range(3):
            axes[r, c+1] = fig.add_subplot(right_gs[r, c])

    # 兼容处理
    if nrows == 1:
        axes = axes.reshape(1, 4)

    for row_idx, data_dir in enumerate(data_dirs):
        row_axes = axes[row_idx]

        # Col 0: 示意图
        if row_idx == 0:
            draw_schematic(
                row_axes[0],
                xy_basic_plot=xy_basic_row1,
                xy_dense_extra_plot=xy_extra_row1,
                is_sparse_mode=False
            )
            row_axes[0].text(
                -0.28, 0.5, "Dense\nMeasurement",
                transform=row_axes[0].transAxes,
                rotation=90, va='center', ha='center',
                fontsize=20, clip_on=False
            )
        else:
            draw_schematic(
                row_axes[0],
                xy_basic_plot=xy_basic_row2,
                xy_dense_extra_plot=xy_extra_row2,
                is_sparse_mode=True,
                rect_info=rect_info_row2
            )
            row_axes[0].text(
                -0.28, 0.5, "Targeted\nMeasurement",
                transform=row_axes[0].transAxes,
                rotation=90, va='center', ha='center',
                fontsize=20, clip_on=False
            )

        # Col 1-3: 演化图
        for i, epoch in enumerate(target_epochs):
            ax = row_axes[i + 1]

            filename = f"phi_epoch_{epoch:08d}.npz"
            filepath = os.path.join(data_dir, filename)
            plot_evolution_panel(ax, filepath, epoch)

            # 在第一行最后一张演化图叠加 ROI 框
            if row_idx == 0 and epoch == target_epochs[-1]:
                p = roi_params
                cx, cy = p['center']
                w, h = p['width'], p['height']
                ang = p['angle']
                ang_rad = np.deg2rad(ang)
                c, s = np.cos(ang_rad), np.sin(ang_rad)
                R = np.array([[c, -s], [s, c]])
                corners_local = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
                corners_world = corners_local @ R.T + np.array([cx, cy])

                poly = patches.Polygon(
                    corners_world, closed=True,
                    linewidth=p['lw'], edgecolor=p['color'],
                    facecolor='none', linestyle=p['ls'],
                    alpha=p['alpha'], zorder=10
                )
                ax.add_patch(poly)

                label_pos_local = np.array([0, h/2])
                label_pos_world = label_pos_local @ R.T + np.array([cx, cy])

                ax.text(
                    label_pos_world[0], label_pos_world[1] + 0.05, "ROI",
                    color=p['color'], fontsize=20,
                    ha='center', va='bottom', zorder=11
                )
            # legend
            if row_idx == 0 and i == 0:
                custom_lines = [
                    Line2D([0], [0], color='blue', lw=2.5),
                    Line2D([0], [0], color='red', lw=2.5),
                ]
                ax.legend(custom_lines, ['True', 'Pred'], loc='upper right', frameon=False, fontsize=20)

    out_path = os.path.join(output_dir, "Fig7.png")
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "Fig7.pdf"), bbox_inches='tight')
    print(f"Plot saved to {out_path}")
    # plt.show()


if __name__ == "__main__":
    main()
