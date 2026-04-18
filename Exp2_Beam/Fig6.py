import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ==========================================
# 1. Nature Computational Science Style Config
# ==========================================

# Figure Size: 180mm (double column) x ~125mm
FIG_WIDTH_INCH = 7.08  # 180 mm
FIG_HEIGHT_INCH = 3.8  # ~125 mm

# Fonts: Arial/Helvetica is standard for Nature journals
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['mathtext.fontset'] = 'stixsans'  # Math font matching sans-serif

# Font Sizes (points)
rcParams['font.size'] = 7
rcParams['axes.labelsize'] = 7
rcParams['axes.titlesize'] = 7
rcParams['xtick.labelsize'] = 6
rcParams['ytick.labelsize'] = 6
rcParams['legend.fontsize'] = 6

# Line Widths and Marker Sizes
rcParams['axes.linewidth'] = 0.6    # Axis spines
rcParams['grid.linewidth'] = 0.5    # Grid lines
rcParams['lines.linewidth'] = 1.2   # Plot lines
rcParams['lines.markersize'] = 3.5  # Marker size

# Colors (Okabe-Ito Palette for Colorblind Safety + Contrast)
C_EXACT = '#404040'   # Dark Gray (Ground Truth)
C_PIMOE = '#D55E00'   # ADD-PINNs
C_PINNs = '#0072B2'    # PINNs
C_APINNs = '#009E73'   # APINNs
C_JUMP = '#999999'    # Vertical jump lines

# Inset position in axes coordinates: (x0, y0, width, height), range [0, 1]
INSET_BBOX = {
    '(b)': (0.2, 0.12, 0.31, 0.28),
    '(c)': (0.5, 0.4, 0.31, 0.28),
}

# ==========================================
# 2. Data Loading & Helper Functions
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PIMOE_NPZ_PATH = str(BASE_DIR / 'data_output' / 'phi_final.npz')
DEFAULT_PINNs_NPZ_PATH = str(BASE_DIR / 'data_output_pinn' / 'phi_final.npz')
DEFAULT_APINNs_NPZ_PATH = str(BASE_DIR / 'data_output_apinn' / 'phi_final.npz')
OUTPUT_DIR = BASE_DIR.parent / 'Figure'

try:
    from config import TRUE_JUMPS
except Exception:
    # Fallback keeps plotting script runnable when training deps (e.g., torch) are unavailable.
    TRUE_JUMPS = (0.30, 0.60)


def get_arr(data: dict, key: str):
    """Safely retrieve and flatten numpy array from dict."""
    if key not in data:
        return None
    arr = data[key]
    if arr.size == 0:
        return None
    return arr.flatten()


def resolve_npz_path(raw_path: str, default_path: str = '') -> str:
    """Resolve user-provided path, then default path, then newest phi_*.npz."""
    if raw_path and os.path.exists(raw_path):
        return raw_path
    if default_path and os.path.exists(default_path):
        return default_path

    candidates = glob.glob(str(BASE_DIR / 'data_output' / 'phi_*.npz'))
    if not candidates:
        candidates = glob.glob(str(BASE_DIR / '**' / 'phi_*.npz'), recursive=True)

    if not candidates:
        raise FileNotFoundError('Could not find any phi_*.npz file. Please specify --npz.')

    return max(candidates, key=os.path.getmtime)


# ==========================================
# 3. Main Plotting Function
# ==========================================

def plot_nature_figure(
    npz_pimoe: str,
    npz_pinn: str = '',
    npz_apinn: str = '',
    output_path: str = '',
    show: bool = False,
):
    pimoe_path = resolve_npz_path(npz_pimoe, DEFAULT_PIMOE_NPZ_PATH)
    pinn_path = resolve_npz_path(npz_pinn, DEFAULT_PINNs_NPZ_PATH)
    apinn_path = resolve_npz_path(npz_apinn, DEFAULT_APINNs_NPZ_PATH)

    print(f'Loading ADD-PINNs data: {pimoe_path}')
    print(f'Loading PINNs data: {pinn_path}')
    print(f'Loading APINNs data: {apinn_path}')

    data_pimoe = np.load(pimoe_path)
    data_pinn = np.load(pinn_path)
    data_apinn = np.load(apinn_path)

    x_exact = get_arr(data_pimoe, 'x')
    if x_exact is None:
        raise ValueError("ADD-PINNs NPZ file is missing 'x' coordinate array.")

    x_pimoe = get_arr(data_pimoe, 'x')
    x_pinn = get_arr(data_pinn, 'x')
    x_apinn = get_arr(data_apinn, 'x')

    if x_pimoe is None:
        x_pimoe = x_exact
    if x_pinn is None:
        x_pinn = x_exact
    if x_apinn is None:
        x_apinn = x_exact

    panels = [
        ('Displacement', 'u_NN', 'u_lab', '$u$ (m)', '(a)'),
        ('Rotation', 'theta_NN', 'theta_lab', r'$\theta$ (rad)', '(b)'),
        ('Curvature', 'kappa_NN', 'kappa_lab', r'$\kappa$ (1/m)', '(c)'),
        ('Bending Moment', 'M_NN', 'M_lab', '$M$ (N$\\cdot$m)', '(d)'),
        ('Shear Force', 'V_NN', 'V_lab', '$V$ (N)', '(e)'),
        ('Stiffness', 'EI_pred', 'EI_true', '$EI$ (N$\\cdot$m$^2$)', '(f)'),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(FIG_WIDTH_INCH, FIG_HEIGHT_INCH), constrained_layout=True)
    axs = axs.ravel()

    for ax, (title, pred_key, exact_key, ylabel, tag) in zip(axs, panels):
        y_exact = get_arr(data_pimoe, exact_key)
        if y_exact is None:
            y_exact = get_arr(data_pinn, exact_key)
        if y_exact is None:
            y_exact = get_arr(data_apinn, exact_key)

        y_pimoe = get_arr(data_pimoe, pred_key)
        y_pinn = get_arr(data_pinn, pred_key)
        y_apinn = get_arr(data_apinn, pred_key)

        if y_exact is not None:
            ax.plot(
                x_exact,
                y_exact,
                color=C_EXACT,
                linestyle='-',
                linewidth=1.8,
                alpha=0.95,
                label='Exact',
                zorder=1,
            )
        if y_pimoe is not None:
            pimoe_kwargs = dict(
                color=C_PIMOE,
                linestyle=(0, (5, 1.6)),
                linewidth=1.6,
                label='ADD-PINNs',
                zorder=2,
            )
            ax.plot(x_pimoe, y_pimoe, **pimoe_kwargs)
        if y_pinn is not None:
            pinn_kwargs = dict(
                color=C_PINNs,
                linestyle=(0, (6, 1.8, 1.6, 1.8)),
                linewidth=1.6,
                label='PINNs',
                zorder=3,
            )
            ax.plot(x_pinn, y_pinn, **pinn_kwargs)
        if y_apinn is not None:
            apinn_kwargs = dict(
                color=C_APINNs,
                linestyle=(0, (1.2, 1.8)),
                linewidth=1.8,
                label='APINNs',
                zorder=4,
            )
            ax.plot(x_apinn, y_apinn, **apinn_kwargs)

        for vx in (TRUE_JUMPS or []):
            ax.axvline(vx, color=C_JUMP, linestyle=':', linewidth=0.8, zorder=0)

        ax.axhline(0, color='#cccccc', linestyle='-', linewidth=0.5, zorder=0)

        ax.text(-0.08, 1.05, tag, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left')
        ax.set_xlabel('$x$ (m)', labelpad=2)
        ax.set_ylabel(ylabel, labelpad=2)
        ax.grid(True, linestyle=':', linewidth=0.5, color='#d9d9d9')

        if tag == '(b)':
            x_zoom_min, x_zoom_max = 0.57, 0.63

            zoom_segments = []
            for xv, yv in (
                (x_exact, y_exact),
                (x_pimoe, y_pimoe),
                (x_pinn, y_pinn),
                (x_apinn, y_apinn),
            ):
                if xv is None or yv is None:
                    continue
                mask = (xv >= x_zoom_min) & (xv <= x_zoom_max)
                if np.any(mask):
                    zoom_segments.append(yv[mask])

            if zoom_segments:
                y_zoom = np.concatenate(zoom_segments)
                y_zoom_min = float(np.min(y_zoom))
                y_zoom_max = float(np.max(y_zoom))
                y_pad = max(1e-5, (y_zoom_max - y_zoom_min) * 0.35)

                x0, y0, w, h = INSET_BBOX['(b)']
                axins = inset_axes(
                    ax,
                    width='100%',
                    height='100%',
                    loc='lower left',
                    bbox_to_anchor=(x0, y0, w, h),
                    bbox_transform=ax.transAxes,
                    borderpad=0.0,
                )

                if y_exact is not None:
                    axins.plot(x_exact, y_exact, color=C_EXACT, linestyle='-', linewidth=1.2, alpha=0.95)
                if y_pimoe is not None:
                    axins.plot(x_pimoe, y_pimoe, color=C_PIMOE, linestyle=(0, (5, 1.6)), linewidth=1.1)
                if y_pinn is not None:
                    axins.plot(x_pinn, y_pinn, color=C_PINNs, linestyle=(0, (6, 1.8, 1.6, 1.8)), linewidth=1.1)
                if y_apinn is not None:
                    axins.plot(x_apinn, y_apinn, color=C_APINNs, linestyle=(0, (1.2, 1.8)), linewidth=1.2)

                axins.set_xlim(x_zoom_min, x_zoom_max)
                axins.set_ylim(y_zoom_min - y_pad, y_zoom_max + y_pad)
                axins.set_facecolor('#fbfbfb')
                axins.grid(True, linestyle=':', linewidth=0.35, color='#d9d9d9')
                axins.tick_params(axis='both', labelsize=4, length=2, pad=1)
                axins.set_xticks([0.58, 0.62])

                for spine in axins.spines.values():
                    spine.set_linewidth(0.6)

                mark_inset(ax, axins, loc1=1, loc2=2, fc='none', ec='#666666', lw=0.65, ls='--')

        if tag == '(c)':
            x_zoom_min, x_zoom_max = 0.57, 0.63

            zoom_segments = []
            for xv, yv in (
                (x_exact, y_exact),
                (x_pimoe, y_pimoe),
                (x_pinn, y_pinn),
                (x_apinn, y_apinn),
            ):
                if xv is None or yv is None:
                    continue
                mask = (xv >= x_zoom_min) & (xv <= x_zoom_max)
                if np.any(mask):
                    zoom_segments.append(yv[mask])

            if zoom_segments:
                y_zoom = np.concatenate(zoom_segments)
                y_zoom_min = float(np.min(y_zoom))
                y_zoom_max = float(np.max(y_zoom))
                y_pad = max(1e-5, (y_zoom_max - y_zoom_min) * 0.35)

                x0, y0, w, h = INSET_BBOX['(c)']
                axins = inset_axes(
                    ax,
                    width='100%',
                    height='100%',
                    loc='lower left',
                    bbox_to_anchor=(x0, y0, w, h),
                    bbox_transform=ax.transAxes,
                    borderpad=0.0,
                )

                if y_exact is not None:
                    axins.plot(x_exact, y_exact, color=C_EXACT, linestyle='-', linewidth=1.2, alpha=0.95)
                if y_pimoe is not None:
                    axins.plot(x_pimoe, y_pimoe, color=C_PIMOE, linestyle=(0, (5, 1.6)), linewidth=1.1)
                if y_pinn is not None:
                    axins.plot(x_pinn, y_pinn, color=C_PINNs, linestyle=(0, (6, 1.8, 1.6, 1.8)), linewidth=1.1)
                if y_apinn is not None:
                    axins.plot(x_apinn, y_apinn, color=C_APINNs, linestyle=(0, (1.2, 1.8)), linewidth=1.2)

                axins.set_xlim(x_zoom_min, x_zoom_max)
                axins.set_ylim(y_zoom_min - y_pad, y_zoom_max + y_pad)
                axins.set_facecolor('#fbfbfb')
                axins.grid(True, linestyle=':', linewidth=0.35, color='#d9d9d9')
                axins.tick_params(axis='both', labelsize=4, length=2, pad=1)
                axins.set_xticks([0.58, 0.62])

                for spine in axins.spines.values():
                    spine.set_linewidth(0.6)

                mark_inset(ax, axins, loc1=1, loc2=2, fc='none', ec='#666666', lw=0.65, ls='--')

        if tag == '(f)':
            y_vals = []
            if y_exact is not None:
                y_vals.extend(y_exact)
            if y_pimoe is not None:
                y_vals.extend(y_pimoe)
            if y_pinn is not None:
                y_vals.extend(y_pinn)
            if y_apinn is not None:
                y_vals.extend(y_apinn)
            if y_vals:
                y_min, y_max = min(y_vals), max(y_vals)
                margin = (y_max - y_min) * 0.2 if y_max > y_min else 0.2
                ax.set_ylim(max(0, y_min - margin), y_max + margin)

        if tag == '(a)':
            ax.legend(
                loc='best',
                bbox_to_anchor=(0.03, 0.0, 1.0, 1.0),
                bbox_transform=ax.transAxes,
                frameon=False,
                handlelength=2.6,
                ncol=2,
                columnspacing=1.2,
            )
        elif tag == '(b)':
            ax.legend(loc='upper right', frameon=False, handlelength=2.6, ncol=1, columnspacing=1.2)
        elif tag == '(c)':
            ax.legend(loc='upper right', frameon=False, handlelength=2.6, ncol=2, columnspacing=1.2)
        else:
            ax.legend(loc='best', frameon=False, handlelength=2.6, ncol=2, columnspacing=1.2)

        ax.margins(x=0.02, y=0.1)

    if not output_path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(OUTPUT_DIR / 'Fig6.pdf')
        png_path = str(OUTPUT_DIR / 'Fig6.png')
    else:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        base, _ext = os.path.splitext(output_path)
        png_path = f'{base}.png'

    print(f'Saving PDF to: {output_path}')
    plt.savefig(output_path, dpi=300, format='pdf')

    print(f'Saving PNG to: {png_path}')
    plt.savefig(png_path, dpi=300, format='png')

    if show:
        plt.show()
    else:
        plt.close(fig)


# ==========================================
# 4. Script Entry Point
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Nature-style figure from NPZ data')
    parser.add_argument('--npz', type=str, default='', help='Path to ADD-PINNs phi_*.npz file')
    parser.add_argument('--npz-pinn', type=str, default='', help='Path to PINNs phi_*.npz file')
    parser.add_argument('--npz-apinn', type=str, default='', help='Path to APINNs phi_*.npz file')
    parser.add_argument('--out', type=str, default='', help='Output file path (PDF)')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')

    args = parser.parse_args()

    try:
        plot_nature_figure(args.npz, args.npz_pinn, args.npz_apinn, args.out, args.show)
    except Exception as e:
        print(f'Error: {e}')
