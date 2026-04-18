import os

import glob

import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import rcParams

try:
    from config import EI_REAL
except Exception:
    # Fallback keeps plotting script runnable when training deps (e.g., torch) are unavailable.
    EI_REAL = 1562.5



# ==========================================

# 1. 配置路径与参数

# ==========================================



BASE_DIR = os.path.dirname(os.path.abspath(__file__))



# 默认路径

DEFAULT_LOSS_CSV = os.path.join(BASE_DIR, "loss_list_global.csv")
DEFAULT_LOSS_CSV_PIMOE = os.path.join(BASE_DIR, "data_output", "loss_list_global.csv")
DEFAULT_LOSS_CSV_PINNs = os.path.join(BASE_DIR, "data_output_pinn", "loss_list_global.csv")
DEFAULT_LOSS_CSV_RD_PINNs = os.path.join(BASE_DIR, "data_output_reduced", "loss_list_global.csv")
DEFAULT_LOSS_CSV_APINNs = os.path.join(BASE_DIR, "data_output_apinn", "loss_list_global.csv")

DEFAULT_PHI_NPZ_DIR = os.path.join(BASE_DIR, "beam_bechmark_viz")



PATH_LOSS_CSV = DEFAULT_LOSS_CSV
PATH_LOSS_CSV_PIMOE = DEFAULT_LOSS_CSV_PIMOE if os.path.exists(DEFAULT_LOSS_CSV_PIMOE) else DEFAULT_LOSS_CSV
PATH_LOSS_CSV_PINNs = DEFAULT_LOSS_CSV_PINNs
PATH_LOSS_CSV_RD_PINNs = DEFAULT_LOSS_CSV_RD_PINNs
PATH_LOSS_CSV_APINNs = DEFAULT_LOSS_CSV_APINNs

DIR_PHI_NPZ = DEFAULT_PHI_NPZ_DIR



# ---------------------------------------------------------

# CSV 列名映射

# ---------------------------------------------------------

COL_NAMES = {

    'epoch': 'epoch',

    'loss_data': 'data',       # 数据项

    'loss_interface': 'interface', # 接口 Loss

   

    # PDE 分量 (将被加和)

    'loss_M': 'M',            

    'loss_V': 'V',            

    'loss_Q': 'Q',            

   

    # EI 列 (用于轨迹图，若无则自动跳过)

    'EI1': 'EI_1_missing',            

    'EI2': 'EI_2_missing',            

    'EI3': 'EI_3_missing'              

}



# 输出路径

OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "Figure")

OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Fig5.pdf")

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Fig5.png")



# ==========================================

# 2. Nature 风格设置

# ==========================================

rcParams['font.family'] = 'sans-serif'

rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

rcParams['mathtext.fontset'] = 'stixsans'

rcParams['font.size'] = 7

rcParams['axes.labelsize'] = 7

rcParams['axes.titlesize'] = 7

rcParams['xtick.labelsize'] = 6

rcParams['ytick.labelsize'] = 6

rcParams['legend.fontsize'] = 6

rcParams['axes.linewidth'] = 0.6

rcParams['grid.linewidth'] = 0.5

rcParams['lines.linewidth'] = 1.2



C_EXACT = '#404040'  

C_PRED  = '#D55E00'   # Interface Loss 颜色

C_AUTO  = '#0072B2'   # Data Loss 颜色

C_RD_PINNs = '#56B4E9' # Reduced PINNs 颜色

C_PDE   = '#E69F00'   # [新] PDE Sum Loss 颜色 (橙色)

C_GT_LINE = '#009E73'

COLORS_EI = ['#56B4E9', '#F0E442', '#CC79A7']



FIG_WIDTH = 7.08

FIG_HEIGHT = 3.6



# ==========================================

# 3. 数据加载函数

# ==========================================



def load_loss_and_params(csv_path):

    print(f"[Info] Loading CSV from: {csv_path}")

    if not os.path.exists(csv_path):

        raise FileNotFoundError(f"CSV file not found: {csv_path}")

   

    df = pd.read_csv(csv_path)

    df.columns = [c.strip() for c in df.columns]

    return df


def get_epoch_total_from_df(df):
    """Extract epoch and unweighted PDE+data loss series from CSV dataframe."""
    epochs = df.get(COL_NAMES['epoch'], df.index).to_numpy()

    # Use only unweighted data + PDE-related terms.
    candidate_cols = ['data', 'fai', 'dfai', 'M', 'V', 'Q']
    available_cols = [col for col in candidate_cols if col in df.columns]

    if available_cols:
        total = np.zeros(len(df), dtype=float)
        for col in available_cols:
            total += pd.to_numeric(df[col], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    elif 'total' in df.columns:
        # Legacy fallback when required raw columns are not available.
        print("[Warning] PDE+data columns not found; fallback to weighted 'total' column.")
        total = pd.to_numeric(df['total'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    else:
        raise ValueError("No usable loss columns found in CSV.")

    return epochs, total



def load_phi_history(npz_dir):

    print(f"[Info] Scanning NPZ files in: {npz_dir}")

    if not os.path.exists(npz_dir):

        raise FileNotFoundError(f"Directory not found: {npz_dir}")

       

    pattern = os.path.join(npz_dir, "phi_*.npz")

    files = glob.glob(pattern)

    if not files:

        pattern = os.path.join(npz_dir, "**", "phi_*.npz")

        files = glob.glob(pattern, recursive=True)

    if not files:

        raise FileNotFoundError("No phi_*.npz files found.")



    def extract_epoch(fpath):

        match = re.search(r'phi_(\d+)', fpath)

        return int(match.group(1)) if match else 0

   

    files = sorted(files, key=extract_epoch)

   

    epochs = []

    phi_matrix = []

    x_coords = None

   

    for f in files:

        ep = extract_epoch(f)

        try:

            data = np.load(f)

            if 'phi' in data: phi_vals = data['phi'].flatten()

            elif 'u' in data: phi_vals = data['u'].flatten()

            else: continue



            if x_coords is None:

                if 'x' in data: x_coords = data['x'].flatten()

                else: x_coords = np.linspace(0, 1, len(phi_vals))

           

            epochs.append(ep)

            phi_matrix.append(phi_vals)

        except Exception as e:

            print(f"[Warning] Failed to load {f}: {e}")



    phi_matrix = np.array(phi_matrix).T

    return np.array(epochs), x_coords, phi_matrix





def load_ei_history(npz_path_or_dir, key="EI_pred"):

    """

    Load EI history from phi_*.npz snapshots.

    Returns epochs, x_coords, ei_matrix (T, N).

    """

    if not os.path.exists(npz_path_or_dir):

        raise FileNotFoundError(f"Path not found: {npz_path_or_dir}")



    files = []

    if os.path.isfile(npz_path_or_dir):

        files = [npz_path_or_dir]

    else:

        pattern = os.path.join(npz_path_or_dir, "phi_*.npz")

        files = glob.glob(pattern)

        if not files:

            pattern = os.path.join(npz_path_or_dir, "**", "phi_*.npz")

            files = glob.glob(pattern, recursive=True)

    if not files:

        raise FileNotFoundError("No phi_*.npz files found for EI history.")



    def extract_epoch(fpath):

        match = re.search(r'phi_(\d+)', fpath)

        return int(match.group(1)) if match else None



    files = sorted(files, key=lambda f: extract_epoch(f) or 0)



    epochs = []

    ei_matrix = []

    x_coords = None



    for idx, f in enumerate(files):

        ep = extract_epoch(f)

        try:

            data = np.load(f)

            if key not in data:

                continue

            ei_vals = data[key].flatten()

            if x_coords is None:

                if 'x' in data:

                    x_coords = data['x'].flatten()

                else:

                    x_coords = np.linspace(0, 1, len(ei_vals))

            epochs.append(ep if ep is not None else idx)

            ei_matrix.append(ei_vals)

        except Exception as e:

            print(f"[Warning] Failed to load {f}: {e}")



    if not ei_matrix:

        raise ValueError(f"No '{key}' found in npz files.")



    ei_matrix = np.array(ei_matrix)  # shape (T, N)

    return np.array(epochs), x_coords, ei_matrix



# ==========================================

# 4. 主绘图逻辑

# ==========================================



def plot_fig2_real_data():

    # --- Step 1: 加载数据 ---

    df = load_loss_and_params(PATH_LOSS_CSV)
    df_pimoe = load_loss_and_params(PATH_LOSS_CSV_PIMOE)
    df_pinn = load_loss_and_params(PATH_LOSS_CSV_PINNs)
    df_rd_pinn = load_loss_and_params(PATH_LOSS_CSV_RD_PINNs)
    df_apinn = load_loss_and_params(PATH_LOSS_CSV_APINNs)

    epochs_pimoe, total_pimoe = get_epoch_total_from_df(df_pimoe)
    epochs_pinn, total_pinn = get_epoch_total_from_df(df_pinn)
    epochs_rd_pinn, total_rd_pinn = get_epoch_total_from_df(df_rd_pinn)
    epochs_apinn, total_apinn = get_epoch_total_from_df(df_apinn)

    heatmap_epochs, x_space, phi_matrix = load_phi_history(DIR_PHI_NPZ)

   

    # EI data: prefer npz snapshots, fallback to CSV columns if available.

    try:

        ei_epochs, x_ei, ei_matrix = load_ei_history(DIR_PHI_NPZ, key="EI_pred")

        if x_ei is None or ei_matrix.size == 0:

            raise ValueError("EI data missing in npz")

        segments = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]

        masks = [

            (x_ei >= seg[0]) & (x_ei < seg[1] if i < 2 else x_ei <= seg[1])

            for i, seg in enumerate(segments)

        ]

        ei_series = []

        for mask in masks:

            if not mask.any():

                ei_series.append(np.full(ei_matrix.shape[0], np.nan))

            else:

                ei_series.append(np.nanmean(ei_matrix[:, mask], axis=1))

        ei1, ei2, ei3 = [s * EI_REAL for s in ei_series]

    except Exception as e:

        print(f"[Warning] EI history from npz failed: {e}")

        ei_epochs = epochs_pimoe

        ei1 = df.get(COL_NAMES['EI1'], None)

        ei2 = df.get(COL_NAMES['EI2'], None)

        ei3 = df.get(COL_NAMES['EI3'], None)



    # --- Step 2: 绘图 ---

    fig, axs = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)

   

    # -------------------------------------------------------

    # Panel (a): Schematic

    # -------------------------------------------------------

    ax_schem = axs[0, 0]

    ax_schem.set_title("Problem Setup", fontweight='normal', pad=4)

    ax_schem.text(-0.15, 1.05, '(a)', transform=ax_schem.transAxes, fontsize=8, fontweight='bold')

   

    colors_seg = ['#56B4E9', '#F0E442', '#CC79A7']

    segments = [(0.0, 0.3, 4166), (0.3, 0.6, 3120), (0.6, 1.0, 5208)]

    for i, (start, end, val) in enumerate(segments):

        rect = plt.Rectangle((start, -0.2), end-start, 0.4,

                             facecolor=colors_seg[i], edgecolor='none', alpha=0.8)

        ax_schem.add_patch(rect)

        ax_schem.text((start+end)/2, 0, f"$EI_{i+1}$", ha='center', va='center', fontsize=7)

        ax_schem.text((start+end)/2, -0.35, f"{val}", ha='center', va='top', fontsize=6, color='#555')



    ax_schem.plot([0, 0], [-0.3, 0.3], color='black', linewidth=2)

    for y in np.linspace(-0.3, 0.3, 6):

        ax_schem.plot([0, -0.05], [y, y-0.05], color='black', linewidth=0.8)

    ax_schem.plot([1], [-0.2], marker='^', color='black', markersize=6)



    ax_schem.set_xlim(-0.1, 1.1)

    ax_schem.set_ylim(-0.6, 0.6)

    ax_schem.axis('off')

   

    ax_schem.axvline(0.3, ymin=0.3, ymax=0.7, color=C_EXACT, linestyle=':', linewidth=0.8)

    ax_schem.axvline(0.6, ymin=0.3, ymax=0.7, color=C_EXACT, linestyle=':', linewidth=0.8)



    # -------------------------------------------------------

    # Panel (b): Loss History (Total Loss of 4 Models)

    # -------------------------------------------------------

    ax_loss = axs[0, 1]

    ax_loss.set_title("Training Convergence", fontweight='normal', pad=4)

    ax_loss.text(-0.15, 1.05, '(b)', transform=ax_loss.transAxes, fontsize=8, fontweight='bold')

   

    valid_pimoe = total_pimoe > 0
    valid_pinn = total_pinn > 0
    valid_rd_pinn = total_rd_pinn > 0
    valid_apinn = total_apinn > 0

    if np.any(valid_pimoe):
        ax_loss.semilogy(
            epochs_pimoe[valid_pimoe],
            total_pimoe[valid_pimoe],
            color=C_PRED,
            label='ADD-PINNs',
            linewidth=1.3,
            linestyle='-',
        )

    if np.any(valid_pinn):
        ax_loss.semilogy(
            epochs_pinn[valid_pinn],
            total_pinn[valid_pinn],
            color=C_AUTO,
            label='PINNs',
            linewidth=1.3,
            linestyle='--',
        )

    if np.any(valid_rd_pinn):
        ax_loss.semilogy(
            epochs_rd_pinn[valid_rd_pinn],
            total_rd_pinn[valid_rd_pinn],
            color=C_RD_PINNs,
            label='RD-PINNs',
            linewidth=1.3,
            linestyle=':',
        )

    if np.any(valid_apinn):
        ax_loss.semilogy(
            epochs_apinn[valid_apinn],
            total_apinn[valid_apinn],
            color=C_GT_LINE,
            label='APINNs',
            linewidth=1.3,
            linestyle='-.',
        )

   

    ax_loss.set_xlabel("Epochs")

    ax_loss.set_ylabel("PDE+Data Loss")

    ax_loss.grid(True, which='both', linestyle=':', linewidth=0.4, alpha=0.5)

   

    # 简洁图例

    ax_loss.legend(frameon=False, loc='upper right', ncol=2)

    max_epoch = max(
        np.max(epochs_pimoe),
        np.max(epochs_pinn),
        np.max(epochs_rd_pinn),
        np.max(epochs_apinn),
    )
    ax_loss.set_xlim(left=0, right=max_epoch)



    # -------------------------------------------------------

    # Panel (c): Interface Evolution

    # -------------------------------------------------------

    ax_evol = axs[1, 0]

    ax_evol.set_title(r"Interface Evolution ($\phi=0$)", fontweight='normal', pad=4)

    ax_evol.text(-0.15, 1.05, '(c)', transform=ax_evol.transAxes, fontsize=8, fontweight='bold')



    t_min, t_max = heatmap_epochs[0], heatmap_epochs[-1]

    x_min, x_max = x_space[0], x_space[-1]

   

    v_abs = np.percentile(np.abs(phi_matrix), 90)

    im = ax_evol.imshow(phi_matrix, aspect='auto', cmap='RdBu_r',

                        extent=[t_min, t_max, x_min, x_max],

                        origin='lower', vmin=-v_abs, vmax=v_abs)



    ax_evol.contour(heatmap_epochs, x_space, phi_matrix, levels=[0], colors='black', linewidths=1.2)

    ax_evol.axhline(0.3, color=C_GT_LINE, linestyle='--', linewidth=1, label='True')

    ax_evol.axhline(0.6, color=C_GT_LINE, linestyle='--', linewidth=1)



    ax_evol.set_xlabel("Epochs")

    ax_evol.set_ylabel("$x$ (m)")

    ax_evol.set_xlim(left=0, right=t_max)

   

    cbar = plt.colorbar(im, ax=ax_evol, fraction=0.046, pad=0.04)

    cbar.set_label(r'$\phi(x)$', rotation=270, labelpad=8)

    cbar.ax.tick_params(labelsize=5)

    cbar.outline.set_linewidth(0.5)



    # -------------------------------------------------------

    # Panel (d): Parameter Trajectory

    # -------------------------------------------------------

    ax_param = axs[1, 1]

    ax_param.set_title("Parameter Identification", fontweight='normal', pad=4)

    ax_param.text(-0.15, 1.05, '(d)', transform=ax_param.transAxes, fontsize=8, fontweight='bold')



    ax_param.axhline(4166, color=C_EXACT, linestyle='--', linewidth=0.8, alpha=0.6)

    ax_param.axhline(3120, color=C_EXACT, linestyle='--', linewidth=0.8, alpha=0.6)

    ax_param.axhline(5208, color=C_EXACT, linestyle='--', linewidth=0.8, alpha=0.6)



    if ei1 is not None and ei2 is not None and ei3 is not None:

        ax_param.plot(ei_epochs, ei1, color=colors_seg[0], label='$EI_1$', linewidth=1.5)

        ax_param.plot(ei_epochs, ei2, color=colors_seg[1], label='$EI_2$', linewidth=1.5)

        ax_param.plot(ei_epochs, ei3, color=colors_seg[2], label='$EI_3$', linewidth=1.5)

       

        y_vals = np.concatenate([ei1, ei2, ei3])

        y_stable = y_vals[int(len(y_vals)*0.1):] if len(y_vals) > 10 else y_vals

        if len(y_stable) > 0:

            y_min, y_max = np.min(y_stable), np.max(y_stable)

            margin = (y_max - y_min) * 0.3

            ax_param.set_ylim(y_min - margin, y_max + margin)

           

        ax_param.legend(frameon=False, loc='lower right')

    else:

        ax_param.text(0.5, 0.5, "Trajectory Data Not Found",

                      ha='center', va='center', fontsize=7, color='gray')



    ax_param.set_xlabel("Epochs")

    ax_param.set_ylabel(r"$EI$ (N$\cdot$m$^2$)")

    ax_param.grid(True, linestyle=':', linewidth=0.5)

    ax_param.set_xlim(left=0, right=max(ei_epochs))



    # --- 保存 ---

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[Info] Saving to {OUTPUT_PDF} ...")

    plt.savefig(OUTPUT_PDF, dpi=300)

    plt.savefig(OUTPUT_PNG, dpi=300)

    print("[Info] Done.")



if __name__ == "__main__":

    try:

        plot_fig2_real_data()

    except Exception as e:

        print(f"[Error] {e}")
