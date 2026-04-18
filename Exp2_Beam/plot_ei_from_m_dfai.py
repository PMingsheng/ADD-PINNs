"""
Plot EI = M / dfai (kappa) from a saved phi_*.npz snapshot.
"""
import argparse
import glob
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import TRUE_JUMPS


def get_arr(data: dict, key: str):
    if key not in data:
        return None
    arr = data[key]
    if arr.size == 0:
        return None
    return np.asarray(arr).squeeze()


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_NPZ_PATH = str(BASE_DIR / "data_output" / "phi_final.npz")
OUTPUT_DIR = BASE_DIR / "figures"


def resolve_npz_path(raw_path: str) -> str:
    raw_path = raw_path or DEFAULT_NPZ_PATH
    if raw_path and os.path.exists(raw_path):
        return raw_path
    if raw_path:
        local_path = str(BASE_DIR / raw_path)
        if os.path.exists(local_path):
            return local_path
        if " " in raw_path:
            for part in raw_path.split():
                if os.path.exists(part):
                    return part
    candidates = glob.glob(str(BASE_DIR / "beam_bechmark_viz" / "phi_*.npz"))
    if not candidates:
        raise FileNotFoundError("no phi_*.npz found; pass the path explicitly")
    return max(candidates, key=os.path.getmtime)


def compute_ei(m: np.ndarray, kappa: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    m = np.asarray(m, dtype=float).squeeze()
    kappa = np.asarray(kappa, dtype=float).squeeze()
    out = np.full_like(m, np.nan, dtype=float)
    mask = np.isfinite(m) & np.isfinite(kappa) & (np.abs(kappa) > eps)
    out[mask] = m[mask] / kappa[mask]
    return out


def plot_from_npz(npz_path: str, output_path: str = "", show: bool = False) -> None:
    npz_path = resolve_npz_path(npz_path)
    data = np.load(npz_path)
    x = get_arr(data, "x")
    if x is None:
        raise ValueError("npz missing 'x' array")

    ei_true = get_arr(data, "EI_true")

    m_sources = [
        ("M_NN", get_arr(data, "M_NN")),
        ("M_AUTO", get_arr(data, "M_AUTO")),
        ("M_lab", get_arr(data, "M_lab")),
    ]
    k_sources = [
        ("kappa_NN", get_arr(data, "kappa_NN")),
        ("kappa_AUTO", get_arr(data, "kappa_AUTO")),
        ("kappa_lab", get_arr(data, "kappa_lab")),
    ]

    combos = []
    for m_name, m_arr in m_sources:
        if m_arr is None:
            continue
        for k_name, k_arr in k_sources:
            if k_arr is None:
                continue
            combos.append((f"{m_name}/{k_name}", m_arr, k_arr))

    if not combos:
        raise ValueError("no valid M or dfai (kappa) arrays found in npz")

    nplots = len(combos) + 1
    ncols = 3
    nrows = math.ceil(nplots / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axs = np.atleast_1d(axs).ravel()

    for i, (label, m_arr, k_arr) in enumerate(combos):
        ax = axs[i]
        ei = compute_ei(m_arr, k_arr)
        ax.plot(x, ei, lw=1.0, label=label)
        if ei_true is not None:
            ax.plot(x, ei_true, "k--", lw=1.0, label="EI_true")
        for vx in (TRUE_JUMPS or []):
            ax.axvline(vx, color="r", ls=":", lw=0.8)
        ax.set_ylim(1.5,3.5)
        ax.set_xlabel("x")
        ax.set_ylabel("EI")
        ax.set_title(label)
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(fontsize=7)

    ax_sum = axs[len(combos)]
    for label, m_arr, k_arr in combos:
        ei = compute_ei(m_arr, k_arr)
        ax_sum.plot(x, ei, lw=0.8, label=label)
    if ei_true is not None:
        ax_sum.plot(x, ei_true, "k--", lw=1.2, label="EI_true")
    for vx in (TRUE_JUMPS or []):
        ax_sum.axvline(vx, color="r", ls=":", lw=0.8)
    ax_sum.set_ylim(1.5,3.5)
    ax_sum.set_xlabel("x")
    ax_sum.set_ylabel("EI")
    ax_sum.set_title("Summary")
    ax_sum.grid(True, ls="--", alpha=0.3)
    ax_sum.legend(fontsize=6, ncol=2)

    for ax in axs[nplots:]:
        ax.axis("off")

    plt.tight_layout()
    if not output_path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(OUTPUT_DIR / "ei_m_dfai.png")
    else:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    print(f"[plot_ei_from_m_dfai] saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot EI = M/dfai from phi_*.npz")
    parser.add_argument(
        "--npz",
        default="",
        help="Path to phi_*.npz (default: data_output/phi_final.npz)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output PNG path (default: next to npz)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="Show the figure interactively",
    )
    args = parser.parse_args()

    plot_from_npz(args.npz, args.out, args.show)


if __name__ == "__main__":
    main()
