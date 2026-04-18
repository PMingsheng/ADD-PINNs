#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["mathtext.fontset"] = "stixsans"
rcParams["font.size"] = 7
rcParams["axes.labelsize"] = 7
rcParams["axes.titlesize"] = 8
rcParams["xtick.labelsize"] = 6
rcParams["ytick.labelsize"] = 6
rcParams["legend.fontsize"] = 6
rcParams["axes.linewidth"] = 0.6
rcParams["grid.linewidth"] = 0.5
rcParams["lines.linewidth"] = 1.2

FIG_WIDTH = 7.08
FIG_HEIGHT = 2.6 * 5.0 / 6.0
XMAX_EPOCH = 10000.0
APINNs_PRETRAIN_EPOCHS = 1000.0

C_PIMOE = "#D55E00"
C_APINNs = "#009E73"
C_PINNs = "#0072B2"

LOSS_PATHS = {
    "ADD-PINNs": ("outputs_add_pinns3d_c1_sphere/loss_list_global.csv", "outputs_add_pinns3d_c1_sphere/data_output/loss_list_global.csv"),
    "APINNs": ("outputs_apinn3d_c1_sphere/loss_list_global.csv", "outputs_apinn3d_c1_sphere/data_output/loss_list_global.csv"),
    "PINNs": ("outputs_pinn3d_c1_sphere/loss_list_global.csv", "outputs_pinn3d_c1_sphere/data_output/loss_list_global.csv"),
}

MODEL_ORDER = ("ADD-PINNs", "APINNs", "PINNs")
MODEL_COLORS = {"ADD-PINNs": C_PIMOE, "APINNs": C_APINNs, "PINNs": C_PINNs}


def _resolve_existing_path(script_dir: Path, rel_paths: Tuple[str, ...]) -> Path:
    tried = []
    for rel_path in rel_paths:
        path = script_dir / rel_path
        tried.append(str(path))
        if path.exists():
            return path
    raise FileNotFoundError("File not found. Tried:\n" + "\n".join(tried))


def _load_csv_columns(csv_path: Path) -> Dict[str, np.ndarray]:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return {str(name).strip(): data[:, i] for i, name in enumerate(header)}


def _shift_apinn_epochs_if_needed(model_name: str, epochs: np.ndarray) -> np.ndarray:
    epochs = np.asarray(epochs, dtype=np.float64)
    if model_name == "APINNs":
        # New APINNs csv already records total-budget epochs.
        if epochs.size > 0 and float(np.max(epochs)) <= XMAX_EPOCH - APINNs_PRETRAIN_EPOCHS:
            return epochs + APINNs_PRETRAIN_EPOCHS
    return epochs


def _extract_comparable_loss(cols: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    epochs = np.asarray(cols.get("epoch", np.arange(len(next(iter(cols.values()))))), dtype=np.float64)
    total = np.asarray(cols.get("data_raw", cols.get("data", 0.0)), dtype=np.float64) + np.asarray(
        cols.get("pde_raw", cols.get("pde", 0.0)), dtype=np.float64
    )
    valid = np.isfinite(epochs) & np.isfinite(total) & (total > 0.0)
    return epochs[valid], total[valid]


def _add_panel_label(ax, tag: str) -> None:
    ax.text(
        -0.16,
        1.01,
        tag,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _plot_loss_compare(ax, script_dir: Path) -> None:
    style_map = {"ADD-PINNs": "-", "APINNs": "-.", "PINNs": "--"}
    for model_name in MODEL_ORDER:
        cols = _load_csv_columns(_resolve_existing_path(script_dir, LOSS_PATHS[model_name]))
        epochs, total = _extract_comparable_loss(cols)
        if epochs.size == 0:
            continue
        epochs = _shift_apinn_epochs_if_needed(model_name, epochs)
        ax.semilogy(
            epochs,
            total,
            color=MODEL_COLORS[model_name],
            linestyle=style_map[model_name],
            linewidth=1.3,
            label=model_name,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Data+PDE Loss")
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_xlim(0.0, XMAX_EPOCH)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 3.0)
    ax.legend(frameon=False, loc="upper left", ncol=3, handlelength=2.4)


def _plot_data_loss(ax, script_dir: Path) -> None:
    style_map = {"ADD-PINNs": "-", "APINNs": "-.", "PINNs": "--"}
    for model_name in MODEL_ORDER:
        cols = _load_csv_columns(_resolve_existing_path(script_dir, LOSS_PATHS[model_name]))
        epochs = np.asarray(cols.get("epoch", np.arange(len(next(iter(cols.values()))))), dtype=np.float64)
        data = np.asarray(cols.get("data_raw", cols.get("data", 0.0)), dtype=np.float64)
        valid = np.isfinite(epochs) & np.isfinite(data) & (data > 0.0)
        epochs = epochs[valid]
        data = data[valid]
        if epochs.size == 0:
            continue
        epochs = _shift_apinn_epochs_if_needed(model_name, epochs)
        ax.semilogy(
            epochs,
            data,
            color=MODEL_COLORS[model_name],
            linestyle=style_map[model_name],
            linewidth=1.3,
            label=model_name,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Data Loss")
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_xlim(0.0, XMAX_EPOCH)
    ax.legend(frameon=False, loc="lower left", ncol=1, handlelength=2.4)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir.parent / "Figure"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=1 / 72, h_pad=2 / 72, wspace=0.08, hspace=0.02)

    _plot_loss_compare(axes[0], script_dir)
    _add_panel_label(axes[0], "(a)")
    _plot_data_loss(axes[1], script_dir)
    _add_panel_label(axes[1], "(b)")

    out_png = output_dir / "Fig13.png"
    out_pdf = output_dir / "Fig13.pdf"
    fig.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
