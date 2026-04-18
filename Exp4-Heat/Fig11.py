#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.lines import Line2D


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
XMAX_EPOCH = 80000.0

C_PIMOE = "#D55E00"
C_APINNs = "#009E73"
C_PINNs = "#0072B2"
C_ROI_ON = "#CC79A7"
C_EXACT = "#666666"

F1_TRUE = 10.0
F2_TRUE = 5.0
APINNs_PRETRAIN_EPOCHS = 1000.0

LOSS_PATHS = {
    "ADD-PINNs": (
        "output_roi_off/data_output/loss_list_global.csv",
        "output_roi_off/loss_list_global.csv",
    ),
    "ADD-PINNs (roi-on)": (
        "output_roi_on/data_output/loss_list_global.csv",
        "output_roi_on/loss_list_global.csv",
    ),
    "APINNs": ("outputs_apinn/loss.csv",),
    "PINNs": ("outputs_pinn_single/loss.csv",),
}

PARAM_PATHS = {
    "ADD-PINNs": (
        "output_roi_off/data_output/loss_list_global.csv",
        "output_roi_off/loss_list_global.csv",
    ),
    "ADD-PINNs (roi-on)": (
        "output_roi_on/data_output/loss_list_global.csv",
        "output_roi_on/loss_list_global.csv",
    ),
    "APINNs": ("outputs_apinn/loss.csv",),
    "PINNs": ("outputs_pinn_single/loss.csv",),
}


def _resolve_existing_path(script_dir: Path, rel_paths: tuple[str, ...]) -> Path:
    tried = []
    for rel_path in rel_paths:
        path = script_dir / rel_path
        tried.append(str(path))
        if path.exists():
            return path
    raise FileNotFoundError("CSV file not found. Tried:\n" + "\n".join(tried))


def _load_csv_columns(csv_path: Path) -> Dict[str, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
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
        return epochs + APINNs_PRETRAIN_EPOCHS
    return epochs


def _extract_comparable_loss(cols: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    epochs = np.asarray(cols.get("epoch", np.arange(len(next(iter(cols.values()))))), dtype=np.float64)
    if "raw_data" in cols:
        total = (
            np.asarray(cols.get("raw_data", 0.0), dtype=np.float64)
            + np.asarray(cols.get("raw_pde", 0.0), dtype=np.float64)
            + np.asarray(cols.get("raw_interface", 0.0), dtype=np.float64)
        )
    else:
        total = (
            np.asarray(cols.get("data", 0.0), dtype=np.float64)
            + np.asarray(cols.get("pde", 0.0), dtype=np.float64)
            + np.asarray(cols.get("interface", 0.0), dtype=np.float64)
        )
    valid = np.isfinite(epochs) & np.isfinite(total) & (total > 0.0)
    return epochs[valid], total[valid]


def _extract_param_series(cols: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs = np.asarray(cols.get("epoch", np.arange(len(next(iter(cols.values()))))), dtype=np.float64)
    f1 = np.asarray(cols.get("f1", np.full_like(epochs, np.nan)), dtype=np.float64)
    f2 = np.asarray(cols.get("f2", np.full_like(epochs, np.nan)), dtype=np.float64)
    valid = np.isfinite(epochs) & np.isfinite(f1) & np.isfinite(f2)
    return epochs[valid], f1[valid], f2[valid]


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
        color="black",
    )


def _plot_loss_compare(ax, script_dir: Path) -> None:
    color_map = {
        "ADD-PINNs": C_PIMOE,
        "ADD-PINNs (roi-on)": C_ROI_ON,
        "APINNs": C_APINNs,
        "PINNs": C_PINNs,
    }
    style_map = {
        "ADD-PINNs": "-",
        "ADD-PINNs (roi-on)": "-",
        "APINNs": "-.",
        "PINNs": "--",
    }

    for model_name, rel_paths in LOSS_PATHS.items():
        cols = _load_csv_columns(_resolve_existing_path(script_dir, rel_paths))
        epochs, total = _extract_comparable_loss(cols)
        if epochs.size == 0:
            continue
        epochs = _shift_apinn_epochs_if_needed(model_name, epochs)
        ax.semilogy(
            epochs,
            total,
            color=color_map[model_name],
            linestyle=style_map[model_name],
            linewidth=1.3,
            label=model_name,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Data+PDE+Interface Loss")
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_xlim(0.0, XMAX_EPOCH)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.62, 0.98),
        ncol=2,
        handlelength=2.4,
    )


def _plot_param_evolution(ax, script_dir: Path) -> None:
    color_map = {
        "ADD-PINNs": C_PIMOE,
        "ADD-PINNs (roi-on)": C_ROI_ON,
        "APINNs": C_APINNs,
        "PINNs": C_PINNs,
    }
    y_values = [F1_TRUE, F2_TRUE]
    model_handles = []
    for model_name, rel_paths in PARAM_PATHS.items():
        cols = _load_csv_columns(_resolve_existing_path(script_dir, rel_paths))
        epochs, f1, f2 = _extract_param_series(cols)
        if epochs.size == 0:
            continue
        epochs = _shift_apinn_epochs_if_needed(model_name, epochs)
        ax.plot(epochs, f1, color=color_map[model_name], linestyle="-", linewidth=1.4)
        ax.plot(epochs, f2, color=color_map[model_name], linestyle="--", linewidth=1.4)
        y_values.extend(np.asarray(f1, dtype=np.float64).tolist())
        y_values.extend(np.asarray(f2, dtype=np.float64).tolist())
        model_handles.append(Line2D([0], [0], color=color_map[model_name], lw=1.4, ls="-", label=model_name))

    exact_epochs = np.linspace(0.0, XMAX_EPOCH, 20, dtype=np.float64)
    ax.scatter(
        exact_epochs,
        np.full_like(exact_epochs, F1_TRUE),
        color=C_EXACT,
        s=5,
        marker="o",
        alpha=0.9,
        zorder=3,
    )
    ax.scatter(
        exact_epochs,
        np.full_like(exact_epochs, F2_TRUE),
        color=C_EXACT,
        s=5,
        marker="^",
        alpha=0.9,
        zorder=3,
    )

    model_legend = ax.legend(
        handles=model_handles,
        frameon=False,
        loc="upper left",
        ncol=2,
        columnspacing=1.0,
        handlelength=2.0,
    )
    ax.add_artist(model_legend)

    exact_handles = [
        Line2D([0], [0], color=C_EXACT, lw=0.0, ls="None", marker="o", markersize=3.6, label=r"$f_1$ Exact"),
        Line2D([0], [0], color=C_EXACT, lw=0.0, ls="None", marker="^", markersize=3.9, label=r"$f_2$ Exact"),
    ]
    ax.legend(
        handles=exact_handles,
        frameon=False,
        loc="lower right",
        ncol=1,
        columnspacing=1.0,
        handlelength=1.0,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_xlim(0.0, XMAX_EPOCH)
    y_arr = np.asarray(y_values, dtype=np.float64)
    y_arr = y_arr[np.isfinite(y_arr)]
    if y_arr.size > 0:
        y_min = float(np.min(y_arr))
        y_max = float(np.max(y_arr))
        y_span = max(y_max - y_min, 1.0)
        pad = 0.06 * y_span
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir.parent / "Figure"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        dpi=300,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=1 / 72, h_pad=2 / 72, wspace=0.08, hspace=0.02)

    _plot_loss_compare(axes[0], script_dir)
    _add_panel_label(axes[0], "(a)")
    _plot_param_evolution(axes[1], script_dir)
    _add_panel_label(axes[1], "(b)")

    out_png = output_dir / "Fig11.png"
    out_pdf = output_dir / "Fig11.pdf"
    fig.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, dpi=400, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
