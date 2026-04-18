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
FIG_HEIGHT = 2.44

C_PIMOE = "#D55E00"
C_APINNs = "#009E73"
C_PINNs = "#0072B2"
C_ROI_ON = "#CC79A7"

LOSS_PATHS = {
    "ADD-PINNs": "output_roi_off/data_output/loss_list_global.csv",
    "ADD-PINNs (roi-on)": "output_roi_on/data_output/loss_list_global.csv",
    "APINNs": "outputs_apinn/loss.csv",
    "PINNs": "outputs_pinn_single/loss.csv",
}

PARAM_ONLY_PATHS = {
    "ADD-PINNs": "output_roi_off/data_output/loss_list_global.csv",
    "ADD-PINNs (roi-on)": "output_roi_on/data_output/loss_list_global.csv",
    "APINNs": "outputs_apinn/loss.csv",
    "PINNs": "outputs_pinn_single/loss.csv",
}


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


def _extract_comparable_loss(cols: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    epochs = np.asarray(cols.get("epoch", np.arange(len(next(iter(cols.values()))))), dtype=np.float64)
    if "raw_data_u" in cols:
        total = (
            np.asarray(cols.get("raw_data_u", 0.0), dtype=np.float64)
            + np.asarray(cols.get("raw_data_eps", 0.0), dtype=np.float64)
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


def _extract_param_series(cols: Dict[str, np.ndarray], model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    epochs = np.asarray(cols.get("epoch", np.arange(len(next(iter(cols.values()))))), dtype=np.float64)
    if model_name.startswith("ADD-PINNs"):
        e = np.asarray(cols.get("E1", np.full_like(epochs, np.nan)), dtype=np.float64)
    else:
        e = np.asarray(cols.get("E_out", np.full_like(epochs, np.nan)), dtype=np.float64)
    valid = np.isfinite(epochs) & np.isfinite(e)
    return epochs[valid], e[valid]


def _remap_apinn_epochs(epochs: np.ndarray) -> np.ndarray:
    epochs = np.asarray(epochs, dtype=np.float64)
    keep = epochs >= 1000.0
    epochs = epochs[keep]
    if epochs.size == 0:
        return epochs
    src0 = 1000.0
    src1 = float(np.max(epochs))
    dst0 = 10000.0
    dst1 = 60000.0
    if src1 <= src0:
        return np.full_like(epochs, dst1)
    return dst0 + (epochs - src0) * (dst1 - dst0) / (src1 - src0)


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

    max_epoch = 0.0
    for model_name, rel_path in LOSS_PATHS.items():
        cols = _load_csv_columns(script_dir / rel_path)
        epochs, total = _extract_comparable_loss(cols)
        if epochs.size == 0:
            continue
        if model_name == "APINNs":
            keep = epochs >= 1000.0
            total = total[keep]
            epochs = _remap_apinn_epochs(epochs)
            if epochs.size == 0:
                continue
        max_epoch = max(max_epoch, float(np.max(epochs)))
        ax.semilogy(
            epochs,
            total,
            color=color_map[model_name],
            linestyle=style_map[model_name],
            linewidth=1.3,
            label=model_name,
        )

    ax.set_title("Comparable Loss History", pad=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Data+PDE+Interface Loss")
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
    if max_epoch > 0:
        ax.set_xlim(0.0, 60000.0)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.3)
    ax.legend(frameon=False, loc="upper right", ncol=2, handlelength=2.4)


def _add_panel_label(ax, tag: str) -> None:
    ax.text(
        -0.13,
        1.15,
        tag,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
    )


def _plot_param_evolution(ax, script_dir: Path) -> None:
    color_map = {
        "ADD-PINNs": C_PIMOE,
        "ADD-PINNs (roi-on)": C_ROI_ON,
        "APINNs": C_APINNs,
        "PINNs": C_PINNs,
    }

    model_handles = []
    for model_name, rel_path in PARAM_ONLY_PATHS.items():
        cols = _load_csv_columns(script_dir / rel_path)
        epochs, e = _extract_param_series(cols, model_name)
        if epochs.size == 0:
            continue
        if model_name == "APINNs":
            keep = epochs >= 1000.0
            e = e[keep]
            epochs = _remap_apinn_epochs(epochs)
            if epochs.size == 0:
                continue
            epochs = np.concatenate([np.asarray([0.0, 10000.0], dtype=np.float64), epochs])
            e = np.concatenate([np.asarray([1.0, 1.0], dtype=np.float64), e])
        ax.plot(epochs, e, color=color_map[model_name], linestyle="-", linewidth=1.5)
        model_handles.append(Line2D([0], [0], color=color_map[model_name], lw=1.3, ls="-", label=model_name))

    ax.axhline(0.5, color="#666666", linestyle=":", linewidth=0.9)
    exact_handle = Line2D([0], [0], color="#666666", lw=0.9, ls=":", label="Exact")

    ax.legend(
        handles=model_handles + [exact_handle],
        frameon=False,
        loc="upper right",
        ncol=2,
        columnspacing=1.0,
        handlelength=2.0,
    )
    ax.set_title(r"$E$ Evolution", pad=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_xlim(0.0, 60000.0)
    ax.set_ylim(-0.08, 1.08)
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
    fig.set_constrained_layout_pads(w_pad=1 / 72, h_pad=2 / 72, wspace=0.06, hspace=0.02)

    _plot_loss_compare(axes[0], script_dir)
    _add_panel_label(axes[0], "(a)")
    _plot_param_evolution(axes[1], script_dir)
    _add_panel_label(axes[1], "(b)")

    out_png = output_dir / "Fig8.png"
    out_pdf = output_dir / "Fig8.pdf"
    fig.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, dpi=400, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
