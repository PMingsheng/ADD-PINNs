#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# ==========================================
# Style (match flower version)
# ==========================================
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["mathtext.fontset"] = "stixsans"
rcParams["font.size"] = 7
rcParams["axes.labelsize"] = 7
rcParams["axes.titlesize"] = 7
rcParams["xtick.labelsize"] = 6
rcParams["ytick.labelsize"] = 6
rcParams["legend.fontsize"] = 6
rcParams["axes.linewidth"] = 0.6
rcParams["grid.linewidth"] = 0.5
rcParams["lines.linewidth"] = 1.2

FIG_WIDTH = 7.08
# Match Fig7 first-row aspect for a 1x2 layout.
FIG_HEIGHT = 2.2
C_PIMOE = "#D55E00"
C_PINN = "#0072B2"
C_APINN = "#CC79A7"


def _load_csv_cols(csv_path: Path) -> Dict[str, np.ndarray]:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        header = [h.strip() for h in f.readline().strip().split(",")]
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != len(header):
        raise ValueError(
            f"Column mismatch for {csv_path}: header={len(header)}, data={arr.shape[1]}"
        )
    return {name: arr[:, i] for i, name in enumerate(header)}


def _extract_epoch_metric(
    cols: Dict[str, np.ndarray],
    csv_path: Path,
    *,
    metric_keys: Tuple[str, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    if "epoch" not in cols:
        raise KeyError(f"{csv_path} must contain 'epoch' column.")
    metric_key = None
    for k in metric_keys:
        if k in cols:
            metric_key = k
            break
    if metric_key is None:
        raise KeyError(f"{csv_path} missing metric columns: {metric_keys}")

    epoch = np.asarray(cols["epoch"], dtype=np.float64).reshape(-1)
    metric = np.asarray(cols[metric_key], dtype=np.float64).reshape(-1)
    mask = np.isfinite(epoch) & np.isfinite(metric) & (metric > 0)
    return epoch[mask], metric[mask]


def _clip_by_epoch(
    epoch: np.ndarray,
    total: np.ndarray,
    max_epoch: float | None,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_epoch is None:
        return epoch, total
    m = epoch <= float(max_epoch)
    return epoch[m], total[m]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "Figure" / "Poisson3D_total_data_loss_compare.png"

    default_pimoe_csv = script_dir / "outputs_add_pinns3d_c1_sphere" / "loss_list_global.csv"
    if not default_pimoe_csv.exists():
        default_pimoe_csv = script_dir / "outputs_add_pinns3d_c1_sphere" / "data_output" / "loss_list_global.csv"

    default_pinn_csv = script_dir / "outputs_pinn3d_c1_sphere" / "loss_list_global.csv"
    if not default_pinn_csv.exists():
        default_pinn_csv = script_dir / "outputs_pinn3d_c1_sphere" / "data_output" / "loss_list_global.csv"

    default_apinn_csv = script_dir / "outputs_apinn3d_c1_sphere" / "loss_list_global.csv"
    if not default_apinn_csv.exists():
        default_apinn_csv = script_dir / "outputs_apinn3d_c1_sphere" / "data_output" / "loss_list_global.csv"

    parser = argparse.ArgumentParser(
        description="Plot total/data loss curves of ADD-PINNs, PINN, APINN in one figure."
    )
    parser.add_argument(
        "--pinn-csv",
        type=str,
        default=str(default_pinn_csv),
        help="PINN loss csv path",
    )
    parser.add_argument(
        "--apinn-csv",
        type=str,
        default=str(default_apinn_csv),
        help="APINN loss csv path",
    )
    parser.add_argument(
        "--pimoe-csv",
        type=str,
        default=str(default_pimoe_csv),
        help="ADD-PINNs loss csv path",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(default_out),
        help="Output figure path (.png/.pdf)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figure interactively.",
    )
    parser.add_argument(
        "--max-epoch",
        type=float,
        default=30000.0,
        help="Only plot points with epoch <= max-epoch (default: 30000).",
    )
    parser.add_argument(
        "--use-raw-total",
        action="store_true",
        help="Use total_raw for total-loss panel when available.",
    )
    args = parser.parse_args()

    pinn_csv = Path(args.pinn_csv).expanduser().resolve()
    apinn_csv = Path(args.apinn_csv).expanduser().resolve()
    pimoe_csv = Path(args.pimoe_csv).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    for p in (pinn_csv, apinn_csv, pimoe_csv):
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")

    pinn_cols = _load_csv_cols(pinn_csv)
    apinn_cols = _load_csv_cols(apinn_csv)
    pimoe_cols = _load_csv_cols(pimoe_csv)

    total_keys = ("total_raw", "total") if args.use_raw_total else ("total",)

    pinn_epoch, pinn_total = _extract_epoch_metric(pinn_cols, pinn_csv, metric_keys=total_keys)
    apinn_epoch, apinn_total = _extract_epoch_metric(apinn_cols, apinn_csv, metric_keys=total_keys)
    pimoe_epoch, pimoe_total = _extract_epoch_metric(pimoe_cols, pimoe_csv, metric_keys=total_keys)

    # Data panel always uses unweighted data loss.
    pinn_epoch_data, pinn_data = _extract_epoch_metric(pinn_cols, pinn_csv, metric_keys=("data_raw", "data"))
    apinn_epoch_data, apinn_data = _extract_epoch_metric(apinn_cols, apinn_csv, metric_keys=("data_raw", "data"))
    pimoe_epoch_data, pimoe_data = _extract_epoch_metric(pimoe_cols, pimoe_csv, metric_keys=("data_raw", "data"))

    pinn_epoch, pinn_total = _clip_by_epoch(pinn_epoch, pinn_total, args.max_epoch)
    apinn_epoch, apinn_total = _clip_by_epoch(apinn_epoch, apinn_total, args.max_epoch)
    pimoe_epoch, pimoe_total = _clip_by_epoch(pimoe_epoch, pimoe_total, args.max_epoch)
    pinn_epoch_data, pinn_data = _clip_by_epoch(pinn_epoch_data, pinn_data, args.max_epoch)
    apinn_epoch_data, apinn_data = _clip_by_epoch(apinn_epoch_data, apinn_data, args.max_epoch)
    pimoe_epoch_data, pimoe_data = _clip_by_epoch(pimoe_epoch_data, pimoe_data, args.max_epoch)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        dpi=300,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=1 / 72, h_pad=2 / 72, wspace=0.02, hspace=0.06)
    ax_total, ax_data = axes

    # Line styles follow flower version: two solid + one dashed.
    ax_total.semilogy(pimoe_epoch, pimoe_total, color=C_PIMOE, lw=1.2, ls="--", label="ADD-PINNs")
    ax_total.semilogy(pinn_epoch, pinn_total, color=C_PINN, lw=1.2, ls="-", label="PINN")
    ax_total.semilogy(apinn_epoch, apinn_total, color=C_APINN, lw=1.2, ls="-", alpha=0.9, label="APINN")
    ax_total.set_xlabel("Epoch")
    ax_total.set_ylabel("Total Loss")
    ax_total.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
    ax_total.legend(frameon=False, loc="lower right", ncol=2)

    ax_data.semilogy(pimoe_epoch_data, pimoe_data, color=C_PIMOE, lw=1.2, ls="--", label="ADD-PINNs")
    ax_data.semilogy(pinn_epoch_data, pinn_data, color=C_PINN, lw=1.2, ls="-", label="PINN")
    ax_data.semilogy(apinn_epoch_data, apinn_data, color=C_APINN, lw=1.2, ls="-", alpha=0.9, label="APINN")
    ax_data.set_xlabel("Epoch")
    ax_data.set_ylabel("Data Loss (unweighted)")
    ax_data.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
    ax_data.legend(frameon=False, loc="lower left", ncol=2)

    if pimoe_data.size and pinn_data.size and apinn_data.size:
        y_min_data = min(
            float(np.nanmin(pimoe_data)),
            float(np.nanmin(pinn_data)),
            float(np.nanmin(apinn_data)),
        )
        y_max_data = max(
            float(np.nanmax(pimoe_data)),
            float(np.nanmax(pinn_data)),
            float(np.nanmax(apinn_data)),
        )
        if y_min_data > 0 and y_max_data > y_min_data:
            ax_data.set_ylim(bottom=y_min_data * 0.4, top=y_max_data * 1.05)

    for ax in (ax_total, ax_data):
        if args.max_epoch is not None:
            ax.set_xlim(left=0.0, right=float(args.max_epoch))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    if out_path.suffix.lower() == ".png":
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {pdf_path}")
    print(f"Saved: {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
