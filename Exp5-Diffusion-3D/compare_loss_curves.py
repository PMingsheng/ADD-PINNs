import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ("ADD-PINNs", "PINN", "APINN")
MODEL_COLORS = {
    "ADD-PINNs": "#1f77b4",
    "PINN": "#ff7f0e",
    "APINN": "#2ca02c",
}
DEFAULT_REL_PATHS = {
    "ADD-PINNs": "outputs_add_pinns3d_c1_sphere/loss_list_global.csv",
    "PINN": "outputs_pinn3d_c1_sphere/loss_list_global.csv",
    "APINN": "outputs_apinn3d_c1_sphere/loss_list_global.csv",
}


def _read_loss_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"missing csv: {csv_path}")
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        raise ValueError(f"empty csv: {csv_path}")
    if data.shape == ():
        data = np.asarray([data])
    cols: Dict[str, np.ndarray] = {}
    for name in data.dtype.names:
        cols[name] = np.asarray(data[name], dtype=np.float64).reshape(-1)
    return cols


def _get_series(cols: Dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in cols:
        raise KeyError(f"column '{key}' not found")
    return cols[key]


def _plot_one_axis(
    ax,
    all_cols: Dict[str, Dict[str, np.ndarray]],
    metric_key: str,
    title: str,
    ylabel: str,
) -> None:
    for model_name in MODEL_ORDER:
        cols = all_cols[model_name]
        epoch = _get_series(cols, "epoch")
        metric = _get_series(cols, metric_key)
        ax.plot(
            epoch,
            metric,
            label=model_name,
            color=MODEL_COLORS[model_name],
            linewidth=1.4,
        )

    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)


def _resolve_csv_paths(args: argparse.Namespace, root: Path) -> Dict[str, Path]:
    override = {
        "ADD-PINNs": args.pimoe_csv,
        "PINN": args.pinn_csv,
        "APINN": args.apinn_csv,
    }
    resolved: Dict[str, Path] = {}
    for model_name in MODEL_ORDER:
        if override[model_name]:
            resolved[model_name] = Path(override[model_name]).expanduser().resolve()
        else:
            resolved[model_name] = (root / DEFAULT_REL_PATHS[model_name]).resolve()
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot loss comparison among ADD-PINNs, PINN, and APINN."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="project root that contains output folders",
    )
    parser.add_argument("--pimoe-csv", type=str, default=None, help="override ADD-PINNs loss csv path")
    parser.add_argument("--pinn-csv", type=str, default=None, help="override PINN loss csv path")
    parser.add_argument("--apinn-csv", type=str, default=None, help="override APINN loss csv path")
    parser.add_argument(
        "--use-raw",
        action="store_true",
        help="plot *_raw columns (total_raw/data_raw/pde_raw) if present",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="loss_compare.png",
        help="output figure path (relative to root if not absolute)",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    csv_paths = _resolve_csv_paths(args, root)

    all_cols: Dict[str, Dict[str, np.ndarray]] = {}
    for model_name in MODEL_ORDER:
        all_cols[model_name] = _read_loss_csv(csv_paths[model_name])

    if args.use_raw:
        key_total, key_data, key_pde = "total_raw", "data_raw", "pde_raw"
        suffix = " (raw)"
    else:
        key_total, key_data, key_pde = "total", "data", "pde"
        suffix = ""

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), dpi=args.dpi)

    _plot_one_axis(
        axes[0],
        all_cols,
        key_total,
        f"Total Loss{suffix}",
        "loss",
    )
    _plot_one_axis(
        axes[1],
        all_cols,
        key_data,
        f"Data Loss{suffix}",
        "loss",
    )
    _plot_one_axis(
        axes[2],
        all_cols,
        key_pde,
        f"PDE Loss{suffix}",
        "loss",
    )

    axes[2].legend(loc="best")
    fig.suptitle("Loss Comparison Across Models", y=1.02)
    fig.tight_layout()

    save_path = Path(args.save)
    if not save_path.is_absolute():
        save_path = root / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("=== CSV Paths ===")
    for model_name in MODEL_ORDER:
        print(f"{model_name:7s}: {csv_paths[model_name]}")
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    main()
