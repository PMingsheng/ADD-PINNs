"""
Plot E1/E2 history from loss_list_global.csv.
"""
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = str(BASE_DIR / "output_roi_on" / "loss_list_global.csv")
DEFAULT_OUTPUT_PATH = ""
DEFAULT_SHOW = True

# User-configurable defaults.
USER_CSV_PATH = ""
USER_OUTPUT_PATH = ""
USER_SHOW = DEFAULT_SHOW


def resolve_csv_path(raw_path: str) -> str:
    raw_path = raw_path or DEFAULT_CSV_PATH
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
    candidates = glob.glob(str(BASE_DIR / "output_*" / "loss_list_global.csv"))
    if not candidates:
        candidates = glob.glob(str(BASE_DIR / "**" / "loss_list_global.csv"), recursive=True)
    if not candidates:
        raise FileNotFoundError("no loss_list_global.csv found; pass the path explicitly")
    return max(candidates, key=os.path.getmtime)


def _load_loss_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    has_header = any(ch.isalpha() for ch in first_line)
    labels = None
    skip_header = 0
    if has_header and "," in first_line:
        labels = [c.strip().lower() for c in first_line.split(",")]
        skip_header = 1

    data = np.genfromtxt(path, delimiter=",", skip_header=skip_header)
    if data.size == 0:
        raise ValueError("loss_list_global.csv is empty")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if labels and len(labels) == data.shape[1]:
        label_map = {name: idx for idx, name in enumerate(labels)}
    else:
        label_map = {}
    return data, label_map


def plot_e_history(csv_path: str, output_path: str = "", show: bool = False) -> None:
    csv_path = resolve_csv_path(csv_path)
    data, label_map = _load_loss_csv(csv_path)

    if "epoch" in label_map:
        epochs = data[:, label_map["epoch"]]
    else:
        epochs = data[:, 0]

    if "e1" in label_map and "e2" in label_map:
        e1 = data[:, label_map["e1"]]
        e2 = data[:, label_map["e2"]]
    else:
        raise ValueError("CSV missing E1/E2 columns")

    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })
    fig, ax = plt.subplots(figsize=(3.6, 2.4), dpi=300)
    ax.plot(epochs, e1, label="E1", color="C0", lw=1.0)
    ax.plot(epochs, e2, label="E2", color="C1", lw=1.0)
    ax.axhline(0.5, color="C0", ls="--", lw=0.9, label="E1 true")
    ax.axhline(1.0, color="C1", ls="--", lw=0.9, label="E2 true")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("E value")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    if not output_path:
        out_dir = os.path.dirname(os.path.abspath(csv_path))
        output_path = os.path.join(out_dir, "E_history.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"[plot_e_history] saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    plot_e_history(
        USER_CSV_PATH or DEFAULT_CSV_PATH,
        USER_OUTPUT_PATH or DEFAULT_OUTPUT_PATH,
        USER_SHOW,
    )


if __name__ == "__main__":
    main()
