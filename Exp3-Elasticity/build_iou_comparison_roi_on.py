#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

import config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ELLIPSE_ROOT = Path(__file__).resolve().parent
ABLATION_ROOT = PROJECT_ROOT / "ADD-PINNs-Ellipse-Ablation"
FIGURE_DIR = PROJECT_ROOT / "Figure"


def signed_ellipse(xx: np.ndarray, yy: np.ndarray, ellipse: Tuple[float, float, float, float, float]) -> np.ndarray:
    xc, yc, a, b, gamma = (float(v) for v in ellipse)
    dx = xx - xc
    dy = yy - yc
    cg = np.cos(gamma)
    sg = np.sin(gamma)
    xp = cg * dx + sg * dy
    yp = -sg * dx + cg * dy
    return (xp / a) ** 2 + (yp / b) ** 2 - 1.0


def compute_iou(snapshot_path: Path, ellipse: Tuple[float, float, float, float, float]) -> Tuple[int, float]:
    with np.load(snapshot_path) as data:
        epoch = int(np.asarray(data["epoch"]).reshape(-1)[0])
        x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
        y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
        phi = np.asarray(data["phi"], dtype=np.float64)

    xx, yy = np.meshgrid(x, y, indexing="ij")
    phi_true = signed_ellipse(xx, yy, ellipse)

    pred_inside = phi >= 0.0
    true_inside = phi_true <= 0.0
    intersection = np.logical_and(pred_inside, true_inside).sum()
    union = np.logical_or(pred_inside, true_inside).sum()
    iou = float(intersection / union) if union else 1.0
    return epoch, iou


def summarize_snapshot_dir(snapshot_dir: Path, ellipse: Tuple[float, float, float, float, float]) -> Dict[int, float]:
    snapshot_dir = snapshot_dir.expanduser().resolve()
    rows = []
    result: Dict[int, float] = {}

    for snapshot_path in sorted(snapshot_dir.glob("phi_epoch_*.npz")):
        epoch, iou = compute_iou(snapshot_path, ellipse)
        rows.append((epoch, iou, snapshot_path.name))
        result[epoch] = iou

    summary_path = snapshot_dir / "iou_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "iou", "snapshot"])
        writer.writerows((epoch, f"{iou:.8f}", name) for epoch, iou, name in rows)

    print(f"Saved: {summary_path}")
    return result


def write_comparison_csv(
    out_path: Path,
    pi_moe_iou: Dict[int, float],
    ablation_iou: Dict[int, float],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = sorted(set(pi_moe_iou) | set(ablation_iou))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "pi_moe_iou", "ablation_iou", "delta_ablation_minus_pimoe"])
        for epoch in epochs:
            p = pi_moe_iou.get(epoch)
            a = ablation_iou.get(epoch)
            delta = (a - p) if (a is not None and p is not None) else None
            writer.writerow(
                [
                    epoch,
                    "" if p is None else f"{p:.8f}",
                    "" if a is None else f"{a:.8f}",
                    "" if delta is None else f"{delta:.8f}",
                ]
            )

    print(f"Saved: {out_path}")


def main() -> None:
    ellipse = tuple(float(v) for v in config.ELLIPSE_PARAMS)
    sources: Iterable[Tuple[Path, str]] = [
        (ELLIPSE_ROOT / "output_roi_on" / "phi_snapshots", "pi-moe"),
        (ABLATION_ROOT / "output_roi_on" / "phi_snapshots", "ablation"),
    ]

    missing = [str(path) for path, _ in sources if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing snapshot directories:\n" + "\n".join(missing))

    pi_moe_iou = summarize_snapshot_dir(ELLIPSE_ROOT / "output_roi_on" / "phi_snapshots", ellipse)
    ablation_iou = summarize_snapshot_dir(ABLATION_ROOT / "output_roi_on" / "phi_snapshots", ellipse)

    # write to simplified IoU filename (sampling mode removed)
    write_comparison_csv(
        FIGURE_DIR / "IoU_Exp1_v2.csv",
        pi_moe_iou,
        ablation_iou,
    )


if __name__ == "__main__":
    main()
