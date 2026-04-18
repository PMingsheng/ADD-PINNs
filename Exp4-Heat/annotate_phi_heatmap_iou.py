#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TRUE_CROSS = (0.4, 0.6, 0.3, 0.3, 0.1, 0.1)


def sdf_rect(xx: np.ndarray, yy: np.ndarray, xc: float, yc: float, a: float, b: float) -> np.ndarray:
    qx = np.abs(xx - xc) - a
    qy = np.abs(yy - yc) - b
    qx_pos = np.maximum(qx, 0.0)
    qy_pos = np.maximum(qy, 0.0)
    outside = np.hypot(qx_pos, qy_pos)
    inside = np.maximum(qx, qy)
    return outside + np.minimum(inside, 0.0)


def sdf_cross(xx: np.ndarray, yy: np.ndarray, xc: float, yc: float, lx: float, ly: float, wh: float, wv: float) -> np.ndarray:
    phi_h = sdf_rect(xx, yy, xc, yc, a=lx / 2.0, b=wh / 2.0)
    phi_v = sdf_rect(xx, yy, xc, yc, a=wv / 2.0, b=ly / 2.0)
    return np.minimum(phi_h, phi_v)


def compute_iou(snapshot_path: Path) -> tuple[int, float]:
    with np.load(snapshot_path) as data:
        epoch = int(np.asarray(data["epoch"]).reshape(-1)[0])
        x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
        y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
        phi = np.asarray(data["phi"], dtype=np.float64)

    xg, yg = np.meshgrid(x, y, indexing="ij")
    phi_true = sdf_cross(xg, yg, *TRUE_CROSS)

    pred_inside = phi >= 0.0
    true_inside = phi_true <= 0.0
    intersection = np.logical_and(pred_inside, true_inside).sum()
    union = np.logical_or(pred_inside, true_inside).sum()
    iou = float(intersection / union) if union else 1.0
    return epoch, iou


def annotate_png(image_path: Path, epoch: int, iou: float) -> None:
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    label = f"Epoch {epoch:d}  IoU={iou:.3f}"

    x0 = 14
    y0 = 12
    bbox = draw.textbbox((x0, y0), label, font=font)
    pad = 6
    rect = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rounded_rectangle(rect, radius=6, fill=(255, 255, 255, 215), outline=(30, 30, 30, 220), width=1)
    draw.text((x0, y0), label, fill=(20, 20, 20, 255), font=font)
    image.save(image_path)


def process_pair(heatmap_dir: Path, snapshot_dir: Path) -> Path:
    heatmap_dir = heatmap_dir.expanduser().resolve()
    snapshot_dir = snapshot_dir.expanduser().resolve()
    summary_path = heatmap_dir / "iou_summary.csv"

    rows = []
    for image_path in sorted(heatmap_dir.glob("phi_heatmap_*.png")):
        stem_suffix = image_path.stem.replace("phi_heatmap_", "")
        snapshot_path = snapshot_dir / f"phi_epoch_{stem_suffix}.npz"
        if not snapshot_path.exists():
            continue
        epoch, iou = compute_iou(snapshot_path)
        annotate_png(image_path, epoch, iou)
        rows.append((epoch, iou, image_path.name))

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "iou", "image"])
        writer.writerows((epoch, f"{iou:.8f}", image_name) for epoch, iou, image_name in rows)

    return summary_path


def main() -> None:
    pairs: Iterable[tuple[Path, Path]] = [
        (
            Path("/home/mingshengpeng/PhD/ADD-PINNs-Good/ADD-PINNs-Possion/output_roi_on/phi_heatmaps"),
            Path("/home/mingshengpeng/PhD/ADD-PINNs-Good/ADD-PINNs-Possion/output_roi_on/phi_snapshots"),
        ),
        (
            Path("/home/mingshengpeng/PhD/ADD-PINNs-Good/ADD-PINNs-Possion-Ablation/output_roi_on/phi_heatmaps"),
            Path("/home/mingshengpeng/PhD/ADD-PINNs-Good/ADD-PINNs-Possion-Ablation/output_roi_on/phi_snapshots"),
        ),
    ]

    for heatmap_dir, snapshot_dir in pairs:
        summary_path = process_pair(heatmap_dir, snapshot_dir)
        print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
