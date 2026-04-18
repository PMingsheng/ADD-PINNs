#!/usr/bin/env python3
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SWEEP_ROOT = PROJECT_ROOT / "outputs_scratch5000_eik_volume_sweep"
SUMMARY_CSV = PROJECT_ROOT.parent / "Figure" / "ADD-PINNs-Possion-3D_scratch5000_eik_volume_sweep_summary.csv"


PARAM_SETS: List[Dict[str, object]] = [
    {"name": "exp01_base", "lam_eik": 1e-4, "lam_volume": 1e-5},
    {"name": "exp02_eik1e5", "lam_eik": 1e-5, "lam_volume": 1e-5},
    {"name": "exp03_eik3e5", "lam_eik": 3e-5, "lam_volume": 1e-5},
    {"name": "exp04_eik3e4", "lam_eik": 3e-4, "lam_volume": 1e-5},
    {"name": "exp05_eik1e3", "lam_eik": 1e-3, "lam_volume": 1e-5},
    {"name": "exp06_vol0", "lam_eik": 1e-4, "lam_volume": 0.0},
    {"name": "exp07_vol1e6", "lam_eik": 1e-4, "lam_volume": 1e-6},
    {"name": "exp08_vol3e6", "lam_eik": 1e-4, "lam_volume": 3e-6},
    {"name": "exp09_vol3e5", "lam_eik": 1e-4, "lam_volume": 3e-5},
    {"name": "exp10_vol1e4", "lam_eik": 1e-4, "lam_volume": 1e-4},
]


def compute_metrics(final_fields_path: Path) -> Dict[str, float]:
    with np.load(final_fields_path) as z:
        phi_true = np.asarray(z["phi_true"], dtype=np.float64)
        phi_pred = np.asarray(z["phi_pred"], dtype=np.float64)
        u_rel_l2 = float(np.asarray(z["u_rel_l2"], dtype=np.float64).reshape(-1)[0])
        f_rel_l2 = float(np.asarray(z["f_rel_l2"], dtype=np.float64).reshape(-1)[0])
        u_fit_rel_l2 = float(np.asarray(z["u_fit_rel_l2"], dtype=np.float64).reshape(-1)[0])

    mask_true = phi_true >= 0.0
    mask_pred = phi_pred >= 0.0
    inter = np.logical_and(mask_true, mask_pred).sum(dtype=np.int64)
    union = np.logical_or(mask_true, mask_pred).sum(dtype=np.int64)
    iou = float(inter / union) if union else 1.0
    return {
        "iou": iou,
        "u_rel_l2": u_rel_l2,
        "f_rel_l2": f_rel_l2,
        "u_fit_rel_l2": u_fit_rel_l2,
    }


def run_one(params: Dict[str, object]) -> Dict[str, object]:
    out_dir = SWEEP_ROOT / str(params["name"])
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--epochs",
        "5000",
        "--output-dir",
        str(out_dir.relative_to(PROJECT_ROOT)),
        "--lam-eik",
        str(params["lam_eik"]),
        "--lam-volume",
        str(params["lam_volume"]),
    ]
    print(f"[Run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    final_fields = out_dir / "data_output" / "final_fields.npz"
    metrics = compute_metrics(final_fields)
    row = dict(params)
    row["output_dir"] = str(out_dir)
    row.update(metrics)
    return row


def write_summary(rows: List[Dict[str, object]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "output_dir",
        "lam_eik",
        "lam_volume",
        "iou",
        "u_rel_l2",
        "f_rel_l2",
        "u_fit_rel_l2",
    ]
    rows_sorted = sorted(rows, key=lambda r: (-float(r["iou"]), float(r["u_rel_l2"]), float(r["f_rel_l2"])))
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)
    print(f"[Saved] {SUMMARY_CSV}")
    for row in rows_sorted:
        print(
            f"[Summary] {row['name']} "
            f"IoU={row['iou']:.6f} "
            f"u_rel={row['u_rel_l2']:.6e} "
            f"f_rel={row['f_rel_l2']:.6e}"
        )


def main() -> None:
    rows: List[Dict[str, object]] = []
    for params in PARAM_SETS:
        row = run_one(params)
        rows.append(row)
        print(
            f"[Result] {row['name']} "
            f"IoU={row['iou']:.6f} "
            f"u_rel={row['u_rel_l2']:.6e} "
            f"f_rel={row['f_rel_l2']:.6e}"
        )
    write_summary(rows)


if __name__ == "__main__":
    main()
