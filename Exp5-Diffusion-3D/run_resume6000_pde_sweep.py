#!/usr/bin/env python3
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_OUTPUT = PROJECT_ROOT / "outputs_add_pinns3d_c1_sphere"
RESUME_CKPT = BASE_OUTPUT / "checkpoints" / "checkpoint_00006000.pt"
SWEEP_ROOT = PROJECT_ROOT / "outputs_resume6000_pde_sweep"
SUMMARY_CSV = PROJECT_ROOT.parent / "Figure" / "ADD-PINNs-Possion-3D_resume6000_pde_sweep_summary.csv"


PARAM_SETS: List[Dict[str, object]] = [
    {"name": "exp01_s8k_dt1e5", "start": 8000, "every": 1000, "dt": 1e-5, "inner": 2, "band": 0.02, "radius": 0.02, "clip_q": 0.99},
    {"name": "exp02_s8k_dt5e5", "start": 8000, "every": 1000, "dt": 5e-5, "inner": 2, "band": 0.02, "radius": 0.02, "clip_q": 0.99},
    {"name": "exp03_s8k_dt1e4", "start": 8000, "every": 1000, "dt": 1e-4, "inner": 2, "band": 0.02, "radius": 0.02, "clip_q": 0.99},
    {"name": "exp04_s7k_dt1e5", "start": 7000, "every": 1000, "dt": 1e-5, "inner": 2, "band": 0.02, "radius": 0.02, "clip_q": 0.99},
    {"name": "exp05_s7k_dt5e5", "start": 7000, "every": 1000, "dt": 5e-5, "inner": 2, "band": 0.02, "radius": 0.02, "clip_q": 0.99},
    {"name": "exp06_s7k_dt1e4", "start": 7000, "every": 1000, "dt": 1e-4, "inner": 2, "band": 0.02, "radius": 0.02, "clip_q": 0.99},
    {"name": "exp07_s7k_e500_dt5e5", "start": 7000, "every": 500, "dt": 5e-5, "inner": 2, "band": 0.02, "radius": 0.02, "clip_q": 0.99},
    {"name": "exp08_s7k_e500_dt1e4", "start": 7000, "every": 500, "dt": 1e-4, "inner": 2, "band": 0.02, "radius": 0.02, "clip_q": 0.99},
    {"name": "exp09_s7k_band003", "start": 7000, "every": 1000, "dt": 5e-5, "inner": 5, "band": 0.03, "radius": 0.03, "clip_q": 0.99},
    {"name": "exp10_s8k_clip095", "start": 8000, "every": 500, "dt": 5e-5, "inner": 5, "band": 0.03, "radius": 0.03, "clip_q": 0.95},
]


def compute_iou(final_fields_path: Path) -> Dict[str, float]:
    with np.load(final_fields_path) as data:
        phi_true = np.asarray(data["phi_true"], dtype=np.float64)
        phi_pred = np.asarray(data["phi_pred"], dtype=np.float64)
        u_rel_l2 = float(np.asarray(data["u_rel_l2"], dtype=np.float64).reshape(-1)[0])
        f_rel_l2 = float(np.asarray(data["f_rel_l2"], dtype=np.float64).reshape(-1)[0])
        u_fit_rel_l2 = float(np.asarray(data["u_fit_rel_l2"], dtype=np.float64).reshape(-1)[0])

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
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--resume",
        str(RESUME_CKPT),
        "--output-dir",
        str(out_dir.relative_to(PROJECT_ROOT)),
        "--allow-phi-update-when-frozen",
        "--phi-update-residual-type",
        "PDE",
        "--phi-update-start-epoch",
        str(params["start"]),
        "--phi-update-every",
        str(params["every"]),
        "--phi-update-dt",
        str(params["dt"]),
        "--phi-update-inner-steps",
        str(params["inner"]),
        "--phi-update-band-eps",
        str(params["band"]),
        "--phi-update-radius",
        str(params["radius"]),
        "--phi-update-clip-q",
        str(params["clip_q"]),
    ]

    print(f"[Run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    final_fields = out_dir / "data_output" / "final_fields.npz"
    metrics = compute_iou(final_fields)
    row = dict(params)
    row["output_dir"] = str(out_dir)
    row.update(metrics)
    return row


def write_summary(rows: List[Dict[str, object]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "output_dir",
        "every",
        "start",
        "dt",
        "inner",
        "band",
        "radius",
        "clip_q",
        "iou",
        "u_rel_l2",
        "f_rel_l2",
        "u_fit_rel_l2",
    ]
    rows_sorted = sorted(rows, key=lambda r: (-float(r["iou"]), float(r["u_rel_l2"])))
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)
    print(f"[Saved] {SUMMARY_CSV}")


def main() -> None:
    if not RESUME_CKPT.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {RESUME_CKPT}")

    rows: List[Dict[str, object]] = []
    for params in PARAM_SETS:
        row = run_one(params)
        rows.append(row)
        print(
            f"[Result] {row['name']} "
            f"IoU={row['iou']:.6f} "
            f"u_rel_l2={row['u_rel_l2']:.6e} "
            f"f_rel_l2={row['f_rel_l2']:.6e}"
        )

    write_summary(rows)


if __name__ == "__main__":
    main()
