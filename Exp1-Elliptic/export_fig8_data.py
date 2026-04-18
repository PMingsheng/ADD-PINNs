#!/usr/bin/env python3
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch

from config import SAMPLING_CONFIGS, DataConfig, VisualizationConfig
from problem import FLOWER_N, phi_signed_flower


def load_uniform_grid_fit_downsample(
    nx=8,
    ny=8,
    *,
    ttxt_filename="Possion.txt",
    circles=None,
    dense_factor=0.5,
    drop_boundary=True,
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    tol=0.02,
):
    xy_full = np.loadtxt(ttxt_filename, usecols=[0, 1], comments="%")
    n_side = int(np.sqrt(len(xy_full)))
    x_full = xy_full[:, 0].reshape(n_side, n_side)
    y_full = xy_full[:, 1].reshape(n_side, n_side)

    x_vec = np.unique(x_full)
    y_vec = np.unique(y_full)

    all_nodes = np.stack([x_full.flatten(), y_full.flatten()], axis=1)
    tree = cKDTree(all_nodes)

    x_tar = np.linspace(x_vec[1], x_vec[-2], nx)
    y_tar = np.linspace(y_vec[1], y_vec[-2], ny)
    xg, yg = np.meshgrid(x_tar, y_tar, indexing="ij")
    ideal_points = np.stack([xg.flatten(), yg.flatten()], axis=1)
    idxs_basic = tree.query(ideal_points, k=1)[1]
    idxs_basic = np.unique(idxs_basic)
    idxs_dense = np.array([], dtype=int)

    if circles:
        for (cx, cy, r) in circles:
            dense_nx = max(2, int(nx / dense_factor))
            dense_ny = max(2, int(ny / dense_factor))
            x_dense = np.linspace(x_vec[1], x_vec[-2], dense_nx)
            y_dense = np.linspace(y_vec[1], y_vec[-2], dense_ny)
            xd, yd = np.meshgrid(x_dense, y_dense, indexing="ij")
            pts_dense = np.stack([xd.flatten(), yd.flatten()], axis=1)
            mask = ((pts_dense[:, 0] - cx) ** 2 + (pts_dense[:, 1] - cy) ** 2) <= r**2
            pts_in = pts_dense[mask]
            if pts_in.size > 0:
                idxs_in = tree.query(pts_in, k=1)[1]
                idxs_dense = np.unique(np.concatenate([idxs_dense, idxs_in]))

    if idxs_dense.size > 0:
        idxs_dense = np.setdiff1d(idxs_dense, idxs_basic, assume_unique=False)

    xy_basic = all_nodes[idxs_basic]
    xy_dense = all_nodes[idxs_dense] if idxs_dense.size > 0 else np.array([]).reshape(0, 2)

    if drop_boundary:
        mask_basic = (
            (xy_basic[:, 0] > xlim[0] + tol)
            & (xy_basic[:, 0] < xlim[1] - tol)
            & (xy_basic[:, 1] > ylim[0] + tol)
            & (xy_basic[:, 1] < ylim[1] - tol)
        )
        xy_basic = xy_basic[mask_basic]

        if xy_dense.size > 0:
            mask_dense = (
                (xy_dense[:, 0] > xlim[0] + tol)
                & (xy_dense[:, 0] < xlim[1] - tol)
                & (xy_dense[:, 1] > ylim[0] + tol)
                & (xy_dense[:, 1] < ylim[1] - tol)
            )
            xy_dense = xy_dense[mask_dense]

    return xy_basic, xy_dense


def _grid_from_npz(npz_path: str):
    data = np.load(npz_path)
    x_grid, y_grid = data["x"], data["y"]
    phi = data["phi"]
    if x_grid.ndim == 1:
        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    else:
        X, Y = x_grid, y_grid
    return X, Y, phi


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(script_dir, "Possion.txt")
    output_dir = os.path.join(os.path.dirname(script_dir), "Figure_Data")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Fig8_data.xlsx")

    data_cfg = DataConfig()
    viz_cfg = VisualizationConfig()

    circle_roi = viz_cfg.phi_compare_circle
    target_epochs = [60000, 90000, 140000]

    sampling_modes: List[Tuple[str, str, bool]] = [
        ("roi-off", "Dense Measurement", False),
        ("roi-on", "Targeted Measurement", True),
        ("full-data", "Full-Field Measurement", True),
    ]

    data_dirs = {
        "roi-off": os.path.join(script_dir, "output_roi_off", "phi_snapshots"),
        "roi-on": os.path.join(script_dir, "output_roi_on", "phi_snapshots"),
        "full-data": os.path.join(script_dir, "output_full_data", "phi_snapshots"),
    }

    missing_files: List[str] = []

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Full grid from Possion.txt
        xy_full = np.loadtxt(txt_path, usecols=[0, 1], comments="%")
        df_full = pd.DataFrame(xy_full, columns=["x", "y"])
        df_full.to_excel(writer, sheet_name="full_grid", index=False)

        # Sampling points per mode
        for mode, _, _ in sampling_modes:
            cfg = SAMPLING_CONFIGS[mode]
            xy_basic, xy_dense = load_uniform_grid_fit_downsample(
                nx=cfg["nx"],
                ny=cfg["ny"],
                ttxt_filename=txt_path,
                circles=list(data_cfg.circles),
                dense_factor=cfg["dense_factor"],
                drop_boundary=cfg["drop_boundary"],
                xlim=cfg["xlim"],
                ylim=cfg["ylim"],
                tol=cfg["tol"],
            )
            df_basic = pd.DataFrame(xy_basic, columns=["x", "y"])
            df_basic["group"] = "basic"
            df_dense = pd.DataFrame(xy_dense, columns=["x", "y"])
            df_dense["group"] = "dense"
            df_sampling = pd.concat([df_basic, df_dense], ignore_index=True)
            df_sampling.to_excel(writer, sheet_name=f"sampling_{mode}", index=False)

        # Phi snapshots per mode/epoch
        for mode, _, _ in sampling_modes:
            data_dir = data_dirs[mode]
            for epoch in target_epochs:
                filename = f"phi_epoch_{epoch:08d}.npz"
                filepath = os.path.join(data_dir, filename)
                sheet_name = f"phi_{mode}_{epoch}"
                if not os.path.exists(filepath):
                    missing_files.append(filepath)
                    pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
                    continue
                X, Y, phi = _grid_from_npz(filepath)
                XY = np.stack([X.ravel(), Y.ravel()], axis=1)
                XY_t = torch.from_numpy(XY).to(dtype=torch.float32)
                phi_true = phi_signed_flower(XY_t).detach().cpu().numpy().reshape(X.shape)
                df_phi = pd.DataFrame(
                    {
                        "x": X.ravel(),
                        "y": Y.ravel(),
                        "phi_pred": phi.ravel(),
                        "phi_true": phi_true.ravel(),
                    }
                )
                df_phi.to_excel(writer, sheet_name=sheet_name, index=False)

        # Metadata sheet
        meta_rows: List[Tuple[str, str]] = [
            ("source_script", "ADD-PINNs-Possion/Fig8.py"),
            ("possion_txt", txt_path),
            ("flower_n", str(FLOWER_N)),
            ("circle_roi", str(circle_roi)),
            ("target_epochs", str(target_epochs)),
        ]
        for mode, label, show_roi in sampling_modes:
            cfg = SAMPLING_CONFIGS[mode]
            meta_rows.append((f"sampling_mode_{mode}_label", label))
            meta_rows.append((f"sampling_mode_{mode}_show_roi", str(show_roi)))
            for key in ["nx", "ny", "dense_factor", "drop_boundary", "xlim", "ylim", "tol"]:
                meta_rows.append((f"sampling_{mode}_{key}", str(cfg[key])))
            meta_rows.append((f"phi_snapshots_dir_{mode}", data_dirs[mode]))

        if missing_files:
            for path in missing_files:
                meta_rows.append(("missing_phi_snapshot", path))

        df_meta = pd.DataFrame(meta_rows, columns=["key", "value"])
        df_meta.to_excel(writer, sheet_name="metadata", index=False)

    print(f"Saved Fig8 plotting data to: {out_path}")


if __name__ == "__main__":
    main()
