#!/usr/bin/env python3
from __future__ import annotations

import argparse
from glob import glob
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator


DEFAULT_CONFIG = {
    "tangent_theta": 0.5 * np.pi,
    "line_points": 401,
    "figsize": (12.0, 10.8),
    "dpi": 260,
    "phi_color": "#2ca02c",
    "ux_color": "#1f77b4",
    "uy_color": "#ff7f0e",
    "umag_color": "#9467bd",
    "true_color": "#17becf",
    "vline_color": "#4d4d4d",
}


@lru_cache(maxsize=8)
def _load_reference_points(txt_filename: str) -> Dict[str, np.ndarray]:
    txt_path = Path(txt_filename).expanduser().resolve()
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("%")]
    arr = np.loadtxt(lines)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("Expected Ellipse.txt with at least columns x y ux uy")
    return {
        "xy": arr[:, 0:2].astype(np.float64),
        "ux": arr[:, 2].astype(np.float64),
        "uy": arr[:, 3].astype(np.float64),
    }


def _build_interpolator(x_axis: np.ndarray, y_axis: np.ndarray, field_map: np.ndarray) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        (np.asarray(x_axis, dtype=np.float64), np.asarray(y_axis, dtype=np.float64)),
        np.asarray(field_map, dtype=np.float64),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def _build_scattered_interpolator(xy: np.ndarray, values: np.ndarray):
    xy = np.asarray(xy, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    linear_interp = LinearNDInterpolator(xy, values, fill_value=np.nan)
    nearest_interp = NearestNDInterpolator(xy, values)

    def _interp(query_xy: np.ndarray) -> np.ndarray:
        query_xy = np.asarray(query_xy, dtype=np.float64)
        linear_val = np.asarray(linear_interp(query_xy), dtype=np.float64).reshape(-1)
        if np.any(~np.isfinite(linear_val)):
            nearest_val = np.asarray(nearest_interp(query_xy), dtype=np.float64).reshape(-1)
            linear_val[~np.isfinite(linear_val)] = nearest_val[~np.isfinite(linear_val)]
        return linear_val

    return _interp


def _crossing_line_segment(
    ellipse: Tuple[float, float, float, float, float],
    bbox: Tuple[float, float, float, float],
    *,
    tangent_theta: float,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xc, yc, a, b, gamma = [float(v) for v in ellipse]
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]

    ct = np.cos(float(tangent_theta))
    st = np.sin(float(tangent_theta))
    cg = np.cos(gamma)
    sg = np.sin(gamma)

    point = np.array(
        [
            xc + cg * (a * ct) - sg * (b * st),
            yc + sg * (a * ct) + cg * (b * st),
        ],
        dtype=np.float64,
    )
    tangent_dir = np.array(
        [
            -a * st * cg - b * ct * sg,
            -a * st * sg + b * ct * cg,
        ],
        dtype=np.float64,
    )
    tangent_dir /= np.linalg.norm(tangent_dir) + 1e-15
    normal_dir = np.array([-tangent_dir[1], tangent_dir[0]], dtype=np.float64)
    normal_dir /= np.linalg.norm(normal_dir) + 1e-15

    t_candidates = []
    eps = 1e-12
    if abs(normal_dir[0]) > eps:
        for x_edge in (xmin, xmax):
            t = (x_edge - point[0]) / normal_dir[0]
            y = point[1] + t * normal_dir[1]
            if ymin - 1e-9 <= y <= ymax + 1e-9:
                t_candidates.append(float(t))
    if abs(normal_dir[1]) > eps:
        for y_edge in (ymin, ymax):
            t = (y_edge - point[1]) / normal_dir[1]
            x = point[0] + t * normal_dir[0]
            if xmin - 1e-9 <= x <= xmax + 1e-9:
                t_candidates.append(float(t))

    if len(t_candidates) < 2:
        span = max(xmax - xmin, ymax - ymin)
        t_candidates = [-span, span]

    t_min = min(t_candidates)
    t_max = max(t_candidates)
    s = np.linspace(t_min, t_max, int(n_points), dtype=np.float64)
    xy_line = point[None, :] + s[:, None] * normal_dir[None, :]
    return s, xy_line, point


def _phi_signed_ellipse_np(
    xy: np.ndarray,
    ellipse: Tuple[float, float, float, float, float],
) -> np.ndarray:
    xc, yc, a, b, gamma = [float(v) for v in ellipse]
    cg = np.cos(gamma)
    sg = np.sin(gamma)
    dx = xy[:, 0:1] - xc
    dy = xy[:, 1:2] - yc
    xp = cg * dx + sg * dy
    yp = -sg * dx + cg * dy
    return ((xp / a) ** 2 + (yp / b) ** 2 - 1.0).reshape(-1)


def _phi_zero_crossings(
    s_line: np.ndarray,
    phi_line: np.ndarray,
) -> np.ndarray:
    s = np.asarray(s_line, dtype=np.float64).reshape(-1)
    phi = np.asarray(phi_line, dtype=np.float64).reshape(-1)
    if s.size != phi.size:
        raise ValueError(f"s/phi length mismatch: {s.size} vs {phi.size}")
    if s.size < 2:
        return np.zeros((0,), dtype=np.float64)

    roots = []
    eps = 1e-14
    for i in range(s.size - 1):
        s0 = float(s[i])
        s1 = float(s[i + 1])
        p0 = float(phi[i])
        p1 = float(phi[i + 1])
        if abs(p0) <= eps:
            roots.append(s0)
        if p0 * p1 < 0.0:
            t = -p0 / (p1 - p0)
            roots.append(s0 + t * (s1 - s0))

    if abs(float(phi[-1])) <= eps:
        roots.append(float(s[-1]))

    if not roots:
        return np.zeros((0,), dtype=np.float64)

    roots = np.asarray(sorted(roots), dtype=np.float64)
    dedup = [roots[0]]
    tol = max(1e-10, 1e-6 * float(np.max(np.abs(s)) + 1.0))
    for r in roots[1:]:
        if abs(float(r) - float(dedup[-1])) > tol:
            dedup.append(float(r))
    return np.asarray(dedup, dtype=np.float64)


def _pick_prediction_keys(data: Dict[str, np.ndarray]) -> Tuple[str, str]:
    ux_key = "ux_pred" if "ux_pred" in data else "ux"
    uy_key = "uy_pred" if "uy_pred" in data else "uy"
    if ux_key not in data or uy_key not in data:
        raise KeyError("NPZ must contain ('ux_pred','uy_pred') or ('ux','uy').")
    return ux_key, uy_key


def save_u_slice_with_phi_plot_from_fields(
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    ux_pred_map: np.ndarray,
    uy_pred_map: np.ndarray,
    ellipse: Tuple[float, float, float, float, float],
    txt_filename: str,
    save_path: Path,
    epoch: int = -1,
    phi_map: Optional[np.ndarray] = None,
    save_npz_path: Optional[Path] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    title_prefix: str = "Ellipse",
) -> Path:
    cfg = dict(DEFAULT_CONFIG)

    x_axis = np.asarray(x_axis, dtype=np.float64).reshape(-1)
    y_axis = np.asarray(y_axis, dtype=np.float64).reshape(-1)
    ux_pred_map = np.asarray(ux_pred_map, dtype=np.float64)
    uy_pred_map = np.asarray(uy_pred_map, dtype=np.float64)
    if bbox is None:
        bbox = (float(x_axis.min()), float(x_axis.max()), float(y_axis.min()), float(y_axis.max()))

    if phi_map is None:
        X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")
        phi_map = _phi_signed_ellipse_np(np.stack([X.reshape(-1), Y.reshape(-1)], axis=1), ellipse).reshape(X.shape)
    else:
        phi_map = np.asarray(phi_map, dtype=np.float64)

    ref = _load_reference_points(str(txt_filename))

    s_line, xy_line, tangent_point = _crossing_line_segment(
        ellipse,
        bbox,
        tangent_theta=float(cfg["tangent_theta"]),
        n_points=int(cfg["line_points"]),
    )

    ux_pred_interp = _build_interpolator(x_axis, y_axis, ux_pred_map)
    uy_pred_interp = _build_interpolator(x_axis, y_axis, uy_pred_map)
    phi_interp = _build_interpolator(x_axis, y_axis, phi_map)
    ux_true_interp = _build_scattered_interpolator(ref["xy"], ref["ux"])
    uy_true_interp = _build_scattered_interpolator(ref["xy"], ref["uy"])

    ux_pred_line = ux_pred_interp(xy_line)
    uy_pred_line = uy_pred_interp(xy_line)
    ux_true_line = ux_true_interp(xy_line)
    uy_true_line = uy_true_interp(xy_line)
    phi_line = phi_interp(xy_line)
    phi_zero_s = _phi_zero_crossings(s_line, phi_line)

    umag_pred_line = np.sqrt(ux_pred_line ** 2 + uy_pred_line ** 2)
    umag_true_line = np.sqrt(ux_true_line ** 2 + uy_true_line ** 2)

    ux_abs_res = np.abs(ux_pred_line - ux_true_line)
    uy_abs_res = np.abs(uy_pred_line - uy_true_line)
    umag_abs_res = np.abs(umag_pred_line - umag_true_line)

    fig, axes = plt.subplots(3, 2, figsize=tuple(cfg["figsize"]), dpi=int(cfg["dpi"]), constrained_layout=True)
    left_specs = [
        ("$u_x$", ux_pred_line, ux_true_line, cfg["ux_color"]),
        ("$u_y$", uy_pred_line, uy_true_line, cfg["uy_color"]),
        ("$|u|$", umag_pred_line, umag_true_line, cfg["umag_color"]),
    ]
    right_specs = [
        ("$|u_x^{pred}-u_x^{true}|$", ux_abs_res),
        ("$|u_y^{pred}-u_y^{true}|$", uy_abs_res),
        ("$||u|^{pred}-|u|^{true}|$", umag_abs_res),
    ]

    for row, (label, pred_line, true_line, color) in enumerate(left_specs):
        axes[row, 0].plot(s_line, pred_line, color=color, linewidth=2.0, label="pred")
        axes[row, 0].plot(s_line, true_line, color=cfg["true_color"], linewidth=2.0, linestyle=":", label="true")
        for s_cross in phi_zero_s:
            axes[row, 0].axvline(float(s_cross), color=cfg["vline_color"], linestyle="-.", linewidth=1.0, alpha=0.85)
        axes[row, 0].set_title(f"{label} along crossing line")
        axes[row, 0].set_xlabel("arc length s")
        axes[row, 0].grid(True, alpha=0.22)
        axes[row, 0].legend(loc="best", frameon=False)

    for row, (label, residual_line) in enumerate(right_specs):
        ax = axes[row, 1]
        ax_phi = ax.twinx()
        ax.plot(s_line, residual_line, color="#8c564b", linewidth=2.0, label=label)
        ax_phi.plot(s_line, phi_line, color=cfg["phi_color"], linewidth=1.8, linestyle="--", label="phi")
        ax_phi.axhline(0.0, color=cfg["phi_color"], linestyle=":", linewidth=1.0, alpha=0.85)
        for s_cross in phi_zero_s:
            ax.axvline(float(s_cross), color=cfg["vline_color"], linestyle="-.", linewidth=1.0, alpha=0.85)
        ax.set_title(f"{label} + phi")
        ax.set_xlabel("arc length s")
        ax.grid(True, alpha=0.22)

    fig.suptitle(
        f"{title_prefix} crossing slice | epoch={epoch} | boundary point=({tangent_point[0]:.4f}, {tangent_point[1]:.4f})"
    )

    save_path = Path(save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    if save_npz_path is not None:
        save_npz_path = Path(save_npz_path).expanduser().resolve()
        save_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_npz_path,
            epoch=np.asarray([epoch], dtype=np.int64),
            s=s_line,
            xy=xy_line,
            tangent_point=tangent_point,
            ux_pred=ux_pred_line,
            uy_pred=uy_pred_line,
            ux_true=ux_true_line,
            uy_true=uy_true_line,
            umag_pred=umag_pred_line,
            umag_true=umag_true_line,
            phi=phi_line,
            phi_zero_s=phi_zero_s,
        )

    return save_path


def save_u_slice_with_phi_plot(
    npz_path: Path,
    *,
    txt_filename: str = "Ellipse.txt",
    ellipse: Tuple[float, float, float, float, float],
    save_path: Optional[Path] = None,
    save_npz_path: Optional[Path] = None,
    title_prefix: str = "Ellipse",
) -> Path:
    npz_path = Path(npz_path).expanduser().resolve()
    with np.load(npz_path) as zf:
        data = {k: zf[k] for k in zf.files}

    ux_key, uy_key = _pick_prediction_keys(data)
    if "x" not in data or "y" not in data:
        raise KeyError("NPZ must contain x and y axes.")

    epoch = -1
    if "epoch" in data:
        epoch_arr = np.asarray(data["epoch"]).reshape(-1)
        if epoch_arr.size > 0:
            epoch = int(epoch_arr[0])

    phi_map = data.get("phi")
    if phi_map is None and "phi_true" in data:
        phi_map = data["phi_true"]

    if save_path is None:
        save_path = npz_path.with_suffix(".slice.png")

    return save_u_slice_with_phi_plot_from_fields(
        x_axis=np.asarray(data["x"], dtype=np.float64),
        y_axis=np.asarray(data["y"], dtype=np.float64),
        ux_pred_map=np.asarray(data[ux_key], dtype=np.float64),
        uy_pred_map=np.asarray(data[uy_key], dtype=np.float64),
        phi_map=None if phi_map is None else np.asarray(phi_map, dtype=np.float64),
        ellipse=ellipse,
        txt_filename=txt_filename,
        save_path=save_path,
        save_npz_path=save_npz_path,
        epoch=epoch,
        title_prefix=title_prefix,
    )


def repair_historical_u_slice_with_phi(
    *,
    snapshot_dir: Path,
    output_dir: Path,
    txt_filename: str,
    ellipse: Tuple[float, float, float, float, float],
    title_prefix: str = "ADD-PINNs",
) -> int:
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_paths = sorted(glob(str(snapshot_dir / "phi_epoch_*.npz")))
    if not snapshot_paths:
        raise FileNotFoundError(f"No phi snapshots found in {snapshot_dir}")

    repaired = 0
    for snapshot_path_str in snapshot_paths:
        snapshot_path = Path(snapshot_path_str)
        with np.load(snapshot_path) as zf:
            data = {k: zf[k] for k in zf.files}

        epoch_arr = np.asarray(data.get("epoch", np.array([-1], dtype=np.int64))).reshape(-1)
        epoch = int(epoch_arr[0]) if epoch_arr.size > 0 else -1
        x_axis = np.asarray(data["x"], dtype=np.float64).reshape(-1)
        y_axis = np.asarray(data["y"], dtype=np.float64).reshape(-1)
        bbox_arr = np.asarray(
            data.get(
                "bbox",
                np.array([x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()], dtype=np.float64),
            ),
            dtype=np.float64,
        ).reshape(-1)
        bbox = (float(bbox_arr[0]), float(bbox_arr[1]), float(bbox_arr[2]), float(bbox_arr[3]))

        out_png = output_dir / f"u_slice_with_phi_{epoch:08d}.png"
        out_npz = output_dir / f"u_slice_with_phi_{epoch:08d}.npz"
        save_u_slice_with_phi_plot_from_fields(
            x_axis=x_axis,
            y_axis=y_axis,
            ux_pred_map=np.asarray(data["ux"], dtype=np.float64),
            uy_pred_map=np.asarray(data["uy"], dtype=np.float64),
            phi_map=np.asarray(data["phi"], dtype=np.float64),
            ellipse=ellipse,
            txt_filename=txt_filename,
            save_path=out_png,
            save_npz_path=out_npz,
            epoch=epoch,
            bbox=bbox,
            title_prefix=title_prefix,
        )
        repaired += 1

    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tangent-line slice plots for Ellipse outputs.")
    parser.add_argument(
        "--history",
        action="store_true",
        help="Regenerate all historical ADD-PINNs slice plots from phi snapshots.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    npz_path = project_root / "output_roi_on" / "data_output" / "final_fields.npz"
    out_path = project_root / "output_roi_on" / "data_output" / "u_slice_with_phi_final.png"
    save_u_slice_with_phi_plot(
        npz_path,
        txt_filename=str(project_root / "Ellipse.txt"),
        ellipse=(0.05, 0.10, 0.35, 0.15, np.deg2rad(-30.0)),
        save_path=out_path,
        save_npz_path=out_path.with_suffix(".npz"),
        title_prefix="ADD-PINNs",
    )
    print(f"[Saved] {out_path}")

    if args.history:
        n_repaired = repair_historical_u_slice_with_phi(
            snapshot_dir=project_root / "output_roi_on" / "phi_snapshots",
            output_dir=project_root / "output_roi_on" / "u_slice_with_phi",
            txt_filename=str(project_root / "Ellipse.txt"),
            ellipse=(0.05, 0.10, 0.35, 0.15, np.deg2rad(-30.0)),
            title_prefix="ADD-PINNs",
        )
        print(f"[Repaired] historical slice count = {n_repaired}")


if __name__ == "__main__":
    main()
