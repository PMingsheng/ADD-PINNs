#!/usr/bin/env python3
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from config import TrainConfig


DEFAULT_CONFIG = {
    "percentile": 99.5,
    "max_points": 401,
    "figsize": (16.2, 8.6),
    "dpi": 260,
    "true_color": "#222222",
    "pred_color": "#D55E00",
    "phi_color": "#2ca02c",
    "dirac_color": "#1f77b4",
    "hline_color": "#7CAE00",
    "vline_color": "#00BFC4",
    "residual_color_h": "#8c564b",
    "residual_color_v": "#9467bd",
    "eps_eik": TrainConfig().eps_eik,
}


@lru_cache(maxsize=8)
def _load_reference_points(txt_filename: str) -> Dict[str, np.ndarray]:
    txt_path = Path(txt_filename).expanduser().resolve()
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("%")]
    arr = np.loadtxt(lines)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Expected Possion.txt with at least columns x y T")
    return {
        "xy": arr[:, 0:2].astype(np.float64),
        "T": arr[:, 2].astype(np.float64),
    }


def _build_interpolator(x_axis: np.ndarray, y_axis: np.ndarray, field_map: np.ndarray) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        (np.asarray(x_axis, dtype=np.float64).reshape(-1), np.asarray(y_axis, dtype=np.float64).reshape(-1)),
        np.asarray(field_map, dtype=np.float64),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def _compute_hotspot_from_points(
    xy: np.ndarray,
    residual: np.ndarray,
    *,
    percentile: float,
) -> Tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    valid = np.isfinite(residual)
    if not np.any(valid):
        raise ValueError("Residual values are entirely invalid.")

    threshold = np.nanpercentile(residual[valid], percentile)
    hot_mask = valid & (residual >= threshold)
    if not np.any(hot_mask):
        idx = int(np.nanargmax(residual))
        return xy[idx], residual

    hot_xy = xy[hot_mask]
    hot_w = residual[hot_mask]
    center = np.average(hot_xy, axis=0, weights=hot_w)
    idx = int(np.argmin(np.sum((xy - center[None, :]) ** 2, axis=1)))
    return xy[idx], residual


def _reshape_points_to_grid(
    xy: np.ndarray,
    values: np.ndarray,
    *,
    decimals: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    values = np.asarray(values, dtype=np.float64).reshape(-1)

    x_round = np.round(xy[:, 0], decimals)
    y_round = np.round(xy[:, 1], decimals)
    x_axis = np.unique(x_round)
    y_axis = np.unique(y_round)
    grid = np.full((x_axis.size, y_axis.size), np.nan, dtype=np.float64)

    x_index = {float(v): i for i, v in enumerate(x_axis)}
    y_index = {float(v): j for j, v in enumerate(y_axis)}
    for (xr, yr), val in zip(np.stack([x_round, y_round], axis=1), values):
        grid[x_index[float(xr)], y_index[float(yr)]] = float(val)
    return x_axis, y_axis, grid


def _extract_true_slices(
    xy: np.ndarray,
    values: np.ndarray,
    center: np.ndarray,
    *,
    max_points: int,
    decimals: int = 8,
) -> Dict[str, np.ndarray]:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    x0 = float(np.round(center[0], decimals))
    y0 = float(np.round(center[1], decimals))

    x_round = np.round(xy[:, 0], decimals)
    y_round = np.round(xy[:, 1], decimals)
    mask_h = y_round == y0
    mask_v = x_round == x0
    if not np.any(mask_h) or not np.any(mask_v):
        raise ValueError("Failed to locate true-data slices through the hotspot.")

    xy_h = xy[mask_h]
    val_h = values[mask_h]
    order_h = np.argsort(xy_h[:, 0])
    xy_h = xy_h[order_h]
    val_h = val_h[order_h]

    xy_v = xy[mask_v]
    val_v = values[mask_v]
    order_v = np.argsort(xy_v[:, 1])
    xy_v = xy_v[order_v]
    val_v = val_v[order_v]

    if max_points > 0 and xy_h.shape[0] > max_points:
        idx = np.linspace(0, xy_h.shape[0] - 1, max_points).round().astype(int)
        xy_h = xy_h[idx]
        val_h = val_h[idx]
    if max_points > 0 and xy_v.shape[0] > max_points:
        idx = np.linspace(0, xy_v.shape[0] - 1, max_points).round().astype(int)
        xy_v = xy_v[idx]
        val_v = val_v[idx]

    return {
        "x0": np.asarray([x0], dtype=np.float64),
        "y0": np.asarray([y0], dtype=np.float64),
        "x_line": xy_h[:, 0],
        "y_line": xy_v[:, 1],
        "xy_h": xy_h,
        "xy_v": xy_v,
        "true_h": val_h,
        "true_v": val_v,
    }


def _phi_zero_crossings(coord: np.ndarray, phi_line: np.ndarray) -> np.ndarray:
    coord = np.asarray(coord, dtype=np.float64).reshape(-1)
    phi = np.asarray(phi_line, dtype=np.float64).reshape(-1)
    if coord.size != phi.size or coord.size < 2:
        return np.zeros((0,), dtype=np.float64)

    roots = []
    eps = 1e-14
    for i in range(coord.size - 1):
        c0 = float(coord[i])
        c1 = float(coord[i + 1])
        p0 = float(phi[i])
        p1 = float(phi[i + 1])
        if abs(p0) <= eps:
            roots.append(c0)
        if p0 * p1 < 0.0:
            t = -p0 / (p1 - p0)
            roots.append(c0 + t * (c1 - c0))
    if abs(float(phi[-1])) <= eps:
        roots.append(float(coord[-1]))
    if not roots:
        return np.zeros((0,), dtype=np.float64)

    roots = np.asarray(sorted(roots), dtype=np.float64)
    dedup = [roots[0]]
    tol = max(1e-10, 1e-6 * float(np.max(np.abs(coord)) + 1.0))
    for r in roots[1:]:
        if abs(float(r) - float(dedup[-1])) > tol:
            dedup.append(float(r))
    return np.asarray(dedup, dtype=np.float64)


def _dirac_smooth_np(phi: np.ndarray, epsilon: float) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64)
    eps = float(epsilon)
    return (1.0 / np.pi) * (eps / (eps**2 + phi**2))


def save_T_slice_with_phi_plot_from_fields(
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    T_pred_map: np.ndarray,
    txt_filename: str,
    save_path: Path,
    epoch: int = -1,
    phi_map: Optional[np.ndarray] = None,
    save_npz_path: Optional[Path] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    title_prefix: str = "ADD-PINNs",
    percentile: float = DEFAULT_CONFIG["percentile"],
    max_points: int = DEFAULT_CONFIG["max_points"],
    eps_eik: float = DEFAULT_CONFIG["eps_eik"],
) -> Path:
    cfg = dict(DEFAULT_CONFIG)
    cfg["percentile"] = float(percentile)
    cfg["max_points"] = int(max_points)
    cfg["eps_eik"] = float(eps_eik)

    x_axis = np.asarray(x_axis, dtype=np.float64).reshape(-1)
    y_axis = np.asarray(y_axis, dtype=np.float64).reshape(-1)
    T_pred_map = np.asarray(T_pred_map, dtype=np.float64)
    phi_map = None if phi_map is None else np.asarray(phi_map, dtype=np.float64)
    if bbox is None:
        bbox = (float(x_axis.min()), float(x_axis.max()), float(y_axis.min()), float(y_axis.max()))

    ref = _load_reference_points(str(txt_filename))
    t_interp = _build_interpolator(x_axis, y_axis, T_pred_map)
    phi_interp = None if phi_map is None else _build_interpolator(x_axis, y_axis, phi_map)

    t_pred_on_true = np.asarray(t_interp(ref["xy"]), dtype=np.float64).reshape(-1)
    residual_points = np.abs(t_pred_on_true - ref["T"])
    hotspot_xy, residual_points = _compute_hotspot_from_points(
        ref["xy"],
        residual_points,
        percentile=float(cfg["percentile"]),
    )

    slice_data = _extract_true_slices(
        ref["xy"],
        ref["T"],
        hotspot_xy,
        max_points=int(cfg["max_points"]),
    )
    x0 = float(slice_data["x0"][0])
    y0 = float(slice_data["y0"][0])
    x_line = slice_data["x_line"]
    y_line = slice_data["y_line"]
    xy_h = slice_data["xy_h"]
    xy_v = slice_data["xy_v"]
    true_h = slice_data["true_h"]
    true_v = slice_data["true_v"]

    pred_h = np.asarray(t_interp(xy_h), dtype=np.float64).reshape(-1)
    pred_v = np.asarray(t_interp(xy_v), dtype=np.float64).reshape(-1)
    res_h = np.abs(pred_h - true_h)
    res_v = np.abs(pred_v - true_v)

    phi_h = np.zeros_like(pred_h)
    phi_v = np.zeros_like(pred_v)
    w_h = np.zeros_like(pred_h)
    w_v = np.zeros_like(pred_v)
    phi_zero_x = np.zeros((0,), dtype=np.float64)
    phi_zero_y = np.zeros((0,), dtype=np.float64)
    if phi_interp is not None:
        phi_h = np.asarray(phi_interp(xy_h), dtype=np.float64).reshape(-1)
        phi_v = np.asarray(phi_interp(xy_v), dtype=np.float64).reshape(-1)
        w_h = _dirac_smooth_np(phi_h, float(cfg["eps_eik"]))
        w_v = _dirac_smooth_np(phi_v, float(cfg["eps_eik"]))
        phi_zero_x = _phi_zero_crossings(x_line, phi_h)
        phi_zero_y = _phi_zero_crossings(y_line, phi_v)

    x_axis_true, y_axis_true, residual_map = _reshape_points_to_grid(ref["xy"], residual_points)
    hotspot_idx = int(np.argmin(np.sum((ref["xy"] - hotspot_xy[None, :]) ** 2, axis=1)))
    hotspot_res = float(residual_points[hotspot_idx])

    fig, axes = plt.subplots(2, 3, figsize=tuple(cfg["figsize"]), dpi=int(cfg["dpi"]), constrained_layout=True)

    im = axes[0, 0].imshow(
        residual_map.T,
        origin="lower",
        extent=[x_axis_true[0], x_axis_true[-1], y_axis_true[0], y_axis_true[-1]],
        cmap="magma",
        aspect="equal",
    )
    axes[0, 0].axvline(x0, color=cfg["vline_color"], linestyle="--", linewidth=1.1)
    axes[0, 0].axhline(y0, color=cfg["hline_color"], linestyle="--", linewidth=1.1)
    axes[0, 0].plot([x0], [y0], marker="o", markersize=3.5, color="white", markeredgecolor="black")
    axes[0, 0].set_title(f"Residual Hotspot | x={x0:.3f}, y={y0:.3f}, |err|={hotspot_res:.3e}")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    cb = fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)

    axes[0, 1].plot(x_line, pred_h, color=cfg["pred_color"], linewidth=1.9, label="pred")
    axes[0, 1].plot(x_line, true_h, color=cfg["true_color"], linewidth=1.8, linestyle=":", label="true")
    for x_cross in phi_zero_x:
        axes[0, 1].axvline(float(x_cross), color=cfg["phi_color"], linestyle="-.", linewidth=1.0, alpha=0.9)
    axes[0, 1].set_title(f"Horizontal Slice at y={y0:.3f}")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("T")
    axes[0, 1].grid(True, alpha=0.22)
    axes[0, 1].legend(loc="best", frameon=False)

    axes[1, 0].plot(y_line, pred_v, color=cfg["pred_color"], linewidth=1.9, label="pred")
    axes[1, 0].plot(y_line, true_v, color=cfg["true_color"], linewidth=1.8, linestyle=":", label="true")
    for y_cross in phi_zero_y:
        axes[1, 0].axvline(float(y_cross), color=cfg["phi_color"], linestyle="-.", linewidth=1.0, alpha=0.9)
    axes[1, 0].set_title(f"Vertical Slice at x={x0:.3f}")
    axes[1, 0].set_xlabel("y")
    axes[1, 0].set_ylabel("T")
    axes[1, 0].grid(True, alpha=0.22)
    axes[1, 0].legend(loc="best", frameon=False)

    ax_res = axes[1, 1]
    ax_res.plot(x_line, res_h, color=cfg["residual_color_h"], linewidth=1.8, label=r"$|e|$ horizontal")
    ax_res.plot(y_line, res_v, color=cfg["residual_color_v"], linewidth=1.8, linestyle="--", label=r"$|e|$ vertical")
    ax_res.set_title("Slice Residual Comparison")
    ax_res.set_xlabel("Slice Coordinate")
    ax_res.set_ylabel(r"$|T^{pred}-T^{true}|$")
    ax_res.grid(True, alpha=0.22)
    ax_res.legend(loc="upper right", frameon=False)

    ax_phi_h = axes[0, 2]
    ax_w_h = ax_phi_h.twinx()
    ax_phi_h.plot(x_line, phi_h, color=cfg["hline_color"], linewidth=1.45, label=r"$\phi$ horizontal")
    ax_w_h.plot(x_line, w_h, color=cfg["dirac_color"], linewidth=1.45, linestyle="--", label=r"$w$ horizontal")
    for x_cross in phi_zero_x:
        ax_phi_h.axvline(float(x_cross), color=cfg["phi_color"], linestyle="-.", linewidth=1.0, alpha=0.9)
    ax_phi_h.axhline(0.0, color=cfg["phi_color"], linestyle=":", linewidth=1.0, alpha=0.8)
    ax_phi_h.set_title(f"Horizontal $\\phi$ / $w$ at y={y0:.3f}")
    ax_phi_h.set_xlabel("x")
    ax_phi_h.set_ylabel(r"$\phi$")
    ax_w_h.set_ylabel(r"$w$")
    ax_phi_h.grid(True, alpha=0.22)
    lines_phi_h, labels_phi_h = ax_phi_h.get_legend_handles_labels()
    lines_w_h, labels_w_h = ax_w_h.get_legend_handles_labels()
    ax_phi_h.legend(lines_phi_h + lines_w_h, labels_phi_h + labels_w_h, loc="upper right", frameon=False)

    ax_phi_v = axes[1, 2]
    ax_w_v = ax_phi_v.twinx()
    ax_phi_v.plot(y_line, phi_v, color=cfg["vline_color"], linewidth=1.45, label=r"$\phi$ vertical")
    ax_w_v.plot(y_line, w_v, color=cfg["dirac_color"], linewidth=1.45, linestyle="--", label=r"$w$ vertical")
    for y_cross in phi_zero_y:
        ax_phi_v.axvline(float(y_cross), color=cfg["phi_color"], linestyle="-.", linewidth=1.0, alpha=0.9)
    ax_phi_v.axhline(0.0, color=cfg["phi_color"], linestyle=":", linewidth=1.0, alpha=0.8)
    ax_phi_v.set_title(f"Vertical $\\phi$ / $w$ at x={x0:.3f}")
    ax_phi_v.set_xlabel("y")
    ax_phi_v.set_ylabel(r"$\phi$")
    ax_w_v.set_ylabel(r"$w$")
    ax_phi_v.grid(True, alpha=0.22)
    lines_phi_v, labels_phi_v = ax_phi_v.get_legend_handles_labels()
    lines_w_v, labels_w_v = ax_w_v.get_legend_handles_labels()
    ax_phi_v.legend(lines_phi_v + lines_w_v, labels_phi_v + labels_w_v, loc="upper right", frameon=False)

    fig.suptitle(
        f"{title_prefix} slice with residual, phi, and w | epoch={epoch}, eps_eik={cfg['eps_eik']:.3g}"
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
            hotspot_xy=np.asarray([x0, y0], dtype=np.float64),
            hotspot_res=np.asarray([hotspot_res], dtype=np.float64),
            x_line=x_line,
            y_line=y_line,
            xy_h=xy_h,
            xy_v=xy_v,
            T_pred_h=pred_h,
            T_true_h=true_h,
            T_pred_v=pred_v,
            T_true_v=true_v,
            T_res_h=res_h,
            T_res_v=res_v,
            phi_h=phi_h,
            phi_v=phi_v,
            w_h=w_h,
            w_v=w_v,
            phi_zero_x=phi_zero_x,
            phi_zero_y=phi_zero_y,
            eps_eik=np.asarray([float(cfg["eps_eik"])], dtype=np.float64),
        )

    return save_path


def _pick_temperature_key(data: Dict[str, np.ndarray]) -> str:
    for key in ("T_pred", "T"):
        if key in data:
            return key
    raise KeyError("NPZ must contain 'T_pred' or 'T'.")


def save_T_slice_with_phi_plot(
    npz_path: Path,
    *,
    txt_filename: str = "Possion.txt",
    save_path: Optional[Path] = None,
    save_npz_path: Optional[Path] = None,
    title_prefix: str = "ADD-PINNs",
    percentile: float = DEFAULT_CONFIG["percentile"],
    max_points: int = DEFAULT_CONFIG["max_points"],
    eps_eik: float = DEFAULT_CONFIG["eps_eik"],
) -> Path:
    npz_path = Path(npz_path).expanduser().resolve()
    with np.load(npz_path) as zf:
        data = {k: zf[k] for k in zf.files}

    epoch = -1
    if "epoch" in data:
        epoch_arr = np.asarray(data["epoch"]).reshape(-1)
        if epoch_arr.size > 0:
            epoch = int(epoch_arr[0])

    if save_path is None:
        save_path = npz_path.with_suffix(".slice.png")

    phi_map = data.get("phi")
    bbox_arr = np.asarray(
        data.get(
            "bbox",
            np.array(
                [
                    np.min(data["x"]),
                    np.max(data["x"]),
                    np.min(data["y"]),
                    np.max(data["y"]),
                ],
                dtype=np.float64,
            ),
        ),
        dtype=np.float64,
    ).reshape(-1)

    return save_T_slice_with_phi_plot_from_fields(
        x_axis=np.asarray(data["x"], dtype=np.float64),
        y_axis=np.asarray(data["y"], dtype=np.float64),
        T_pred_map=np.asarray(data[_pick_temperature_key(data)], dtype=np.float64),
        phi_map=None if phi_map is None else np.asarray(phi_map, dtype=np.float64),
        txt_filename=txt_filename,
        save_path=save_path,
        save_npz_path=save_npz_path,
        epoch=epoch,
        bbox=(float(bbox_arr[0]), float(bbox_arr[1]), float(bbox_arr[2]), float(bbox_arr[3])),
        title_prefix=title_prefix,
        percentile=percentile,
        max_points=max_points,
        eps_eik=eps_eik,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate slice-with-residual plots for Poisson outputs.")
    parser.add_argument("--npz", type=str, default="output_roi_off/data_output/final_fields.npz")
    parser.add_argument("--txt", type=str, default="Possion.txt")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--title-prefix", type=str, default="ADD-PINNs")
    parser.add_argument("--percentile", type=float, default=DEFAULT_CONFIG["percentile"])
    parser.add_argument("--max-points", type=int, default=DEFAULT_CONFIG["max_points"])
    parser.add_argument("--eps-eik", type=float, default=DEFAULT_CONFIG["eps_eik"])
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    npz_path = (project_root / args.npz).resolve() if not Path(args.npz).is_absolute() else Path(args.npz).resolve()
    txt_path = (project_root / args.txt).resolve() if not Path(args.txt).is_absolute() else Path(args.txt).resolve()
    out_path = None
    if args.out:
        out_path = (project_root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()

    saved = save_T_slice_with_phi_plot(
        npz_path,
        txt_filename=str(txt_path),
        save_path=out_path,
        save_npz_path=None if out_path is None else out_path.with_suffix(".npz"),
        title_prefix=args.title_prefix,
        percentile=args.percentile,
        max_points=args.max_points,
        eps_eik=args.eps_eik,
    )
    print(f"[Saved] {saved}")


if __name__ == "__main__":
    main()
