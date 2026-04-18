#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator


rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["mathtext.fontset"] = "stixsans"
rcParams["font.size"] = 7
rcParams["axes.labelsize"] = 7
rcParams["axes.titlesize"] = 8
rcParams["xtick.labelsize"] = 6
rcParams["ytick.labelsize"] = 6
rcParams["legend.fontsize"] = 6
rcParams["axes.linewidth"] = 0.6
rcParams["grid.linewidth"] = 0.45
rcParams["lines.linewidth"] = 1.25

C_TRUE = "#222222"
C_PIMOE = "#D55E00"
C_APINN = "#009E73"
C_PINN = "#0072B2"


def _load_reference_dataset(txt_path: Path) -> Dict[str, np.ndarray]:
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("%")]
    arr = np.loadtxt(lines)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Expected Possion.txt with at least columns x y T")
    return {
        "xy": arr[:, 0:2].astype(np.float64),
        "T": arr[:, 2].astype(np.float64),
    }


def _first_existing(data: np.lib.npyio.NpzFile, keys: Tuple[str, ...]) -> str | None:
    for key in keys:
        if key in data:
            return key
    return None


def _load_model_temperature(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        key = _first_existing(data, ("T_pred", "T"))
        if key is None:
            raise KeyError(f"{npz_path} missing temperature field.")
        x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
        y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
        t_map = np.asarray(data[key], dtype=np.float64)
        if "bbox" in data:
            bbox = np.asarray(data["bbox"], dtype=np.float64).reshape(-1)[:4]
        else:
            bbox = np.asarray([x.min(), x.max(), y.min(), y.max()], dtype=np.float64)
    interp = RegularGridInterpolator(
        (x, y),
        t_map,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    return {
        "x": x,
        "y": y,
        "T": t_map,
        "bbox": bbox,
        "interp": interp,
    }


def _compute_hotspot_from_points(
    xy: np.ndarray,
    residual: np.ndarray,
    *,
    percentile: float,
) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    valid = np.isfinite(residual)
    if not np.any(valid):
        raise ValueError("Residual values are entirely invalid.")

    threshold = np.nanpercentile(residual[valid], percentile)
    hot_mask = valid & (residual >= threshold)
    if not np.any(hot_mask):
        return xy[int(np.nanargmax(residual))]

    hot_xy = xy[hot_mask]
    hot_w = residual[hot_mask]
    center = np.average(hot_xy, axis=0, weights=hot_w)
    idx = int(np.argmin(np.sum((hot_xy - center[None, :]) ** 2, axis=1)))
    return hot_xy[idx]


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
        "x_line": xy_h[:, 0],
        "y_line": xy_v[:, 1],
        "xy_h": xy_h,
        "xy_v": xy_v,
        "true_h": val_h,
        "true_v": val_v,
        "x0": x0,
        "y0": y0,
    }


def _eval_interp(interp, xy: np.ndarray) -> np.ndarray:
    return np.asarray(interp(np.asarray(xy, dtype=np.float64)), dtype=np.float64).reshape(-1)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    out_default = script_dir.parent / "Figure" / "ResidualProfile.png"

    parser = argparse.ArgumentParser(
        description="Plot cross-sections through the high-temperature-residual region in ADD-PINNs-Possion."
    )
    parser.add_argument("--txt", type=str, default=str(script_dir / "Possion.txt"))
    parser.add_argument("--pimoe", type=str, default=str(script_dir / "output_roi_off" / "data_output" / "final_fields.npz"))
    parser.add_argument("--apinn", type=str, default=str(script_dir / "outputs_apinn" / "final_fields.npz"))
    parser.add_argument("--pinn", type=str, default=str(script_dir / "outputs_pinn_single" / "final_fields.npz"))
    parser.add_argument("--out", type=str, default=str(out_default))
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--samples", type=int, default=401)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    txt_path = Path(args.txt).expanduser().resolve()
    pimoe_path = Path(args.pimoe).expanduser().resolve()
    apinn_path = Path(args.apinn).expanduser().resolve()
    pinn_path = Path(args.pinn).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    for path in (txt_path, pimoe_path, apinn_path, pinn_path):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    ref = _load_reference_dataset(txt_path)

    pimoe = _load_model_temperature(pimoe_path)
    apinn = _load_model_temperature(apinn_path)
    pinn = _load_model_temperature(pinn_path)

    pimoe_on_true = _eval_interp(pimoe["interp"], ref["xy"])
    pimoe_point_res = np.abs(pimoe_on_true - ref["T"])
    hotspot = _compute_hotspot_from_points(ref["xy"], pimoe_point_res, percentile=float(args.percentile))

    line_data = _extract_true_slices(ref["xy"], ref["T"], hotspot, max_points=int(args.samples))
    x_line = line_data["x_line"]
    y_line = line_data["y_line"]
    xy_h = line_data["xy_h"]
    xy_v = line_data["xy_v"]
    true_h = line_data["true_h"]
    true_v = line_data["true_v"]
    x0 = float(line_data["x0"])
    y0 = float(line_data["y0"])

    x_axis_true, y_axis_true, residual_map = _reshape_points_to_grid(ref["xy"], pimoe_point_res)

    model_specs = {
        "ADD-PINNs": (pimoe, C_PIMOE),
        "APINN": (apinn, C_APINN),
        "PINN": (pinn, C_PINN),
    }

    pred_h: Dict[str, np.ndarray] = {}
    pred_v: Dict[str, np.ndarray] = {}
    res_h: Dict[str, np.ndarray] = {}
    res_v: Dict[str, np.ndarray] = {}
    for model_name, (model_data, _color) in model_specs.items():
        pred_h[model_name] = _eval_interp(model_data["interp"], xy_h)
        pred_v[model_name] = _eval_interp(model_data["interp"], xy_v)
        res_h[model_name] = np.abs(pred_h[model_name] - true_h)
        res_v[model_name] = np.abs(pred_v[model_name] - true_v)

    hotspot_res = float(np.abs(_eval_interp(pimoe["interp"], hotspot[None, :])[0] - ref["T"][np.argmin(np.sum((ref["xy"] - hotspot[None, :]) ** 2, axis=1))]))

    fig, axes = plt.subplots(2, 2, figsize=(7.08, 5.2), dpi=300, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=1 / 72, h_pad=2 / 72, wspace=0.08, hspace=0.08)

    im = axes[0, 0].imshow(
        np.asarray(residual_map, dtype=np.float64).T,
        origin="lower",
        extent=[x_axis_true[0], x_axis_true[-1], y_axis_true[0], y_axis_true[-1]],
        cmap="magma",
        aspect="equal",
    )
    axes[0, 0].axvline(x0, color="#00BFC4", linestyle="--", linewidth=1.0)
    axes[0, 0].axhline(y0, color="#7CAE00", linestyle="--", linewidth=1.0)
    axes[0, 0].plot([x0], [y0], marker="o", markersize=3.5, color="white", markeredgecolor="black")
    axes[0, 0].set_title(f"ADD-PINNs T Residual Hotspot\n(x={x0:.3f}, y={y0:.3f}, |err|={hotspot_res:.4e})", pad=4)
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    cb = fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)

    axes[0, 1].plot(x_line, true_h, color=C_TRUE, linestyle="-", linewidth=1.3, label="True")
    for model_name, (_model_data, color) in model_specs.items():
        axes[0, 1].plot(x_line, pred_h[model_name], color=color, linestyle="-", linewidth=1.2, label=model_name)
    axes[0, 1].set_title(f"Horizontal Slice at y={y0:.3f}", pad=4)
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("T")
    axes[0, 1].grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    axes[0, 1].legend(frameon=False, loc="best", ncol=2, handlelength=2.0)

    axes[1, 0].plot(y_line, true_v, color=C_TRUE, linestyle="-", linewidth=1.3, label="True")
    for model_name, (_model_data, color) in model_specs.items():
        axes[1, 0].plot(y_line, pred_v[model_name], color=color, linestyle="-", linewidth=1.2, label=model_name)
    axes[1, 0].set_title(f"Vertical Slice at x={x0:.3f}", pad=4)
    axes[1, 0].set_xlabel("y")
    axes[1, 0].set_ylabel("T")
    axes[1, 0].grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    for model_name, (_model_data, color) in model_specs.items():
        axes[1, 1].plot(x_line, res_h[model_name], color=color, linestyle="-", linewidth=1.2)
        axes[1, 1].plot(y_line, res_v[model_name], color=color, linestyle="--", linewidth=1.2)
    axes[1, 1].set_title("Slice Residual Comparison", pad=4)
    axes[1, 1].set_xlabel("Coordinate Along Slice")
    axes[1, 1].set_ylabel(r"$|T^{pred}-T^{true}|$")
    axes[1, 1].grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    model_handles = [
        Line2D([0], [0], color=C_TRUE, lw=1.3, ls="-", label="True"),
        Line2D([0], [0], color=C_PIMOE, lw=1.2, ls="-", label="ADD-PINNs"),
        Line2D([0], [0], color=C_APINN, lw=1.2, ls="-", label="APINN"),
        Line2D([0], [0], color=C_PINN, lw=1.2, ls="-", label="PINN"),
    ]
    slice_handles = [
        Line2D([0], [0], color="black", lw=1.1, ls="-", label="Horizontal"),
        Line2D([0], [0], color="black", lw=1.1, ls="--", label="Vertical"),
    ]
    legend_models = axes[1, 1].legend(
        handles=model_handles[1:],
        frameon=False,
        loc="upper right",
        ncol=1,
        handlelength=2.0,
    )
    axes[1, 1].add_artist(legend_models)
    axes[1, 1].legend(
        handles=slice_handles,
        frameon=False,
        loc="upper left",
        ncol=1,
        handlelength=2.0,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"), dpi=400, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {out_path}")
    print(f"Hotspot: x={x0:.6f}, y={y0:.6f}, abs_err={hotspot_res:.6e}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
