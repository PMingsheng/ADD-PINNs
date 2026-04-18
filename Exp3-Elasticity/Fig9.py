#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator


def _true_phi_signed_on_grid(
    bbox: np.ndarray,
    shape: Tuple[int, int],
    *,
    ellipse: Tuple[float, float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(bbox[0], bbox[1], shape[0], dtype=np.float64)
    ys = np.linspace(bbox[2], bbox[3], shape[1], dtype=np.float64)
    xg, yg = np.meshgrid(xs, ys, indexing="ij")

    xc, yc, a, b, gamma = [float(v) for v in ellipse]
    cg = np.cos(gamma)
    sg = np.sin(gamma)
    dx = xg - xc
    dy = yg - yc
    xp = cg * dx + sg * dy
    yp = -sg * dx + cg * dy
    phi_true = (xp / a) ** 2 + (yp / b) ** 2 - 1.0
    return xg, yg, phi_true


def _first_existing(data: np.lib.npyio.NpzFile, keys: Tuple[str, ...]) -> str | None:
    for key in keys:
        if key in data:
            return key
    return None


def _bbox_from_npz(data: np.lib.npyio.NpzFile, field_map: np.ndarray) -> np.ndarray:
    if "bbox" in data:
        bbox = np.asarray(data["bbox"], dtype=np.float64).reshape(-1)
        if bbox.size >= 4:
            return bbox[:4]
    if "x" in data and "y" in data:
        x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
        y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
        return np.asarray([x.min(), x.max(), y.min(), y.max()], dtype=np.float64)
    n1, n2 = field_map.shape
    return np.asarray([0.0, float(n1 - 1), 0.0, float(n2 - 1)], dtype=np.float64)


def _extract_epoch_from_name(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else -1


def _latest_npz(paths: Iterable[Path]) -> Path:
    valid = [p for p in paths if p.exists()]
    if not valid:
        raise FileNotFoundError("No candidate npz files found.")
    return max(valid, key=lambda p: (_extract_epoch_from_name(p.stem), p.stat().st_mtime))


def pick_default_model_paths(project_root: Path) -> Dict[str, Path]:
    pinn_final = project_root / "outputs_pinn_single" / "final_fields.npz"
    apinn_final = project_root / "outputs_apinn" / "final_fields.npz"
    pimoe_final = project_root / "output_roi_on" / "data_output" / "final_fields.npz"

    pinn_snapshots = sorted((project_root / "outputs_pinn_single" / "snapshots").glob("epoch_*.npz"))
    apinn_snapshots = sorted((project_root / "outputs_apinn" / "snapshots").glob("epoch_*.npz"))

    return {
        "pinn": pinn_final if pinn_final.exists() else _latest_npz(pinn_snapshots),
        "apinn": apinn_final if apinn_final.exists() else _latest_npz(apinn_snapshots),
        "pimoe": pimoe_final,
        "phi_dir": project_root / "output_roi_on" / "phi_snapshots",
    }


def load_reference_dataset(txt_path: Path) -> Dict[str, np.ndarray]:
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


def _sample_grid_field_at_points(
    field_map: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    xy_points: np.ndarray,
) -> np.ndarray:
    interp = RegularGridInterpolator(
        (np.asarray(x_axis, dtype=np.float64).reshape(-1), np.asarray(y_axis, dtype=np.float64).reshape(-1)),
        np.asarray(field_map, dtype=np.float64),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    return np.asarray(interp(np.asarray(xy_points, dtype=np.float64)), dtype=np.float64).reshape(-1)


def _scatter_to_grid(
    xy_points: np.ndarray,
    values: np.ndarray,
    bbox: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    xs = np.linspace(float(bbox[0]), float(bbox[1]), int(shape[0]), dtype=np.float64)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), int(shape[1]), dtype=np.float64)
    xg, yg = np.meshgrid(xs, ys, indexing="ij")
    interp = LinearNDInterpolator(np.asarray(xy_points, dtype=np.float64), np.asarray(values, dtype=np.float64).reshape(-1))
    grid = np.asarray(interp(xg, yg), dtype=np.float64)
    return grid.reshape(int(shape[0]), int(shape[1]))


def load_displacement_maps(npz_path: Path, *, reference: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
        y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
        ux_key = _first_existing(data, ("ux_pred", "ux"))
        uy_key = _first_existing(data, ("uy_pred", "uy"))
        if ux_key is None or uy_key is None:
            raise KeyError(f"{npz_path} missing ux/uy predictions.")

        ux_pred = np.asarray(data[ux_key], dtype=np.float64)
        uy_pred = np.asarray(data[uy_key], dtype=np.float64)
        bbox = _bbox_from_npz(data, ux_pred)
        grid_shape = ux_pred.shape

    ux_pred_scatter = _sample_grid_field_at_points(ux_pred, x, y, reference["xy"])
    uy_pred_scatter = _sample_grid_field_at_points(uy_pred, x, y, reference["xy"])
    ux_true_scatter = np.asarray(reference["ux"], dtype=np.float64).reshape(-1)
    uy_true_scatter = np.asarray(reference["uy"], dtype=np.float64).reshape(-1)

    return {
        "x": x,
        "y": y,
        "bbox": bbox,
        "ux_true": _scatter_to_grid(reference["xy"], ux_true_scatter, bbox, grid_shape),
        "uy_true": _scatter_to_grid(reference["xy"], uy_true_scatter, bbox, grid_shape),
        "ux_res": _scatter_to_grid(reference["xy"], np.abs(ux_pred_scatter - ux_true_scatter), bbox, grid_shape),
        "uy_res": _scatter_to_grid(reference["xy"], np.abs(uy_pred_scatter - uy_true_scatter), bbox, grid_shape),
    }


def load_pde_residual_map(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        key = _first_existing(data, ("pde_residual", "f_residual", "f_pred"))
        if key is None:
            raise KeyError(f"{npz_path} missing PDE residual-like field.")
        residual = np.asarray(data[key], dtype=np.float64)
        bbox = _bbox_from_npz(data, residual)
    return {
        "true": np.zeros_like(residual, dtype=np.float64),
        "res": np.abs(residual),
        "bbox": bbox,
    }


def load_phi_snapshot(npz_path: Path) -> Dict[str, np.ndarray | int]:
    with np.load(npz_path) as data:
        if "phi" not in data:
            raise KeyError(f"{npz_path} missing phi")
        phi = np.asarray(data["phi"], dtype=np.float64)
        bbox = _bbox_from_npz(data, phi)
        epoch = int(np.asarray(data["epoch"]).reshape(-1)[0]) if "epoch" in data else _extract_epoch_from_name(npz_path.stem)
    return {"phi": phi, "bbox": bbox, "epoch": epoch}


def choose_phi_snapshots(phi_dir: Path, desired_epochs: Iterable[int]) -> List[Path]:
    all_paths = sorted(phi_dir.glob("phi_epoch_*.npz"))
    if not all_paths:
        raise FileNotFoundError(f"No phi snapshots found in {phi_dir}")
    epoch_to_path = {int(_extract_epoch_from_name(p.stem)): p for p in all_paths}
    selected = []
    available_epochs = np.asarray(sorted(epoch_to_path.keys()), dtype=np.int64)
    for epoch in desired_epochs:
        if epoch in epoch_to_path:
            selected.append(epoch_to_path[epoch])
            continue
        nearest = int(available_epochs[np.abs(available_epochs - int(epoch)).argmin()])
        selected.append(epoch_to_path[nearest])
    return selected


def _plot_map(ax, arr: np.ndarray, bbox: np.ndarray, *, title: str, cmap: str, vmin=None, vmax=None):
    im = ax.imshow(
        np.ma.masked_invalid(np.asarray(arr, dtype=np.float64)).T,
        origin="lower",
        extent=[bbox[0], bbox[1], bbox[2], bbox[3]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return im


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    defaults = pick_default_model_paths(script_dir)
    default_out = script_dir.parent / "Figure" / "Fig9.png"

    parser = argparse.ArgumentParser(description="Ellipse analog of Fig4: ux/uy true and residual maps with phi evolution.")
    parser.add_argument("--txt", type=str, default=str(script_dir / "Ellipse.txt"))
    parser.add_argument("--pinn", type=str, default=str(defaults["pinn"]))
    parser.add_argument("--apinn", type=str, default=str(defaults["apinn"]))
    parser.add_argument("--pimoe", type=str, default=str(defaults["pimoe"]))
    parser.add_argument("--phi-dir", type=str, default=str(defaults["phi_dir"]))
    parser.add_argument("--out", type=str, default=str(default_out))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    txt_path = Path(args.txt).expanduser().resolve()
    pinn_path = Path(args.pinn).expanduser().resolve()
    apinn_path = Path(args.apinn).expanduser().resolve()
    pimoe_path = Path(args.pimoe).expanduser().resolve()
    phi_dir = Path(args.phi_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    for path in (txt_path, pinn_path, apinn_path, pimoe_path):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
    if not phi_dir.exists():
        raise FileNotFoundError(f"Directory not found: {phi_dir}")

    reference = load_reference_dataset(txt_path)
    pinn = load_displacement_maps(pinn_path, reference=reference)
    apinn = load_displacement_maps(apinn_path, reference=reference)
    pimoe = load_displacement_maps(pimoe_path, reference=reference)
    pinn_f = load_pde_residual_map(pinn_path)
    apinn_f = load_pde_residual_map(apinn_path)

    ux_res_max = max(
        float(np.nanmax(pinn["ux_res"])),
        float(np.nanmax(apinn["ux_res"])),
        float(np.nanmax(pimoe["ux_res"])),
    )
    uy_res_max = max(
        float(np.nanmax(pinn["uy_res"])),
        float(np.nanmax(apinn["uy_res"])),
        float(np.nanmax(pimoe["uy_res"])),
    )
    ux_res_max = ux_res_max if ux_res_max > 0 else 1.0
    uy_res_max = uy_res_max if uy_res_max > 0 else 1.0

    pimoe_f = load_pde_residual_map(phi_dir / "phi_epoch_00060000.npz")
    f_res_max = max(
        float(np.nanmax(pinn_f["res"])),
        float(np.nanmax(apinn_f["res"])),
        float(np.nanmax(pimoe_f["res"])),
    )
    f_res_max = f_res_max if f_res_max > 0 else 1.0

    ux_true = pimoe["ux_true"]
    uy_true = pimoe["uy_true"]
    ux_true_bbox = pimoe["bbox"]
    uy_true_bbox = pimoe["bbox"]

    fig, axes = plt.subplots(3, 4, figsize=(13.5, 8.9), dpi=250, constrained_layout=True)

    im00 = _plot_map(axes[0, 0], ux_true, ux_true_bbox, title=r"$u_x$ True", cmap="viridis")
    im01 = _plot_map(axes[0, 1], pinn["ux_res"], pinn["bbox"], title=r"$u_x$ Residual (PINNs)", cmap="magma", vmin=0.0, vmax=ux_res_max)
    im02 = _plot_map(axes[0, 2], apinn["ux_res"], apinn["bbox"], title=r"$u_x$ Residual (APINNs)", cmap="magma", vmin=0.0, vmax=ux_res_max)
    im03 = _plot_map(axes[0, 3], pimoe["ux_res"], pimoe["bbox"], title=r"$u_x$ Residual (ADD-PINNs)", cmap="magma", vmin=0.0, vmax=ux_res_max)

    im10 = _plot_map(axes[1, 0], uy_true, uy_true_bbox, title=r"$u_y$ True", cmap="viridis")
    im11 = _plot_map(axes[1, 1], pinn["uy_res"], pinn["bbox"], title=r"$u_y$ Residual (PINNs)", cmap="magma", vmin=0.0, vmax=uy_res_max)
    im12 = _plot_map(axes[1, 2], apinn["uy_res"], apinn["bbox"], title=r"$u_y$ Residual (APINNs)", cmap="magma", vmin=0.0, vmax=uy_res_max)
    im13 = _plot_map(axes[1, 3], pimoe["uy_res"], pimoe["bbox"], title=r"$u_y$ Residual (ADD-PINNs)", cmap="magma", vmin=0.0, vmax=uy_res_max)

    im20 = _plot_map(axes[2, 0], pimoe_f["true"], pimoe_f["bbox"], title=r"$f$ True (=0)", cmap="viridis")
    im21 = _plot_map(axes[2, 1], pinn_f["res"], pinn_f["bbox"], title=r"$f$ Residual (PINNs)", cmap="magma", vmin=0.0, vmax=f_res_max)
    im22 = _plot_map(axes[2, 2], apinn_f["res"], apinn_f["bbox"], title=r"$f$ Residual (APINNs)", cmap="magma", vmin=0.0, vmax=f_res_max)
    im23 = _plot_map(axes[2, 3], pimoe_f["res"], pimoe_f["bbox"], title=r"$f$ Residual (ADD-PINNs)", cmap="magma", vmin=0.0, vmax=f_res_max)

    for i, j, im in (
        (0, 0, im00), (0, 1, im01), (0, 2, im02), (0, 3, im03),
        (1, 0, im10), (1, 1, im11), (1, 2, im12), (1, 3, im13),
        (2, 0, im20), (2, 1, im21), (2, 2, im22), (2, 3, im23),
    ):
        cb = fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.03)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        cb.formatter = formatter
        cb.update_ticks()
        cb.ax.tick_params(labelsize=7)
        cb.ax.yaxis.set_offset_position("left")
        cb.ax.yaxis.get_offset_text().set(size=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_path.with_suffix(".png")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
