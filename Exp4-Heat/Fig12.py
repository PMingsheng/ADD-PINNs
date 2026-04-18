#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator


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
    phi_dir = project_root / "output_roi_on" / "phi_snapshots"
    phi_snapshots = sorted(phi_dir.glob("phi_epoch_*.npz"))
    return {
        "pinn": pinn_final,
        "apinn": apinn_final,
        "pimoe": pimoe_final,
        "pimoe_pde": _latest_npz(phi_snapshots),
    }


def load_reference_dataset(txt_path: Path) -> Dict[str, np.ndarray]:
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("%")]
    arr = np.loadtxt(lines)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Expected Possion.txt with at least columns x y T")
    return {
        "xy": arr[:, 0:2].astype(np.float64),
        "T": arr[:, 2].astype(np.float64),
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


def load_temperature_maps(npz_path: Path, *, reference: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
        y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
        t_key = _first_existing(data, ("T_pred", "T"))
        if t_key is None:
            raise KeyError(f"{npz_path} missing temperature field.")
        t_pred = np.asarray(data[t_key], dtype=np.float64)
        bbox = _bbox_from_npz(data, t_pred)
        grid_shape = t_pred.shape

    t_pred_scatter = _sample_grid_field_at_points(t_pred, x, y, reference["xy"])
    t_true_scatter = np.asarray(reference["T"], dtype=np.float64).reshape(-1)

    return {
        "bbox": bbox,
        "shape": np.asarray(grid_shape, dtype=np.int64),
        "T_true": _scatter_to_grid(reference["xy"], t_true_scatter, bbox, grid_shape),
        "T_res": _scatter_to_grid(reference["xy"], np.abs(t_pred_scatter - t_true_scatter), bbox, grid_shape),
    }


def load_pde_residual_map(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        key = _first_existing(data, ("pde_residual", "grad_pde_residual"))
        if key is None:
            raise KeyError(f"{npz_path} missing PDE residual-like field.")
        residual = np.asarray(data[key], dtype=np.float64)
        bbox = _bbox_from_npz(data, residual)
    return {
        "true": np.zeros_like(residual, dtype=np.float64),
        "res": np.abs(residual),
        "bbox": bbox,
    }


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


def _finite_max(*arrays: np.ndarray) -> float:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=np.float64)
        finite = a[np.isfinite(a)]
        if finite.size > 0:
            vals.append(float(np.max(finite)))
    if not vals:
        return 1.0
    vmax = max(vals)
    return vmax if vmax > 0.0 else 1.0


def save_fig12_comparison(
    *,
    txt_path: Path,
    pinn_path: Path,
    apinn_path: Path,
    pimoe_path: Path,
    pimoe_pde_path: Path,
    out_path: Path,
    save_pdf: bool = False,
) -> Path:
    txt_path = Path(txt_path).expanduser().resolve()
    pinn_path = Path(pinn_path).expanduser().resolve()
    apinn_path = Path(apinn_path).expanduser().resolve()
    pimoe_path = Path(pimoe_path).expanduser().resolve()
    pimoe_pde_path = Path(pimoe_pde_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()

    for path in (txt_path, pinn_path, apinn_path, pimoe_path, pimoe_pde_path):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    reference = load_reference_dataset(txt_path)
    pinn = load_temperature_maps(pinn_path, reference=reference)
    apinn = load_temperature_maps(apinn_path, reference=reference)
    pimoe = load_temperature_maps(pimoe_path, reference=reference)

    pinn_pde = load_pde_residual_map(pinn_path)
    apinn_pde = load_pde_residual_map(apinn_path)
    pimoe_pde = load_pde_residual_map(pimoe_pde_path)

    t_res_vmax = _finite_max(pinn["T_res"], apinn["T_res"], pimoe["T_res"])
    t_true = pimoe["T_true"]
    t_true_bbox = pimoe["bbox"]
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.0), dpi=250, constrained_layout=True)

    im00 = _plot_map(axes[0, 0], t_true, t_true_bbox, title=r"$T$ True", cmap="viridis")
    im01 = _plot_map(axes[0, 1], pinn["T_res"], pinn["bbox"], title=r"$T$ Residual (PINNs)", cmap="magma", vmin=0.0, vmax=t_res_vmax)
    im02 = _plot_map(axes[0, 2], apinn["T_res"], apinn["bbox"], title=r"$T$ Residual (APINNs)", cmap="magma", vmin=0.0, vmax=t_res_vmax)
    im03 = _plot_map(axes[0, 3], pimoe["T_res"], pimoe["bbox"], title=r"$T$ Residual (ADD-PINNs)", cmap="magma", vmin=0.0, vmax=t_res_vmax)

    im10 = _plot_map(axes[1, 0], pimoe_pde["true"], pimoe_pde["bbox"], title=r"PDE True (=0)", cmap="viridis")
    im11 = _plot_map(axes[1, 1], pinn_pde["res"], pinn_pde["bbox"], title=r"PDE Residual (PINNs)", cmap="magma", vmin=0.0)
    im12 = _plot_map(axes[1, 2], apinn_pde["res"], apinn_pde["bbox"], title=r"PDE Residual (APINNs)", cmap="magma", vmin=0.0)
    im13 = _plot_map(axes[1, 3], pimoe_pde["res"], pimoe_pde["bbox"], title=r"PDE Residual (ADD-PINNs)", cmap="magma", vmin=0.0)

    for i, j, im in (
        (0, 0, im00), (0, 1, im01), (0, 2, im02), (0, 3, im03),
        (1, 0, im10), (1, 1, im11), (1, 2, im12), (1, 3, im13),
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
    plt.close(fig)
    return png_path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    defaults = pick_default_model_paths(script_dir)
    default_out = script_dir.parent / "Figure" / "Fig12.png"

    parser = argparse.ArgumentParser(description="Poisson analog of Ellipse Fig8: temperature and PDE residual comparisons.")
    parser.add_argument("--txt", type=str, default=str(script_dir / "Possion.txt"))
    parser.add_argument("--pinn", type=str, default=str(defaults["pinn"]))
    parser.add_argument("--apinn", type=str, default=str(defaults["apinn"]))
    parser.add_argument("--pimoe", type=str, default=str(defaults["pimoe"]))
    parser.add_argument("--pimoe-pde", type=str, default=str(defaults["pimoe_pde"]))
    parser.add_argument("--out", type=str, default=str(default_out))
    parser.add_argument("--save-pdf", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    out_path = save_fig12_comparison(
        txt_path=Path(args.txt),
        pinn_path=Path(args.pinn),
        apinn_path=Path(args.apinn),
        pimoe_path=Path(args.pimoe),
        pimoe_pde_path=Path(args.pimoe_pde),
        out_path=Path(args.out),
        save_pdf=bool(args.save_pdf),
    )
    print(f"Saved: {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
