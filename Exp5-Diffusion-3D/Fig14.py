#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

C1 = (0.4, 0.5, 0.5)


def pick_default_model_paths(project_root: Path) -> Dict[str, Path]:
    return {
        "pinn": project_root / "outputs_pinn3d_c1_sphere" / "data_output" / "final_fields.npz",
        "apinn": project_root / "outputs_apinn3d_c1_sphere" / "data_output" / "final_fields.npz",
        "pimoe": project_root / "outputs_add_pinns3d_c1_sphere" / "data_output" / "final_fields.npz",
    }


def _load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _nearest_index(arr: np.ndarray, value: float) -> int:
    return int(np.abs(np.asarray(arr, dtype=np.float64) - float(value)).argmin())


def _extract_yz_slice(field_3d: np.ndarray, x_axis: np.ndarray, x_value: float) -> Tuple[np.ndarray, int, float]:
    idx = _nearest_index(x_axis, x_value)
    return np.asarray(field_3d[idx, :, :], dtype=np.float64).T, idx, float(x_axis[idx])


def _plot_slice(ax, arr: np.ndarray, y: np.ndarray, z: np.ndarray, *, title: str, cmap: str, vmin=None, vmax=None):
    im = ax.imshow(
        np.ma.masked_invalid(arr),
        origin="lower",
        extent=[float(y.min()), float(y.max()), float(z.min()), float(z.max())],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    return im


def _plot_phi_evolution_slice(
    ax,
    phi_pred: np.ndarray,
    phi_true: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    epoch: int,
):
    yg, zg = np.meshgrid(np.asarray(y, dtype=np.float64), np.asarray(z, dtype=np.float64), indexing="ij")
    ax.set_facecolor("white")
    ax.contour(yg, zg, np.asarray(phi_true, dtype=np.float64).T, levels=[0.0], colors="blue", linewidths=2.0)
    ax.contour(yg, zg, np.asarray(phi_pred, dtype=np.float64).T, levels=[0.0], colors="red", linewidths=2.0)
    ax.set_xlim(float(np.min(y)), float(np.max(y)))
    ax.set_ylim(float(np.min(z)), float(np.max(z)))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(rf"$\phi$ (epoch {epoch})")
    ax.set_xlabel("y")
    ax.set_ylabel("z")


def _load_phi_snapshot(npz_path: Path) -> Dict[str, np.ndarray | int]:
    with np.load(npz_path) as data:
        if "phi_pred" not in data or "phi_true" not in data:
            raise KeyError(f"{npz_path} missing phi_pred/phi_true.")
        epoch = int(np.asarray(data["epoch"]).reshape(-1)[0]) if "epoch" in data else -1
        return {
            "epoch": epoch,
            "x": np.asarray(data["x"], dtype=np.float64).reshape(-1),
            "y": np.asarray(data["y"], dtype=np.float64).reshape(-1),
            "z": np.asarray(data["z"], dtype=np.float64).reshape(-1),
            "phi_pred": np.asarray(data["phi_pred"], dtype=np.float64),
            "phi_true": np.asarray(data["phi_true"], dtype=np.float64),
        }


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


def save_fig13_comparison(
    *,
    pinn_path: Path,
    apinn_path: Path,
    pimoe_path: Path,
    phi_dir: Path,
    out_path: Path,
    save_pdf: bool = True,
) -> Path:
    pinn = _load_npz(Path(pinn_path).expanduser().resolve())
    apinn = _load_npz(Path(apinn_path).expanduser().resolve())
    pimoe = _load_npz(Path(pimoe_path).expanduser().resolve())

    y = np.asarray(pimoe["y"], dtype=np.float64).reshape(-1)
    z = np.asarray(pimoe["z"], dtype=np.float64).reshape(-1)
    x = np.asarray(pimoe["x"], dtype=np.float64).reshape(-1)

    u_true_slice, _, x_used = _extract_yz_slice(pimoe["u_true"], x, C1[0])
    zero_slice = np.zeros_like(u_true_slice, dtype=np.float64)

    pinn_u_res, _, _ = _extract_yz_slice(np.abs(pinn["u_residual"]), np.asarray(pinn["x"]), C1[0])
    apinn_u_res, _, _ = _extract_yz_slice(np.abs(apinn["u_residual"]), np.asarray(apinn["x"]), C1[0])
    pimoe_u_res, _, _ = _extract_yz_slice(np.abs(pimoe["u_residual"]), x, C1[0])

    pinn_f_res, _, _ = _extract_yz_slice(np.abs(pinn["f_residual"]), np.asarray(pinn["x"]), C1[0])
    apinn_f_res, _, _ = _extract_yz_slice(np.abs(apinn["f_residual"]), np.asarray(apinn["x"]), C1[0])
    pimoe_f_res, _, _ = _extract_yz_slice(np.abs(pimoe["f_residual"]), x, C1[0])

    u_res_vmax = _finite_max(pinn_u_res, apinn_u_res, pimoe_u_res)
    phi_epochs = [1000, 2000, 4000, 10000]
    phi_items = [_load_phi_snapshot(phi_dir / f"phi_heatmap_{epoch:08d}.npz") for epoch in phi_epochs]
    phi_slices = []
    for item in phi_items:
        phi_pred_slice, _, _ = _extract_yz_slice(item["phi_pred"], item["x"], C1[0])
        phi_true_slice, _, _ = _extract_yz_slice(item["phi_true"], item["x"], C1[0])
        phi_slices.append(
            {
                "epoch": int(item["epoch"]),
                "y": np.asarray(item["y"], dtype=np.float64).reshape(-1),
                "z": np.asarray(item["z"], dtype=np.float64).reshape(-1),
                "phi_pred": phi_pred_slice,
                "phi_true": phi_true_slice,
            }
        )

    fig, axes = plt.subplots(3, 4, figsize=(13.5, 8.9), dpi=250, constrained_layout=True)

    im00 = _plot_slice(axes[0, 0], u_true_slice, y, z, title=rf"$u$ True @ $x={x_used:.3f}$", cmap="viridis")
    im01 = _plot_slice(axes[0, 1], pinn_u_res, y, z, title=r"$u$ Residual (PINNs)", cmap="magma", vmin=0.0, vmax=u_res_vmax)
    im02 = _plot_slice(axes[0, 2], apinn_u_res, y, z, title=r"$u$ Residual (APINNs)", cmap="magma", vmin=0.0, vmax=u_res_vmax)
    im03 = _plot_slice(axes[0, 3], pimoe_u_res, y, z, title=r"$u$ Residual (ADD-PINNs)", cmap="magma", vmin=0.0, vmax=u_res_vmax)

    im10 = _plot_slice(axes[1, 0], zero_slice, y, z, title=rf"PDE True (=0) @ $x={x_used:.3f}$", cmap="viridis")
    im11 = _plot_slice(axes[1, 1], pinn_f_res, y, z, title=r"PDE Residual (PINNs)", cmap="magma", vmin=0.0)
    im12 = _plot_slice(axes[1, 2], apinn_f_res, y, z, title=r"PDE Residual (APINNs)", cmap="magma", vmin=0.0)
    im13 = _plot_slice(axes[1, 3], pimoe_f_res, y, z, title=r"PDE Residual (ADD-PINNs)", cmap="magma", vmin=0.0)

    for j, item in enumerate(phi_slices):
        _plot_phi_evolution_slice(
            axes[2, j],
            item["phi_pred"],
            item["phi_true"],
            item["y"],
            item["z"],
            epoch=int(item["epoch"]),
        )

    for i, j, im in (
        (0, 0, im00), (0, 1, im01), (0, 2, im02), (0, 3, im03),
        (1, 0, im10), (1, 1, im11), (1, 2, im12), (1, 3, im13),
    ):
        cb = fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)

    pred_handle = Line2D([0], [0], color="red", linewidth=2.0, linestyle="-", label="Predicted Interface")
    true_handle = Line2D([0], [0], color="blue", linewidth=2.0, linestyle="-", label="True Interface")
    axes[2, 0].legend(
        handles=[pred_handle, true_handle],
        loc="upper right",
        frameon=False,
        fontsize=8.5,
        handlelength=2.2,
    )

    out_path = Path(out_path).expanduser().resolve()
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
    default_out = script_dir.parent / "Figure" / "Fig14.png"

    parser = argparse.ArgumentParser(description="3D Poisson comparison figure: u and PDE residual slices.")
    parser.add_argument("--pinn", type=str, default=str(defaults["pinn"]))
    parser.add_argument("--apinn", type=str, default=str(defaults["apinn"]))
    parser.add_argument("--pimoe", type=str, default=str(defaults["pimoe"]))
    parser.add_argument(
        "--phi-dir",
        type=str,
        default=str(script_dir / "outputs_add_pinns3d_c1_sphere" / "phi_heatmaps"),
    )
    parser.add_argument("--out", type=str, default=str(default_out))
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()

    out_path = save_fig13_comparison(
        pinn_path=Path(args.pinn),
        apinn_path=Path(args.apinn),
        pimoe_path=Path(args.pimoe),
        phi_dir=Path(args.phi_dir),
        out_path=Path(args.out),
        save_pdf=not bool(args.no_pdf),
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
