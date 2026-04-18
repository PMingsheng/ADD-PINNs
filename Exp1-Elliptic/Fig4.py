#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def _true_phi_signed_on_grid(bbox: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(bbox[0], bbox[1], shape[0], dtype=np.float64)
    ys = np.linspace(bbox[2], bbox[3], shape[1], dtype=np.float64)
    xg, yg = np.meshgrid(xs, ys, indexing="ij")
    rho = xg * xg + yg * yg
    theta = np.arctan2(xg, yg)
    radius = 0.4 + 0.1 * np.sin(0.0 * theta)
    phi_true = radius * radius - rho
    return xg, yg, phi_true


def _first_existing(data: np.lib.npyio.NpzFile, keys: Tuple[str, ...]) -> str | None:
    for k in keys:
        if k in data:
            return k
    return None


def _bbox_from_npz(data: np.lib.npyio.NpzFile, true_map: np.ndarray) -> np.ndarray:
    if "bbox" in data:
        bbox = np.asarray(data["bbox"], dtype=np.float64).reshape(-1)
        if bbox.size >= 4:
            return bbox[:4]

    if "x" in data and "y" in data:
        x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
        y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
        if x.size > 1 and y.size > 1:
            return np.asarray([x.min(), x.max(), y.min(), y.max()], dtype=np.float64)

    if "Xg" in data and "Yg" in data:
        xg = np.asarray(data["Xg"], dtype=np.float64)
        yg = np.asarray(data["Yg"], dtype=np.float64)
        return np.asarray([xg.min(), xg.max(), yg.min(), yg.max()], dtype=np.float64)

    n1, n2 = true_map.shape
    return np.asarray([0.0, float(n1 - 1), 0.0, float(n2 - 1)], dtype=np.float64)


def load_field_maps(npz_path: Path, field: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    true_key = _first_existing(data, (f"{field}_true", f"{field}_true_map"))
    pred_key = _first_existing(data, (f"{field}_pred", f"{field}_pred_map"))
    res_key = _first_existing(data, (f"{field}_residual", f"{field}_res_map"))
    if true_key is None or pred_key is None:
        raise KeyError(
            f"{npz_path} missing true/pred keys for '{field}'. "
            f"Expected one of {field}_true/{field}_true_map and {field}_pred/{field}_pred_map."
        )

    true_map = np.asarray(data[true_key], dtype=np.float64)
    pred_map = np.asarray(data[pred_key], dtype=np.float64)
    if true_map.shape != pred_map.shape:
        raise ValueError(f"Shape mismatch in {npz_path}: true={true_map.shape}, pred={pred_map.shape}")

    if res_key is not None:
        res_map = np.asarray(data[res_key], dtype=np.float64)
    else:
        res_map = pred_map - true_map

    if res_map.shape != true_map.shape:
        raise ValueError(f"Residual shape mismatch in {npz_path}: residual={res_map.shape}, true={true_map.shape}")

    bbox = _bbox_from_npz(data, true_map)
    return {
        "true": true_map,
        "pred": pred_map,
        "res": res_map,
        "bbox": bbox,
    }


def load_phi_snapshot(npz_path: Path) -> Dict[str, np.ndarray | int]:
    data = np.load(npz_path)
    if "phi" not in data:
        raise KeyError(f"{npz_path} missing 'phi' key.")
    phi_map = np.asarray(data["phi"], dtype=np.float64)
    bbox = _bbox_from_npz(data, phi_map)
    epoch = int(np.asarray(data["epoch"]).reshape(-1)[0]) if "epoch" in data else -1
    return {
        "phi": phi_map,
        "bbox": bbox,
        "epoch": epoch,
    }


def _plot_map(
    ax,
    arr: np.ndarray,
    bbox: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    im = ax.imshow(
        arr.T,
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


def _plot_phi_evolution_panel(
    ax,
    phi_pred: np.ndarray,
    bbox: np.ndarray,
    *,
    epoch: int,
):
    xs = np.linspace(float(bbox[0]), float(bbox[1]), int(phi_pred.shape[0]), dtype=np.float64)
    ys = np.linspace(float(bbox[2]), float(bbox[3]), int(phi_pred.shape[1]), dtype=np.float64)
    xg, yg = np.meshgrid(xs, ys, indexing="ij")

    ax.set_facecolor("white")
    ax.contour(xg, yg, phi_pred, levels=[0.0], colors="red", linewidths=2.0)
    xg_true, yg_true, phi_true = _true_phi_signed_on_grid(bbox, phi_pred.shape)
    ax.contour(xg_true, yg_true, phi_true, levels=[0.0], colors="blue", linewidths=2.0)
    ax.set_xlim(float(bbox[0]), float(bbox[1]))
    ax.set_ylim(float(bbox[2]), float(bbox[3]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(rf"$\phi$ (epoch {epoch})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent / "Figure" / "Fig4.png"

    parser = argparse.ArgumentParser(
        description="Plot 2x4 heatmaps: true + residuals(PINNs/APINNs/ADD-PINNs) for u and f."
    )
    parser.add_argument(
        "--pinn",
        type=str,
        default=str(script_dir / "outputs_pinn_single" / "final_fields.npz"),
        help="PINNs final_fields npz",
    )
    parser.add_argument(
        "--apinn",
        type=str,
        default=str(script_dir / "outputs_apinn" / "final_fields.npz"),
        help="APINNs final_fields npz",
    )
    parser.add_argument(
        "--pimoe-u",
        type=str,
        default=str(script_dir / "outputs_flower" / "roi_off" / "u_heatmaps" / "u_heatmap_00030000.npz"),
        help="ADD-PINNs u heatmap npz",
    )
    parser.add_argument(
        "--pimoe-f",
        type=str,
        default=str(script_dir / "outputs_flower" / "roi_off" / "f_heatmaps" / "f_heatmap_00030000.npz"),
        help="ADD-PINNs f heatmap npz",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(default_out),
        help="Output figure path (.png/.pdf)",
    )
    parser.add_argument(
        "--phi-dir",
        type=str,
        default=str(script_dir / "outputs_flower" / "roi_off" / "phi_snapshots"),
        help="Directory containing phi_epoch_*.npz snapshots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figure interactively.",
    )
    args = parser.parse_args()

    pinn_path = Path(args.pinn).expanduser().resolve()
    apinn_path = Path(args.apinn).expanduser().resolve()
    pimoe_u_path = Path(args.pimoe_u).expanduser().resolve()
    pimoe_f_path = Path(args.pimoe_f).expanduser().resolve()
    phi_dir = Path(args.phi_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    for p in (pinn_path, apinn_path, pimoe_u_path, pimoe_f_path):
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
    if not phi_dir.exists():
        raise FileNotFoundError(f"Directory not found: {phi_dir}")

    pinn_u = load_field_maps(pinn_path, "u")
    apinn_u = load_field_maps(apinn_path, "u")
    pimoe_u = load_field_maps(pimoe_u_path, "u")

    pinn_f = load_field_maps(pinn_path, "f")
    apinn_f = load_field_maps(apinn_path, "f")
    pimoe_f = load_field_maps(pimoe_f_path, "f")

    pinn_u["res"] = np.abs(pinn_u["res"])
    apinn_u["res"] = np.abs(apinn_u["res"])
    pimoe_u["res"] = np.abs(pimoe_u["res"])
    pinn_f["res"] = np.abs(pinn_f["res"])
    apinn_f["res"] = np.abs(apinn_f["res"])
    pimoe_f["res"] = np.abs(pimoe_f["res"])

    u_true = pinn_u["true"]
    f_true = pinn_f["true"]
    u_true_bbox = pinn_u["bbox"]
    f_true_bbox = pinn_f["bbox"]

    u_res_abs_max = max(
        np.nanmax(np.abs(pinn_u["res"])),
        np.nanmax(np.abs(apinn_u["res"])),
        np.nanmax(np.abs(pimoe_u["res"])),
    )
    f_res_abs_max = max(
        np.nanmax(np.abs(pinn_f["res"])),
        np.nanmax(np.abs(apinn_f["res"])),
        np.nanmax(np.abs(pimoe_f["res"])),
    )
    u_res_abs_max = float(u_res_abs_max if u_res_abs_max > 0 else 1.0)
    f_res_abs_max = float(f_res_abs_max if f_res_abs_max > 0 else 1.0)
    pinn_f_res_abs_max = float(np.nanmax(pinn_f["res"]) if np.nanmax(pinn_f["res"]) > 0 else 1.0)
    apinn_f_res_abs_max = float(np.nanmax(apinn_f["res"]) if np.nanmax(apinn_f["res"]) > 0 else 1.0)
    pimoe_f_res_abs_max = float(np.nanmax(pimoe_f["res"]) if np.nanmax(pimoe_f["res"]) > 0 else 1.0)

    phi_epochs = [15000, 20000, 25000, 30000]
    phi_maps = [
        load_phi_snapshot(phi_dir / f"phi_epoch_{epoch:08d}.npz")
        for epoch in phi_epochs
    ]
    phi_abs_max = max(float(np.nanmax(np.abs(item["phi"]))) for item in phi_maps)
    phi_abs_max = phi_abs_max if phi_abs_max > 0 else 1.0

    fig, axes = plt.subplots(3, 4, figsize=(13.5, 8.9), dpi=250, constrained_layout=True)

    im00 = _plot_map(axes[0, 0], u_true, u_true_bbox, title="u True", cmap="viridis")
    im01 = _plot_map(
        axes[0, 1],
        pinn_u["res"],
        pinn_u["bbox"],
        title="u Residual (PINNs)",
        cmap="magma",
        vmin=0.0,
        vmax=u_res_abs_max,
    )
    im02 = _plot_map(
        axes[0, 2],
        apinn_u["res"],
        apinn_u["bbox"],
        title="u Residual (APINNs)",
        cmap="magma",
        vmin=0.0,
        vmax=u_res_abs_max,
    )
    im03 = _plot_map(
        axes[0, 3],
        pimoe_u["res"],
        pimoe_u["bbox"],
        title="u Residual (ADD-PINNs)",
        cmap="magma",
        vmin=0.0,
        vmax=u_res_abs_max,
    )

    im10 = _plot_map(axes[1, 0], f_true, f_true_bbox, title="f True", cmap="viridis")
    im11 = _plot_map(
        axes[1, 1],
        pinn_f["res"],
        pinn_f["bbox"],
        title="f Residual (PINNs)",
        cmap="magma",
        vmin=0.0,
        vmax=pinn_f_res_abs_max,
    )
    im12 = _plot_map(
        axes[1, 2],
        apinn_f["res"],
        apinn_f["bbox"],
        title="f Residual (APINNs)",
        cmap="magma",
        vmin=0.0,
        vmax=apinn_f_res_abs_max,
    )
    im13 = _plot_map(
        axes[1, 3],
        pimoe_f["res"],
        pimoe_f["bbox"],
        title="f Residual (ADD-PINNs)",
        cmap="magma",
        vmin=0.0,
        vmax=pimoe_f_res_abs_max,
    )

    for j, phi_item in enumerate(phi_maps):
        _plot_phi_evolution_panel(
            axes[2, j],
            phi_item["phi"],
            phi_item["bbox"],
            epoch=int(phi_item["epoch"]),
        )

    for i, j, im in [
        (0, 0, im00),
        (0, 1, im01),
        (0, 2, im02),
        (0, 3, im03),
        (1, 0, im10),
        (1, 1, im11),
        (1, 2, im12),
        (1, 3, im13),
    ]:
        cb = fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)

    pred_handle = Line2D([0], [0], color="red", linewidth=2.0, linestyle="-", label="Predicted Interface")
    true_handle = Line2D([0], [0], color="blue", linewidth=2.0, linestyle="-", label="True Interface")
    for j in range(4):
        handles = [true_handle] if j == 0 else [pred_handle, true_handle]
        axes[2, j].legend(
            handles=handles,
            loc="upper right",
            frameon=False,
            fontsize=8.5,
            handlelength=2.2,
        )

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
