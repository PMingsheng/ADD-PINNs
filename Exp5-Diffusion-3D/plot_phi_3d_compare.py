#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent

# Edit parameters here directly.
CONFIG = {
    "npz_path": str(
        PROJECT_ROOT
        / "outputs_add_pinns3d_c1_sphere"
        / "data_output"
        / "final_fields.npz"
    ),
    "phi_true_key": "phi_true",
    "phi_pred_key": "phi_pred",
    "phi_true_fallbacks": (),
    "phi_pred_fallbacks": (),
    # zero-level shell thickness: eps = eps_scale * max(dx,dy,dz)
    "eps_scale": 1.3,
    "max_points_each": 20000,
    "random_seed": 0,
    "true_color": "#1f77b4",
    "pred_color": "#d62728",
    "point_size": 4.0,
    "alpha_single": 0.50,
    "alpha_overlay_true": 0.28,
    "alpha_overlay_pred": 0.28,
    "elev": 22.0,
    "azim": -52.0,
    "figsize": (14.8, 4.8),
    "slice_figsize": (12.8, 6.8),
    "dpi": 260,
    "save_path": None,  # None -> <npz_stem>_phi_3d_compare.png
    "slice_save_path": None,  # None -> <npz_stem>_phi_slice_2x3.png
    "c1_xyz": (0.4, 0.5, 0.5),
    # 2x3 slice panel style:
    # "x_only_circle" -> six x-slices inside the sphere range (to show circles)
    # "xyz_mixed"     -> old style (x-mid/x-c1/y-mid/y-c1/z-mid/z-c1)
    "slice_style": "x_only_circle",
    "x_slice_count": 6,
    "x_slice_inner_ratio": 0.80,  # use x in [cx-r*ratio, cx+r*ratio]
}


def _pick_key(
    data: Dict[str, np.ndarray],
    primary: str,
    fallbacks: Iterable[str],
    display_name: str,
) -> str:
    candidates = [primary] + [k for k in fallbacks if k != primary]
    for key in candidates:
        if key in data:
            return key
    raise KeyError(
        f"Cannot find '{display_name}'. Tried keys: {candidates}. "
        f"Available keys: {list(data.keys())}"
    )


def _sample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    ids = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[ids]


def _extract_zero_shell_points(
    phi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    eps: float,
) -> np.ndarray:
    mask = np.abs(phi) <= float(eps)
    ids = np.argwhere(mask)
    if ids.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts = np.empty((ids.shape[0], 3), dtype=np.float64)
    pts[:, 0] = x[ids[:, 0]]
    pts[:, 1] = y[ids[:, 1]]
    pts[:, 2] = z[ids[:, 2]]
    return pts


def _set_axes_equal(
    ax: plt.Axes,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))


def _nearest_index(arr: np.ndarray, value: float) -> int:
    return int(np.abs(arr - float(value)).argmin())


def _extract_plane(field: np.ndarray, axis: str, idx: int) -> np.ndarray:
    if axis == "x":
        return field[idx, :, :].T
    if axis == "y":
        return field[:, idx, :].T
    if axis == "z":
        return field[:, :, idx].T
    raise ValueError(f"Unknown axis: {axis}")


def _plane_extent(axis: str, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    if axis == "x":
        return [float(y.min()), float(y.max()), float(z.min()), float(z.max())], "y", "z"
    if axis == "y":
        return [float(x.min()), float(x.max()), float(z.min()), float(z.max())], "x", "z"
    if axis == "z":
        return [float(x.min()), float(x.max()), float(y.min()), float(y.max())], "x", "y"
    raise ValueError(f"Unknown axis: {axis}")


def _plane_coords(axis: str, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    if axis == "x":
        return y, z
    if axis == "y":
        return x, z
    if axis == "z":
        return x, y
    raise ValueError(f"Unknown axis: {axis}")


def _iou(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    inter = np.logical_and(mask_true, mask_pred).sum(dtype=np.int64)
    union = np.logical_or(mask_true, mask_pred).sum(dtype=np.int64)
    if union == 0:
        return 1.0
    return float(inter / union)


def _build_plane_specs_mixed(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    c1_xyz: Tuple[float, float, float],
) -> List[Tuple[str, str, int, float]]:
    c1x, c1y, c1z = [float(v) for v in c1_xyz]
    return [
        ("x-mid", "x", len(x) // 2, float(x[len(x) // 2])),
        ("x-c1", "x", _nearest_index(x, c1x), float(x[_nearest_index(x, c1x)])),
        ("y-mid", "y", len(y) // 2, float(y[len(y) // 2])),
        ("y-c1", "y", _nearest_index(y, c1y), float(y[_nearest_index(y, c1y)])),
        ("z-mid", "z", len(z) // 2, float(z[len(z) // 2])),
        ("z-c1", "z", _nearest_index(z, c1z), float(z[_nearest_index(z, c1z)])),
    ]


def _build_plane_specs_x_only_circle(
    x: np.ndarray,
    phi_true: np.ndarray,
    c1_x: float,
    x_slice_count: int,
    x_slice_inner_ratio: float,
) -> List[Tuple[str, str, int, float]]:
    if x_slice_count <= 0:
        raise ValueError("x_slice_count must be positive.")

    radius = float(np.sqrt(max(0.0, float(np.max(phi_true)))))
    inner = max(0.05, min(0.98, float(x_slice_inner_ratio)))
    x_left = float(c1_x - inner * radius)
    x_right = float(c1_x + inner * radius)
    x_targets = np.linspace(x_left, x_right, int(x_slice_count), dtype=np.float64)

    idx_raw = [_nearest_index(x, xv) for xv in x_targets]
    idx_list: List[int] = []
    for idx in idx_raw:
        if idx not in idx_list:
            idx_list.append(idx)

    # If nearest indices collapse (coarse grid), pad from valid circle-cutting planes.
    if len(idx_list) < x_slice_count:
        candidate = []
        for i in range(len(x)):
            pos_cells = int((_extract_plane(phi_true, "x", i) > 0.0).sum())
            if pos_cells >= 8:  # avoid tangent-like slices with too few positive cells
                candidate.append(i)
        for i in candidate:
            if i not in idx_list:
                idx_list.append(i)
            if len(idx_list) >= x_slice_count:
                break

    idx_list = idx_list[:x_slice_count]
    return [
        (f"x-{k+1}", "x", i, float(x[i]))
        for k, i in enumerate(idx_list)
    ]


def _save_phi_slice_2x3(
    phi_true: np.ndarray,
    phi_pred: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    out_png: Path,
    *,
    figsize: Tuple[float, float],
    dpi: int,
    c1_xyz: Tuple[float, float, float],
    slice_style: str,
    x_slice_count: int,
    x_slice_inner_ratio: float,
    title_prefix: str,
) -> Tuple[float, float]:
    if slice_style == "x_only_circle":
        specs = _build_plane_specs_x_only_circle(
            x=x,
            phi_true=phi_true,
            c1_x=float(c1_xyz[0]),
            x_slice_count=x_slice_count,
            x_slice_inner_ratio=x_slice_inner_ratio,
        )
    elif slice_style == "xyz_mixed":
        specs = _build_plane_specs_mixed(x, y, z, c1_xyz)
    else:
        raise ValueError(f"Unknown slice_style: {slice_style}")

    if len(specs) != 6:
        raise ValueError(f"Expected 6 slice specs for 2x3 panel, got {len(specs)}.")
    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi, squeeze=False)
    vmin = float(phi_pred.min())
    vmax = float(phi_pred.max())
    if abs(vmax - vmin) < 1e-14:
        vmax = vmin + 1e-14

    ious = []
    im = None
    for i, (_, axis, idx, coord) in enumerate(specs):
        r, c = divmod(i, 3)
        ax = axes[r, c]
        true_2d = _extract_plane(phi_true, axis, idx)
        pred_2d = _extract_plane(phi_pred, axis, idx)
        extent, xlab, ylab = _plane_extent(axis, x, y, z)
        cx, cy = _plane_coords(axis, x, y, z)
        Xc, Yc = np.meshgrid(cx, cy, indexing="xy")

        im = ax.imshow(
            pred_2d,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        if true_2d.min() <= 0.0 <= true_2d.max():
            ax.contour(Xc, Yc, true_2d, levels=[0.0], colors="black", linewidths=1.2, linestyles="-")
        if pred_2d.min() <= 0.0 <= pred_2d.max():
            ax.contour(Xc, Yc, pred_2d, levels=[0.0], colors="yellow", linewidths=1.2, linestyles="--")

        iou_val = _iou(true_2d > 0.0, pred_2d > 0.0)
        ious.append(iou_val)
        ax.text(
            0.02,
            0.98,
            f"IoU={iou_val:.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
        )
        ax.set_title(f"pred | {axis}={coord:.3f}")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.08, top=0.90, wspace=0.25, hspace=0.28)
    cax = fig.add_axes([0.92, 0.16, 0.018, 0.68])
    fig.colorbar(im, cax=cax)

    iou_mean = float(np.mean(ious)) if ious else 0.0
    iou_min = float(np.min(ious)) if ious else 0.0
    iou_vol = _iou(phi_true > 0.0, phi_pred > 0.0)
    if slice_style == "x_only_circle":
        x_txt = ", ".join([f"{s[3]:.3f}" for s in specs])
        title = (
            f"{title_prefix} | x-only circle cuts: [{x_txt}] | "
            f"IoU(slice mean/min)={iou_mean:.4f}/{iou_min:.4f}, IoU(vol)={iou_vol:.4f}"
        )
    else:
        title = (
            f"{title_prefix} | IoU(slice mean/min)={iou_mean:.4f}/{iou_min:.4f}, "
            f"IoU(vol)={iou_vol:.4f}"
        )
    fig.suptitle(title, y=0.98)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return iou_mean, iou_min


def main() -> None:
    cfg = dict(CONFIG)
    npz_path = Path(str(cfg["npz_path"])).expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"npz not found: {npz_path}")

    with np.load(npz_path) as zf:
        data = {k: zf[k] for k in zf.files}

    for ck in ("x", "y", "z"):
        if ck not in data:
            raise KeyError(f"{npz_path} missing coordinate key: '{ck}'")
    x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
    y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
    z = np.asarray(data["z"], dtype=np.float64).reshape(-1)

    phi_true_key = _pick_key(
        data, str(cfg["phi_true_key"]), cfg["phi_true_fallbacks"], "phi_true"
    )
    phi_pred_key = _pick_key(
        data, str(cfg["phi_pred_key"]), cfg["phi_pred_fallbacks"], "phi_pred"
    )

    phi_true = np.asarray(data[phi_true_key], dtype=np.float64)
    phi_pred = np.asarray(data[phi_pred_key], dtype=np.float64)
    if phi_true.ndim != 3 or phi_pred.ndim != 3:
        raise ValueError("phi_true and phi_pred must be 3D arrays.")
    if phi_true.shape != phi_pred.shape:
        raise ValueError(f"Shape mismatch: true={phi_true.shape}, pred={phi_pred.shape}")

    dx = float(x[1] - x[0]) if x.size > 1 else 1.0
    dy = float(y[1] - y[0]) if y.size > 1 else 1.0
    dz = float(z[1] - z[0]) if z.size > 1 else 1.0
    eps = float(cfg["eps_scale"]) * max(abs(dx), abs(dy), abs(dz))

    pts_true = _extract_zero_shell_points(phi_true, x, y, z, eps=eps)
    pts_pred = _extract_zero_shell_points(phi_pred, x, y, z, eps=eps)

    rng = np.random.default_rng(int(cfg["random_seed"]))
    pts_true = _sample_points(pts_true, int(cfg["max_points_each"]), rng)
    pts_pred = _sample_points(pts_pred, int(cfg["max_points_each"]), rng)

    mt = phi_true > 0.0
    mp = phi_pred > 0.0
    inter = int(np.logical_and(mt, mp).sum())
    union = int(np.logical_or(mt, mp).sum())
    iou_vol = 1.0 if union == 0 else float(inter / union)

    fig = plt.figure(figsize=tuple(cfg["figsize"]), dpi=int(cfg["dpi"]))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    if pts_true.shape[0] > 0:
        ax1.scatter(
            pts_true[:, 0],
            pts_true[:, 1],
            pts_true[:, 2],
            s=float(cfg["point_size"]),
            c=str(cfg["true_color"]),
            alpha=float(cfg["alpha_single"]),
            linewidths=0.0,
        )
    ax1.set_title(f"True phi=0 shell\npoints={pts_true.shape[0]}")

    if pts_pred.shape[0] > 0:
        ax2.scatter(
            pts_pred[:, 0],
            pts_pred[:, 1],
            pts_pred[:, 2],
            s=float(cfg["point_size"]),
            c=str(cfg["pred_color"]),
            alpha=float(cfg["alpha_single"]),
            linewidths=0.0,
        )
    ax2.set_title(f"Pred phi=0 shell\npoints={pts_pred.shape[0]}")

    if pts_true.shape[0] > 0:
        ax3.scatter(
            pts_true[:, 0],
            pts_true[:, 1],
            pts_true[:, 2],
            s=float(cfg["point_size"]),
            c=str(cfg["true_color"]),
            alpha=float(cfg["alpha_overlay_true"]),
            linewidths=0.0,
            label="true",
        )
    if pts_pred.shape[0] > 0:
        ax3.scatter(
            pts_pred[:, 0],
            pts_pred[:, 1],
            pts_pred[:, 2],
            s=float(cfg["point_size"]),
            c=str(cfg["pred_color"]),
            alpha=float(cfg["alpha_overlay_pred"]),
            linewidths=0.0,
            label="pred",
        )
    ax3.set_title(f"Overlay\nIoU(vol)={iou_vol:.4f}")
    ax3.legend(loc="upper right", frameon=False)

    xlim = (float(x.min()), float(x.max()))
    ylim = (float(y.min()), float(y.max()))
    zlim = (float(z.min()), float(z.max()))
    for ax in (ax1, ax2, ax3):
        _set_axes_equal(ax, xlim, ylim, zlim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=float(cfg["elev"]), azim=float(cfg["azim"]))

    fig.suptitle(
        f"3D Phi Sphere Compare | eps={eps:.4f} | source={npz_path.name}",
        y=0.98,
    )
    fig.tight_layout()

    if cfg["save_path"] is None:
        out_png = npz_path.with_name(f"{npz_path.stem}_phi_3d_compare.png")
    else:
        out_png = Path(str(cfg["save_path"])).expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    if cfg["slice_save_path"] is None:
        out_slice_png = npz_path.with_name(f"{npz_path.stem}_phi_slice_2x3.png")
    else:
        out_slice_png = Path(str(cfg["slice_save_path"])).expanduser().resolve()
    iou_mean, iou_min = _save_phi_slice_2x3(
        phi_true=phi_true,
        phi_pred=phi_pred,
        x=x,
        y=y,
        z=z,
        out_png=out_slice_png,
        figsize=tuple(cfg["slice_figsize"]),
        dpi=int(cfg["dpi"]),
        c1_xyz=tuple(cfg["c1_xyz"]),
        slice_style=str(cfg["slice_style"]),
        x_slice_count=int(cfg["x_slice_count"]),
        x_slice_inner_ratio=float(cfg["x_slice_inner_ratio"]),
        title_prefix="phi pred(+true contour)",
    )

    print(f"[Input]     {npz_path}")
    print(f"[phi_true]  {phi_true_key}")
    print(f"[phi_pred]  {phi_pred_key}")
    print(f"[eps]       {eps:.6f}")
    print(f"[points]    true={pts_true.shape[0]}, pred={pts_pred.shape[0]}")
    print(f"[IoU vol]   {iou_vol:.6f}")
    print(f"[IoU 2x3]   mean={iou_mean:.6f}, min={iou_min:.6f}")
    print(f"[Saved]     {out_png}")
    print(f"[Saved]     {out_slice_png}")


if __name__ == "__main__":
    main()
