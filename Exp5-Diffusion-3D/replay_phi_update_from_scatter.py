#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


DIRAC_BANDWIDTH = 0.05
PROJECT_ROOT = Path(__file__).resolve().parent


# Set True to use in-file config below (recommended).
# Set False if you still want to pass parameters by terminal arguments.
USE_PROGRAM_CONFIG = True
PROGRAM_CONFIG = {
    "npz": str(
        PROJECT_ROOT
        / "outputs_add_pinns3d_c1_sphere"
        / "viz_scatter"
        / "scatter_PDE_epoch_00030000.npz"
    ),
    "vel_type": "auto",  # PDE / GRAD / CV / DATA / auto
    "dt": 1e-3,
    "band_eps": 0.05,
    "h": 0.05,
    "tau": 1.0,
    "clip_q": 0.99,
    "out_prefix": None,  # None -> <input_stem>_replay
}


def _dirac_smooth_np(phi: np.ndarray, epsilon: float = DIRAC_BANDWIDTH) -> np.ndarray:
    eps = float(epsilon)
    if eps <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    out = np.zeros_like(phi, dtype=np.float64)
    mask = np.abs(phi) <= eps
    out[mask] = 0.5 / eps * (1.0 + np.cos(np.pi * phi[mask] / eps))
    return out


def _grid_from_bbox(bbox: Tuple[float, float, float, float], n: int):
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    xs = np.linspace(xmin, xmax, n, dtype=np.float64)
    ys = np.linspace(ymin, ymax, n, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return xs, ys, X, Y


def _grad_norm(phi_map: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    gx = np.gradient(phi_map, dx, axis=0)
    gy = np.gradient(phi_map, dy, axis=1)
    return np.sqrt(gx * gx + gy * gy + 1e-16)


def _extract_vel_type_from_path(npz_path: Path) -> str:
    m = re.search(r"scatter_([A-Za-z]+)_epoch_", npz_path.name)
    if m:
        return m.group(1).upper()
    return "PDE"


def _resolve_xy_points(
    data: Dict[str, np.ndarray],
    bbox: Tuple[float, float, float, float],
    n: int,
) -> np.ndarray:
    if "XY_residual" in data:
        xy = np.asarray(data["XY_residual"], dtype=np.float64)
        if xy.ndim == 2 and xy.shape[1] == 2:
            return xy
    if "YZ_residual" in data:
        yz = np.asarray(data["YZ_residual"], dtype=np.float64)
        if yz.ndim == 2 and yz.shape[1] == 2:
            return yz

    _, _, X, Y = _grid_from_bbox(bbox, n)
    return np.stack([X.ravel(), Y.ravel()], axis=1)


def _neighbor_lists(xy_all: np.ndarray, xy_band: np.ndarray, radius: float):
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(xy_all)
        return tree.query_ball_point(xy_band, r=radius)
    except Exception:
        # Fallback without scipy.
        r2 = float(radius) * float(radius)
        nbrs = []
        for p in xy_band:
            d2 = np.sum((xy_all - p[None, :]) ** 2, axis=1)
            nbrs.append(np.where(d2 <= r2)[0].tolist())
        return nbrs


def _compute_vn_band(
    phi_flat: np.ndarray,
    val_flat: np.ndarray,
    xy_all: np.ndarray,
    *,
    band_eps: float,
    h: float,
    tau: float,
    clip_q: float,
    mode: str,
) -> Tuple[np.ndarray, bool]:
    eps = 1e-8
    band_mask = np.abs(phi_flat) < float(band_eps)
    fai_empty = False

    if mode == "CV" and not np.any(band_mask):
        fai_empty = True
        k = max(64, phi_flat.size // 50)
        idx = np.argsort(np.abs(phi_flat))[:k]
        band_mask = np.zeros_like(phi_flat, dtype=bool)
        band_mask[idx] = True

    if not np.any(band_mask):
        return np.zeros_like(phi_flat, dtype=np.float64), fai_empty

    xy_band = xy_all[band_mask]
    phi_band = phi_flat[band_mask]

    neighbor_ids = _neighbor_lists(xy_all, xy_band, float(h))

    pos_mask = phi_flat > 0
    neg_mask = ~pos_mask

    r_pos = np.zeros(len(neighbor_ids), dtype=np.float64)
    r_neg = np.zeros(len(neighbor_ids), dtype=np.float64)

    for i, nbrs in enumerate(neighbor_ids):
        if not nbrs:
            continue
        nbrs = np.asarray(nbrs, dtype=np.int64)
        pos_ids = nbrs[pos_mask[nbrs]]
        neg_ids = nbrs[neg_mask[nbrs]]
        if pos_ids.size > 0:
            r_pos[i] = float(np.mean(val_flat[pos_ids]))
        if neg_ids.size > 0:
            r_neg[i] = float(np.mean(val_flat[neg_ids]))

    if mode == "PDE":
        delta = r_neg - r_pos
        scale = np.quantile(np.abs(delta), clip_q) + eps
        vel_band = np.tanh(delta / scale)
        vel_band = vel_band * _dirac_smooth_np(phi_band, epsilon=band_eps)
    elif mode == "GRAD":
        delta = r_neg - r_pos
        delta = delta / (np.mean(np.abs(delta)) + eps)
        vel_band = delta * (np.abs(delta) / (np.abs(delta) + tau))
        vel_band = vel_band * _dirac_smooth_np(phi_band, epsilon=band_eps)
        scale = np.quantile(np.abs(vel_band), 0.95) + eps
        vel_band = np.tanh(vel_band / scale)
    elif mode == "CV":
        r_abs = val_flat
        inside = phi_flat > 0
        c1 = float(np.mean(r_abs[inside])) if np.any(inside) else 0.0
        c2 = float(np.mean(r_abs[~inside])) if np.any(~inside) else 0.0
        r_b = r_abs[band_mask]
        cv = (r_b - c1) ** 2 - (r_b - c2) ** 2
        vnb = _dirac_smooth_np(phi_band, epsilon=band_eps) * cv
        s0 = np.quantile(np.abs(vnb), clip_q) + eps
        vnb = np.tanh(vnb / s0)
        phi_max = np.max(np.abs(phi_flat)) + eps
        vn_trim = np.quantile(np.abs(vnb), clip_q) + tau
        scale = phi_max / vn_trim
        vel_band = vnb * scale * 2.0
    else:
        raise ValueError(f"Unknown mode: {mode}. Use one of PDE/GRAD/CV.")

    vn_full = np.zeros_like(phi_flat, dtype=np.float64)
    vn_full[band_mask] = vel_band
    return vn_full, fai_empty


def compute_phi_next_from_npz(
    npz_path: Path,
    *,
    vel_type: str,
    dt: float,
    band_eps: float,
    h: float,
    tau: float,
    clip_q: float,
) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as raw:
        data = {k: raw[k] for k in raw.files}

    if "phi_model" not in data or "RES_val" not in data:
        raise KeyError(f"{npz_path} must contain 'phi_model' and 'RES_val'.")

    phi_model = np.asarray(data["phi_model"], dtype=np.float64)
    if phi_model.ndim != 2 or phi_model.shape[0] != phi_model.shape[1]:
        raise ValueError("phi_model must be a square 2D array.")
    n = int(phi_model.shape[0])

    if "bbox" in data:
        bbox_arr = np.asarray(data["bbox"], dtype=np.float64).reshape(-1)
        if bbox_arr.size < 4:
            raise ValueError("bbox must contain at least 4 values.")
        bbox = (float(bbox_arr[0]), float(bbox_arr[1]), float(bbox_arr[2]), float(bbox_arr[3]))
    else:
        # Fallback to [0,1]x[0,1] for compatibility.
        bbox = (0.0, 1.0, 0.0, 1.0)

    xy_all = _resolve_xy_points(data, bbox, n)
    res_flat = np.asarray(data["RES_val"], dtype=np.float64).reshape(-1)
    phi_flat = phi_model.reshape(-1)
    if res_flat.size != phi_flat.size:
        raise ValueError(
            f"RES_val size ({res_flat.size}) does not match phi_model size ({phi_flat.size})."
        )
    if xy_all.shape[0] != phi_flat.size:
        raise ValueError(
            f"residual points ({xy_all.shape[0]}) do not match grid size ({phi_flat.size})."
        )

    mode = vel_type.upper().strip()
    if mode == "DATA":
        mode = "CV"
    if mode not in ("PDE", "GRAD", "CV"):
        raise ValueError("vel_type must be one of: PDE, GRAD, CV, DATA.")

    vn_flat, fai_empty = _compute_vn_band(
        phi_flat,
        res_flat,
        xy_all,
        band_eps=band_eps,
        h=h,
        tau=tau,
        clip_q=clip_q,
        mode=mode,
    )
    vn_map = vn_flat.reshape(n, n)

    xs, ys, X, Y = _grid_from_bbox(bbox, n)
    grad_norm = _grad_norm(phi_model, xs, ys)
    if fai_empty:
        phi_next = phi_model + dt * vn_map
    else:
        phi_next = phi_model + dt * vn_map * grad_norm
    delta_phi = phi_next - phi_model

    return {
        "X": X,
        "Y": Y,
        "phi_model": phi_model,
        "phi_next": phi_next,
        "delta_phi": delta_phi,
        "vn": vn_map,
        "res_flat": res_flat,
        "xy_all": xy_all,
        "phi_true": None if "phi_true" not in data else np.asarray(data["phi_true"], dtype=np.float64),
        "bbox": np.asarray(bbox, dtype=np.float64),
        "mode": np.asarray([mode]),
    }


def _plot_result(
    out_png: Path,
    result: Dict[str, np.ndarray],
    *,
    title: str,
) -> None:
    X = result["X"]
    Y = result["Y"]
    phi_model = result["phi_model"]
    phi_next = result["phi_next"]
    delta_phi = result["delta_phi"]
    phi_true = result["phi_true"]
    xy = result["xy_all"]
    res = result["res_flat"]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), dpi=260)

    vmax_lin = float(np.percentile(res, 99.0))
    if vmax_lin <= 0:
        vmax_lin = float(np.max(res)) if res.size > 0 else 1.0
    if vmax_lin <= 0:
        vmax_lin = 1.0
    res_clip = np.clip(res, 0.0, vmax_lin)

    sc = axes[0].scatter(
        xy[:, 0],
        xy[:, 1],
        c=res_clip,
        s=8,
        marker="s",
        linewidths=0.0,
        cmap="cividis",
        vmin=0.0,
        vmax=vmax_lin,
    )
    if phi_true is not None and np.min(phi_true) <= 0.0 <= np.max(phi_true):
        axes[0].contour(X, Y, phi_true, levels=[0.0], colors="black", linewidths=1.0, linestyles="-")
    if np.min(phi_model) <= 0.0 <= np.max(phi_model):
        axes[0].contour(X, Y, phi_model, levels=[0.0], colors="red", linewidths=1.0, linestyles="--")
    if np.min(phi_next) <= 0.0 <= np.max(phi_next):
        axes[0].contour(X, Y, phi_next, levels=[0.0], colors="lime", linewidths=0.8, linestyles="dashdot")
    axes[0].set_title("Residual + phi contours")
    axes[0].set_xlabel("axis-1")
    axes[0].set_ylabel("axis-2")
    axes[0].set_aspect("equal", adjustable="box")
    plt.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        phi_next.T,
        origin="lower",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        cmap="viridis",
        aspect="auto",
    )
    axes[1].set_title("phi_next")
    axes[1].set_xlabel("axis-1")
    axes[1].set_ylabel("axis-2")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    vmax = float(np.max(np.abs(delta_phi)))
    if vmax < 1e-14:
        vmax = 1e-14
    im2 = axes[2].imshow(
        delta_phi.T,
        origin="lower",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    axes[2].set_title("delta_phi = phi_next - phi_now")
    axes[2].set_xlabel("axis-1")
    axes[2].set_ylabel("axis-2")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one-step phi update from saved scatter residual/phi npz."
    )
    parser.add_argument(
        "--npz",
        type=str,
        default=str(
            PROJECT_ROOT
            / "outputs_add_pinns3d_c1_sphere"
            / "viz_scatter"
            / "scatter_PDE_epoch_00030000.npz"
        ),
        help="Input scatter npz (contains RES_val + phi_model).",
    )
    parser.add_argument(
        "--vel-type",
        type=str,
        default="auto",
        help="PDE/GRAD/CV/DATA or auto (infer from file name).",
    )
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--band-eps", type=float, default=0.05)
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--clip-q", type=float, default=0.99)
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Output prefix path without extension. Default: <input_stem>_replay",
    )
    return parser.parse_args()


def _runtime_config_from_source() -> Dict[str, object]:
    if USE_PROGRAM_CONFIG:
        return dict(PROGRAM_CONFIG)

    args = parse_args()
    return {
        "npz": args.npz,
        "vel_type": args.vel_type,
        "dt": args.dt,
        "band_eps": args.band_eps,
        "h": args.h,
        "tau": args.tau,
        "clip_q": args.clip_q,
        "out_prefix": args.out_prefix,
    }


def _normalize_runtime_config(cfg: Dict[str, object]) -> Dict[str, object]:
    return {
        "npz": str(cfg["npz"]),
        "vel_type": str(cfg.get("vel_type", "auto")),
        "dt": float(cfg.get("dt", 1e-3)),
        "band_eps": float(cfg.get("band_eps", 0.05)),
        "h": float(cfg.get("h", 0.05)),
        "tau": float(cfg.get("tau", 1.0)),
        "clip_q": float(cfg.get("clip_q", 0.99)),
        "out_prefix": cfg.get("out_prefix", None),
    }


def main(config: Optional[Dict[str, object]] = None) -> None:
    raw_cfg = _runtime_config_from_source() if config is None else dict(config)
    cfg = _normalize_runtime_config(raw_cfg)

    npz_path = Path(str(cfg["npz"])).expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"npz not found: {npz_path}")

    vel_type_raw = str(cfg["vel_type"])
    if vel_type_raw.lower() == "auto":
        vel_type = _extract_vel_type_from_path(npz_path)
    else:
        vel_type = vel_type_raw

    result = compute_phi_next_from_npz(
        npz_path,
        vel_type=vel_type,
        dt=float(cfg["dt"]),
        band_eps=float(cfg["band_eps"]),
        h=float(cfg["h"]),
        tau=float(cfg["tau"]),
        clip_q=float(cfg["clip_q"]),
    )

    out_prefix = (
        npz_path.with_name(npz_path.stem + "_replay")
        if cfg["out_prefix"] is None
        else Path(str(cfg["out_prefix"])).expanduser().resolve()
    )
    out_png = out_prefix.with_suffix(".png")
    out_npz = out_prefix.with_suffix(".npz")

    title = (
        f"Replay phi update | mode={result['mode'][0]} | "
        f"dt={float(cfg['dt']):.2e}, band_eps={float(cfg['band_eps']):.3g}, "
        f"h={float(cfg['h']):.3g}, tau={float(cfg['tau']):.3g}, "
        f"clip_q={float(cfg['clip_q']):.3g}"
    )
    _plot_result(out_png, result, title=title)

    np.savez_compressed(
        out_npz,
        phi_model=result["phi_model"].astype(np.float32),
        phi_next=result["phi_next"].astype(np.float32),
        delta_phi=result["delta_phi"].astype(np.float32),
        vn=result["vn"].astype(np.float32),
        XY_residual=result["xy_all"].astype(np.float32),
        RES_val=result["res_flat"].astype(np.float32),
        bbox=result["bbox"].astype(np.float32),
        mode=result["mode"],
        params_json=np.asarray(
            [
                json.dumps(
                    {
                        "dt": float(cfg["dt"]),
                        "band_eps": float(cfg["band_eps"]),
                        "h": float(cfg["h"]),
                        "tau": float(cfg["tau"]),
                        "clip_q": float(cfg["clip_q"]),
                        "vel_type_input": vel_type,
                    }
                )
            ]
        ),
    )

    print(f"[Input]  {npz_path}")
    print(f"[Mode]   {result['mode'][0]}")
    print(f"[Saved]  {out_png}")
    print(f"[Saved]  {out_npz}")


if __name__ == "__main__":
    main()
