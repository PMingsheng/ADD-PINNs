#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


CONFIG = {
    "line_axis": "x",  # "x" -> vary x, fix y; "y" -> vary y, fix x
    "line_mode": "value",  # "value" or "index"
    "line_value": 0.0,
    "line_index": 0,
    "phi_zero_force_nearest_if_empty": True,
    "u_color": "#1f77b4",
    "u_true_color": "#17becf",
    "f_color": "#d62728",
    "f_true_color": "#ff9896",
    "phi_color": "#2ca02c",
    "u_linewidth": 2.0,
    "f_linewidth": 2.0,
    "phi_linewidth": 1.8,
    "phi_zero_vline": True,
    "phi_zero_vline_color": "#4d4d4d",
    "phi_zero_vline_style": "-.",
    "phi_zero_vline_width": 1.0,
    "phi_zero_vline_alpha": 0.85,
    "figsize": (12.0, 10.8),
    "dpi": 260,
}


def _nearest_index(arr: np.ndarray, value: float) -> int:
    return int(np.abs(arr - float(value)).argmin())


def _resolve_index(arr: np.ndarray, mode: str, value: float, index: int, name: str) -> int:
    n = int(arr.shape[0])
    if mode == "index":
        if index < 0 or index >= n:
            raise ValueError(f"{name} index out of range: {index}, valid [0, {n - 1}]")
        return int(index)
    if mode == "value":
        return _nearest_index(arr, value)
    raise ValueError(f"{name} mode must be 'value' or 'index', got: {mode}")


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
        f"Cannot find '{display_name}' field. Tried keys: {candidates}. "
        f"Available keys: {list(data.keys())}"
    )


def _coords_from_data(data: Dict[str, np.ndarray], field_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if "x" in data and "y" in data:
        x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
        y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
        return x, y

    if "Xg" in data and "Yg" in data:
        x = np.asarray(data["Xg"], dtype=np.float64)[:, 0].reshape(-1)
        y = np.asarray(data["Yg"], dtype=np.float64)[0, :].reshape(-1)
        return x, y

    if "bbox" in data:
        bbox = np.asarray(data["bbox"], dtype=np.float64).reshape(-1)
        if bbox.size >= 4:
            n1, n2 = field_map.shape
            x = np.linspace(float(bbox[0]), float(bbox[1]), n1, dtype=np.float64)
            y = np.linspace(float(bbox[2]), float(bbox[3]), n2, dtype=np.float64)
            return x, y

    raise KeyError("Cannot infer x/y coordinates from npz (need x/y or Xg/Yg or bbox).")


def _phi_zero_crossings(
    x_line: np.ndarray,
    phi_line: np.ndarray,
    *,
    force_nearest_if_empty: bool = True,
) -> np.ndarray:
    x = np.asarray(x_line, dtype=np.float64).reshape(-1)
    y = np.asarray(phi_line, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"x/phi length mismatch: {x.size} vs {y.size}")
    if x.size < 2:
        return np.zeros((0,), dtype=np.float64)

    roots = []
    eps = 1e-14
    for i in range(x.size - 1):
        x0 = float(x[i])
        x1 = float(x[i + 1])
        y0 = float(y[i])
        y1 = float(y[i + 1])
        if abs(y0) <= eps:
            roots.append(x0)
        if y0 * y1 < 0.0:
            t = -y0 / (y1 - y0)
            roots.append(x0 + t * (x1 - x0))

    if abs(float(y[-1])) <= eps:
        roots.append(float(x[-1]))

    if len(roots) == 0:
        if not force_nearest_if_empty:
            return np.zeros((0,), dtype=np.float64)
        i0 = int(np.argmin(np.abs(y)))
        return np.asarray([float(x[i0])], dtype=np.float64)

    roots = np.asarray(roots, dtype=np.float64)
    roots = np.sort(roots)
    dedup = [roots[0]]
    tol = max(1e-10, 1e-6 * float(np.max(np.abs(x)) + 1.0))
    for r in roots[1:]:
        if abs(float(r) - float(dedup[-1])) > tol:
            dedup.append(float(r))
    return np.asarray(dedup, dtype=np.float64)


def _read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as zf:
        return {k: zf[k] for k in zf.files}


def _extract_line_2d(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    line_axis: str,
    line_mode: str,
    line_value: float,
    line_index: int,
) -> Tuple[np.ndarray, np.ndarray, str, str, int, float]:
    if line_axis == "x":
        fixed_idx = _resolve_index(
            arr=y,
            mode=line_mode,
            value=line_value,
            index=line_index,
            name="line fixed axis(y)",
        )
        return (
            x,
            field[:, fixed_idx],
            "x",
            "y",
            fixed_idx,
            float(y[fixed_idx]),
        )
    if line_axis == "y":
        fixed_idx = _resolve_index(
            arr=x,
            mode=line_mode,
            value=line_value,
            index=line_index,
            name="line fixed axis(x)",
        )
        return (
            y,
            field[fixed_idx, :],
            "y",
            "x",
            fixed_idx,
            float(x[fixed_idx]),
        )
    raise ValueError(f"line_axis must be 'x' or 'y', got: {line_axis}")


def _interp_line(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    x_src = np.asarray(x_src, dtype=np.float64).reshape(-1)
    y_src = np.asarray(y_src, dtype=np.float64).reshape(-1)
    x_dst = np.asarray(x_dst, dtype=np.float64).reshape(-1)
    if x_src.shape[0] != y_src.shape[0]:
        raise ValueError("x_src and y_src length mismatch")
    return np.interp(x_dst, x_src, y_src)


def save_uf_slice_with_phi_plot(
    u_npz_path: Path,
    f_npz_path: Path,
    phi_npz_path: Path,
    *,
    save_path: Optional[Path] = None,
    line_axis: Optional[str] = None,
    line_mode: Optional[str] = None,
    line_value: Optional[float] = None,
    line_index: Optional[int] = None,
) -> Path:
    cfg = dict(CONFIG)
    if line_axis is not None:
        cfg["line_axis"] = line_axis
    if line_mode is not None:
        cfg["line_mode"] = line_mode
    if line_value is not None:
        cfg["line_value"] = float(line_value)
    if line_index is not None:
        cfg["line_index"] = int(line_index)

    u_npz_path = Path(u_npz_path).expanduser().resolve()
    f_npz_path = Path(f_npz_path).expanduser().resolve()
    phi_npz_path = Path(phi_npz_path).expanduser().resolve()
    if not u_npz_path.exists():
        raise FileNotFoundError(f"u npz not found: {u_npz_path}")
    if not f_npz_path.exists():
        raise FileNotFoundError(f"f npz not found: {f_npz_path}")
    if not phi_npz_path.exists():
        raise FileNotFoundError(f"phi npz not found: {phi_npz_path}")

    data_u = _read_npz(u_npz_path)
    data_f = _read_npz(f_npz_path)
    data_phi = _read_npz(phi_npz_path)

    u_true_key = _pick_key(data_u, "u_true_map", ("u_true",), "u_true")
    u_pred_key = _pick_key(data_u, "u_pred_map", ("u_pred",), "u_pred")
    f_true_key = _pick_key(data_f, "f_true_map", ("f_true",), "f_true")
    f_pred_key = _pick_key(data_f, "f_pred_map", ("f_pred",), "f_pred")
    phi_key = _pick_key(data_phi, "phi", ("phi_pred", "phi_map"), "phi")

    u_true_map = np.asarray(data_u[u_true_key], dtype=np.float64)
    u_pred_map = np.asarray(data_u[u_pred_key], dtype=np.float64)
    f_true_map = np.asarray(data_f[f_true_key], dtype=np.float64)
    f_pred_map = np.asarray(data_f[f_pred_key], dtype=np.float64)
    phi_map = np.asarray(data_phi[phi_key], dtype=np.float64)

    if u_true_map.shape != u_pred_map.shape:
        raise ValueError(f"u true/pred shape mismatch: {u_true_map.shape} vs {u_pred_map.shape}")
    if f_true_map.shape != f_pred_map.shape:
        raise ValueError(f"f true/pred shape mismatch: {f_true_map.shape} vs {f_pred_map.shape}")
    if u_true_map.ndim != 2 or f_true_map.ndim != 2 or phi_map.ndim != 2:
        raise ValueError("u/f/phi fields must be 2D arrays")

    xu, yu = _coords_from_data(data_u, u_true_map)
    xf, yf = _coords_from_data(data_f, f_true_map)
    xp, yp = _coords_from_data(data_phi, phi_map)

    axis = str(cfg["line_axis"]).lower().strip()
    mode = str(cfg["line_mode"]).lower().strip()
    value = float(cfg["line_value"])
    index = int(cfg["line_index"])

    x_line, u_true_line, x_name, fixed_name, fixed_idx, fixed_value = _extract_line_2d(
        u_true_map,
        xu,
        yu,
        line_axis=axis,
        line_mode=mode,
        line_value=value,
        line_index=index,
    )
    _, u_pred_line, _, _, _, _ = _extract_line_2d(
        u_pred_map,
        xu,
        yu,
        line_axis=axis,
        line_mode=mode,
        line_value=value,
        line_index=index,
    )

    if axis == "x":
        f_fixed = _nearest_index(yf, fixed_value)
        f_true_line_raw = f_true_map[:, f_fixed]
        f_pred_line_raw = f_pred_map[:, f_fixed]
        p_fixed = _nearest_index(yp, fixed_value)
        phi_line_raw = phi_map[:, p_fixed]
        x_f = xf
        x_p = xp
    else:
        f_fixed = _nearest_index(xf, fixed_value)
        f_true_line_raw = f_true_map[f_fixed, :]
        f_pred_line_raw = f_pred_map[f_fixed, :]
        p_fixed = _nearest_index(xp, fixed_value)
        phi_line_raw = phi_map[p_fixed, :]
        x_f = yf
        x_p = yp

    f_true_line = _interp_line(x_f, f_true_line_raw, x_line)
    f_pred_line = _interp_line(x_f, f_pred_line_raw, x_line)
    phi_line = _interp_line(x_p, phi_line_raw, x_line)

    u_abs_res_line = np.abs(u_pred_line - u_true_line)
    f_abs_res_line = np.abs(f_pred_line - f_true_line)
    w_pos_line = (phi_line >= 0.0).astype(np.float64)
    w_neg_line = 1.0 - w_pos_line
    relu_pos_line = np.maximum(phi_line, 0.0)
    relu_neg_line = np.maximum(-phi_line, 0.0)

    phi_zero_x = _phi_zero_crossings(
        x_line,
        phi_line,
        force_nearest_if_empty=bool(cfg["phi_zero_force_nearest_if_empty"]),
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=tuple(cfg["figsize"]),
        dpi=int(cfg["dpi"]),
        constrained_layout=True,
    )

    suffix = f"{fixed_name}={fixed_value:.4f} (idx={fixed_idx})"

    axes[0, 0].plot(x_line, u_pred_line, color=str(cfg["u_color"]), linewidth=float(cfg["u_linewidth"]), label="u_pred")
    axes[0, 0].plot(
        x_line,
        u_true_line,
        color=str(cfg["u_true_color"]),
        linewidth=float(cfg["u_linewidth"]),
        linestyle=":",
        label="u_true",
    )
    if bool(cfg["phi_zero_vline"]):
        for xv in phi_zero_x:
            axes[0, 0].axvline(
                float(xv),
                color=str(cfg["phi_zero_vline_color"]),
                linestyle=str(cfg["phi_zero_vline_style"]),
                linewidth=float(cfg["phi_zero_vline_width"]),
                alpha=float(cfg["phi_zero_vline_alpha"]),
            )
    axes[0, 0].set_title(f"u (pred/true) | {suffix}")
    axes[0, 0].set_xlabel(x_name)
    axes[0, 0].grid(True, alpha=0.22)
    axes[0, 0].legend(loc="best", frameon=False)

    axes[0, 1].plot(x_line, f_pred_line, color=str(cfg["f_color"]), linewidth=float(cfg["f_linewidth"]), label="f_pred")
    axes[0, 1].plot(
        x_line,
        f_true_line,
        color=str(cfg["f_true_color"]),
        linewidth=float(cfg["f_linewidth"]),
        linestyle=":",
        label="f_true",
    )
    if bool(cfg["phi_zero_vline"]):
        for xv in phi_zero_x:
            axes[0, 1].axvline(
                float(xv),
                color=str(cfg["phi_zero_vline_color"]),
                linestyle=str(cfg["phi_zero_vline_style"]),
                linewidth=float(cfg["phi_zero_vline_width"]),
                alpha=float(cfg["phi_zero_vline_alpha"]),
            )
    axes[0, 1].set_title(f"f (pred/true) | {suffix}")
    axes[0, 1].set_xlabel(x_name)
    axes[0, 1].grid(True, alpha=0.22)
    axes[0, 1].legend(loc="best", frameon=False)

    ax_u_res = axes[1, 0]
    ax_u_phi = ax_u_res.twinx()
    ax_u_res.plot(x_line, u_abs_res_line, color="#9467bd", linewidth=float(cfg["u_linewidth"]), label="|u_pred-u_true|")
    ax_u_phi.plot(
        x_line,
        phi_line,
        color=str(cfg["phi_color"]),
        linewidth=float(cfg["phi_linewidth"]),
        linestyle="--",
        label="phi",
    )
    ax_u_phi.axhline(0.0, color=str(cfg["phi_color"]), linestyle=":", linewidth=1.0, alpha=0.8)
    if bool(cfg["phi_zero_vline"]):
        for xv in phi_zero_x:
            ax_u_res.axvline(
                float(xv),
                color=str(cfg["phi_zero_vline_color"]),
                linestyle=str(cfg["phi_zero_vline_style"]),
                linewidth=float(cfg["phi_zero_vline_width"]),
                alpha=float(cfg["phi_zero_vline_alpha"]),
            )
    ax_u_res.set_title(f"|u residual| + phi | {suffix}")
    ax_u_res.set_xlabel(x_name)
    ax_u_res.grid(True, alpha=0.22)

    ax_f_res = axes[1, 1]
    ax_f_phi = ax_f_res.twinx()
    ax_f_res.plot(x_line, f_abs_res_line, color="#8c564b", linewidth=float(cfg["f_linewidth"]), label="|f_pred-f_true|")
    ax_f_phi.plot(
        x_line,
        phi_line,
        color=str(cfg["phi_color"]),
        linewidth=float(cfg["phi_linewidth"]),
        linestyle="--",
        label="phi",
    )
    ax_f_phi.axhline(0.0, color=str(cfg["phi_color"]), linestyle=":", linewidth=1.0, alpha=0.8)
    if bool(cfg["phi_zero_vline"]):
        for xv in phi_zero_x:
            ax_f_res.axvline(
                float(xv),
                color=str(cfg["phi_zero_vline_color"]),
                linestyle=str(cfg["phi_zero_vline_style"]),
                linewidth=float(cfg["phi_zero_vline_width"]),
                alpha=float(cfg["phi_zero_vline_alpha"]),
            )
    ax_f_res.set_title(f"|f residual| + phi | {suffix}")
    ax_f_res.set_xlabel(x_name)
    ax_f_res.grid(True, alpha=0.22)

    ax_w_mask = axes[2, 0]
    ax_w_relu = ax_w_mask.twinx()
    ax_w_mask.plot(x_line, w_pos_line, color="#1f77b4", linewidth=2.0, label="Mask(phi>=0)")
    ax_w_mask.plot(x_line, w_neg_line, color="#ff7f0e", linewidth=2.0, linestyle="--", label="Mask(phi<0)")
    ax_w_relu.plot(x_line, relu_pos_line, color="#1f77b4", linewidth=1.8, linestyle=":", label="ReLU(phi)")
    ax_w_relu.plot(
        x_line,
        relu_neg_line,
        color="#ff7f0e",
        linewidth=1.8,
        linestyle="-.",
        label="ReLU(-phi)",
    )
    if bool(cfg["phi_zero_vline"]):
        for xv in phi_zero_x:
            ax_w_mask.axvline(
                float(xv),
                color=str(cfg["phi_zero_vline_color"]),
                linestyle=str(cfg["phi_zero_vline_style"]),
                linewidth=float(cfg["phi_zero_vline_width"]),
                alpha=float(cfg["phi_zero_vline_alpha"]),
            )
    ax_w_mask.set_title(f"weights w1/w2 | {suffix}")
    ax_w_mask.set_xlabel(x_name)
    ax_w_mask.set_ylabel("mask weight")
    ax_w_relu.set_ylabel("ReLU weight (unnormalized)")
    ax_w_mask.grid(True, alpha=0.22)
    h1, l1 = ax_w_mask.get_legend_handles_labels()
    h2, l2 = ax_w_relu.get_legend_handles_labels()
    ax_w_mask.legend(h1 + h2, l1 + l2, loc="best", frameon=False)

    axes[2, 1].plot(x_line, phi_line, color=str(cfg["phi_color"]), linewidth=float(cfg["phi_linewidth"]), label="phi")
    axes[2, 1].axhline(0.0, color=str(cfg["phi_color"]), linestyle=":", linewidth=1.0, alpha=0.8)
    if bool(cfg["phi_zero_vline"]):
        for xv in phi_zero_x:
            axes[2, 1].axvline(
                float(xv),
                color=str(cfg["phi_zero_vline_color"]),
                linestyle=str(cfg["phi_zero_vline_style"]),
                linewidth=float(cfg["phi_zero_vline_width"]),
                alpha=float(cfg["phi_zero_vline_alpha"]),
            )
    axes[2, 1].set_title(f"phi line | {suffix}")
    axes[2, 1].set_xlabel(x_name)
    axes[2, 1].grid(True, alpha=0.22)
    axes[2, 1].legend(loc="best", frameon=False)

    fig.suptitle(
        "Section Curves with phi "
        f"| u={u_npz_path.name}, f={f_npz_path.name}, phi={phi_npz_path.name}"
    )

    if save_path is None:
        out_png = u_npz_path.with_name(f"uf_phi_line_{axis}_{fixed_idx:03d}.png")
    else:
        out_png = Path(save_path).expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> None:
    project_root = Path(__file__).resolve().parent
    output_root = project_root / "outputs_flower" / "roi_off"
    ep = 30000
    u_npz = output_root / "u_heatmaps" / f"u_heatmap_{ep:08d}.npz"
    f_npz = output_root / "f_heatmaps" / f"f_heatmap_{ep:08d}.npz"
    phi_npz = output_root / "phi_snapshots" / f"phi_epoch_{ep:08d}.npz"
    out = output_root / "uf_slice_with_phi" / f"uf_phi_line_epoch_{ep:08d}.png"
    saved = save_uf_slice_with_phi_plot(u_npz, f_npz, phi_npz, save_path=out)
    print(f"[Saved] {saved}")


if __name__ == "__main__":
    main()
