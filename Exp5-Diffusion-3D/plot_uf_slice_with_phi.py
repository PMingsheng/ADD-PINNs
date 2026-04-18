#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from config import PIMOE3DDataConfig


PROJECT_ROOT = Path(__file__).resolve().parent

_BETA_INSIDE = 5.0
_ALPHA_INSIDE = 0.0
_ALPHA_OUTSIDE = 0.0

# Edit parameters here directly.
CONFIG = {
    "npz_path": str(
        PROJECT_ROOT
        / "outputs_add_pinns3d_c1_sphere"
        / "data_output"
        / "final_fields.npz"
    ),
    "slice_axis": "x",  # "x" / "y" / "z"
    "slice_mode": "value",  # "value" or "index"
    "slice_value": 0.4,  # used when slice_mode == "value"
    "slice_index": 25,  # used when slice_mode == "index"
    # Line cut inside the selected 2D slice.
    # "auto" -> first in-plane axis. Or choose explicit: "x"/"y"/"z"
    "line_axis": "auto",
    "line_mode": "value",  # "value" or "index" for the fixed in-plane coordinate
    "line_value": 0.5,  # used when line_mode == "value"
    "line_index": 25,  # used when line_mode == "index"
    "u_key": "u_pred",
    "u_pred_from_phi_pred_mask": False,
    "u_pred_u1_key": "u1_pred",
    "u_pred_u2_key": "u2_pred",
    "u_true_key": "u_true",
    "f_key": "f_pred",
    "f_true_key": "f_true",
    "phi_key": "phi_pred",
    "u_fallbacks": ("u_true", "u_residual"),
    "u_true_fallbacks": ("u_pred",),
    "f_fallbacks": ("f_true", "f_residual"),
    "f_true_fallbacks": ("f_pred",),
    "phi_fallbacks": ("phi_true", "phi_residual"),
    "u_color": "#1f77b4",
    "u_color_pos": "#1f77b4",
    "u_color_neg": "#ff7f0e",
    "u_true_color": "#17becf",
    "f_color": "#d62728",
    "f_color_pos": "#d62728",
    "f_color_neg": "#8c564b",
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
    "phi_zero_hline_row2": True,
    "phi_zero_hline_color": "#2ca02c",
    "phi_zero_hline_style": ":",
    "phi_zero_hline_width": 1.0,
    "phi_zero_hline_alpha": 0.9,
    # If no exact sign-change crossing exists, still draw one vertical line at argmin(|phi|).
    "phi_zero_force_nearest_if_empty": True,
    # First panel label-data scatter.
    # "train_input" -> regenerate model input labels from data config (nx,ny,nz, drop_boundary),
    #                  then pick points on current line by nearest slice/fixed coordinates.
    # "line_true"   -> fallback to current line's u_true samples (not model input points).
    "first_panel_label_source": "train_input",
    "label_nx": None,  # None -> read from PIMOE3DDataConfig
    "label_ny": None,  # None -> read from PIMOE3DDataConfig
    "label_nz": None,  # None -> read from PIMOE3DDataConfig
    "label_drop_boundary": True,
    "label_sphere_center": (0.4, 0.5, 0.5),
    "label_sphere_radius": 0.1,
    "first_panel_label_scatter": True,
    "first_panel_label_stride": 1,
    "first_panel_label_color": "#111111",
    "first_panel_label_size": 18.0,
    "first_panel_label_alpha": 0.85,
    "first_panel_label_marker": "o",
    "pred_point_scatter": True,
    "pred_point_size": 13.0,
    "pred_point_alpha": 0.72,
    "pred_point_marker": "o",
    # Third row weight curves: w1=ReLU(phi), w2=ReLU(-phi)
    "w_pos_color": "#1f77b4",
    "w_neg_color": "#ff7f0e",
    "w_linewidth": 2.0,
    "figsize": (12.0, 12.2),
    "dpi": 260,
    "save_path": None,  # None -> <npz_stem>_uf_phi_line_<slice_axis><slice_idx>.png
    "f_pred_from_phi_pred_mask": False,
    "f_pred_u1_key": "u1_pred",
    "f_pred_u2_key": "u2_pred",
    "f_true_from_phi_pred": False,
    "f_true_pred_mask_threshold": 0.0,
}


def _extract_plane(field: np.ndarray, axis: str, idx: int) -> np.ndarray:
    if axis == "x":
        return field[idx, :, :].T
    if axis == "y":
        return field[:, idx, :].T
    if axis == "z":
        return field[:, :, idx].T
    raise ValueError(f"Unknown axis: {axis}")


def _nearest_index(arr: np.ndarray, value: float) -> int:
    return int(np.abs(arr - float(value)).argmin())


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


def _resolve_index(arr: np.ndarray, mode: str, value: float, index: int, name: str) -> int:
    n = int(arr.shape[0])
    if mode == "index":
        if index < 0 or index >= n:
            raise ValueError(f"{name} index out of range: {index}, valid [0, {n - 1}]")
        return int(index)
    if mode == "value":
        return _nearest_index(arr, value)
    raise ValueError(f"{name} mode must be 'value' or 'index', got: {mode}")


def _plane_coords(axis: str, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    if axis == "x":
        return y, z, "y", "z"
    if axis == "y":
        return x, z, "x", "z"
    if axis == "z":
        return x, y, "x", "y"
    raise ValueError(f"Unknown axis: {axis}")


def _line_from_plane(
    plane: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    n1: str,
    n2: str,
    cfg: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, str, str, int, float]:
    line_axis = str(cfg["line_axis"]).lower().strip()
    if line_axis == "auto":
        line_axis = n1
    if line_axis not in (n1, n2):
        raise ValueError(
            f"line_axis='{line_axis}' not in selected slice plane axes ({n1}, {n2})."
        )

    if line_axis == n1:
        fixed_idx = _resolve_index(
            arr=c2,
            mode=str(cfg["line_mode"]).lower().strip(),
            value=float(cfg["line_value"]),
            index=int(cfg["line_index"]),
            name=f"line fixed axis({n2})",
        )
        x_line = c1
        y_line = plane[fixed_idx, :]
        x_name = n1
        fixed_name = n2
        fixed_value = float(c2[fixed_idx])
    else:
        fixed_idx = _resolve_index(
            arr=c1,
            mode=str(cfg["line_mode"]).lower().strip(),
            value=float(cfg["line_value"]),
            index=int(cfg["line_index"]),
            name=f"line fixed axis({n1})",
        )
        x_line = c2
        y_line = plane[:, fixed_idx]
        x_name = n2
        fixed_name = n1
        fixed_value = float(c1[fixed_idx])

    return x_line, y_line, x_name, fixed_name, fixed_idx, fixed_value


def _line_xyz_coords(
    x_line: np.ndarray,
    slice_axis: str,
    slice_value: float,
    x_name: str,
    fixed_name: str,
    fixed_value: float,
) -> np.ndarray:
    coords = np.zeros((x_line.size, 3), dtype=np.float64)
    coords[:, _axis_idx(slice_axis)] = float(slice_value)
    coords[:, _axis_idx(x_name)] = x_line.astype(np.float64)
    coords[:, _axis_idx(fixed_name)] = float(fixed_value)
    return coords


def _source_region_inside_numpy(xyz: np.ndarray) -> np.ndarray:
    x = xyz[:, 0:1]
    y = xyz[:, 1:2]
    z = xyz[:, 2:3]
    u = np.exp(x * y * z)
    lap_u = u * (y * y * z * z + x * x * z * z + x * x * y * y)
    return (-_BETA_INSIDE * lap_u + _ALPHA_INSIDE * u).reshape(-1)


def _source_region_outside_numpy(xyz: np.ndarray) -> np.ndarray:
    x = xyz[:, 0:1]
    y = xyz[:, 1:2]
    z = xyz[:, 2:3]
    beta = 2.0 + np.cos(x + y)
    sin_sum = np.sin(x + y + z)
    cos_sum = np.cos(x + y + z)
    sin_xy = np.sin(x + y)
    div_beta_grad_u = -3.0 * beta * sin_sum - 2.0 * sin_xy * cos_sum
    return (-div_beta_grad_u + _ALPHA_OUTSIDE * sin_sum).reshape(-1)


def _pred_source_inside_numpy(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    edge_order = 2 if min(len(x), len(y), len(z)) >= 3 else 1
    du_dx, du_dy, du_dz = np.gradient(
        u.astype(np.float64),
        x,
        y,
        z,
        axis=(0, 1, 2),
        edge_order=edge_order,
    )
    d2u_dx2 = np.gradient(du_dx, x, axis=0, edge_order=edge_order)
    d2u_dy2 = np.gradient(du_dy, y, axis=1, edge_order=edge_order)
    d2u_dz2 = np.gradient(du_dz, z, axis=2, edge_order=edge_order)
    lap_u = d2u_dx2 + d2u_dy2 + d2u_dz2
    return -_BETA_INSIDE * lap_u + _ALPHA_INSIDE * u


def _pred_source_outside_numpy(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    edge_order = 2 if min(len(x), len(y), len(z)) >= 3 else 1
    du_dx, du_dy, du_dz = np.gradient(
        u.astype(np.float64),
        x,
        y,
        z,
        axis=(0, 1, 2),
        edge_order=edge_order,
    )
    d2u_dx2 = np.gradient(du_dx, x, axis=0, edge_order=edge_order)
    d2u_dy2 = np.gradient(du_dy, y, axis=1, edge_order=edge_order)
    d2u_dz2 = np.gradient(du_dz, z, axis=2, edge_order=edge_order)
    lap_u = d2u_dx2 + d2u_dy2 + d2u_dz2

    X, Y, _ = np.meshgrid(x, y, z, indexing="ij")
    beta = 2.0 + np.cos(X + Y)
    sin_xy = np.sin(X + Y)
    div_beta_grad_u = beta * lap_u - sin_xy * du_dx - sin_xy * du_dy
    return -div_beta_grad_u + _ALPHA_OUTSIDE * u


def _masked_f_pred_from_phi_numpy(
    phi3d: np.ndarray,
    u1_3d: np.ndarray,
    u2_3d: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    f1_3d = _pred_source_inside_numpy(u1_3d, x, y, z)
    f2_3d = _pred_source_outside_numpy(u2_3d, x, y, z)
    mask = phi3d >= float(threshold)
    return np.where(mask, f1_3d, f2_3d)


def _plot_dual_axis(
    ax: plt.Axes,
    x_line: np.ndarray,
    left_pred_line: np.ndarray,
    left_true_line: Optional[np.ndarray],
    right_line: Optional[np.ndarray],
    *,
    x_name: str,
    left_pred_name: str,
    left_true_name: Optional[str],
    right_name: Optional[str],
    left_pred_color: str,
    left_true_color: Optional[str],
    right_color: Optional[str],
    left_pred_lw: float,
    left_true_lw: Optional[float],
    right_lw: Optional[float],
    pred_region_mask: Optional[np.ndarray],
    pred_region_pos_color: Optional[str],
    pred_region_neg_color: Optional[str],
    pred_region_pos_label: Optional[str],
    pred_region_neg_label: Optional[str],
    pred_point_scatter: bool,
    pred_point_size: Optional[float],
    pred_point_alpha: Optional[float],
    pred_point_marker: Optional[str],
    phi_zero_x: np.ndarray,
    draw_phi_zero_vline: bool,
    vline_color: str,
    vline_style: str,
    vline_width: float,
    vline_alpha: float,
    scatter_x: Optional[np.ndarray],
    scatter_y: Optional[np.ndarray],
    scatter_label: Optional[str],
    scatter_color: Optional[str],
    scatter_size: Optional[float],
    scatter_alpha: Optional[float],
    scatter_marker: Optional[str],
    draw_phi_zero_hline: bool,
    hline_color: str,
    hline_style: str,
    hline_width: float,
    hline_alpha: float,
    title: str,
) -> None:
    has_right = right_line is not None and right_name is not None
    ax_r = ax.twinx() if has_right else None

    l1 = []
    if pred_region_mask is not None and pred_region_pos_color is not None and pred_region_neg_color is not None:
        mask = np.asarray(pred_region_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != x_line.shape[0]:
            raise ValueError(f"pred region mask length mismatch: {mask.shape[0]} vs {x_line.shape[0]}")
        y_pos = np.where(mask, left_pred_line, np.nan)
        y_neg = np.where(~mask, left_pred_line, np.nan)
        l1 += ax.plot(
            x_line,
            y_pos,
            color=pred_region_pos_color,
            linewidth=left_pred_lw,
            label=pred_region_pos_label if pred_region_pos_label is not None else left_pred_name,
        )
        l1 += ax.plot(
            x_line,
            y_neg,
            color=pred_region_neg_color,
            linewidth=left_pred_lw,
            label=pred_region_neg_label if pred_region_neg_label is not None else left_pred_name,
        )
        if pred_point_scatter:
            size = float(pred_point_size if pred_point_size is not None else 12.0)
            alpha = float(pred_point_alpha if pred_point_alpha is not None else 0.7)
            marker = str(pred_point_marker if pred_point_marker is not None else "o")
            if np.any(mask):
                ax.scatter(
                    x_line[mask],
                    left_pred_line[mask],
                    s=size,
                    c=str(pred_region_pos_color),
                    alpha=alpha,
                    marker=marker,
                    edgecolors="none",
                    zorder=3,
                )
            if np.any(~mask):
                ax.scatter(
                    x_line[~mask],
                    left_pred_line[~mask],
                    s=size,
                    c=str(pred_region_neg_color),
                    alpha=alpha,
                    marker=marker,
                    edgecolors="none",
                    zorder=3,
                )
    else:
        l1 = ax.plot(
            x_line,
            left_pred_line,
            color=left_pred_color,
            linewidth=left_pred_lw,
            label=left_pred_name,
        )
        if pred_point_scatter:
            ax.scatter(
                x_line,
                left_pred_line,
                s=float(pred_point_size if pred_point_size is not None else 12.0),
                c=str(left_pred_color),
                alpha=float(pred_point_alpha if pred_point_alpha is not None else 0.7),
                marker=str(pred_point_marker if pred_point_marker is not None else "o"),
                edgecolors="none",
                zorder=3,
            )
    l1t = []
    if left_true_line is not None and left_true_name is not None and left_true_color is not None:
        lw = float(left_true_lw) if left_true_lw is not None else float(left_pred_lw)
        l1t = ax.plot(
            x_line,
            left_true_line,
            color=left_true_color,
            linewidth=lw,
            linestyle=":",
            label=left_true_name,
        )
    l2 = []
    if has_right and ax_r is not None and right_color is not None:
        lw_r = float(right_lw) if right_lw is not None else 1.5
        l2 = ax_r.plot(
            x_line,
            right_line,
            color=right_color,
            linewidth=lw_r,
            linestyle="--",
            label=right_name,
        )

    ax.set_xlabel(x_name)
    ax.set_ylabel("u/f", color=left_pred_color)
    ax.tick_params(axis="y", colors=left_pred_color)
    if has_right and ax_r is not None and right_color is not None:
        ax_r.set_ylabel(right_name, color=right_color)
        ax_r.tick_params(axis="y", colors=right_color)
    if has_right and ax_r is not None and draw_phi_zero_hline:
        ax_r.axhline(
            0.0,
            color=hline_color,
            linestyle=hline_style,
            linewidth=hline_width,
            alpha=hline_alpha,
            zorder=0,
        )
    if draw_phi_zero_vline:
        for xv in phi_zero_x:
            ax.axvline(
                float(xv),
                color=vline_color,
                linestyle=vline_style,
                linewidth=vline_width,
                alpha=vline_alpha,
                zorder=0,
            )
    if scatter_x is not None and scatter_y is not None and scatter_label is not None:
        ax.scatter(
            scatter_x,
            scatter_y,
            s=float(scatter_size if scatter_size is not None else 16.0),
            c=str(scatter_color if scatter_color is not None else "#111111"),
            alpha=float(scatter_alpha if scatter_alpha is not None else 0.8),
            marker=str(scatter_marker if scatter_marker is not None else "o"),
            label=scatter_label,
            zorder=3,
        )
    ax.grid(True, alpha=0.22)
    ax.set_title(title)

    lines = l1 + l1t + l2
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, loc="best", frameon=False)


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

        # Sign change on segment -> one crossing by linear interpolation.
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


def _axis_idx(name: str) -> int:
    m = {"x": 0, "y": 1, "z": 2}
    if name not in m:
        raise ValueError(f"unknown axis name: {name}")
    return m[name]


def _label_grid_coords(n: int, *, drop_boundary: bool) -> np.ndarray:
    if n < 2:
        raise ValueError("label grid side must be >=2")
    if drop_boundary:
        return np.linspace(0.0, 1.0, n + 2, dtype=np.float64)[1:-1]
    return np.linspace(0.0, 1.0, n, dtype=np.float64)


def _generate_train_input_labels(nx: int, ny: int, nz: int, *, drop_boundary: bool):
    xs = _label_grid_coords(nx, drop_boundary=drop_boundary)
    ys = _label_grid_coords(ny, drop_boundary=drop_boundary)
    zs = _label_grid_coords(nz, drop_boundary=drop_boundary)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    xyz = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)

    # exact_solution in problem_3d:
    # inside sphere -> exp(x*y*z), outside -> sin(x+y+z)
    c1x, c1y, c1z = [float(v) for v in CONFIG["label_sphere_center"]]
    radius = float(CONFIG["label_sphere_radius"])
    phi = radius * radius - (
        (xyz[:, 0] - c1x) ** 2
        + (xyz[:, 1] - c1y) ** 2
        + (xyz[:, 2] - c1z) ** 2
    )
    inside = phi >= 0.0
    u = np.where(inside, np.exp(xyz[:, 0] * xyz[:, 1] * xyz[:, 2]), np.sin(xyz[:, 0] + xyz[:, 1] + xyz[:, 2]))
    return xyz.astype(np.float64), u.astype(np.float64), xs, ys, zs


def _extract_train_label_line(
    xyz_fit: np.ndarray,
    u_fit: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    *,
    vary_axis: str,
    slice_axis: str,
    slice_value: float,
    fixed_axis: str,
    fixed_value: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    grids = {"x": xs, "y": ys, "z": zs}
    s_used = float(grids[slice_axis][_nearest_index(grids[slice_axis], slice_value)])
    f_used = float(grids[fixed_axis][_nearest_index(grids[fixed_axis], fixed_value)])
    tol = 1e-12
    sm = np.isclose(xyz_fit[:, _axis_idx(slice_axis)], s_used, atol=tol)
    fm = np.isclose(xyz_fit[:, _axis_idx(fixed_axis)], f_used, atol=tol)
    mask = sm & fm
    pts = xyz_fit[mask]
    vals = u_fit[mask]
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64), s_used, f_used
    vx = pts[:, _axis_idx(vary_axis)]
    order = np.argsort(vx)
    return vx[order], vals[order], s_used, f_used


def _plot_with_config(cfg: Dict[str, object]) -> Path:
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

    slice_axis = str(cfg["slice_axis"]).lower().strip()
    if slice_axis not in ("x", "y", "z"):
        raise ValueError("slice_axis must be one of: x / y / z")
    slice_grid = {"x": x, "y": y, "z": z}[slice_axis]
    slice_idx = _resolve_index(
        arr=slice_grid,
        mode=str(cfg["slice_mode"]).lower().strip(),
        value=float(cfg["slice_value"]),
        index=int(cfg["slice_index"]),
        name="slice",
    )
    slice_value = float(slice_grid[slice_idx])

    u_key = _pick_key(data, str(cfg["u_key"]), cfg["u_fallbacks"], "u")
    u_true_key = _pick_key(data, str(cfg["u_true_key"]), cfg["u_true_fallbacks"], "u_true")
    f_key = _pick_key(data, str(cfg["f_key"]), cfg["f_fallbacks"], "f")
    f_true_key = _pick_key(data, str(cfg["f_true_key"]), cfg["f_true_fallbacks"], "f_true")
    phi_key = _pick_key(data, str(cfg["phi_key"]), cfg["phi_fallbacks"], "phi")

    u3d = np.asarray(data[u_key], dtype=np.float64)
    u_true3d = np.asarray(data[u_true_key], dtype=np.float64)
    f3d = np.asarray(data[f_key], dtype=np.float64)
    f_true3d = np.asarray(data[f_true_key], dtype=np.float64)
    phi3d = np.asarray(data[phi_key], dtype=np.float64)
    if u3d.ndim != 3 or u_true3d.ndim != 3 or f3d.ndim != 3 or f_true3d.ndim != 3 or phi3d.ndim != 3:
        raise ValueError("u/f/phi fields must be 3D arrays.")
    if (
        u3d.shape != u_true3d.shape
        or u3d.shape != f3d.shape
        or u3d.shape != f_true3d.shape
        or u3d.shape != phi3d.shape
    ):
        raise ValueError(
            "Shape mismatch: "
            f"u={u3d.shape}, u_true={u_true3d.shape}, "
            f"f={f3d.shape}, f_true={f_true3d.shape}, phi={phi3d.shape}"
        )

    u_display_name = u_key
    if bool(cfg["u_pred_from_phi_pred_mask"]):
        u1_key = str(cfg["u_pred_u1_key"])
        u2_key = str(cfg["u_pred_u2_key"])
        if u1_key in data and u2_key in data:
            u1_3d = np.asarray(data[u1_key], dtype=np.float64)
            u2_3d = np.asarray(data[u2_key], dtype=np.float64)
            if u1_3d.shape == u3d.shape and u2_3d.shape == u3d.shape:
                mask = phi3d >= float(cfg["f_true_pred_mask_threshold"])
                u3d = np.where(mask, u1_3d, u2_3d)
                u_display_name = f"{u_key} (masked by phi_pred)"

    f_display_name = f_key
    if bool(cfg["f_pred_from_phi_pred_mask"]):
        u1_key = str(cfg["f_pred_u1_key"])
        u2_key = str(cfg["f_pred_u2_key"])
        if u1_key in data and u2_key in data:
            u1_3d = np.asarray(data[u1_key], dtype=np.float64)
            u2_3d = np.asarray(data[u2_key], dtype=np.float64)
            if u1_3d.shape == u3d.shape and u2_3d.shape == u3d.shape:
                f3d = _masked_f_pred_from_phi_numpy(
                    phi3d,
                    u1_3d,
                    u2_3d,
                    x,
                    y,
                    z,
                    threshold=float(cfg["f_true_pred_mask_threshold"]),
                )
                f_display_name = f"{f_key} (masked by phi_pred)"

    u_plane = _extract_plane(u3d, slice_axis, slice_idx)
    u_true_plane = _extract_plane(u_true3d, slice_axis, slice_idx)
    f_plane = _extract_plane(f3d, slice_axis, slice_idx)
    f_true_plane = _extract_plane(f_true3d, slice_axis, slice_idx)
    phi_plane = _extract_plane(phi3d, slice_axis, slice_idx)
    c1, c2, n1, n2 = _plane_coords(slice_axis, x, y, z)

    x_line, u_line, x_name, fixed_name, fixed_idx, fixed_value = _line_from_plane(
        u_plane, c1, c2, n1, n2, cfg
    )
    _, u_true_line, _, _, _, _ = _line_from_plane(
        u_true_plane, c1, c2, n1, n2, cfg
    )
    _, f_line, _, _, _, _ = _line_from_plane(
        f_plane, c1, c2, n1, n2, cfg
    )
    _, f_true_line, _, _, _, _ = _line_from_plane(
        f_true_plane, c1, c2, n1, n2, cfg
    )
    _, phi_line, _, _, _, _ = _line_from_plane(
        phi_plane, c1, c2, n1, n2, cfg
    )
    f_true_display_name = f_true_key
    if bool(cfg["f_true_from_phi_pred"]):
        xyz_line = _line_xyz_coords(
            x_line=x_line,
            slice_axis=slice_axis,
            slice_value=slice_value,
            x_name=x_name,
            fixed_name=fixed_name,
            fixed_value=fixed_value,
        )
        f_in_line = _source_region_inside_numpy(xyz_line.astype(np.float64))
        f_out_line = _source_region_outside_numpy(xyz_line.astype(np.float64))
        mask_pred = phi_line >= float(cfg["f_true_pred_mask_threshold"])
        f_true_line = np.where(mask_pred, f_in_line, f_out_line)
        f_true_display_name = f"{f_true_key} (from phi_pred sign)"
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

    u_abs_res_line = np.abs(u_line - u_true_line)
    f_abs_res_line = np.abs(f_line - f_true_line)
    stride = max(1, int(cfg["first_panel_label_stride"]))

    label_x = np.zeros((0,), dtype=np.float64)
    label_u = np.zeros((0,), dtype=np.float64)
    label_src = str(cfg["first_panel_label_source"]).strip().lower()
    used_slice_label = float(slice_value)
    used_fixed_label = float(fixed_value)
    if label_src == "train_input":
        dcfg = PIMOE3DDataConfig()
        nx = int(dcfg.nx) if cfg["label_nx"] is None else int(cfg["label_nx"])
        ny = int(dcfg.ny) if cfg["label_ny"] is None else int(cfg["label_ny"])
        nz = int(dcfg.nz) if cfg["label_nz"] is None else int(cfg["label_nz"])
        xyz_fit, u_fit, xs_fit, ys_fit, zs_fit = _generate_train_input_labels(
            nx,
            ny,
            nz,
            drop_boundary=bool(cfg["label_drop_boundary"]),
        )
        label_x, label_u, used_slice_label, used_fixed_label = _extract_train_label_line(
            xyz_fit,
            u_fit,
            xs_fit,
            ys_fit,
            zs_fit,
            vary_axis=x_name,
            slice_axis=slice_axis,
            slice_value=float(slice_value),
            fixed_axis=fixed_name,
            fixed_value=float(fixed_value),
        )
        label_x = label_x[::stride]
        label_u = label_u[::stride]
    else:
        label_x = x_line[::stride]
        label_u = u_true_line[::stride]

    common_suffix = (
        f"slice {slice_axis}={slice_value:.4f} (idx={slice_idx}), "
        f"{fixed_name}={fixed_value:.4f} (idx={fixed_idx})"
    )
    _plot_dual_axis(
        axes[0, 0],
        x_line,
        u_line,
        u_true_line,
        phi_line,
        x_name=x_name,
        left_pred_name=u_display_name,
        left_true_name=u_true_key,
        right_name=None,
        left_pred_color=str(cfg["u_color"]),
        left_true_color=str(cfg["u_true_color"]),
        right_color=None,
        left_pred_lw=float(cfg["u_linewidth"]),
        left_true_lw=float(cfg["u_linewidth"]),
        right_lw=None,
        pred_region_mask=phi_line >= float(cfg["f_true_pred_mask_threshold"]),
        pred_region_pos_color=str(cfg["u_color_pos"]),
        pred_region_neg_color=str(cfg["u_color_neg"]),
        pred_region_pos_label=f"{u_display_name} (phi>=0)",
        pred_region_neg_label=f"{u_display_name} (phi<0)",
        pred_point_scatter=bool(cfg["pred_point_scatter"]),
        pred_point_size=float(cfg["pred_point_size"]),
        pred_point_alpha=float(cfg["pred_point_alpha"]),
        pred_point_marker=str(cfg["pred_point_marker"]),
        phi_zero_x=phi_zero_x,
        draw_phi_zero_vline=bool(cfg["phi_zero_vline"]),
        vline_color=str(cfg["phi_zero_vline_color"]),
        vline_style=str(cfg["phi_zero_vline_style"]),
        vline_width=float(cfg["phi_zero_vline_width"]),
        vline_alpha=float(cfg["phi_zero_vline_alpha"]),
        scatter_x=label_x if bool(cfg["first_panel_label_scatter"]) else None,
        scatter_y=label_u if bool(cfg["first_panel_label_scatter"]) else None,
        scatter_label="label(u_fit)",
        scatter_color=str(cfg["first_panel_label_color"]),
        scatter_size=float(cfg["first_panel_label_size"]),
        scatter_alpha=float(cfg["first_panel_label_alpha"]),
        scatter_marker=str(cfg["first_panel_label_marker"]),
        draw_phi_zero_hline=False,
        hline_color=str(cfg["phi_zero_hline_color"]),
        hline_style=str(cfg["phi_zero_hline_style"]),
        hline_width=float(cfg["phi_zero_hline_width"]),
        hline_alpha=float(cfg["phi_zero_hline_alpha"]),
        title=f"u (pred/true) | {common_suffix}",
    )
    _plot_dual_axis(
        axes[0, 1],
        x_line,
        f_line,
        f_true_line,
        None,
        x_name=x_name,
        left_pred_name=f_display_name,
        left_true_name=f_true_display_name,
        right_name=None,
        left_pred_color=str(cfg["f_color"]),
        left_true_color=str(cfg["f_true_color"]),
        right_color=None,
        left_pred_lw=float(cfg["f_linewidth"]),
        left_true_lw=float(cfg["f_linewidth"]),
        right_lw=None,
        pred_region_mask=phi_line >= float(cfg["f_true_pred_mask_threshold"]),
        pred_region_pos_color=str(cfg["f_color_pos"]),
        pred_region_neg_color=str(cfg["f_color_neg"]),
        pred_region_pos_label=f"{f_display_name} (phi>=0)",
        pred_region_neg_label=f"{f_display_name} (phi<0)",
        pred_point_scatter=bool(cfg["pred_point_scatter"]),
        pred_point_size=float(cfg["pred_point_size"]),
        pred_point_alpha=float(cfg["pred_point_alpha"]),
        pred_point_marker=str(cfg["pred_point_marker"]),
        phi_zero_x=phi_zero_x,
        draw_phi_zero_vline=bool(cfg["phi_zero_vline"]),
        vline_color=str(cfg["phi_zero_vline_color"]),
        vline_style=str(cfg["phi_zero_vline_style"]),
        vline_width=float(cfg["phi_zero_vline_width"]),
        vline_alpha=float(cfg["phi_zero_vline_alpha"]),
        scatter_x=None,
        scatter_y=None,
        scatter_label=None,
        scatter_color=None,
        scatter_size=None,
        scatter_alpha=None,
        scatter_marker=None,
        draw_phi_zero_hline=False,
        hline_color=str(cfg["phi_zero_hline_color"]),
        hline_style=str(cfg["phi_zero_hline_style"]),
        hline_width=float(cfg["phi_zero_hline_width"]),
        hline_alpha=float(cfg["phi_zero_hline_alpha"]),
        title=f"f (pred/true) | {common_suffix}",
    )
    _plot_dual_axis(
        axes[1, 0],
        x_line,
        u_abs_res_line,
        None,
        phi_line,
        x_name=x_name,
        left_pred_name=f"|{u_display_name} - {u_true_key}|",
        left_true_name=None,
        right_name=phi_key,
        left_pred_color="#9467bd",
        left_true_color=None,
        right_color=str(cfg["phi_color"]),
        left_pred_lw=float(cfg["u_linewidth"]),
        left_true_lw=None,
        right_lw=float(cfg["phi_linewidth"]),
        pred_region_mask=None,
        pred_region_pos_color=None,
        pred_region_neg_color=None,
        pred_region_pos_label=None,
        pred_region_neg_label=None,
        pred_point_scatter=False,
        pred_point_size=None,
        pred_point_alpha=None,
        pred_point_marker=None,
        phi_zero_x=phi_zero_x,
        draw_phi_zero_vline=bool(cfg["phi_zero_vline"]),
        vline_color=str(cfg["phi_zero_vline_color"]),
        vline_style=str(cfg["phi_zero_vline_style"]),
        vline_width=float(cfg["phi_zero_vline_width"]),
        vline_alpha=float(cfg["phi_zero_vline_alpha"]),
        scatter_x=None,
        scatter_y=None,
        scatter_label=None,
        scatter_color=None,
        scatter_size=None,
        scatter_alpha=None,
        scatter_marker=None,
        draw_phi_zero_hline=bool(cfg["phi_zero_hline_row2"]),
        hline_color=str(cfg["phi_zero_hline_color"]),
        hline_style=str(cfg["phi_zero_hline_style"]),
        hline_width=float(cfg["phi_zero_hline_width"]),
        hline_alpha=float(cfg["phi_zero_hline_alpha"]),
        title=f"|u residual| + phi | {common_suffix}",
    )
    _plot_dual_axis(
        axes[1, 1],
        x_line,
        f_abs_res_line,
        None,
        phi_line,
        x_name=x_name,
        left_pred_name=f"|{f_display_name} - {f_true_display_name}|",
        left_true_name=None,
        right_name=phi_key,
        left_pred_color="#8c564b",
        left_true_color=None,
        right_color=str(cfg["phi_color"]),
        left_pred_lw=float(cfg["f_linewidth"]),
        left_true_lw=None,
        right_lw=float(cfg["phi_linewidth"]),
        pred_region_mask=None,
        pred_region_pos_color=None,
        pred_region_neg_color=None,
        pred_region_pos_label=None,
        pred_region_neg_label=None,
        pred_point_scatter=False,
        pred_point_size=None,
        pred_point_alpha=None,
        pred_point_marker=None,
        phi_zero_x=phi_zero_x,
        draw_phi_zero_vline=bool(cfg["phi_zero_vline"]),
        vline_color=str(cfg["phi_zero_vline_color"]),
        vline_style=str(cfg["phi_zero_vline_style"]),
        vline_width=float(cfg["phi_zero_vline_width"]),
        vline_alpha=float(cfg["phi_zero_vline_alpha"]),
        scatter_x=None,
        scatter_y=None,
        scatter_label=None,
        scatter_color=None,
        scatter_size=None,
        scatter_alpha=None,
        scatter_marker=None,
        draw_phi_zero_hline=bool(cfg["phi_zero_hline_row2"]),
        hline_color=str(cfg["phi_zero_hline_color"]),
        hline_style=str(cfg["phi_zero_hline_style"]),
        hline_width=float(cfg["phi_zero_hline_width"]),
        hline_alpha=float(cfg["phi_zero_hline_alpha"]),
        title=f"|f residual| + phi | {common_suffix}",
    )

    w_pos_line = np.maximum(phi_line, 0.0)
    w_neg_line = np.maximum(-phi_line, 0.0)

    axes[2, 0].plot(
        x_line,
        w_pos_line,
        color=str(cfg["w_pos_color"]),
        linewidth=float(cfg["w_linewidth"]),
        label="ReLU(phi)",
    )
    axes[2, 0].plot(
        x_line,
        w_neg_line,
        color=str(cfg["w_neg_color"]),
        linewidth=float(cfg["w_linewidth"]),
        linestyle="--",
        label="ReLU(-phi)",
    )
    if bool(cfg["phi_zero_vline"]):
        for xv in phi_zero_x:
            axes[2, 0].axvline(
                float(xv),
                color=str(cfg["phi_zero_vline_color"]),
                linestyle=str(cfg["phi_zero_vline_style"]),
                linewidth=float(cfg["phi_zero_vline_width"]),
                alpha=float(cfg["phi_zero_vline_alpha"]),
                zorder=0,
            )
    axes[2, 0].set_xlabel(x_name)
    axes[2, 0].set_ylabel("weight")
    axes[2, 0].grid(True, alpha=0.22)
    axes[2, 0].set_title(f"weights w1/w2 | {common_suffix}")
    axes[2, 0].legend(loc="best", frameon=False)

    axes[2, 1].plot(
        x_line,
        w_pos_line,
        color=str(cfg["w_pos_color"]),
        linewidth=float(cfg["w_linewidth"]),
        label="ReLU(phi)",
    )
    axes[2, 1].plot(
        x_line,
        w_neg_line,
        color=str(cfg["w_neg_color"]),
        linewidth=float(cfg["w_linewidth"]),
        linestyle="--",
        label="ReLU(-phi)",
    )
    if bool(cfg["phi_zero_vline"]):
        for xv in phi_zero_x:
            axes[2, 1].axvline(
                float(xv),
                color=str(cfg["phi_zero_vline_color"]),
                linestyle=str(cfg["phi_zero_vline_style"]),
                linewidth=float(cfg["phi_zero_vline_width"]),
                alpha=float(cfg["phi_zero_vline_alpha"]),
                zorder=0,
            )
    axes[2, 1].set_xlabel(x_name)
    axes[2, 1].set_ylabel("weight")
    axes[2, 1].grid(True, alpha=0.22)
    axes[2, 1].set_title(f"weights w1/w2 | {common_suffix}")
    axes[2, 1].legend(loc="best", frameon=False)

    fig.suptitle(
        "Section Curves with Dual Y-Axis "
        "(row2: |residual|+phi, row3: weights ReLU(phi)/ReLU(-phi)) "
        f"| source={npz_path.name}"
    )

    if cfg["save_path"] is None:
        out_png = npz_path.with_name(f"{npz_path.stem}_uf_phi_line_{slice_axis}_{slice_idx:03d}.png")
    else:
        out_png = Path(str(cfg["save_path"])).expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"[Input]     {npz_path}")
    print(f"[u key]     {u_display_name}")
    print(f"[u true]    {u_true_key}")
    print(f"[f key]     {f_display_name}")
    if bool(cfg["f_true_from_phi_pred"]):
        print(
            "[f true]    "
            f"{f_true_display_name}, threshold={float(cfg['f_true_pred_mask_threshold']):.6f}"
        )
    else:
        print(f"[f true]    {f_true_key}")
    print(f"[phi key]   {phi_key}")
    print(f"[phi=0 x]   {np.array2string(phi_zero_x, precision=6, separator=', ')}")
    print(
        f"[label]     source={label_src}, points={label_x.size}, "
        f"slice_used={slice_axis}={used_slice_label:.6f}, "
        f"fixed_used={fixed_name}={used_fixed_label:.6f}"
    )
    print(f"[Slice]     {slice_axis}={slice_value:.6f} (idx={slice_idx})")
    print(f"[Line]      x={x_name}, fixed {fixed_name}={fixed_value:.6f} (idx={fixed_idx})")
    print(f"[Saved]     {out_png}")
    return out_png


def save_uf_slice_with_phi_plot(
    npz_path: Path,
    save_path: Optional[Path] = None,
    *,
    f_true_from_phi_pred: Optional[bool] = None,
) -> Path:
    cfg = dict(CONFIG)
    cfg["npz_path"] = str(Path(npz_path).expanduser().resolve())
    if save_path is not None:
        cfg["save_path"] = str(Path(save_path).expanduser().resolve())
    if f_true_from_phi_pred is not None:
        cfg["f_true_from_phi_pred"] = bool(f_true_from_phi_pred)
    return _plot_with_config(cfg)


def main() -> None:
    _plot_with_config(dict(CONFIG))


if __name__ == "__main__":
    main()
