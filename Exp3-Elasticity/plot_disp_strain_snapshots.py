"""
Plot displacement/strain snapshots saved during training.
"""
from pathlib import Path
import glob
import os

import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DIR = BASE_DIR / "output_roi_on" / "disp_strain_snapshots"

# User-configurable defaults.
USER_NPZ_PATH = "output_roi_on/disp_strain_snapshots/disp_strain_epoch_00010000.npz"  # file or directory; empty -> auto-latest
USER_OUTPUT_PATH = ""  # empty -> save next to npz
USER_SHOW = True
USER_DISP_COMPONENT = "mag"  # "ux", "uy", "mag"
USER_STRAIN_COMPONENT = "mag"  # "exx", "eyy", "exy", "mag"
USER_MARKER_SIZE = 10


def _resolve_npz_path(raw_path: str) -> Path:
    if raw_path:
        path = Path(raw_path).expanduser()
        if path.is_file():
            return path.resolve()
        if path.is_dir():
            candidates = sorted(path.glob("disp_strain_epoch_*.npz"))
            if candidates:
                return candidates[-1].resolve()
        alt = BASE_DIR / raw_path
        if alt.is_file():
            return alt.resolve()
        if alt.is_dir():
            candidates = sorted(alt.glob("disp_strain_epoch_*.npz"))
            if candidates:
                return candidates[-1].resolve()

    if DEFAULT_DIR.is_dir():
        candidates = sorted(DEFAULT_DIR.glob("disp_strain_epoch_*.npz"))
        if candidates:
            return candidates[-1].resolve()

    candidates = sorted(
        glob.glob(str(BASE_DIR / "**" / "disp_strain_epoch_*.npz"), recursive=True)
    )
    if not candidates:
        raise FileNotFoundError("No disp_strain_epoch_*.npz found.")
    return Path(max(candidates, key=os.path.getmtime)).resolve()


def _select_disp_component(U: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == "ux":
        return U[:, 0]
    if mode == "uy":
        return U[:, 1]
    if mode == "mag":
        return np.sqrt((U ** 2).sum(axis=1))
    raise ValueError(f"Unknown displacement component: {mode}")


def _select_strain_component(E: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == "exx":
        return E[:, 0]
    if mode == "eyy":
        return E[:, 1]
    if mode == "exy":
        return E[:, 2]
    if mode == "mag":
        return np.sqrt((E ** 2).sum(axis=1))
    raise ValueError(f"Unknown strain component: {mode}")


def _scatter_panel(ax, xy, values, title, cmap, vmin=None, vmax=None):
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=values,
        s=USER_MARKER_SIZE,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return sc


def main() -> None:
    npz_path = _resolve_npz_path(USER_NPZ_PATH)
    data = np.load(npz_path)

    xy_u = data.get("xy_u")
    U_pred = data.get("U_pred")
    U_true = data.get("U_true")
    U_err = data.get("U_err")

    if xy_u is None or U_pred is None or U_true is None:
        raise ValueError("Missing displacement data (xy_u/U_pred/U_true).")

    if U_err is None:
        U_err = U_pred - U_true

    u_true = _select_disp_component(U_true, USER_DISP_COMPONENT)
    u_pred = _select_disp_component(U_pred, USER_DISP_COMPONENT)
    if USER_DISP_COMPONENT == "mag":
        u_err = np.sqrt((U_err ** 2).sum(axis=1))
    else:
        u_err = np.abs(_select_disp_component(U_err, USER_DISP_COMPONENT))

    xy_eps = data.get("xy_eps")
    E_pred = data.get("E_pred")
    E_true = data.get("E_true")
    E_err = data.get("E_err")
    have_strain = xy_eps is not None and E_pred is not None and E_true is not None

    if have_strain and E_err is None:
        E_err = E_pred - E_true

    if have_strain:
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), dpi=300)
        axes = np.asarray(axes)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), dpi=300)
        axes = np.asarray(axes).reshape(1, 3)

    u_min = min(np.min(u_true), np.min(u_pred))
    u_max = max(np.max(u_true), np.max(u_pred))

    sc0 = _scatter_panel(axes[0, 0], xy_u, u_true, "U true", "viridis", u_min, u_max)
    sc1 = _scatter_panel(axes[0, 1], xy_u, u_pred, "U pred", "viridis", u_min, u_max)
    sc2 = _scatter_panel(axes[0, 2], xy_u, u_err, "|U pred - U true|", "magma")
    fig.colorbar(sc0, ax=axes[0, 0])
    fig.colorbar(sc1, ax=axes[0, 1])
    fig.colorbar(sc2, ax=axes[0, 2])

    if have_strain:
        e_true = _select_strain_component(E_true, USER_STRAIN_COMPONENT)
        e_pred = _select_strain_component(E_pred, USER_STRAIN_COMPONENT)
        if USER_STRAIN_COMPONENT == "mag":
            e_err = np.sqrt((E_err ** 2).sum(axis=1))
        else:
            e_err = np.abs(_select_strain_component(E_err, USER_STRAIN_COMPONENT))

        e_min = min(np.min(e_true), np.min(e_pred))
        e_max = max(np.max(e_true), np.max(e_pred))

        sc3 = _scatter_panel(axes[1, 0], xy_eps, e_true, "E true", "viridis", e_min, e_max)
        sc4 = _scatter_panel(axes[1, 1], xy_eps, e_pred, "E pred", "viridis", e_min, e_max)
        sc5 = _scatter_panel(axes[1, 2], xy_eps, e_err, "|E pred - E true|", "magma")
        fig.colorbar(sc3, ax=axes[1, 0])
        fig.colorbar(sc4, ax=axes[1, 1])
        fig.colorbar(sc5, ax=axes[1, 2])

    output_path = Path(USER_OUTPUT_PATH).expanduser() if USER_OUTPUT_PATH else npz_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"[SAVE] {output_path}")

    if USER_SHOW:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
