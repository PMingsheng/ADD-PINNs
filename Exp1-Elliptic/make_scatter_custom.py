#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from config import DataConfig, ModelConfig, SAMPLING_CONFIGS, VisualizationConfig
from data import load_uniform_grid_fit
from model import PartitionPINN
from problem import f_region_inside, f_region_outside
from utils import set_seed
from visualization import plot_residual_scatter_heat


# Edit parameters directly here.
RUN_CONFIG = {
    "residual_source": "grad",  # data / pde / grad
    "residual_data_source": "epoch_npz",  # compute / epoch_npz
    "residual_npz_path": None,  # optional explicit path; None -> auto by vel_type+epoch
    "residual_npz_tag": None,  # optional override tag in scatter_<TAG>_epoch_xxxxxxxx.npz
    "vel_type": "GRAD",  # PDE / GRAD / CV / DATA
    "phi_source": "epoch",  # model / epoch
    "sampling_mode": "roi-off",  # roi-on / roi-off / full-data
    "checkpoint": None,  # e.g. "outputs_flower/roi_off/checkpoint_xxx.pth" or None
    "seed": 1234,
    "n": 200,
    "batch_size": 4096,
    "bbox": (-1.0, 1.0, -1.0, 1.0),  # xmin, xmax, ymin, ymax
    "dt_next": 10,
    "band_eps_vel": 0.1,
    "h_vel": 0.1,
    "tau_vel": 1e-3,
    "clip_q_vel": 0.99,
    "cmap": "cividis",
    "show_next": True,
    "show": True,
    "epoch_label": 20000,  # used for default filename only
    "base_name": None,  # if None -> scatter_<VEL>_epoch_<EPOCH>
    "output_dir": None,  # if None -> outputs_flower/<sampling_tag>/viz_scatter
}


def _get_f1_f2(xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return f_region_inside(xy), f_region_outside(xy)


def _load_model_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    obj = torch.load(str(checkpoint), map_location=torch.device("cpu"))

    if isinstance(obj, dict):
        state_dict = None
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                state_dict = obj[key]
                break
        if state_dict is None and all(isinstance(k, str) for k in obj.keys()):
            state_dict = obj
        if state_dict is None:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint}")
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(obj)}")

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Checkpoint] loaded {checkpoint} | missing={len(missing)} unexpected={len(unexpected)}")


def _load_phi_from_epoch_snapshot(
    project_root: Path,
    *,
    sampling_tag: str,
    epoch_label: int,
) -> np.ndarray:
    phi_npz = (
        project_root
        / "outputs_flower"
        / sampling_tag
        / "phi_snapshots"
        / f"phi_epoch_{int(epoch_label):08d}.npz"
    )
    if not phi_npz.exists():
        raise FileNotFoundError(f"phi snapshot not found: {phi_npz}")
    with np.load(phi_npz) as zf:
        if "phi" not in zf.files:
            raise KeyError(f"'phi' key not found in {phi_npz}")
        phi_map = np.asarray(zf["phi"], dtype=np.float32)
    return phi_map


def _load_residual_from_epoch_npz(npz_path: Path) -> dict:
    if not npz_path.exists():
        raise FileNotFoundError(f"residual npz not found: {npz_path}")

    with np.load(str(npz_path), allow_pickle=True) as zf:
        required = ("XY_residual", "RES_val", "phi_model", "phi_true", "bbox")
        for key in required:
            if key not in zf.files:
                raise KeyError(f"'{key}' key not found in {npz_path}")

        out = {
            "XY_residual": np.asarray(zf["XY_residual"], dtype=np.float32),
            "RES_val": np.asarray(zf["RES_val"], dtype=np.float32),
            "phi_model": np.asarray(zf["phi_model"], dtype=np.float32),
            "phi_true": np.asarray(zf["phi_true"], dtype=np.float32),
            "bbox": np.asarray(zf["bbox"], dtype=np.float32),
        }

        if "phi_next" in zf.files:
            phi_next_raw = zf["phi_next"]
            if phi_next_raw.dtype == object and phi_next_raw.size == 1 and phi_next_raw.item() is None:
                out["phi_next"] = None
            else:
                out["phi_next"] = np.asarray(phi_next_raw, dtype=np.float32)
        else:
            out["phi_next"] = None

    return out


def _save_scatter_outputs(
    *,
    XY_residual: np.ndarray,
    RES_val: np.ndarray,
    phi_model: np.ndarray,
    phi_true: np.ndarray,
    phi_next: np.ndarray | None,
    bbox: Tuple[float, float, float, float],
    kind: str,
    cmap: str,
    show_next: bool,
    show: bool,
    save_png: Path,
    save_npz: Path,
) -> None:
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]
    nx, ny = int(phi_model.shape[0]), int(phi_model.shape[1])
    xs = np.linspace(xmin, xmax, nx, dtype=np.float32)
    ys = np.linspace(ymin, ymax, ny, dtype=np.float32)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")

    RES_plot = np.asarray(RES_val, dtype=np.float32).copy()
    vmax_lin = float(np.percentile(RES_plot, 99.0))
    if vmax_lin <= 0:
        vmax_lin = float(RES_plot.max()) if float(RES_plot.max()) > 0 else 1.0
    RES_clip = np.clip(RES_plot, 0.0, vmax_lin)

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    fig, ax = plt.subplots(dpi=300, figsize=(3.8, 2.5))

    sc = ax.scatter(
        XY_residual[:, 0],
        XY_residual[:, 1],
        c=RES_clip,
        s=(40 if kind == "data" else 8),
        marker=("o" if kind == "data" else "s"),
        linewidths=0.0,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax_lin,
    )

    ax.contour(Xg, Yg, phi_true, levels=[0.0], colors="black", linewidths=1.0, linestyles="-")
    ax.contour(Xg, Yg, phi_model, levels=[0.0], colors="red", linewidths=1.0, linestyles="--")

    if show_next and (phi_next is not None):
        ax.contour(Xg, Yg, phi_next, levels=[0.0], colors="lime", linewidths=0.7, linestyles="dashdot")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.locator_params(axis="x", nbins=4)
    ax.locator_params(axis="y", nbins=4)

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.0, ls="-", label="Exact"),
        Line2D([0], [0], color="red", lw=1.0, ls="--", label="Current"),
    ]
    if show_next and (phi_next is not None):
        legend_handles.append(Line2D([0], [0], color="lime", lw=1.0, ls="-.", label="Next"))
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        framealpha=0.8,
        facecolor="white",
        edgecolor="none",
        fontsize=6,
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=1.0)
    cbar.mappable.set_clim(0.0, vmax_lin)
    sf = ScalarFormatter(useMathText=True)
    sf.set_powerlimits((-2, 2))
    sf.set_scientific(True)
    cbar.ax.yaxis.set_major_formatter(sf)
    cbar.ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))
    offset_txt = cbar.ax.yaxis.get_offset_text()
    offset_txt.set_x(2.4)
    offset_txt.set_va("bottom")
    offset_txt.set_fontsize(7)

    fig.tight_layout()
    fig.savefig(str(save_png), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    np.savez_compressed(
        str(save_npz),
        XY_residual=np.asarray(XY_residual, dtype=np.float32),
        RES_val=np.asarray(RES_val, dtype=np.float32),
        phi_model=np.asarray(phi_model, dtype=np.float32),
        phi_true=np.asarray(phi_true, dtype=np.float32),
        phi_next=None if phi_next is None else np.asarray(phi_next, dtype=np.float32),
        bbox=np.asarray([xmin, xmax, ymin, ymax], dtype=np.float32),
    )


def main() -> None:
    cfg = dict(RUN_CONFIG)
    residual_source = str(cfg["residual_source"]).lower()
    residual_data_source = str(cfg["residual_data_source"]).lower()
    residual_npz_path_cfg = cfg["residual_npz_path"]
    residual_npz_tag_cfg = cfg.get("residual_npz_tag")
    vel_type = str(cfg["vel_type"]).upper()
    phi_source = str(cfg["phi_source"]).lower()
    sampling_mode = str(cfg["sampling_mode"]).lower()
    checkpoint = cfg["checkpoint"]
    seed = int(cfg["seed"])
    n = int(cfg["n"])
    batch_size = int(cfg["batch_size"])
    bbox = tuple(float(v) for v in cfg["bbox"])
    dt_next = float(cfg["dt_next"])
    band_eps_vel = float(cfg["band_eps_vel"])
    h_vel = float(cfg["h_vel"])
    tau_vel = float(cfg["tau_vel"])
    clip_q_vel = float(cfg["clip_q_vel"])
    cmap = str(cfg["cmap"])
    show_next = bool(cfg["show_next"])
    show = bool(cfg["show"])
    epoch_label = int(cfg["epoch_label"])
    base_name_cfg = cfg["base_name"]
    output_dir_cfg = cfg["output_dir"]

    if residual_source not in {"data", "pde", "grad"}:
        raise ValueError(f"residual_source must be one of data/pde/grad, got: {residual_source}")
    if residual_data_source not in {"compute", "epoch_npz"}:
        raise ValueError(
            f"residual_data_source must be one of compute/epoch_npz, got: {residual_data_source}"
        )
    if vel_type not in {"PDE", "GRAD", "CV", "DATA"}:
        raise ValueError(f"vel_type must be one of PDE/GRAD/CV/DATA, got: {vel_type}")
    if phi_source not in {"model", "epoch"}:
        raise ValueError(f"phi_source must be one of model/epoch, got: {phi_source}")
    if sampling_mode not in {"roi-on", "roi-off", "full-data"}:
        raise ValueError(f"sampling_mode must be one of roi-on/roi-off/full-data, got: {sampling_mode}")

    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    set_seed(seed)

    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    viz_cfg = VisualizationConfig()

    sampling_cfg = SAMPLING_CONFIGS.get(sampling_mode, SAMPLING_CONFIGS["roi-off"])
    sampling_tag = sampling_mode.replace("-", "_")

    if output_dir_cfg is None:
        out_dir = project_root / "outputs_flower" / sampling_tag / viz_cfg.viz_scatter_dir
    else:
        out_dir = Path(str(output_dir_cfg)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if base_name_cfg:
        base_name = str(base_name_cfg)
    else:
        base_name = f"scatter_{vel_type}_epoch_{epoch_label:08d}"

    save_png = out_dir / f"{base_name}.png"
    save_npz = out_dir / f"{base_name}.npz"

    if residual_data_source == "epoch_npz":
        default_npz_tag = {
            "pde": "PDE",
            "grad": "GRAD",
            "data": "DATA",
        }[residual_source]
        npz_tag = default_npz_tag if residual_npz_tag_cfg is None else str(residual_npz_tag_cfg).upper()
        if residual_npz_path_cfg is None:
            residual_npz_path = (
                project_root
                / "outputs_flower"
                / sampling_tag
                / viz_cfg.viz_scatter_dir
                / f"scatter_{npz_tag}_epoch_{epoch_label:08d}.npz"
            )
        else:
            residual_npz_path = Path(str(residual_npz_path_cfg)).expanduser().resolve()
        out = _load_residual_from_epoch_npz(residual_npz_path)
        print(f"[Residual] using epoch npz: {residual_npz_path}")
    else:
        xy_fit, u_fit = load_uniform_grid_fit(
            nx=sampling_cfg["nx"],
            ny=sampling_cfg["ny"],
            use_synthetic=data_cfg.use_synthetic,
            synthetic_n_side=data_cfg.synthetic_n_side,
            ttxt_filename=str((project_root / data_cfg.ttxt_filename).resolve()),
            device=torch.device("cpu"),
            circles=list(data_cfg.circles),
            annuli=list(data_cfg.annuli),
            dense_factor=sampling_cfg["dense_factor"],
            drop_boundary=sampling_cfg["drop_boundary"],
            xlim=sampling_cfg["xlim"],
            ylim=sampling_cfg["ylim"],
            tol=sampling_cfg["tol"],
            target_total=sampling_cfg.get("target_total"),
        )

        model = PartitionPINN(width=model_cfg.width, depth=model_cfg.depth).to(torch.device("cpu"))
        if checkpoint:
            ckpt = Path(str(checkpoint)).expanduser().resolve()
            if not ckpt.exists():
                raise FileNotFoundError(f"checkpoint not found: {ckpt}")
            _load_model_checkpoint(model, ckpt)
        else:
            print("[Checkpoint] none provided, using current model initialization.")

        out = plot_residual_scatter_heat(
            model,
            kind=residual_source,
            xy_fit=xy_fit,
            u_fit=u_fit,
            n=n,
            bbox=bbox,
            batch_size=batch_size,
            cmap=cmap,
            savepath=None,
            save_npz_path=None,
            show=False,
            show_next=show_next,
            vel_type_for_next=vel_type,
            dt_next=dt_next,
            band_eps_vel=band_eps_vel,
            h_vel=h_vel,
            tau_vel=tau_vel,
            clip_q_vel=clip_q_vel,
            get_f1_f2=_get_f1_f2,
            fallback_circles=data_cfg.circles,
        )
        print("[Residual] using on-the-fly computed residual")

    phi_model_map = np.asarray(out["phi_model"], dtype=np.float32)
    phi_next_map = None if out["phi_next"] is None else np.asarray(out["phi_next"], dtype=np.float32)
    if phi_source == "epoch":
        phi_model_map = _load_phi_from_epoch_snapshot(
            project_root,
            sampling_tag=sampling_tag,
            epoch_label=epoch_label,
        )
        if phi_model_map.shape != np.asarray(out["phi_model"]).shape:
            raise ValueError(
                "phi snapshot shape mismatch: "
                f"snapshot={phi_model_map.shape}, model_grid={np.asarray(out['phi_model']).shape}. "
                "Please set RUN_CONFIG['n'] to match snapshot resolution."
            )
        # Keep the same one-step increment used by training-style visualization
        # and apply it on top of epoch phi.
        if show_next and (phi_next_map is not None):
            delta_phi = np.asarray(out["phi_next"], dtype=np.float32) - np.asarray(out["phi_model"], dtype=np.float32)
            phi_next_map = phi_model_map + delta_phi
        print(f"[Phi] using epoch snapshot phi at epoch={epoch_label}")
    else:
        print("[Phi] using model-computed phi")

    _save_scatter_outputs(
        XY_residual=np.asarray(out["XY_residual"], dtype=np.float32),
        RES_val=np.asarray(out["RES_val"], dtype=np.float32),
        phi_model=phi_model_map,
        phi_true=np.asarray(out["phi_true"], dtype=np.float32),
        phi_next=phi_next_map,
        bbox=tuple(float(v) for v in np.asarray(out.get("bbox", np.asarray(bbox, dtype=np.float32))).reshape(-1)),
        kind=residual_source,
        cmap=cmap,
        show_next=show_next,
        show=show,
        save_png=save_png,
        save_npz=save_npz,
    )

    print(f"[Done] png={save_png}")
    print(f"[Done] npz={save_npz}")


if __name__ == "__main__":
    main()
