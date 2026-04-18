from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Tuple

import csv
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from config import TrainConfig, VisualizationConfig
from Fig12 import save_fig12_comparison
from level_set import evolve_phi_local, rar_refine
from loss import compute_loss
from plot_T_slice_with_phi import save_T_slice_with_phi_plot_from_fields
from pde import _grad_norm, div_kgrad
from utils import masked_partition_value
from visualization import plot_phi_heatmap, plot_residual_scatter_heat


@dataclass
class TrainState:
    f1_raw: torch.nn.Parameter
    f2_raw: torch.nn.Parameter
    loss_list_global: list = field(default_factory=list)
    epoch_list_global: list = field(default_factory=list)
    loss_list_global_item: list = field(default_factory=list)
    epoch_offset_global: int = 0
    best_loss: float = 1e10
    xy_int_const: Optional[torch.Tensor] = None

    def get_f1_f2(self) -> Tuple[torch.Tensor, torch.Tensor]:
        f1 = 10 * self.f1_raw
        f2 = 10 * self.f2_raw
        return f1, f2


def get_vel_params_for_epoch(current_epoch: int):
    if current_epoch >= 30000:
        return dict(dt_next=1e-2, band_eps_vel=0.002, h_vel=0.02, tau_vel=1e-2, clip_q_vel=0.99)
    if current_epoch >= 20000:
        return dict(dt_next=1e-3, band_eps_vel=0.02, h_vel=0.05, tau_vel=1.0, clip_q_vel=0.99)
    return dict(dt_next=1e-3, band_eps_vel=0.02, h_vel=0.05, tau_vel=1.0, clip_q_vel=0.99)


def write_loss_history_csv(loss_records: Iterable[Iterable[float]], filename: str) -> None:
    if not loss_records:
        return
    block = np.array(loss_records, dtype=np.float64)
    header = "epoch,total,data,pde,bc,interface,eik,area,f1,f2"
    np.savetxt(filename, block, delimiter=",", header=header, comments="")


def _sdf_rect(xx: np.ndarray, yy: np.ndarray, xc: float, yc: float, a: float, b: float) -> np.ndarray:
    qx = np.abs(xx - xc) - a
    qy = np.abs(yy - yc) - b
    qx_pos = np.maximum(qx, 0.0)
    qy_pos = np.maximum(qy, 0.0)
    outside = np.hypot(qx_pos, qy_pos)
    inside = np.maximum(qx, qy)
    return outside + np.minimum(inside, 0.0)


def _sdf_cross(xx: np.ndarray, yy: np.ndarray, xc: float, yc: float, lx: float, ly: float, wh: float, wv: float) -> np.ndarray:
    phi_h = _sdf_rect(xx, yy, xc, yc, a=lx / 2.0, b=wh / 2.0)
    phi_v = _sdf_rect(xx, yy, xc, yc, a=wv / 2.0, b=ly / 2.0)
    return np.minimum(phi_h, phi_v)


def _compute_phi_iou(phi_map: np.ndarray, xg: np.ndarray, yg: np.ndarray, cross_params: Tuple[float, float, float, float, float, float]) -> float:
    phi_true_map = _sdf_cross(xg, yg, *cross_params)
    pred_inside = np.asarray(phi_map, dtype=np.float64) >= 0.0
    true_inside = np.asarray(phi_true_map, dtype=np.float64) <= 0.0
    intersection = np.logical_and(pred_inside, true_inside).sum()
    union = np.logical_or(pred_inside, true_inside).sum()
    return float(intersection / union) if union else 1.0


def _annotate_heatmap_with_iou(image_path: Path, *, epoch: int, iou: float) -> None:
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    label = f"Epoch {epoch:d}  IoU={iou:.3f}"
    x0 = 14
    y0 = 12
    bbox = draw.textbbox((x0, y0), label, font=font)
    pad = 6
    rect = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rounded_rectangle(rect, radius=6, fill=(255, 255, 255, 215), outline=(30, 30, 30, 220), width=1)
    draw.text((x0, y0), label, fill=(20, 20, 20, 255), font=font)
    image.save(image_path)


def _update_iou_summary(summary_path: Path, *, epoch: int, iou: float, image_name: str) -> None:
    summary_path = Path(summary_path)
    rows = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    epoch_str = str(int(epoch))
    updated = False
    for row in rows:
        if row.get("epoch", "") == epoch_str:
            row["iou"] = f"{iou:.8f}"
            row["image"] = image_name
            updated = True
            break
    if not updated:
        rows.append({"epoch": epoch_str, "iou": f"{iou:.8f}", "image": image_name})

    rows.sort(key=lambda item: int(item["epoch"]))
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "iou", "image"])
        writer.writeheader()
        writer.writerows(rows)


def train_main(
    model,
    state: TrainState,
    epochs: int,
    *,
    xy_fit: Optional[torch.Tensor] = None,
    T_fit: Optional[torch.Tensor] = None,
    opt: Optional[torch.optim.Optimizer] = None,
    opt_phi: Optional[torch.optim.Optimizer] = None,
    lr: float = 1e-3,
    lam_weights: Optional[dict] = None,
    train_cfg: Optional[TrainConfig] = None,
    viz_cfg: Optional[VisualizationConfig] = None,
    fallback_circles: Optional[Iterable[Tuple[float, float, float]]] = None,
):
    if train_cfg is None:
        train_cfg = TrainConfig()
    if viz_cfg is None:
        viz_cfg = VisualizationConfig()

    device = next(model.parameters()).device

    if opt is None:
        opt = torch.optim.Adam(list(model.parameters()) + [state.f1_raw, state.f2_raw], lr=lr)
    elif lr is not None:
        for group in opt.param_groups:
            group["lr"] = lr

    if opt_phi is None:
        opt_phi = torch.optim.Adam(model.phi.parameters(), lr)
    elif lr is not None:
        for group in opt_phi.param_groups:
            group["lr"] = lr

    for ep in range(1, epochs + 1):
        state.epoch_offset_global += 1
        current_epoch = state.epoch_offset_global

        total_loss, d, core_loss = compute_loss(
            model,
            state.xy_int_const,
            xy_fit=xy_fit,
            T_fit=T_fit,
            lam=lam_weights,
            get_f1_f2=state.get_f1_f2,
            eps_eik=train_cfg.eps_eik,
        )
        core_loss = total_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if ep % train_cfg.rar_every == 0:
            state.xy_int_const = rar_refine(
                state.xy_int_const,
                model,
                state.get_f1_f2,
                n_cand=train_cfg.rar_n_cand,
                n_new=train_cfg.rar_n_new,
                band_eps=train_cfg.rar_band_eps,
                corner_tol=train_cfg.rar_corner_tol,
                batch_size=train_cfg.rar_batch_size,
            )
            print(f"[RAR] epoch(local) {ep:>6d} | new training points -> {len(state.xy_int_const):,}")

        if ep % 2000 == 0 and (3e4 >= current_epoch >= 2e4):
            evolve_phi_local(
                model,
                state.xy_int_const,
                opt_phi,
                state.get_f1_f2,
                dt=1.0,
                n_inner=10,
                band_eps=0.05,
                h=0.05,
                tau=1.0,
                typeVn="CV",
                fallback_circles=fallback_circles,
            )

        if ep % 500 == 0 and (4e4 >= current_epoch >= 3e4):
            evolve_phi_local(
                model,
                state.xy_int_const,
                opt_phi,
                state.get_f1_f2,
                dt=1e-3,
                n_inner=10,
                band_eps=0.02,
                h=0.05,
                tau=1.0,
                typeVn="PDE",
                fallback_circles=fallback_circles,
            )
        if ep % 500 == 0 and (current_epoch >= 4e4):
            evolve_phi_local(
                model,
                state.xy_int_const,
                opt_phi,
                state.get_f1_f2,
                dt=1e-3,
                n_inner=20,
                band_eps=0.005,
                h=0.02,
                tau=1e-1,
                typeVn="PDE",
                fallback_circles=fallback_circles,
            )

        if ep % train_cfg.print_every == 0:
            f1_val, f2_val = state.get_f1_f2()
            f1_val_f = float(f1_val.detach().cpu())
            f2_val_f = float(f2_val.detach().cpu())

            state.loss_list_global.append(total_loss.item())
            state.epoch_list_global.append(current_epoch)

            print(
                f"E{current_epoch:>6d}  "
                f"total={total_loss.item():.3e}  "
                f"fit={d['data'].item():.1e}  "
                f"PDE={d['pde'].item():.1e}  "
                f"bc={d['bc'].item():.1e}  "
                f"if={d['interface'].item():.1e}  "
                f"eik={d['eik'].item():.1e}  "
                f"area={d['area'].item():.1e}  "
                f"f1={f1_val_f:.4f}  "
                f"f2={f2_val_f:.4f}"
            )

            phi_dir = viz_cfg.phi_snapshots_dir
            os.makedirs(phi_dir, exist_ok=True)
            n_phi = train_cfg.phi_snapshot_n
            bbox_phi = train_cfg.phi_snapshot_bbox
            was_training = model.training
            model.eval()
            xg = torch.linspace(bbox_phi[0], bbox_phi[1], n_phi, device=device)
            yg = torch.linspace(bbox_phi[2], bbox_phi[3], n_phi, device=device)
            Xg, Yg = torch.meshgrid(xg, yg, indexing="ij")
            xy_grid = torch.stack([Xg, Yg], dim=-1).reshape(-1, 2)
            with torch.no_grad():
                phi_grid = model.phi(xy_grid).reshape(n_phi, n_phi).detach().cpu().numpy()
                xg_np = xg.detach().cpu().numpy()
                yg_np = yg.detach().cpu().numpy()

            xy_req = xy_grid.detach().clone().requires_grad_(True)
            phi_req, T1_req, T2_req = model(xy_req)
            f1_req, f2_req = state.get_f1_f2()
            R1 = div_kgrad(T1_req, f1_req, xy_req, keep_graph=True)
            R2 = div_kgrad(T2_req, f2_req, xy_req, keep_graph=True)
            T_blend_np = masked_partition_value(phi_req, T1_req, T2_req).detach().cpu().numpy().reshape(n_phi, n_phi)
            pde_res = masked_partition_value(phi_req, R1, R2)
            gR1 = _grad_norm(R1, xy_req)
            gR2 = _grad_norm(R2, xy_req)
            grad_pde_res = masked_partition_value(phi_req, gR1, gR2)
            pde_res_np = pde_res.detach().cpu().numpy().reshape(n_phi, n_phi)
            grad_pde_res_np = grad_pde_res.detach().cpu().numpy().reshape(n_phi, n_phi)

            if was_training:
                model.train()
            np.savez_compressed(
                os.path.join(phi_dir, f"phi_epoch_{current_epoch:08d}.npz"),
                epoch=int(current_epoch),
                x=xg_np,
                y=yg_np,
                phi=phi_grid,
                T=T_blend_np,
                pde_residual=pde_res_np,
                grad_pde_residual=grad_pde_res_np,
                bbox=np.asarray(bbox_phi, dtype=np.float64),
                f1=float(f1_req.detach().cpu()),
                f2=float(f2_req.detach().cpu()),
            )

            if viz_cfg.slice_snapshot_every and (current_epoch % viz_cfg.slice_snapshot_every == 0):
                slice_dir = Path(viz_cfg.slice_snapshots_dir)
                slice_dir.mkdir(parents=True, exist_ok=True)
                save_T_slice_with_phi_plot_from_fields(
                    x_axis=xg_np,
                    y_axis=yg_np,
                    T_pred_map=T_blend_np,
                    phi_map=phi_grid,
                    txt_filename=str(Path(__file__).resolve().parent / "Possion.txt"),
                    save_path=slice_dir / f"T_slice_with_phi_{current_epoch:08d}.png",
                    save_npz_path=slice_dir / f"T_slice_with_phi_{current_epoch:08d}.npz",
                    epoch=current_epoch,
                    bbox=tuple(float(v) for v in bbox_phi),
                    title_prefix="ADD-PINNs",
                    percentile=viz_cfg.slice_percentile,
                    max_points=viz_cfg.slice_points,
                    eps_eik=train_cfg.eps_eik,
                )

            if viz_cfg.phi_heatmap_every and (current_epoch % viz_cfg.phi_heatmap_every == 0):
                heat_dir = viz_cfg.phi_heatmap_dir
                os.makedirs(heat_dir, exist_ok=True)
                heat_path = os.path.join(heat_dir, f"phi_heatmap_{current_epoch:08d}.png")
                heatmap_data = plot_phi_heatmap(
                    model,
                    bbox=viz_cfg.phi_heatmap_bbox,
                    n=viz_cfg.phi_heatmap_n,
                    savepath=heat_path,
                    dpi=viz_cfg.phi_heatmap_dpi,
                    show=False,
                )
                iou = _compute_phi_iou(
                    heatmap_data["phi_map"],
                    heatmap_data["Xg"],
                    heatmap_data["Yg"],
                    viz_cfg.phi_compare_cross,
                )
                _annotate_heatmap_with_iou(Path(heat_path), epoch=current_epoch, iou=iou)
                _update_iou_summary(
                    Path(heat_dir) / "iou_summary.csv",
                    epoch=current_epoch,
                    iou=iou,
                    image_name=Path(heat_path).name,
                )
                print(f"[IOU] epoch={current_epoch:>6d}  IoU={iou:.6f}")

            if ep % train_cfg.viz_every == 0:
                viz_dir = viz_cfg.viz_scatter_dir
                os.makedirs(viz_dir, exist_ok=True)
                project_root = Path(__file__).resolve().parent
                snapshot_npz = Path(phi_dir) / f"phi_epoch_{current_epoch:08d}.npz"
                fig12_dir = Path(viz_cfg.fig12_snapshots_dir)
                fig12_dir.mkdir(parents=True, exist_ok=True)
                save_fig12_comparison(
                    txt_path=project_root / "Possion.txt",
                    pinn_path=project_root / "outputs_pinn_single" / "final_fields.npz",
                    apinn_path=project_root / "outputs_apinn" / "final_fields.npz",
                    pimoe_path=snapshot_npz,
                    pimoe_pde_path=snapshot_npz,
                    out_path=fig12_dir / f"Fig12_epoch_{current_epoch:08d}.png",
                )

                param_common = get_vel_params_for_epoch(current_epoch)
                viz_cfgs = [
                    ("PDE", param_common.copy()),
                    ("GRAD", param_common.copy()),
                    ("CV", param_common.copy()),
                ]

                for vel_type_for_next, param_dict in viz_cfgs:
                    kind_to_use = "grad" if vel_type_for_next.upper() == "GRAD" else "pde"
                    viz_path = os.path.join(
                        viz_dir, f"scatter_{vel_type_for_next}_epoch_{current_epoch:08d}.png"
                    )
                    npz_path = os.path.splitext(viz_path)[0] + ".npz"
                    plot_residual_scatter_heat(
                        model,
                        kind=kind_to_use,
                        xy_fit=xy_fit,
                        T_fit=T_fit,
                        n=viz_cfg.scatter_n,
                        bbox=viz_cfg.scatter_bbox,
                        batch_size=viz_cfg.scatter_batch_size,
                        cross_params=viz_cfg.cross_params,
                        cmap="cividis",
                        savepath=viz_path,
                        save_npz_path=npz_path,
                        show=False,
                        show_next=True,
                        vel_type_for_next=vel_type_for_next,
                        get_f1_f2=state.get_f1_f2,
                        fallback_circles=fallback_circles,
                        **param_dict,
                    )

        if ep % train_cfg.record_every == 0:
            f1_val, f2_val = state.get_f1_f2()
            f1_val_f = float(f1_val.detach().cpu())
            f2_val_f = float(f2_val.detach().cpu())
            row = [
                float(current_epoch),
                float(total_loss.detach().cpu()),
                float(d["data"].detach().cpu()),
                float(d["pde"].detach().cpu()),
                float(d["bc"].detach().cpu()),
                float(d["interface"].detach().cpu()),
                float(d["eik"].detach().cpu()),
                float(d["area"].detach().cpu()),
                f1_val_f,
                f2_val_f,
            ]
            state.loss_list_global_item.append(row)
            write_loss_history_csv(state.loss_list_global_item, viz_cfg.loss_csv_file)

    return model, opt, opt_phi, state.xy_int_const
