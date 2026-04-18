from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import os
import numpy as np
import torch

from config import TrainConfig, VisualizationConfig
from level_set import evolve_phi_local, rar_refine
from loss import compute_loss
from plot_uf_slice_with_phi import save_uf_slice_with_phi_plot
from pde import _grad_norm, div_kgrad
from problem import exact_solution, f_region_inside, f_region_outside, phi_signed_flower
from visualization import (
    plot_f_true_pred_residual_heatmap,
    plot_phi_heatmap,
    plot_residual_scatter_heat,
    plot_u_true_pred_residual_heatmap,
)


@dataclass
class TrainState:
    loss_list_global: list = field(default_factory=list)
    epoch_list_global: list = field(default_factory=list)
    loss_list_global_item: list = field(default_factory=list)
    epoch_offset_global: int = 0
    best_loss: float = 1e10
    xy_int_const: Optional[torch.Tensor] = None
    xy_bnd_const: Optional[torch.Tensor] = None

    def get_f1_f2(self, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Keep the same branch/sign convention as loss.py:
        #   phi > 0 -> u1 (inside), phi < 0 -> u2 (outside)
        return f_region_inside(xy), f_region_outside(xy)


def get_vel_params_for_epoch(current_epoch: int):
    if current_epoch >= 30000:
        return dict(
            dt_next=1e-2,
            band_eps_vel=0.002,
            h_vel=0.02,
            tau_vel=1e-2,
            clip_q_vel=0.99,
        )
    if current_epoch >= 20000:
        return dict(
            dt_next=1e-3,
            band_eps_vel=0.02,
            h_vel=0.05,
            tau_vel=1.0,
            clip_q_vel=0.99,
        )
    return dict(dt_next=1e-3, band_eps_vel=0.02, h_vel=0.05, tau_vel=1.0, clip_q_vel=0.99)


def _relative_l2(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> float:
    num = torch.linalg.norm((pred - true).reshape(-1))
    den = torch.linalg.norm(true.reshape(-1)).clamp_min(eps)
    return float((num / den).detach().cpu())


def _compute_rel_l2_u_f(
    model,
    *,
    bbox: Tuple[float, float, float, float],
    n: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float]:
    was_training = model.training
    model.eval()

    x = torch.linspace(bbox[0], bbox[1], n, device=device)
    y = torch.linspace(bbox[2], bbox[3], n, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    XY_grid = torch.stack([X, Y], dim=-1).reshape(-1, 2)

    u_pred_chunks = []
    u_true_chunks = []
    f_pred_chunks = []
    f_true_chunks = []

    for i0 in range(0, XY_grid.shape[0], batch_size):
        i1 = min(XY_grid.shape[0], i0 + batch_size)
        XY_b = XY_grid[i0:i1].detach().clone().requires_grad_(True)

        phi_b, u1_b, u2_b = model(XY_b)
        mask_pos = (phi_b >= 0).to(phi_b.dtype)
        mask_neg = 1.0 - mask_pos
        u_pred_b = mask_pos * u1_b + mask_neg * u2_b

        zeros = torch.zeros_like(u1_b)
        lap_u1_b = div_kgrad(u1_b, zeros, XY_b, keep_graph=True)
        lap_u2_b = div_kgrad(u2_b, zeros, XY_b)
        f1_pred_b = -lap_u1_b
        f2_pred_b = -lap_u2_b
        f_pred_b = mask_pos * f1_pred_b + mask_neg * f2_pred_b

        XY_det = XY_b.detach()
        with torch.no_grad():
            u_true_b = exact_solution(XY_det)
            f1_true_b = f_region_inside(XY_det)
            f2_true_b = f_region_outside(XY_det)
            phi_true_b = phi_signed_flower(XY_det)
            f_true_b = torch.where(phi_true_b >= 0.0, f1_true_b, f2_true_b)

        u_pred_chunks.append(u_pred_b.detach().cpu())
        u_true_chunks.append(u_true_b.detach().cpu())
        f_pred_chunks.append(f_pred_b.detach().cpu())
        f_true_chunks.append(f_true_b.detach().cpu())

    if was_training:
        model.train()

    u_pred_all = torch.cat(u_pred_chunks, dim=0)
    u_true_all = torch.cat(u_true_chunks, dim=0)
    f_pred_all = torch.cat(f_pred_chunks, dim=0)
    f_true_all = torch.cat(f_true_chunks, dim=0)

    return _relative_l2(u_pred_all, u_true_all), _relative_l2(f_pred_all, f_true_all)


def _compute_rel_l2_label_u(
    model,
    *,
    xy_fit: Optional[torch.Tensor],
    u_fit: Optional[torch.Tensor],
) -> float:
    if xy_fit is None or u_fit is None:
        return float("nan")

    was_training = model.training
    model.eval()
    with torch.no_grad():
        phi_f, u1_f, u2_f = model(xy_fit)
        mask_pos = (phi_f >= 0).to(phi_f.dtype)
        mask_neg = 1.0 - mask_pos
        u_pred_f = mask_pos * u1_f + mask_neg * u2_f
        rel = _relative_l2(u_pred_f, u_fit)
    if was_training:
        model.train()
    return rel


def _rebalance_xy_by_model_phi(
    model,
    *,
    n_pos: int,
    n_neg: int,
    device: torch.device,
    corner_tol: float,
    batch_size: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    max_rounds: int = 400,
) -> torch.Tensor:
    x0, x1 = xlim
    y0, y1 = ylim
    mx = corner_tol * (x1 - x0)
    my = corner_tol * (y1 - y0)

    was_training = model.training
    model.eval()

    pos_list = []
    neg_list = []
    n_pos_now = 0
    n_neg_now = 0
    use_true_phi_fallback = False

    for _ in range(max_rounds):
        if n_pos_now >= n_pos and n_neg_now >= n_neg:
            break

        x = torch.rand(batch_size, 1, device=device) * (x1 - x0) + x0
        y = torch.rand(batch_size, 1, device=device) * (y1 - y0) + y0
        xy_batch = torch.cat([x, y], dim=1)
        mask_inner = (
            (xy_batch[:, 0] > x0 + mx)
            & (xy_batch[:, 0] < x1 - mx)
            & (xy_batch[:, 1] > y0 + my)
            & (xy_batch[:, 1] < y1 - my)
        )
        xy_valid = xy_batch[mask_inner]
        if xy_valid.numel() == 0:
            continue

        with torch.no_grad():
            if use_true_phi_fallback:
                phi_v = phi_signed_flower(xy_valid).squeeze(1)
            else:
                phi_v = model.phi(xy_valid).squeeze(1)

        if n_pos_now < n_pos:
            xy_pos = xy_valid[phi_v > 0]
            if xy_pos.numel() > 0:
                need = n_pos - n_pos_now
                take = min(need, int(xy_pos.shape[0]))
                pos_list.append(xy_pos[:take])
                n_pos_now += take

        if n_neg_now < n_neg:
            xy_neg = xy_valid[phi_v < 0]
            if xy_neg.numel() > 0:
                need = n_neg - n_neg_now
                take = min(need, int(xy_neg.shape[0]))
                neg_list.append(xy_neg[:take])
                n_neg_now += take

        # If model phi is nearly one-sided, switch to true-phi fallback after half rounds.
        if (_ + 1) >= (max_rounds // 2) and (n_pos_now == 0 or n_neg_now == 0):
            use_true_phi_fallback = True

    if was_training:
        model.train()

    if n_pos_now < n_pos or n_neg_now < n_neg:
        raise RuntimeError(
            f"rebalance failed: got pos={n_pos_now}/{n_pos}, neg={n_neg_now}/{n_neg}"
        )

    xy_pos_all = torch.cat(pos_list, dim=0)[:n_pos]
    xy_neg_all = torch.cat(neg_list, dim=0)[:n_neg]
    xy_all = torch.cat([xy_pos_all, xy_neg_all], dim=0)
    perm = torch.randperm(xy_all.shape[0], device=device)
    return xy_all[perm]


def write_loss_history_csv(loss_records: Iterable[Iterable[float]], filename: str) -> None:
    if not loss_records:
        return
    block = np.array(loss_records, dtype=np.float64)
    if block.shape[1] >= 17:
        header = (
            "epoch,total,data,pde,bc,interface,eik,area,perimeter,"
            "total_raw,data_raw,pde_raw,bc_raw,interface_raw,eik_raw,area_raw,perimeter_raw"
        )
    else:
        header = "epoch,total,data,pde,bc,interface,eik,area,perimeter"
    np.savetxt(filename, block, delimiter=",", header=header, comments="")


def train_main(
    model,
    state: TrainState,
    epochs: int,
    *,
    xy_fit: Optional[torch.Tensor] = None,
    u_fit: Optional[torch.Tensor] = None,
    opt: Optional[torch.optim.Optimizer] = None,
    opt_phi: Optional[torch.optim.Optimizer] = None,
    lr: float = 1e-3,
    phi_lr: Optional[float] = None,
    lam_weights: Optional[dict] = None,
    train_cfg: Optional[TrainConfig] = None,
    viz_cfg: Optional[VisualizationConfig] = None,
    fallback_circles: Optional[Iterable[Tuple[float, float, float]]] = None,
):
    if train_cfg is None:
        train_cfg = TrainConfig()
    if viz_cfg is None:
        viz_cfg = VisualizationConfig()
    if lam_weights is None:
        lam_weights = train_cfg.lam_weights

    device = next(model.parameters()).device

    if opt is None:
        opt = torch.optim.Adam(list(model.parameters()), lr=lr)
    elif lr is not None:
        for group in opt.param_groups:
            group["lr"] = lr

    phi_lr_eff = lr if phi_lr is None else phi_lr
    if phi_lr_eff is None:
        phi_lr_eff = 1e-3

    if opt_phi is None:
        opt_phi = torch.optim.Adam(model.phi.parameters(), lr=phi_lr_eff)
    elif phi_lr_eff is not None:
        for group in opt_phi.param_groups:
            group["lr"] = phi_lr_eff

    for ep in range(1, epochs + 1):
        state.epoch_offset_global += 1
        current_epoch = state.epoch_offset_global

        lam_step = dict(lam_weights)
        if train_cfg.eik_decay_enabled:
            steps = max(int(train_cfg.eik_decay_steps), 1)
            ratio = max(float(train_cfg.eik_decay_ratio), 1e-12)
            decay_rate = np.log(1.0 / ratio) / steps
            decay_mult = np.exp(-decay_rate * current_epoch)
            for key in ("eik", "perimeter"):
                if key in lam_step:
                    lam_step[key] = float(lam_weights[key]) * decay_mult

        total_loss, d, d_raw = compute_loss(
            model,
            state.xy_int_const,
            xy_fit=xy_fit,
            u_fit=u_fit,
            xy_bnd=state.xy_bnd_const,
            lam=lam_step,
        )
        core_loss = total_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if current_epoch % 5000 == 0:
            rel_u, rel_f = _compute_rel_l2_u_f(
                model,
                bbox=train_cfg.phi_snapshot_bbox,
                n=train_cfg.phi_snapshot_n,
                batch_size=train_cfg.rar_batch_size,
                device=device,
            )
            rel_u_label = _compute_rel_l2_label_u(
                model,
                xy_fit=xy_fit,
                u_fit=u_fit,
            )
            print(
                f"[RelL2] epoch(global) {current_epoch:>6d} | "
                f"u={rel_u:.3e}, f={rel_f:.3e}, label_u={rel_u_label:.3e}",
                flush=True,
            )

            state.xy_int_const = _rebalance_xy_by_model_phi(
                model,
                n_pos=2000,
                n_neg=2000,
                device=device,
                corner_tol=train_cfg.corner_tol,
                batch_size=train_cfg.sample_batch_size,
                xlim=train_cfg.xy_int_xlim,
                ylim=train_cfg.xy_int_ylim,
            )
            print(
                f"[Rebalance] epoch(global) {current_epoch:>6d} | "
                f"phi>0=2000, phi<0=2000 -> {len(state.xy_int_const):,}"
            )

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
                xlim=train_cfg.xy_int_xlim,
                ylim=train_cfg.xy_int_ylim,
            )
            print(
                f"[RAR] epoch(local) {ep:>6d} | new training points -> {len(state.xy_int_const):,}"
            )

        # if ep % 2000 == 0 and (3e4 >= current_epoch >= 1e4):
        #     evolve_phi_local(
        #         model,
        #         state.xy_int_const,
        #         opt_phi,
        #         state.get_f1_f2,
        #         dt=1.0,
        #         n_inner=10,
        #         band_eps=0.05,
        #         h=0.05,
        #         tau=1.0,
        #         typeVn="CV",
        #         fallback_circles=fallback_circles,
        #     )

        # if ep % 500 == 0 and (current_epoch >= 1e4):
        #     evolve_phi_local(
        #         model,
        #         state.xy_int_const,
        #         opt_phi,
        #         state.get_f1_f2,
        #         dt=1e-3,
        #         n_inner=10,
        #         band_eps=0.02,
        #         h=0.05,
        #         tau=1.0,
        #         typeVn="PDE",
        #         fallback_circles=fallback_circles,
        #     )

        if ep % 500 == 0 and (current_epoch >= 5e3):
            evolve_phi_local(
                model,
                state.xy_int_const,
                opt_phi,
                state.get_f1_f2,
                dt=1e-3,
                n_inner=20,
                band_eps=0.05,
                h=0.05,
                tau=1e-3,
                typeVn="PDE",
                fallback_circles=fallback_circles,
            )

        if ep % train_cfg.print_every == 0:
            state.loss_list_global.append(total_loss.item())
            state.epoch_list_global.append(current_epoch)

            print(
                f"E{current_epoch:>6d}  "
                f"total={total_loss.item():.3e}  "
                f"fit={d['data'].item():.3e}  "
                f"PDE={d['pde'].item():.3e}  "
                f"bc={d['bc'].item():.3e}  "
                f"if={d['interface'].item():.3e}  "
                f"eik={d['eik'].item():.3e}  "
                f"area={d['area'].item():.3e}  "
                f"per={d['perimeter'].item():.3e}"
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
            phi_req, u1_req, u2_req = model(xy_req)
            f1_req, f2_req = state.get_f1_f2(xy_req)
            R1 = div_kgrad(u1_req, f1_req, xy_req, keep_graph=True)
            R2 = div_kgrad(u2_req, f2_req, xy_req, keep_graph=True)
            mask_pos = (phi_req >= 0).to(phi_req.dtype)
            mask_neg = 1.0 - mask_pos
            pde_res = mask_pos * R1 + mask_neg * R2
            gR1 = _grad_norm(R1, xy_req)
            gR2 = _grad_norm(R2, xy_req)
            grad_pde_res = mask_pos * gR1 + mask_neg * gR2
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
                pde_residual=pde_res_np,
                grad_pde_residual=grad_pde_res_np,
                bbox=np.asarray(bbox_phi, dtype=np.float64),
                f1_mean=float(f1_req.detach().cpu().mean()),
                f2_mean=float(f2_req.detach().cpu().mean()),
            )

            if viz_cfg.phi_heatmap_every and (current_epoch % viz_cfg.phi_heatmap_every == 0):
                heat_dir = viz_cfg.phi_heatmap_dir
                os.makedirs(heat_dir, exist_ok=True)
                heat_path = os.path.join(heat_dir, f"phi_heatmap_{current_epoch:08d}.png")
                plot_phi_heatmap(
                    model,
                    bbox=viz_cfg.phi_heatmap_bbox,
                    n=viz_cfg.phi_heatmap_n,
                    savepath=heat_path,
                    dpi=viz_cfg.phi_heatmap_dpi,
                    show=False,
                )
            if ep % train_cfg.viz_every == 0:
                viz_dir = viz_cfg.viz_scatter_dir
                os.makedirs(viz_dir, exist_ok=True)

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
                        u_fit=u_fit,
                        n=viz_cfg.scatter_n,
                        bbox=viz_cfg.scatter_bbox,
                        batch_size=viz_cfg.scatter_batch_size,
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

        if viz_cfg.u_heatmap_every and (current_epoch % viz_cfg.u_heatmap_every == 0):
            u_npz_path = None
            u_heat_dir = viz_cfg.u_heatmap_dir
            os.makedirs(u_heat_dir, exist_ok=True)
            u_heat_path = os.path.join(u_heat_dir, f"u_heatmap_{current_epoch:08d}.png")
            u_npz_path = os.path.splitext(u_heat_path)[0] + ".npz"
            plot_u_true_pred_residual_heatmap(
                model,
                bbox=viz_cfg.u_heatmap_bbox,
                n=viz_cfg.u_heatmap_n,
                savepath=u_heat_path,
                save_npz_path=u_npz_path,
                dpi=viz_cfg.u_heatmap_dpi,
                show=False,
            )
        else:
            u_npz_path = None
        if viz_cfg.f_heatmap_every and (current_epoch % viz_cfg.f_heatmap_every == 0):
            f_npz_path = None
            f_heat_dir = viz_cfg.f_heatmap_dir
            os.makedirs(f_heat_dir, exist_ok=True)
            f_heat_path = os.path.join(f_heat_dir, f"f_heatmap_{current_epoch:08d}.png")
            f_npz_path = os.path.splitext(f_heat_path)[0] + ".npz"
            plot_f_true_pred_residual_heatmap(
                model,
                bbox=viz_cfg.f_heatmap_bbox,
                n=viz_cfg.f_heatmap_n,
                savepath=f_heat_path,
                save_npz_path=f_npz_path,
                dpi=viz_cfg.f_heatmap_dpi,
                show=False,
            )
        else:
            f_npz_path = None

        if viz_cfg.uf_slice_with_phi_every and (current_epoch % viz_cfg.uf_slice_with_phi_every == 0):
            if u_npz_path is None:
                candidate_u = os.path.join(
                    viz_cfg.u_heatmap_dir, f"u_heatmap_{current_epoch:08d}.npz"
                )
                if os.path.exists(candidate_u):
                    u_npz_path = candidate_u
            if f_npz_path is None:
                candidate_f = os.path.join(
                    viz_cfg.f_heatmap_dir, f"f_heatmap_{current_epoch:08d}.npz"
                )
                if os.path.exists(candidate_f):
                    f_npz_path = candidate_f

            phi_npz_path = os.path.join(
                viz_cfg.phi_snapshots_dir, f"phi_epoch_{current_epoch:08d}.npz"
            )

            try:
                if (
                    u_npz_path is not None
                    and f_npz_path is not None
                    and os.path.exists(phi_npz_path)
                ):
                    os.makedirs(viz_cfg.uf_slice_with_phi_dir, exist_ok=True)
                    uf_slice_path = os.path.join(
                        viz_cfg.uf_slice_with_phi_dir,
                        f"uf_phi_line_epoch_{current_epoch:08d}.png",
                    )
                    save_uf_slice_with_phi_plot(
                        u_npz_path,
                        f_npz_path,
                        phi_npz_path,
                        save_path=uf_slice_path,
                        line_axis=viz_cfg.uf_slice_line_axis,
                        line_mode=viz_cfg.uf_slice_line_mode,
                        line_value=viz_cfg.uf_slice_line_value,
                        line_index=viz_cfg.uf_slice_line_index,
                    )
                    print("[SAVE]", uf_slice_path)
                else:
                    print(
                        f"[WARN] skip uf-slice-phi at epoch {current_epoch}: "
                        f"missing u/f/phi npz"
                    )
            except Exception as exc:
                print(
                    f"[WARN] failed to save uf-slice-phi at epoch {current_epoch}: {exc}"
                )

        if ep % train_cfg.record_every == 0:
            row = [
                float(current_epoch),
                float(total_loss.detach().cpu()),
                float(d["data"].detach().cpu()),
                float(d["pde"].detach().cpu()),
                float(d["bc"].detach().cpu()),
                float(d["interface"].detach().cpu()),
                float(d["eik"].detach().cpu()),
                float(d["area"].detach().cpu()),
                float(d["perimeter"].detach().cpu()),
                float(d_raw["total"].detach().cpu()),
                float(d_raw["data"].detach().cpu()),
                float(d_raw["pde"].detach().cpu()),
                float(d_raw["bc"].detach().cpu()),
                float(d_raw["interface"].detach().cpu()),
                float(d_raw["eik"].detach().cpu()),
                float(d_raw["area"].detach().cpu()),
                float(d_raw["perimeter"].detach().cpu()),
            ]
            state.loss_list_global_item.append(row)
            write_loss_history_csv(state.loss_list_global_item, viz_cfg.loss_csv_file)

    return model, opt, opt_phi, state.xy_int_const
