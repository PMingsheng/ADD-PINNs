import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import config
from loss import compute_loss
from plot_u_slice_with_phi import save_u_slice_with_phi_plot_from_fields
from pde import strain_from_u
from level_set import evolve_phi_local, rar_refine
from pde import lame_from_E, div_sigma_batch
from utils import TorchVarLogger
from visualization import plot_residual_scatter

EPOCH_OFFSET_GLOBAL = 0
LOSS_LIST_GLOBAL_ITEM = []


def _spatial_grad_norm(Z_2d, xmin, xmax, ymin, ymax):
    n, m = Z_2d.shape
    dx = (xmax - xmin) / (n - 1)
    dy = (ymax - ymin) / (m - 1)
    gx = np.zeros_like(Z_2d, dtype=np.float32)
    gy = np.zeros_like(Z_2d, dtype=np.float32)
    gx[1:-1, :] = (Z_2d[2:, :] - Z_2d[:-2, :]) / (2 * dx)
    gy[:, 1:-1] = (Z_2d[:, 2:] - Z_2d[:, :-2]) / (2 * dy)
    return np.sqrt(gx * gx + gy * gy)


def _get_scaled_E(model):
    if hasattr(model, "get_E_scaled"):
        return model.get_E_scaled()
    return model.E_1, model.E_2


def _blend_u(phi, ux1, uy1, ux2, uy2):
    w1 = torch.relu(phi)
    w2 = torch.relu(-phi)
    denom = w1 + w2 + 1e-12
    ux = (w1 * ux1 + w2 * ux2) / denom
    uy = (w1 * uy1 + w2 * uy2) / denom
    return torch.cat([ux, uy], dim=1)


def _blend_strain(phi, exx_1, eyy_1, exy_1, exx_2, eyy_2, exy_2):
    w1 = torch.relu(phi)
    w2 = torch.relu(-phi)
    denom = w1 + w2 + 1e-12
    E1 = torch.cat([exx_1, eyy_1, exy_1], dim=1)
    E2 = torch.cat([exx_2, eyy_2, exy_2], dim=1)
    return (w1 * E1 + w2 * E2) / denom


def train_main(
    model,
    xy_int_const,
    *,
    xy_u=None,
    U_fit=None,
    xy_eps=None,
    E_fit=None,
    epochs=None,
    lr=None,
    phi_lr=None,
    lam_weights=None,
    nu=None,
    eps0=None,
):
    if epochs is None:
        epochs = config.TRAIN_CONFIG["epochs"]
    if lr is None:
        lr = config.TRAIN_CONFIG["lr"]
    if phi_lr is None:
        phi_lr = config.TRAIN_CONFIG["phi_lr"]
    if lam_weights is None:
        lam_weights = config.LAM_WEIGHTS
    if nu is None:
        nu = config.NU
    if eps0 is None:
        eps0 = config.EPS0

    global EPOCH_OFFSET_GLOBAL, LOSS_LIST_GLOBAL_ITEM

    device = next(model.parameters()).device

    if not hasattr(model, "E_1"):
        model.E_1 = torch.nn.Parameter(torch.tensor(0.1, device=device))
    if not hasattr(model, "E_2"):
        model.E_2 = torch.nn.Parameter(torch.tensor(0.1, device=device))

    main_params = list(model.net_1.parameters()) + list(model.net_2.parameters())
    opt = torch.optim.Adam(main_params, lr=lr)

    phi_params = list(model.phi.parameters()) + [model.E_1, model.E_2]
    opt_phi = torch.optim.Adam(phi_params, lr=phi_lr)

    loss_hist = []

    if EPOCH_OFFSET_GLOBAL == 0:
        LOSS_LIST_GLOBAL_ITEM = []
        if config.LOSS_CSV_PATH.exists():
            try:
                os.remove(config.LOSS_CSV_PATH)
            except OSError:
                pass

    if EPOCH_OFFSET_GLOBAL == 0 and config.E_HISTORY_PATH.exists():
        try:
            os.remove(config.E_HISTORY_PATH)
        except OSError:
            pass
    logger = TorchVarLogger(str(config.E_HISTORY_PATH), overwrite=False)

    loss_list_global = []
    epoch_list_global = []
    for ep in range(1, epochs + 1):
        EPOCH_OFFSET_GLOBAL += 1
        current_epoch = EPOCH_OFFSET_GLOBAL

        tot, d, core_loss = compute_loss(
            model,
            xy_int_const,
            xy_u=xy_u,
            U_fit=U_fit,
            xy_eps=xy_eps,
            E_fit=E_fit,
            lam=lam_weights,
            nu=nu,
            eps0=eps0,
        )
        core_loss = tot

        opt.zero_grad()
        opt_phi.zero_grad()
        tot.backward()
        opt.step()
        opt_phi.step()

        loss_hist.append(tot.item())

        if ep % config.TRAIN_CONFIG["rar_every"] == 0:
            try:
                xy_int_const = rar_refine(
                    xy_int_const,
                    model,
                    None,
                    None,
                    n_cand=config.TRAIN_CONFIG["rar_n_cand"],
                    n_new=config.TRAIN_CONFIG["rar_n_new"],
                    nu=nu,
                )
            except TypeError:
                xy_int_const = rar_refine(
                    xy_int_const,
                    model,
                    n_cand=config.TRAIN_CONFIG["rar_n_cand"],
                    n_new=config.TRAIN_CONFIG["rar_n_new"],
                    nu=nu,
                )
            print(f"[RAR] epoch {ep:>6d} | new training points {len(xy_int_const):,}")

        if (
            ep % config.TRAIN_CONFIG["phi_evolve_every"] == 0
            and config.TRAIN_CONFIG["phi_evolve_stage1"]["start"]
            <= current_epoch
            <= config.TRAIN_CONFIG["phi_evolve_stage1"]["end"]
        ):
            stage = config.TRAIN_CONFIG["phi_evolve_stage1"]
            evolve_phi_local(
                model,
                xy_int_const,
                opt_phi,
                dt=stage["dt"],
                n_inner=stage["n_inner"],
                band_eps=stage["band_eps"],
                h=stage["h"],
                tau=stage["tau"],
                typeVn=stage["typeVn"],
                nu=nu,
            )

        if (
            ep % config.TRAIN_CONFIG["phi_evolve_every"] == 0
            and current_epoch >= config.TRAIN_CONFIG["phi_evolve_stage2"]["start"]
        ):
            stage = config.TRAIN_CONFIG["phi_evolve_stage2"]
            evolve_phi_local(
                model,
                xy_int_const,
                opt_phi,
                dt=stage["dt"],
                n_inner=stage["n_inner"],
                band_eps=stage["band_eps"],
                h=stage["h"],
                tau=stage["tau"],
                typeVn=stage["typeVn"],
                nu=nu,
            )

        if ep % config.TRAIN_CONFIG["log_every"] == 0:
            loss_list_global.append(tot.item())
            epoch_list_global.append(current_epoch)
            w_data = lam_weights.get("data", 1.0)
            w_pde = lam_weights.get("pde", 1.0)
            w_bc = lam_weights.get("bc", 1.0)
            w_if = lam_weights.get("interface", 1.0)
            w_eik = lam_weights.get("eik", 1.0)
            w_area = lam_weights.get("area", 1.0)
            data_w = d["data"].item() * w_data
            pde_w = d["pde"].item() * w_pde
            bc_w = d["bc"].item() * w_bc
            if_w = d["interface"].item() * w_if
            eik_w = d["eik"].item() * w_eik
            area_w = d["area"].item() * w_area
            E_1_scaled, E_2_scaled = _get_scaled_E(model)
            print(
                f"E{current_epoch:>5d}  total={tot.item():.3e}  "
                f"fit={data_w:.1e}  "
                f"PDE={pde_w:.1e}  "
                f"bc={bc_w:.1e}  "
                f"if={if_w:.1e}  "
                f"eik={eik_w:.1e}  "
                f"area={area_w:.1e}  "
                f"E_1={float(E_1_scaled.detach().cpu()):.4f}  "
                f"E_2={float(E_2_scaled.detach().cpu()):.4f}"
            )

        if ep % config.TRAIN_CONFIG["phi_snapshot_every"] == 0:
            config.PHI_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            n_phi = config.TRAIN_CONFIG["phi_snapshot_n"]
            bbox_phi = config.TRAIN_CONFIG["phi_snapshot_bbox"]
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

            E_1_scaled, E_2_scaled = _get_scaled_E(model)
            lam_1, mu_1 = lame_from_E(E_1_scaled, nu)
            lam_2, mu_2 = lame_from_E(E_2_scaled, nu)
            xy_req = xy_grid.detach().clone().requires_grad_(True)
            phi_req, ux1_req, uy1_req, ux2_req, uy2_req = model(xy_req)
            Rx1, Ry1 = div_sigma_batch(xy_req, ux1_req, uy1_req, lam_1, mu_1)
            Rx2, Ry2 = div_sigma_batch(xy_req, ux2_req, uy2_req, lam_2, mu_2)
            Rmag1 = Rx1 ** 2 + Ry1 ** 2
            Rmag2 = Rx2 ** 2 + Ry2 ** 2
            w1 = torch.relu(phi_req)
            w2 = torch.relu(-phi_req)
            denom = (w1 + w2 + 1e-12)
            pde_res = (w1 * Rmag1 + w2 * Rmag2) / denom
            pde_res_np = pde_res.detach().cpu().numpy().reshape(n_phi, n_phi)
            grad_pde_res_np = _spatial_grad_norm(
                pde_res_np, bbox_phi[0], bbox_phi[1], bbox_phi[2], bbox_phi[3]
            ).astype(np.float32)

            if was_training:
                model.train()
            np.savez_compressed(
                config.PHI_SNAPSHOT_DIR / f"phi_epoch_{current_epoch:08d}.npz",
                epoch=int(current_epoch),
                x=xg_np,
                y=yg_np,
                phi=phi_grid,
                ux=((w1 * ux1_req + w2 * ux2_req) / denom).detach().cpu().numpy().reshape(n_phi, n_phi),
                uy=((w1 * uy1_req + w2 * uy2_req) / denom).detach().cpu().numpy().reshape(n_phi, n_phi),
                pde_residual=pde_res_np,
                grad_pde_residual=grad_pde_res_np,
                bbox=np.asarray(bbox_phi, dtype=np.float64),
                E1=float(E_1_scaled.detach().cpu()),
                E2=float(E_2_scaled.detach().cpu()),
            )
            config.SLICE_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            save_u_slice_with_phi_plot_from_fields(
                x_axis=xg_np,
                y_axis=yg_np,
                ux_pred_map=((w1 * ux1_req + w2 * ux2_req) / denom).detach().cpu().numpy().reshape(n_phi, n_phi),
                uy_pred_map=((w1 * uy1_req + w2 * uy2_req) / denom).detach().cpu().numpy().reshape(n_phi, n_phi),
                phi_map=phi_grid,
                ellipse=config.ELLIPSE_PARAMS,
                txt_filename=config.DATA_FILE,
                save_path=config.SLICE_SNAPSHOT_DIR / f"u_slice_with_phi_{current_epoch:08d}.png",
                save_npz_path=config.SLICE_SNAPSHOT_DIR / f"u_slice_with_phi_{current_epoch:08d}.npz",
                epoch=current_epoch,
                bbox=bbox_phi,
                title_prefix="ADD-PINNs",
            )

            if xy_u is not None and U_fit is not None:
                config.DISP_STRAIN_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                xy_u_req = xy_u.detach().clone()
                with torch.no_grad():
                    phi_u, ux1_u, uy1_u, ux2_u, uy2_u = model(xy_u_req)
                    U_pred = _blend_u(phi_u, ux1_u, uy1_u, ux2_u, uy2_u)
                U_true = U_fit.detach()
                U_err = U_pred - U_true
                U_err_norm = torch.linalg.norm(U_err, dim=1, keepdim=True)

                save_data = {
                    "epoch": int(current_epoch),
                    "xy_u": xy_u_req.detach().cpu().numpy(),
                    "U_pred": U_pred.detach().cpu().numpy(),
                    "U_true": U_true.detach().cpu().numpy(),
                    "U_err": U_err.detach().cpu().numpy(),
                    "U_err_norm": U_err_norm.detach().cpu().numpy(),
                }

                if xy_eps is not None and E_fit is not None:
                    xy_eps_req = xy_eps.detach().clone().requires_grad_(True)
                    phi_e, ux1_e, uy1_e, ux2_e, uy2_e = model(xy_eps_req)
                    U1_e = torch.cat([ux1_e, uy1_e], dim=1)
                    U2_e = torch.cat([ux2_e, uy2_e], dim=1)
                    exx_1, eyy_1, exy_1 = strain_from_u(U1_e, xy_eps_req)
                    exx_2, eyy_2, exy_2 = strain_from_u(U2_e, xy_eps_req)
                    E_pred = _blend_strain(phi_e, exx_1, eyy_1, exy_1, exx_2, eyy_2, exy_2)
                    E_true = E_fit.detach()
                    E_err = E_pred - E_true
                    E_err_norm = torch.linalg.norm(E_err, dim=1, keepdim=True)

                    save_data.update(
                        {
                            "xy_eps": xy_eps_req.detach().cpu().numpy(),
                            "E_pred": E_pred.detach().cpu().numpy(),
                            "E_true": E_true.detach().cpu().numpy(),
                            "E_err": E_err.detach().cpu().numpy(),
                            "E_err_norm": E_err_norm.detach().cpu().numpy(),
                        }
                    )

                np.savez_compressed(
                    config.DISP_STRAIN_SNAPSHOT_DIR / f"disp_strain_epoch_{current_epoch:08d}.npz",
                    **save_data,
                )

        if ep % config.TRAIN_CONFIG["scatter_every"] == 0:
            config.VIZ_SCATTER_DIR.mkdir(parents=True, exist_ok=True)
            for vel_type_for_next, param_dict in config.VIZ_SCATTER_CONFIGS:
                viz_path = config.VIZ_SCATTER_DIR / f"scatter_{vel_type_for_next}_epoch_{current_epoch:08d}.png"
                npz_path = viz_path.with_suffix(".npz")
                plot_residual_scatter(
                    model,
                    kind="pde",
                    nu=nu,
                    xy_u=xy_u,
                    U_fit=U_fit,
                    n=200,
                    bbox=(-1, 1, -1, 1),
                    ellipse=config.ELLIPSE_PARAMS,
                    savepath=str(viz_path),
                    save_npz_path=str(npz_path),
                    cmap="cividis",
                    use_log_norm=False,
                    show_next=True,
                    vel_type_for_next=vel_type_for_next,
                    show=False,
                    **param_dict,
                )

        if ep % config.TRAIN_CONFIG["history_every"] == 0:
            E_1_scaled, E_2_scaled = _get_scaled_E(model)
            logger.log(current_epoch, E_1_scaled, E_2_scaled)
            row = [
                float(current_epoch),
                float(tot.detach().cpu()),
                float(d["data"].detach().cpu()),
                float(d["pde"].detach().cpu()),
                float(d["bc"].detach().cpu()),
                float(d["interface"].detach().cpu()),
                float(d["eik"].detach().cpu()),
                float(d["area"].detach().cpu()),
                float(E_1_scaled.detach().cpu()),
                float(E_2_scaled.detach().cpu()),
            ]
            LOSS_LIST_GLOBAL_ITEM.append(row)
            header = "epoch,total,data,pde,bc,interface,eik,area,E1,E2"
            np.savetxt(
                config.LOSS_CSV_PATH,
                np.asarray(LOSS_LIST_GLOBAL_ITEM, dtype=np.float64),
                delimiter=",",
                header=header,
                comments="",
            )

    logger.close()

    loss_hist_np = np.asarray(loss_hist, dtype=np.float64)
    _ = loss_hist_np

    if len(LOSS_LIST_GLOBAL_ITEM) > 0:
        block = np.array(LOSS_LIST_GLOBAL_ITEM, dtype=np.float64)

        epochs_abs = block[:, 0]
        total_arr = block[:, 1]
        data_arr = block[:, 2]
        pde_arr = block[:, 3]
        bc_arr = block[:, 4]
        if_arr = block[:, 5]
        eik_arr = block[:, 6]
        area_arr = block[:, 7]

        w_data = lam_weights["data"]
        w_pde = lam_weights["pde"]
        w_bc = lam_weights["bc"]
        w_if = lam_weights["interface"]
        w_eik = lam_weights["eik"]
        w_area = lam_weights["area"]

        data_plot = np.sqrt(data_arr / w_data)
        pde_plot = np.sqrt(pde_arr / w_pde)
        bc_plot = np.sqrt(bc_arr / w_bc)
        if_plot = np.sqrt(if_arr / w_if)
        eik_plot = np.sqrt(eik_arr / w_eik)
        area_plot = np.sqrt(area_arr / w_area)

        total_plot = total_arr

        plt.figure(figsize=(6, 4), dpi=200)
        plt.semilogy(epochs_abs, data_plot, label="data")
        plt.semilogy(epochs_abs, pde_plot, label="pde")
        plt.semilogy(epochs_abs, bc_plot, label="bc")
        plt.semilogy(epochs_abs, if_plot, label="interface")
        plt.semilogy(
            epochs_abs,
            total_plot,
            label="total",
            linestyle="--",
            linewidth=1.5,
            color="black",
        )

        plt.xlabel("global epoch")
        plt.ylabel("loss (scaled)")
        plt.title("Loss components (normalized + sqrt), and total")
        plt.grid(True, which="both", linestyle="--", alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.close()

    return model, opt, opt_phi, xy_int_const
