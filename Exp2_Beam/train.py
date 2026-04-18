"""
Training functions for LS-PINN Beam project.
"""
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple, Any

from config import (
    DEVICE, LEARNING_RATE, LOSS_WEIGHTS,
    RAR_ENABLED, RAR_INTERVAL, PLOT_INTERVAL, EI_REAL,
    SAVE_INTERVAL_EARLY, SAVE_INTERVAL_LATE, SAVE_EPOCH_CUTOFF,
)
from loss import compute_loss
from pde import euler_beam_pde
from visualization import plot_loss_curves


def rar_refine(
    x_int: torch.Tensor,
    model: nn.Module,
    *,
    n_cand: int = 4096,
    n_new: int = 256,
    band_eps: float = 0.02,
    corner_tol: float = 0.05,
    batch_size: int = 8192,
) -> torch.Tensor:
    """
    Residual-based Adaptive Refinement (RAR).
    
    Select n_new highest residual points from candidates to add to training set.
    
    Args:
        x_int: Current training points
        model: Neural network model
        n_cand: Number of candidate points
        n_new: Number of new points to add
        band_eps: Narrow band width
        corner_tol: Corner exclusion tolerance
        batch_size: Batch size for candidate generation
        
    Returns:
        Updated training points
    """
    device = x_int.device
    
    # Generate candidate points
    x_list = []
    while True:
        xb = torch.rand(batch_size, 1, device=device)
        m = (xb[:, 0] > corner_tol) & (xb[:, 0] < 1 - corner_tol)
        x_list.append(xb[m])
        x_all = torch.cat(x_list, 0)
        if x_all.shape[0] >= n_cand:
            break
    x_cand = x_all[:n_cand].detach().clone().requires_grad_(True)

    # Forward pass
    (phi, u1_NN, u2_NN, fai1_NN, fai2_NN, Dfai1_NN, Dfai2_NN,
     M1_NN, M2_NN, V1_NN, V2_NN, EI1, EI2) = model(x_cand)

    # PDE residuals
    Rf1, Rdtheta1, RM1, RV1, RQ1 = euler_beam_pde(x_cand, u1_NN, fai1_NN, Dfai1_NN, M1_NN, V1_NN, EI1)
    Rf2, Rdtheta2, RM2, RV2, RQ2 = euler_beam_pde(x_cand, u2_NN, fai2_NN, Dfai2_NN, M2_NN, V2_NN, EI2)

    w1, w2 = torch.relu(phi), torch.relu(-phi)
    R_comb = (
        w1.detach() * (RQ1**2 + RV1**2)
        + w2.detach() * (RQ2**2 + RV2**2)
    ) / (w1.detach() + w2.detach() + 1e-12)
    R_comb = R_comb.detach().flatten()

    # Select top-k points
    topk_idx = torch.topk(R_comb.abs(), n_new)[1]
    x_new = x_cand[topk_idx].detach()

    return torch.cat([x_int, x_new], 0)


def _should_save_snapshot(
    current_epoch: int,
    cutoff: int,
    interval_early: int,
    interval_late: int,
) -> bool:
    if current_epoch <= cutoff:
        return interval_early > 0 and current_epoch % interval_early == 0
    return interval_late > 0 and current_epoch % interval_late == 0


def train_main(
    model: nn.Module,
    epochs: int,
    xy_fit: Optional[torch.Tensor] = None,
    label_fit: Optional[torch.Tensor] = None,
    label_fit_disp: Optional[torch.Tensor] = None,
    xy_int_const: Optional[torch.Tensor] = None,
    opt: Optional[torch.optim.Optimizer] = None,
    opt_phi: Optional[torch.optim.Optimizer] = None,
    lr: float = LEARNING_RATE,
    RAR: bool = RAR_ENABLED,
    loss_save_path: str = "loss_list_global.csv",
    lam: Optional[Dict[str, float]] = None,
    epoch_offset_global: int = 0,
    history: Optional[Dict[str, list]] = None,
    live_phi: bool = False,
    phi_live: Optional[Dict[str, Any]] = None,
    phi_plot_interval: Optional[int] = None,
    phi_plot_points: int = 1024,
    live_panels: bool = False,
    panels_live: Optional[Dict[str, Any]] = None,
    panels_out_dir: str = "beam_bechmark_viz",
    panels_label_path: Optional[str] = None,
    panels_EI_real: float = EI_REAL,
    save_snapshots: bool = False,
    snapshot_interval_early: int = SAVE_INTERVAL_EARLY,
    snapshot_interval_late: int = SAVE_INTERVAL_LATE,
    snapshot_epoch_cutoff: int = SAVE_EPOCH_CUTOFF,
    snapshot_plot_panels: bool = False,
) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.Optimizer, torch.Tensor, int, Dict[str, list]]:
    """
    Main training loop.
    
    Args:
        model: Neural network model
        epochs: Number of training epochs
        xy_fit: Data fitting points
        label_fit: Data labels
        label_fit_disp: Displacement labels at xy_fit (optional)
        xy_int_const: Collocation points
        opt: Model optimizer
        opt_phi: Phi network optimizer
        lr: Learning rate
        RAR: Enable residual adaptive refinement
        loss_save_path: Path to save loss history
        lam: Dictionary of loss weights. If None, uses default from config.
        live_phi: Enable live phi plot during training.
        phi_live: Mutable dict to reuse the same plot across calls.
        phi_plot_interval: Plot update interval (defaults to PLOT_INTERVAL).
        phi_plot_points: Number of points for phi visualization.
        live_panels: Enable live panels plot during training.
        panels_live: Mutable dict to reuse the same panels plot across calls.
        panels_out_dir: Output directory for panels snapshot.
        panels_label_path: Optional label path for panels overlay.
        panels_EI_real: Reference EI value for panels scaling.
        save_snapshots: Enable periodic snapshot saving.
        snapshot_interval_early: Snapshot interval before cutoff (epochs).
        snapshot_interval_late: Snapshot interval after cutoff (epochs).
        snapshot_epoch_cutoff: Cutoff epoch for switching intervals.
        snapshot_plot_panels: Also save panels for each snapshot.
        
    Returns:
        model: Trained model
        opt: Optimizer
        opt_phi: Phi optimizer
        xy_int_const: Final collocation points
        epoch_offset_global: Updated global epoch offset
        history: Updated history dict
    """
    device = next(model.parameters()).device

    # Initialize tracking lists
    if history is None:
        history = {
            "loss_list_global": [],
            "loss_list_global_item": [],
            "epoch_list_global": [],
        }
    loss_list_global = history["loss_list_global"]
    loss_list_global_item = history["loss_list_global_item"]
    epoch_list_global = history["epoch_list_global"]

    # Use default weights if not provided
    if lam is None:
        lam = LOSS_WEIGHTS

    # Initialize optimizers
    params = list(model.parameters())
        
    if opt is None:
        opt = torch.optim.Adam(params, lr=lr)

    if opt_phi is None:
        opt_phi = torch.optim.Adam(model.phi.parameters(), lr)

    plt = None
    phi_plot = None
    panels_plot = None
    save_beam_phi_snapshot = None
    if live_phi:
        import matplotlib.pyplot as plt  # local import to avoid backend issues

        if phi_live is None:
            phi_live = {}
        if phi_plot_interval is None:
            phi_plot_interval = PLOT_INTERVAL

        if phi_live.get("line") is None:
            plt.ion()
            x_vis = torch.linspace(0, 1, phi_plot_points, device=device).unsqueeze(1)
            x_np = x_vis.detach().cpu().squeeze().numpy()
            fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
            phi_live["closed"] = False
            fig.canvas.mpl_connect("close_event", lambda _evt: phi_live.update({"closed": True}))
            was_training = model.training
            model.eval()
            with torch.no_grad():
                phi_np = model(x_vis)[0].detach().cpu().squeeze().numpy()
            if was_training:
                model.train()
            (line,) = ax.plot(x_np, phi_np, "b-", lw=0.9)
            ax.axhline(0.0, color="k", ls=":", lw=0.8)
            ax.set_xlabel("x")
            ax.set_ylabel("phi")
            ax.set_title("phi evolution")
            ax.grid(True, ls="--", alpha=0.3)
            fig.canvas.draw()
            fig.canvas.flush_events()
            phi_live.update(
                {"fig": fig, "ax": ax, "line": line, "x_vis": x_vis, "x_np": x_np}
            )
        else:
            x_vis = phi_live.get("x_vis")
            if x_vis is None or x_vis.device != device:
                x_vis = torch.linspace(0, 1, phi_plot_points, device=device).unsqueeze(1)
                phi_live["x_vis"] = x_vis
                phi_live["x_np"] = x_vis.detach().cpu().squeeze().numpy()

        phi_plot = phi_live

        if live_panels:
            if panels_live is None:
                panels_live = {}
            if panels_live.get("im") is None:
                from visualization import save_beam_phi_snapshot  # local import

                plt.ion()
                panels_live["closed"] = False
                fig_p, ax_p = plt.subplots(figsize=(8, 6), dpi=150)
                fig_p.canvas.mpl_connect("close_event", lambda _evt: panels_live.update({"closed": True}))

                save_beam_phi_snapshot(
                    model,
                    label_path=panels_label_path,
                    out_dir=panels_out_dir,
                    suffix="live",
                    plot_panels=True,
                    EI_real=panels_EI_real,
                )
                panel_path = os.path.join(panels_out_dir, "panels_live.png")
                img = plt.imread(panel_path)
                im = ax_p.imshow(img)
                ax_p.axis("off")
                ax_p.set_title("panels evolution")
                fig_p.canvas.draw()
                fig_p.canvas.flush_events()
                panels_live.update(
                    {"fig": fig_p, "ax": ax_p, "im": im, "panel_path": panel_path}
                )
            panels_plot = panels_live

    loss_terms_hist = []
    if save_snapshots or epochs == 0:
        if save_beam_phi_snapshot is None:
            from visualization import save_beam_phi_snapshot  # local import
        save_beam_phi_snapshot(
            model,
            label_path=panels_label_path,
            out_dir=panels_out_dir,
            suffix=epoch_offset_global,
            plot_panels=snapshot_plot_panels if save_snapshots else False,
            EI_real=panels_EI_real,
        )

    for ep in range(1, epochs + 1):
        current_epoch = epoch_offset_global + ep
        # Compute total loss
        tot_loss, d, core_loss = compute_loss(
            model,
            x_int=xy_int_const,
            x_fit=xy_fit,
            strain_fit=label_fit,
            u_fit_disp=label_fit_disp,
            lam=lam,
            xy_int_const=xy_int_const,
        )

        # Update network
        opt.zero_grad()
        tot_loss.backward()
        opt.step()

        current_terms = {'total': tot_loss.item()}
        current_terms.update({k: v.item() for k, v in d.items()})
        loss_terms_hist.append(current_terms)

        if save_snapshots and _should_save_snapshot(
            current_epoch,
            snapshot_epoch_cutoff,
            snapshot_interval_early,
            snapshot_interval_late,
        ):
            if save_beam_phi_snapshot is None:
                from visualization import save_beam_phi_snapshot  # local import

            save_beam_phi_snapshot(
                model,
                label_path=panels_label_path,
                out_dir=panels_out_dir,
                suffix=current_epoch,
                plot_panels=snapshot_plot_panels,
                EI_real=panels_EI_real,
            )

        # RAR refinement
        if RAR and (epoch_offset_global + ep) % RAR_INTERVAL == 0:
            xy_int_const = rar_refine(
                xy_int_const, model,
                n_cand=4096, n_new=121
            )
            print(f"[RAR] epoch {ep:>6d} | new training points -> {len(xy_int_const):,}")

        # Logging
        if ep % PLOT_INTERVAL == 0:
            loss_list_global.append(tot_loss.item())
            epoch_list_global.append(current_epoch)

            loss_row = [
                current_epoch,
                tot_loss.item(),
                d['data'].item(),
                d['data_strain'].item(),
                d['data_disp'].item(),
                d['fai'].item(),
                d['dfai'].item(),
                d['weight'].item(),
                d['M'].item(),
                d['V'].item(),
                d['Q'].item(),
                d.get('dEI', torch.tensor(0.0, device=device)).item(),
                d['interface'].item(),
                d['eik'].item(),
                d['area'].item(),
            ]
            loss_list_global_item.append(loss_row)

            if loss_save_path:
                header = 'epoch,total,data,data_strain,data_disp,fai,dfai,weight,M,V,Q,dEI,interface,eik,area'
                np.savetxt(
                    loss_save_path,
                    np.asarray(loss_list_global_item, dtype=float),
                    delimiter=',',
                    header=header,
                    comments='',
                )

            print(
                f"E{current_epoch:>5d}  total={tot_loss.item():.3e}  "
                f"strain={d['data_strain'].item() * lam.get('data_strain', lam.get('data', 0)):.1e}  "
                f"disp={d['data_disp'].item() * lam.get('data_disp', lam.get('data', 0)):.1e}  "
                f"data={d['data'].item():.1e}  "
                f"fai={d['fai'].item() * lam['fai']:.1e}  "
                f"dfai={d['dfai'].item() * lam['dfai']:.1e}  "
                f"weight={d['weight'].item() * lam['weight']:.1e}  "
                f"M={d['M'].item() * lam['M']:.1e}  "
                f"V={d['V'].item() * lam['V']:.1e}  "
                f"Q={d['Q'].item() * lam['Q']:.1e}  "
                f"dEI={d['dEI'].item() * lam['dEI']:.1e}  "
                f"if={d['interface'].item() * lam['interface']:.1e}  "
                f"eik={d['eik'].item() * lam['eik']:.1e}  "
                f"area={d['area'].item() * lam['area']:.1e}"
            )

            if live_phi and phi_plot_interval and (current_epoch % phi_plot_interval == 0):
                if phi_plot.get("closed"):
                    continue
                x_vis = phi_plot["x_vis"]
                x_np = phi_plot["x_np"]
                ax = phi_plot["ax"]
                fig = phi_plot["fig"]
                line = phi_plot["line"]
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    phi_np = model(x_vis)[0].detach().cpu().squeeze().numpy()
                if was_training:
                    model.train()
                try:
                    line.set_xdata(x_np)
                    line.set_ydata(phi_np)
                    ax.relim()
                    ax.autoscale_view()
                    ax.set_title(f"phi @ epoch {current_epoch}")
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    if plt is not None:
                        plt.pause(0.001)
                except Exception:
                    phi_plot["closed"] = True

                if live_panels and panels_plot and not panels_plot.get("closed"):
                    if save_beam_phi_snapshot is None:
                        from visualization import save_beam_phi_snapshot  # local import

                    save_beam_phi_snapshot(
                        model,
                        label_path=panels_label_path,
                        out_dir=panels_out_dir,
                        suffix="live",
                        plot_panels=True,
                        EI_real=panels_EI_real,
                    )
                    try:
                        img = plt.imread(panels_plot["panel_path"])
                        panels_plot["im"].set_data(img)
                        panels_plot["ax"].set_title(f"panels @ epoch {current_epoch}")
                        panels_plot["fig"].canvas.draw_idle()
                        panels_plot["fig"].canvas.flush_events()
                        if plt is not None:
                            plt.pause(0.001)
                    except Exception:
                        panels_plot["closed"] = True

    epoch_offset_global += epochs

    # Plot global loss curves
    if loss_list_global_item:
        keys = [
            "total", "data", "data_strain", "data_disp", "fai", "dfai", "weight",
            "M", "V", "Q", "dEI", "interface", "eik", "area",
        ]
        loss_terms_global = [
            dict(zip(keys, row[1:])) for row in loss_list_global_item
        ]
        plot_loss_curves(loss_terms_global, epochs_axis=epoch_list_global, show=False)
    else:
        plot_loss_curves(loss_terms_hist, show=False)

    return model, opt, opt_phi, xy_int_const, epoch_offset_global, history
