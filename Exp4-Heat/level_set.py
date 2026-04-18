from typing import Iterable, Optional, Tuple

import numpy as np
import torch

from pde import div_kgrad, _grad_norm
from utils import masked_partition_value


def heaviside(phi: torch.Tensor, epsilon: float = 0.005) -> torch.Tensor:
    eps = torch.as_tensor(epsilon, dtype=phi.dtype, device=phi.device)
    return 0.5 * (1 + (2.0 / torch.pi) * torch.atan(phi / eps))


def dirac_smooth(phi: torch.Tensor, epsilon: float = 0.005) -> torch.Tensor:
    eps = torch.as_tensor(epsilon, dtype=phi.dtype, device=phi.device)
    return (1.0 / torch.pi) * (eps / (eps**2 + phi**2))


def sdf_rect_torch(xx, yy, xc, yc, a, b):
    qx = torch.abs(xx - xc) - a
    qy = torch.abs(yy - yc) - b
    qx_pos = torch.maximum(qx, torch.tensor(0.0, device=xx.device))
    qy_pos = torch.maximum(qy, torch.tensor(0.0, device=xx.device))
    epsilon = 1e-12
    outside = torch.sqrt(qx_pos**2 + qy_pos**2 + epsilon)
    inside = torch.maximum(qx, qy)
    return outside + torch.minimum(inside, torch.tensor(0.0, device=xx.device))


def sdf_cross_torch(xx, yy, xc, yc, Lx, Ly, wh, wv):
    phi_h = sdf_rect_torch(xx, yy, xc, yc, a=Lx / 2.0, b=wh / 2.0)
    phi_v = sdf_rect_torch(xx, yy, xc, yc, a=wv / 2.0, b=Ly / 2.0)
    return torch.minimum(phi_h, phi_v)


def local_velocity(
    model,
    xy: torch.Tensor,
    get_f1_f2,
    *,
    band_eps: float = 2e-2,
    h: float = 3e-2,
    eps: float = 1e-8,
    clip_q: float = 0.99,
    tau: float = 1e-6,
) -> torch.Tensor:
    xy = xy.detach().clone().requires_grad_(True)
    phi, T1, T2 = model(xy)
    f1, f2 = get_f1_f2()

    R1 = div_kgrad(T1, f1, xy)
    R2 = div_kgrad(T2, f2, xy)
    r_val = masked_partition_value(phi, R1, R2).detach().abs()

    band_mask = (phi.detach().abs() < band_eps).squeeze()
    if band_mask.sum() == 0:
        return torch.zeros_like(phi)

    xy_band = xy[band_mask]
    phi_band = phi.detach()[band_mask]

    dmat = torch.cdist(xy_band, xy.detach(), p=2)
    knn = dmat < h

    pos_side = (phi.detach() > 0).T
    neg_side = ~pos_side

    r_pos = (knn & pos_side).float() @ r_val / (knn & pos_side).float().sum(
        1, keepdim=True
    ).clamp_min(1)
    r_neg = (knn & neg_side).float() @ r_val / (knn & neg_side).float().sum(
        1, keepdim=True
    ).clamp_min(1)

    delta = r_neg - r_pos
    if delta.numel() == 0:
        return torch.zeros_like(phi)

    scale = torch.quantile(delta.abs(), clip_q) + eps
    vel = torch.tanh(delta / scale)
    vel *= (1 / np.pi) * band_eps / (phi_band**2 + band_eps**2 + eps)

    Vn = torch.zeros_like(phi)
    Vn[band_mask] = vel
    return Vn


@torch.no_grad()
def local_velocity_fit(
    model,
    xy_band_all: torch.Tensor,
    xy_fit: torch.Tensor,
    T_fit: torch.Tensor,
    band_eps: float = 0.02,
    h: float = 0.03,
    tau: float = 1e-4,
    eps: float = 1e-8,
    verbose: bool = False,
) -> torch.Tensor:
    xy_all = xy_band_all.detach().clone().requires_grad_(False)

    phi_f, T1_f, T2_f = model(xy_fit)
    T_pred_f = masked_partition_value(phi_f, T1_f, T2_f)
    r_label = (T_pred_f - T_fit).abs()

    phi_all = model.phi(xy_all)

    band_mask = (phi_all.abs() < band_eps).squeeze()
    xy_band = xy_all[band_mask]
    phi_band = phi_all[band_mask]

    if xy_band.shape[0] == 0:
        if verbose:
            print(
                "[local_velocity_fit] Warning: no band points found, skip step."
            )
        return torch.zeros_like(phi_all)

    dmat = torch.cdist(xy_band, xy_fit, p=2)
    knn = dmat < h

    pos_side_f = (phi_f > 0).T
    neg_side_f = ~pos_side_f

    r_pos = (knn & pos_side_f).float() @ r_label / (knn & pos_side_f).float().sum(
        1, keepdim=True
    ).clamp_min(1)
    r_neg = (knn & neg_side_f).float() @ r_label / (knn & neg_side_f).float().sum(
        1, keepdim=True
    ).clamp_min(1)

    delta = r_neg - r_pos
    delta_mean = delta.abs().mean() + eps
    delta_norm = delta / delta_mean
    vel = delta_norm * (delta_norm.abs() / (delta_norm.abs() + tau))
    vel *= (1 / np.pi) * band_eps / (phi_band.pow(2) + band_eps**2 + eps)

    if vel.numel() == 0:
        if verbose:
            print("[local_velocity_fit] Warning: empty velocity, skip step.")
        return torch.zeros_like(phi_all)

    scale = torch.quantile(torch.abs(vel), 0.95) + eps
    vel_clipped = torch.tanh(vel / scale)

    Vn = torch.zeros_like(phi_all)
    Vn[band_mask] = vel_clipped

    if verbose:
        print(
            f"[local_velocity_fit] band_pts={xy_band.shape[0]}, "
            f"delta_mean={delta_mean.item():.3e}, vel.max={vel.max().item():.3e}"
        )

    return Vn


def local_velocity_grad(
    model,
    xy: torch.Tensor,
    get_f1_f2,
    band_eps: float = 0.02,
    h: float = 0.03,
    tau: float = 1e-4,
    eps: float = 1e-8,
) -> torch.Tensor:
    xy = xy.detach().clone().requires_grad_(True)
    phi, T1, T2 = model(xy)
    w_pos, w_neg = torch.relu(phi), torch.relu(-phi)

    f1, f2 = get_f1_f2()
    R1 = div_kgrad(T1, f1, xy, keep_graph=True)
    R2 = div_kgrad(T2, f2, xy, keep_graph=True)

    gR1 = _grad_norm(R1, xy)
    gR2 = _grad_norm(R2, xy)

    g_val = (w_pos * gR1 + w_neg * gR2).detach()
    phi_det = phi.detach()

    band_mask = (phi_det.abs() < band_eps).squeeze()
    if not band_mask.any():
        return torch.zeros_like(phi_det)

    xy_band = xy[band_mask]
    phi_band = phi_det[band_mask]

    dmat = torch.cdist(xy_band, xy, p=2)
    knn = dmat < h
    pos_side = (phi_det > 0).T
    neg_side = ~pos_side

    g_pos = (knn & pos_side).float() @ g_val / (knn & pos_side).float().sum(
        1, keepdim=True
    ).clamp_min(1)
    g_neg = (knn & neg_side).float() @ g_val / (knn & neg_side).float().sum(
        1, keepdim=True
    ).clamp_min(1)

    delta = g_neg - g_pos
    delta = delta / (delta.abs().mean() + eps)
    vel = delta * (delta.abs() / (delta.abs() + tau))
    vel *= (1 / np.pi) * band_eps / (phi_band.pow(2) + band_eps**2 + eps)

    scale = torch.quantile(vel.abs(), 0.95) + eps
    vel = torch.tanh(vel / scale)

    Vn = torch.zeros_like(phi_det)
    Vn[band_mask] = vel
    return Vn


def local_velocity_CV_full(
    model,
    xy: torch.Tensor,
    get_f1_f2,
    *,
    band_eps: float = 2e-2,
    h: float = 3e-2,
    eps: float = 1e-8,
    tau: float = 1e-8,
    clip_q: float = 0.99,
    fallback_circles: Optional[Iterable[Tuple[float, float, float]]] = None,
):
    xy = xy.detach().clone().requires_grad_(True)
    phi, T1, T2 = model(xy)

    f1, f2 = get_f1_f2()
    R1 = div_kgrad(T1, f1, xy)
    R2 = div_kgrad(T2, f2, xy)
    R_abs = masked_partition_value(phi, R1, R2).abs().detach()

    band = (phi.detach().abs() < band_eps).squeeze()
    Fai_empty = False
    if not band.any() and fallback_circles:
        cx, cy, r0 = list(fallback_circles)[0]
        dist = ((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2).sqrt()
        band = (dist - r0).abs() < band_eps
    if not band.any():
        return torch.zeros_like(phi), True

    R_b = R_abs[band]
    phi_b = phi[band]
    inside_mask = (phi.detach() > 0).squeeze()
    C1 = R_abs[inside_mask].mean() if inside_mask.any() else 0.0
    C2 = R_abs[~inside_mask].mean() if (~inside_mask).any() else 0.0
    CV = (R_b - C1) ** 2 - (R_b - C2) ** 2

    delta = (band_eps / np.pi) / (phi_b**2 + band_eps**2 + eps)
    Vn_b = delta * CV

    s0 = torch.quantile(Vn_b.abs(), clip_q) + eps
    Vn_b = torch.tanh(Vn_b / s0)

    phi_max = phi.abs().max()
    vn_trim = torch.quantile(Vn_b.abs(), clip_q) + tau
    scale = phi_max / vn_trim
    Vn_b = Vn_b * scale * 2.0

    Vn = torch.zeros_like(phi)
    Vn[band] = Vn_b
    return Vn, Fai_empty


def evolve_phi_local(
    model,
    xy: torch.Tensor,
    opt_phi,
    get_f1_f2,
    *,
    xy_fit: Optional[torch.Tensor] = None,
    T_fit: Optional[torch.Tensor] = None,
    dt: float = 1e-2,
    n_inner: int = 10,
    stop_tol: float = 1e-6,
    band_eps: float = 0.05,
    h: float = 0.05,
    tau: float = 1e-3,
    typeVn: str = "PDE_grad",
    plot_interval: int = 5,
    fallback_circles: Optional[Iterable[Tuple[float, float, float]]] = None,
):
    xy = xy.detach().clone().requires_grad_(True)

    Fai_empty = False
    if typeVn == "PDE":
        Vn = local_velocity(model, xy, get_f1_f2, band_eps=band_eps, h=h, tau=tau)
    elif typeVn == "Grad":
        Vn = local_velocity_grad(model, xy, get_f1_f2, band_eps, h, tau)
    elif typeVn == "Data":
        Vn = local_velocity_fit(
            model, xy, xy_fit, T_fit, band_eps=band_eps, h=h, tau=tau
        )
    elif typeVn == "CV":
        Vn, Fai_empty = local_velocity_CV_full(
            model,
            xy,
            get_f1_f2,
            band_eps=band_eps,
            h=h,
            fallback_circles=fallback_circles,
        )
    else:
        raise ValueError(f"Unknown typeVn '{typeVn}'")

    Vn = Vn.detach()

    phi = model.phi(xy)
    if not Fai_empty:
        grad_phi = torch.autograd.grad(
            phi, xy, torch.ones_like(phi), create_graph=True
        )[0]
        norm_g = grad_phi.norm(dim=1, keepdim=True).clamp_min(1e-6)
        phi_tar = phi + dt * Vn * norm_g
    else:
        phi_tar = phi + dt * Vn

    for _ in range(n_inner):
        loss_phi = torch.nn.functional.mse_loss(model.phi(xy), phi_tar.detach())
        opt_phi.zero_grad()
        loss_phi.backward()
        opt_phi.step()
        if loss_phi.item() < stop_tol:
            print(f"Converged (loss < {stop_tol}), stopping.")
            break


def rar_refine(
    xy_int: torch.Tensor,
    model,
    get_f1_f2,
    n_cand: int = 4096,
    n_new: int = 256,
    band_eps: float = 0.02,
    corner_tol: float = 0.05,
    batch_size: int = 8192,
) -> torch.Tensor:
    xy_list = []
    while True:
        xy_batch = torch.rand(batch_size, 2, device=xy_int.device)
        mask = (
            (xy_batch[:, 0] > corner_tol)
            & (xy_batch[:, 0] < 1 - corner_tol)
            & (xy_batch[:, 1] > corner_tol)
            & (xy_batch[:, 1] < 1 - corner_tol)
        )
        xy_valid = xy_batch[mask]
        xy_list.append(xy_valid)
        xy_all = torch.cat(xy_list, dim=0)
        if len(xy_all) >= n_cand:
            break
    xy_cand = xy_all[:n_cand].detach().clone().requires_grad_(True)

    phi_cand, T1, T2 = model(xy_cand)
    f1, f2 = get_f1_f2()
    R1 = div_kgrad(T1, f1, xy_cand).detach()
    R2 = div_kgrad(T2, f2, xy_cand).detach()

    R = masked_partition_value(phi_cand, R1, R2)

    topk_idx = torch.topk(R.abs().flatten(), n_new)[1]
    xy_new = xy_cand[topk_idx].detach()

    xy_int_new = torch.cat([xy_int, xy_new], dim=0)
    return xy_int_new
