from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from problem_3d import (
    ALPHA_INSIDE,
    ALPHA_OUTSIDE,
    BETA_INSIDE,
    beta_outside,
    grad_beta_outside,
    source_region_inside,
    source_region_outside,
)


HEAVISIDE_BANDWIDTH: float = 0.05
DIRAC_BANDWIDTH: float = 0.05


def heaviside(phi: torch.Tensor, epsilon: float = HEAVISIDE_BANDWIDTH) -> torch.Tensor:
    eps = torch.as_tensor(epsilon, dtype=phi.dtype, device=phi.device)
    h = torch.zeros_like(phi)
    h = torch.where(phi > eps, torch.ones_like(h), h)
    mask = phi.abs() <= eps
    if mask.any():
        phi_m = phi[mask]
        h_m = 0.5 * (1.0 + phi_m / eps + (1.0 / torch.pi) * torch.sin(torch.pi * phi_m / eps))
        h = h.clone()
        h[mask] = h_m
    return h


def dirac_smooth(phi: torch.Tensor, epsilon: float = DIRAC_BANDWIDTH) -> torch.Tensor:
    eps = torch.as_tensor(epsilon, dtype=phi.dtype, device=phi.device)
    inside = 0.5 / eps * (1.0 + torch.cos(torch.pi * phi / eps))
    return torch.where(phi.abs() <= eps, inside, torch.zeros_like(phi))


def _grad_lap(u: torch.Tensor, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=xyz,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2x = torch.autograd.grad(
        outputs=grad_u[:, 0:1],
        inputs=xyz,
        grad_outputs=torch.ones_like(grad_u[:, 0:1]),
        create_graph=False,
        retain_graph=True,
    )[0][:, 0:1]
    d2y = torch.autograd.grad(
        outputs=grad_u[:, 1:2],
        inputs=xyz,
        grad_outputs=torch.ones_like(grad_u[:, 1:2]),
        create_graph=False,
        retain_graph=True,
    )[0][:, 1:2]
    d2z = torch.autograd.grad(
        outputs=grad_u[:, 2:3],
        inputs=xyz,
        grad_outputs=torch.ones_like(grad_u[:, 2:3]),
        create_graph=False,
        retain_graph=True,
    )[0][:, 2:3]
    return grad_u, d2x + d2y + d2z


def _branch_residuals(
    phi: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor,
    xyz: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = 1e-8
    grad_u1, lap_u1 = _grad_lap(u1, xyz)
    grad_u2, lap_u2 = _grad_lap(u2, xyz)

    f1 = source_region_inside(xyz).detach()
    f2 = source_region_outside(xyz).detach()

    r1 = -BETA_INSIDE * lap_u1 + ALPHA_INSIDE * u1 - f1
    beta2 = beta_outside(xyz)
    gbeta2 = grad_beta_outside(xyz)
    div_beta_grad_u2 = beta2 * lap_u2 + (gbeta2 * grad_u2).sum(dim=1, keepdim=True)
    r2 = -div_beta_grad_u2 + ALPHA_OUTSIDE * u2 - f2

    w1 = torch.relu(phi)
    w2 = torch.relu(-phi)
    r_mix = (w1 * r1 + w2 * r2) / (w1 + w2 + eps)
    return r1, r2, r_mix


def _grad_norm(field: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    grad_f = torch.autograd.grad(
        outputs=field,
        inputs=xyz,
        grad_outputs=torch.ones_like(field),
        create_graph=False,
        retain_graph=True,
    )[0]
    return grad_f.norm(dim=1, keepdim=True)


def local_velocity_pde(
    model,
    xyz: torch.Tensor,
    *,
    band_eps: float = 0.02,
    h: float = 0.08,
    clip_q: float = 0.99,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    xyz_req = xyz.detach().clone().requires_grad_(True)
    phi, u1, u2 = model(xyz_req)

    _, _, r_mix = _branch_residuals(phi, u1, u2, xyz_req)
    r_metric = r_mix.detach().abs()
    phi_det = phi.detach()

    band_mask = (phi_det.abs() < band_eps).squeeze(-1)
    if not torch.any(band_mask):
        return {
            "vn": torch.zeros_like(phi_det),
            "band_count": torch.as_tensor(0, device=phi_det.device),
            "r_mean": torch.as_tensor(0.0, device=phi_det.device),
            "r_max": torch.as_tensor(0.0, device=phi_det.device),
        }

    xyz_band = xyz_req[band_mask]
    phi_band = phi_det[band_mask]

    dmat = torch.cdist(xyz_band.detach(), xyz_req.detach(), p=2)
    knn = dmat < h

    pos_side = (phi_det > 0).T
    neg_side = ~pos_side
    r_pos = (knn & pos_side).float() @ r_metric / (knn & pos_side).float().sum(1, keepdim=True).clamp_min(1.0)
    r_neg = (knn & neg_side).float() @ r_metric / (knn & neg_side).float().sum(1, keepdim=True).clamp_min(1.0)

    delta = r_neg - r_pos
    scale = torch.quantile(delta.abs(), clip_q) + eps
    vel = torch.tanh(delta / scale)
    vel *= dirac_smooth(phi_band)

    vn = torch.zeros_like(phi_det)
    vn[band_mask] = vel
    return {
        "vn": vn,
        "band_count": band_mask.sum(),
        "r_mean": r_metric.mean(),
        "r_max": r_metric.max(),
    }


def local_velocity_grad(
    model,
    xyz: torch.Tensor,
    *,
    band_eps: float = 0.02,
    h: float = 0.08,
    tau: float = 1e-3,
    clip_q: float = 0.99,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    xyz_req = xyz.detach().clone().requires_grad_(True)
    phi, u1, u2 = model(xyz_req)
    r1, r2, _ = _branch_residuals(phi, u1, u2, xyz_req)

    g_r1 = _grad_norm(r1, xyz_req)
    g_r2 = _grad_norm(r2, xyz_req)

    w1 = torch.relu(phi)
    w2 = torch.relu(-phi)
    g_metric = ((w1 * g_r1 + w2 * g_r2) / (w1 + w2 + eps)).detach()
    phi_det = phi.detach()

    band_mask = (phi_det.abs() < band_eps).squeeze(-1)
    if not torch.any(band_mask):
        return {
            "vn": torch.zeros_like(phi_det),
            "band_count": torch.as_tensor(0, device=phi_det.device),
            "r_mean": torch.as_tensor(0.0, device=phi_det.device),
            "r_max": torch.as_tensor(0.0, device=phi_det.device),
        }

    xyz_band = xyz_req[band_mask]
    phi_band = phi_det[band_mask]

    dmat = torch.cdist(xyz_band.detach(), xyz_req.detach(), p=2)
    knn = dmat < h

    pos_side = (phi_det > 0).T
    neg_side = ~pos_side
    g_pos = (knn & pos_side).float() @ g_metric / (knn & pos_side).float().sum(1, keepdim=True).clamp_min(1.0)
    g_neg = (knn & neg_side).float() @ g_metric / (knn & neg_side).float().sum(1, keepdim=True).clamp_min(1.0)

    delta = g_neg - g_pos
    delta = delta / (delta.abs().mean() + eps)
    vel = delta * (delta.abs() / (delta.abs() + tau))
    vel *= dirac_smooth(phi_band)
    scale = torch.quantile(vel.abs(), clip_q) + eps
    vel = torch.tanh(vel / scale)

    vn = torch.zeros_like(phi_det)
    vn[band_mask] = vel
    return {
        "vn": vn,
        "band_count": band_mask.sum(),
        "r_mean": g_metric.mean(),
        "r_max": g_metric.max(),
    }


@torch.no_grad()
def local_velocity_data(
    model,
    xyz: torch.Tensor,
    xyz_fit: torch.Tensor,
    u_fit: torch.Tensor,
    *,
    band_eps: float = 0.02,
    h: float = 0.08,
    tau: float = 1e-3,
    clip_q: float = 0.99,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    phi_fit, u1_fit, u2_fit = model(xyz_fit)
    w1_fit = torch.relu(phi_fit)
    w2_fit = torch.relu(-phi_fit)
    u_pred_fit = (w1_fit * u1_fit + w2_fit * u2_fit) / (w1_fit + w2_fit + eps)
    r_label = (u_pred_fit - u_fit).abs()

    phi_all = model.phi(xyz).detach()
    band_mask = (phi_all.abs() < band_eps).squeeze(-1)
    if not torch.any(band_mask):
        return {
            "vn": torch.zeros_like(phi_all),
            "band_count": torch.as_tensor(0, device=phi_all.device),
            "r_mean": torch.as_tensor(0.0, device=phi_all.device),
            "r_max": torch.as_tensor(0.0, device=phi_all.device),
        }

    xyz_band = xyz[band_mask]
    phi_band = phi_all[band_mask]

    dmat = torch.cdist(xyz_band, xyz_fit, p=2)
    knn = dmat < h

    pos_side_fit = (phi_fit.detach() > 0).T
    neg_side_fit = ~pos_side_fit

    r_pos = (knn & pos_side_fit).float() @ r_label / (knn & pos_side_fit).float().sum(1, keepdim=True).clamp_min(1.0)
    r_neg = (knn & neg_side_fit).float() @ r_label / (knn & neg_side_fit).float().sum(1, keepdim=True).clamp_min(1.0)

    delta = r_neg - r_pos
    delta_norm = delta / (delta.abs().mean() + eps)
    vel = delta_norm * (delta_norm.abs() / (delta_norm.abs() + tau))
    vel *= dirac_smooth(phi_band)
    scale = torch.quantile(vel.abs(), clip_q) + eps
    vel = torch.tanh(vel / scale)

    vn = torch.zeros_like(phi_all)
    vn[band_mask] = vel
    return {
        "vn": vn,
        "band_count": band_mask.sum(),
        "r_mean": r_label.mean(),
        "r_max": r_label.max(),
    }


def evolve_phi_by_residual(
    model,
    xyz: torch.Tensor,
    opt_phi,
    *,
    residual_type: str = "PDE",
    xyz_fit: Optional[torch.Tensor] = None,
    u_fit: Optional[torch.Tensor] = None,
    dt: float = 1e-3,
    n_inner: int = 10,
    stop_tol: float = 1e-6,
    band_eps: float = 0.02,
    h: float = 0.08,
    tau: float = 1e-3,
    clip_q: float = 0.99,
) -> Dict[str, float]:
    kind = residual_type.strip().upper()
    # Keep a flower-style alias in visualization/experiments.
    if kind == "CV":
        kind = "DATA"
    if kind == "PDE":
        v_info = local_velocity_pde(
            model,
            xyz,
            band_eps=band_eps,
            h=h,
            clip_q=clip_q,
        )
    elif kind == "GRAD":
        v_info = local_velocity_grad(
            model,
            xyz,
            band_eps=band_eps,
            h=h,
            tau=tau,
            clip_q=clip_q,
        )
    elif kind == "DATA":
        if xyz_fit is None or u_fit is None:
            raise ValueError("DATA residual_type requires xyz_fit and u_fit.")
        v_info = local_velocity_data(
            model,
            xyz,
            xyz_fit,
            u_fit,
            band_eps=band_eps,
            h=h,
            tau=tau,
            clip_q=clip_q,
        )
    else:
        raise ValueError("Unknown residual_type. Use one of: PDE, GRAD, DATA, CV.")

    vn = v_info["vn"].detach()
    band_count = int(v_info["band_count"].item())

    xyz_req = xyz.detach().clone().requires_grad_(True)
    phi_now = model.phi(xyz_req)
    grad_phi = torch.autograd.grad(
        outputs=phi_now,
        inputs=xyz_req,
        grad_outputs=torch.ones_like(phi_now),
        create_graph=False,
        retain_graph=False,
    )[0]
    norm_g = grad_phi.norm(dim=1, keepdim=True).clamp_min(1e-6)
    phi_target = (phi_now + dt * vn * norm_g).detach()

    loss_phi_val = 0.0
    for _ in range(max(int(n_inner), 1)):
        pred = model.phi(xyz)
        loss_phi = torch.nn.functional.mse_loss(pred, phi_target)
        opt_phi.zero_grad()
        loss_phi.backward()
        opt_phi.step()
        loss_phi_val = float(loss_phi.item())
        if loss_phi_val < stop_tol:
            break

    return {
        "band_count": float(band_count),
        "r_mean": float(v_info["r_mean"].item()),
        "r_max": float(v_info["r_max"].item()),
        "vn_max": float(vn.abs().max().item()),
        "phi_fit_loss": float(loss_phi_val),
        "type": kind,
    }


def predict_phi_next_by_residual(
    model,
    xyz: torch.Tensor,
    *,
    residual_type: str = "PDE",
    xyz_fit: Optional[torch.Tensor] = None,
    u_fit: Optional[torch.Tensor] = None,
    dt: float = 1e-3,
    band_eps: float = 0.02,
    h: float = 0.08,
    tau: float = 1e-3,
    clip_q: float = 0.99,
) -> Dict[str, torch.Tensor]:
    kind = residual_type.strip().upper()
    # Keep a flower-style alias in visualization/experiments.
    if kind == "CV":
        kind = "DATA"
    if kind == "PDE":
        v_info = local_velocity_pde(
            model,
            xyz,
            band_eps=band_eps,
            h=h,
            clip_q=clip_q,
        )
    elif kind == "GRAD":
        v_info = local_velocity_grad(
            model,
            xyz,
            band_eps=band_eps,
            h=h,
            tau=tau,
            clip_q=clip_q,
        )
    elif kind == "DATA":
        if xyz_fit is None or u_fit is None:
            raise ValueError("DATA residual_type requires xyz_fit and u_fit.")
        v_info = local_velocity_data(
            model,
            xyz,
            xyz_fit,
            u_fit,
            band_eps=band_eps,
            h=h,
            tau=tau,
            clip_q=clip_q,
        )
    else:
        raise ValueError("Unknown residual_type. Use one of: PDE, GRAD, DATA, CV.")

    xyz_req = xyz.detach().clone().requires_grad_(True)
    phi_now = model.phi(xyz_req)
    grad_phi = torch.autograd.grad(
        outputs=phi_now,
        inputs=xyz_req,
        grad_outputs=torch.ones_like(phi_now),
        create_graph=False,
        retain_graph=False,
    )[0]
    norm_g = grad_phi.norm(dim=1, keepdim=True).clamp_min(1e-6)
    vn = v_info["vn"].detach()
    phi_next = (phi_now + dt * vn * norm_g).detach()

    return {
        "type": kind,
        "phi_now": phi_now.detach(),
        "phi_next": phi_next,
        "vn": vn,
        "band_count": v_info["band_count"],
        "r_mean": v_info["r_mean"],
        "r_max": v_info["r_max"],
    }
