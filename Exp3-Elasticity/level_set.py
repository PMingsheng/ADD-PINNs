import numpy as np
import torch
import torch.nn.functional as F

from pde import div_sigma_from_u, lame_from_E


def heaviside(phi, epsilon=0.005):
    return 0.5 * (1 + (2 / torch.pi) * torch.atan(phi / epsilon))


def heaviside_derivative(phi, epsilon=0.1):
    return (1 / (np.pi * epsilon)) * (1 / (1 + (phi / epsilon) ** 2))


def phi_ellipse(xy, *, xc, yc, A, B, Gamma=None, c=None, s=None):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    if c is None or s is None:
        if Gamma is None:
            raise ValueError("phi_ellipse requires Gamma or (c, s).")
        if isinstance(Gamma, (float, int)):
            c = torch.tensor(np.cos(Gamma), dtype=xy.dtype, device=xy.device)
            s = torch.tensor(np.sin(Gamma), dtype=xy.dtype, device=xy.device)
        else:
            c = torch.cos(Gamma)
            s = torch.sin(Gamma)
    dx = x - xc
    dy = y - yc
    xp = c * dx + s * dy
    yp = -s * dx + c * dy
    return (xp / A) ** 2 + (yp / B) ** 2 - 1.0


def get_Es(model, device, dtype):
    if hasattr(model, "get_E_scaled"):
        E_out, E_in = model.get_E_scaled()
        if torch.is_tensor(E_out):
            E_out = E_out.to(device=device, dtype=dtype)
        else:
            E_out = torch.tensor(E_out, device=device, dtype=dtype)
        if torch.is_tensor(E_in):
            E_in = E_in.to(device=device, dtype=dtype)
        else:
            E_in = torch.tensor(E_in, device=device, dtype=dtype)
    elif hasattr(model, "E_1") and hasattr(model, "E_2"):
        E_out = model.E_1 if torch.is_tensor(model.E_1) else torch.tensor(model.E_1, device=device, dtype=dtype)
        E_in = model.E_2 if torch.is_tensor(model.E_2) else torch.tensor(model.E_2, device=device, dtype=dtype)
    elif hasattr(model, "E_out") and hasattr(model, "E_in"):
        E_out = model.E_out if torch.is_tensor(model.E_out) else torch.tensor(model.E_out, device=device, dtype=dtype)
        E_in = model.E_in if torch.is_tensor(model.E_in) else torch.tensor(model.E_in, device=device, dtype=dtype)
    else:
        E_out = torch.as_tensor(1.0, device=device, dtype=dtype)
        E_in = torch.as_tensor(1.0, device=device, dtype=dtype)
    return E_out, E_in


def u_channels_from_model(model, xy):
    phi, ux_out, uy_out, ux_in, uy_in = model(xy)
    U_out = torch.cat([ux_out, uy_out], dim=1)
    U_in = torch.cat([ux_in, uy_in], dim=1)
    return phi, U_out, U_in


def local_velocity(
    model,
    xy: torch.Tensor,
    *,
    band_eps: float = 2e-2,
    h: float = 3e-2,
    eps: float = 1e-8,
    clip_q: float = 0.99,
    tau: float = 1e-6,
    nu: float = 0.30,
) -> torch.Tensor:
    xy = xy.detach().clone().requires_grad_(True)
    phi, U_out, U_in = u_channels_from_model(model, xy)
    device, dtype = xy.device, xy.dtype
    E_out, E_in = get_Es(model, device, dtype)
    lam_out, mu_out = lame_from_E(E_out, nu)
    lam_in, mu_in = lame_from_E(E_in, nu)

    Rx_o, Ry_o = div_sigma_from_u(U_out, xy, lam_out, mu_out, keep_graph=True)
    Rx_i, Ry_i = div_sigma_from_u(U_in, xy, lam_in, mu_in, keep_graph=True)
    R_o = torch.sqrt(Rx_o ** 2 + Ry_o ** 2 + eps)
    R_i = torch.sqrt(Rx_i ** 2 + Ry_i ** 2 + eps)

    w_pos = torch.relu(phi)
    w_neg = torch.relu(-phi)
    r_val = ((w_pos * R_o + w_neg * R_i) / (w_pos + w_neg + eps)).detach()

    band_mask = (phi.detach().abs() < band_eps).squeeze()
    if band_mask.sum() == 0:
        return torch.zeros_like(phi)

    xy_band = xy[band_mask]
    phi_band = phi.detach()[band_mask]

    dmat = torch.cdist(xy_band, xy.detach(), p=2)
    knn = dmat < h
    pos_side = (phi.detach() > 0).T
    neg_side = ~pos_side

    r_pos = (knn & pos_side).float() @ r_val / (
        (knn & pos_side).float().sum(1, keepdim=True).clamp_min(1)
    )
    r_neg = (knn & neg_side).float() @ r_val / (
        (knn & neg_side).float().sum(1, keepdim=True).clamp_min(1)
    )

    delta = r_neg - r_pos
    if delta.numel() == 0:
        return torch.zeros_like(phi)

    scale = torch.quantile(delta.abs(), clip_q) + eps
    vel = torch.tanh(delta / scale)

    vel *= (1 / np.pi) * band_eps / (phi_band ** 2 + band_eps ** 2 + eps)

    Vn = torch.zeros_like(phi)
    Vn[band_mask] = vel
    return Vn


@torch.no_grad()
def local_velocity_fit(
    model,
    xy_band_all,
    xy_fit,
    T_fit,
    band_eps=0.02,
    h=0.03,
    tau=1e-4,
    eps=1e-8,
    verbose=False,
):
    xy_all = xy_band_all.detach().clone().requires_grad_(False)

    phi_f, U_out_f, U_in_f = u_channels_from_model(model, xy_fit)
    w1f = torch.relu(phi_f)
    w2f = torch.relu(-phi_f)
    U_pred_f = (w1f * U_out_f + w2f * U_in_f) / (w1f + w2f + eps)
    r_label = torch.linalg.norm(U_pred_f - T_fit, dim=1, keepdim=True)

    phi_all = model.phi(xy_all)

    band_mask = (phi_all.abs() < band_eps).squeeze()
    if band_mask.sum() == 0:
        if verbose:
            print("[local_velocity_fit] Warning: no band points.")
        return torch.zeros_like(phi_all)

    xy_band = xy_all[band_mask]
    phi_band = phi_all[band_mask]

    dmat = torch.cdist(xy_band, xy_fit, p=2)
    knn = dmat < h
    pos_side_f = (phi_f > 0).T
    neg_side_f = ~pos_side_f

    r_pos = (knn & pos_side_f).float() @ r_label / (
        (knn & pos_side_f).float().sum(1, keepdim=True).clamp_min(1)
    )
    r_neg = (knn & neg_side_f).float() @ r_label / (
        (knn & neg_side_f).float().sum(1, keepdim=True).clamp_min(1)
    )

    delta = r_neg - r_pos
    if delta.numel() == 0:
        return torch.zeros_like(phi_all)

    delta_mean = delta.abs().mean() + eps
    delta_norm = delta / delta_mean
    vel = delta_norm * (delta_norm.abs() / (delta_norm.abs() + tau))
    vel *= (1 / np.pi) * band_eps / (phi_band.pow(2) + band_eps ** 2 + eps)

    scale = torch.quantile(vel.abs(), 0.95) + eps
    vel = torch.tanh(vel / scale)

    Vn = torch.zeros_like(phi_all)
    Vn[band_mask] = vel
    if verbose:
        mx = float(vel.max().abs().cpu())
        print(f"[local_velocity_fit] band={xy_band.shape[0]}, max|vel|={mx:.3e}")
    return Vn


def local_velocity_grad(
    model,
    xy: torch.Tensor,
    band_eps: float = 0.02,
    h: float = 0.03,
    tau: float = 1e-4,
    eps: float = 1e-8,
    nu: float = 0.30,
) -> torch.Tensor:
    xy = xy.detach().clone().requires_grad_(True)
    phi, U_out, U_in = u_channels_from_model(model, xy)
    device, dtype = xy.device, xy.dtype
    E_out, E_in = get_Es(model, device, dtype)
    lam_out, mu_out = lame_from_E(E_out, nu)
    lam_in, mu_in = lame_from_E(E_in, nu)

    Rx_o, Ry_o = div_sigma_from_u(U_out, xy, lam_out, mu_out, keep_graph=True)
    Rx_i, Ry_i = div_sigma_from_u(U_in, xy, lam_in, mu_in, keep_graph=True)

    Ro = torch.sqrt(Rx_o ** 2 + Ry_o ** 2 + eps)
    Ri = torch.sqrt(Rx_i ** 2 + Ry_i ** 2 + eps)

    gRo = torch.autograd.grad(Ro, xy, torch.ones_like(Ro), create_graph=True, retain_graph=True)[0]
    gRi = torch.autograd.grad(Ri, xy, torch.ones_like(Ri), create_graph=True, retain_graph=True)[0]
    gRo = torch.sqrt(gRo[:, 0:1] ** 2 + gRo[:, 1:2] ** 2 + eps)
    gRi = torch.sqrt(gRi[:, 0:1] ** 2 + gRi[:, 1:2] ** 2 + eps)

    g_val = (torch.relu(phi) * gRo + torch.relu(-phi) * gRi).detach()
    phi_det = phi.detach()

    band_mask = (phi_det.abs() < band_eps).squeeze()
    if not band_mask.any():
        return torch.zeros_like(phi_det)

    xy_band = xy[band_mask]
    g_band = g_val[band_mask]
    phi_band = phi_det[band_mask]

    dmat = torch.cdist(xy_band, xy, p=2)
    knn = dmat < h
    pos_side = (phi_det > 0).T
    neg_side = ~pos_side

    g_pos = (knn & pos_side).float() @ g_val / (
        (knn & pos_side).float().sum(1, keepdim=True).clamp_min(1)
    )
    g_neg = (knn & neg_side).float() @ g_val / (
        (knn & neg_side).float().sum(1, keepdim=True).clamp_min(1)
    )

    delta = g_neg - g_pos

    delta = delta / (delta.abs().mean() + eps)
    vel = delta * (delta.abs() / (delta.abs() + tau))
    vel *= (1 / np.pi) * band_eps / (phi_band.pow(2) + band_eps ** 2 + eps)

    scale = torch.quantile(vel.abs(), 0.95) + eps
    vel = torch.tanh(vel / scale)

    Vn = torch.zeros_like(phi_det)
    Vn[band_mask] = vel
    return Vn


def local_velocity_CV_full(
    model,
    xy,
    *,
    band_eps=2e-2,
    h=3e-2,
    eps=1e-8,
    tau=1e-8,
    clip_q=0.99,
    nu: float = 0.30,
):
    xy = xy.detach().clone().requires_grad_(True)
    phi, ux_out, uy_out, ux_in, uy_in = model(xy)
    U_out = torch.cat([ux_out, uy_out], dim=1)
    U_in = torch.cat([ux_in, uy_in], dim=1)

    device, dtype = xy.device, xy.dtype
    E_out, E_in = get_Es(model, device, dtype)
    lam_out, mu_out = lame_from_E(E_out, nu)
    lam_in, mu_in = lame_from_E(E_in, nu)

    Rx_o, Ry_o = div_sigma_from_u(U_out, xy, lam_out, mu_out, keep_graph=True)
    Rx_i, Ry_i = div_sigma_from_u(U_in, xy, lam_in, mu_in, keep_graph=True)
    R_o = torch.sqrt(Rx_o ** 2 + Ry_o ** 2 + eps)
    R_i = torch.sqrt(Rx_i ** 2 + Ry_i ** 2 + eps)

    w1, w2 = torch.relu(phi), torch.relu(-phi)
    R_abs = ((w1 * R_o + w2 * R_i) / (w1 + w2 + 1e-12)).detach()

    band = (phi.detach().abs() < band_eps).squeeze()
    Fai_empty = False
    if not band.any():
        Fai_empty = True
        k = max(64, xy.shape[0] // 50)
        _, idx = torch.topk(-phi.detach().abs().squeeze(), k=k, largest=True)
        band = torch.zeros_like(phi, dtype=torch.bool).squeeze()
        band[idx] = True

    R_b, phi_b = R_abs[band], phi[band]

    inside = (phi.detach() > 0).squeeze()
    C1 = R_abs[inside].mean() if inside.any() else 0.0
    C2 = R_abs[~inside].mean() if (~inside).any() else 0.0
    CV = (R_b - C1) ** 2 - (R_b - C2) ** 2

    dirac = (band_eps / np.pi) / (phi_b ** 2 + band_eps ** 2 + eps)
    Vn_b = dirac * CV

    s0 = torch.quantile(Vn_b.abs(), clip_q) + eps
    Vn_b = torch.tanh(Vn_b / s0)

    phi_max = phi.abs().max().clamp_min(1e-12)
    vn_trim = torch.quantile(Vn_b.abs(), clip_q) + tau
    scale = phi_max / vn_trim
    Vn_b = Vn_b * scale * 2.0

    Vn = torch.zeros_like(phi)
    Vn[band] = Vn_b
    return Vn, Fai_empty


def evolve_phi_local(
    model,
    xy,
    opt_phi,
    dt=1e-2,
    n_inner=10,
    stop_tol=1e-6,
    band_eps=0.05,
    h=0.05,
    tau=1e-3,
    typeVn="PDE_grad",
    xy_fit=None,
    T_fit=None,
    nu: float = 0.30,
):
    xy = xy.detach().clone().requires_grad_(True)

    Fai_empty = False
    typeVn = typeVn.upper()
    if typeVn == "PDE":
        Vn = local_velocity(model, xy, band_eps=band_eps, h=h, tau=tau, nu=nu)
    elif typeVn == "GRAD":
        Vn = local_velocity_grad(model, xy, band_eps, h, tau, nu=nu)
    elif typeVn == "DATA":
        if xy_fit is None or T_fit is None:
            raise ValueError("typeVn='DATA' requires xy_fit and T_fit")
        Vn = local_velocity_fit(model, xy, xy_fit, T_fit, band_eps=band_eps, h=h, tau=tau)
    elif typeVn == "CV":
        Vn, Fai_empty = local_velocity_CV_full(model, xy, band_eps=band_eps, h=h, tau=tau, nu=nu)
    else:
        raise ValueError(f"Unknown typeVn '{typeVn}'")

    Vn = Vn.detach()

    phi = model.phi(xy)
    if not Fai_empty:
        grad_phi = torch.autograd.grad(phi, xy, torch.ones_like(phi), create_graph=True)[0]
        norm_g = grad_phi.norm(dim=1, keepdim=True).clamp_min(1e-6)
        phi_tar = phi + dt * Vn * norm_g
    else:
        phi_tar = phi + dt * Vn

    for _ in range(n_inner):
        loss_phi = F.mse_loss(model.phi(xy), phi_tar.detach())
        opt_phi.zero_grad()
        loss_phi.backward()
        opt_phi.step()
        if loss_phi.item() < stop_tol:
            break


def rar_refine(
    xy_int,
    model,
    f1_raw=None,
    f2_raw=None,
    n_cand=4096,
    n_new=256,
    band_eps=0.02,
    corner_tol=0.05,
    batch_size=8192,
    nu: float = 0.30,
):
    device = xy_int.device
    xy_list = []
    while True:
        xy_batch = torch.rand(batch_size, 2, device=device)
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

    phi_cand, ux_out, uy_out, ux_in, uy_in = model(xy_cand)
    U_out = torch.cat([ux_out, uy_out], dim=1)
    U_in = torch.cat([ux_in, uy_in], dim=1)

    E_out, E_in = get_Es(model, device=xy_cand.device, dtype=xy_cand.dtype)
    lam_out, mu_out = lame_from_E(E_out, nu)
    lam_in, mu_in = lame_from_E(E_in, nu)

    Rx_o, Ry_o = div_sigma_from_u(U_out, xy_cand, lam_out, mu_out, keep_graph=True)
    Rx_i, Ry_i = div_sigma_from_u(U_in, xy_cand, lam_in, mu_in, keep_graph=True)

    R_o = torch.sqrt(Rx_o ** 2 + Ry_o ** 2 + 1e-12)
    R_i = torch.sqrt(Rx_i ** 2 + Ry_i ** 2 + 1e-12)
    w1, w2 = torch.relu(phi_cand), torch.relu(-phi_cand)
    R = (w1 * R_o + w2 * R_i) / (w1 + w2 + 1e-12)

    topk_idx = torch.topk(R.flatten().abs(), n_new)[1]
    xy_new = xy_cand[topk_idx].detach()

    xy_int_new = torch.cat([xy_int, xy_new], dim=0)
    return xy_int_new
