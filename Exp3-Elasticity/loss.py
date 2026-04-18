import torch

from level_set import heaviside
from pde import lame_from_E, strain_from_u, stress_from_u, div_sigma_from_u


def sample_boundary(n, device):
    t = torch.rand(n, 1, device=device) * 2 - 1
    top = torch.cat([t, torch.ones_like(t)], dim=1)
    bot = torch.cat([t, -torch.ones_like(t)], dim=1)
    left = torch.cat([-torch.ones_like(t), t], dim=1)
    right = torch.cat([torch.ones_like(t), t], dim=1)
    return top, bot, left, right


def bc_loss(model, n=256, eps0=3.0e-3, nu=0.30, eps=1e-12):
    dev = next(model.parameters()).device
    p_top, p_bot, p_left, p_right = sample_boundary(n, dev)

    def _target_u(points_phys):
        x = points_phys[:, 0:1]
        y = points_phys[:, 1:2]
        ux = eps0 * x
        uy = -nu * eps0 * y
        return ux, uy

    def _edge_loss(points_phys, comp):
        points_model = points_phys.to(dev)
        phi, ux_out, uy_out, ux_in, uy_in = model(points_model)
        w1 = torch.relu(phi)
        w2 = torch.relu(-phi)
        denom = (w1 + w2 + eps)
        if comp == "ux":
            u_tg, _ = _target_u(points_phys)
            u_pred = (w1.detach() * ux_out + w2.detach() * ux_in) / denom.detach()
        else:
            _, u_tg = _target_u(points_phys)
            u_pred = (w1.detach() * uy_out + w2.detach() * uy_in) / denom.detach()
        return ((u_pred - u_tg) ** 2).mean()

    loss_top = _edge_loss(p_top, "uy")
    loss_bot = _edge_loss(p_bot, "uy")
    loss_lft = _edge_loss(p_left, "ux")
    loss_rgt = _edge_loss(p_right, "ux")

    return (loss_top + loss_bot + loss_lft + loss_rgt) / 4.0


def compute_loss(
    model,
    xy_int,
    xy_u=None,
    U_fit=None,
    xy_eps=None,
    E_fit=None,
    target_area=None,
    lam=None,
    nu=0.30,
    band_eps_if=1e-2,
    eps0=3.0e-3,
):
    if lam is None:
        lam = {"data": 1.0, "pde": 1.0, "bc": 0.0, "interface": 0.0, "eik": 0.0, "area": 0.0}

    device = next(model.parameters()).device
    eps = 1e-16

    if hasattr(model, "get_E_scaled"):
        E_1, E_2 = model.get_E_scaled()
    else:
        E_1, E_2 = model.E_1, model.E_2
    lam_1, mu_1 = lame_from_E(E_1, nu)
    lam_2, mu_2 = lame_from_E(E_2, nu)

    xy_int_req = xy_int.detach().clone().requires_grad_(True)

    phi, ux_1, uy_1, ux_2, uy_2 = model(xy_int_req)
    U_1 = torch.cat([ux_1, uy_1], dim=1)
    U_2 = torch.cat([ux_2, uy_2], dim=1)

    w1 = torch.relu(phi)
    w2 = torch.relu(-phi)
    denom = (w1 + w2 + eps)

    loss_data = torch.tensor(0.0, device=device)

    if xy_u is not None and U_fit is not None:
        phi_u, ux1_u, uy1_u, ux2_u, uy2_u = model(xy_u)
        U1_u = torch.cat([ux1_u, uy1_u], dim=1)
        U2_u = torch.cat([ux2_u, uy2_u], dim=1)

        w1_u = torch.relu(phi_u)
        w2_u = torch.relu(-phi_u)
        denom_u = (w1_u + w2_u + eps)

        res_1 = (U1_u - U_fit) ** 2
        res_2 = (U2_u - U_fit) ** 2

        res_u = (w1_u.detach() * res_1 + w2_u.detach() * res_2) / denom_u.detach()
        loss_data = loss_data + res_u.mean()

    if xy_eps is not None and E_fit is not None:
        xy_eps_req = xy_eps.detach().clone().requires_grad_(True)

        phi_e, ux1_e, uy1_e, ux2_e, uy2_e = model(xy_eps_req)
        U1_e = torch.cat([ux1_e, uy1_e], dim=1)
        U2_e = torch.cat([ux2_e, uy2_e], dim=1)

        exx_1, eyy_1, exy_1 = strain_from_u(U1_e, xy_eps_req)
        exx_2, eyy_2, exy_2 = strain_from_u(U2_e, xy_eps_req)

        E_1 = torch.cat([exx_1, eyy_1, exy_1], dim=1)
        E_2 = torch.cat([exx_2, eyy_2, exy_2], dim=1)

        w1_e = torch.relu(phi_e)
        w2_e = torch.relu(-phi_e)
        denom_e = (w1_e + w2_e + eps)

        res_1 = (E_1 - E_fit) ** 2
        res_2 = (E_2 - E_fit) ** 2

        res_e = (w1_e.detach() * res_1 + w2_e.detach() * res_2) / denom_e.detach()
        loss_data = loss_data + res_e.mean()

    Rx_1, Ry_1 = div_sigma_from_u(U_1, xy_int_req, lam_1, mu_1, keep_graph=True)
    Rx_2, Ry_2 = div_sigma_from_u(U_2, xy_int_req, lam_2, mu_2, keep_graph=True)

    R_1 = Rx_1 ** 2 + Ry_1 ** 2
    R_2 = Rx_2 ** 2 + Ry_2 ** 2

    loss_pde = (w1 * R_1 + w2 * R_2).mean()
    # \+ ((w1.detach() * R_1 + w2.detach() * R_2)/denom.detach()).mean()*0.01

    loss_bc = bc_loss(model, n=256, eps0=eps0, nu=nu)

    gphi = torch.autograd.grad(
        phi,
        xy_int_req,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True,
    )[0]
    gx, gy = gphi[:, 0:1], gphi[:, 1:2]
    nrm = (gx * gx + gy * gy).sqrt().clamp_min(1e-12)
    nx, ny = gx / nrm, gy / nrm

    loss_if = torch.tensor(0.0, device=device)
    if lam.get("interface", 0.0) > 0.0:
        band = (phi.detach().abs() < band_eps_if).squeeze()
        if band.any():
            sxx_1, syy_1, sxy_1 = stress_from_u(U_1, xy_int_req, lam_1, mu_1)
            sxx_2, syy_2, sxy_2 = stress_from_u(U_2, xy_int_req, lam_2, mu_2)

            s_nn_1 = sxx_1 * nx * nx + syy_1 * ny * ny + 2.0 * sxy_1 * nx * ny
            s_nn_2 = sxx_2 * nx * nx + syy_2 * ny * ny + 2.0 * sxy_2 * nx * ny

            diff_stress = (s_nn_2 - s_nn_1)[band]
            loss_stress = (diff_stress ** 2).mean()

            jump_u = (U_2 - U_1)[band, :]
            loss_disp = (jump_u ** 2).mean()

            loss_if = loss_stress + 10.0 * loss_disp

    band = (phi.detach().abs() < band_eps_if).squeeze()
    if band.any():
        grad_norm = gphi.norm(2, dim=1, keepdim=True)[band]
        loss_eik = ((grad_norm - 1.0) ** 2).mean()
    else:
        loss_eik = torch.tensor(0.0, device=device)

    H = heaviside(phi)
    area_pred = H.mean()
    loss_area = area_pred if target_area is None else (area_pred - target_area) ** 2

    total = (
        lam["data"] * loss_data
        + lam["pde"] * loss_pde
        + lam["bc"] * loss_bc
        + lam["interface"] * loss_if
        + lam["eik"] * loss_eik
        + lam["area"] * loss_area
    )

    core_loss = total

    return (
        total,
        {
            "data": loss_data,
            "pde": loss_pde,
            "bc": loss_bc,
            "interface": loss_if,
            "eik": loss_eik,
            "area": loss_area,
        },
        core_loss,
    )
