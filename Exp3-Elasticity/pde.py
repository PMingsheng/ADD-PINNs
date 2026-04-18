import torch


def lame_from_E(E: torch.Tensor, nu: float):
    nu_t = torch.as_tensor(nu, dtype=E.dtype, device=E.device)
    mu = E / (2.0 * (1.0 + nu_t))
    lam = (E * nu_t) / (1.0 - nu_t * nu_t)
    return lam, mu


def strain_from_u(U, xy):
    ux, uy = U[:, 0:1], U[:, 1:2]
    gux = torch.autograd.grad(ux, xy, torch.ones_like(ux), create_graph=True, retain_graph=True)[0]
    guy = torch.autograd.grad(uy, xy, torch.ones_like(uy), create_graph=True, retain_graph=True)[0]
    dux_dx, dux_dy = gux[:, 0:1], gux[:, 1:2]
    duy_dx, duy_dy = guy[:, 0:1], guy[:, 1:2]
    exx = dux_dx
    eyy = duy_dy
    exy = 0.5 * (dux_dy + duy_dx)
    return exx, eyy, exy


def stress_from_u(U, xy, lam, mu):
    exx, eyy, exy = strain_from_u(U, xy)
    tr = exx + eyy
    sxx = lam * tr + 2.0 * mu * exx
    syy = lam * tr + 2.0 * mu * eyy
    sxy = 2.0 * mu * exy
    return sxx, syy, sxy


def div_sigma_from_u(U, xy, lam, mu, keep_graph: bool = False):
    sxx, syy, sxy = stress_from_u(U, xy, lam, mu)
    cg = bool(keep_graph)
    dsxx = torch.autograd.grad(sxx, xy, torch.ones_like(sxx), create_graph=cg, retain_graph=True)[0]
    dsxy = torch.autograd.grad(sxy, xy, torch.ones_like(sxy), create_graph=cg, retain_graph=True)[0]
    dsyy = torch.autograd.grad(syy, xy, torch.ones_like(syy), create_graph=cg, retain_graph=keep_graph)[0]
    fx = dsxx[:, 0:1] + dsxy[:, 1:2]
    fy = dsxy[:, 0:1] + dsyy[:, 1:2]
    return fx, fy


def strain_from_u_batch(xy_req, ux, uy):
    du = torch.autograd.grad(ux, xy_req, grad_outputs=torch.ones_like(ux), create_graph=True, retain_graph=True)[0]
    dv = torch.autograd.grad(uy, xy_req, grad_outputs=torch.ones_like(uy), create_graph=True, retain_graph=True)[0]
    ux_x = du[:, 0:1]
    ux_y = du[:, 1:2]
    uy_x = dv[:, 0:1]
    uy_y = dv[:, 1:2]
    exx = ux_x
    eyy = uy_y
    exy = 0.5 * (ux_y + uy_x)
    return exx, eyy, exy


def stress_from_u_batch(xy_req, ux, uy, lam, mu):
    exx, eyy, exy = strain_from_u_batch(xy_req, ux, uy)
    tr = exx + eyy
    sxx = 2.0 * mu * exx + lam * tr
    syy = 2.0 * mu * eyy + lam * tr
    sxy = 2.0 * mu * exy
    return sxx, syy, sxy


def div_sigma_batch(xy_req, ux, uy, lam, mu):
    sxx, syy, sxy = stress_from_u_batch(xy_req, ux, uy, lam, mu)
    dsxx = torch.autograd.grad(sxx, xy_req, grad_outputs=torch.ones_like(sxx), create_graph=False, retain_graph=True)[0]
    dsxy = torch.autograd.grad(sxy, xy_req, grad_outputs=torch.ones_like(sxy), create_graph=False, retain_graph=True)[0]
    dsyy = torch.autograd.grad(syy, xy_req, grad_outputs=torch.ones_like(syy), create_graph=False, retain_graph=True)[0]
    fx = dsxx[:, 0:1] + dsxy[:, 1:2]
    fy = dsxy[:, 0:1] + dsyy[:, 1:2]
    return fx, fy
