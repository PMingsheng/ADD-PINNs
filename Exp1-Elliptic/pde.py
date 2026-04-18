import torch


def div_kgrad(
    u_sub: torch.Tensor,
    f_sub: torch.Tensor,
    xy: torch.Tensor,
    keep_graph: bool = False,
) -> torch.Tensor:
    grad_u = torch.autograd.grad(
        u_sub,
        xy,
        torch.ones_like(u_sub),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2u_dx2 = torch.autograd.grad(
        grad_u[:, 0:1],
        xy,
        torch.ones_like(grad_u[:, 0:1]),
        create_graph=keep_graph,
        retain_graph=True,
    )[0][:, 0:1]

    d2u_dy2 = torch.autograd.grad(
        grad_u[:, 1:2],
        xy,
        torch.ones_like(grad_u[:, 1:2]),
        create_graph=keep_graph,
        retain_graph=keep_graph,
    )[0][:, 1:2]

    if f_sub.ndim == 0:
        f_sub = f_sub.expand_as(u_sub)

    return d2u_dx2 + d2u_dy2 + f_sub


def _grad_norm(scalar_field: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(
        outputs=scalar_field,
        inputs=xy,
        grad_outputs=torch.ones_like(scalar_field),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad.norm(dim=1, keepdim=True)
