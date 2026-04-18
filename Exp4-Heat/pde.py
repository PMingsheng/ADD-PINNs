import torch


def div_kgrad(
    T_sub: torch.Tensor,
    f_sub: torch.Tensor,
    xy: torch.Tensor,
    keep_graph: bool = False,
) -> torch.Tensor:
    gradT = torch.autograd.grad(
        T_sub,
        xy,
        torch.ones_like(T_sub),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2T_dx2 = torch.autograd.grad(
        gradT[:, 0:1],
        xy,
        torch.ones_like(gradT[:, 0:1]),
        create_graph=keep_graph,
        retain_graph=True,
    )[0][:, 0:1]

    d2T_dy2 = torch.autograd.grad(
        gradT[:, 1:2],
        xy,
        torch.ones_like(gradT[:, 1:2]),
        create_graph=keep_graph,
        retain_graph=keep_graph,
    )[0][:, 1:2]

    if f_sub.ndim == 0:
        f_sub = f_sub.expand_as(T_sub)

    return d2T_dx2 + d2T_dy2 + f_sub


def _grad_norm(scalar_field: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(
        outputs=scalar_field,
        inputs=xy,
        grad_outputs=torch.ones_like(scalar_field),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad.norm(dim=1, keepdim=True)
