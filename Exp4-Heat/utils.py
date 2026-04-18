import os
import random

import numpy as np
import torch


def set_seed(seed: int = 1234) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def activation_masks(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask_pos = (phi >= 0).to(dtype=phi.dtype)
    mask_neg = 1.0 - mask_pos
    return mask_pos, mask_neg


def masked_partition_value(
    phi: torch.Tensor,
    pos_value: torch.Tensor,
    neg_value: torch.Tensor,
) -> torch.Tensor:
    mask_pos, mask_neg = activation_masks(phi)
    return mask_pos * pos_value + mask_neg * neg_value
