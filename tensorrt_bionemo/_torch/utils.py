import torch
import torch.nn.functional as F


def pad_dim(t: torch.Tensor,
            dim: int,
            pad_len: int,
            value: float = 0) -> torch.Tensor:
    if pad_len == 0:
        return t

    total_dims = len(t.shape)
    padding = [0] * (2 * (total_dims - dim))
    padding[2 * (total_dims - 1 - dim) + 1] = pad_len
    return F.pad(t, tuple(padding), value=value)
