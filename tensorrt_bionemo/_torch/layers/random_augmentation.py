import math
from typing import Optional

import torch

# the following is copied from Torch3D, BSD License, Copyright (c) Meta Platforms, Inc. and affiliates.


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def random_quaternions(n: int,
                       dtype: Optional[torch.dtype] = None,
                       device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(n: int,
                     dtype: Optional[torch.dtype] = None,
                     device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions)


def centre_random_augmentation(x: torch.Tensor,
                               mask: Optional[torch.Tensor] = None,
                               s_trans: float = 1.0) -> torch.Tensor:
    """Center, rotate, and translate ``[..., N_atom, 3]`` coordinates."""
    lead = x.shape[:-2]
    n = math.prod(lead) if lead else 1
    rots = random_rotations(n, dtype=x.dtype,
                            device=x.device).reshape(*lead, 3, 3)
    trans = s_trans * torch.randn((*lead, 3), dtype=x.dtype, device=x.device)
    if mask is None:
        centre = x.mean(dim=-2, keepdim=True)
    else:
        m = mask.unsqueeze(-1).to(x.dtype)
        centre = (x * m).sum(dim=-2, keepdim=True) / m.sum(
            dim=-2, keepdim=True).clamp(min=1e-7)
    x = (x - centre) @ rots.transpose(-1, -2) + trans[..., None, :]
    if mask is not None:
        x = x * mask.unsqueeze(-1).to(x.dtype)
    return x
