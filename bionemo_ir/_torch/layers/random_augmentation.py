# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this file are copied from PyTorch3D:
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# The PyTorch3D-derived portion is licensed under BSD-3-Clause; see
# LICENSES/BSD-3-Clause.txt in the repository root.
# Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py

import math

import torch


def compute_random_augmentation(
    batch_size: int = 1,
    multiplicity: int = 1,
    s_trans: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute random augmentation for the coordinates.
    Args:
        multiplicity (int):
            The number of diffusion samples. Default: 1
        s_trans (float):
            The translation scale. Default: 1.0
        device (Optional[torch.device]):
            The device to compute the random augmentation. Default: None
        dtype (torch.dtype):
            The dtype to compute the random augmentation. Default: torch.float32
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The random rotation matrix and the random translation.
    """
    # Using quaternion to create random rotation matrix shape [*, 3, 3]
    R = random_rotations(multiplicity * batch_size, dtype=dtype, device=device, generator=generator).view(
        batch_size, multiplicity, 3, 3
    )

    # Using randn to create random translation matrix shape [*, 1, 3]
    random_trans = (
        torch.randn((batch_size, multiplicity, 1, 3), dtype=dtype, device=device, generator=generator) * s_trans
    )
    return R, random_trans


# PyTorch3D-derived helpers begin here.


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


def random_quaternions(
    n: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.
        generator: Optional RNG to draw from. When supplied, randomness is
            taken from this generator instead of the default one — used to keep
            sampling off the default CUDA generator that ``torch.cuda.graph``
            capture registers.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device, generator=generator)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.
        generator: Optional explicit random-number generator.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, dtype=dtype, device=device, generator=generator)
    return quaternion_to_matrix(quaternions)


def broadcast_atom_mask(positions: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    """Reshape ``atom_mask`` to broadcast against an atom-position tensor.

    ``positions`` is ``[*, ..., N_atom, 3]`` and ``atom_mask`` is
    ``[*, N_atom]``, where ``positions`` may carry extra batch dims (e.g. a
    diffusion-samples axis ``S``) between the mask's batch dims and the atom
    axis -- as in ``[B, S, N_atom, 3]`` vs ``[B, N_atom]``. Returns the mask
    reshaped to ``[*mask_batch, 1, ..., 1, N_atom, 1]`` (cast to ``positions``'s
    dtype) so it broadcasts over those extra dims and the coordinate axis. A
    bare ``atom_mask[..., None]`` would instead right-align the mask's batch dim
    with ``positions``'s samples dim and fail / misbroadcast.
    """
    extra_batch_dims = positions.ndim - atom_mask.ndim - 1
    return atom_mask.reshape(
        *atom_mask.shape[:-1],
        *((1,) * extra_batch_dims),
        atom_mask.shape[-1],
        1,
    ).to(positions.dtype)


def centre_random_augmentation(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    s_trans: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Center, rotate, and translate ``[..., N_atom, 3]`` coordinates.

    ``mask`` may carry fewer dims than ``x`` -- the EDM rollout augments
    ``[B, S, N_atom, 3]`` coordinates under a ``[B, N_atom]`` atom mask -- so it
    is broadcast rather than merely unsqueezed.
    """
    lead = x.shape[:-2]
    n = math.prod(lead) if lead else 1
    rots = random_rotations(n, dtype=x.dtype, device=x.device, generator=generator).reshape(*lead, 3, 3)
    trans = s_trans * torch.randn((*lead, 3), dtype=x.dtype, device=x.device, generator=generator)
    if mask is None:
        centre = x.mean(dim=-2, keepdim=True)
    else:
        m = broadcast_atom_mask(x, mask)
        centre = (x * m).sum(dim=-2, keepdim=True) / m.sum(dim=-2, keepdim=True).clamp(min=1e-7)
    x = (x - centre) @ rots.transpose(-1, -2) + trans[..., None, :]
    if mask is not None:
        x = x * broadcast_atom_mask(x, mask)
    return x
