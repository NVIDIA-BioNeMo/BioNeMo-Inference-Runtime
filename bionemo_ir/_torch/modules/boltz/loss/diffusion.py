# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
from einops import einsum

from bionemo_ir.dsl_kernels.triton.rigid_align import (
    COV_EPS,
    WEIGHT_EPS,
    rigid_align_transform,
    supports_rigid_align,
)
from bionemo_ir.logger import logger


def horn_rotation(cov_matrix: torch.Tensor) -> torch.Tensor:
    """Proper rotation ``R`` maximizing ``trace(R^T H)`` for ``[..., 3, 3]`` covariances ``H``.

    Horn's quaternion method: the optimal unit quaternion is the top
    eigenvector of a symmetric 4x4 built from ``H``, solved in float64. It is
    the SVD/Kabsch solution ``U diag(1, 1, det(UV^T)) V^T`` without the
    determinant branch. ``H[i, j] = sum_n w_n pred_n[i] true_n[j]``; the result
    maps ``true`` onto ``pred``.
    """
    h = cov_matrix.to(torch.float64)
    sxx, sxy, sxz = h[..., 0, 0], h[..., 1, 0], h[..., 2, 0]
    syx, syy, syz = h[..., 0, 1], h[..., 1, 1], h[..., 2, 1]
    szx, szy, szz = h[..., 0, 2], h[..., 1, 2], h[..., 2, 2]
    rows = (
        (sxx + syy + szz, syz - szy, szx - sxz, sxy - syx),
        (syz - szy, sxx - syy - szz, sxy + syx, szx + sxz),
        (szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy),
        (sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz),
    )
    n = torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)
    _, vectors = torch.linalg.eigh(n)
    q = vectors[..., :, -1]
    w, x, y, z = q.unbind(dim=-1)
    rot = torch.stack(
        (
            w * w + x * x - y * y - z * z,
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            w * w - x * x + y * y - z * z,
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            w * w - x * x - y * y + z * z,
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)
    return rot.to(cov_matrix.dtype)


def weighted_rigid_align(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Algorithm 28
    Compute weighted alignment.

    Contiguous CUDA float32 inputs take a fused solver and pointwise rotation
    with no host synchronization (CUDA-graph safe); other inputs run the
    batched torch path below. Both solve the same Horn/Kabsch rotation in
    float64.

    Parameters
    ----------
    true_coords: torch.Tensor
        The ground truth atom coordinates. Shape [B, multiplicity, N, 3]
    pred_coords: torch.Tensor
        The predicted atom coordinates. Shape [B, multiplicity, N, 3]
    weights: torch.Tensor
        The weights for alignment. Shape [B, multiplicity, N]
    mask: torch.Tensor
        The atoms mask. Shape [B, multiplicity, N]

    Returns
    -------
    torch.Tensor
        Aligned coordinates

    """
    if supports_rigid_align(true_coords, pred_coords, weights, mask):
        rotation, shift = rigid_align_transform(true_coords, pred_coords, weights, mask)
        # Keep coordinates independent of TF32 settings.
        aligned = shift.unsqueeze(-2).expand_as(true_coords)
        for axis in range(3):
            aligned = torch.addcmul(aligned, true_coords[..., axis : axis + 1], rotation[..., None, :, axis])
        return aligned

    num_points, dim = true_coords.shape[-2:]
    weights = (mask * weights).unsqueeze(-1)

    # Compute weighted centroids (clamp mass so empty masks do not NaN).
    weight_sum = weights.sum(dim=2, keepdim=True).clamp(min=WEIGHT_EPS)
    true_centroid = (true_coords * weights).sum(dim=2, keepdim=True) / weight_sum
    pred_centroid = (pred_coords * weights).sum(dim=2, keepdim=True) / weight_sum

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        logger.warning(
            "The size of one of the point clouds is <= dim+1. `WeightedRigidAlign` cannot return a unique rotation."
        )

    # Weighted cross-covariance. Divide by total mass so the absolute scale
    # does not grow with atom count.
    cov_matrix = einsum(weights * pred_coords_centered, true_coords_centered, "b m n i, b m n j -> b m i j")
    # weight_sum is [B, M, 1, 1]; do not squeeze — a [B, M, 1] divisor
    # right-aligns incorrectly against [B, M, 3, 3] and can expand M.
    cov_matrix = cov_matrix / weight_sum

    # A tiny diagonal keeps collapsed / empty clouds on the identity rotation.
    # Half inputs upcast so the jitter survives; float64 inputs stay float64.
    solve_dtype = torch.promote_types(cov_matrix.dtype, torch.float32)
    eye = torch.eye(dim, dtype=solve_dtype, device=cov_matrix.device)
    rot_matrix = horn_rotation(cov_matrix.to(solve_dtype) + eye * COV_EPS).to(dtype=cov_matrix.dtype)

    # Apply the rotation and translation
    aligned_coords = einsum(true_coords_centered, rot_matrix, "b m n i, b m j i -> b m n j") + pred_centroid
    return aligned_coords
