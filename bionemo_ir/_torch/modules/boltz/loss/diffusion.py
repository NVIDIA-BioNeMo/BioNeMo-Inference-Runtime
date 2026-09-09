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

from bionemo_ir.logger import logger

# Floor used when summing alignment weights / normalizing the covariance.
# Prevents NaN centroids (empty mask) and keeps SVD inputs well-scaled.
_WEIGHT_EPS = 1e-8
# Diagonal jitter added to the float32 covariance before SVD. Makes near-
# singular / ill-conditioned 3x3 matrices (common mid-denoising on large
# complexes) converge under cusolver ``gesvd``.
_COV_EPS = 1e-6


def weighted_rigid_align(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Algorithm 28
    Compute weighted alignment.

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
    batch_size, multiplicity, num_points, dim = true_coords.shape
    weights = (mask * weights).unsqueeze(-1)

    # Compute weighted centroids (clamp mass so empty masks do not NaN).
    weight_sum = weights.sum(dim=2, keepdim=True).clamp(min=_WEIGHT_EPS)
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
    # does not grow with atom count (large complexes otherwise stress SVD).
    cov_matrix = einsum(weights * pred_coords_centered, true_coords_centered, "b m n i, b m n j -> b m i j")
    # weight_sum is [B, M, 1, 1]; do not squeeze — a [B, M, 1] divisor
    # right-aligns incorrectly against [B, M, 3, 3] and can expand M.
    cov_matrix = cov_matrix / weight_sum

    # SVD / det require float32. Add a tiny diagonal so near-singular batches
    # (collinear / collapsed clouds mid-denoising) stay invertible.
    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    eye = torch.eye(dim, dtype=cov_matrix_32.dtype, device=cov_matrix_32.device)
    cov_matrix_32 = cov_matrix_32 + eye * _COV_EPS

    try:
        U, S, V = torch.linalg.svd(cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None)
    except (torch.linalg.LinAlgError, RuntimeError) as exc:
        # Skip the rigid rotation for this step; keep the translation onto the
        # predicted centroid so reverse-diffusion can continue.
        logger.warning(f"weighted_rigid_align SVD failed ({exc}); falling back to identity rotation.")
        return (true_coords_centered + pred_centroid).to(dtype=true_coords.dtype)

    V = V.mH

    # Catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        logger.warning(
            "Excessively low rank of cross-correlation between aligned "
            "point clouds. `WeightedRigidAlign` cannot return a unique "
            "rotation."
        )

    # Compute the rotation matrix
    rot_matrix = torch.einsum("b m i j, b m k j -> b m i k", U, V).to(dtype=torch.float32)

    # Ensure proper rotation matrix with determinant 1
    F = eye[None].repeat(batch_size, multiplicity, 1, 1)
    F[:, :, -1, -1] = torch.det(rot_matrix)
    rot_matrix = einsum(U, F, V, "b m i j, b m j k, b m l k -> b m i l")
    rot_matrix = rot_matrix.to(dtype=original_dtype)

    # Apply the rotation and translation
    aligned_coords = einsum(true_coords_centered, rot_matrix, "b m n i, b m j i -> b m n j") + pred_centroid
    return aligned_coords
