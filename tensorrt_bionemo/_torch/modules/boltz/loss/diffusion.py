# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    # Compute weighted centroids
    true_centroid = (true_coords * weights).sum(
        dim=2, keepdim=True) / weights.sum(dim=2, keepdim=True)
    pred_centroid = (pred_coords * weights).sum(
        dim=2, keepdim=True) / weights.sum(dim=2, keepdim=True)

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        print("Warning: The size of one of the point clouds is <= dim+1. " +
              "`WeightedRigidAlign` cannot return a unique rotation.")

    # Compute the weighted covariance matrix
    cov_matrix = einsum(weights * pred_coords_centered, true_coords_centered,
                        "b m n i, b m n j -> b m i j")

    # Compute the SVD of the covariance matrix, required float32 for svd and determinant
    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    U, S, V = torch.linalg.svd(
        cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None)
    V = V.mH

    # Catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        print("Warning: Excessively low rank of " +
              "cross-correlation between aligned point clouds. " +
              "`WeightedRigidAlign` cannot return a unique rotation.")

    # Compute the rotation matrix
    rot_matrix = torch.einsum("b m i j, b m k j -> b m i k", U,
                              V).to(dtype=torch.float32)

    # Ensure proper rotation matrix with determinant 1
    F = torch.eye(dim, dtype=cov_matrix_32.dtype,
                  device=cov_matrix.device)[None].repeat(
                      batch_size, multiplicity, 1, 1)
    F[:, :, -1, -1] = torch.det(rot_matrix)
    rot_matrix = einsum(U, F, V, "b m i j, b m j k, b m l k -> b m i l")
    rot_matrix = rot_matrix.to(dtype=original_dtype)

    # Apply the rotation and translation
    aligned_coords = (einsum(true_coords_centered, rot_matrix,
                             "b m n i, b m j i -> b m n j") + pred_centroid)
    return aligned_coords
