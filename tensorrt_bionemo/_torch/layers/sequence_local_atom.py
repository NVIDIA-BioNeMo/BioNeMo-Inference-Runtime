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
import torch.nn.functional as F


def create_indexing_matrix(K: int, W: int, H: int,
                           device: torch.device) -> torch.Tensor:
    """
    Create the indexing matrix for the sequence local atom attention.
    Args:
        K: int
            The number of windows.
        W: int
            Query window size.
        H: int
            Key window size.
        device: torch.device
            The device to create the indexing matrix on.
    """
    assert W % 2 == 0
    assert H % (W // 2) == 0

    # W//2 = area size, a window has two areas
    # h is the number of areas for each query window
    h = H // (W // 2)
    assert h % 2 == 0

    arange = torch.arange(2 * K, device=device)
    index = ((arange.unsqueeze(0) - arange.unsqueeze(1)) + h // 2).clamp(
        min=0, max=h + 1)
    index = index.view(K, 2, 2 * K)[:, 0, :]
    onehot = F.one_hot(index, num_classes=h + 2)[..., 1:-1].transpose(1, 0)
    return onehot.reshape(2 * K, h * K).float()


def create_gather_indices(
        K: int, W: int, H: int,
        device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute gather indices + validity mask from the indexing matrix.

    Returns:
        gather_indices: [h*K] long tensor — source row in the (2K+1)-padded query
        valid_mask:     [h*K] bool tensor — True where the column has a mapping
    """
    mat = create_indexing_matrix(K, W, H, device)
    col_has_nonzero = mat.sum(dim=0) > 0  # [h*K]
    indices = mat.argmax(dim=0)  # [h*K]
    # Zero-columns: point to a sentinel row (2*K) that will be zero-padded
    indices = torch.where(col_has_nonzero, indices,
                          torch.tensor(2 * K, device=device))
    return indices.long(), col_has_nonzero


def query_to_keys_optimized(
        query: torch.Tensor,
        gather_indices: torch.Tensor,
        # valid_mask: torch.Tensor,
        W: int = None,
        H: int = None) -> torch.Tensor:
    if not query.is_floating_point():
        query = query.float()
    assert H is not None
    multiplicity = 1
    if query.ndim == 3:
        B, N, D = query.shape
        assert W is not None
        K = N // W
        query = query.view(B, K, W, D)
    elif query.ndim == 4:
        B, K, W, D = query.shape
    elif query.ndim == 5:
        B, multiplicity, K, W, D = query.shape
    else:
        raise ValueError("Query tensor must be 3, 4, or 5 dimensions")

    half_W = W // 2
    # Reshape to area representation: [B, mult, 2*K, W//2, D]
    query_areas = query.view(B, multiplicity, 2 * K, half_W, D)

    # Pad with a zero sentinel row at index 2*K: [B, mult, 2*K+1, W//2, D]
    zero_row = query_areas.new_zeros(B, multiplicity, 1, half_W, D)
    query_padded = torch.cat([query_areas, zero_row], dim=2)

    # Gather: indices [h*K] -> [B, mult, h*K, W//2, D]
    result = query_padded[:, :, gather_indices]  # index along dim=2

    return result.reshape(B, multiplicity, K, H, D)


def query_to_keys(query: torch.Tensor,
                  keys_indexing_matrix: torch.Tensor,
                  W: int = None,
                  H: int = None) -> torch.Tensor:
    """
    Convert the query to keys for the sequence local atom attention.
    Args:
        query: torch.Tensor
            The query tensor. Shape [B, N, D]
        W: int
            Query window size.
        H: int
            Key window size.
        keys_indexing_matrix: torch.Tensor,
            The keys indexing matrix. Shape [2 * K, h * K]
    """
    if not query.is_floating_point():
        query = query.float()
    assert H is not None, "Key window size is required"
    # B: batch size, N: number of atoms, D: feature dimension of the atoms
    multiplicity = 1
    if query.ndim == 3:
        B, N, D = query.shape
        # K: number of windows, W: query window size
        assert W is not None, "Query window size is required"
        K = N // W
        query = query.view(B, K, W, D)

    elif query.ndim == 4:
        B, K, W, D = query.shape
    elif query.ndim == 5:
        B, multiplicity, K, W, D = query.shape
    else:
        raise ValueError("Query tensor must be 3, 4, or 5 dimensions")
    # 2*K: number of areas, W//2: area size
    query = query.view(B, multiplicity, 2 * K, W // 2, D)
    return torch.einsum("b m j i d, j k -> b m k i d", query,
                        keys_indexing_matrix.to(query.dtype)).reshape(
                            B, multiplicity, K, H, D)
