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
import torch.nn.functional as F


def pad_to_multiple_and_divide(tensor: torch.Tensor,
                               multiple: int,
                               dim: int = 1):
    """
    Pad a tensor to a multiple of a given value along a given dimension.
    Args:
        tensor: The tensor to pad.
        multiple: The multiple to pad to.
        dim: The dimension to pad along.
    Returns:
        The padded and divided tensor.
    """
    current_size = tensor.shape[dim]
    pad_size = (multiple - (tensor.shape[dim] % multiple))
    extend_size = tensor.shape[dim] + pad_size
    pad = [0, 0] * (tensor.dim() - dim - 1) + [0, pad_size]
    tensor = torch.nn.functional.pad(tensor, pad, mode="constant", value=0.0)
    tensor_shape = list(tensor.shape)
    tensor_shape[dim] = extend_size // multiple
    tensor_shape.insert(dim + 1, multiple)
    tensor = tensor.reshape(tensor_shape)
    return tensor, current_size


def to_blocks(x: torch.Tensor, num_blocks: int, window: int) -> torch.Tensor:
    """Pad ``[B, N, D]`` and reshape to ``[B, num_blocks, window, D]``.

    An explicit block count avoids an extra block when ``N`` is divisible by
    ``window``.
    """
    B, N, D = x.shape
    pad = num_blocks * window - N
    if pad < 0:
        raise ValueError(
            f"num_blocks*window ({num_blocks * window}) < N ({N})")
    if pad > 0:
        x = F.pad(x, (0, 0, 0, pad))
    return x.reshape(B, num_blocks, window, D)


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
    """Gather-based query→keys for sequence-local atom attention.

    Bit-exact to the einsum formulation but avoids TF32 precision loss on
    Ampere+ GPUs (matmul silently uses TF32, rounds large integer indices,
    causing OOB gathers downstream).

    The output preserves the input's number of dimensions:
        [B, N, D]              → [B, K, H, D]      (3-d in, 4-d block out)
        [B, K, W, D]           → [B, K, H, D]      (multiplicity dim squeezed)
        [B, mult, K, W, D]     → [B, mult, K, H, D]
    OOB columns gather from a sentinel zero row appended at flat index
    ``K*W``, matching the OSS reference's ``F.pad(value=0)`` + ``unfold``.
    """
    if not query.is_floating_point():
        query = query.float()
    assert H is not None
    input_ndim = query.ndim
    multiplicity = 1
    if input_ndim == 3:
        B, N, D = query.shape
        assert W is not None
        K = N // W
        query = query.view(B, K, W, D)
    elif input_ndim == 4:
        B, K, W, D = query.shape
    elif input_ndim == 5:
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
    result = result.reshape(B, multiplicity, K, H, D)

    # Preserve input ndim: squeeze the synthetic multiplicity dim when caller
    # didn't supply one. Boltz passes 5-d inputs (real multiplicity) and
    # receives 5-d outputs unchanged.
    if input_ndim < 5:
        result = result.squeeze(1)
    return result


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
