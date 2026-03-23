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


def query_to_key_width_edge_masking(n_q: int,
                                    n_k: int,
                                    idx: torch.Tensor,
                                    atom_mask: torch.Tensor | None = None):
    """
    Return a callable(ql) -> ql_key that wraps query/key block generation.

    When atom_mask [B, K, n_q] is provided:
      - All edge gather indices (edge_b, edge_k, idx_gather, oob_gather) are
        precomputed once here, eliminating nonzero/any device syncs at forward time.
      - atom_mask_k is edge-fixed here (no D dependency), so the forward pass
        only needs a single gather on trt_key for edge blocks.

    When atom_mask is None, edge indices are derived from ql shape each call.

    Usage:
        idx = create_indexing_matrix(K, n_q, n_k, device)
        gqk = wrap_query_to_key(n_q=32, n_k=128, idx=idx, atom_mask=mask_b)
        ql_key = gqk(ql_blocked)
    """
    to_keys = lambda x, _i=idx: query_to_keys(x, _i, H=n_k)

    if atom_mask is not None:
        B, K, n_q_ = atom_mask.shape
        device = atom_mask.device
        mask_flat = atom_mask.reshape(B, K * n_q_)

        total_shift, is_edge, n_atom_true = compute_block_indices(
            mask_flat, K, n_q_, n_k)

        # Precompute key mask via sliding window
        left_pad = n_k // 2 - n_q_ // 2
        mask_padded = torch.nn.functional.pad(mask_flat, (left_pad, n_k),
                                              value=0.0)
        atom_mask_k = mask_padded.unfold(-1, n_k,
                                         n_q_)[:, :K].clone()  # [B, K, n_k]

        # Precompute edge gather indices — eliminates device syncs at forward time
        edge_b, edge_k = is_edge.nonzero(as_tuple=True)  # [E]
        E = edge_b.numel()

        if E > 0:
            arange_k = torch.arange(n_k, device=device)
            n_atom_e = n_atom_true[edge_b]
            shift_e = total_shift[edge_b, edge_k]

            raw_start = arange_k.unsqueeze(0).expand(E, n_k)
            oob_start = raw_start >= n_atom_e.unsqueeze(-1)
            idx_start = torch.minimum(raw_start, n_atom_e.unsqueeze(-1) - 1)

            raw_end = n_atom_e.unsqueeze(-1) - n_k + arange_k
            oob_end = raw_end < 0
            idx_end = raw_end.clamp(min=0)

            is_start = (shift_e > 0).unsqueeze(-1)
            idx_gather = torch.where(is_start, idx_start, idx_end)  # [E, n_k]
            oob_gather = torch.where(is_start, oob_start, oob_end)  # [E, n_k]

            # Pre-fix atom_mask_k for edge blocks (no D → done once, free at forward)
            mask_patches = torch.gather(mask_flat[edge_b], 1,
                                        idx_gather).masked_fill_(
                                            oob_gather, 0.0)
            atom_mask_k[edge_b, edge_k] = mask_patches
        else:
            idx_gather = torch.zeros(0, n_k, dtype=torch.long, device=device)
            oob_gather = torch.zeros(0, n_k, dtype=torch.bool, device=device)

        def _wrapped(ql: torch.Tensor) -> torch.Tensor:
            has_sample_dim = ql.ndim == 5
            if has_sample_dim:
                Bi, S, Ki, nq, D = ql.shape
                ql_r = ql.reshape(Bi * S, Ki, nq, D)
            else:
                ql_r = ql

            BS, Ki, nq, D = ql_r.shape
            trt_key = to_keys(ql_r).squeeze(1)  # [BS, K, n_k, D]

            if E > 0:
                ql_flat = ql_r.reshape(BS, Ki * nq, D)
                if ql_flat.dtype != trt_key.dtype:
                    ql_flat = ql_flat.to(trt_key.dtype)

                if has_sample_dim:
                    _eb = (edge_b.unsqueeze(1) * S +
                           torch.arange(S, device=device)).reshape(-1)  # [E*S]
                    _ek = edge_k.unsqueeze(1).expand(E, S).reshape(-1)  # [E*S]
                    _idx = idx_gather.unsqueeze(1).expand(E, S, n_k).reshape(
                        -1, n_k)
                    _oob = oob_gather.unsqueeze(1).expand(E, S, n_k).reshape(
                        -1, n_k)
                else:
                    _eb, _ek, _idx, _oob = edge_b, edge_k, idx_gather, oob_gather

                En = _eb.numel()
                patches = torch.gather(ql_flat[_eb], 1,
                                       _idx.unsqueeze(-1).expand(
                                           En, n_k, D)).masked_fill_(
                                               _oob.unsqueeze(-1), 0.0)
                trt_key[_eb, _ek] = patches

            if has_sample_dim:
                trt_key = trt_key.reshape(Bi, S, Ki, n_k, D)
            return trt_key

        return _wrapped

    else:
        # No mask: derive edge indices from ql shape on each call
        def _wrapped(ql: torch.Tensor) -> torch.Tensor:
            if ql.ndim == 5:
                B_, S_, K_, n_q_, D_ = ql.shape
                BS_ = B_ * S_
            else:
                BS_, K_, n_q_, D_ = ql.shape
            ones = torch.ones(BS_, K_ * n_q_, device=ql.device)
            total_shift_, is_edge_, n_atom_true_ = compute_block_indices(
                ones, K_, n_q_, n_k)
            pc = (total_shift_, is_edge_, n_atom_true_)
            return apply_edge_mask_query_and_key(ql, n_k, to_keys, pc)

        return _wrapped


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
    pad_size = (multiple - (tensor.shape[dim] % multiple)) % multiple
    extend_size = tensor.shape[dim] + pad_size
    pad = [0, 0] * (tensor.dim() - dim - 1) + [0, pad_size]
    tensor = torch.nn.functional.pad(tensor, pad, mode="constant", value=0.0)
    tensor_shape = list(tensor.shape)
    tensor_shape[dim] = extend_size // multiple
    tensor_shape.insert(dim + 1, multiple)
    tensor = tensor.reshape(tensor_shape)
    return tensor, current_size


def compute_block_indices(
    atom_mask: torch.Tensor,
    K: int,
    n_q: int,
    n_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-block shift, edge flag, and true atom counts.

    Args:
        atom_mask: [B, N_atom]
        K:         number of blocks
        n_q:       query window size
        n_k:       key window size
    Returns:
        total_shift:  [B, K]  >0 left edge (underflow), <0 right edge (overflow), 0 middle
        is_edge:      [B, K]  True where the block's window needed a shift
        n_atom_true:  [B]     number of valid atoms per batch item
    """
    B = atom_mask.shape[0]
    device = atom_mask.device
    n_atom_true = atom_mask.sum(dim=-1).long()  # [B]

    offset = n_q // 2
    centers = offset + torch.arange(K, device=device) * n_q  # [K]

    # Only need window start/end — avoid materialising [B, K, n_k]
    win_start = (centers - n_k // 2).unsqueeze(0).expand(B, K)  # [B, K]
    win_end = (centers + n_k // 2 - 1).unsqueeze(0).expand(B, K)  # [B, K]
    n_atom_bk = n_atom_true.view(B, 1).expand(B, K)  # [B, K]

    underflow = torch.relu(-win_start)
    overflow = torch.relu(win_end - (n_atom_bk - 1))
    total_shift = torch.where(underflow > 0, underflow, -overflow)  # [B, K]
    is_edge = total_shift.abs() > 0  # [B, K]

    return total_shift, is_edge, n_atom_true


def fix_boundary_blocks(
    trt_key: torch.Tensor,
    ql: torch.Tensor,
    n_atom_true: torch.Tensor,
    total_shift: torch.Tensor,
    is_edge: torch.Tensor,
) -> torch.Tensor:
    """
    Patch edge blocks using the sign of total_shift.
    Used only in the no-mask fallback path (atom_mask=None).

    Args:
        trt_key:     [B, K, n_k, D]
        ql:          [B, N_atom, D]
        n_atom_true: [B]
        total_shift: [B, K]
        is_edge:     [B, K]
    Returns:
        fixed:       [B, K, n_k, D]
    """
    if not is_edge.any():
        return trt_key

    _, K, n_k, D = trt_key.shape
    device = trt_key.device

    edge_b, edge_k = is_edge.nonzero(as_tuple=True)  # [E] each
    E = edge_b.numel()

    n_atom_e = n_atom_true[edge_b]
    shift_e = total_shift[edge_b, edge_k]

    arange_k = torch.arange(n_k, device=device)

    raw_start = arange_k.unsqueeze(0).expand(E, n_k)
    oob_start = raw_start >= n_atom_e.unsqueeze(-1)
    idx_start = torch.minimum(raw_start, n_atom_e.unsqueeze(-1) - 1)

    raw_end = n_atom_e.unsqueeze(-1) - n_k + arange_k
    oob_end = raw_end < 0
    idx_end = raw_end.clamp(min=0)

    is_start = (shift_e > 0).unsqueeze(-1)
    idx = torch.where(is_start, idx_start, idx_end)
    oob = torch.where(is_start, oob_start, oob_end)

    gathered = torch.gather(ql[edge_b], 1,
                            idx.unsqueeze(-1).expand(E, n_k, D)).masked_fill_(
                                oob.unsqueeze(-1), 0.0)

    trt_key[edge_b, edge_k] = gathered.to(dtype=trt_key.dtype)
    return trt_key


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


def apply_edge_mask_query_and_key(
    ql: torch.Tensor,
    n_k: int,
    to_keys: callable,
    precomputed: tuple,
) -> torch.Tensor:
    """
    Fallback path used when atom_mask=None.

    Args:
        ql:          [B, K, n_q, D] or [B, S, K, n_q, D]  pre-blocked
        n_k:         key window size
        to_keys:     callable([B, K, n_q, D]) -> [B, 1, K, n_k, D]
        precomputed: (total_shift, is_edge, n_atom_true)
    Returns:
        ql_key: [B, K, n_k, D] or [B, S, K, n_k, D]
    """
    has_sample_dim = ql.ndim == 5

    if has_sample_dim:
        B, S, K, n_q, D = ql.shape
        ql = ql.reshape(B * S, K, n_q, D)

    BS, K, n_q, D = ql.shape

    trt_key = to_keys(ql).squeeze(1)  # [BS, K, n_k, D]

    ql_flat = ql.reshape(BS, K * n_q, D)
    if ql_flat.dtype != trt_key.dtype:
        ql_flat = ql_flat.to(trt_key.dtype)

    total_shift, is_edge, n_atom_true = precomputed

    if has_sample_dim:
        total_shift = total_shift.unsqueeze(1).expand(B, S, K).reshape(BS, K)
        is_edge = is_edge.unsqueeze(1).expand(B, S, K).reshape(BS, K)
        n_atom_true = n_atom_true.unsqueeze(1).expand(B, S).reshape(BS)

    ql_key = fix_boundary_blocks(trt_key, ql_flat, n_atom_true, total_shift,
                                 is_edge)

    if has_sample_dim:
        ql_key = ql_key.reshape(B, S, K, n_k, D)
    return ql_key

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
