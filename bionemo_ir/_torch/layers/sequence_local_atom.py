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

# Portions of this file are adapted from OpenFold3:
# Copyright 2025 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
"""Sequence-local attention and layout-neutral token/atom primitives."""

import contextlib
from typing import Literal

import torch
import torch.nn.functional as F

from bionemo_ir._torch.utils.common import _deterministic_algorithms


def pad_to_multiple_and_divide(tensor: torch.Tensor, multiple: int, dim: int = 1):
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
    pad_size = multiple - (tensor.shape[dim] % multiple)
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
        raise ValueError(f"num_blocks*window ({num_blocks * window}) < N ({N})")
    if pad > 0:
        x = F.pad(x, (0, 0, 0, pad))
    return x.reshape(B, num_blocks, window, D)


def create_indexing_matrix(K: int, W: int, H: int, device: torch.device) -> torch.Tensor:
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
    index = ((arange.unsqueeze(0) - arange.unsqueeze(1)) + h // 2).clamp(min=0, max=h + 1)
    index = index.view(K, 2, 2 * K)[:, 0, :]
    onehot = F.one_hot(index, num_classes=h + 2)[..., 1:-1].transpose(1, 0)
    return onehot.reshape(2 * K, h * K).float()


def create_gather_indices(K: int, W: int, H: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute gather indices + validity mask from the indexing matrix.

    Returns:
        gather_indices: [h*K] long tensor — source row in the (2K+1)-padded query
        valid_mask:     [h*K] bool tensor — True where the column has a mapping
    """
    mat = create_indexing_matrix(K, W, H, device)
    col_has_nonzero = mat.sum(dim=0) > 0  # [h*K]
    indices = mat.argmax(dim=0)  # [h*K]
    # Zero-columns: point to a sentinel row (2*K) that will be zero-padded
    indices = torch.where(col_has_nonzero, indices, torch.tensor(2 * K, device=device))
    return indices.long(), col_has_nonzero


def query_to_keys_optimized(
    query: torch.Tensor, gather_indices: torch.Tensor, W: int = None, H: int = None
) -> torch.Tensor:
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


def query_to_keys(
    query: torch.Tensor, keys_indexing_matrix: torch.Tensor, W: int = None, H: int = None
) -> torch.Tensor:
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
    return torch.einsum("b m j i d, j k -> b m k i d", query, keys_indexing_matrix.to(query.dtype)).reshape(
        B, multiplicity, K, H, D
    )


def compute_atom_broadcast_index(
    token_mask: torch.Tensor,
    num_atoms_per_token: torch.Tensor,
) -> torch.Tensor:
    """Precompute a gather index for graph-safe token-to-atom expansion.

    The index reproduces the dynamic ``repeat_interleave`` expansion used by
    :func:`broadcast_token_features_to_atoms`. Each valid token index is
    repeated by its atom count; remaining rows select a trailing zero-padding
    token. Because the index length depends on tensor values, build it eagerly
    before CUDA graph capture and reuse it during graph replay.

    Args:
        token_mask: Valid-token mask shaped ``[*batch, N_token]``.
        num_atoms_per_token: Atom counts with the same shape as ``token_mask``.

    Returns:
        A flat integer gather index of length
        ``prod(batch) * max_atoms_in_batch``.

    Used by:
        OpenFold3 diffusion setup, which caches the index for CUDA graph replay.
    """
    counts = num_atoms_per_token * token_mask.int()
    max_num_atoms = torch.max(torch.sum(counts, dim=-1)).int()
    padded_counts = (
        torch.concat(
            [counts, max_num_atoms - torch.sum(counts, dim=-1, keepdim=True)],
            dim=-1,
        )
        .reshape(-1)
        .int()
    )
    row_ids = torch.arange(padded_counts.numel(), device=token_mask.device)
    return torch.repeat_interleave(row_ids, padded_counts)


def broadcast_token_features_to_atoms(
    token_mask: torch.Tensor,
    num_atoms_per_token: torch.Tensor,
    token_features: torch.Tensor,
    token_dim: int | None = -1,
    max_num_atoms_per_token: int | None = None,
    expand_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Repeat each token feature for the atoms owned by that token.

    Invalid tokens contribute no atoms. By default, atom rows are packed to
    the largest atom count in the batch and unused rows are zero. Supplying
    ``max_num_atoms_per_token`` instead creates a fixed-width block per token.
    A precomputed ``expand_index`` selects a CUDA-graph-safe ``index_select``
    path; otherwise the function uses value-dependent ``repeat_interleave``.

    Feature tensors may have payload dimensions after the token axis and may
    include an extra repeated/sample batch axis.

    Args:
        token_mask: Valid-token mask shaped ``[*batch, N_token]``.
        num_atoms_per_token: Number of atoms owned by each token.
        token_features: Features containing a token axis.
        token_dim: Position of the token axis in ``token_features``.
        max_num_atoms_per_token: Optional fixed atom capacity for every token.
        expand_index: Optional index from :func:`compute_atom_broadcast_index`.

    Returns:
        Atom-level features shaped
        ``[*feature_batch, N_atom, *feature_payload]``.

    Used by:
        OpenFold3 atom attention, confidence heads, and representative/frame
        atom utilities through its compatibility wrapper.
    """
    n_token = token_mask.shape[-1]
    batch_dims = token_mask.shape[:-1]
    feat_batch_dims = token_features.shape[:token_dim]
    feat_dims = token_features.shape[token_dim:][1:]

    num_atoms_per_token = num_atoms_per_token * token_mask.int()
    token_features = token_features * token_mask.reshape((*batch_dims, n_token, *((1,) * len(feat_dims))))

    if max_num_atoms_per_token is not None:
        num_atoms_per_token = torch.stack(
            [num_atoms_per_token, max_num_atoms_per_token - num_atoms_per_token],
            dim=-1,
        ).reshape((*batch_dims, 2 * n_token))
        normalized_token_dim = token_dim if token_dim >= 0 else token_dim + token_features.ndim
        token_features = token_features.unsqueeze(normalized_token_dim + 1)
        pad = [0, 0] * token_features.ndim
        pad[2 * (token_features.ndim - normalized_token_dim - 2) + 1] = 1
        token_features = torch.nn.functional.pad(token_features, pad).reshape((*batch_dims, 2 * n_token, *feat_dims))

    padded_token_features = torch.concat(
        [
            token_features,
            torch.zeros(
                (*feat_batch_dims, 1, *feat_dims),
                dtype=token_features.dtype,
                device=token_features.device,
            ),
        ],
        dim=token_dim,
    ).reshape(-1, *feat_dims)

    n_batch = 1
    for dim in batch_dims:
        n_batch *= int(dim)
    n_feat_batch = 1
    for dim in feat_batch_dims:
        n_feat_batch *= int(dim)
    can_tile = feat_batch_dims == batch_dims or n_feat_batch % n_batch == 0
    if expand_index is not None and can_tile:
        max_num_atoms = expand_index.numel() // n_batch
        if n_feat_batch != n_batch:
            n_groups = n_feat_batch // n_batch
            batch_of = torch.arange(expand_index.numel(), device=expand_index.device) // max_num_atoms
            base = (expand_index + batch_of * (n_groups - 1) * (n_token + 1)).reshape(n_batch, max_num_atoms)
            group_offset = (torch.arange(n_groups, device=expand_index.device) * (n_token + 1)).reshape(1, n_groups, 1)
            full_index = (base.unsqueeze(1) + group_offset).reshape(-1)
        else:
            full_index = expand_index
        atom_features = padded_token_features.index_select(0, full_index)
        return atom_features.reshape((*feat_batch_dims, max_num_atoms, *feat_dims))

    max_num_atoms = torch.max(torch.sum(num_atoms_per_token, dim=-1)).int()
    padded_num_atoms_per_token = torch.concat(
        [
            num_atoms_per_token,
            max_num_atoms - torch.sum(num_atoms_per_token, dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    if batch_dims != feat_batch_dims:
        # Match C-order flatten of [batch, samples, tokens]: each batch's
        # counts stay contiguous across its samples.
        n_groups = n_feat_batch // n_batch
        padded_num_atoms_per_token = padded_num_atoms_per_token.reshape(n_batch, -1).repeat_interleave(n_groups, dim=0)
    padded_num_atoms_per_token = padded_num_atoms_per_token.reshape(-1).int()

    atom_features = torch.repeat_interleave(
        input=padded_token_features,
        repeats=padded_num_atoms_per_token,
        dim=0,
    )
    return atom_features.reshape((*feat_batch_dims, max_num_atoms, *feat_dims))


def aggregate_atom_features_to_tokens(
    token_mask: torch.Tensor,
    atom_to_token_index: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_features: torch.Tensor,
    atom_dim: int | None = -1,
    aggregate_fn: Literal["mean", "sum"] = "mean",
    eps: float = 1e-9,
) -> torch.Tensor:
    """Reduce packed atom features into their owning token rows.

    Masked atoms contribute neither values nor counts. Their owner index is
    redirected to a dropped padding token. Floating-point scatter accumulation
    runs under deterministic-algorithm mode so repeated diffusion steps do not
    amplify CUDA atomic ordering differences.

    Args:
        token_mask: Token layout shaped ``[*batch, N_token]``; its final
            dimension determines the output token count.
        atom_to_token_index: Integer owner for each packed atom, shaped
            ``[*batch, N_atom]``.
        atom_mask: Valid-atom mask with the same shape as the owner index.
        atom_features: Features containing the atom axis and optional payload.
        atom_dim: Position of the atom axis in ``atom_features``.
        aggregate_fn: Either ``"sum"`` or a valid-atom ``"mean"``.
        eps: Denominator offset used by mean aggregation.

    Returns:
        Token-level features with the atom axis replaced by ``N_token``.

    Raises:
        ValueError: If ``aggregate_fn`` is unsupported.

    Used by:
        OpenFold3 sequence-local atom attention when returning atom features to
        token representations.
    """
    n_token = token_mask.shape[-1]
    batch_dims = token_mask.shape[:-1]
    feat_batch_dims = atom_features.shape[:atom_dim]
    feat_dims = atom_features.shape[atom_dim:][1:]
    atom_features = atom_features * atom_mask.reshape(atom_mask.shape + (1,) * len(feat_dims))
    atom_to_token_index = torch.where(atom_mask.bool(), atom_to_token_index, n_token)

    if batch_dims == feat_batch_dims:
        repeated_index = atom_to_token_index.reshape(
            *atom_to_token_index.shape,
            *(1,) * len(feat_dims),
        ).repeat(*((1,) * (len(batch_dims) + 1) + feat_dims))
    else:
        batch_n_repeat = feat_batch_dims[-1]
        repeated_index = atom_to_token_index.reshape(
            *atom_to_token_index.shape,
            *(1,) * len(feat_dims),
        ).repeat(*((1,) * (len(batch_dims) - 1) + (batch_n_repeat,) + (1,) + feat_dims))

    if aggregate_fn not in ("mean", "sum"):
        raise ValueError(f"Invalid aggregation function: {aggregate_fn}")

    token_features = torch.zeros(
        (*feat_batch_dims, n_token + 1, *feat_dims),
        device=atom_features.device,
        dtype=atom_features.dtype,
    )
    with _deterministic_algorithms():
        token_features.scatter_add_(
            index=repeated_index.long(),
            src=atom_features,
            dim=atom_dim,
        )
    token_features = token_features.reshape((*feat_batch_dims, n_token + 1, -1))[..., :n_token, :].reshape(
        (*feat_batch_dims, n_token, *feat_dims)
    )

    if aggregate_fn == "mean":
        token_num_atoms = torch.zeros(
            (*batch_dims, n_token + 1),
            device=atom_features.device,
            dtype=torch.int32,
        ).scatter_add_(
            index=atom_to_token_index.to(torch.int64),
            src=atom_mask.to(torch.int32),
            dim=-1,
        )[..., :n_token]
        token_features = token_features / (token_num_atoms.reshape(token_num_atoms.shape + (1,) * len(feat_dims)) + eps)

    return token_features


def select_atoms_from_padded_tokens(
    atom_features: torch.Tensor,
    max_atom_per_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Compact fixed-width per-token atom blocks into packed atom rows.

    Every batch item is compacted according to ``max_atom_per_token_mask`` and
    then padded to the largest selected atom count in the batch.

    Args:
        atom_features: Features shaped
            ``[*batch, N_token * max_atoms_per_token, C]``.
        max_atom_per_token_mask: Valid-row mask for the fixed-width atom axis.

    Returns:
        Packed features shaped ``[*batch, max_valid_atoms, C]``.

    Used by:
        OpenFold3 confidence heads to remove max-atoms-per-token padding.
    """
    batch_dims = atom_features.shape[:-2]
    feature_dim = atom_features.shape[-1]
    max_atoms_in_batch = torch.max(torch.sum(max_atom_per_token_mask.int(), dim=-1))
    flat_atom_features = atom_features.reshape(-1, *atom_features.shape[-2:])
    flat_mask = max_atom_per_token_mask.reshape(-1, max_atom_per_token_mask.shape[-1])
    n_feat = flat_atom_features.shape[0]
    n_mask = flat_mask.shape[0]
    if n_feat != n_mask:
        if n_mask == 0 or n_feat % n_mask != 0:
            raise ValueError(
                "max_atom_per_token_mask batch size must divide the flattened "
                f"feature batch, got {tuple(atom_features.shape)} and "
                f"{tuple(max_atom_per_token_mask.shape)}"
            )
        # Each mask row applies contiguously across that item's sample axis.
        flat_mask = flat_mask.repeat_interleave(n_feat // n_mask, dim=0)

    def select_atoms(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compact one batch item.

        Args:
            features: Fixed-width rows shaped ``[N_slot, C]``.
            mask: Valid-row mask shaped ``[N_slot]``.

        Returns:
            Rows shaped ``[max_valid_atoms_in_batch, C]``.

        Used by:
            :func:`select_atoms_from_padded_tokens` only.
        """
        selected = torch.masked_select(features, mask[..., None].bool()).reshape(-1, feature_dim)
        return torch.nn.functional.pad(selected, (0, 0, 0, max_atoms_in_batch - selected.shape[-2]))

    if batch_dims:
        atom_features = torch.stack(
            [
                select_atoms(features, mask)
                for features, mask in zip(
                    torch.unbind(flat_atom_features, dim=0),
                    torch.unbind(flat_mask, dim=0),
                    strict=True,
                )
            ],
            dim=0,
        )
    else:
        atom_features = select_atoms(flat_atom_features[0], flat_mask[0])
    return atom_features.reshape(*batch_dims, -1, feature_dim)


def gather_token_features_to_atoms(
    token_features: torch.Tensor,
    atom_to_token_index: torch.Tensor,
) -> torch.Tensor:
    """Gather indexed token features into atom rows.

    Ownership batch dimensions must prefix the feature batch dimensions. This
    lets a mapping shaped ``[B, N_atom]`` broadcast over features shaped
    ``[B, S, N_token, C]`` without materializing ``S`` copies of the mapping.

    Args:
        token_features: Features shaped
            ``[*batch, *samples, N_token, C]``.
        atom_to_token_index: Integer owners shaped ``[*batch, N_atom]``.

    Returns:
        Features shaped ``[*batch, *samples, N_atom, C]``.

    Raises:
        ValueError: If ownership batch dimensions do not prefix feature batch
            dimensions.

    Used by:
        Protenix atom-attention encoder and decoder token-to-atom broadcasts.
    """
    feature_batch_shape = tuple(token_features.shape[:-2])
    index_batch_shape = tuple(atom_to_token_index.shape[:-1])
    if feature_batch_shape[: len(index_batch_shape)] != index_batch_shape:
        raise ValueError(
            "atom_to_token_index batch dimensions must prefix token_features, "
            f"got {tuple(token_features.shape)} and "
            f"{tuple(atom_to_token_index.shape)}"
        )
    extra_batch_dims = len(feature_batch_shape) - len(index_batch_shape)
    index = atom_to_token_index.reshape(
        *index_batch_shape,
        *((1,) * extra_batch_dims),
        atom_to_token_index.shape[-1],
        1,
    ).expand(*feature_batch_shape, atom_to_token_index.shape[-1], token_features.shape[-1])
    return torch.gather(token_features, -2, index.long())


def aggregate_indexed_atom_features(
    atom_features: torch.Tensor,
    atom_to_token_index: torch.Tensor,
    num_tokens: int,
    *,
    atom_mask: torch.Tensor | None = None,
    count_mask: torch.Tensor | None = None,
    aggregate_fn: Literal["mean", "sum"] = "mean",
    deterministic: bool,
) -> torch.Tensor:
    """Reduce indexed atom rows while keeping mask semantics explicit.

    ``atom_mask`` controls which values enter the sum, while ``count_mask``
    independently controls the denominator of a mean. If ``count_mask`` is
    ``None``, every atom row is counted—even rows zeroed by ``atom_mask``.
    This distinction preserves differing OF3 and Protenix reduction contracts.

    Ownership can omit extra sample dimensions carried by ``atom_features``;
    the index and masks are broadcast over those dimensions. The caller must
    explicitly choose deterministic scatter behavior because CUDA
    determinism can affect both performance and iterative diffusion results.

    Args:
        atom_features: Features shaped
            ``[*batch, *samples, N_atom, C]``.
        atom_to_token_index: Integer owners shaped ``[*batch, N_atom]``.
        num_tokens: Number of rows in the token output.
        atom_mask: Optional mask for values entering the reduction.
        count_mask: Optional, independent mask for mean counts. ``None`` counts
            all atom rows.
        aggregate_fn: Either ``"sum"`` or ``"mean"``.
        deterministic: Whether to enable deterministic algorithms around the
            floating-point scatter.

    Returns:
        Features shaped ``[*batch, *samples, N_token, C]``.

    Raises:
        ValueError: If shapes are incompatible or ``aggregate_fn`` is invalid.

    Used by:
        Protenix atom-attention encoder when reducing atom activations back to
        token activations.
    """
    feature_batch_shape = tuple(atom_features.shape[:-2])
    index_batch_shape = tuple(atom_to_token_index.shape[:-1])
    if feature_batch_shape[: len(index_batch_shape)] != index_batch_shape:
        raise ValueError(
            "atom_to_token_index batch dimensions must prefix atom_features, "
            f"got {tuple(atom_features.shape)} and "
            f"{tuple(atom_to_token_index.shape)}"
        )
    if atom_features.shape[-2] != atom_to_token_index.shape[-1]:
        raise ValueError(
            "atom_features and atom_to_token_index atom dimensions must match, "
            f"got {atom_features.shape[-2]} and "
            f"{atom_to_token_index.shape[-1]}"
        )
    if aggregate_fn not in ("mean", "sum"):
        raise ValueError(f"Invalid aggregation function: {aggregate_fn}")

    extra_batch_dims = len(feature_batch_shape) - len(index_batch_shape)
    expanded_index = atom_to_token_index.reshape(
        *index_batch_shape,
        *((1,) * extra_batch_dims),
        atom_to_token_index.shape[-1],
    ).expand(*feature_batch_shape, atom_to_token_index.shape[-1])
    index = expanded_index.unsqueeze(-1).expand_as(atom_features).long()
    source = atom_features
    if atom_mask is not None:
        atom_mask = _expand_atom_metadata(
            atom_mask,
            atom_to_token_index,
            feature_batch_shape,
            "atom_mask",
        )
        source = source * atom_mask.unsqueeze(-1).to(source.dtype)

    output = atom_features.new_zeros(*feature_batch_shape, num_tokens, atom_features.shape[-1])
    context = _deterministic_algorithms() if deterministic else contextlib.nullcontext()
    with context:
        output.scatter_add_(-2, index, source)
    if aggregate_fn == "sum":
        return output

    counts = atom_features.new_zeros(*feature_batch_shape, num_tokens, 1)
    if count_mask is None:
        count_source = atom_features.new_ones(*atom_features.shape[:-1], 1)
    else:
        count_mask = _expand_atom_metadata(
            count_mask,
            atom_to_token_index,
            feature_batch_shape,
            "count_mask",
        )
        count_source = count_mask.unsqueeze(-1).to(atom_features.dtype)
    counts.scatter_add_(-2, expanded_index.unsqueeze(-1).long(), count_source)
    return output / counts.clamp(min=1)


def _expand_atom_metadata(
    metadata: torch.Tensor,
    atom_to_token_index: torch.Tensor,
    feature_batch_shape: tuple[int, ...],
    name: str,
) -> torch.Tensor:
    """Broadcast per-atom metadata over feature-only sample dimensions.

    Args:
        metadata: Tensor shaped like ``atom_to_token_index`` or already
            expanded to ``[*feature_batch, N_atom]``.
        atom_to_token_index: Ownership tensor defining base batch and atom axes.
        feature_batch_shape: Desired leading dimensions after expansion.
        name: Metadata name included in shape errors.

    Returns:
        A broadcast view shaped ``[*feature_batch, N_atom]``.

    Raises:
        ValueError: If ``metadata`` matches neither accepted shape.

    Used by:
        :func:`aggregate_indexed_atom_features` for Protenix value/count masks
        and any future indexed atom layout with sample axes.
    """
    base_shape = tuple(atom_to_token_index.shape)
    expanded_shape = (*feature_batch_shape, atom_to_token_index.shape[-1])
    if metadata.shape == expanded_shape:
        return metadata
    if metadata.shape != base_shape:
        raise ValueError(f"{name} must have shape {base_shape} or {expanded_shape}, got {tuple(metadata.shape)}")
    extra_batch_dims = len(feature_batch_shape) - len(base_shape) + 1
    return metadata.reshape(
        *base_shape[:-1],
        *((1,) * extra_batch_dims),
        base_shape[-1],
    ).expand(*expanded_shape)
