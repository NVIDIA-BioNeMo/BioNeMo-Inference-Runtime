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

"""Deterministic, per-token atom reduction without scatter atomics.

Sizes and strides are 64-bit runtime arguments: one CUBIN per slot bucket and
access width serves every batch, sample, atom, token, and channel count.
"""

from functools import cache

import torch
import triton
import triton.language as tl

from bionemo_ir.dsl_kernels.triton_cache import CachedKernel, TritonKernelCache

# Bound verified against deterministic scatter.
MAX_ORDERED_ATOMS = 31
_BLOCK_CHANNELS = 128
# FP32 channels per 16-byte access.
_VECTOR = 4
_SLOT_BUCKET = 8


@triton.jit(
    do_not_specialize=[
        "n_samples",
        "n_atoms",
        "n_tokens",
        "n_slots",
        "n_vectors",
        "feature_batch_stride",
        "feature_sample_stride",
        "feature_atom_stride",
        "feature_channel_stride",
        "index_batch_stride",
        "valid_batch_stride",
        "count_batch_stride",
        "count_token_stride",
        "eps",
    ]
)
def _reduce_atom_slots(
    features_ptr,
    indices_ptr,
    valid_ptr,
    counts_ptr,
    output_ptr,
    n_samples: tl.int64,
    n_atoms: tl.int64,
    n_tokens: tl.int64,
    n_slots: tl.int64,
    n_vectors: tl.int64,
    feature_batch_stride: tl.int64,
    feature_sample_stride: tl.int64,
    feature_atom_stride: tl.int64,
    feature_channel_stride: tl.int64,
    index_batch_stride: tl.int64,
    valid_batch_stride: tl.int64,
    count_batch_stride: tl.int64,
    count_token_stride: tl.int64,
    eps,
    max_slots: tl.constexpr,
    vector: tl.constexpr,
    block_channels: tl.constexpr,
):
    # Channel counts and row strides arrive in ``vector`` units, so every row
    # offset provably keeps the 16-byte alignment that wide accesses need.
    n_channels = n_vectors * vector
    channel = tl.program_id(0) * block_channels + tl.arange(0, block_channels)
    in_channels = channel < n_channels
    token = tl.program_id(1)
    sample_batch = tl.program_id(2)
    batch = sample_batch // n_samples
    sample = sample_batch % n_samples
    index_row = indices_ptr + batch * index_batch_stride + token * n_slots
    valid_row = valid_ptr + batch * valid_batch_stride + token * n_slots
    rows = features_ptr + (batch * feature_batch_stride + sample * feature_sample_stride) * vector
    if vector == 1:
        columns = channel * feature_channel_stride
    else:
        columns = channel
    total = tl.zeros((block_channels,), tl.float32)
    # Slots past n_slots add +0.0, so the sum keeps packed-atom order.
    for slot in tl.static_range(max_slots):
        in_row = slot < n_slots
        atom = tl.load(index_row + slot, mask=in_row, other=0)
        is_valid = tl.load(valid_row + slot, mask=in_row, other=0) != 0
        value = tl.load(
            rows + atom * feature_atom_stride * vector + columns,
            mask=in_channels & in_row & is_valid & (atom < n_atoms),
            other=0.0,
        )
        total = total + value
    count = tl.load(counts_ptr + batch * count_batch_stride + token * count_token_stride).to(tl.float32)
    output = output_ptr + (sample_batch * n_tokens + token) * n_channels + channel
    tl.store(output, tl.div_rn(total, count + eps), mask=in_channels)


class _AtomReduction(TritonKernelCache):
    def __init__(self, dtypes: tuple[torch.dtype, ...], max_slots: int, vector: int) -> None:
        feature_dtype, index_dtype, valid_dtype, count_dtype = dtypes
        self.kernel = self.compile_for_dtypes(
            _reduce_atom_slots,
            dtypes=[feature_dtype],
            make_dummy_args=lambda dtype: (
                torch.empty(1, device="cuda", dtype=dtype),
                torch.empty(1, device="cuda", dtype=index_dtype),
                torch.empty(1, device="cuda", dtype=valid_dtype),
                torch.empty(1, device="cuda", dtype=count_dtype),
                torch.empty(1, device="cuda", dtype=dtype),
                *(2,) * 13,
                0.0,
            ),
            # Compile without touching uninitialized dummy tensors.
            grid=(0,),
            max_slots=max_slots,
            vector=vector,
            block_channels=_BLOCK_CHANNELS,
            num_warps=_BLOCK_CHANNELS // (32 * vector),
            enable_fp_fusion=False,
        )[feature_dtype]


@cache
def _cached_reduction(device: int, dtypes: tuple[torch.dtype, ...], max_slots: int, vector: int) -> CachedKernel:
    with torch.cuda.device(device):
        return _AtomReduction(dtypes, max_slots, vector).kernel


def reduce_atom_slots(
    features: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor,
    counts: torch.Tensor,
    n_tokens: int,
    eps: float,
) -> torch.Tensor:
    """Mean-reduce validated short segments in packed-atom order.

    Each token's slots are summed sequentially in FP32 and divided by
    ``counts + eps`` with IEEE rounding, matching deterministic scatter
    followed by division.

    Args:
        features: CUDA FP32 features shaped ``[B, S, A, C]``, any strides.
        indices: Prepared packed indices shaped ``[B, 1, T * M]``.
        valid: Binary slot mask shaped ``[B, 1, T, M]``.
        counts: Integer valid-atom counts shaped ``[B, T]`` or ``[B, 1, T]``.
        n_tokens: Number of output tokens, ``T``.
        eps: Offset added to the count denominator.

    Returns:
        Token means shaped ``[B, S, T, C]``. Callers validate the layout and
        restrict ``M`` to ``MAX_ORDERED_ATOMS`` before dispatch; the slot axes
        of ``indices`` and ``valid`` must be contiguous.
    """
    batch, n_samples, n_atoms, n_channels = features.shape
    n_slots = indices.shape[-1] // n_tokens
    output = torch.empty((batch, n_samples, n_tokens, n_channels), device=features.device, dtype=features.dtype)
    if output.numel() == 0:
        return output
    counts = counts.reshape(batch, n_tokens)
    # Size-one axes never advance, so their strides cannot misalign wide accesses.
    strides = [stride if size > 1 else 0 for size, stride in zip(features.shape, features.stride(), strict=True)]
    wide = strides[3] == 1 and n_channels % _VECTOR == 0 and not any(stride % _VECTOR for stride in strides[:3])
    vector = _VECTOR if wide else 1
    scalars = (
        n_samples,
        n_atoms,
        n_tokens,
        n_slots,
        n_channels // vector,
        strides[0] // vector,
        strides[1] // vector,
        strides[2] // vector,
        strides[3],
        indices.stride(0),
        valid.stride(0),
        counts.stride(0),
        counts.stride(1),
        float(eps),
    )
    max_slots = _SLOT_BUCKET * max(1, triton.cdiv(n_slots, _SLOT_BUCKET))
    grid = (triton.cdiv(n_channels, _BLOCK_CHANNELS), n_tokens, batch * n_samples)
    tensors = (features, indices, valid, counts, output)
    with torch.cuda.device(features.device):
        # Unaligned views break the dummy-pointer alignment.
        if any(tensor.data_ptr() % 16 for tensor in tensors):
            _reduce_atom_slots[grid](
                *tensors,
                *scalars,
                max_slots=max_slots,
                vector=vector,
                block_channels=_BLOCK_CHANNELS,
                num_warps=_BLOCK_CHANNELS // (32 * vector),
                enable_fp_fusion=False,
            )
            return output
        dtypes = (features.dtype, indices.dtype, valid.dtype, counts.dtype)
        kernel = _cached_reduction(features.device.index, dtypes, max_slots, vector)
        driver = kernel.driver
        if driver is not None:
            values = (*(tensor.data_ptr() for tensor in tensors), *scalars)
            # Triton's trailing scratch pointers stay null.
            for param, value in zip(driver.params, values, strict=False):
                param.value = value
            driver.launch(*grid)
        else:
            kernel.launch(grid, *tensors, *scalars, max_slots, vector, _BLOCK_CHANNELS)
    return output
