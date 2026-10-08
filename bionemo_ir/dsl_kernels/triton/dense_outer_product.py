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
"""Bounded two-stage half-precision outer product projection."""

from functools import cache

import torch
import triton
import triton.language as tl

from bionemo_ir.dsl_kernels.triton_cache import CachedKernel, TritonKernelCache

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_CZ = (128, 256)
# Rows of ``a`` per contraction; bounds the intermediate to ``[_CHUNK, J, 1024]``.
_CHUNK = 256
_BLOCK = 128
_BLOCK_K = 64
# Sequence offsets at or past this many elements need 64-bit indexing.
_INDEX_LIMIT = 2**31


@triton.jit(do_not_specialize=["OFFSET", "S", "I", "J", "ROWS"])
def opm_contract(
    A,
    B,
    Z,
    OFFSET: tl.int32,
    S: tl.int32,
    I: tl.int32,
    J: tl.int32,
    ROWS: tl.int32,
    WIDE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Write ``Z[r, j, c * 32 + d] = sum_s A[s, OFFSET + r, c] * B[s, j, d]`` for ``r < ROWS``."""
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    if WIDE:
        k = k.to(tl.int64)
    acc = tl.zeros((BM, BN), tl.float32)
    # Whole sequence tiles skip the sequence mask; the tail tile applies it.
    for start in range(S // BK):
        rows = start * BK + k
        a = tl.load(A + rows[None, :] * I * 32 + (m[:, None] + OFFSET * 32), m[:, None] < ROWS * 32, other=0)
        b = tl.load(B + rows[:, None] * J * 32 + n[None, :], n[None, :] < J * 32, other=0)
        acc = tl.dot(a, b, acc)
    if S % BK != 0:
        rows = S // BK * BK + k
        a = tl.load(
            A + rows[None, :] * I * 32 + (m[:, None] + OFFSET * 32),
            (rows[None, :] < S) & (m[:, None] < ROWS * 32),
            other=0,
        )
        b = tl.load(B + rows[:, None] * J * 32 + n[None, :], (rows[:, None] < S) & (n[None, :] < J * 32), other=0)
        acc = tl.dot(a, b, acc)
    pairs = (m[:, None] // 32 * J + n[None, :] // 32).to(tl.int64)
    offsets = pairs * 1024 + m[:, None] % 32 * 32 + n[None, :] % 32
    tl.store(Z + offsets, acc, (m[:, None] < ROWS * 32) & (n[None, :] < J * 32))


@triton.jit(do_not_specialize=["OFFSET", "J", "ROWS"])
def opm_project(
    Z,
    W,
    MASK,
    BIAS,
    OUT,
    OFFSET: tl.int32,
    J: tl.int32,
    ROWS: tl.int32,
    CZ: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BEFORE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Project ``ROWS * J`` intermediate rows to ``CZ`` channels and normalize by ``MASK``."""
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    rows = m.to(tl.int64) * 1024
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(1024 // BK):
        cols = start * BK + k
        z = tl.load(Z + rows[:, None] + cols[None, :], m[:, None] < ROWS * J, other=0)
        w = tl.load(W + n[None, :] * 1024 + cols[:, None])
        acc = tl.dot(z, w, acc)
    pairs = OFFSET.to(tl.int64) * J + m
    mask = tl.load(MASK + pairs, m < ROWS * J, other=1)
    inv = tl.div_rn(1.0, mask)
    if BEFORE:
        if HAS_BIAS:
            acc = tl.fma(acc, inv[:, None], tl.load(BIAS + n)[None, :].to(tl.float32))
        else:
            acc = acc * inv[:, None]
    else:
        if HAS_BIAS:
            acc = acc + tl.load(BIAS + n)[None, :].to(tl.float32)
        acc = acc * inv[:, None]
    tl.store(OUT + pairs[:, None] * CZ + n[None, :], acc, m[:, None] < ROWS * J)


class FusedOPM(TritonKernelCache):
    """Contraction and projection kernels for one dtype, output width, epilogue, and index width."""

    def __init__(self, dtype: torch.dtype, c_z: int, has_bias: bool, before: bool, wide: bool) -> None:
        self.contract = self.compile_for_dtypes(
            opm_contract,
            dtypes=[dtype],
            make_dummy_args=lambda dtype: (
                *(torch.empty(1, device="cuda", dtype=dtype) for _ in range(3)),
                2,
                2,
                2,
                2,
                2,
            ),
            grid=(0,),
            WIDE=wide,
            BM=_BLOCK,
            BN=_BLOCK,
            BK=_BLOCK_K,
            num_warps=8,
            num_stages=3,
        )[dtype]
        self.project = self.compile_for_dtypes(
            opm_project,
            dtypes=[dtype],
            make_dummy_args=lambda dtype: (
                torch.empty(1, device="cuda", dtype=dtype),
                torch.empty(1, device="cuda", dtype=dtype),
                torch.empty(1, device="cuda", dtype=torch.float32),
                torch.empty(1, device="cuda", dtype=dtype),
                torch.empty(1, device="cuda", dtype=dtype),
                2,
                2,
                2,
            ),
            grid=(0,),
            CZ=c_z,
            HAS_BIAS=has_bias,
            BEFORE=before,
            BM=_BLOCK,
            BN=_BLOCK,
            BK=_BLOCK_K,
            num_warps=8,
            num_stages=3,
            enable_fp_fusion=False,
        )[dtype]


@cache
def _cached_kernels(device: int, dtype: torch.dtype, c_z: int, has_bias: bool, before: bool, wide: bool) -> FusedOPM:
    with torch.cuda.device(device):
        return FusedOPM(dtype, c_z, has_bias, before, wide)


def _aligned(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.contiguous()
    # Cached CUBINs assume aligned tensor pointers.
    return tensor.clone() if tensor.data_ptr() % 16 else tensor


def _launch(
    kernel: CachedKernel,
    grid: tuple[int, int],
    tensors: tuple[torch.Tensor, ...],
    scalars: tuple[int, ...],
    constexprs: tuple[int | bool, ...],
) -> None:
    if kernel.driver is not None:
        kernel.driver.launch_with((*(tensor.data_ptr() for tensor in tensors), *scalars), *grid)
    else:
        kernel.launch(grid, *tensors, *scalars, *constexprs)


def dense_outer_product(
    a: torch.Tensor,
    b: torch.Tensor,
    num_mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    norm_before: bool,
) -> torch.Tensor:
    """Contract half-precision ``[B, S, I, 32]`` and ``[B, S, J, 32]`` inputs and project to ``C_z`` channels.

    An intermediate in the input dtype preserves the fused kernel's rounding boundary.

    Args:
        a: Left input ``[B, S, I, 32]``, BF16 or FP16.
        b: Right input ``[B, S, J, 32]`` in ``a``'s dtype.
        num_mask: Normalization counts ``[B, I, J]``, read as FP32.
        weight: Projection ``[C_z, 1024]`` in ``a``'s dtype, with ``C_z`` 128 or 256.
        bias: Optional projection bias ``[C_z]`` in ``a``'s dtype.
        norm_before: Normalize before adding the bias instead of after.

    Returns:
        ``[B, I, J, C_z]`` in ``a``'s dtype.

    Raises:
        ValueError: The operands fall outside this contract.
    """
    batch, sequences, rows_i, channels = a.shape
    cols_j = b.shape[2]
    c_z = weight.shape[0]
    operands = (a, b, weight) if bias is None else (a, b, weight, bias)
    if (
        not a.is_cuda
        or channels != 32
        or b.shape != (batch, sequences, cols_j, 32)
        or c_z not in _SUPPORTED_CZ
        or weight.shape != (c_z, 1024)
        or (bias is not None and bias.shape != (c_z,))
        or num_mask.shape != (batch, rows_i, cols_j)
        or a.dtype not in _SUPPORTED_DTYPES
        or any(tensor.dtype != a.dtype for tensor in operands)
        or any(tensor.device != a.device for tensor in (*operands, num_mask))
    ):
        raise ValueError("dense_outer_product takes BF16 or FP16 [B, S, I, 32] x [B, S, J, 32] with C_z in (128, 256)")
    a, b = _aligned(a), _aligned(b)
    num_mask = num_mask.to(torch.float32).contiguous()
    weight = _aligned(weight)
    has_bias = bias is not None
    bias = _aligned(bias) if has_bias else weight
    output = torch.empty((batch, rows_i, cols_j, c_z), device=a.device, dtype=a.dtype)
    if output.numel() == 0:
        return output
    chunk = min(_CHUNK, rows_i)
    wide = sequences * max(rows_i, cols_j) * 32 >= _INDEX_LIMIT
    with torch.cuda.device(a.device):
        kernels = _cached_kernels(a.device.index, a.dtype, c_z, has_bias, norm_before, wide)
        intermediate = torch.empty((chunk, cols_j, 1024), device=a.device, dtype=a.dtype)
        for index in range(batch):
            contract_args = (a[index], b[index], intermediate)
            project_args = (intermediate, weight, _aligned(num_mask[index]), bias, output[index])
            for offset in range(0, rows_i, chunk):
                rows = min(chunk, rows_i - offset)
                _launch(
                    kernels.contract,
                    (triton.cdiv(rows * 32, _BLOCK), triton.cdiv(cols_j * 32, _BLOCK)),
                    contract_args,
                    (offset, sequences, rows_i, cols_j, rows),
                    (wide, _BLOCK, _BLOCK, _BLOCK_K),
                )
                _launch(
                    kernels.project,
                    (triton.cdiv(rows * cols_j, _BLOCK), c_z // _BLOCK),
                    project_args,
                    (offset, cols_j, rows),
                    (c_z, has_bias, norm_before, _BLOCK, _BLOCK, _BLOCK_K),
                )
    return output
