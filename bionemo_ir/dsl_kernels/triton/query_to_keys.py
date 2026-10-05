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

"""Single-launch Triton window gather for sequence-local atom attention.

Query block ``k`` holds atoms ``[k*W, (k+1)*W)`` and its key window is
``[k*W + W/2 - H/2, k*W + W/2 + H/2)``; rows outside ``[0, K*W)`` are zero.
``query_to_keys_triton`` materializes those windows in one launch, bit-exact
to ``query_to_keys_optimized`` without its concatenate-and-index gather.

The kernel is compiled once per ``(W, H, BLOCK_D)`` for fp32, bf16 and fp16
and launched through :class:`~bionemo_ir.dsl_kernels.cache_base.DriverLauncher`
when ``cuda.bindings`` is available, otherwise through Triton's ``.run()``, so
calls skip Triton's JIT dispatch.
"""

from functools import cache

import torch
import triton
import triton.language as tl

from bionemo_ir.dsl_kernels.triton_cache import CachedKernel, TritonKernelCache

__all__ = ["query_to_keys_triton"]


@triton.jit
def _window_start(block, n_queries: tl.constexpr, n_keys: tl.constexpr):
    """First atom of query block ``block``'s key window (negative near the start)."""
    return block * n_queries + n_queries // 2 - n_keys // 2


_BLOCK_J = 32
_CACHED_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
# The cached kernel types its scalar arguments as 32-bit.
_MAX_SCALAR = 2**31 - 1


@triton.jit(do_not_specialize=["num_blocks", "stride_sb", "stride_sr", "stride_db", "stride_dr", "D"])
def _query_to_keys_kernel(
    src,
    dst,
    num_blocks,
    stride_sb,
    stride_sr,
    stride_db,
    stride_dr,
    D,
    n_queries: tl.constexpr,
    n_keys: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Copy ``BLOCK_J`` key-window rows of one block per program, zero outside ``[0, K*W)``."""
    pid = tl.program_id(0).to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    block = pid // tl.cdiv(n_keys, BLOCK_J)
    j = (pid % tl.cdiv(n_keys, BLOCK_J)) * BLOCK_J + tl.arange(0, BLOCK_J).to(tl.int64)
    rows = _window_start(block, n_queries, n_keys) + j
    row_ok = (j < n_keys) & (rows >= 0) & (rows < num_blocks * n_queries)
    src_rows = src + batch * stride_sb + rows[:, None] * stride_sr
    dst_rows = dst + batch * stride_db + (block * n_keys + j)[:, None] * stride_dr
    for d0 in range(0, D, BLOCK_D):
        cols = d0 + tl.arange(0, BLOCK_D)
        col_ok = cols < D
        x = tl.load(src_rows + cols[None, :], mask=row_ok[:, None] & col_ok[None, :], other=0.0)
        tl.store(dst_rows + cols[None, :], x, mask=(j < n_keys)[:, None] & col_ok[None, :])


class _QueryToKeysKernel(TritonKernelCache):
    """The window gather compiled for one ``(W, H, BLOCK_D)`` across the cached dtypes."""

    def __init__(self, n_queries: int, n_keys: int, block_d: int) -> None:
        self.constexprs = (n_queries, n_keys, _BLOCK_J, block_d)
        self.kernels: dict[torch.dtype, CachedKernel] = self.compile_for_dtypes(
            _query_to_keys_kernel,
            dtypes=_CACHED_DTYPES,
            make_dummy_args=lambda dtype: (
                torch.empty(1, n_queries, block_d, device="cuda", dtype=dtype),
                torch.empty(1, n_keys, block_d, device="cuda", dtype=dtype),
                1,
                n_queries * block_d,
                block_d,
                n_keys * block_d,
                block_d,
                block_d,
            ),
            grid=(0, 1),
            n_queries=n_queries,
            n_keys=n_keys,
            BLOCK_J=_BLOCK_J,
            BLOCK_D=block_d,
        )


@cache
def _query_to_keys_kernel_cache(device: int, n_queries: int, n_keys: int, block_d: int) -> _QueryToKeysKernel:
    with torch.cuda.device(device):
        return _QueryToKeysKernel(n_queries, n_keys, block_d)


def query_to_keys_triton(query: torch.Tensor, W: int | None = None, H: int | None = None) -> torch.Tensor:
    """Gather each query block's key window in one Triton launch.

    Drop-in for ``query_to_keys_optimized``: same shapes, same zero padding
    outside ``[0, K*W)``, and no precomputed gather indices.

    Args:
        query: ``[B, N, D]`` (with ``N = K*W``), ``[B, K, W, D]`` or
            ``[B, M, K, W, D]`` CUDA tensor. Non-floating inputs are cast to
            ``float32``, as in the reference.
        W: Query window size; required for 3-d input.
        H: Key window size.

    Returns:
        ``[B, K, H, D]`` for 3-d / 4-d input, ``[B, M, K, H, D]`` for 5-d input.
    """
    if H is None:
        raise ValueError("Key window size H is required")
    if not query.is_cuda:
        raise ValueError("query_to_keys_triton needs a CUDA tensor")
    if not query.is_floating_point():
        query = query.float()
    if query.ndim == 3:
        if W is None:
            raise ValueError("Query window size W is required for 3-d input")
        B, N, D = query.shape
        if N % W:
            raise ValueError(f"N={N} is not a multiple of W={W}")
        out_shape = (B, N // W, H, D)
        flat = query
    elif query.ndim == 4:
        B, K, W, D = query.shape
        out_shape = (B, K, H, D)
        flat = query.reshape(B, K * W, D)
    elif query.ndim == 5:
        B, M, K, W, D = query.shape
        out_shape = (B, M, K, H, D)
        flat = query.reshape(B * M, K * W, D)
    else:
        raise ValueError("Query tensor must be 3, 4, or 5 dimensions")
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    num_blocks = out_shape[-3]
    out = torch.empty(out_shape, dtype=query.dtype, device=query.device)
    out_flat = out.view(flat.shape[0], num_blocks * H, D)
    if out.numel() == 0:
        return out
    # Plain integer math: triton.cdiv / next_power_of_2 dispatch through the JIT on the host.
    block_d = min(1 << (D - 1).bit_length(), 128)
    grid = (num_blocks * -(-H // _BLOCK_J), flat.shape[0])
    scalars = (num_blocks, flat.stride(0), flat.stride(1), out_flat.stride(0), out_flat.stride(1), D)
    if flat.dtype not in _CACHED_DTYPES or max(scalars) > _MAX_SCALAR:
        _query_to_keys_kernel[grid](flat, out_flat, *scalars, n_queries=W, n_keys=H, BLOCK_J=_BLOCK_J, BLOCK_D=block_d)
        return out
    # The cached kernel was compiled for a 16-byte aligned source.
    if flat.data_ptr() % 16:
        flat = flat.clone()
    with torch.cuda.device(query.device):
        compiled = _query_to_keys_kernel_cache(query.device.index, W, H, block_d)
        kernel = compiled.kernels[flat.dtype]
        driver = kernel.driver
        if driver is not None:
            driver.launch_with((flat.data_ptr(), out_flat.data_ptr(), *scalars), *grid)
        else:
            kernel.launch(grid, flat, out_flat, *scalars, *compiled.constexprs)
    return out
