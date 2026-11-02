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

"""Fused sequence-to-pair injection with masking.

Supports both RFProteína pair-write modes::

    sum:       pair[b,i,j,:] + row[b,j,:] + column[b,i,:]
    hadamard:  pair[b,i,j,:] + row[b,j,:] * column[b,i,:]

The kernel performs one pair-tensor pass and is bit-identical to the eager
operations at valid positions. Invalid positions are exact zero.
"""

from collections.abc import Mapping
from contextlib import nullcontext
from types import MappingProxyType
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from bionemo_ir.dsl_kernels.triton_cache import CachedKernel, TritonKernelCache
from bionemo_ir.utils import get_sm_version

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "token_stride_b",
        "token_stride_n",
    ]
)
def _fused_pair_injection_kernel(
    out_ptr,  # [B, N, N, C]  contiguous
    pair_ptr,  # [B, N, N, C]  contiguous
    row_ptr,  # [B, N, C]     C innermost, b/n strides passed below
    column_ptr,  # [B, N, C]     same strides as row_ptr
    mask_ptr,  # [B, N, N]     contiguous
    num_tokens,  # C is PAIR_DIM; B is the slowest grid axis
    # row and column are halves of one chunked projection, so n stride is 2*C.
    token_stride_b,
    token_stride_n,
    HADAMARD: tl.constexpr,
    PAIR_DIM: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_TILES: tl.constexpr,
):
    # Tile index leads so consecutive blocks walk one (b, i) plane in address
    # order, keeping that plane's column and mask operands hot.
    pid_tile = tl.program_id(0)
    i_idx = tl.program_id(1).to(tl.int64)
    b_idx = tl.program_id(2).to(tl.int64)

    pid_j = pid_tile // D_TILES
    pid_d = pid_tile % D_TILES

    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)  # [BLOCK_J]
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]
    mask_j = offs_j < num_tokens
    mask_d = offs_d < PAIR_DIM
    tile_mask = mask_j[:, None] & mask_d[None, :]

    # i64: B * N * N * C crosses INT32_MAX at N=1024, C=384, B=8.
    tokens = num_tokens.to(tl.int64)
    plane_stride = tokens * PAIR_DIM
    pair_base = b_idx * tokens * plane_stride + i_idx * plane_stride
    pair_offs = pair_base + offs_j[:, None].to(tl.int64) * PAIR_DIM + offs_d[None, :]

    token_base = b_idx * token_stride_b
    row_offs = token_base + offs_j[:, None].to(tl.int64) * token_stride_n + offs_d[None, :]
    column_offs = token_base + i_idx * token_stride_n + offs_d

    mask_offs = b_idx * tokens * tokens + i_idx * tokens + offs_j

    pair_tile = tl.load(pair_ptr + pair_offs, mask=tile_mask, other=0.0)
    row_tile = tl.load(row_ptr + row_offs, mask=tile_mask, other=0.0)
    column_vec = tl.load(column_ptr + column_offs, mask=mask_d, other=0.0)
    mask_vec = tl.load(mask_ptr + mask_offs, mask=mask_j, other=0.0)

    # Torch computes these in fp32 and rounds on every store, so rounding each
    # intermediate in the same order is what keeps the result bit-identical.
    dtype = pair_tile.dtype
    column_tile = tl.broadcast_to(column_vec[None, :], row_tile.shape).to(tl.float32)
    accumulated = pair_tile.to(tl.float32)
    if HADAMARD:
        # pair + row * column, as asm because LLVM otherwise contracts the
        # multiply into an FMA and rounds once where torch rounds twice.
        product = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [row_tile.to(tl.float32), column_tile],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        ).to(dtype)
        accumulated = (accumulated + product.to(tl.float32)).to(dtype)
    else:
        # (pair + row).add_(column)
        accumulated = (accumulated + row_tile.to(tl.float32)).to(dtype)
        accumulated = (accumulated.to(tl.float32) + column_tile).to(dtype)
    result = tl.where(mask_vec[:, None] != 0, accumulated, 0.0).to(dtype)

    tl.store(out_ptr + pair_offs, result, mask=tile_mask)


# ---------------------------------------------------------------------------
# Tile-shape heuristic
# ---------------------------------------------------------------------------

#: H100 BF16 tuning for released pair widths; see
#: ``bench_pair_injection_triton.py --sweep-blocks``.
_BLOCK_J_DEFAULT = 16
_BLOCK_D_CANDIDATES = (128, 256, 64)


class _LaunchConfig(NamedTuple):
    """Launch constants for one pair width and SM version."""

    block_j: int
    block_d: int
    num_warps: int | None
    num_stages: int | None


class _CompiledLaunch(NamedTuple):
    """Device-specific launch state kept local from selection through launch."""

    config: _LaunchConfig
    d_tiles: int
    kernels: Mapping[torch.dtype, CachedKernel]
    cache_key: tuple


#: Exact-SM tuning for released pair widths. Unknown architectures, including
#: SM90, retain the H100 defaults above.
_LAUNCH_TUNING: dict[tuple[int, int], _LaunchConfig] = {
    (100, 384): _LaunchConfig(block_j=32, block_d=128, num_warps=4, num_stages=3),
    (103, 384): _LaunchConfig(block_j=32, block_d=128, num_warps=4, num_stages=3),
}


def _pick_block_d(pair_dim: int) -> int:
    """Pick the feature tile that leaves the fewest lanes masked off.

    A tile that does not divide ``pair_dim`` leaves idle lanes in the last tile,
    so rank candidates by that waste and let the measured order break ties.
    """
    candidates = [block_d for block_d in _BLOCK_D_CANDIDATES if block_d <= pair_dim]
    if not candidates:
        return triton.next_power_of_2(pair_dim)
    return min(candidates, key=lambda block_d: triton.cdiv(pair_dim, block_d) * block_d - pair_dim)


def _pick_launch_config(pair_dim: int, sm_version: int) -> _LaunchConfig:
    """Return exact-SM tuning, or the unchanged H100-compatible defaults."""
    tuned = _LAUNCH_TUNING.get((sm_version, pair_dim))
    if tuned is not None:
        return tuned
    return _LaunchConfig(
        block_j=_BLOCK_J_DEFAULT,
        block_d=_pick_block_d(pair_dim),
        num_warps=None,
        num_stages=None,
    )


def _resolve_launch_config(
    pair_dim: int,
    block_j: int | None,
    block_d: int | None,
    sm_version: int | None = None,
) -> _LaunchConfig:
    """Apply explicit diagnostic overrides, otherwise select exact-SM tuning."""
    if block_j is not None or block_d is not None:
        return _LaunchConfig(
            block_j=block_j if block_j is not None else _BLOCK_J_DEFAULT,
            block_d=block_d if block_d is not None else _pick_block_d(pair_dim),
            num_warps=None,
            num_stages=None,
        )
    return _pick_launch_config(pair_dim, get_sm_version() if sm_version is None else sm_version)


def _resolve_device_launch_config(
    pair_dim: int,
    block_j: int | None,
    block_d: int | None,
    device: torch.device | int,
) -> _LaunchConfig:
    """Resolve launch constants against the device that will execute them."""
    return _resolve_launch_config(pair_dim, block_j, block_d, get_sm_version(device))


def _launch_options(config: _LaunchConfig) -> dict[str, int]:
    """Triton launch keywords, omitted on architectures whose defaults stay unchanged."""
    options = {}
    if config.num_warps is not None:
        options["num_warps"] = config.num_warps
    if config.num_stages is not None:
        options["num_stages"] = config.num_stages
    return options


def _grid(num_tokens: int, pair_dim: int, batch_size: int, block_j: int, block_d: int) -> tuple[int, int, int]:
    tiles = triton.cdiv(num_tokens, block_j) * triton.cdiv(pair_dim, block_d)
    return (tiles, num_tokens, batch_size)


def _check_operands(
    pair: torch.Tensor,
    row: torch.Tensor,
    column: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[int, int, int]:
    """Validate the layout contract and return ``(batch, tokens, pair_dim)``."""
    if pair.ndim != 4:
        raise ValueError(f"pair must be [B, N, N, C], got shape {tuple(pair.shape)}")
    batch, tokens, tokens_2, pair_dim = pair.shape
    if tokens != tokens_2:
        raise ValueError(f"pair must be square in its token axes, got {tuple(pair.shape)}")
    if not pair.is_contiguous():
        raise ValueError("pair must be contiguous")
    if row.shape != (batch, tokens, pair_dim) or column.shape != row.shape:
        raise ValueError(
            f"row and column must both be [B, N, C] = {(batch, tokens, pair_dim)}, "
            f"got {tuple(row.shape)} and {tuple(column.shape)}"
        )
    if row.stride() != column.stride():
        raise ValueError(f"row and column must share strides, got {row.stride()} and {column.stride()}")
    if row.stride(-1) != 1 or column.stride(-1) != 1:
        raise ValueError("row and column must have a contiguous feature axis")
    if mask.shape not in ((batch, tokens, tokens), (batch, tokens, tokens, 1)):
        raise ValueError(f"mask must be [B, N, N] or [B, N, N, 1], got {tuple(mask.shape)}")
    if not mask.is_contiguous():
        raise ValueError("mask must be contiguous")
    if not (pair.dtype == row.dtype == column.dtype == mask.dtype):
        raise ValueError(
            f"all operands must share a dtype, got {pair.dtype}, {row.dtype}, {column.dtype}, {mask.dtype}"
        )
    if not (pair.get_device() == row.get_device() == column.get_device() == mask.get_device()):
        raise ValueError(
            f"all operands must share a device, got {pair.device}, {row.device}, {column.device}, {mask.device}"
        )
    return batch, tokens, pair_dim


# ---------------------------------------------------------------------------
# Class-based API — pre-compile, fast dispatch via cuda.bindings
# ---------------------------------------------------------------------------


class FusedPairInjection(TritonKernelCache):
    """Fused pair injection with a pre-compiled kernel and driver launch.

    Compiles once per ``(device, SM, pair_dim, hadamard, dtype, block shape)`` across
    the common dtypes so each call is a bare launch, sharing compilations
    between instances through a class-level cache. ``num_tokens`` and the batch
    size stay runtime values, so one instance serves every shape a rollout
    visits.
    """

    _global_cache: dict[tuple, dict] = {}

    def __init__(
        self,
        pair_dim: int,
        hadamard: bool,
        block_j: int | None = None,
        block_d: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Create a cached pair-injection kernel.

        Args:
            pair_dim: Pair representation width.
            hadamard: Multiply row and column representations instead of adding them.
            block_j: Optional token-axis tile override.
            block_d: Optional feature-axis tile override.
            dtype: Initial compilation dtype, alongside FP32, BF16 and FP16.

        Setting either tile override bypasses exact-SM tuning. An unspecified
        tile then uses its H100-compatible default.
        """
        self.pair_dim = pair_dim
        self.hadamard = hadamard
        self._block_j_override = block_j
        self._block_d_override = block_d
        # Each launch is immutable and read once per call, so concurrent calls on
        # different devices never mix one device's constants with another's kernels.
        self._launches: dict[int, _CompiledLaunch] = {}
        self._last_launch: _CompiledLaunch | None = None
        self._dtype = dtype

        if torch.cuda.is_available():
            self._ensure_compiled(torch.cuda.current_device())

    def _ensure_compiled(self, device: torch.device | int) -> _CompiledLaunch:
        device_index = device if isinstance(device, int) else device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        launch = self._launches.get(device_index)
        if launch is None:
            launch = self._compile_launch(device_index)
            self._launches[device_index] = launch
        self._last_launch = launch
        return launch

    def _compile_launch(self, device_index: int) -> _CompiledLaunch:
        sm_version = get_sm_version(device_index)
        config = _resolve_launch_config(
            self.pair_dim,
            self._block_j_override,
            self._block_d_override,
            sm_version,
        )
        cache_key = (
            device_index,
            sm_version,
            self.pair_dim,
            self.hadamard,
            self._dtype,
            config.block_j,
            config.block_d,
            config.num_warps,
            config.num_stages,
        )
        d_tiles = triton.cdiv(self.pair_dim, config.block_d)
        cached = FusedPairInjection._global_cache.get(cache_key)
        if cached is not None:
            return _CompiledLaunch(config, d_tiles, MappingProxyType(dict(cached)), cache_key)

        pair_dim = self.pair_dim
        with torch.cuda.device(device_index):
            result = self.compile_for_dtypes(
                _fused_pair_injection_kernel,
                dtypes=[self._dtype, torch.float32, torch.bfloat16, torch.float16],
                make_dummy_args=lambda dt: (
                    torch.empty(1, 1, 1, pair_dim, dtype=dt, device="cuda"),
                    torch.empty(1, 1, 1, pair_dim, dtype=dt, device="cuda"),
                    torch.empty(1, 1, pair_dim, dtype=dt, device="cuda"),
                    torch.empty(1, 1, pair_dim, dtype=dt, device="cuda"),
                    torch.empty(1, 1, 1, dtype=dt, device="cuda"),
                    1,
                    pair_dim,
                    pair_dim,
                ),
                grid=(1, 1, 1),
                HADAMARD=self.hadamard,
                PAIR_DIM=pair_dim,
                BLOCK_J=config.block_j,
                BLOCK_D=config.block_d,
                D_TILES=d_tiles,
                **_launch_options(config),
            )

        kernels = FusedPairInjection._global_cache.setdefault(cache_key, result)
        return _CompiledLaunch(config, d_tiles, MappingProxyType(dict(kernels)), cache_key)

    def __call__(
        self,
        pair: torch.Tensor,
        row: torch.Tensor,
        column: torch.Tensor,
        mask: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inject ``row`` and ``column`` into ``pair`` and re-apply ``mask``.

        Args:
            pair: Contiguous ``[B, N, N, C]`` pair representation.
            row: ``[B, N, C]`` projection indexed by the second token axis.
            column: ``[B, N, C]`` projection indexed by the first token axis,
                sharing ``row``'s strides.
            mask: Contiguous ``[B, N, N]`` or ``[B, N, N, 1]`` mask of 0 or 1,
                in the operand dtype.
            output: Optional pre-allocated ``[B, N, N, C]`` buffer.

        Returns:
            The injected pair representation.
        """
        batch, tokens, pair_dim = _check_operands(pair, row, column, mask)
        if pair_dim != self.pair_dim:
            raise ValueError(f"pair width {pair_dim} does not match the compiled width {self.pair_dim}")
        device_index = pair.get_device()
        switch = device_index != torch.cuda.current_device()
        with torch.cuda.device(device_index) if switch else nullcontext():
            launch = self._ensure_compiled(device_index)

            out = output if output is not None else torch.empty_like(pair)
            if out.shape != pair.shape or out.get_device() != device_index or not out.is_contiguous():
                raise ValueError("output must be a contiguous tensor shaped like pair on the same device")

            config = launch.config
            grid = _grid(tokens, pair_dim, batch, config.block_j, config.block_d)
            kernel = launch.kernels[pair.dtype]
            drv = kernel.driver
            if drv is not None:
                values = (
                    out.data_ptr(),
                    pair.data_ptr(),
                    row.data_ptr(),
                    column.data_ptr(),
                    mask.data_ptr(),
                    tokens,
                    row.stride(0),
                    row.stride(1),
                )
                drv.launch_with(values, *grid)
            else:
                kernel.launch(
                    grid,
                    out,
                    pair,
                    row,
                    column,
                    mask,
                    tokens,
                    row.stride(0),
                    row.stride(1),
                    self.hadamard,
                    pair_dim,
                    config.block_j,
                    config.block_d,
                    launch.d_tiles,
                )
        return out


# ---------------------------------------------------------------------------
# Functional API (convenience wrapper)
# ---------------------------------------------------------------------------


def fused_pair_injection(
    pair: torch.Tensor,
    row: torch.Tensor,
    column: torch.Tensor,
    mask: torch.Tensor,
    hadamard: bool,
    output: torch.Tensor | None = None,
    block_j: int | None = None,
    block_d: int | None = None,
) -> torch.Tensor:
    """Fused sequence-to-pair injection with masking.

    Compiles per call. For hot loops prefer :class:`FusedPairInjection`, whose
    ``__call__`` documents the operands.

    Args:
        pair: Contiguous ``[B, N, N, C]`` pair representation.
        row: ``[B, N, C]`` projection indexed by the second token axis.
        column: ``[B, N, C]`` projection indexed by the first token axis.
        mask: Contiguous ``[B, N, N]`` or ``[B, N, N, 1]`` mask of 0 or 1.
        hadamard: Multiply ``row`` and ``column`` instead of adding both.
        output: Optional pre-allocated ``[B, N, N, C]`` buffer.
        block_j: Optional token-axis tile override.
        block_d: Optional feature-axis tile override.

    Returns:
        The injected pair representation.
    """
    batch, tokens, pair_dim = _check_operands(pair, row, column, mask)
    out = output if output is not None else torch.empty_like(pair)
    if out.shape != pair.shape or out.device != pair.device or not out.is_contiguous():
        raise ValueError("output must be a contiguous tensor shaped like pair on the same device")
    if tokens == 0:
        return out

    with torch.cuda.device(pair.device):
        config = _resolve_device_launch_config(pair_dim, block_j, block_d, pair.device)
        grid = _grid(tokens, pair_dim, batch, config.block_j, config.block_d)
        _fused_pair_injection_kernel[grid](
            out,
            pair,
            row,
            column,
            mask,
            tokens,
            row.stride(0),
            row.stride(1),
            HADAMARD=hadamard,
            PAIR_DIM=pair_dim,
            BLOCK_J=config.block_j,
            BLOCK_D=config.block_d,
            D_TILES=triton.cdiv(pair_dim, config.block_d),
            **_launch_options(config),
        )
    return out
