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

"""Inference-only fused LayerNorm/RMSNorm with layout transposes.

Normalizes the last dimension only; ``nn.LayerNorm``'s multi-axis
``normalized_shape`` is not supported.

Derived from cuEquivariance's fused LayerNorm kernels, with three changes:

1. Pick 64-bit indexing from the tiled channel extent
   ``ceil(D / TILE_D) * TILE_D``, not logical ``D``. Upstream can choose i32
   for ``D=196`` while the kernel spans a 256-wide tile and overflow on long
   pair sequences.
2. ``pad_multiple`` zero-extends the output channels so a consumer GEMM can
   take a vector-aligned K. The tile already covers that width, so the extra
   store is free.
3. Fit the whole row in one tile when possible so mean and variance come
   from one load. Wider rows tile with ``E[x^2] - E[x]^2``; neither path
   walks ``x`` a third time.
"""

from __future__ import annotations

import enum
import functools

import torch
import triton
import triton.language as tl

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class Layout(enum.IntEnum):
    BND_BND = 0
    BDN_BND = 1
    BND_BDN = 2
    DBN_BND = 3
    BND_DBN = 4


@triton.jit
def _affine(x_hat, w, b):
    """Fuse ``x_hat * w + b`` into one FMA.

    A plain multiply-add is contracted for some constexpr ``D_OUT`` widths
    and not others, so padded and unpadded stores can round the ``[0, D)``
    head differently. Keep ``D_OUT`` specialized so the store stays
    vectorized.
    """
    return tl.math.fma(x_hat, w, b)


@triton.jit
def layer_norm_transpose_forward_kernel(
    # inputs
    x_ptr,
    w_ptr,
    b_ptr,
    # outputs
    out_ptr,
    B,
    N,
    D: tl.constexpr,
    D_OUT: tl.constexpr,
    EPS: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_D: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RMS_NORM: tl.constexpr,
    LAYOUT: tl.constexpr,
    SINGLE_TILE: tl.constexpr,
    NEEDS_INT64: tl.constexpr = True,
):
    # ``D_OUT >= D`` widens the output row stride so the result is a
    # zero-extended operand for kernels that need a K aligned to a vector
    # width.  Statistics are still taken over the true ``D`` channels; the
    # ``[D, D_OUT)`` tail is written as explicit zeros.
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    if NEEDS_INT64:
        pid_n = tl.cast(pid_n, tl.int64)
        pid_b = tl.cast(pid_b, tl.int64)
        N = tl.cast(N, tl.int64)
        B = tl.cast(B, tl.int64)

    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_d = tl.arange(0, TILE_D)
    mask_n = offs_n < N

    # Layouts 1 and 3 stride the channel axis, so a channel tile advances by a
    # row of N; the others hold D contiguous.
    if LAYOUT == 1:  # bdn->bnd
        x_ptrs = x_ptr + pid_b * D * N + offs_d[None, :] * N + offs_n[:, None]
        x_step = TILE_D * N
    elif LAYOUT == 3:  # dbn->bnd
        x_ptrs = x_ptr + offs_d[None, :] * B * N + pid_b * N + offs_n[:, None]
        x_step = TILE_D * B * N
    else:  # bnd->bnd, bnd->bdn, bnd->dbn
        x_ptrs = x_ptr + pid_b * N * D + offs_n[:, None] * D + offs_d[None, :]
        x_step = TILE_D

    # Layouts 0/1/3 write D contiguously, so ``D_OUT`` is their row stride.
    # Layouts 2/4 keep D as an outer axis (the launcher pins ``D_OUT == D``
    # for them, since widening D there would not align anything).
    if LAYOUT == 2:  # bnd->bdn
        out_ptrs = out_ptr + pid_b * N * D + offs_d[None, :] * N + offs_n[:, None]
        out_step = TILE_D * N
    elif LAYOUT == 4:  # bnd->dbn
        out_ptrs = out_ptr + offs_d[None, :] * B * N + pid_b * N + offs_n[:, None]
        out_step = TILE_D * B * N
    else:
        out_ptrs = out_ptr + pid_b * N * D_OUT + offs_n[:, None] * D_OUT + offs_d[None, :]
        out_step = TILE_D

    if SINGLE_TILE:
        # The tile spans the row, so one load serves both moments and the
        # store. This is the whole reason the kernel is fast: the traffic is
        # one read and one write, which is the floor for a normalisation.
        mask_d = offs_d < D
        x = tl.load(x_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        if RMS_NORM:
            x_centred = x
        else:
            mean = tl.sum(x, axis=1) / D
            # Lanes past D loaded zero and would carry ``-mean`` into the variance.
            # Clearing them once leaves a centred row the store reuses as-is.
            x_centred = tl.where(mask_d[None, :], x - mean[:, None], 0.0)
        rstd = tl.rsqrt(tl.sum(x_centred * x_centred, axis=1) / D + EPS)
        x_hat = x_centred * rstd[:, None]
        if HAS_WEIGHT and HAS_BIAS:
            w = tl.load(w_ptr + offs_d, mask=mask_d, other=1.0).to(tl.float32)
            b = tl.load(b_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
            y = _affine(x_hat, w[None, :], b[None, :])
        elif HAS_WEIGHT:
            w = tl.load(w_ptr + offs_d, mask=mask_d, other=1.0).to(tl.float32)
            y = x_hat * w[None, :]
        elif HAS_BIAS:
            b = tl.load(b_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
            y = x_hat + b[None, :]
        else:
            y = x_hat
        # Affine defaults can reach out-of-range lanes; clear them so the
        # store can widen over the pad tail. With ``D_OUT == D`` this is a no-op.
        y = tl.where(mask_d[None, :], y, 0.0)
        tl.store(out_ptrs, y, mask=mask_n[:, None] & (offs_d < D_OUT))
    else:
        num_tiles_d = tl.cdiv(D, TILE_D)
        acc = tl.zeros([TILE_N, TILE_D], dtype=tl.float32)
        acc_sq = tl.zeros([TILE_N, TILE_D], dtype=tl.float32)
        for di in range(num_tiles_d):
            mask_d = offs_d < (D - di * TILE_D)
            x = tl.load(x_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            if not RMS_NORM:
                acc += x
            acc_sq += x * x
            x_ptrs += x_step

        if RMS_NORM:
            mean = tl.zeros([TILE_N], dtype=tl.float32)
            var = tl.sum(acc_sq, axis=1) / D
        else:
            mean = tl.sum(acc, axis=1) / D
            # ``E[x^2] - E[x]^2`` fuses what would otherwise be two loops over x.
            # Accumulation is fp32 over normalised activations, so the cancellation
            # this form is known for stays well inside the output dtype; the clamp
            # only guards against a negative rounding.
            var = tl.maximum(tl.sum(acc_sq, axis=1) / D - mean * mean, 0.0)
        rstd = tl.rsqrt(var + EPS)
        x_ptrs -= x_step * num_tiles_d

        if HAS_WEIGHT:
            w_ptrs = w_ptr + offs_d
        if HAS_BIAS:
            b_ptrs = b_ptr + offs_d

        for di in range(num_tiles_d):
            mask_d = offs_d < (D - di * TILE_D)
            x = tl.load(x_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            x_hat = (x - mean[:, None]) * rstd[:, None]
            if HAS_WEIGHT and HAS_BIAS:
                w = tl.load(w_ptrs, mask=mask_d, other=1.0).to(tl.float32)
                b = tl.load(b_ptrs, mask=mask_d, other=0.0).to(tl.float32)
                y = _affine(x_hat, w[None, :], b[None, :])
            elif HAS_WEIGHT:
                w = tl.load(w_ptrs, mask=mask_d, other=1.0).to(tl.float32)
                y = x_hat * w[None, :]
            elif HAS_BIAS:
                b = tl.load(b_ptrs, mask=mask_d, other=0.0).to(tl.float32)
                y = x_hat + b[None, :]
            else:
                y = x_hat
            if HAS_WEIGHT:
                w_ptrs += TILE_D
            if HAS_BIAS:
                b_ptrs += TILE_D
            y = tl.where(mask_d[None, :], y, 0.0)
            tl.store(out_ptrs, y, mask=mask_n[:, None] & (offs_d < (D_OUT - di * TILE_D)))
            x_ptrs += x_step
            out_ptrs += out_step


_TILE_D = 64


def _padded_channels(D: int, pad_multiple: int) -> int:
    """Round ``D`` up to ``pad_multiple``, or return ``D`` when disabled.

    ``pad_multiple`` must divide the channel tile so the widened row still
    fits inside the tiles the kernel already walks; otherwise the pad tail
    would fall outside the store loop.
    """
    if pad_multiple <= 0:
        return D
    if _TILE_D % pad_multiple != 0:
        raise ValueError(f"pad_multiple must divide the {_TILE_D}-channel tile, got {pad_multiple}")
    return triton.cdiv(D, pad_multiple) * pad_multiple


def _allocate_output(
    x: torch.Tensor,
    layout: Layout,
    pad_multiple: int = -1,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, int, int, int, int]:
    out_dtype = x.dtype if out_dtype is None else out_dtype
    if layout == Layout.BND_BND:
        B, N, D = x.shape
        D_OUT = _padded_channels(D, pad_multiple)
        out = torch.empty((B, N, D_OUT), dtype=out_dtype, device=x.device)
    elif layout == Layout.BDN_BND:
        B, D, N = x.shape
        D_OUT = _padded_channels(D, pad_multiple)
        out = torch.empty((B, N, D_OUT), dtype=out_dtype, device=x.device)
    elif layout == Layout.BND_BDN:
        B, N, D = x.shape
        D_OUT = D
        out = torch.empty((B, D, N), dtype=out_dtype, device=x.device)
    elif layout == Layout.DBN_BND:
        D, B, N = x.shape
        D_OUT = _padded_channels(D, pad_multiple)
        out = torch.empty((B, N, D_OUT), dtype=out_dtype, device=x.device)
    elif layout == Layout.BND_DBN:
        B, N, D = x.shape
        D_OUT = D
        out = torch.empty((D, B, N), dtype=out_dtype, device=x.device)
    else:
        raise ValueError(f"unsupported layout {layout}")

    return out, B, N, D, D_OUT


def _needs_int64(B: int, N: int, D: int, tile_d: int = 64) -> bool:
    """Select i64 using the largest tiled channel offset."""
    tiled_D = triton.cdiv(D, tile_d) * tile_d
    return B * N * tiled_D >= 2**31 - 1


#: fp32 values a single-tile program may hold. 16384 is 64 KB of accumulator,
#: which fits without spilling at every warp count this launcher picks.
_SINGLE_TILE_MAX_ELEMS = 16384
#: Rows per program floor. Below 4 the tile stops amortising the row-wise
#: reduction and the store loses its 2D shape.
_MIN_TILE_N = 4
_TARGET_FP32_PER_THREAD = 32


def _strides_channel_axis(layout: Layout) -> bool:
    """True when the load walks D with a stride, so N must carry coalescing."""
    return layout in (Layout.BDN_BND, Layout.DBN_BND)


def _prev_power_of_2(n: int) -> int:
    """Largest power of two not exceeding ``n``, at least 1.

    Every tile extent reaches ``tl.arange``, which rejects anything else, and
    the grid-filling bound below is an arbitrary quotient.
    """
    return 1 << (n.bit_length() - 1) if n > 0 else 1


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _launch_params(D: int, D_OUT: int, rows: int, layout: Layout, sm_count: int) -> tuple[int, int, int, bool]:
    """Pick ``(tile_d, tile_n, num_warps, single_tile)``.

    A channel tile that spans the row lets the kernel take both moments from
    one load, so it wins whenever the register footprint fits. Within that, the
    row tile trades two effects measured on an H100:

    * layouts that stride the channel axis read ``TILE_N`` contiguous elements
      along N, so they want the widest row tile -- 64 rows of bf16 fill a
      sector, and dropping to 16 cost 13% at ``D=196``;
    * layouts that hold D contiguous only need enough rows to fill the machine,
      and a wider row tile past that shrinks the grid below the SM count, which
      is what makes the short sequence-shaped norms slow.

    The warp count then targets ~32 fp32 values per thread, which was flat to
    within 1.5% of the swept optimum at all four production shapes.
    """
    tile_d = triton.next_power_of_2(max(D, D_OUT))
    if tile_d * _MIN_TILE_N > _SINGLE_TILE_MAX_ELEMS:
        return _TILE_D, 64, 8, False

    budget = _SINGLE_TILE_MAX_ELEMS if _strides_channel_axis(layout) else _SINGLE_TILE_MAX_ELEMS // 2
    tile_n = min(64, max(1, budget // tile_d))
    tile_n = max(_MIN_TILE_N, min(tile_n, _prev_power_of_2(rows // sm_count)))
    num_warps = tile_n * tile_d // (32 * _TARGET_FP32_PER_THREAD)
    num_warps = min(8, max(2, triton.next_power_of_2(max(1, num_warps))))
    return tile_d, tile_n, num_warps, True


def _launch_layer_norm_transpose(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    elementwise_affine: bool,
    rms_norm: bool,
    layout: Layout,
    pad_multiple: int = -1,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("fused LayerNorm/RMSNorm requires a CUDA input")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported input dtype {x.dtype}; expected fp16, bf16, or fp32")
    if out_dtype is not None and out_dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported output dtype {out_dtype}; expected fp16, bf16, or fp32")
    if pad_multiple > 0 and layout in (Layout.BND_BDN, Layout.BND_DBN):
        raise ValueError(
            f"pad_multiple needs an output that is contiguous in D; layout {layout!r} keeps D as an outer axis"
        )
    out, B, N, D, D_OUT = _allocate_output(x, layout, pad_multiple, out_dtype)
    has_weight = elementwise_affine and weight is not None
    has_bias = elementwise_affine and bias is not None
    if rms_norm and has_bias:
        raise ValueError("RMSNorm does not support an additive bias")
    for name, parameter, enabled in (("weight", weight, has_weight), ("bias", bias, has_bias)):
        if not enabled or parameter is None:
            continue
        if parameter.shape != (D,):
            raise ValueError(f"{name} must have shape ({D},), got {tuple(parameter.shape)}")
        if parameter.device != x.device:
            raise ValueError(f"{name} must be on {x.device}, got {parameter.device}")
        if parameter.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"unsupported {name} dtype {parameter.dtype}; expected fp16, bf16, or fp32")
    if out.numel() == 0:
        return out

    tile_d, tile_n, num_warps, single_tile = _launch_params(D, D_OUT, B * N, layout, _sm_count(x.device.index or 0))
    x = x.contiguous()
    weight_arg = weight.contiguous() if has_weight else x
    bias_arg = bias.contiguous() if has_bias else x
    layer_norm_transpose_forward_kernel[(triton.cdiv(N, tile_n), B, 1)](
        x,
        weight_arg,
        bias_arg,
        out,
        B,
        N,
        D=D,
        D_OUT=D_OUT,
        EPS=eps,
        TILE_N=tile_n,
        TILE_D=tile_d,
        HAS_WEIGHT=has_weight,
        HAS_BIAS=has_bias,
        RMS_NORM=rms_norm,
        LAYOUT=layout,
        SINGLE_TILE=single_tile,
        NEEDS_INT64=_needs_int64(B, N, D, tile_d),
        num_warps=num_warps,
        num_stages=2,
    )
    return out


def layer_norm_transpose(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
    elementwise_affine: bool = True,
    layout: str = "nd->nd",  # codespell:ignore nd
    pad_multiple: int = -1,
    rms_norm: bool = False,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply inference-only fused LayerNorm/RMSNorm with an optional layout change.

    Normalizes the last dimension only; ``nn.LayerNorm``'s multi-axis
    ``normalized_shape`` is not supported.

    Args:
        x: Input tensor in fp16, bf16, or fp32.
        weight: Optional per-channel affine weight.
        bias: Optional per-channel LayerNorm bias. RMSNorm rejects a bias.
        eps: Epsilon added to the variance.
        elementwise_affine: Whether to apply available ``weight`` and ``bias``.
        layout: Input-to-output layout transform.
        pad_multiple: round the output channel dim up to this multiple,
            writing zeros in the tail (``< 0`` disables). Normalisation
            statistics still use the true channel count, so the result is the
            unpadded output zero-extended along D -- a valid operand for a
            GEMM whose K must be vector-aligned. Only valid for layouts whose
            output is contiguous in D, and the multiple must divide the
            64-channel tile.
        rms_norm: Compute RMSNorm instead of LayerNorm.
        out_dtype: Optional output dtype. Defaults to the input dtype.
    """
    if rms_norm and elementwise_affine and bias is not None:
        raise ValueError("RMSNorm does not support an additive bias")

    supported_layouts = (
        "nd->nd",  # codespell:ignore nd
        "nd->dn",
        "dn->nd",  # codespell:ignore nd
        "bnd->bnd",
        "bnd->bdn",
        "bdn->bnd",
        "dbn->bnd",
        "bnd->dbn",
        "bijd->bijd",
        "bijd->bdij",
        "bdij->bijd",
        "dbij->bijd",
        "bijd->dbij",
    )

    if layout == "nd->nd":  # codespell:ignore nd
        N, D = x.shape
        x = x.contiguous().view(1, N, D)
        out_shape = (N, D)
        kernel_layout = Layout.BND_BND
    elif layout == "nd->dn":
        N, D = x.shape
        x = x.contiguous().view(1, N, D)
        out_shape = (D, N)
        kernel_layout = Layout.BND_BDN
    elif layout == "dn->nd":  # codespell:ignore nd
        D, N = x.shape
        x = x.contiguous().view(1, D, N)
        out_shape = (N, D)
        kernel_layout = Layout.BDN_BND
    elif layout == "bnd->bnd":
        B, N, D = x.shape
        out_shape = (B, N, D)
        kernel_layout = Layout.BND_BND
    elif layout == "bdn->bnd":
        B, D, N = x.shape
        out_shape = (B, N, D)
        kernel_layout = Layout.BDN_BND
    elif layout == "bnd->bdn":
        B, N, D = x.shape
        out_shape = (B, D, N)
        kernel_layout = Layout.BND_BDN
    elif layout == "dbn->bnd":
        D, B, N = x.shape
        out_shape = (B, N, D)
        kernel_layout = Layout.DBN_BND
    elif layout == "bnd->dbn":
        B, N, D = x.shape
        out_shape = (D, B, N)
        kernel_layout = Layout.BND_DBN
    elif layout == "bijd->bijd":
        B, I, J, D = x.shape
        out_shape = (B, I, J, D)
        x = x.contiguous().view(B, I * J, D)
        kernel_layout = Layout.BND_BND
    elif layout == "bijd->bdij":
        B, I, J, D = x.shape
        out_shape = (B, D, I, J)
        x = x.contiguous().view(B, I * J, D)
        kernel_layout = Layout.BND_BDN
    elif layout == "bdij->bijd":
        B, D, I, J = x.shape
        out_shape = (B, I, J, D)
        x = x.contiguous().view(B, D, I * J)
        kernel_layout = Layout.BDN_BND
    elif layout == "dbij->bijd":
        D, B, I, J = x.shape
        out_shape = (B, I, J, D)
        x = x.contiguous().view(D, B, I * J)
        kernel_layout = Layout.DBN_BND
    elif layout == "bijd->dbij":
        B, I, J, D = x.shape
        out_shape = (D, B, I, J)
        x = x.contiguous().view(B, I * J, D)
        kernel_layout = Layout.BND_DBN
    else:
        raise ValueError(f"layout {layout} not supported; expected one of {supported_layouts}")

    out = _launch_layer_norm_transpose(
        x,
        weight,
        bias,
        eps,
        elementwise_affine,
        rms_norm,
        kernel_layout,
        pad_multiple,
        out_dtype,
    )
    if pad_multiple > 0:
        # Padding is rejected above unless D is the trailing output axis.
        out_shape = (*out_shape[:-1], out.shape[-1])
    return out.contiguous().view(*out_shape)
