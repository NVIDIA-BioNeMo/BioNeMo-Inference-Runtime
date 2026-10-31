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
"""Two norms of one width in one launch.

An attention layer with ``use_qk_norm`` normalizes ``q`` and ``k``, two column slices of its input projection, over
their whole width. :class:`~bionemo_ir._torch.layers.normalization.FusedRMSNorm` and ``FusedLayerNorm`` each need a
contiguous input, so the pair costs two copies and two launches. A conditioned transformer block likewise
normalizes its one condition twice, once with each AdaLN's weight. :func:`dual_norm_qk` reads both inputs where they
lie and normalizes them in one launch. Its rows take the single-tile path of ``layer_norm_transpose_forward_kernel``
with the same tile and warp counts, so each output row is that kernel's, bit for bit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from bionemo_ir.dsl_kernels.triton.fused_layer_norm_transpose import (
    _STRIDED_WARP_TILE_L2_MULTIPLE,
    _SUPPORTED_DTYPES,
    Layout,
    _affine,
    _l2_bytes,
    _launch_params,
    _sm_count,
    _sm_version,
)

__all__ = ["dual_norm_qk"]


@triton.jit
def dual_norm_qk_kernel(
    q_ptr,
    k_ptr,
    wq_ptr,
    wk_ptr,
    bq_ptr,
    bk_ptr,
    q_out_ptr,
    k_out_ptr,
    rows,
    q_stride,
    k_stride,
    D: tl.constexpr,
    EPS: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_D: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RMS_NORM: tl.constexpr,
    NEEDS_INT64: tl.constexpr,
):
    """Program ``(n, 0)`` normalizes ``q``'s rows ``n TILE_N ..``, program ``(n, 1)`` the same rows of ``k``."""
    pid_n = tl.program_id(0)
    if NEEDS_INT64:
        pid_n = tl.cast(pid_n, tl.int64)
        rows = tl.cast(rows, tl.int64)
    if tl.program_id(1) == 0:
        x_ptr, w_ptr, b_ptr, out_ptr, stride = q_ptr, wq_ptr, bq_ptr, q_out_ptr, q_stride
    else:
        x_ptr, w_ptr, b_ptr, out_ptr, stride = k_ptr, wk_ptr, bk_ptr, k_out_ptr, k_stride

    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_d = tl.arange(0, TILE_D)
    mask_n = offs_n < rows
    mask_d = offs_d < D
    # layer_norm_transpose_forward_kernel's single-tile path, row by row.
    x = tl.load(
        x_ptr + offs_n[:, None] * stride + offs_d[None, :], mask=mask_n[:, None] & mask_d[None, :], other=0.0
    ).to(tl.float32)
    if RMS_NORM:
        x_centred = x
    else:
        mean = tl.sum(x, axis=1) / D
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
    y = tl.where(mask_d[None, :], y, 0.0)
    tl.store(out_ptr + offs_n[:, None] * D + offs_d[None, :], y, mask=mask_n[:, None] & mask_d[None, :])


def _row_stride(x: torch.Tensor) -> int | None:
    """The stride between consecutive rows of ``x`` ``[..., D]`` read as ``[rows, D]``, or ``None`` if its leading
    axes do not step rows evenly or its channels are not contiguous."""
    if x.dim() < 2 or x.stride(-1) != 1:
        return None
    stride, rows = x.stride(-2), 1
    for axis in range(x.dim() - 2, -1, -1):
        if x.shape[axis] != 1 and x.stride(axis) != stride * rows:
            return None
        rows *= x.shape[axis]
    return stride


def _vector(t: torch.Tensor | None, width: int, x: torch.Tensor) -> bool:
    return t is None or (t.shape == (width,) and t.device == x.device and t.dtype in _SUPPORTED_DTYPES)


def dual_norm_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    q_bias: torch.Tensor | None,
    k_bias: torch.Tensor | None,
    *,
    eps: float,
    rms_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """LayerNorm or RMSNorm of ``q`` and of ``k`` over their last axis, in one launch.

    Each output is what ``layer_norm_transpose(x.reshape(-1, D), weight, bias, eps, rms_norm=...)`` returns for its
    input, bit for bit, as a contiguous tensor shaped like the input. The inputs may be strided views, such as column
    slices of a fused projection, as long as their rows are evenly spaced and their channels contiguous.

    Returns:
        ``(q_normed, k_normed)``, or ``None`` when one launch cannot serve the pair: different row counts, widths or
        dtypes, unevenly spaced rows, an affine parameter on one side only, an RMSNorm bias, or a width past the
        single-tile limit. The caller then normalizes each with its own module.
    """
    if not (q.is_cuda and k.is_cuda) or q.device != k.device or q.dtype != k.dtype or q.dtype not in _SUPPORTED_DTYPES:
        return None
    D = q.shape[-1]
    if k.shape[-1] != D or q.numel() != k.numel() or q.numel() == 0:
        return None
    if (q_weight is None) != (k_weight is None) or (q_bias is None) != (k_bias is None):
        return None
    if rms_norm and q_bias is not None:
        return None
    if not all(_vector(t, D, q) for t in (q_weight, k_weight, q_bias, k_bias)):
        return None
    q_stride, k_stride = _row_stride(q), _row_stride(k)
    if q_stride is None or k_stride is None:
        return None
    rows = q.numel() // D

    # FusedRMSNorm and FusedLayerNorm launch [1, rows, D] -> [1, rows, D]; take their tile and warp counts so the
    # per-row reduction is theirs too.
    device_index = q.device.index or 0
    traffic = rows * 2 * D * q.element_size()
    tile_d, tile_n, num_warps, single_tile = _launch_params(
        D,
        D,
        1,
        rows,
        Layout.BND_BND,
        _sm_version(device_index),
        _sm_count(device_index),
        traffic <= _STRIDED_WARP_TILE_L2_MULTIPLE * _l2_bytes(device_index),
    )
    if not single_tile:
        return None
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    has_weight, has_bias = q_weight is not None, q_bias is not None
    dual_norm_qk_kernel[(triton.cdiv(rows, tile_n), 2, 1)](
        q,
        k,
        q_weight.contiguous() if has_weight else q,
        k_weight.contiguous() if has_weight else k,
        q_bias.contiguous() if has_bias else q,
        k_bias.contiguous() if has_bias else k,
        q_out,
        k_out,
        rows,
        q_stride,
        k_stride,
        D=D,
        EPS=eps,
        TILE_N=tile_n,
        TILE_D=tile_d,
        HAS_WEIGHT=has_weight,
        HAS_BIAS=has_bias,
        RMS_NORM=rms_norm,
        NEEDS_INT64=rows * max(q_stride, k_stride, tile_d) >= 2**31 - 1,
        num_warps=num_warps,
        num_stages=2,
    )
    return q_out, k_out
