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
"""Dual-GEMM ``x_x`` public operator selection."""

from __future__ import annotations

from collections.abc import Callable

import torch
from cuequivariance_ops_torch.gated_gemm_torch import fused_sigmoid_gated_dual_gemm

from tensorrt_bionemo.utils import get_sm_version

from .cutedsl import DualGemmXxCuTe


def _invoke_vanilla_dual_gemm_x_x(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    transpose_out: bool = False,
    actual_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the unfused PyTorch fallback."""
    del actual_seqlen
    if bias1 is not None and bias2 is not None:
        result = (x @ w1.T + bias1).sigmoid() * (x @ w2.T + bias2)
    else:
        result = (x @ w1.T).sigmoid() * (x @ w2.T)
    if mask is not None:
        result = result * mask.unsqueeze(-1)
    if transpose_out:
        result = result.moveaxis(-1, 0)
    return result.contiguous()


def _invoke_cuequiv_dual_gemm_x_x(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    transpose_out: bool = False,
    actual_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the cuEquivariance fused fallback."""
    del actual_seqlen
    return fused_sigmoid_gated_dual_gemm(
        x,
        w1,
        w2,
        mask,
        transpose_out=transpose_out,
        b1=bias1,
        b2=bias2,
        precision=-1,
    )


_DUAL_GEMM_XX_CUTE: DualGemmXxCuTe | None = None


def _get_cute_dual_gemm_x_x() -> DualGemmXxCuTe:
    """Return the process-wide CuTe backend instance."""
    global _DUAL_GEMM_XX_CUTE
    if _DUAL_GEMM_XX_CUTE is None:
        _DUAL_GEMM_XX_CUTE = DualGemmXxCuTe()
    return _DUAL_GEMM_XX_CUTE


def _invoke_cute_dual_gemm_x_x(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    transpose_out: bool = False,
    actual_seqlen: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the source-or-CUBIN CuTe backend."""
    return _get_cute_dual_gemm_x_x()(
        x,
        w1,
        w2,
        bias0=bias1,
        bias1=bias2,
        mask=mask,
        transpose_out=transpose_out,
        actual_seqlen=actual_seqlen,
    )


def get_dual_gemm_x_x_op(
    dtype: torch.dtype,
    transpose_out: bool = False,
    N: int = 128,
    K: int = 128,
    pair_mask_left_aligned: bool = True,
) -> Callable:
    """Return the best backend for one dtype, shape, and device."""
    sm = get_sm_version()
    cute_shapes = {(128, 128), (256, 128)}
    if sm in (80, 86, 89, 90):
        cute_shapes = cute_shapes | {(512, 256)}
    if (N, K) not in cute_shapes or dtype not in (torch.float16, torch.bfloat16):
        return _invoke_vanilla_dual_gemm_x_x

    if pair_mask_left_aligned and sm in (80, 86, 89, 90):
        return _invoke_cute_dual_gemm_x_x
    return _invoke_cuequiv_dual_gemm_x_x
