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
"""Dual-GEMM ``x0_x1`` public operator selection."""

from __future__ import annotations

from collections.abc import Callable

import torch

from bionemo_ir.utils import get_sm_version

from .cutedsl import DualGemmX0X1CuTe


def _invoke_vanilla_dual_gemm_x0_x1(
    x1: torch.Tensor,
    x2: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    transpose_out: bool = False,
) -> torch.Tensor:
    """Run the unfused PyTorch fallback."""
    if bias1 is not None and bias2 is not None:
        result = (x1 @ w1.T + bias1).sigmoid() * (x2 @ w2.T + bias2)
    else:
        result = (x1 @ w1.T).sigmoid() * (x2 @ w2.T)
    if mask is not None:
        result = result * mask.unsqueeze(-1)
    if transpose_out:
        result = result.moveaxis(-1, 0)
    return result.contiguous()


def _invoke_cuequiv_dual_gemm_x0_x1(
    x1: torch.Tensor,
    x2: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    transpose_out: bool = False,
) -> torch.Tensor:
    """Run the cuEquivariance fused fallback."""
    from cuequivariance_ops_torch.gated_gemm_torch import fused_sigmoid_gated_dual_gemm_dual_x

    return fused_sigmoid_gated_dual_gemm_dual_x(
        x1,
        x2,
        w1,
        w2,
        mask,
        transpose_out=transpose_out,
        b1=bias1,
        b2=bias2,
        precision=-1,
    )


_DUAL_GEMM_X0X1_CUTE: DualGemmX0X1CuTe | None = None


def _get_cute_dual_gemm_x0_x1() -> DualGemmX0X1CuTe:
    """Return the process-wide CuTe backend instance."""
    global _DUAL_GEMM_X0X1_CUTE
    if _DUAL_GEMM_X0X1_CUTE is None:
        _DUAL_GEMM_X0X1_CUTE = DualGemmX0X1CuTe()
    return _DUAL_GEMM_X0X1_CUTE


def _invoke_cute_dual_gemm_x0_x1(
    x1: torch.Tensor,
    x2: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    transpose_out: bool = False,
) -> torch.Tensor:
    """Run the source-or-CUBIN CuTe backend."""
    if mask is not None:
        raise NotImplementedError(
            "CuTe dual_gemm_x0_x1 backend does not support `mask`; route to the "
            "cuEquivariance backend via get_dual_gemm_x0_x1_op for masked calls."
        )
    if transpose_out:
        raise NotImplementedError(
            "CuTe dual_gemm_x0_x1 backend does not support `transpose_out`; route "
            "to the cuEquivariance backend via get_dual_gemm_x0_x1_op for "
            "transposed output."
        )
    return _get_cute_dual_gemm_x0_x1()(x1, x2, w1, w2, bias0=bias1, bias1=bias2)


def get_dual_gemm_x0_x1_op(
    dtype: torch.dtype,
    transpose_out: bool = False,
    N: int = 128,
    K: int = 128,
) -> Callable:
    """Return the best backend for one dtype, shape, and device."""
    sm = get_sm_version()
    cute_shapes = {(128, 128)}
    if sm in (80, 86, 89, 90):
        cute_shapes = cute_shapes | {(256, 256)}
    cuequiv_shapes = {(256, 128)}

    if dtype not in (torch.float16, torch.bfloat16):
        return _invoke_vanilla_dual_gemm_x0_x1
    if (N, K) not in cute_shapes and (N, K) not in cuequiv_shapes:
        return _invoke_vanilla_dual_gemm_x0_x1
    if transpose_out or (N, K) in cuequiv_shapes:
        return _invoke_cuequiv_dual_gemm_x0_x1
    if sm in (80, 86, 89, 90):
        return _invoke_cute_dual_gemm_x0_x1
    return _invoke_cuequiv_dual_gemm_x0_x1
