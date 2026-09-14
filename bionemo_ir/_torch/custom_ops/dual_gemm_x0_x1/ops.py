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
    K0: int | None = None,
    K1: int | None = None,
) -> Callable:
    """Return the best backend for one dtype, shape, and device.

    ``K`` stays the equal-width default. Supplying ``K0`` / ``K1`` selects a
    genuinely asymmetric projection with pair width gating a
    trimul-hidden-width projection.
    """
    sm = get_sm_version()
    K0 = K if K0 is None else K0
    K1 = K if K1 is None else K1
    # Keyed on ``(N, K0, K1)``:
    #   * (128, 128, 128) -- OpenFold3 / Boltz; SM80/86/89/90
    #   * (256, 256, 256) -- ProtenixV2; SM80/86/89/90
    #   * (256, 256, 200) / (384, 384, 200 | 256) -- asymmetric trimul; SM80/86/89/90
    cute_shapes = {(128, 128, 128)}
    if sm in (80, 86, 89, 90):
        cute_shapes = cute_shapes | {(256, 256, 256)}
    if sm in (80, 86, 89, 90):
        # 200 is trimul hidden width 196 padded to a 128-bit copy atom.
        cute_shapes = cute_shapes | {
            (256, 256, 200),
            (384, 384, 200),
            (384, 384, 256),
        }
    if sm in (80, 86, 89, 90):
        cute_shapes = cute_shapes | {
            # Template-level trimul in OpenFold2/3 and Protenix.
            (64, 64, 64),
            # Legacy out-projection whose wide output needs independent strides.
            (256, 128, 128),
        }
    if sm in (80, 86, 89, 90):
        # z12 hero trimul, pair_dim 512 / tri_mult_c 256.
        cute_shapes.add((512, 512, 256))
    # The legacy OpenFold3-width out-projection now has direct CuTe tuning on
    # every shipped SM.
    cuequiv_shapes: set[tuple[int, int]] = set() if sm in (80, 86, 89, 90) else {(256, 128)}

    if dtype not in (torch.float16, torch.bfloat16):
        return _invoke_vanilla_dual_gemm_x0_x1
    if (N, K0, K1) not in cute_shapes and (N, K0) not in cuequiv_shapes:
        return _invoke_vanilla_dual_gemm_x0_x1
    # cuEquivariance requires equal inner dims and owns the transposed-output
    # path; asymmetric widths and every other transpose go to vanilla.
    if K0 == K1 and (transpose_out or (N, K0) in cuequiv_shapes):
        return _invoke_cuequiv_dual_gemm_x0_x1
    if transpose_out:
        return _invoke_vanilla_dual_gemm_x0_x1
    if sm in (80, 86, 89, 90):
        return _invoke_cute_dual_gemm_x0_x1
    if K0 == K1:
        return _invoke_cuequiv_dual_gemm_x0_x1
    return _invoke_vanilla_dual_gemm_x0_x1
