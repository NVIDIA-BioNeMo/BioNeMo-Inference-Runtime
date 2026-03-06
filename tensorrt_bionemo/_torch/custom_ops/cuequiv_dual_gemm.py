# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

import torch
from cuequivariance_ops_torch.gated_gemm_torch import (
    fused_sigmoid_gated_dual_gemm, fused_sigmoid_gated_dual_gemm_dual_x)

from tensorrt_bionemo._torch.custom_ops.base import (IGNORE_BENCHMARK_SCORE,
                                                     MID_BENCHMARK_SCORE,
                                                     CustomOpBase)


def _generate_support_dict(dual_gemm_type: str = "x_x"):
    support_dict = {}
    if dual_gemm_type == "x_x":
        for dtype in ["torch.float16", "torch.bfloat16", "torch.float32"]:
            for N in [128, 256]:
                for K in [128]:
                    for mask in [True, False]:
                        for bias in [True, False]:
                            support_dict[
                                f"{dtype}-N={N}-K={K}-Mask={mask}-Bias={bias}"] = True
    elif dual_gemm_type == "x0_x1":
        for dtype in ["torch.float16", "torch.bfloat16", "torch.float32"]:
            for N in [128, 256]:
                for K in [128]:
                    for bias in [True, False]:
                        support_dict[f"{dtype}-N={N}-K={K}-Bias={bias}"] = True
    return support_dict


class CuEquivFusedSigmoidGatedDualGemm(CustomOpBase):
    """Apply fused sigmoid-gated dual GEMM operation."""

    _SUPPORT_DICT = _generate_support_dict(dual_gemm_type="x_x")

    @staticmethod
    def apply(x: torch.Tensor,
              w1: torch.Tensor,
              w2: torch.Tensor,
              mask: Optional[torch.Tensor] = None,
              b1: Optional[torch.Tensor] = None,
              b2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply fused sigmoid-gated dual GEMM operation.

        Args:
            x: Input tensor
            w1: First weight tensor
            w2: Second weight tensor
            mask: Mask tensor
            b1: First bias tensor
            b2: Second bias tensor

        Returns:
            torch.Tensor: sigmoid(x@w1 + b1) * (x@w2 + b2)
        """
        return fused_sigmoid_gated_dual_gemm(x,
                                             w1,
                                             w2,
                                             mask,
                                             transpose_out=False,
                                             b1=b1,
                                             b2=b2,
                                             precision=-1)

    @staticmethod
    def is_supported(x: torch.Tensor,
                     w1: torch.Tensor,
                     w2: torch.Tensor,
                     mask: Optional[torch.Tensor] = None,
                     b1: Optional[torch.Tensor] = None,
                     b2: Optional[torch.Tensor] = None) -> bool:
        """Check if the operation is supported for given tensor configurations."""
        key = (
            f"{x.dtype}-N={w1.shape[0]}-K={w1.shape[1]}"
            f"-Mask={mask is not None}-Bias={b1 is not None and b2 is not None}"
        )
        return key in CuEquivFusedSigmoidGatedDualGemm._SUPPORT_DICT

    @staticmethod
    def get_benchmark_score(x: torch.Tensor,
                            w1: torch.Tensor,
                            w2: torch.Tensor,
                            mask: Optional[torch.Tensor] = None,
                            b1: Optional[torch.Tensor] = None,
                            b2: Optional[torch.Tensor] = None) -> int:
        """Get benchmark score for the operation."""
        M = 1
        for dim in x.shape[:-1]:
            M *= dim
        if M < 65536:
            return IGNORE_BENCHMARK_SCORE
        return MID_BENCHMARK_SCORE


class CuEquivFusedSigmoidGatedDualGemmDualX(CustomOpBase):
    """Apply fused sigmoid-gated dual GEMM operation with two input tensors."""

    _SUPPORT_DICT = _generate_support_dict(dual_gemm_type="x0_x1")

    @staticmethod
    def apply(x1: torch.Tensor,
              x2: torch.Tensor,
              w1: torch.Tensor,
              w2: torch.Tensor,
              mask: Optional[torch.Tensor] = None,
              b1: Optional[torch.Tensor] = None,
              b2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply fused sigmoid-gated dual GEMM operation with two input tensors.

        Args:
            x1: First input tensor
            x2: Second input tensor
            w1: First weight tensor
            w2: Second weight tensor
            mask: Mask tensor
            b1: First bias tensor
            b2: Second bias tensor

        Returns:
            torch.Tensor: sigmoid(x1@w1 + b1) * (x2@w2 + b2)
        """
        return fused_sigmoid_gated_dual_gemm_dual_x(x1,
                                                    x2,
                                                    w1,
                                                    w2,
                                                    mask,
                                                    transpose_out=False,
                                                    b1=b1,
                                                    b2=b2,
                                                    precision=-1)

    @staticmethod
    def is_supported(x1: torch.Tensor,
                     x2: torch.Tensor,
                     w1: torch.Tensor,
                     w2: torch.Tensor,
                     mask: Optional[torch.Tensor] = None,
                     b1: Optional[torch.Tensor] = None,
                     b2: Optional[torch.Tensor] = None) -> bool:
        """Check if the operation is supported for given tensor configurations."""
        key = (f"{x1.dtype}-N={w1.shape[0]}-K={w1.shape[1]}"
               f"-Bias={b1 is not None and b2 is not None}")
        return key in CuEquivFusedSigmoidGatedDualGemmDualX._SUPPORT_DICT

    @staticmethod
    def get_benchmark_score(x1: torch.Tensor,
                            x2: torch.Tensor,
                            w1: torch.Tensor,
                            w2: torch.Tensor,
                            mask: Optional[torch.Tensor] = None,
                            b1: Optional[torch.Tensor] = None,
                            b2: Optional[torch.Tensor] = None) -> int:
        """Get benchmark score for the operation."""
        M = 1
        for dim in x1.shape[:-1]:
            M *= dim
        if M < 65536:
            return IGNORE_BENCHMARK_SCORE
        return MID_BENCHMARK_SCORE


CUEQUIV_DUAL_GEMM_KERNELS = {
    "fused_sigmoid_gated_dual_gemm": CuEquivFusedSigmoidGatedDualGemm,
    "fused_sigmoid_gated_dual_gemm_dual_x":
    CuEquivFusedSigmoidGatedDualGemmDualX,
}
