# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from tensorrt_llm_lite._utils import get_sm_version

from tensorrt_bionemo import ops
from tensorrt_bionemo._torch.custom_ops.base import (MAX_BENCHMARK_SCORE,
                                                     MIN_BENCHMARK_SCORE,
                                                     CustomOpBase)


def _generate_support_dict(dual_gemm_type: str = "x_x"):
    support_dict = {}
    if dual_gemm_type == "x_x":
        for sm in ["80", "86", "89"]:
            for dtype in ["torch.float16", "torch.bfloat16"]:
                for N in [128, 256]:
                    for K in [128]:
                        for mask in [True, False]:
                            for bias in [True, False]:
                                support_dict[
                                    f"sm{sm}-{dtype}-N={N}-K={K}-Mask={mask}-Bias={bias}"] = True
    elif dual_gemm_type == "x0_x1":
        for sm in ["80", "86", "89"]:
            for dtype in ["torch.float16", "torch.bfloat16"]:
                for N in [128, 256]:
                    for K in [128]:
                        for bias in [True, False]:
                            support_dict[
                                f"sm{sm}-{dtype}-N={N}-K={K}-Bias={bias}"] = True
    return support_dict


class CutlassFusedSigmoidGatedDualGemm(CustomOpBase):
    """Apply fused sigmoid-gated dual GEMM operation.

    Wrapper for x_x_dual_gemm operation providing a consistent API.
    """
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
            x: Input tensor (CUDA, contiguous)
            w1: First weight tensor (CUDA, contiguous)
            w2: Second weight tensor (CUDA, contiguous)
            mask: Mask tensor (CUDA, contiguous)
            b1: First bias tensor (CUDA, contiguous)
            b2: Second bias tensor (CUDA, contiguous)
        Note: For weights and biases, we always assume they are contiguous.
        Returns:
            torch.Tensor: X@W1 * sigmoid(X@W2) * mask
        """
        x = x.contiguous()
        mask = mask.contiguous() if mask is not None else None
        ret = ops.x_x_dual_gemm(x, w1, w2, b1, b2, mask)
        return ret

    @staticmethod
    def is_supported(x: torch.Tensor,
                     w1: torch.Tensor,
                     w2: torch.Tensor,
                     mask: Optional[torch.Tensor] = None,
                     b1: Optional[torch.Tensor] = None,
                     b2: Optional[torch.Tensor] = None) -> bool:
        """Check if the operation is supported for given tensor configurations.

        Args:
            x: Input tensor
            w1: First weight tensor
            w2: Second weight tensor
            mask: Mask tensor
            b1: First bias tensor
            b2: Second bias tensor
        Note: For weights and biases, we always assume they are contiguous.
        Returns:
            bool: True if the operation is supported, False otherwise
        """
        sm = get_sm_version()
        key = f"sm{sm}-{x.dtype}-N={w1.shape[0]}-K={w1.shape[1]}-Mask={mask is not None}-Bias={b1 is not None and b2 is not None}"
        return key in CutlassFusedSigmoidGatedDualGemm._SUPPORT_DICT

    @staticmethod
    def get_benchmark_score(x: torch.Tensor,
                            w1: torch.Tensor,
                            w2: torch.Tensor,
                            mask: Optional[torch.Tensor] = None,
                            b1: Optional[torch.Tensor] = None,
                            b2: Optional[torch.Tensor] = None) -> int:
        """Get benchmark score for the operation."""
        sm = get_sm_version()
        if sm == 80 or sm == 89:
            return MAX_BENCHMARK_SCORE
        M = 1
        for dim in x.shape[:-1]:
            M *= dim
        if sm == 86 and M < 512 * 512:
            return MAX_BENCHMARK_SCORE
        return MIN_BENCHMARK_SCORE


class CutlassFusedSigmoidGatedDualGemmDualX(CustomOpBase):
    """Apply fused sigmoid-gated dual GEMM operation with two input tensors.

    Wrapper for x0_x1_dual_gemm operation providing a consistent API.
    """
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
            x1: First input tensor (CUDA, contiguous)
            x2: Second input tensor (CUDA, contiguous)
            w1: First weight tensor (CUDA, contiguous)
            w2: Second weight tensor (CUDA, contiguous)
            mask: Mask tensor (CUDA, contiguous) - currently unused
            b1: First bias tensor (CUDA, contiguous)
            b2: Second bias tensor (CUDA, contiguous)

        Returns:
            torch.Tensor: sigmoid(X1@W1) * (X2@W2)
        """
        # Note: The underlying op doesn't support mask parameter
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        return ops.x0_x1_dual_gemm(x1, x2, w1, w2, b1, b2)

    @staticmethod
    def is_supported(x1: torch.Tensor,
                     x2: torch.Tensor,
                     w1: torch.Tensor,
                     w2: torch.Tensor,
                     mask: Optional[torch.Tensor] = None,
                     b1: Optional[torch.Tensor] = None,
                     b2: Optional[torch.Tensor] = None) -> bool:
        """Check if the operation is supported for given tensor configurations.

        Args:
            x1: First input tensor
            x2: Second input tensor
            w1: First weight tensor
            w2: Second weight tensor
            mask: Mask tensor (not used)
            b1: First bias tensor
            b2: Second bias tensor

        Returns:
            bool: True if the operation is supported, False otherwise
        """
        sm = get_sm_version()
        key = f"sm{sm}-{x1.dtype}-N={w1.shape[0]}-K={w1.shape[1]}-Bias={b1 is not None and b2 is not None}"
        return key in CutlassFusedSigmoidGatedDualGemmDualX._SUPPORT_DICT

    @staticmethod
    def get_benchmark_score(x1: torch.Tensor,
                            x2: torch.Tensor,
                            w1: torch.Tensor,
                            w2: torch.Tensor,
                            mask: Optional[torch.Tensor] = None,
                            b1: Optional[torch.Tensor] = None,
                            b2: Optional[torch.Tensor] = None) -> int:
        """Get benchmark score for the operation."""
        sm = get_sm_version()
        if sm == 80 or sm == 89:
            return MAX_BENCHMARK_SCORE
        M = 1
        for dim in x1.shape[:-1]:
            M *= dim
        if sm == 86 and M < 512 * 512:
            return MAX_BENCHMARK_SCORE
        return MIN_BENCHMARK_SCORE


CUTLASS_DUAL_GEMM_KERNELS = {
    "fused_sigmoid_gated_dual_gemm": CutlassFusedSigmoidGatedDualGemm,
    "fused_sigmoid_gated_dual_gemm_dual_x":
    CutlassFusedSigmoidGatedDualGemmDualX,
}
