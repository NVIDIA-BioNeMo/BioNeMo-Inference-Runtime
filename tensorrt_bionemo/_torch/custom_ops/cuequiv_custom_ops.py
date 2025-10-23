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


class CuEquivFusedSigmoidGatedDualGemm:
    """ Apply fused sigmoid-gated dual GEMM operation."""

    @staticmethod
    def apply(x: torch.Tensor,
              w1: torch.Tensor,
              w2: torch.Tensor,
              mask: Optional[torch.Tensor] = None,
              b1: Optional[torch.Tensor] = None,
              b2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x(torch.Tensor): Input tensor.
            w1(torch.Tensor): First weight tensor.
            w2(torch.Tensor): Second weight tensor.
            mask(torch.Tensor): Mask tensor.
            b1(torch.Tensor): First bias tensor.
            b2(torch.Tensor): Second bias tensor.
        Returns:
            torch.Tensor: sigmoid(x@w1 + b1) * (x@w2 + b2)
        """
        return fused_sigmoid_gated_dual_gemm(x,
                                             w1,
                                             w2,
                                             mask,
                                             transpose_out=False,
                                             b1=b1,
                                             b2=b2)

    @staticmethod
    def is_supported(x: torch.Tensor,
                     w1: torch.Tensor,
                     w2: torch.Tensor,
                     mask: Optional[torch.Tensor] = None,
                     b1: Optional[torch.Tensor] = None,
                     b2: Optional[torch.Tensor] = None) -> bool:
        M = 1
        for dim in x.shape[:-1]:
            M *= dim
        if M < 32:
            return False
        N = w1.shape[0]
        K = w1.shape[1]
        if N not in [128, 256]:
            return False
        if K not in [128]:
            return False
        if not x.dtype in [torch.float16, torch.bfloat16]:
            return False
        return True


class CuEquivFusedSigmoidGatedDualGemmDualX:
    """ Apply fused sigmoid-gated dual GEMM operation with two input tensors."""

    @staticmethod
    def apply(x1: torch.Tensor,
              x2: torch.Tensor,
              w1: torch.Tensor,
              w2: torch.Tensor,
              mask: Optional[torch.Tensor] = None,
              b1: Optional[torch.Tensor] = None,
              b2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Apply fused sigmoid-gated dual GEMM operation with two input tensors.
            Args:
                x1(torch.Tensor): First input tensor.
                x2(torch.Tensor): Second input tensor.
                w1(torch.Tensor): First weight tensor.
                w2(torch.Tensor): Second weight tensor.
                mask(torch.Tensor): Mask tensor.
                b1(torch.Tensor): First bias tensor.
                b2(torch.Tensor): Second bias tensor.
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
                                                    b2=b2)

    @staticmethod
    def is_supported(x1: torch.Tensor,
                     x2: torch.Tensor,
                     w1: torch.Tensor,
                     w2: torch.Tensor,
                     mask: Optional[torch.Tensor] = None,
                     b1: Optional[torch.Tensor] = None,
                     b2: Optional[torch.Tensor] = None) -> bool:
        M = 1
        for dim in x1.shape[:-1]:
            M *= dim
        if M < 32:
            return False
        N = w1.shape[0]
        K = w1.shape[1]
        if N not in [128, 256]:
            return False
        if K not in [128]:
            return False
        if x1.shape[-1] != x2.shape[-1]:
            return False
        if not x1.dtype in [torch.float16, torch.bfloat16]:
            return False
        return True
