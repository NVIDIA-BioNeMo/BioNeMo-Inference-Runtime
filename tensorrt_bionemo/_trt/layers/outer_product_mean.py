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

import tensorrt as trt
from tensorrt_llm_lite.functional import (Tensor, cast, concat, einsum,
                                          elementwise_binary, shape, split,
                                          sum)
from tensorrt_llm_lite.layers.linear import Linear
from tensorrt_llm_lite.layers.normalization import LayerNorm
from tensorrt_llm_lite.module import Module


class OuterProductMean(Module):

    def __init__(self,
                 c_in: int,
                 c_hidden: int,
                 c_out: int,
                 eps: float = 1e-5,
                 mask_eps: float = 1e-3,
                 norm_mask_by_eps: bool = False,
                 norm_before_output: bool = True,
                 cast_to_float_before_einsum: bool = True,
                 bias_flags: dict[str, bool] = {
                     "proj_a": False,
                     "proj_b": False,
                     "proj_o": True
                 },
                 dtype: str = None):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.c_out = c_out
        self.eps = eps
        self.mask_eps = mask_eps
        self.norm_mask_by_eps = norm_mask_by_eps
        self.norm_before_output = norm_before_output
        self.cast_to_float_before_einsum = cast_to_float_before_einsum
        self.dtype = dtype

        self.norm = LayerNorm([c_in], eps=eps, dtype=dtype)
        self.fused_proj_a_b = Linear(c_in,
                                     2 * self.c_hidden,
                                     bias=bias_flags["proj_a"]
                                     or bias_flags["proj_b"],
                                     dtype=dtype)
        self.proj_o = Linear(self.c_hidden * self.c_hidden,
                             c_out,
                             bias=bias_flags["proj_o"],
                             dtype=dtype)

    def forward(self, m: Tensor, mask: Tensor) -> Tensor:
        """
        Note: For TRT side, we need to re-work for a distributed gemm with ring-commnication.
              At now, linear layers will be reduced at the end.
        Args:
            m: [B, I, J, c_in]
            mask: [B, I, J]
        Returns:
            [B, J, J, c_out]
        """
        m = self.norm(m)
        mask = mask.unsqueeze(-1)

        ab = self.fused_proj_a_b(m)  # At here, ab is reduced.
        a, b = split(ab, [self.c_hidden, self.c_hidden], dim=-1)

        if self.cast_to_float_before_einsum:
            a = cast(a * mask, "float32")
            b = cast(b * mask, "float32")
        else:
            a = a * mask
            b = b * mask

        left_mask = mask.unsqueeze(-3)
        right_mask = mask.unsqueeze(-2)

        mask = sum(left_mask * right_mask, dim=1)

        mask = cast(mask, "float32")
        if self.norm_mask_by_eps:
            num_mask = mask + self.mask_eps
        else:
            num_mask = elementwise_binary(mask, float(1),
                                          trt.ElementWiseOperation.MAX)

        z = einsum("bsic,bsjd->bijcd", [a, b])

        cxd = self.c_hidden * self.c_hidden
        new_shape = concat([shape(z, 0), shape(z, 1), shape(z, 2), cxd])
        z = z.view(new_shape)
        if z.dtype != m.dtype:
            z = cast(z, m.dtype)
            num_mask = cast(num_mask, m.dtype)
        if self.norm_before_output:
            z = z / num_mask
        z = self.proj_o(z)
        if not self.norm_before_output:
            z = z / num_mask
        return z
