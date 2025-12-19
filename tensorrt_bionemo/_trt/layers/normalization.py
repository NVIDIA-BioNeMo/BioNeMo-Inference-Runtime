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
from tensorrt_llm_lite.functional import Tensor, activation, shape, split
from tensorrt_llm_lite.layers.linear import Linear
from tensorrt_llm_lite.layers.normalization import LayerNorm
from tensorrt_llm_lite.module import Module


class AdaLN(Module):

    def __init__(self,
                 dim: int,
                 dim_single_cond: int,
                 eps: float = 1e-5,
                 dtype: str = None):
        super().__init__()
        self.dim = dim
        self.dim_single_cond = dim_single_cond
        self.a_norm = LayerNorm(normalized_shape=[self.dim],
                                eps=eps,
                                dtype=dtype,
                                elementwise_affine=False)
        self.s_norm = LayerNorm(normalized_shape=[dim_single_cond],
                                eps=eps,
                                dtype=dtype,
                                bias=False)
        # Fused s_scale and s_bias, s_bias has no bias
        # remember to set it to zero correctly
        self.fused_s_scale_s_bias = Linear(self.dim_single_cond,
                                           2 * self.dim,
                                           bias=True,
                                           dtype=dtype)

    def forward(self, a: Tensor, s: Tensor):
        """
        Args:
            a: [B, I, d]
            s: [B, I, d_cond]

        Returns:
            a: [B, I, d]
        """
        shape(a, 0)
        shape(a, 1)
        a = self.a_norm(a)
        s = self.s_norm(s)
        ss = self.fused_s_scale_s_bias(s)
        s_scale, s_bias = split(ss, [self.dim, self.dim], dim=-1)

        a = activation(s_scale, trt.ActivationType.SIGMOID) * a + s_bias

        return a
