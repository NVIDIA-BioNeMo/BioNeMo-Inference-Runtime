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
from tensorrt_llm.functional import Tensor, activation, shape, slice, split
from tensorrt_llm.layers.linear import ColumnLinear
from tensorrt_llm.layers.normalization import LayerNorm
from tensorrt_llm.module import Module

from tensorrt_bionemo.mapping import Mapping


class AdaLN(Module):

    def __init__(self,
                 dim: int,
                 dim_single_cond: int,
                 eps: float = 1e-5,
                 dtype: str = None,
                 mapping: Mapping = Mapping()):
        super().__init__()
        self.dim = dim // mapping.tp_size
        self.dim_single_cond = dim_single_cond
        self.mapping = mapping
        self.tp_group = mapping.tp_group
        self.a_norm = LayerNorm(normalized_shape=[self.dim * mapping.tp_size],
                                eps=eps,
                                dtype=dtype,
                                elementwise_affine=False,
                                tp_size=1,
                                tp_dim=0)
        self.s_norm = LayerNorm(normalized_shape=[dim_single_cond],
                                eps=eps,
                                dtype=dtype,
                                tp_size=1,
                                tp_dim=0,
                                bias=False)
        # Fused s_scale and s_bias, s_bias has no bias
        # remember to set it to zero correctly
        self.fused_s_scale_s_bias = ColumnLinear(self.dim_single_cond,
                                                 2 * mapping.tp_size * self.dim,
                                                 bias=True,
                                                 dtype=dtype,
                                                 tp_group=mapping.tp_group,
                                                 tp_size=mapping.tp_size,
                                                 gather_output=False)

    def forward(self, a: Tensor, s: Tensor):
        """
        Args:
            a: [B, I, d]
            s: [B, I, d_cond]

        Returns:
            a: [B, I, d]
        """
        bs = shape(a, 0)
        seqlen = shape(a, 1)
        a = self.a_norm(a)
        s = self.s_norm(s)
        ss = self.fused_s_scale_s_bias(s)
        s_scale, s_bias = split(ss, [self.dim, self.dim], dim=-1)
        if self.mapping.tp_size > 1:
            # slice a tensor to get the local tensor
            s_idx = self.mapping.tp_rank * self.dim
            starts = concat([0, 0, s_idx])
            sizes = concat([bs, seqlen, self.dim])
            a = slice(a, starts, sizes)

        a = activation(s_scale, trt.ActivationType.SIGMOID) * a + s_bias

        if self.mapping.tp_size > 1:
            a = allgather(a, self.tp_group, gather_dim=-1)
        return a
