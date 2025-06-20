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

import tensorrt as trt
from tensorrt_llm.functional import (AllReduceParams, Tensor, activation, split,
                                     swiglu)
from tensorrt_llm.layers.linear import ColumnLinear, RowLinear
from tensorrt_llm.layers.normalization import LayerNorm
from tensorrt_llm.module import Module

from tensorrt_bionemo._trt.layers.normalization import AdaLN
from tensorrt_bionemo.mapping import Mapping


class Transition(Module):

    def __init__(self,
                 *,
                 local_layer_idx: int,
                 dim: int = 128,
                 hidden: int = 512,
                 out_dim: Optional[int] = None,
                 eps: float = 1e-05,
                 dtype: str = None,
                 mapping: Mapping = Mapping()) -> None:
        super().__init__()
        if out_dim is None:
            out_dim = dim

        self.mapping = mapping
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.tp_group = mapping.tp_group

        self.local_layer_idx = local_layer_idx
        self.dim = dim
        self.hidden = hidden // self.tp_size
        self.out_dim = out_dim

        self.norm = LayerNorm(normalized_shape=[dim], eps=eps, dtype=dtype)
        self.fused_fc2_fc1 = ColumnLinear(self.dim,
                                          2 * self.tp_size * self.hidden,
                                          bias=False,
                                          dtype=dtype,
                                          tp_group=self.tp_group,
                                          tp_size=self.tp_size,
                                          gather_output=False)
        self.fc3 = RowLinear(self.tp_size * self.hidden,
                             self.out_dim,
                             bias=False,
                             dtype=dtype,
                             tp_group=self.tp_group,
                             tp_size=self.tp_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.fused_fc2_fc1(x)
        x = swiglu(x)
        x = self.fc3(x)
        return x


class ConditionedTransitionBlock(Module):
    """Algorithm 25"""

    def __init__(self,
                 dim_single: int,
                 dim_single_cond: int,
                 expansion_factor: int = 2,
                 eps: float = 1e-5,
                 dtype: str = None,
                 mapping: Mapping = Mapping()):
        super().__init__()
        self.mapping = mapping
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.tp_group = mapping.tp_group

        self.dim_single = dim_single
        self.dim_single_cond = dim_single_cond
        self.expansion_factor = expansion_factor

        self.adaln = AdaLN(dim_single,
                           dim_single_cond,
                           eps=eps,
                           dtype=dtype,
                           mapping=mapping)
        self.dim_inner = int(dim_single * expansion_factor) // mapping.tp_size
        # Fused swiglu_gate linear and a_to_b
        self.fused_swl_a_to_b = ColumnLinear(self.dim_single,
                                             self.dim_inner * 3,
                                             bias=False,
                                             dtype=dtype,
                                             tp_group=self.tp_group,
                                             tp_size=self.tp_size,
                                             gather_output=False,
                                             is_qkv=True)
        self.b_to_a = RowLinear(self.dim_inner,
                                self.dim_single,
                                bias=False,
                                dtype=dtype,
                                tp_group=self.tp_group,
                                tp_size=self.tp_size)
        self.output_projection = ColumnLinear(self.dim_single_cond,
                                              self.dim_single,
                                              bias=True,
                                              dtype=dtype,
                                              tp_group=self.tp_group,
                                              tp_size=self.tp_size,
                                              gather_output=True)

    def forward(self,
                a: Tensor,
                s: Tensor,
                all_reduce_params: Optional[AllReduceParams] = None) -> Tensor:
        """
        Args:
            a: [B, I, d]
            s: [B, I, d_cond]

        Returns:
            a: [B, I, d]
        """
        a = self.adaln(a, s)
        z = self.fused_swl_a_to_b(a)
        m, n = split(z, [self.dim_inner * 2, self.dim_inner], dim=-1)
        b = swiglu(m) * n  # TODO: Fused swiglu here
        a = self.output_projection(s)
        a = activation(a, trt.ActivationType.SIGMOID) * self.b_to_a(
            b, all_reduce_params=all_reduce_params)
        return a
