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

from enum import IntEnum
from typing import Optional

from tensorrt_llm.functional import (Tensor, allgather, concat, expand_dims,
                                     floordiv, permute, shape, slice, squeeze)
from tensorrt_llm.layers.linear import ColumnLinear
from tensorrt_llm.layers.normalization import LayerNorm
from tensorrt_llm.module import Module

from tensorrt_bionemo.functional import chunk_loop

from .attention import AttentionParams, TriangleAttention


class TriangleAttentionNodeType(IntEnum):
    STARTING = 0
    ENDING = 1


class TriangleMultiplicationNodeType(IntEnum):
    INCOMING = 0
    OUTGOING = 1


class TriangleAttentionNode(Module):

    def __init__(
            self,
            *,
            local_layer_idx: int,
            c_in: int,
            c_hidden: int,
            num_heads: int,
            node_type: TriangleAttentionNodeType = TriangleAttentionNodeType.
        STARTING,
            inf: float = 1e9,
            eps: float = 1e-05,
            dtype: str = None,
            chunk_size: int = 0,
            tp_group: Optional[list[int]] = None,
            tp_size: int = 1,
            tp_rank: int = 0,
            dp_group: Optional[list[int]] = None,
            dp_size: int = 1,
            dp_rank: int = 0):
        super().__init__()
        self.local_layer_idx = local_layer_idx
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.node_type = node_type
        self.inf = inf

        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.dp_group = dp_group

        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group

        self.chunk_size = chunk_size

        assert self.num_heads % self.tp_size == 0, \
            "num_attention_heads must be divisible by tp_size"

        if chunk_size > 0:
            assert self.chunk_size % self.dp_size == 0, \
                "chunk_size must be divisible by dp_size"
            self.chunk_size = chunk_size // self.dp_size
        self.layer_norm = LayerNorm(normalized_shape=[self.c_in],
                                    eps=eps,
                                    dtype=dtype,
                                    tp_size=1,
                                    tp_dim=0)
        self.linear = ColumnLinear(self.c_in,
                                   self.num_heads,
                                   bias=False,
                                   dtype=dtype,
                                   tp_group=tp_group,
                                   tp_size=tp_size,
                                   gather_output=True)
        self.mha = TriangleAttention(local_layer_idx=self.local_layer_idx,
                                     hidden_size=self.c_in,
                                     num_attention_heads=self.num_heads,
                                     num_kv_heads=self.num_heads,
                                     dtype=dtype,
                                     bias=False,
                                     gating=True,
                                     tp_group=tp_group,
                                     tp_size=tp_size,
                                     tp_rank=tp_rank)

    def forward(self, x: Tensor, mask: Tensor,
                attention_params: AttentionParams):
        if x.ndim() > 3:
            x = squeeze(x, 0)
        if mask.ndim() > 2:
            mask = squeeze(mask, 0)
        assert x.ndim() == 3
        assert mask.ndim() == 2

        if self.node_type == TriangleAttentionNodeType.ENDING:
            x = x.transpose(0, 1)
            mask = mask.transpose(0, 1)
        x = self.layer_norm(x)

        # Compute mask bias
        mask_bias = (self.inf * (mask - 1))
        mask_bias = expand_dims(mask_bias, [1, 2])

        # Compute triangle bias
        lx = self.linear(x)

        triangle_bias = permute(lx, [2, 0, 1])
        triangle_bias = expand_dims(triangle_bias, 0)  # [1, H, I, J]
        # First if dp_size > 1, we need to split the input by dp_size
        seq_len = shape(x, 0)
        if self.dp_size > 1:
            seq_len = floordiv(seq_len, self.dp_size)
            s_idx = seq_len * self.dp_rank
            slice_size = seq_len
            starts = concat([s_idx, 0, 0])
            sizes = concat([slice_size, shape(x, 1), shape(x, 2)])
            x = slice(x, starts, sizes)
            starts = concat([s_idx, 0, 0, 0])
            sizes = concat([slice_size, 1, 1, shape(mask_bias, 3)])
            mask_bias = slice(mask_bias, starts, sizes)

        def _loop_body(sub_chunk):
            sub_x, sub_mask_bias = sub_chunk
            biases = [sub_mask_bias, triangle_bias]
            context = self.mha(sub_x,
                               biases=biases,
                               attention_params=attention_params)
            return context

        output = chunk_loop([x, mask_bias],
                            self.chunk_size,
                            _loop_body,
                            reshape_output=False)
        if self.dp_size > 1:
            output = allgather(output, self.dp_group, gather_dim=0)

        if self.node_type == TriangleAttentionNodeType.ENDING:
            output = output = output.transpose(1, 0)

        return output
