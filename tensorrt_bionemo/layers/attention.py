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

import math
from typing import Optional

# isort: off
import tensorrt as trt
# isort: on

from tensorrt_llm._common import precision
from tensorrt_llm.functional import (AllReduceParams, Tensor, activation, cast,
                                     concat, expand_dims, matmul, shape, slice,
                                     softmax, split)
from tensorrt_llm.layers.linear import ColumnLinear, RowLinear
from tensorrt_llm.layers.normalization import LayerNorm
from tensorrt_llm.module import Module

from tensorrt_bionemo.mapping import Mapping


class AttentionParams(object):

    def __init__(self, plain_attn_precision: str = 'float32'):
        self.plain_attn_precision = plain_attn_precision


class TriangleAttention(Module):

    def __init__(self,
                 *,
                 local_layer_idx: int,
                 hidden_size: int,
                 num_attention_heads: int,
                 num_kv_heads: Optional[int] = None,
                 bias: bool = False,
                 gating: bool = True,
                 dtype: str = None,
                 mapping: Mapping = Mapping()):
        super().__init__()
        self.local_layer_idx = local_layer_idx

        self.attention_head_size = hidden_size // num_attention_heads
        self.num_kv_heads = num_kv_heads
        assert num_attention_heads % mapping.tp_size == 0, \
            "num_attention_heads must be divisible by tp_size"
        self.num_attention_heads = num_attention_heads // mapping.tp_size
        self.num_attention_kv_heads = (
            num_kv_heads + mapping.tp_size - 1
        ) // mapping.tp_size if num_kv_heads is not None else self.num_attention_heads
        assert self.num_attention_heads == self.num_attention_kv_heads, \
            "num_attention_heads must be equal to num_attention_kv_heads for the triangular attention"
        self.hidden_size = hidden_size
        self.attention_hidden_size = self.attention_head_size * self.num_attention_heads

        self.tp_group = mapping.tp_group
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.dtype = dtype
        self.bias = bias

        self.norm_factor = math.sqrt(self.attention_head_size)

        self.q_size = self.num_attention_heads * self.attention_head_size
        self.kv_size = self.num_attention_kv_heads * self.attention_head_size
        self.qkv_proj = ColumnLinear(hidden_size,
                                     mapping.tp_size * self.q_size +
                                     2 * mapping.tp_size * self.kv_size,
                                     bias=bias,
                                     dtype=dtype,
                                     tp_group=mapping.tp_group,
                                     tp_size=mapping.tp_size,
                                     gather_output=False,
                                     is_qkv=True)
        self.o_proj = RowLinear(mapping.tp_size * self.q_size,
                                hidden_size,
                                bias=False,
                                dtype=dtype,
                                tp_group=mapping.tp_group,
                                tp_size=mapping.tp_size)
        self.g_proj = None
        if gating:
            self.g_proj = ColumnLinear(hidden_size,
                                       mapping.tp_size * self.q_size,
                                       bias=False,
                                       dtype=dtype,
                                       tp_group=mapping.tp_group,
                                       tp_size=mapping.tp_size,
                                       gather_output=False)

    def forward(self,
                hidden_states: Tensor,
                biases: Optional[list[Tensor]] = None,
                norm_before_bmm1: bool = False,
                attention_params: AttentionParams = None,
                all_reduce_params: Optional[AllReduceParams] = None):
        qkv = self.qkv_proj(hidden_states, None)

        if False:
            # TODO: Call to alpha-fold self-attention plugin, at here
            context = None
        else:
            # plain TensorRT mode
            def transpose_for_scores(x, is_kv: bool = False):
                _num_attention_heads = self.num_attention_kv_heads if is_kv else self.num_attention_heads
                new_x_shape = concat([
                    shape(x, 0),
                    shape(x, 1), _num_attention_heads, self.attention_head_size
                ])

                return x.view(new_x_shape).permute([0, 2, 1, 3])

            query, key, value = split(
                qkv, [self.attention_hidden_size, self.kv_size, self.kv_size],
                dim=2)

            query = transpose_for_scores(query, is_kv=False)
            key = transpose_for_scores(key, is_kv=True)
            value = transpose_for_scores(value, is_kv=True)
            # At here, query has shape [batch_size, num_heads, seq_len, head_dim]
            # key and value have shape [batch_size, num_heads, seq_len, head_dim]
            mask_bias = None
            triangle_bias = None
            batch_size = shape(query, 0)
            seq_len = shape(query, 2)

            if biases is not None:
                mask_bias = biases[0]
                triangle_bias = biases[1]
                # slice the triangle bias for tp by the head dimension
                if self.tp_size > 1:
                    starts = concat(
                        [0, self.num_attention_heads * self.tp_rank, 0, 0])
                    ends = concat(
                        [1, self.num_attention_heads, seq_len, seq_len])
                    triangle_bias = slice(triangle_bias, starts, ends)

            key = key.permute([0, 1, 3, 2])
            model_type = query.dtype

            # Using attn precision different from model precision to avoid NaN results
            with precision(attention_params.plain_attn_precision):
                query = cast(query, attention_params.plain_attn_precision)
                key = cast(key, attention_params.plain_attn_precision)
                if norm_before_bmm1:
                    query /= self.norm_factor
                attention_scores = matmul(query, key)
                if not norm_before_bmm1:
                    attention_scores /= self.norm_factor
                if mask_bias is not None:
                    attention_scores += mask_bias
                if triangle_bias is not None:
                    attention_scores += triangle_bias

                attention_probs = softmax(attention_scores, dim=-1)
                attention_probs = cast(attention_probs, model_type)
            attention_probs = attention_probs.view(
                concat([
                    shape(attention_probs, 0),
                    shape(attention_probs, 1),
                    shape(attention_probs, 2),
                    shape(value, 2)
                ]))

            context = matmul(attention_probs, value,
                             use_fp32_acc=False).permute([0, 2, 1, 3])
            if self.g_proj is not None:
                g = self.g_proj(hidden_states)
                g = activation(g, trt.ActivationType.SIGMOID)
                g = g.view(
                    concat([
                        batch_size, seq_len, self.num_attention_heads,
                        self.attention_head_size
                    ]))
                context *= g
            context = context.view(
                concat([
                    shape(context, 0),
                    shape(context, 1),
                    self.num_attention_heads * self.attention_head_size
                ]))
            context = self.o_proj(context, all_reduce_params=all_reduce_params)
        return context


class SelfAttentionPairBias(Module):

    def __init__(self,
                 *,
                 local_layer_idx: int,
                 c_s: int,
                 c_z: int,
                 num_heads: int,
                 initial_norm: bool = True,
                 inf: float = 1e6,
                 eps: float = 1e-05,
                 dtype: str = None,
                 mapping: Mapping = Mapping()):
        super().__init__()
        self.local_layer_idx = local_layer_idx
        self.c_s = c_s
        self.c_z = c_z
        self.attention_head_size = c_s // num_heads
        self.initial_norm = initial_norm
        self.inf = inf

        self.num_attention_kv_heads = num_heads
        # This equal to 1 for self-attention
        self.num_key_value_groups = num_heads // self.num_attention_kv_heads

        assert num_heads % mapping.tp_size == 0
        self.num_attention_heads = num_heads // mapping.tp_size
        self.num_attention_kv_heads = self.num_attention_kv_heads // mapping.tp_size
        self.q_size = self.num_attention_heads * self.attention_head_size
        self.kv_size = self.num_attention_kv_heads * self.attention_head_size

        self.tp_group = mapping.tp_group
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.dtype = dtype

        self.norm_factor = math.sqrt(self.attention_head_size)

        self.norm_s = None
        if initial_norm:
            self.norm_s = LayerNorm(normalized_shape=[c_s],
                                    eps=eps,
                                    dtype=dtype,
                                    tp_size=1,
                                    tp_dim=0)

        # Couldn't fused q,k,v as one because of the different bias
        self.proj_q = ColumnLinear(self.c_s,
                                   mapping.tp_size * self.q_size,
                                   bias=True,
                                   dtype=dtype,
                                   tp_group=mapping.tp_group,
                                   tp_size=mapping.tp_size,
                                   gather_output=False)
        # Fused k,v at here
        self.proj_kv = ColumnLinear(self.c_s,
                                    2 * mapping.tp_size * self.kv_size,
                                    bias=False,
                                    dtype=dtype,
                                    tp_group=mapping.tp_group,
                                    tp_size=mapping.tp_size,
                                    gather_output=False)
        self.proj_g = ColumnLinear(self.c_s,
                                   mapping.tp_size * self.q_size,
                                   bias=False,
                                   dtype=dtype,
                                   tp_group=mapping.tp_group,
                                   tp_size=mapping.tp_size,
                                   gather_output=False)
        self.proj_z_norm = LayerNorm(normalized_shape=[c_z],
                                     dtype=dtype,
                                     eps=eps,
                                     tp_size=1,
                                     tp_dim=0)
        self.proj_z = ColumnLinear(self.c_z,
                                   mapping.tp_size * self.num_attention_heads,
                                   bias=False,
                                   dtype=dtype,
                                   tp_group=mapping.tp_group,
                                   tp_size=mapping.tp_size,
                                   gather_output=False)
        self.proj_o = RowLinear(mapping.tp_size * self.q_size,
                                self.c_s,
                                bias=False,
                                dtype=dtype,
                                tp_group=mapping.tp_group,
                                tp_size=mapping.tp_size)

    def forward(self,
                s: Tensor,
                z: Tensor,
                mask: Tensor,
                norm_before_bmm1: bool = False,
                attention_params: AttentionParams = None,
                all_reduce_params: Optional[AllReduceParams] = None):
        if self.norm_s:
            norm_s = self.norm_s(s)
        else:
            norm_s = s

        query = self.proj_q(norm_s)
        kv = self.proj_kv(norm_s)
        key, value = split(kv, [self.kv_size, self.kv_size], dim=-1)
        res_s = self.proj_g(norm_s)
        res_s = activation(res_s, trt.ActivationType.SIGMOID)

        if False:
            # TODO: Call to alpha-fold self-attention plugin, at here
            context = None
        else:
            # plain TensorRT mode
            def transpose_for_scores(x, is_kv: bool = False):
                _num_attention_heads = self.num_attention_kv_heads if is_kv else self.num_attention_heads
                new_x_shape = concat([
                    shape(x, 0),
                    shape(x, 1), _num_attention_heads, self.attention_head_size
                ])

                return x.view(new_x_shape).permute([0, 2, 1, 3])

            query = transpose_for_scores(query, is_kv=False)
            key = transpose_for_scores(key, is_kv=True)
            value = transpose_for_scores(value, is_kv=True)
            # At here, query has shape [batch_size, num_heads, seq_len, attention_head_size]
            # key and value have also the same shape
            key = key.permute([0, 1, 3, 2])
            model_type = query.dtype
            pair_bias = self.proj_z(z)
            pair_bias = pair_bias.permute([0, 3, 1,
                                           2])  # [B, N, N, H] -> [B, H, N, N]
            mask = cast(mask, 'float32')
            mask_bias = (1 - expand_dims(mask, [1, 2])) * (-self.inf)

            with precision(attention_params.plain_attn_precision):
                query = cast(query, attention_params.plain_attn_precision)
                key = cast(key, attention_params.plain_attn_precision)
                value = cast(value, attention_params.plain_attn_precision)
                pair_bias = cast(pair_bias,
                                 attention_params.plain_attn_precision)
                mask_bias = cast(mask_bias,
                                 attention_params.plain_attn_precision)
                if norm_before_bmm1:
                    query /= self.norm_factor
                attention_scores = matmul(query, key)
                if not norm_before_bmm1:
                    attention_scores /= self.norm_factor
                attention_scores += mask_bias
                attention_scores += pair_bias
                attention_probs = softmax(attention_scores, dim=-1)
            attention_probs = cast(attention_probs, model_type)
            context = matmul(attention_probs, value,
                             use_fp32_acc=False).permute([0, 2, 1, 3])
            context = context.view(
                concat([
                    shape(context, 0),
                    shape(context, 1),
                    self.num_attention_heads * self.attention_head_size
                ]))
            context = context * res_s
            context = self.proj_o(context, all_reduce_params=all_reduce_params)
        return context
