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

from tensorrt_llm.functional import (AllReduceParams, Tensor, activation, cast,
                                     concat, constant_to_tensor_, expand_dims,
                                     matmul, shape, slice, softmax, split,
                                     squeeze)
from tensorrt_llm.layers.linear import ColumnLinear, RowLinear
from tensorrt_llm.layers.normalization import LayerNorm
from tensorrt_llm.logger import logger
from tensorrt_llm.module import Module

from tensorrt_bionemo._trt.functional import triangle_attention
from tensorrt_bionemo.mapping import Mapping


class AttentionParams(object):

    def __init__(self):
        # TODO: Add attention params
        pass


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
                 triangle_attn_backend: str = 'VANILLA',
                 support_batch: bool = False,
                 mapping: Mapping = Mapping()):
        super().__init__()
        self.local_layer_idx = local_layer_idx
        self.triangle_attn_backend = triangle_attn_backend
        self.support_batch = support_batch
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
        """
        Implementation of the triangle attention in TensorRT.

        Args:
            For batch support:
                hidden_states: [B, I, J, F]
                biases: Include two biases:
                    - mask_bias: [B, I, 1, 1, J]
                    - triangle_bias: [B, H, J, J]
            Without batch support:
                hidden_states: [I, J, F]
                biases: Include two biases:
                    - mask_bias: [I, 1, 1, J]
                    - triangle_bias: [B, H, J, J]
        """
        if self.support_batch:
            bs = shape(hidden_states, 0)
            si = shape(hidden_states, 1)
            sj = shape(hidden_states, 2)
        else:
            bs = 1
            si = shape(hidden_states, 0)
            sj = shape(hidden_states, 1)
        qkv = self.qkv_proj(hidden_states,
                            None)  # [B, I, J, 3*H*D] or [I, J, 3*H*D]
        mask_bias = None
        triangle_bias = None

        if biases is not None:
            mask_bias = biases[0]
            triangle_bias = biases[1]
            # slice the triangle bias for tp by the head dimension
            if self.tp_size > 1:
                if self.support_batch:
                    starts = concat(
                        [0, self.num_attention_heads * self.tp_rank, 0, 0])
                    ends = concat([bs, self.num_attention_heads, sj, sj])
                else:
                    starts = concat(
                        [0, self.num_attention_heads * self.tp_rank, 0, 0])
                    ends = concat([1, self.num_attention_heads, sj, sj])
                triangle_bias = slice(triangle_bias, starts, ends)
            if self.support_batch:
                triangle_bias = triangle_bias.unsqueeze(1)

        if self.triangle_attn_backend != 'VANILLA':
            assert self.triangle_attn_backend == "TRIFAST", "Only TRIFAST is supported for now"
            logger.debug(
                f"Using {self.triangle_attn_backend} triangle attention backend, {self.dtype}"
            )
            context = None

            query, key, value = split(
                qkv, [self.attention_hidden_size, self.kv_size, self.kv_size],
                dim=-1)
            if self.triangle_attn_backend == "TRIFAST":

                def transpose_for_bh(x, is_kv: bool = False):
                    _num_attention_heads = self.num_attention_kv_heads if is_kv else self.num_attention_heads
                    if self.support_batch:
                        new_x_shape = concat([
                            bs, si, sj, _num_attention_heads,
                            self.attention_head_size
                        ])
                        x = x.view(new_x_shape).permute([0, 3, 1, 2, 4])
                        bh_shape = concat([
                            bs * _num_attention_heads, si, sj,
                            self.attention_head_size
                        ])
                        return x.view(bh_shape)
                    else:
                        new_x_shape = concat([
                            si, sj, _num_attention_heads,
                            self.attention_head_size
                        ])
                        x = x.view(new_x_shape).permute([2, 0, 1, 3])
                        return x

                query = transpose_for_bh(query, is_kv=False)  # [B*H, I, J, D]
                key = transpose_for_bh(key, is_kv=True)  # [B*H, I, J, D]
                value = transpose_for_bh(value, is_kv=True)  # [B*H, I, J, D]
                assert triangle_bias is not None, "Triangle bias is required for triangle attention"
                assert mask_bias is not None, "Mask bias is required for triangle attention"

                if not self.support_batch:
                    mask_bias = mask_bias.unsqueeze(0)
                mask_bias = squeeze(mask_bias, (2, 3))
                mask_bias = cast(mask_bias, "bool")
                bias_shape = concat([bs * self.num_attention_heads, sj, sj])
                triangle_bias = triangle_bias.view(bias_shape)

                context, _ = triangle_attention(
                    query,
                    key,
                    value,
                    triangle_bias,
                    mask_bias,
                    self.num_attention_heads,
                    self.attention_head_size,
                    dtype=query.dtype,
                    use_trifast=True)  # [B*H, I, J, D]
                if self.support_batch:
                    context = context.view(
                        concat([
                            bs, self.num_attention_heads, si, sj,
                            self.attention_head_size
                        ]))  # [B, H, I, J, D]
                    context = context.permute([0, 2, 3, 1,
                                               4])  # [B, I, J, H, D]
                else:
                    context = context.view(
                        concat([
                            self.num_attention_heads, si, sj,
                            self.attention_head_size
                        ]))  # [H, I, J, D]
                    context = context.permute([1, 2, 0, 3])  # [I, J, H, D]
        else:
            # plain TensorRT mode
            def transpose_for_scores(x, is_kv: bool = False):
                _num_attention_heads = self.num_attention_kv_heads if is_kv else self.num_attention_heads
                if self.support_batch:
                    new_x_shape = concat([
                        bs, si, sj, _num_attention_heads,
                        self.attention_head_size
                    ])
                    return x.view(new_x_shape).permute([0, 1, 3, 2,
                                                        4])  # [B, I, H, J, D]
                else:
                    new_x_shape = concat([
                        si, sj, _num_attention_heads, self.attention_head_size
                    ])
                    return x.view(new_x_shape).permute([0, 2, 1,
                                                        3])  # [I, H, J, D]

            query, key, value = split(
                qkv, [self.attention_hidden_size, self.kv_size, self.kv_size],
                dim=-1)

            query = transpose_for_scores(
                query, is_kv=False)  # [B, I, H, J, D] or [I, H, J, D]
            key = transpose_for_scores(
                key, is_kv=True)  # [B, I, H, J, D] or [I, H, J, D]
            value = transpose_for_scores(
                value, is_kv=True)  # [B, I, H, J, D] or [I, H, J, D]
            if self.support_batch:
                key = key.permute([0, 1, 2, 4, 3])  # [B, I, H, D, J] # K^T
            else:
                key = key.permute([0, 1, 3, 2])  # [I, H, D, J] # K^T
            norm_factor_const = constant_to_tensor_(self.norm_factor,
                                                    dtype=query.dtype,
                                                    to_array=False)
            if norm_before_bmm1:
                query /= norm_factor_const
            attention_scores = matmul(query, key)
            if not norm_before_bmm1:
                attention_scores /= norm_factor_const
            if mask_bias is not None:
                attention_scores += mask_bias
            if triangle_bias is not None:
                attention_scores += triangle_bias
            attention_probs = softmax(attention_scores,
                                      dim=-1)  # [B, I, H, J, J] or [I, H, J, J]
            if self.support_batch:
                context = matmul(attention_probs, value,
                                 use_fp32_acc=False).permute(
                                     [0, 1, 3, 2, 4])  # [B, I, J, H, D]
            else:
                context = matmul(attention_probs, value,
                                 use_fp32_acc=False).permute([0, 2, 1, 3
                                                              ])  # [I, J, H, D]

        if self.g_proj is not None:
            g = self.g_proj(hidden_states)  # [B, I, J, H*D] or [I, J, H*D]
            g = activation(g, trt.ActivationType.SIGMOID)
            if self.support_batch:
                g = g.view(
                    concat([
                        bs, si, sj, self.num_attention_heads,
                        self.attention_head_size
                    ]))  # [B, I, J, H, D]
            else:
                g = g.view(
                    concat([
                        si, sj, self.num_attention_heads,
                        self.attention_head_size
                    ]))  # [I, J, H, D]
            context *= g
        if self.support_batch:
            context = context.view(
                concat([
                    bs, si, sj,
                    self.num_attention_heads * self.attention_head_size
                ]))  # [B, I, J, H*D]
        else:
            context = context.view(
                concat([
                    si, sj, self.num_attention_heads * self.attention_head_size
                ]))  # [I, J, H*D]
        context = self.o_proj(
            context,
            all_reduce_params=all_reduce_params)  # [B, I, J, F] or [I, J, F]
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
        self.inf = 1e6

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
        """
        Implementation of the self-attention pair bias in TensorRT.

        Args:
            s: [B, I, C_S]
            z: [B, I, I, C_Z]
            mask: [B, I]
        """
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
            z = self.proj_z_norm(z)
            pair_bias = self.proj_z(z)  # [B, N, N, H]
            pair_bias = pair_bias.permute([0, 3, 1,
                                           2])  # [B, N, N, H] -> [B, H, N, N]
            mask = cast(mask, model_type)
            inf_const = constant_to_tensor_(-self.inf,
                                            dtype=model_type,
                                            to_array=False)
            one_const = constant_to_tensor_(1.0,
                                            dtype=model_type,
                                            to_array=False)
            mask_bias = (one_const - expand_dims(mask, [1, 2])) * inf_const
            norm_factor_const = constant_to_tensor_(self.norm_factor,
                                                    dtype=model_type,
                                                    to_array=False)
            if norm_before_bmm1:
                query /= norm_factor_const
            attention_scores = matmul(query, key)
            if not norm_before_bmm1:
                attention_scores /= norm_factor_const
            attention_scores += pair_bias
            attention_scores += mask_bias
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
