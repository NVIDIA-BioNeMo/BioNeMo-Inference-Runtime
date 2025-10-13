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

from tensorrt_llm import functional as trt_f
from tensorrt_llm.functional import (AllReduceParams, Tensor, activation, cast,
                                     concat, constant_to_tensor_, expand_dims,
                                     matmul, not_op, shape, slice, softmax,
                                     split, squeeze)
from tensorrt_llm.layers.linear import ColumnLinear, RowLinear
from tensorrt_llm.layers.normalization import LayerNorm
from tensorrt_llm.logger import logger
from tensorrt_llm.module import Module

from tensorrt_bionemo._trt.functional import (dynamic_const_tensor,
                                              triangle_attention)
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
                 bias_flags: dict[str, bool] = {
                     "q": False,
                     "k": False,
                     "v": False,
                     "g": False,
                     "z": False,
                     "o": False,
                 },
                 gating: bool = True,
                 dtype: str = None,
                 triangle_attn_backend: str = 'VANILLA',
                 support_batch: bool = False,
                 mapping: Mapping = Mapping(),
                 fallback_threshold=0):
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
        self.bias_flags = bias_flags
        self.fallback_threshold = fallback_threshold
        self.norm_factor = math.sqrt(self.attention_head_size)

        self.q_size = self.num_attention_heads * self.attention_head_size
        self.kv_size = self.num_attention_kv_heads * self.attention_head_size

        self.qkv_proj = ColumnLinear(
            hidden_size,
            mapping.tp_size * self.q_size + 2 * mapping.tp_size * self.kv_size,
            bias=bias_flags["q"] or bias_flags["k"] or bias_flags["v"],
            dtype=dtype,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            gather_output=False,
            is_qkv=True)
        self.o_proj = RowLinear(mapping.tp_size * self.q_size,
                                hidden_size,
                                bias=bias_flags["o"],
                                dtype=dtype,
                                tp_group=mapping.tp_group,
                                tp_size=mapping.tp_size)
        self.g_proj = None
        if gating:
            self.g_proj = ColumnLinear(hidden_size,
                                       mapping.tp_size * self.q_size,
                                       bias=bias_flags["g"],
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
            batch_dims = 1
        else:
            bs = 1
            batch_dims = 0
        si = shape(hidden_states, batch_dims + 0)
        sj = shape(hidden_states, batch_dims + 1)
        qkv = self.qkv_proj(hidden_states,
                            None)  # [B, I, J, 3*H*D] or [I, J, 3*H*D]
        mask_bias = None
        triangle_bias = None

        if biases is not None:
            mask_bias = biases[0]
            triangle_bias = biases[1]
            if triangle_bias is not None:
                # slice the triangle bias for tp by the head dimension
                if self.tp_size > 1:
                    starts = concat(
                        [0, self.num_attention_heads * self.tp_rank, 0, 0])
                    ends = concat([bs, self.num_attention_heads, sj, sj])
                    triangle_bias = slice(triangle_bias, starts, ends)
                if self.support_batch:
                    triangle_bias = triangle_bias.unsqueeze(1)

        def transpose_for_scores(x, is_kv: bool = False):
            """
            Transpose the tensor for the scores computation. Used for CUEQUIV backend and plain TensorRT mode.
            """
            _num_attention_heads = self.num_attention_kv_heads if is_kv else self.num_attention_heads
            if self.support_batch:
                new_x_shape = concat([
                    bs, si, sj, _num_attention_heads, self.attention_head_size
                ])
                return x.view(new_x_shape).permute([0, 1, 3, 2,
                                                    4])  # [B, I, H, J, D]
            else:
                new_x_shape = concat(
                    [si, sj, _num_attention_heads, self.attention_head_size])
                return x.view(new_x_shape).permute([0, 2, 1, 3])  # [I, H, J, D]

        def vanilla_attention(query, key, value, triangle_bias, mask_bias):
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
            return context

        query, key, value = split(
            qkv, [self.attention_hidden_size, self.kv_size, self.kv_size],
            dim=-1)

        if self.triangle_attn_backend != 'VANILLA':
            logger.debug(
                f"Using {self.triangle_attn_backend} triangle attention backend, {self.dtype}"
            )
            if self.fallback_threshold > 0:
                # This experiment: Please do not use this fallback mechanism at now.
                hs_shape = shape(hidden_states)
                dim_sj = slice(hs_shape, starts=[batch_dims + 1], sizes=[1])
                threshold = constant_to_tensor_(
                    self.fallback_threshold,  # Threshold value
                    dtype=dim_sj.dtype,
                    to_array=False)

                condition = trt_f.gt(dim_sj, threshold).squeeze(0, False)
                cond_node = trt_f.Conditional(condition)
                query = cond_node.add_input(query)
                key = cond_node.add_input(key)
                value = cond_node.add_input(value)
                triangle_bias = cond_node.add_input(triangle_bias)
                if mask_bias is not None:
                    mask_bias = cond_node.add_input(mask_bias)
                # if sj < threshold, just call vanilla attention
                fallback = vanilla_attention(query, key, value, triangle_bias,
                                             mask_bias)

            context = None

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
                if self.support_batch:
                    bias_shape = concat([bs * self.num_attention_heads, sj, sj])
                    triangle_bias = triangle_bias.view(bias_shape)
                else:
                    triangle_bias = triangle_bias.squeeze(0, False)

                context, _ = triangle_attention(
                    query,
                    key,
                    value,
                    triangle_bias,
                    mask_bias,
                    self.num_attention_heads,
                    self.attention_head_size,
                    dtype=query.dtype,
                    backend=self.triangle_attn_backend)  # [B*H, I, J, D]
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
            elif self.triangle_attn_backend == "CUEQUIV":
                query = transpose_for_scores(
                    query, is_kv=False)  # [B, I, H, J, D] or [I, H, J, D]
                key = transpose_for_scores(
                    key, is_kv=True)  # [B, I, H, J, D] or [I, H, J, D]
                value = transpose_for_scores(
                    value, is_kv=True)  # [B, I, H, J, D] or [I, H, J, D]
                if mask_bias is not None:
                    mask_bias = cast(mask_bias, "bool")
                    mask_bias = not_op(
                        mask_bias)  # flip the mask for CUEQUIV backend
                if not self.support_batch:
                    # CUEQUIV requires the batch dimension
                    if mask_bias is not None:
                        mask_bias = mask_bias.unsqueeze(0)  # [B, I, 1, 1, J]
                    query = query.unsqueeze(0)
                    key = key.unsqueeze(0)
                    value = value.unsqueeze(0)
                    triangle_bias = triangle_bias.unsqueeze(
                        1)  # [B, 1, H, J, J]
                context, _ = triangle_attention(
                    query,
                    key,
                    value,
                    triangle_bias,
                    mask_bias,
                    self.num_attention_heads,
                    self.attention_head_size,
                    dtype=query.dtype,
                    backend=self.triangle_attn_backend)  # [B, I, H, J, D]
                context = context.permute([0, 1, 3, 2, 4])  # [B, I, J, H, D]
                if not self.support_batch:
                    context = context.squeeze(0, False)
            # closing conditional
            if self.fallback_threshold > 0:
                context = cond_node.add_output(context, fallback)
        else:
            # plain TensorRT mode
            context = vanilla_attention(query, key, value, triangle_bias,
                                        mask_bias)

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

    def __init__(
        self,
        *,
        local_layer_idx: int,
        c_s: int,
        c_z: int,
        num_heads: int,
        initial_norm: bool = True,
        bias_flags: dict[str, bool] = {
            "q": True,
            "k": False,
            "v": False,
            "g": False,
            "z": False,
            "norm_z": True,
            "o": False,
        },
        transform_mask: bool = True,
        inf: float = 1e9,
        eps: float = 1e-05,
        dtype: str = None,
        need_project_z: bool = True,
        max_batch_size: int = 1,
        mapping: Mapping = Mapping()):
        super().__init__()
        self.local_layer_idx = local_layer_idx
        self.c_s = c_s
        self.c_z = c_z
        self.attention_head_size = c_s // num_heads
        self.initial_norm = initial_norm
        self.inf = inf
        self.max_batch_size = max_batch_size  # this for multi-diffusion samples
        self.transform_mask = transform_mask

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
                                   bias=bias_flags["q"],
                                   dtype=dtype,
                                   tp_group=mapping.tp_group,
                                   tp_size=mapping.tp_size,
                                   gather_output=False)
        # Fused k,v at here.
        # TODO: potential for fused q,k,v here, if bias_flags["q"] = bias_flags["k"] = bias_flags["v"]
        self.proj_kv = ColumnLinear(self.c_s,
                                    2 * mapping.tp_size * self.kv_size,
                                    bias=bias_flags["k"] or bias_flags["v"],
                                    dtype=dtype,
                                    tp_group=mapping.tp_group,
                                    tp_size=mapping.tp_size,
                                    gather_output=False)
        self.proj_g = ColumnLinear(self.c_s,
                                   mapping.tp_size * self.q_size,
                                   bias=bias_flags["g"],
                                   dtype=dtype,
                                   tp_group=mapping.tp_group,
                                   tp_size=mapping.tp_size,
                                   gather_output=False)
        self.need_project_z = need_project_z
        if need_project_z:
            self.proj_z_norm = LayerNorm(normalized_shape=[c_z],
                                         dtype=dtype,
                                         eps=eps,
                                         tp_size=1,
                                         tp_dim=0,
                                         bias=bias_flags.get("norm_z", True))
            self.proj_z = ColumnLinear(self.c_z,
                                       mapping.tp_size *
                                       self.num_attention_heads,
                                       bias=bias_flags["z"],
                                       dtype=dtype,
                                       tp_group=mapping.tp_group,
                                       tp_size=mapping.tp_size,
                                       gather_output=False)
        self.proj_o = RowLinear(mapping.tp_size * self.q_size,
                                self.c_s,
                                bias=bias_flags["o"],
                                dtype=dtype,
                                tp_group=mapping.tp_group,
                                tp_size=mapping.tp_size)

    def forward(self,
                s: Tensor,
                z: Tensor,
                mask: Tensor,
                norm_before_bmm1: bool = False,
                compute_pair_bias: bool = True,
                attention_params: AttentionParams = None,
                all_reduce_params: Optional[AllReduceParams] = None):
        """
        Implementation of the self-attention pair bias in TensorRT.

        Args:
            s: [B, I, C_S] or [B, J, I, C_S]
            z: [1, I, I, C_Z] or [B, I, I, C_Z]
            mask: [B, I] or [B, J, I]
        Note: For the case s is [B, J, I, C_S], the mask is [B, J, I], please use the TriangleAttention layer instead.
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
                if x.ndim() == 3:
                    _num_attention_heads = self.num_attention_kv_heads if is_kv else self.num_attention_heads
                    new_x_shape = concat([
                        shape(x, 0),
                        shape(x, 1), _num_attention_heads,
                        self.attention_head_size
                    ])  # [B, I, H, D]

                    return x.view(new_x_shape).permute([0, 2, 1,
                                                        3])  # [B, H, I, D]
                if x.ndim() == 4:
                    _num_attention_heads = self.num_attention_kv_heads if is_kv else self.num_attention_heads
                    new_x_shape = concat([
                        shape(x, 0),
                        shape(x, 1),
                        shape(x, 2), _num_attention_heads,
                        self.attention_head_size
                    ])  # [B, J, I, H, D]
                    return x.view(new_x_shape).permute([0, 1, 3, 2,
                                                        4])  # [B, J, H, I, D]
                assert False, f"Invalid input shape, not supported number of dimensions {x.ndim()}"

            query = transpose_for_scores(query, is_kv=False)
            key = transpose_for_scores(key, is_kv=True)
            value = transpose_for_scores(
                value, is_kv=True)  # [B, H, I, D] or [B, J, H, I, D]
            # At here, query has shape [batch_size, num_heads, seq_len, attention_head_size]
            # key and value have also the same shape
            if key.ndim() == 4:
                key = key.permute([0, 1, 3, 2])  # [B, H, D, I]
            elif key.ndim() == 5:
                key = key.permute([0, 1, 2, 4, 3])  # [B, J, H, D, I]
            else:
                assert False, f"Invalid input shape, not supported number of dimensions {key.ndim()}"
            model_type = query.dtype
            pair_bias = z
            if compute_pair_bias and self.need_project_z:
                z = self.proj_z_norm(z)
                pair_bias = self.proj_z(z)  # [B, I, I, H]
                pair_bias = pair_bias.permute(
                    [0, 3, 1, 2])  # [B, I, I, H] -> [B, H, I, I]
                if key.ndim() == 5:
                    pair_bias = pair_bias.unsqueeze(1)  # [B, 1, H, I, I]

            # For multi-diffusion samples, but the broadcasting automatically do the repeat_interleave
            # if self.max_batch_size > 1:
            #     batch_size = shape(s, 0)
            #     pair_bias = repeat_interleave(pair_bias, batch_size, dim=0)
            mask = cast(mask, model_type)
            inf_const = constant_to_tensor_(-self.inf,
                                            dtype=model_type,
                                            to_array=False)
            one_const = constant_to_tensor_(1.0,
                                            dtype=model_type,
                                            to_array=False)
            if self.transform_mask:
                if key.ndim() == 4:
                    mask_bias = (one_const - expand_dims(
                        mask, [1, 2])) * inf_const  # [B, 1, 1, I]
                elif key.ndim() == 5:
                    mask_bias = (one_const - expand_dims(
                        mask, [2, 3])) * inf_const  # [B, J, 1, 1, I]
                else:
                    assert False, f"Invalid input shape, not supported number of dimensions {key.ndim()}"
            else:
                mask_bias = mask
            norm_factor_const = constant_to_tensor_(self.norm_factor,
                                                    dtype=model_type,
                                                    to_array=False)
            if norm_before_bmm1:
                query /= norm_factor_const
            attention_scores = matmul(query,
                                      key)  # [B, H, I, I] or [B, J, H, I, I]
            if not norm_before_bmm1:
                attention_scores /= norm_factor_const
            if pair_bias is not None:
                attention_scores += pair_bias
            attention_scores += mask_bias
            attention_probs = softmax(attention_scores, dim=-1)
            attention_probs = cast(attention_probs, model_type)

            if key.ndim() == 4:
                context = matmul(attention_probs, value,
                                 use_fp32_acc=False).permute([0, 2, 1, 3
                                                              ])  # [B, I, H, D]
                context = context.view(
                    concat([
                        shape(context, 0),
                        shape(context, 1),
                        self.num_attention_heads * self.attention_head_size
                    ]))
            elif key.ndim() == 5:
                context = matmul(attention_probs, value,
                                 use_fp32_acc=False).permute(
                                     [0, 1, 3, 2, 4])  # [B, J, I, H, D]
                context = context.view(
                    concat([
                        shape(context, 0),
                        shape(context, 1),
                        shape(context, 2),
                        self.num_attention_heads * self.attention_head_size
                    ]))
            else:
                assert False, f"Invalid input shape, not supported number of dimensions {key.ndim()}"
            context = context * res_s  # [B, I, H*D] or [B, J, I, H*D]
            context = self.proj_o(context, all_reduce_params=all_reduce_params
                                  )  # [B, I, H*D] or [B, J, I, H*D]
        return context


class MSAAttention(Module):
    """ Implementation of OpenFold MSAAttention layer."""

    def __init__(self,
                 *,
                 local_layer_idx: int,
                 c_in: int,
                 num_heads: int,
                 c_z: Optional[int] = None,
                 triangle_attn_backend: str = 'VANILLA',
                 support_batch: bool = True,
                 need_project_z: bool = True,
                 transpose_input: bool = False,
                 bias_flags: dict[str, bool] = {"z": False},
                 eps: float = 1e-5,
                 inf: float = 1e9,
                 dtype: str = None,
                 mapping: Mapping = Mapping()):
        super().__init__()
        self.local_layer_idx = local_layer_idx
        self.num_heads = num_heads
        self.c_in = c_in
        self.c_z = c_z
        self.inf = inf
        self.support_batch = support_batch
        self.dtype = dtype
        self.transpose_input = transpose_input
        self.triangle_attn_backend = triangle_attn_backend

        self.layer_norm_m = LayerNorm(normalized_shape=[c_in],
                                      dtype=dtype,
                                      eps=eps,
                                      tp_size=1,
                                      tp_dim=0)
        self.proj_z_norm = None
        self.proj_z = None
        if need_project_z:
            self.proj_z_norm = LayerNorm(normalized_shape=[c_z],
                                         dtype=dtype,
                                         eps=eps,
                                         tp_size=1,
                                         tp_dim=0)
            self.proj_z = ColumnLinear(self.c_z,
                                       num_heads,
                                       bias=bias_flags["z"],
                                       dtype=dtype,
                                       tp_group=mapping.tp_group,
                                       tp_size=mapping.tp_size,
                                       gather_output=True)
        self.mha = TriangleAttention(
            local_layer_idx=local_layer_idx,
            hidden_size=c_in,
            num_attention_heads=num_heads,
            triangle_attn_backend=triangle_attn_backend,
            support_batch=support_batch,
            mapping=mapping,
            dtype=dtype,
            bias_flags={
                "q": False,
                "k": False,
                "v": False,
                "g": True,
                "z": False,
                "o": True,
            })

    def forward(self,
                m: Tensor,
                z: Tensor,
                mask: Tensor,
                attention_params: AttentionParams = None,
                all_reduce_params: Optional[AllReduceParams] = None):
        """
        Args:
            m: [B, J, I, c_in]
            z: [B, I, I, c_z]
            mask: [B, J, I]
        """
        if self.transpose_input:
            m = m.permute([0, 2, 1, 3])
            mask = mask.permute([0, 2, 1])

        inf_const = constant_to_tensor_(self.inf,
                                        dtype=mask.dtype,
                                        to_array=False)
        one_const = constant_to_tensor_(1.0, dtype=mask.dtype, to_array=False)

        # Compute mask bias
        mask_bias = ((mask - one_const) * inf_const)
        if self.support_batch:
            mask_bias = expand_dims(mask_bias, [2, 3])
        else:
            mask_bias = expand_dims(mask_bias, [1, 2])
        if self.proj_z_norm and self.proj_z and z is not None:
            z = self.proj_z_norm(z)
            z = self.proj_z(z)
            z = z.permute([0, 3, 1, 2])  # [B, N, N, H] -> [B, H, N, N]
        else:
            if self.triangle_attn_backend != 'VANILLA':
                # set z to zeros for non-vanilla triangle attention
                z_shape = concat(
                    [shape(m, 0), self.num_heads,
                     shape(m, 2),
                     shape(m, 2)])
                z = dynamic_const_tensor(z_shape, value=0.0, dtype=self.dtype)
        biases = [mask_bias, z]
        m = self.layer_norm_m(m)

        o = self.mha(m,
                     biases=biases,
                     attention_params=attention_params,
                     all_reduce_params=all_reduce_params)
        if self.transpose_input:
            o = o.permute([0, 2, 1, 3])
        return o
