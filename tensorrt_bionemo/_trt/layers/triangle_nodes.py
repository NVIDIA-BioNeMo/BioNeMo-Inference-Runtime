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

import tensorrt as trt
from tensorrt_llm.functional import (AllReduceParams, Tensor, activation,
                                     allgather, cast, concat,
                                     constant_to_tensor_, einsum, expand_dims,
                                     floordiv, permute, shape, slice, split)
from tensorrt_llm.layers.linear import ColumnLinear
from tensorrt_llm.layers.normalization import LayerNorm
from tensorrt_llm.module import Module

from tensorrt_bionemo._trt.functional import chunk_loop, send_recv
from tensorrt_bionemo.mapping import Mapping

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
        triangle_attn_backend: str = 'VANILLA',
        support_batch: bool = True,
        fallback_threshold=0,
        mapping: Mapping = Mapping()):
        super().__init__()
        self.local_layer_idx = local_layer_idx
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.node_type = node_type
        self.inf = inf
        self.triangle_attn_backend = triangle_attn_backend
        self.dcp_size = mapping.dcp_size
        self.dcp_rank = mapping.dcp_rank
        self.dcp_group = mapping.dcp_group
        self.support_batch = support_batch
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.tp_group = mapping.tp_group
        self.fallback_threshold = fallback_threshold
        self.chunk_size = chunk_size

        assert self.num_heads % self.tp_size == 0, \
            "num_attention_heads must be divisible by tp_size"

        if chunk_size > 0:
            assert self.chunk_size % self.dcp_size == 0, \
                "chunk_size must be divisible by dcp_size"
            self.chunk_size = chunk_size // self.dcp_size
        self.layer_norm = LayerNorm(normalized_shape=[self.c_in],
                                    eps=eps,
                                    dtype=dtype)
        self.linear = ColumnLinear(self.c_in,
                                   self.num_heads,
                                   bias=False,
                                   dtype=dtype,
                                   tp_group=self.tp_group,
                                   tp_size=self.tp_size,
                                   gather_output=True)
        self.mha = TriangleAttention(
            local_layer_idx=self.local_layer_idx,
            hidden_size=self.c_in,
            num_attention_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            dtype=dtype,
            bias=False,
            gating=True,
            triangle_attn_backend=self.triangle_attn_backend,
            support_batch=self.support_batch,
            fallback_threshold=self.fallback_threshold,
            mapping=mapping)

    def forward(self,
                x: Tensor,
                mask: Tensor,
                attention_params: AttentionParams = None,
                all_reduce_params: Optional[AllReduceParams] = None):
        """
        Args:
            x: [B, I, J, F] or [I, J, F]
            mask: [B, I, J] or [I, J]
        """
        if self.node_type == TriangleAttentionNodeType.ENDING:
            if self.support_batch:
                x = x.transpose(1, 2)
                mask = mask.transpose(1, 2)
            else:
                x = x.transpose(0, 1)
                mask = mask.transpose(0, 1)
        x = self.layer_norm(x)
        inf_const = constant_to_tensor_(self.inf, dtype=x.dtype, to_array=False)
        one_const = constant_to_tensor_(1.0, dtype=x.dtype, to_array=False)
        # Compute mask bias
        mask_bias = ((mask - one_const) * inf_const)
        if self.support_batch:
            mask_bias = expand_dims(mask_bias, [2, 3])
        else:
            mask_bias = expand_dims(mask_bias, [1, 2])

        # Compute triangle bias
        lx = self.linear(x)  # [B, I, J, H] or [I, J, H]

        if self.support_batch:
            triangle_bias = permute(lx, [0, 3, 1, 2])  # [B, H, I, J]
        else:
            triangle_bias = permute(lx, [2, 0, 1])  # [I, H, J]
            triangle_bias = triangle_bias.unsqueeze(0)

        # First if dcp_size > 1, we need to split the input by dcp_size
        if self.support_batch:
            bs = shape(x, 0)
            si = shape(x, 1)
            sj = shape(x, 2)
        else:
            si = shape(x, 0)
            sj = shape(x, 1)
            bs = 1

        if self.dcp_size > 1:
            slice_size = floordiv(si, self.dcp_size)
            s_idx = slice_size * self.dcp_rank
            # Slice the input
            if self.support_batch:
                starts = concat([0, s_idx, 0, 0])
                sizes = concat([bs, slice_size, si, self.c_in])
            else:
                starts = concat([s_idx, 0, 0])
                sizes = concat([slice_size, si, self.c_in])
            x = slice(x, starts, sizes)

            # Slice the mask bias
            if self.support_batch:
                starts = concat([0, s_idx, 0, 0, 0])
                sizes = concat([bs, slice_size, 1, 1, sj])
            else:
                starts = concat([s_idx, 0, 0, 0])
                sizes = concat([slice_size, 1, 1, sj])
            mask_bias = slice(mask_bias, starts, sizes)

        def _loop_body(sub_chunk):
            sub_x, sub_mask_bias = sub_chunk
            biases = [sub_mask_bias, triangle_bias]
            context = self.mha(sub_x,
                               biases=biases,
                               attention_params=attention_params,
                               all_reduce_params=all_reduce_params)
            return context

        output = chunk_loop([x, mask_bias],
                            self.chunk_size,
                            _loop_body,
                            reshape_output=False)
        if self.dcp_size > 1:
            output = allgather(output, self.dcp_group, gather_dim=1)

        if self.node_type == TriangleAttentionNodeType.ENDING:
            if self.support_batch:
                output = output.transpose(2, 1)
            else:
                output = output.transpose(1, 0)

        return output


class TriangleMultiplicationNode(Module):

    def __init__(
        self,
        *,
        local_layer_idx: int,
        dim: int,
        eps: float = 1e-5,
        multiplication_type:
        TriangleMultiplicationNodeType = TriangleMultiplicationNodeType.
        OUTGOING,
        dtype: str = None,
        support_batch: bool = False,
        mapping: Mapping = Mapping()):
        super().__init__()
        self.local_layer_idx = local_layer_idx
        self.dcp_size = mapping.dcp_size
        self.dcp_rank = mapping.dcp_rank
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.dcp_group = mapping.dcp_group
        self.tp_group = mapping.tp_group
        self.dtype = dtype
        self.support_batch = support_batch
        self.dim = dim // self.tp_size
        self.multiplication_type = multiplication_type

        self.norm_in = LayerNorm(normalized_shape=[self.dim * self.tp_size],
                                 eps=eps,
                                 dtype=dtype)
        self.p_in = ColumnLinear(dim,
                                 2 * dim,
                                 bias=False,
                                 dtype=dtype,
                                 tp_group=self.tp_group,
                                 tp_size=self.tp_size,
                                 gather_output=False)
        self.g_in = ColumnLinear(dim,
                                 2 * dim,
                                 bias=False,
                                 dtype=dtype,
                                 tp_group=self.tp_group,
                                 tp_size=self.tp_size,
                                 gather_output=False)

        self.norm_out = LayerNorm(normalized_shape=[dim],
                                  eps=eps,
                                  dtype="float32")
        self.p_out = ColumnLinear(dim,
                                  dim,
                                  bias=False,
                                  dtype="float32",
                                  tp_group=self.tp_group,
                                  tp_size=self.tp_size,
                                  gather_output=True)
        self.g_out = ColumnLinear(dim,
                                  dim,
                                  bias=False,
                                  dtype="float32",
                                  tp_group=self.tp_group,
                                  tp_size=self.tp_size,
                                  gather_output=True)
        self.mapping = mapping

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: [B, I, J, D] or [I, J, D]
            mask: [B, I, J] or [I, J]
        Note: The ring-communication on the dcp group (dcp_size > 1) is experimental and may not work,
            or make engines go large and slow than normal. And it will be reworked in the future.
        """
        original_dtype = mask.dtype
        if self.support_batch:
            bs = shape(x, 0)
            si = shape(x, 1)
            sj = shape(x, 2)
            d = shape(x, 3)
        else:
            bs = 1
            si = shape(x, 0)
            sj = shape(x, 1)
            d = shape(x, 2)
        x = self.norm_in(x)
        if self.dcp_size > 1:
            if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
                slice_size = floordiv(si, self.dcp_size)
                s_idx = slice_size * self.dcp_rank
                # slice x
                if self.support_batch:
                    starts = concat([0, s_idx, 0, 0])
                    sizes = concat([bs, slice_size, sj, d])
                else:
                    starts = concat([s_idx, 0, 0])
                    sizes = concat([slice_size, sj, d])
                x = slice(x, starts, sizes)
                # slice mask
                if self.support_batch:
                    starts = concat([0, s_idx, 0])
                    sizes = concat([bs, slice_size, sj])
                else:
                    starts = concat([s_idx, 0])
                    sizes = concat([slice_size, sj])
                mask = slice(mask, starts, sizes)
            elif self.multiplication_type == TriangleMultiplicationNodeType.INCOMING:
                slice_size = floordiv(sj, self.dcp_size)
                s_idx = slice_size * self.dcp_rank
                # slice x
                if self.support_batch:
                    starts = concat([0, 0, s_idx, 0])
                    sizes = concat([bs, si, slice_size, d])
                else:
                    starts = concat([0, s_idx, 0])
                    sizes = concat([si, slice_size, d])
                x = slice(x, starts, sizes)
                # slice mask
                if self.support_batch:
                    starts = concat([0, 0, s_idx])
                    sizes = concat([bs, si, slice_size])
                else:
                    starts = concat([0, s_idx])
                    sizes = concat([si, slice_size])
                mask = slice(mask, starts, sizes)

        x_in = x
        # TODO: SWiGLU, fuse p_in and g_in here
        x = self.p_in(x) * activation(self.g_in(x), trt.ActivationType.SIGMOID)
        x = x * mask.unsqueeze(-1)
        x = cast(x, "float32")
        a, b = split(x, [self.dim, self.dim], dim=-1)

        def _enisum_compute(a_, b_):
            if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
                if self.support_batch:
                    return einsum("bikd,bjkd->bijd", [a_, b_])
                else:
                    return einsum("ikd,jkd->ijd", [a_, b_])
            else:
                if self.support_batch:
                    return einsum("bkid,bkjd->bijd", [a_, b_])
                else:
                    return einsum("kid,kjd->ijd", [a_, b_])

        if self.dcp_size > 1:
            enisum_results = [
                None,
            ] * self.dcp_size
            enisum_results[self.dcp_rank] = _enisum_compute(a, b)
            if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
                b_recv = b
                for i in range(1, self.dcp_size):
                    b_recv = send_recv(b_recv,
                                       self.mapping.prev_dcp_rank(),
                                       self.mapping.next_dcp_rank(),
                                       self.mapping.dcp_group,
                                       group_stride=self.tp_size)
                    enisum_results[(self.dcp_rank - i) %
                                   self.dcp_size] = _enisum_compute(a, b_recv)
                if self.support_batch:
                    x = concat(enisum_results, dim=2)
                else:
                    x = concat(enisum_results, dim=1)
            else:
                a_recv = a
                for i in range(1, self.dcp_size):
                    a_recv = send_recv(a_recv,
                                       self.mapping.prev_dcp_rank(),
                                       self.mapping.next_dcp_rank(),
                                       self.mapping.dcp_group,
                                       group_stride=self.tp_size)
                    enisum_results[(self.dcp_rank - i) %
                                   self.dcp_size] = _enisum_compute(a_recv, b)
                if self.support_batch:
                    x = concat(enisum_results, dim=1)
                else:
                    x = concat(enisum_results, dim=0)
        else:
            x = _enisum_compute(a, b)

        if self.tp_size > 1:
            x = allgather(x, self.tp_group, gather_dim=-1)

        x_in = cast(x_in, "float32")
        norm_x = self.norm_out(x)
        pout_x = self.p_out(norm_x)
        gout_x = activation(self.g_out(x_in), trt.ActivationType.SIGMOID)
        x = pout_x * gout_x

        # Gather on the dcp group
        if self.dcp_size > 1:
            if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
                gather_dim = 1 if self.support_batch else 0
            else:
                gather_dim = 2 if self.support_batch else 1
            x = allgather(x, self.dcp_group, gather_dim=gather_dim)
        if x.dtype != original_dtype:
            x = cast(x, original_dtype)
        return x
