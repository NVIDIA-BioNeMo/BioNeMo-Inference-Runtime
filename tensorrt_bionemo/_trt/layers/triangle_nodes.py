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

import tensorrt as trt
from tensorrt_llm_lite.functional import (Tensor, activation, cast,
                                          constant_to_tensor_, einsum,
                                          expand_dims, permute, shape, split)
from tensorrt_llm_lite.layers.linear import Linear
from tensorrt_llm_lite.layers.normalization import LayerNorm
from tensorrt_llm_lite.module import Module

from tensorrt_bionemo._trt.functional import chunk_loop

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
        fallback_threshold: int = 0,
        mha_bias_flags: dict[str, bool] = {
            "q": False,
            "k": False,
            "v": False,
            "g": False,
            "z": False,
            "o": False
        }):
        super().__init__()
        self.local_layer_idx = local_layer_idx
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.node_type = node_type
        self.inf = inf
        self.triangle_attn_backend = triangle_attn_backend
        self.support_batch = support_batch
        self.fallback_threshold = fallback_threshold
        self.chunk_size = chunk_size

        self.layer_norm = LayerNorm(normalized_shape=[self.c_in],
                                    eps=eps,
                                    dtype=dtype)
        self.linear = Linear(self.c_in,
                             self.num_heads,
                             bias=False,
                             dtype=dtype)
        self.mha = TriangleAttention(
            local_layer_idx=self.local_layer_idx,
            hidden_size=self.c_in,
            num_attention_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            dtype=dtype,
            bias_flags=mha_bias_flags,
            gating=True,
            triangle_attn_backend=self.triangle_attn_backend,
            support_batch=self.support_batch,
            fallback_threshold=self.fallback_threshold)

    def forward(self,
                x: Tensor,
                mask: Tensor,
                attention_params: AttentionParams = None):
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
        inf_const = constant_to_tensor_(self.inf,
                                        dtype=x.dtype,
                                        to_array=False)
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
            shape(x, 0)
            shape(x, 1)
            shape(x, 2)
        else:
            shape(x, 0)
            shape(x, 1)

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
            high_precision: bool = True,
            bias_flags: dict[str, bool] = {
                "p_in": False,
                "g_in": False,
                "p_out": False,
                "g_out": False,
            },
            dtype: str = None,
            support_batch: bool = False):
        super().__init__()
        self.local_layer_idx = local_layer_idx
        self.dtype = dtype
        self.support_batch = support_batch
        self.dim = dim
        self.multiplication_type = multiplication_type
        self.high_precision = high_precision
        self.norm_in = LayerNorm(normalized_shape=[self.dim],
                                 eps=eps,
                                 dtype=dtype)
        self.p_in = Linear(dim, 2 * dim, bias=bias_flags["p_in"], dtype=dtype)
        self.g_in = Linear(dim, 2 * dim, bias=bias_flags["g_in"], dtype=dtype)

        high_precision_dtype = "float32" if high_precision else dtype
        self.norm_out = LayerNorm(normalized_shape=[dim],
                                  eps=eps,
                                  dtype=high_precision_dtype)
        self.p_out = Linear(dim,
                            dim,
                            bias=bias_flags["p_out"],
                            dtype=high_precision_dtype)
        self.g_out = Linear(dim,
                            dim,
                            bias=bias_flags["g_out"],
                            dtype=high_precision_dtype)

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
            shape(x, 0)
            shape(x, 1)
            shape(x, 2)
            shape(x, 3)
        else:
            shape(x, 0)
            shape(x, 1)
            shape(x, 2)
        x = self.norm_in(x)
        x_in = x
        # TODO: SWiGLU, fuse p_in and g_in here
        x = self.p_in(x) * activation(self.g_in(x), trt.ActivationType.SIGMOID)
        x = x * mask.unsqueeze(-1)
        if self.high_precision:
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

        x = _enisum_compute(a, b)

        if self.high_precision:
            x_in = cast(x_in, "float32")
        norm_x = self.norm_out(x)
        pout_x = self.p_out(norm_x)
        gout_x = activation(self.g_out(x_in), trt.ActivationType.SIGMOID)
        x = pout_x * gout_x

        if x.dtype != original_dtype:
            x = cast(x, original_dtype)
        return x
