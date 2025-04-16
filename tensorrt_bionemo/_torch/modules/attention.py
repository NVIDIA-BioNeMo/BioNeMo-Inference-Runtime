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

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorrt_llm.functional import AllReduceParams

from tensorrt_bionemo._torch.distributed import (TensorParallelMode,
                                                 create_parallel_config)
from tensorrt_bionemo._torch.modules.linear import (Linear, WeightMode,
                                                    WeightsLoadingConfig)
from tensorrt_bionemo.mapping import Mapping, create_max_tp_mapping

from ..attention_backend import AttentionMetadata, PredefinedAttentionBiases
from ..attention_backend.utils import create_attention


class TriangleAttention(nn.Module):
    """
    A module that implements the triangle attention mechanism with tensor parallelism in torch
    """

    def __init__(self,
                 *,
                 hidden_size: int,
                 num_attention_heads: int,
                 num_key_value_heads: Optional[int] = None,
                 layer_idx: int,
                 bias: bool = False,
                 gating: bool = True,
                 dtype: torch.dtype = None,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False,
                 attn_backend: str = "VANILLA"):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        mapping = mapping or Mapping()

        tp_size = mapping.tp_size
        mapping.tp_rank
        mapping.gpus_per_node

        assert self.num_heads % tp_size == 0
        self.num_heads = self.num_heads // tp_size
        self.num_key_value_heads = (self.num_key_value_heads + tp_size -
                                    1) // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.qkv_proj = Linear(
            self.hidden_size,
            tp_size * self.q_size + 2 * tp_size * self.kv_size,
            bias=bias,
            dtype=dtype,
            parallel_config=create_parallel_config(
                mapping, tensor_parallel_mode=TensorParallelMode.COLUMN),
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_QKV_LINEAR),
            skip_create_weights=skip_create_weights,
        )
        self.o_proj = Linear(
            tp_size * self.q_size,
            self.hidden_size,
            bias=False,
            dtype=dtype,
            parallel_config=create_parallel_config(
                mapping, tensor_parallel_mode=TensorParallelMode.ROW),
            skip_create_weights=skip_create_weights,
        )
        self.g_proj = None
        if gating:
            self.g_proj = Linear(
                self.hidden_size,
                tp_size * self.q_size,
                bias=False,
                dtype=dtype,
                parallel_config=create_parallel_config(
                    mapping, tensor_parallel_mode=TensorParallelMode.COLUMN),
                skip_create_weights=skip_create_weights,
            )
        self.attn = create_attention(
            attn_backend,
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        biases: Optional[list[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:
        if attn_metadata.mapping.tp_size > 1:
            new_biases = []
            new_biases.append(biases[0])
            bias_1_shape = biases[1].shape
            scatter_size = bias_1_shape[1] // attn_metadata.mapping.tp_size
            scatter_bias = biases[1][:, attn_metadata.mapping.tp_rank *
                                     scatter_size:
                                     (attn_metadata.mapping.tp_rank + 1) *
                                     scatter_size, :]
            new_biases.append(scatter_bias)
            biases = new_biases
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        mha_o = self.attn.forward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            biases=biases,
            metadata=attn_metadata,
            biases_type=PredefinedAttentionBiases.TRIANGLE)

        if self.g_proj is not None:
            g = self.g_proj(hidden_states)
            g = F.sigmoid(g)
            # [*, Q, H, C_hidden]
            g = g.view(g.size(0), -1, self.num_heads, self.head_dim)
            attn_output = mha_o * g
        else:
            attn_output = mha_o
        attn_output = attn_output.view(attn_output.size(0), -1,
                                       self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output,
                                  all_reduce_params=all_reduce_params)
        return attn_output


class SelfAttentionPairBias(nn.Module):
    """
    A module that implements the self-attention pair bias mechanism with tensor parallelism in torch.
    This kind of attention is used in the pairformer modules.
    """

    def __init__(self,
                 layer_idx: int,
                 c_s: int,
                 c_z: int,
                 num_heads: int,
                 initial_norm: bool = True,
                 dtype: torch.dtype = None,
                 inf: float = 1e6,
                 max_attention_pairwise_tp_size: bool = True,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False,
                 attn_backend: str = "VANILLA"):
        super().__init__()
        self.layer_idx = layer_idx
        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.initial_norm = initial_norm
        self.inf = inf

        self.num_key_value_heads = num_heads
        # This equal to 1 for self-attention
        self.num_key_value_groups = num_heads // self.num_key_value_heads

        mapping = mapping or Mapping()
        if max_attention_pairwise_tp_size:
            mapping = create_max_tp_mapping(mapping, num_heads)
        tp_size = mapping.tp_size
        mapping.tp_rank
        mapping.gpus_per_node

        assert self.num_heads % tp_size == 0
        self.num_heads = self.num_heads // tp_size
        self.num_key_value_heads = self.num_key_value_heads // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.norm_s = None
        if initial_norm:
            self.norm_s = nn.LayerNorm(c_s, dtype=dtype)

        column_parallel_config = create_parallel_config(
            mapping, tensor_parallel_mode=TensorParallelMode.COLUMN)
        self.proj_q = Linear(
            self.c_s,
            tp_size * self.q_size,
            bias=True,
            dtype=dtype,
            parallel_config=column_parallel_config,
            skip_create_weights=skip_create_weights,
        )
        self.proj_kv = Linear(
            self.c_s,
            2 * tp_size * self.kv_size,
            bias=False,
            dtype=dtype,
            parallel_config=column_parallel_config,
            skip_create_weights=skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_KV_LINEAR),
        )
        self.proj_g = Linear(
            self.c_s,
            tp_size * self.q_size,
            bias=False,
            dtype=dtype,
            parallel_config=column_parallel_config,
            skip_create_weights=skip_create_weights,
        )

        self.proj_z = nn.Sequential(
            nn.LayerNorm(c_z, dtype=dtype),
            Linear(
                c_z,
                tp_size * self.num_heads,
                bias=False,
                dtype=dtype,
                parallel_config=column_parallel_config,
                skip_create_weights=skip_create_weights,
            ),
        )
        self.proj_o = Linear(
            tp_size * self.q_size,
            self.c_s,
            bias=False,
            dtype=dtype,
            parallel_config=create_parallel_config(
                mapping, tensor_parallel_mode=TensorParallelMode.ROW),
            skip_create_weights=skip_create_weights,
        )
        self.attn = create_attention(
            attn_backend,
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:
        B = s.size(0)
        if self.initial_norm:
            s = self.norm_s(s)
        q = self.proj_q(s)
        kv = self.proj_kv(s)
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        z = self.proj_z(z)
        z = torch.moveaxis(z, 3, 1)  # [B, N, N, H] -> [B, H, N, N]

        mask_bias = (1 - mask[:, None, None].float()) * -self.inf

        biases = [mask_bias, z]
        mha_o = self.attn.forward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            biases=biases,
            metadata=attn_metadata,
            biases_type=PredefinedAttentionBiases.PAIRWISE)
        o = mha_o.reshape(B, -1, self.num_heads * self.head_dim)

        g = self.proj_g(s).sigmoid()
        o = self.proj_o(g * o, all_reduce_params=all_reduce_params)
        return o
