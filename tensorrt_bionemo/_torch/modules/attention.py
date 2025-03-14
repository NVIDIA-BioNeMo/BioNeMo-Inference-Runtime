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
from tensorrt_llm._torch.distributed import ParallelConfig, TensorParallelMode
from tensorrt_llm._torch.modules.linear import (Linear, WeightMode,
                                                WeightsLoadingConfig)

from ..attention_backend import AttentionMetadata, PredefinedAttentionBiases
from ..attention_backend.utils import create_attention
from ..model_config import ModelConfig


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
                 config: Optional[ModelConfig] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        config = config or ModelConfig()
        tp_size = config.mapping.tp_size
        tp_rank = config.mapping.tp_rank
        gpus_per_node = config.mapping.gpus_per_node
        if config.mapping.enable_attention_dp:
            tp_size = 1
            tp_rank = 0

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
            parallel_config=ParallelConfig(
                tensor_parallel_rank=tp_rank,
                tensor_parallel_size=tp_size,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gpus_per_node=gpus_per_node),
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_QKV_LINEAR),
            skip_create_weights=config.skip_create_weights,
        )
        self.o_proj = Linear(
            self.hidden_size,
            self.q_size,
            bias=False,
            dtype=dtype,
            parallel_config=ParallelConfig(
                tensor_parallel_rank=tp_rank,
                tensor_parallel_size=tp_size,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gpus_per_node=gpus_per_node),
            skip_create_weights=config.skip_create_weights,
        )
        self.g_proj = None
        if gating:
            self.g_proj = Linear(
                self.q_size,
                self.hidden_size,
                bias=False,
                dtype=dtype,
                parallel_config=ParallelConfig(
                    tensor_parallel_rank=tp_rank,
                    tensor_parallel_size=tp_size,
                    tensor_parallel_mode=TensorParallelMode.ROW,
                    gpus_per_node=gpus_per_node),
                skip_create_weights=config.skip_create_weights,
            )
        self.attn = create_attention(
            config.attn_backend,
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
    ) -> torch.Tensor:
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
        attn_output = self.o_proj(attn_output)
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
                 config: Optional[ModelConfig] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.initial_norm = initial_norm

        self.num_key_value_heads = num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        config = config or ModelConfig()
        tp_size = config.mapping.tp_size
        tp_rank = config.mapping.tp_rank
        gpus_per_node = config.mapping.gpus_per_node
        if config.mapping.enable_attention_dp:
            tp_size = 1
            tp_rank = 0

        assert self.num_heads % tp_size == 0
        self.num_heads = self.num_heads // tp_size
        self.num_key_value_heads = (self.num_key_value_heads + tp_size -
                                    1) // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.norm_s = None
        if initial_norm:
            self.norm_s = nn.LayerNorm(c_s, dtype=dtype)

        self.proj_q = Linear(
            self.c_s,
            tp_size * self.q_size,
            bias=True,
            dtype=dtype,
            parallel_config=ParallelConfig(
                tensor_parallel_rank=tp_rank,
                tensor_parallel_size=tp_size,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gpus_per_node=gpus_per_node),
            skip_create_weights=config.skip_create_weights,
        )
        self.proj_kv = Linear(
            self.c_s,
            2 * tp_size * self.kv_size,
            bias=False,
            dtype=dtype,
            parallel_config=ParallelConfig(
                tensor_parallel_rank=tp_rank,
                tensor_parallel_size=tp_size,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gpus_per_node=gpus_per_node),
            skip_create_weights=config.skip_create_weights,
        )

        self.proj_g = Linear(
            self.c_s,
            self.c_s,
            bias=False,
            dtype=dtype,
            parallel_config=ParallelConfig(
                tensor_parallel_rank=tp_rank,
                tensor_parallel_size=tp_size,
                tensor_parallel_mode=TensorParallelMode.ROW,
                gpus_per_node=gpus_per_node),
            skip_create_weights=config.skip_create_weights,
        )

        self.proj_z = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(
                c_z,
                tp_size * self.num_heads,
                bias=False,
                dtype=dtype,
                parallel_config=ParallelConfig(
                    tensor_parallel_rank=tp_rank,
                    tensor_parallel_size=tp_size,
                    tensor_parallel_mode=TensorParallelMode.ROW,
                    gpus_per_node=gpus_per_node),
                skip_create_weights=config.skip_create_weights,
            ),
        )
        self.proj_o = Linear(
            self.c_s,
            self.c_s,
            bias=False,
            dtype=dtype,
            parallel_config=ParallelConfig(
                tensor_parallel_rank=tp_rank,
                tensor_parallel_size=tp_size,
                tensor_parallel_mode=TensorParallelMode.ROW,
                gpus_per_node=gpus_per_node),
            skip_create_weights=config.skip_create_weights,
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        B = s.size(0)
        if self.initial_norm:
            s = self.norm_s(s)
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        kv = self.proj_kv(s)
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        k = k.view(B, -1, self.num_key_value_heads, self.head_dim)
        v = v.view(B, -1, self.num_key_value_heads, self.head_dim)
        z = self.proj_z(z)
        z = torch.moveaxis(z, 3, 1)  # [B, N, N, H] -> [B, H, N, N]
        g = self.proj_g(s).sigmoid()
        mask_bias = (1 - mask[:, None, None].float()) * -self.inf

        biases = [mask_bias, z]
        mha_o = self.attn.forward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            biases=biases,
            metadata=attn_metadata,
            biases_type=PredefinedAttentionBiases.PAIRWISE)
        o = mha_o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)
        return o
