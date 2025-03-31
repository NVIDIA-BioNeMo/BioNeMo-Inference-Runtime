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

from tensorrt_bionemo._torch.distributed import (AllGatherMode, ParallelConfig,
                                                 TensorParallelMode, allgather)
from tensorrt_bionemo._torch.modules.linear import Linear
from tensorrt_bionemo.layers.triangle_nodes import TriangleAttentionNodeType

from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .attention import TriangleAttention


class TriangleAttentionNode(nn.Module):

    def __init__(
            self,
            c_in: int,
            c_hidden: int,
            num_heads: int,
            node_type: TriangleAttentionNodeType = TriangleAttentionNodeType.
        STARTING,
            inf: float = 1e9,
            layer_idx: int = 0,
            dtype: torch.dtype = None,
            config: Optional[ModelConfig] = None):
        """
        Args:
            c_in (int): input channel dimension
            c_hidden (int): hidden channel dimension
            num_heads (int): number of attention heads
            node_type (TriangleAttentionNodeType): whether this is the starting node
            inf (float): infinity value
            dtype (torch.dtype): data type
            config (ModelConfig): model config
        """
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.node_type = node_type
        self.inf = inf
        config = config or ModelConfig()
        self.dp_size = config.mapping.dp_size
        self.dp_rank = config.mapping.dp_rank
        self.tp_size = config.mapping.tp_size
        self.tp_rank = config.mapping.tp_rank
        self.gpus_per_node = config.mapping.gpus_per_node

        assert self.num_heads % self.tp_size == 0
        self.num_heads = self.num_heads // self.tp_size
        self.chunk_size = config.triangle_attn_node_chunk_size

        if self.chunk_size > 0:
            assert self.chunk_size % self.dp_size == 0
            self.chunk_size = self.chunk_size // self.dp_size
        self.layer_norm = nn.LayerNorm(self.c_in, dtype=dtype)
        self.linear = Linear(
            self.c_in,
            self.tp_size * self.num_heads,
            bias=False,
            dtype=dtype,
            parallel_config=ParallelConfig(
                tensor_parallel_rank=self.tp_rank,
                tensor_parallel_size=self.tp_size,
                data_parallel_size=self.dp_size,
                data_parallel_rank=self.dp_rank,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gpus_per_node=self.gpus_per_node,
                gather_output=True),
            skip_create_weights=config.skip_create_weights,
        )

        self.mha = TriangleAttention(
            layer_idx=layer_idx,
            hidden_size=self.c_in,
            num_attention_heads=self.num_heads * self.tp_size,
            num_key_value_heads=self.num_heads * self.tp_size,
            gating=True,
            bias=False,
            dtype=dtype,
            config=config)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        """
        Forward pass for the triangle attention node. If dp_size > 1 and chunk_size,
        make sure the sequence length is a multiple of chunk_size*dp_size. Currently,
        supports only batch_size = 1

        Args:
            x (torch.Tensor): input tensor, shape [1, I, J, c_in]
            mask (Optional[torch.Tensor]): mask tensor [1, I, J]
            attn_metadata (Optional[AttentionMetadata]): attention metadata
        """
        if x.ndim == 4:
            x = x.squeeze(0)
        assert x.ndim == 3
        if mask is not None:
            mask = mask.squeeze(0)
        else:
            mask = x.new_ones(x.shape[:-1])

        if self.node_type == TriangleAttentionNodeType.ENDING:
            x = x.transpose(0, 1)
            mask = mask.transpose(0, 1)
        x = self.layer_norm(x)

        # Compute mask bias
        mask_bias = (self.inf * (mask - 1))[:, None, None, :]

        # Compute triangle bias
        lx = self.linear(x)
        triangle_bias = torch.permute(lx,
                                      (2, 0, 1)).unsqueeze(0)  # [1, H, I, J]

        # First if dp_size > 1, we need to split the input by dp_size
        seq_len = x.shape[0]
        if self.dp_size > 1:
            seq_len = seq_len // self.dp_size
            x = x[self.dp_rank * seq_len:(self.dp_rank + 1) * seq_len, ...]
            mask_bias = mask_bias[self.dp_rank * seq_len:(self.dp_rank + 1) *
                                  seq_len, ...]

        if self.chunk_size > 0:
            niters = seq_len // self.chunk_size
            outputs = []
            for i in range(niters):
                start = i * self.chunk_size
                end = start + self.chunk_size
                x_chunk = x[start:end, ...]
                chunk_mask_bias = mask_bias[start:end, ...]
                biases = [chunk_mask_bias, triangle_bias]
                chunk_output = self.mha(x_chunk,
                                        biases=biases,
                                        attn_metadata=attn_metadata)
                outputs.append(chunk_output)
            output = torch.cat(outputs, dim=0)
        else:
            biases = [mask_bias, triangle_bias]
            output = self.mha(x, biases=biases, attn_metadata=attn_metadata)

        if self.dp_size > 1:
            parallel_config = ParallelConfig(
                tensor_parallel_rank=self.tp_rank,
                tensor_parallel_size=self.tp_size,
                data_parallel_size=self.dp_size,
                data_parallel_rank=self.dp_rank,
                gpus_per_node=self.gpus_per_node,
                gather_output=True,
            )
            output = allgather(output,
                               parallel_config,
                               gather_dim=0,
                               mode=AllGatherMode.DP)

        if self.node_type == TriangleAttentionNodeType.ENDING:
            output = output.transpose(1, 0)
        return output


class TriangleAttentionStartingNode(TriangleAttentionNode):

    def __init__(self,
                 c_in: int,
                 c_hidden: int,
                 num_heads: int,
                 inf: float = 1e9,
                 layer_idx: int = 0,
                 dtype: torch.dtype = None,
                 config: Optional[ModelConfig] = None):
        super().__init__(c_in, c_hidden, num_heads,
                         TriangleAttentionNodeType.STARTING, inf, layer_idx,
                         dtype, config)


class TriangleAttentionEndingNode(TriangleAttentionNode):

    def __init__(self,
                 c_in: int,
                 c_hidden: int,
                 num_heads: int,
                 inf: float = 1e9,
                 layer_idx: int = 0,
                 dtype: torch.dtype = None,
                 config: Optional[ModelConfig] = None):
        super().__init__(c_in, c_hidden, num_heads,
                         TriangleAttentionNodeType.ENDING, inf, layer_idx,
                         dtype, config)
