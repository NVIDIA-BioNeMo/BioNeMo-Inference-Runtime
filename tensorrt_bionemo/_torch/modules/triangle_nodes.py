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
from tensorrt_llm.functional import AllReduceParams

from tensorrt_bionemo._torch.distributed import (AllGatherMode, DPCommManager,
                                                 TensorParallelMode, allgather,
                                                 create_parallel_config)
from tensorrt_bionemo._torch.modules.linear import (Linear, WeightMode,
                                                    WeightsLoadingConfig)
from tensorrt_bionemo.layers.triangle_nodes import (
    TriangleAttentionNodeType, TriangleMultiplicationNodeType)

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
        self.mapping = config.mapping
        self.dcp_size = self.mapping.dcp_size
        self.dcp_rank = self.mapping.dcp_rank
        self.tp_size = self.mapping.tp_size
        self.tp_rank = self.mapping.tp_rank
        self.gpus_per_node = self.mapping.gpus_per_node

        assert self.num_heads % self.tp_size == 0
        self.num_heads = self.num_heads // self.tp_size
        self.chunk_size = config.triangle_attn_node_chunk_size

        if self.chunk_size > 0:
            assert self.chunk_size % self.dcp_size == 0
            self.chunk_size = self.chunk_size // self.dcp_size
        self.layer_norm = nn.LayerNorm(self.c_in, dtype=dtype)
        self.linear = Linear(
            self.c_in,
            self.tp_size * self.num_heads,
            bias=False,
            dtype=dtype,
            parallel_config=create_parallel_config(
                self.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
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
            attn_metadata: Optional[AttentionMetadata] = None,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        """
        Forward pass for the triangle attention node. If dcp_size > 1 and chunk_size,
        make sure the sequence length is a multiple of chunk_size*dcp_size. Currently,
        supports only batch_size = 1

        Args:
            x (torch.Tensor): input tensor, shape [1, I, J, c_in]
            mask (Optional[torch.Tensor]): mask tensor [1, I, J]
            attn_metadata (Optional[AttentionMetadata]): attention metadata
        """
        if x.ndim == 4:
            x = x.squeeze(0)
        assert x.ndim == 3
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        if mask is not None and mask.ndim == 3:
            mask = mask.squeeze(0)

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

        # First if dcp_size > 1, we need to split the input by dcp_size
        seq_len = x.shape[0]
        if self.dcp_size > 1:
            seq_len = seq_len // self.dcp_size
            x = x[self.dcp_rank * seq_len:(self.dcp_rank + 1) * seq_len, ...]
            mask_bias = mask_bias[self.dcp_rank * seq_len:(self.dcp_rank + 1) *
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
                                        attn_metadata=attn_metadata,
                                        all_reduce_params=all_reduce_params)
                outputs.append(chunk_output)
            output = torch.cat(outputs, dim=0)
        else:
            biases = [mask_bias, triangle_bias]
            output = self.mha(x,
                              biases=biases,
                              attn_metadata=attn_metadata,
                              all_reduce_params=all_reduce_params)
        if self.dcp_size > 1:
            parallel_config = create_parallel_config(
                self.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
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

    def __init__(self, *args, **kwargs):
        kwargs['node_type'] = TriangleAttentionNodeType.STARTING
        super().__init__(*args, **kwargs)


class TriangleAttentionEndingNode(TriangleAttentionNode):

    def __init__(self, *args, **kwargs):
        kwargs['node_type'] = TriangleAttentionNodeType.ENDING
        super().__init__(*args, **kwargs)


class TriangleMultiplicationNode(nn.Module):

    def __init__(self,
                 layer_idx: int = 0,
                 dim: int = 128,
                 eps: float = 1e-5,
                 multiplication_type:
                 TriangleMultiplicationNodeType = TriangleMultiplicationNodeType
                 .OUTGOING,
                 dtype: torch.dtype = None,
                 config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.mapping = config.mapping
        self.dcp_size = self.mapping.dcp_size
        self.dcp_rank = self.mapping.dcp_rank
        self.tp_size = self.mapping.tp_size
        self.tp_rank = self.mapping.tp_rank
        self.gpus_per_node = self.mapping.gpus_per_node

        self.dp_comm = None
        if self.dcp_size > 1:
            DPCommManager.init_dp_comm(self.mapping)
            self.dp_comm = DPCommManager()
        self.dim = dim // self.tp_size
        self.multiplication_type = multiplication_type
        self.norm_in = nn.LayerNorm(self.dim * self.tp_size,
                                    dtype=dtype,
                                    eps=eps)
        col_parallel_config = create_parallel_config(
            self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False)
        self.p_in = Linear(self.dim * self.tp_size,
                           2 * self.dim * self.tp_size,
                           bias=False,
                           dtype=dtype,
                           parallel_config=col_parallel_config,
                           weights_loading_config=WeightsLoadingConfig(
                               weight_mode=WeightMode.FUSED_KV_LINEAR),
                           skip_create_weights=config.skip_create_weights)
        self.g_in = Linear(self.dim * self.tp_size,
                           2 * self.dim * self.tp_size,
                           bias=False,
                           dtype=dtype,
                           parallel_config=col_parallel_config,
                           weights_loading_config=WeightsLoadingConfig(
                               weight_mode=WeightMode.FUSED_KV_LINEAR),
                           skip_create_weights=config.skip_create_weights)
        # Use float32 for the output layers
        self.norm_out = nn.LayerNorm(self.dim * self.tp_size,
                                     dtype=torch.float32,
                                     eps=eps)
        self.p_out = Linear(self.dim * self.tp_size,
                            self.dim * self.tp_size,
                            bias=False,
                            dtype=torch.float32,
                            parallel_config=create_parallel_config(
                                self.mapping,
                                tensor_parallel_mode=TensorParallelMode.COLUMN,
                                gather_output=True),
                            skip_create_weights=config.skip_create_weights)
        self.g_out = Linear(self.dim * self.tp_size,
                            self.dim * self.tp_size,
                            bias=False,
                            dtype=torch.float32,
                            parallel_config=create_parallel_config(
                                self.mapping,
                                tensor_parallel_mode=TensorParallelMode.COLUMN,
                                gather_output=True),
                            skip_create_weights=config.skip_create_weights)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor, shape [1, I, J, c_in]
            mask (torch.Tensor): mask tensor [1, I, J]
        """
        # if self.tp_size > 1 or self.dcp_size > 1:
        parallel_config = create_parallel_config(
            self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        if x.ndim == 4:
            x = x.squeeze(0)
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        x = self.norm_in(x)
        seq_len = x.shape[0]
        if self.dcp_size > 1:
            seq_len = seq_len // self.dcp_size
            st = self.dcp_rank * seq_len
            et = (self.dcp_rank + 1) * seq_len
            if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
                x = x[st:et, ...]
                mask = mask[st:et, ...]
            elif self.multiplication_type == TriangleMultiplicationNodeType.INCOMING:
                x = x[:, st:et, ...]
                mask = mask[:, st:et, ...]
        x_in = x
        # TODO: SwiGLU fused here
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)

        a, b = x.float().split([self.dim, self.dim], dim=-1)
        a = a.contiguous()
        b = b.contiguous()

        def _enisum_compute(a_, b_):
            if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
                return torch.einsum("ikd,jkd->ijd", a_, b_)
            else:
                return torch.einsum("kid,kjd->ijd", a_, b_)

        # Ring communication
        if self.dcp_size > 1:
            enisum_results = [
                None,
            ] * self.dcp_size
            enisum_results[self.dcp_rank] = _enisum_compute(a, b)
            if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
                b_recv = torch.zeros_like(b)
                buffers = [b, b_recv]  # double buffers
                send_idx = 0
                recv_idx = 1
                for i in range(1, self.dcp_size):
                    self.dp_comm.batch_isend_irecv(buffers[send_idx],
                                                   buffers[recv_idx])
                    enisum_results[(self.dcp_rank - i) %
                                   self.dcp_size] = _enisum_compute(
                                       a, buffers[recv_idx])
                    recv_idx = send_idx
                    send_idx = (send_idx + 1) % 2
                x = torch.cat(enisum_results, dim=1)
            else:
                a_recv = torch.zeros_like(a)
                buffers = [a, a_recv]  # double buffers
                send_idx = 0
                recv_idx = 1
                for i in range(1, self.dcp_size):
                    self.dp_comm.batch_isend_irecv(buffers[send_idx],
                                                   buffers[recv_idx])
                    enisum_results[(self.dcp_rank - i) %
                                   self.dcp_size] = _enisum_compute(
                                       buffers[recv_idx], b)
                    recv_idx = send_idx
                    send_idx = (send_idx + 1) % 2
                x = torch.cat(enisum_results, dim=0)
        else:
            x = _enisum_compute(a, b)
        x = x.contiguous()
        # need to gather here for LayerNorm
        if self.tp_size > 1:
            x = allgather(x, parallel_config, mode=AllGatherMode.TP)
        pout_x = self.p_out(self.norm_out(x))
        gout_x = self.g_out(x_in.float()).sigmoid()
        x = pout_x * gout_x
        x = x.contiguous()
        if self.dcp_size > 1:
            gather_dim = 0 if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING else 1
            x = allgather(x,
                          parallel_config,
                          gather_dim=gather_dim,
                          mode=AllGatherMode.DP)
        return x
