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
                                                 allgather)
from tensorrt_bionemo._torch.layers.linear import (Linear, TensorParallelMode,
                                                   WeightMode,
                                                   WeightsLoadingConfig)
from tensorrt_bionemo._trt.layers.triangle_nodes import (
    TriangleAttentionNodeType, TriangleMultiplicationNodeType)
from tensorrt_bionemo.mapping import Mapping, create_max_tp_mapping

from ..attention_backend import AttentionMetadata
from ..custom_ops import get_custom_ops_impl
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
        chunk_size: int = 0,
        mapping: Optional[Mapping] = None,
        skip_create_weights: bool = False,
        attn_backend: str = "VANILLA",
        mha_bias_flags: dict[str, bool] = {
            "q": False,
            "k": False,
            "v": False,
            "g": False,
            "z": False,
            "o": False
        }):
        """
        Args:
            c_in (int): input channel dimension
            c_hidden (int): hidden channel dimension
            num_heads (int): number of attention heads
            node_type (TriangleAttentionNodeType): whether this is the starting node
            inf (float): infinity value
            dtype (torch.dtype): data type
            chunk_size (int): chunk size
            mapping (Mapping): mapping
            skip_create_weights (bool): whether to skip creating weights
            attn_backend (str): attention backend
        """
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.node_type = node_type
        self.inf = inf
        self.mapping = mapping or Mapping()
        self.dcp_size = self.mapping.dcp_size
        self.dcp_rank = self.mapping.dcp_rank
        self.tp_size = self.mapping.tp_size
        self.tp_rank = self.mapping.tp_rank
        self.gpus_per_node = self.mapping.gpus_per_node
        self.dtype = dtype

        assert self.num_heads % self.tp_size == 0
        self.num_heads = self.num_heads // self.tp_size
        self.chunk_size = chunk_size

        if self.chunk_size > 0:
            assert self.chunk_size % self.dcp_size == 0
            self.chunk_size = self.chunk_size // self.dcp_size
        self.layer_norm = nn.LayerNorm(self.c_in, dtype=dtype)
        self.linear = Linear(
            self.c_in,
            self.tp_size * self.num_heads,
            bias=False,
            dtype=dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights,
        )

        self.mha = TriangleAttention(
            layer_idx=layer_idx,
            hidden_size=self.c_in,
            num_attention_heads=self.num_heads * self.tp_size,
            num_key_value_heads=self.num_heads * self.tp_size,
            gating=True,
            bias_flags=mha_bias_flags,
            dtype=dtype,
            mapping=self.mapping,
            skip_create_weights=skip_create_weights,
            attn_backend=attn_backend,
        )

    def _dcp_slice(
            self, x: torch.Tensor,
            mask_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Deal with the dcp size > 1 """
        seq_len = x.shape[1]
        if self.dcp_size > 1:
            seq_len = seq_len // self.dcp_size
            start = self.dcp_rank * seq_len
            end = (self.dcp_rank + 1) * seq_len
            x = x[:, start:end, ...]
            mask_bias = mask_bias[:, start:end, ...]
        if not x.is_contiguous():
            x = x.contiguous()
        if not mask_bias.is_contiguous():
            mask_bias = mask_bias.contiguous()
        return x, mask_bias

    def _dcp_gather(self, output: torch.Tensor) -> torch.Tensor:
        """ Gather the input by dcp size """
        if self.dcp_size > 1:
            output = allgather(output,
                               self.mapping,
                               gather_dim=1,
                               mode=AllGatherMode.DP)
        return output

    def _ensure_dtype(self, x: torch.Tensor,
                      mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Ensure the dtype of the input and mask """
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        if mask.dtype != self.dtype:
            mask = mask.to(self.dtype)
        return x, mask

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
            x (torch.Tensor): input tensor, shape [B, I, J, c_in]
            mask (Optional[torch.Tensor]): mask tensor [B, I, J]
            attn_metadata (Optional[AttentionMetadata]): attention metadata
        """
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        x, mask = self._ensure_dtype(x, mask)
        if self.node_type == TriangleAttentionNodeType.ENDING:
            x = x.transpose(1, 2)
            mask = mask.transpose(1, 2)

        x = self.layer_norm(x)
        # Compute mask bias
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # Compute triangle bias
        lx = self.linear(x)  # [B, I, J, H]
        triangle_bias = torch.permute(lx, (0, 3, 1, 2))

        seq_len = x.shape[1]
        x, mask_bias = self._dcp_slice(x, mask_bias)
        if self.chunk_size > 0:
            niters = seq_len // self.chunk_size
            outputs = []
            for i in range(niters):
                start = i * self.chunk_size
                end = start + self.chunk_size
                x_chunk = x[:, start:end, ...]
                chunk_mask_bias = mask_bias[:, start:end, ...]
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
        output = self._dcp_gather(output)
        if self.node_type == TriangleAttentionNodeType.ENDING:
            output = output.transpose(2, 1)
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

    def __init__(
            self,
            layer_idx: int = 0,
            dim: int = 128,
            eps: float = 1e-5,
            multiplication_type:
        TriangleMultiplicationNodeType = TriangleMultiplicationNodeType.
        OUTGOING,
            bias_flags: dict[str, bool] = {
                "p_in": False,
                "g_in": False,
                "p_out": False,
                "g_out": False
            },
            dtype: torch.dtype = None,
            mapping: Optional[Mapping] = None,
            skip_create_weights: bool = False,
            max_tri_mul_tp_size: bool = True,
            high_precision: bool = True):
        super().__init__()
        self.mapping = mapping or Mapping()
        if max_tri_mul_tp_size:
            self.mapping = create_max_tp_mapping(self.mapping, dim)
        self.dcp_size = self.mapping.dcp_size
        self.dcp_rank = self.mapping.dcp_rank
        self.tp_size = self.mapping.tp_size
        self.tp_rank = self.mapping.tp_rank
        self.gpus_per_node = self.mapping.gpus_per_node
        self.dtype = dtype
        self.high_precision = high_precision

        self.dp_comm = None
        if self.dcp_size > 1:
            DPCommManager.init_dp_comm(self.mapping)
            self.dp_comm = DPCommManager()
        self.dim = dim // self.tp_size
        self.multiplication_type = multiplication_type
        self.norm_in = nn.LayerNorm(self.dim * self.tp_size,
                                    dtype=dtype,
                                    eps=eps)
        self.p_in = Linear(self.dim * self.tp_size,
                           2 * self.dim * self.tp_size,
                           bias=bias_flags["p_in"],
                           dtype=dtype,
                           mapping=self.mapping,
                           tensor_parallel_mode=TensorParallelMode.COLUMN,
                           gather_output=False,
                           weights_loading_config=WeightsLoadingConfig(
                               weight_mode=WeightMode.FUSED_KV_LINEAR),
                           skip_create_weights=skip_create_weights)
        self.g_in = Linear(self.dim * self.tp_size,
                           2 * self.dim * self.tp_size,
                           bias=bias_flags["g_in"],
                           dtype=dtype,
                           mapping=self.mapping,
                           tensor_parallel_mode=TensorParallelMode.COLUMN,
                           gather_output=False,
                           weights_loading_config=WeightsLoadingConfig(
                               weight_mode=WeightMode.FUSED_KV_LINEAR),
                           skip_create_weights=skip_create_weights)
        # Use float32 for the output layers
        if self.high_precision:
            self.high_precision_dtype = torch.float32
        else:
            self.high_precision_dtype = dtype
        self.norm_out = nn.LayerNorm(self.dim * self.tp_size,
                                     dtype=self.high_precision_dtype,
                                     eps=eps)
        self.p_out = Linear(self.dim * self.tp_size,
                            self.dim * self.tp_size,
                            bias=bias_flags["p_out"],
                            dtype=self.high_precision_dtype,
                            mapping=self.mapping,
                            tensor_parallel_mode=TensorParallelMode.COLUMN,
                            gather_output=True,
                            skip_create_weights=skip_create_weights)
        self.g_out = Linear(self.dim * self.tp_size,
                            self.dim * self.tp_size,
                            bias=bias_flags["g_out"],
                            dtype=self.high_precision_dtype,
                            mapping=self.mapping,
                            tensor_parallel_mode=TensorParallelMode.COLUMN,
                            gather_output=True,
                            skip_create_weights=skip_create_weights)

    @torch.compiler.disable
    def _fused_dual_gemm(self, x: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        fused_ops = get_custom_ops_impl("fused_sigmoid_gated_dual_gemm", x,
                                        self.g_in.weight, self.p_in.weight,
                                        mask, self.g_in.bias, self.p_in.bias)
        if fused_ops is not None:
            x = fused_ops()
        else:
            x = self.p_in(x) * self.g_in(x).sigmoid()
            x = x * mask.unsqueeze(-1)
        return x

    @torch.compiler.disable
    def _fused_dual_gemm_dual_x(self, x_0_out: torch.Tensor,
                                x_1_out: torch.Tensor) -> torch.Tensor:
        fused_ops = get_custom_ops_impl("fused_sigmoid_gated_dual_gemm_dual_x",
                                        x_1_out, x_0_out, self.g_out.weight,
                                        self.p_out.weight, None,
                                        self.g_out.bias, self.p_out.bias)
        if fused_ops is not None:
            x = fused_ops()
        else:
            pout_x = self.p_out(x_0_out)
            gout_x = self.g_out(x_1_out).sigmoid()
            x = pout_x * gout_x
        return x

    def _dcp_slice(self, x: torch.Tensor,
                   mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Slice the input by dcp size """
        seq_len = x.shape[1]
        if self.dcp_size > 1:
            seq_len = seq_len // self.dcp_size
            st = self.dcp_rank * seq_len
            et = (self.dcp_rank + 1) * seq_len
            if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
                x = x[:, st:et, ...]
                mask = mask[:, st:et, ...]
            elif self.multiplication_type == TriangleMultiplicationNodeType.INCOMING:
                x = x[:, :, st:et, ...]
                mask = mask[:, :, st:et]
            x = x.contiguous()
            mask = mask.contiguous()
        return x, mask

    def _dcp_gather(self, x: torch.Tensor) -> torch.Tensor:
        """ Gather the input by dcp size """
        if self.dcp_size > 1:
            x = x.contiguous()
            gather_dim = 1 if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING else 2
            x = allgather(x,
                          self.mapping,
                          gather_dim=gather_dim,
                          mode=AllGatherMode.DP)
        return x

    def _tp_gather(self, x: torch.Tensor) -> torch.Tensor:
        """ Gather the input by tp size """
        if self.tp_size > 1:
            x = x.contiguous()
            x = allgather(x, self.mapping, mode=AllGatherMode.TP)
        return x

    def _ring_einsum_compute(self, a: torch.Tensor,
                             b: torch.Tensor) -> torch.Tensor:
        """ Compute the enisum operation in a ring manner """

        def _einsum_compute(a_, b_):
            if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
                return torch.einsum("bikd,bjkd->bijd", a_, b_)
            else:
                return torch.einsum("bkid,bkjd->bijd", a_, b_)

        # Ring communication
        if self.dcp_size > 1:
            a = a.contiguous()
            b = b.contiguous()
            enisum_results = [
                None,
            ] * self.dcp_size
            enisum_results[self.dcp_rank] = _einsum_compute(a, b)
            if self.multiplication_type == TriangleMultiplicationNodeType.OUTGOING:
                b_recv = torch.zeros_like(b)
                buffers = [b, b_recv]  # double buffers
                send_idx = 0
                recv_idx = 1
                for i in range(1, self.dcp_size):
                    self.dp_comm.batch_isend_irecv(buffers[send_idx],
                                                   buffers[recv_idx])
                    enisum_results[(self.dcp_rank - i) %
                                   self.dcp_size] = _einsum_compute(
                                       a, buffers[recv_idx])
                    recv_idx = send_idx
                    send_idx = (send_idx + 1) % 2
                x = torch.cat(enisum_results, dim=2)
            else:
                a_recv = torch.zeros_like(a)
                buffers = [a, a_recv]  # double buffers
                send_idx = 0
                recv_idx = 1
                for i in range(1, self.dcp_size):
                    self.dp_comm.batch_isend_irecv(buffers[send_idx],
                                                   buffers[recv_idx])
                    enisum_results[(self.dcp_rank - i) %
                                   self.dcp_size] = _einsum_compute(
                                       buffers[recv_idx], b)
                    recv_idx = send_idx
                    send_idx = (send_idx + 1) % 2
                x = torch.cat(enisum_results, dim=1)
        else:
            x = _einsum_compute(a, b)
        return x

    def _ensure_dtype(self, x: torch.Tensor) -> torch.Tensor:
        """ Ensure the dtype of the input """
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor, shape [B, I, J, c_in]
            mask (torch.Tensor): mask tensor [B, I, J]
        """
        x = self._ensure_dtype(x)
        x = self.norm_in(x)
        x, mask = self._dcp_slice(x, mask)
        x_in = x
        x = self._fused_dual_gemm(x, mask)
        x = x.to(self.high_precision_dtype)
        a, b = x.split([self.dim, self.dim], dim=-1)
        x = self._ring_einsum_compute(a, b)
        # need to gather here for LayerNorm
        x = self._tp_gather(x)
        x_0_out = self.norm_out(x)
        x_1_out = x_in.to(self.high_precision_dtype)
        x = self._fused_dual_gemm_dual_x(x_0_out, x_1_out)
        x = self._dcp_gather(x)
        x = self._ensure_dtype(x)
        return x
