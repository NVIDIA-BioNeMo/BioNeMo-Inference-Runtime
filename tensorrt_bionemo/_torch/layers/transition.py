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

from tensorrt_bionemo._torch.custom_ops.gated_sigmoid import \
    get_gated_sigmoid_op
from tensorrt_bionemo._torch.distributed import AllReduceParams
from tensorrt_bionemo._torch.layers.linear import (Linear, TensorParallelMode,
                                                   WeightMode,
                                                   WeightsLoadingConfig)
from tensorrt_bionemo._torch.layers.normalization import AdaLN
from tensorrt_bionemo.dsl_kernels.triton.fused_swiglu import FusedSwiGLU
from tensorrt_bionemo.mapping import Mapping, create_max_tp_mapping
from tensorrt_bionemo.runtime.buffers import PreallocatedBuffers


class Transition(nn.Module):

    def __init__(self,
                 dim: int,
                 hidden: int,
                 out_dim: Optional[int] = None,
                 layer_idx: int = 0,
                 eps: float = 1e-5,
                 dtype: torch.dtype = None,
                 max_transition_tp_size: bool = True,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False):
        super().__init__()
        if out_dim is None:
            out_dim = dim

        mapping = mapping or Mapping()
        if max_transition_tp_size:
            mapping = create_max_tp_mapping(mapping, hidden)
        self.dtype = dtype
        self.hidden = hidden // mapping.tp_size
        self.norm = nn.LayerNorm(dim, eps=eps, dtype=dtype)

        self.fused_fc2_fc1 = Linear(
            dim,
            2 * hidden,
            dtype=dtype,
            bias=False,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
            skip_create_weights=skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_KV_LINEAR),
        )
        self._swiglu = FusedSwiGLU(d=self.hidden,
                                   three_way=False,
                                   dtype=dtype or torch.bfloat16)
        self.fc3 = Linear(hidden,
                          out_dim,
                          dtype=dtype,
                          bias=False,
                          mapping=mapping,
                          tensor_parallel_mode=TensorParallelMode.ROW,
                          reduce_output=True,
                          skip_create_weights=skip_create_weights)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if chunk_size is not None and x.dim() >= 3:
            return self._forward_chunked(x, all_reduce_params, chunk_size)
        x = self.norm(x)
        z = self.fused_fc2_fc1(x)
        x = self._swiglu(z)
        x = self.fc3(x, all_reduce_params=all_reduce_params)

        if mask is not None:
            if mask.ndim == x.ndim - 1:
                mask = mask.unsqueeze(-1)
            x = x * mask
        return x

    def _forward_chunked(
        self,
        x: torch.Tensor,
        all_reduce_params: Optional[AllReduceParams],
        chunk_size: int,
    ) -> torch.Tensor:
        """Chunk along dim=1 (e.g. MSA sequence dim) to bound peak memory."""
        chunks = []
        for i in range(0, x.shape[1], chunk_size):
            xi = x[:, i:i + chunk_size]
            xi = self.norm(xi)
            zi = self.fused_fc2_fc1(xi)
            xi = self._swiglu(zi)
            xi = self.fc3(xi, all_reduce_params=all_reduce_params)
            chunks.append(xi)
        return torch.cat(chunks, dim=1)


class ConditionedTransitionBlock(nn.Module):

    def __init__(self,
                 dim_single: int,
                 dim_single_cond: int,
                 expansion_factor: int = 2,
                 eps: float = 1e-5,
                 dtype: torch.dtype = None,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False,
                 using_silu: bool = False):
        super().__init__()
        mapping = mapping or Mapping()
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.tp_group = mapping.tp_group
        self.mapping = mapping
        self.dtype = dtype

        self.dim_single = dim_single
        self.dim_single_cond = dim_single_cond
        self.expansion_factor = expansion_factor

        self.adaln = AdaLN(dim_single,
                           dim_single_cond,
                           eps=eps,
                           dtype=dtype,
                           mapping=mapping)
        self.dim_inner = int(dim_single * expansion_factor) // mapping.tp_size
        # Fused swiglu_gate linear and a_to_b
        self.using_silu = using_silu
        self._swiglu = FusedSwiGLU(d=self.dim_inner,
                                   three_way=not using_silu,
                                   dtype=dtype or torch.bfloat16)
        if not using_silu:
            self.fused_swl_a_to_b = Linear(
                self.dim_single,
                3 * self.dim_inner * mapping.tp_size,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=False,
                skip_create_weights=skip_create_weights,
                weights_loading_config=WeightsLoadingConfig(
                    weight_mode=WeightMode.FUSED_QKV_LINEAR))
        else:
            self.fused_swl_a_to_b = Linear(
                self.dim_single,
                2 * self.dim_inner * mapping.tp_size,
                bias=False,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=False,
                skip_create_weights=skip_create_weights,
                weights_loading_config=WeightsLoadingConfig(
                    weight_mode=WeightMode.FUSED_KV_LINEAR))

        self.b_to_a = Linear(self.dim_inner * mapping.tp_size,
                             self.dim_single,
                             bias=False,
                             dtype=dtype,
                             reduce_output=True,
                             mapping=mapping,
                             tensor_parallel_mode=TensorParallelMode.ROW,
                             skip_create_weights=skip_create_weights)

        self.output_projection = Linear(
            self.dim_single_cond,
            self.dim_single,
            bias=True,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights)

        self._can_fuse_output_gate = (mapping.tp_size == 1)

    def forward(
            self,
            a: torch.Tensor,
            s: torch.Tensor,
            all_reduce_params: Optional[AllReduceParams] = None,
            buffers: Optional[PreallocatedBuffers] = None,
            buffer_key: str = "cond_trans_adaln",
    ) -> torch.Tensor:
        """
        Args:
            a: [B, I, d]
            s: [B, I, d_cond]
            buffers: optional preallocated buffer dict, forwarded to AdaLN.
            buffer_key: key into ``buffers`` for the AdaLN output tensor.

        Returns:
            a: [B, I, d]
        """
        a = self.adaln(a, s, buffers=buffers, buffer_key=buffer_key)
        z = self.fused_swl_a_to_b(a)
        b = self._swiglu(z)
        a = self.b_to_a(b, all_reduce_params=all_reduce_params)

        if self._can_fuse_output_gate:
            # The gated-sigmoid op broadcasts `s` (gate) across the
            # multiplicity dim of `a` when their leading shapes differ,
            # falling back to torch internally for unsupported patterns.
            # Reuse the AdaLN output buffer — fused_swl_a_to_b consumed it
            # above, same shape as the gated_sigmoid output.
            a = get_gated_sigmoid_op(s.dtype)(
                s, self.output_projection.weight,
                a, self.output_projection.bias,
                output=buffers.get(buffer_key) if buffers is not None else None)
        else:
            a = F.sigmoid(self.output_projection(s)) * a
        return a


class PairTransition(nn.Module):

    def __init__(self,
                 c_z: int,
                 n: int,
                 eps: float = 1e-5,
                 dtype: torch.dtype = None,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False):
        super().__init__()
        self.dtype = dtype
        self.mapping = mapping
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.tp_group = mapping.tp_group
        self.c_z = c_z
        self.n = n

        self.layer_norm = nn.LayerNorm(c_z, eps=eps, dtype=dtype)
        self.linear_1 = Linear(c_z,
                               n * c_z,
                               bias=True,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=False)
        self.linear_2 = Linear(n * c_z,
                               c_z,
                               bias=True,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=TensorParallelMode.ROW,
                               reduce_output=True)
        self.relu = nn.ReLU()

    def forward(self,
                z: torch.Tensor,
                mask: torch.Tensor,
                all_reduce_params: Optional[AllReduceParams] = None):
        mask = mask.unsqueeze(-1)
        # [*, N_res, N_res, C_z]
        z = self.layer_norm(z)

        # [*, N_res, N_res, C_hidden]
        z = self.linear_1(z)
        z = self.relu(z)

        # [*, N_res, N_res, C_z]
        z = self.linear_2(z)
        z = z * mask

        return z


class MSATransition(nn.Module):

    def __init__(self,
                 c_m: int,
                 n: int,
                 eps: float = 1e-5,
                 dtype: torch.dtype = None,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False):
        super().__init__()
        self.dtype = dtype
        self.mapping = mapping
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self.tp_group = mapping.tp_group
        self.c_m = c_m
        self.n = n

        self.layer_norm = nn.LayerNorm(c_m, eps=eps, dtype=dtype)
        self.linear_1 = Linear(c_m,
                               n * c_m,
                               bias=True,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=False)
        self.linear_2 = Linear(n * c_m,
                               c_m,
                               bias=True,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=TensorParallelMode.ROW,
                               reduce_output=True)
        self.relu = nn.ReLU()

    def forward(self,
                m: torch.Tensor,
                mask: torch.Tensor,
                all_reduce_params: Optional[AllReduceParams] = None):
        # Similar to PairTransition, but with different names
        mask = mask.unsqueeze(-1)
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m)
        m = m * mask

        return m
