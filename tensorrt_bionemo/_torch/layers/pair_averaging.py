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

from tensorrt_bionemo._torch.distributed import AllReduceParams
from tensorrt_bionemo.mapping import Mapping

from .linear import (Linear, TensorParallelMode, WeightMode,
                     WeightsLoadingConfig)


class PairWeightedAveraging(nn.Module):
    """Pair weighted averaging layer."""

    def __init__(self,
                 c_m: int,
                 c_z: int,
                 c_h: int,
                 num_heads: int,
                 inf: float = 1e9,
                 eps: float = 1e-5,
                 dtype: torch.dtype = None,
                 skip_create_weights: bool = False,
                 mapping: Optional[Mapping] = None) -> None:
        """
        Args:
            c_m(int): The dimension of the input sequence.
            c_z(int): The dimension of the input pairwise tensor.
            c_h(int): The dimension of the hidden.
            num_heads(int): The number of heads.
            inf(float): The infinity value.
            eps(float): The epsilon value.
            dtype(torch.dtype): The data type of the input tensor.
            skip_create_weights(bool): Whether to skip creating weights.
            mapping(Optional[Mapping]): The mapping of the input tensor.
        """
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_h = c_h
        self.inf = inf

        self.mapping = mapping
        if mapping is None:
            self.mapping = Mapping()
        assert num_heads % self.mapping.tp_size == 0, "num_heads must be divisible by tp_size"
        self.num_heads = num_heads // self.mapping.tp_size
        self.norm_m = nn.LayerNorm(self.c_m, dtype=dtype, eps=eps)
        self.norm_z = nn.LayerNorm(self.c_z, dtype=dtype, eps=eps)

        self.fused_proj_m_g = Linear(
            self.c_m,
            2 * self.c_h * self.num_heads * self.mapping.tp_size,
            bias=False,
            dtype=dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_KV_LINEAR),
            skip_create_weights=skip_create_weights,
        )
        self.proj_z = Linear(
            self.c_z,
            self.num_heads * self.mapping.tp_size,
            bias=False,
            dtype=dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
            skip_create_weights=skip_create_weights,
        )
        self.proj_o = Linear(
            self.c_h * self.num_heads * self.mapping.tp_size,
            c_m,
            bias=False,
            dtype=dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            reduce_output=True,
            skip_create_weights=skip_create_weights,
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> torch.Tensor:
        """
        TODO: Support chunking by num_heads
        Args:
            m(torch.Tensor): The input sequence tensor (B, S, N, D)
            z(torch.Tensor): The input pairwise tensor (B, N, N, D)
            mask(torch.Tensor): The pairwise mask tensor (B, N, N)
        Returns:
            torch.Tensor: The output tensor (B, S, N, D)
        """
        m = self.norm_m(m)
        z = self.norm_z(z)
        vg = self.fused_proj_m_g(m)
        v, g = vg.split([self.c_h * self.num_heads, self.c_h * self.num_heads],
                        dim=-1)
        v = v.reshape(*v.shape[:3], self.num_heads, self.c_h)
        v = v.permute(0, 3, 1, 2, 4)
        g = g.sigmoid()

        b = self.proj_z(z)
        b = b.permute(0, 3, 1, 2)
        b = b + (1 - mask[:, None]) * -self.inf
        w = torch.softmax(b, dim=-1)

        # Compute output
        o = torch.einsum("bhij,bhsjd->bhsid", w, v)
        o = o.permute(0, 2, 3, 1, 4)  # [B, S, N, H, D]
        o = o.reshape(*o.shape[:3], self.num_heads * self.c_h)
        o = self.proj_o(
            g * o, all_reduce_params=all_reduce_params)  # Reduce output here
        return o
