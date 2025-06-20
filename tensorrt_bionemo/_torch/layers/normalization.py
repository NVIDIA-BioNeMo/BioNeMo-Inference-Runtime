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

from tensorrt_bionemo._torch.distributed import allgather
from tensorrt_bionemo._torch.layers.linear import (Linear, TensorParallelMode,
                                                   WeightMode,
                                                   WeightsLoadingConfig)
from tensorrt_bionemo.mapping import Mapping


class AdaLN(nn.Module):

    def __init__(self,
                 dim: int,
                 dim_single_cond: int,
                 eps: float = 1e-5,
                 dtype: torch.dtype = None,
                 skip_create_weights: bool = False,
                 mapping: Optional[Mapping] = None):
        super().__init__()
        if mapping is None:
            mapping = Mapping()
        self.dim = dim // mapping.tp_size
        self.dim_single_cond = dim_single_cond
        self.mapping = mapping
        self.tp_group = mapping.tp_group
        self.a_norm = nn.LayerNorm(self.dim * mapping.tp_size,
                                   dtype=dtype,
                                   eps=eps,
                                   elementwise_affine=False,
                                   bias=False)
        self.s_norm = nn.LayerNorm(self.dim_single_cond,
                                   dtype=dtype,
                                   eps=eps,
                                   bias=False)
        # Fused s_scale and s_bias, but s_bias has no bias
        # remember to set it to zero correctly
        self.fused_s_scale_s_bias = Linear(
            self.dim_single_cond,
            2 * mapping.tp_size * self.dim,
            bias=True,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
            skip_create_weights=skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_KV_LINEAR))

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: [B, I, d]
            s: [B, I, d_cond]

        Returns:
            a: [B, I, d]
        """
        a = self.a_norm(a)
        s = self.s_norm(s)
        ss = self.fused_s_scale_s_bias(s)
        s_scale, s_bias = ss.split([self.dim, self.dim], dim=-1)
        if self.mapping.tp_size > 1:
            start = self.mapping.tp_rank * self.dim
            end = (self.mapping.tp_rank + 1) * self.dim
            a = a[:, :, start:end]

        a = F.sigmoid(s_scale) * a + s_bias

        if self.mapping.tp_size > 1:
            a = allgather(a, self.mapping, gather_dim=-1)
        return a
