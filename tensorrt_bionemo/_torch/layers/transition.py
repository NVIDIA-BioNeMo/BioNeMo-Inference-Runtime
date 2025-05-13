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

from tensorrt_bionemo._torch.layers.linear import (Linear, TensorParallelMode,
                                                   WeightMode,
                                                   WeightsLoadingConfig)
from tensorrt_bionemo.mapping import Mapping, create_max_tp_mapping


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
        self.silu = nn.SiLU()
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
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        x = self.norm(x)
        x = self.fused_fc2_fc1(x)
        x, gate = x.split([self.hidden, self.hidden], dim=-1)
        x = self.silu(gate) * x
        x = self.fc3(x, all_reduce_params=all_reduce_params)
        return x
