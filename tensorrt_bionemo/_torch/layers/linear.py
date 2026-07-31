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
#
# Adapt from tensorrt_llm/_torch/modules/linear.py

import enum
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


class WeightMode(str, enum.Enum):
    # weight of a vanilla layer
    VANILLA = 'vanilla'
    # weight of a fused QKV linear layer
    FUSED_QKV_LINEAR = 'fused_qkv_linear'
    # weight of a fused gate and up linear layer
    FUSED_KV_LINEAR = 'fused_kv_linear'
    # weight of a fused all linear layer, the last dimension is the fused dimension
    FUSED_ALL_LINEAR_LAST_DIM = 'fused_all_linear_last_dim'


@dataclass(kw_only=True)
class WeightsLoadingConfig:
    weight_mode: WeightMode = WeightMode.VANILLA


def load_weight(
        weight,
        device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    if isinstance(weight, torch.Tensor):
        # Avoid unnecessary copy
        return weight.to(device)
    # WAR to check whether it is a safetensor slice since safetensor didn't register the type to the module
    # safetensors slice, supports lazy loading, type(weight) is `builtin.PySafeSlice`
    elif hasattr(weight, "get_shape"):
        return weight[slice(None)].to(device)
    else:
        raise ValueError(f'unsupported weight type: {type(weight)}')


class Linear(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 weights_loading_config: Optional[WeightsLoadingConfig] = None,
                 skip_create_weights: bool = False):
        super().__init__()
        self.has_bias = bias
        self.dtype = dtype
        # could be modified later
        self.weights_loading_config = weights_loading_config or WeightsLoadingConfig(
        )

        self.in_features = in_features
        self.out_features = out_features

        self._weights_created = False

        if not skip_create_weights:
            self.create_weights()

    def create_weights(self):
        if self._weights_created:
            return
        device = torch.device('cuda')
        weight_shape = (self.out_features, self.in_features)

        self.weight = Parameter(torch.empty(weight_shape,
                                            dtype=self.dtype,
                                            device=device),
                                requires_grad=False)

        if self.has_bias:
            self.bias = Parameter(torch.empty((self.out_features, ),
                                              dtype=self.dtype,
                                              device=device),
                                  requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self._weights_created = True

    def apply_linear(self, input, weight, bias):
        output = F.linear(input, weight, bias)
        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.apply_linear(input, self.weight, self.bias)

    def load_weights(self, weights: List[Dict]):
        assert self._weights_created

        def copy(dst: Parameter, src: torch.Tensor):
            if dst.dtype != src.dtype:
                src = src.to(dst.dtype)
            assert dst.dtype == src.dtype, f"Incompatible dtype. dst: {dst.dtype}, src: {src.dtype}"
            dst.data.copy_(src)

        weight_mode = self.weights_loading_config.weight_mode
        # load weights onto GPU to speed up the fusion operations below
        device = torch.device('cuda')

        if weight_mode == WeightMode.VANILLA:
            assert len(weights) == 1

            weight = load_weight(weights[0]['weight'], device)
            copy(self.weight, weight)

            if self.bias is not None:
                bias = load_weight(weights[0]['bias'], device)
                copy(self.bias, bias)
        elif weight_mode == WeightMode.FUSED_QKV_LINEAR:
            assert len(weights) == 3

            q_weight = load_weight(weights[0]['weight'], device)
            k_weight = load_weight(weights[1]['weight'], device)
            v_weight = load_weight(weights[2]['weight'], device)

            fused_weight = torch.cat((q_weight, k_weight, v_weight))

            copy(self.weight, fused_weight)

            if self.bias is not None:
                q_bias = load_weight(weights[0]['bias'], device)
                k_bias = load_weight(weights[1]['bias'], device)
                v_bias = load_weight(weights[2]['bias'], device)
                copy(self.bias, torch.cat((q_bias, k_bias, v_bias)))
        elif weight_mode == WeightMode.FUSED_KV_LINEAR:
            assert len(weights) == 2

            k_weight = load_weight(weights[0]['weight'], device)
            v_weight = load_weight(weights[1]['weight'], device)

            fused_weight = torch.cat((k_weight, v_weight))

            copy(self.weight, fused_weight)

            if self.bias is not None:
                k_bias = load_weight(weights[0]['bias'], device)
                v_bias = load_weight(weights[1]['bias'], device)
                copy(self.bias, torch.cat((k_bias, v_bias)))
        elif weight_mode == WeightMode.FUSED_ALL_LINEAR_LAST_DIM:
            fused_weight = []
            for weight_index in range(len(weights)):
                weight = load_weight(weights[weight_index]['weight'], device)
                fused_weight.append(weight)
            copy(self.weight, torch.cat(fused_weight, dim=-1))

            if self.bias is not None:
                fused_bias = []
                for bias_index in range(len(weights)):
                    bias = load_weight(weights[bias_index]['bias'], device)
                    fused_bias.append(bias.unsqueeze(-1))

                copy(self.bias, torch.sum(torch.cat(fused_bias, dim=-1),
                                          dim=-1))
        else:
            raise ValueError(f'unsupported weight mode: {weight_mode}')
