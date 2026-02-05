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
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from tensorrt_bionemo._torch.distributed import AllReduceParams
from tensorrt_bionemo.mapping import Mapping


class TensorParallelMode(str, enum.Enum):
    COLUMN = 'column'
    ROW = 'row'

    @classmethod
    def split_dim(cls, mode):
        return 1 if mode == cls.ROW else 0

    # Helper to shard the corresponding per-channel activation scales
    # Which shard along the dimension orthogonal to the weights
    @classmethod
    def flip(cls, mode):
        return cls.ROW if mode == cls.COLUMN else cls.COLUMN


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
    ignore_tensor_parallel: bool = False


def load_weight_shard(
        weight,
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    if isinstance(weight, torch.Tensor):
        tensor_shape = weight.shape

        def maybe_convert_to_torch_tensor(tensor: torch.Tensor,
                                          indices: slice = None):
            if indices == None:
                # Avoid unnecessary copy
                return tensor.to(device)
            else:
                return tensor[indices].to(device)
    # WAR to check whether it is a safetensor slice since safetensor didn't register the type to the module
    # safetensors slice, supports lazy loading, type(weight) is `builtin.PySafeSlice`
    elif hasattr(weight, "get_shape"):
        tensor_shape = weight.get_shape()

        def maybe_convert_to_torch_tensor(
            tensor, indices: Union[slice | tuple[slice]] = slice(None)):
            return tensor[indices].to(device)
    else:
        raise ValueError(f'unsupported weight type: {type(weight)}')
    if tensor_parallel_mode is None or tensor_parallel_size <= 1:
        return maybe_convert_to_torch_tensor(weight)

    split_dim = TensorParallelMode.split_dim(tensor_parallel_mode)

    if len(tensor_shape) == 1 and split_dim == 1:
        return maybe_convert_to_torch_tensor(weight)

    width = tensor_shape[split_dim]
    if width == 1:
        return maybe_convert_to_torch_tensor(weight)

    slice_width = math.ceil(width / tensor_parallel_size)
    slice_start = tensor_parallel_rank * slice_width
    slice_end = min((tensor_parallel_rank + 1) * slice_width, width)
    slice_obj = [slice(None)] * len(tensor_shape)
    slice_obj[split_dim] = slice(slice_start, slice_end)
    return maybe_convert_to_torch_tensor(weight, tuple(slice_obj))


class Linear(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            dtype: torch.dtype = None,
            mapping: Optional[Mapping] = None,
            tensor_parallel_mode: Optional[TensorParallelMode] = None,
            gather_output: bool = False,  # COLUMN parallel only
            reduce_output: bool = True,  # ROW parallel only
            weights_loading_config: Optional[WeightsLoadingConfig] = None,
            skip_create_weights: bool = False):
        super().__init__()
        # TODO: support custom tensor parallel group
        from tensorrt_bionemo._torch.distributed import \
            get_default_tp_group_coordinator
        self.has_bias = bias
        self.dtype = dtype
        self.mapping = mapping or Mapping()
        # could be modified later
        self.weights_loading_config = weights_loading_config or WeightsLoadingConfig(
        )
        self.tp_size = self.mapping.tp_size
        self.tp_rank = self.mapping.tp_rank
        self.tp_mode = tensor_parallel_mode
        self.gather_output = gather_output
        self.reduce_output = reduce_output

        local_in_features = in_features
        local_out_features = out_features

        self.group_comm = None
        if self.tp_size > 1:
            self.group_comm = get_default_tp_group_coordinator()
            assert self.group_comm(
            ) is not None, "TP group coordinator is not initialized, please call register_tp_group_coordinator first"

        if self.tp_mode == TensorParallelMode.ROW:
            assert in_features % self.tp_size == 0, (
                f'in_features {in_features} must be divisible by tp_size {self.tp_size}'
            )
            local_in_features = in_features // self.tp_size
        elif self.tp_mode == TensorParallelMode.COLUMN:
            assert out_features % self.tp_size == 0, (
                f'out_features {out_features} must be divisible by tp_size {self.tp_size}'
            )
            local_out_features = out_features // self.tp_size
        else:
            assert self.tp_mode is None, (
                'unsupported tensor parallel mode: {self.tp_mode}')

        self.in_features = local_in_features
        self.out_features = local_out_features

        # The default strategy is NCCL, MIN_LATENCY has some errors
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

    def forward(
            self,
            input: torch.Tensor,
            *,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:

        if self.tp_mode == TensorParallelMode.ROW:
            bias = None if (self.tp_rank > 0) else self.bias
            output = self.apply_linear(input, self.weight, bias)
            if self.tp_size > 1:
                output = self.group_comm().all_reduce(
                    output,
                    params=all_reduce_params,
                )

        elif self.tp_mode == TensorParallelMode.COLUMN:
            output = self.apply_linear(input, self.weight, self.bias)
            if self.gather_output and self.tp_size > 1:
                output = self.group_comm().all_gather(output)
        else:
            output = self.apply_linear(input, self.weight, self.bias)

        return output

    def load_weights(self, weights: List[Dict]):
        assert self._weights_created

        def copy(dst: Parameter, src: torch.Tensor):
            if dst.dtype != src.dtype:
                src = src.to(dst.dtype)
            assert dst.dtype == src.dtype, f"Incompatible dtype. dst: {dst.dtype}, src: {src.dtype}"
            dst.data.copy_(src)

        weight_mode = self.weights_loading_config.weight_mode
        # load weight shard onto GPU to speed up operations on the shards
        device = torch.device('cuda')

        if weight_mode == WeightMode.VANILLA:
            assert len(weights) == 1

            weight = load_weight_shard(weights[0]['weight'], self.tp_size,
                                       self.tp_rank, self.tp_mode, device)
            copy(self.weight, weight)

            if self.bias is not None:
                bias = load_weight_shard(weights[0]['bias'], self.tp_size,
                                         self.tp_rank, self.tp_mode, device)
                copy(self.bias, bias)
        elif weight_mode == WeightMode.FUSED_QKV_LINEAR:
            assert len(weights) == 3

            q_weight = load_weight_shard(weights[0]['weight'], self.tp_size,
                                         self.tp_rank, self.tp_mode, device)
            k_weight = load_weight_shard(weights[1]['weight'], self.tp_size,
                                         self.tp_rank, self.tp_mode, device)
            v_weight = load_weight_shard(weights[2]['weight'], self.tp_size,
                                         self.tp_rank, self.tp_mode, device)

            fused_weight = torch.cat((q_weight, k_weight, v_weight))

            copy(self.weight, fused_weight)

            if self.bias is not None:
                q_bias = load_weight_shard(weights[0]['bias'], self.tp_size,
                                           self.tp_rank, self.tp_mode, device)
                k_bias = load_weight_shard(weights[1]['bias'], self.tp_size,
                                           self.tp_rank, self.tp_mode, device)
                v_bias = load_weight_shard(weights[2]['bias'], self.tp_size,
                                           self.tp_rank, self.tp_mode, device)
                copy(self.bias, torch.cat((q_bias, k_bias, v_bias)))
        elif weight_mode == WeightMode.FUSED_KV_LINEAR:
            assert len(weights) == 2

            k_weight = load_weight_shard(weights[0]['weight'], self.tp_size,
                                         self.tp_rank, self.tp_mode, device)
            v_weight = load_weight_shard(weights[1]['weight'], self.tp_size,
                                         self.tp_rank, self.tp_mode, device)

            fused_weight = torch.cat((k_weight, v_weight))

            copy(self.weight, fused_weight)

            if self.bias is not None:
                k_bias = load_weight_shard(weights[0]['bias'], self.tp_size,
                                           self.tp_rank, self.tp_mode, device)
                v_bias = load_weight_shard(weights[1]['bias'], self.tp_size,
                                           self.tp_rank, self.tp_mode, device)
                copy(self.bias, torch.cat((k_bias, v_bias)))
        elif weight_mode == WeightMode.FUSED_ALL_LINEAR_LAST_DIM:
            fused_weight = []
            for weight_index in range(len(weights)):
                weight = load_weight_shard(weights[weight_index]['weight'],
                                           self.tp_size, self.tp_rank,
                                           self.tp_mode, device)
                fused_weight.append(weight)
            copy(self.weight, torch.cat(fused_weight, dim=-1))

            if self.bias is not None:
                fused_bias = []
                for bias_index in range(len(weights)):
                    bias = load_weight_shard(weights[bias_index]['bias'],
                                             self.tp_size, self.tp_rank,
                                             self.tp_mode, device)
                    fused_bias.append(bias.unsqueeze(-1))

                copy(self.bias, torch.sum(torch.cat(fused_bias, dim=-1),
                                          dim=-1))
        else:
            raise ValueError(f'unsupported weight mode: {weight_mode}')
