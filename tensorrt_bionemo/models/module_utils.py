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

import os
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

import safetensors
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.functional import Tensor
from tensorrt_llm.logger import logger
from tensorrt_llm.module import Module
from tensorrt_llm.plugin import (current_all_reduce_helper,
                                 init_all_reduce_helper)

from tensorrt_bionemo.confs.model_config import PretrainedModuleConfig
from tensorrt_bionemo.layers.attention import AttentionParams


class PretrainedModule(Module):

    def __init__(self, config: PretrainedModuleConfig):
        super().__init__()
        init_all_reduce_helper()
        self.config = config

    def check_config(self, config):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @classmethod
    def from_config(cls, config: PretrainedModuleConfig) -> 'PretrainedModule':
        return cls(config)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str,
        rank: Optional[int] = None,
        config: Optional[PretrainedModuleConfig] = None,
        *,
        preprocess_weights_hook: Optional[Callable[[Dict[str, Tensor]],
                                                   Dict[str, Tensor]]] = None
    ) -> 'PretrainedModule':
        if config is None:
            config = PretrainedModuleConfig.from_json_file(
                cls, os.path.join(ckpt_dir, 'config.json'))
        if rank is not None:
            config.set_rank(rank)
        rank = config.mapping.rank
        # TODO:If has cp for attention, modify rank
        weights_path = os.path.join(ckpt_dir, f'rank{rank}.safetensors')
        assert os.path.isfile(weights_path)
        weights = safetensors.torch.load_file(weights_path)
        if preprocess_weights_hook is not None:
            weights = preprocess_weights_hook(weights)
        model = cls(config)
        model.load(weights, from_pruned=False)
        return model

    def load(self, weights, from_pruned=False):
        required_names = set()
        for name, param in self.named_parameters():
            if param.is_inited():
                continue
            required_names.add(name)

        provided_names = set(weights.keys())

        if not required_names.issubset(provided_names):
            raise RuntimeError(
                f"Required but not provided tensors:{required_names.difference(provided_names)}"
            )
        if not provided_names.issubset(required_names):
            logger.warning(
                f"Provided but not required tensors: {provided_names.difference(required_names)}"
            )

        for name, param in self.named_parameters():
            if name in provided_names:
                if not from_pruned:
                    try:
                        param.value = weights[name]
                    except Exception as e:
                        raise RuntimeError(
                            f"Encounter error '{e}' for parameter '{name}'")
                else:
                    param.set_value_or_dummy(weights[name])

    def save_checkpoint(self, output_dir, save_config=True):
        # multiple ranks could share same config.json, so adding a save_config parameter to let user avoiding writing config.json in all ranks
        rank = self.config.mapping.rank
        weights = {
            name: numpy_to_torch(param.raw_value)
            for name, param in self.named_parameters()
        }
        # If there are some tensors share memory, this will lead to error when we call "save_file". So, for repeated tensors, we
        # clone the tensors to prevent this issue.
        data_ptrs = set()
        for name, param in weights.items():
            if param.data_ptr() in data_ptrs:
                weights[name] = param.clone()
            data_ptrs.add(weights[name].data_ptr())
        safetensors.torch.save_file(
            weights, os.path.join(output_dir, f'rank{rank}.safetensors'))
        if save_config:
            self.config.to_json_file(os.path.join(output_dir, 'config.json'))

    def prepare_inputs(
            self,
            opt_profiles: list[dict] = None,
            has_attention: bool = False,
            disable_custom_all_reduce: bool = False) -> dict[str, Any]:
        input_shapes = self.config.get_input_shapes()
        mapping = self.config.mapping
        dtype = str_dtype_to_trt(self.config.dtype)

        if mapping.tp_size > 1 and not disable_custom_all_reduce:
            if len(opt_profiles) > 0:
                current_all_reduce_helper().set_workspace_tensor(
                    mapping, len(opt_profiles))
            else:
                current_all_reduce_helper().set_workspace_tensor(mapping, 1)

        basic_inputs = {}

        if opt_profiles is None and len(opt_profiles) == 0:
            for k, v in input_shapes.items():
                for dim in v:
                    if dim.dynamic:
                        raise ValueError(
                            f"Dynamic input {k} has no any opt_profiles")
                shape = [dim.size for dim in v]
                basic_inputs[k] = Tensor(name=k, dtype=dtype, shape=shape)
        else:
            for k, v in input_shapes.items():
                dim_ranges = OrderedDict()
                shape = []
                for i, dim in enumerate(v):
                    dim_range = []
                    for opt_profile in opt_profiles:
                        dim_range.append((
                            opt_profile[k][0][i],  # min
                            opt_profile[k][1][i],  # opt
                            opt_profile[k][2][i]))  # max
                    count = 0
                    for dim_name in dim_ranges.keys():
                        if dim_name.startswith(dim.name):
                            count += 1
                    dim_ranges[dim.name + "_" + str(count)] = dim_range
                    shape.append(dim.size)
                logger.info(f"Dynamic input {k} with shape: {shape}")
                logger.info(f"  And dim ranges: {dim_ranges}")
                basic_inputs[k] = Tensor(name=k,
                                         dtype=dtype,
                                         shape=shape,
                                         dim_range=dim_ranges)

        if has_attention:
            basic_inputs["attention_params"] = AttentionParams(
                vanilla_attn_precision=self.config.vanilla_attn_precision)
        return basic_inputs
