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

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar, Union

import torch
import transformers
from tensorrt_llm._utils import str_dtype_to_torch, torch_dtype_to_str
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.plugin import PluginConfig

from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.version import __version__

TConfig = TypeVar("TConfig", bound=transformers.PretrainedConfig)


@dataclass
class DimSpec:
    size: int = -1
    name: str = ""
    dynamic: bool = False
    min: int = -1
    max: int = -1


@dataclass(kw_only=True)
class ModelConfig(Generic[TConfig]):
    pretrained_config: Optional[TConfig] = None
    mapping: Mapping = field(default_factory=Mapping)
    skip_create_weights: bool = False


class PretrainedModuleConfig:

    def __init__(self,
                 *,
                 architecture: str,
                 dtype: str,
                 logits_dtype: str = 'float32',
                 norm_epsilon: float = 1e-5,
                 mask_inf: float = 1e9,
                 skip_create_weights: bool = False,
                 mapping: Optional[Union[Mapping, dict]] = None,
                 disable_custom_all_reduce: bool = False,
                 package_version: str = __version__,
                 **kwargs):
        self.architecture = architecture
        self.dtype = dtype
        self.logits_dtype = logits_dtype
        self.norm_epsilon = norm_epsilon
        self.mask_inf = mask_inf
        self.skip_create_weights = skip_create_weights
        self.quant_algo = None
        # This flag will be disable custom all reduce if has dynamic mapping config
        self.disable_custom_all_reduce = disable_custom_all_reduce
        self.package_version = package_version
        self.set_dtype(dtype)

        if mapping is None:
            mapping = Mapping()
        elif isinstance(mapping, dict):
            mapping = Mapping.from_dict(mapping)
        assert isinstance(mapping, Mapping)
        self.mapping = mapping

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
                logger.warning(
                    f"Implicitly setting {self.__class__.__name__}.{key} = {value}"
                )
            except AttributeError as err:
                raise err

    def get_input_names(self):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    def get_output_names(self):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    def get_input_shapes(self):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    def get_output_shapes(self):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    def update_from_dict(self, config: dict):
        for name, value in config.items():
            if not hasattr(self, name):
                raise AttributeError(
                    f"{self.__class__} object has no attribute {name}")
            setattr(self, name, value)

    def set_rank(self, rank: int):
        self.mapping.rank = rank

    @classmethod
    def from_dict(cls, module_cls, config_dict: dict):
        config_cls = getattr(module_cls, 'config_class')
        return config_cls(**config_dict)

    @classmethod
    def from_json_file(cls, module_cls, config_file: str):
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(module_cls, config_dict)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)

        output['mapping'] = self.mapping.to_dict()
        output['mapping'].pop('rank')
        if 'torch_dtype' in output:
            output.pop('torch_dtype')

        return output

    def set_dtype(self, value: Union[str, torch.dtype]):
        if value is None:
            value = "float32"
        if isinstance(value, str):
            self.torch_dtype = str_dtype_to_torch(value)
            self.dtype = value
        else:
            self.torch_dtype = value
            self.dtype = torch_dtype_to_str(value)


@dataclass
class BuildModuleConfig:
    """ TensorRT-BNM build configurations """

    strongly_typed: bool = True
    weakly_dtype: str = None
    force_num_profiles: Optional[int] = None
    profiling_verbosity: str = 'layer_names_only'
    plugin_config: PluginConfig = field(default_factory=PluginConfig)
    module_config: PretrainedModuleConfig = None
    input_timing_cache: str = None
    output_timing_cache: str = 'model.cache'
    has_attention: bool = True
    vanilla_attn_precision: str = "float32"
    dry_run: bool = False
    monitor_memory: bool = False
    enable_debug_output: bool = False
    lora_config: LoraConfig = field(
        default_factory=LoraConfig)  # Patch for save engine

    @property
    def optimization_profiles(self) -> list[Any]:
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def from_json_file(cls, config_file, plugin_config=None):
        with open(config_file) as f:
            config = json.load(f)
        return cls.from_dict(config, plugin_config=plugin_config)

    def update_from_dict(self, config: dict):
        for name, value in config.items():
            if not hasattr(self, name):
                raise AttributeError(
                    f"{self.__class__} object has no attribute {name}")
            setattr(self, name, value)

    @classmethod
    def from_dict(cls, config, plugin_config=None):
        config = copy.deepcopy(config)
        strongly_typed = config.pop('strongly_typed', True)
        force_num_profiles = config.pop('force_num_profiles', None)
        profiling_verbosity = config.pop('profiling_verbosity',
                                         'layer_names_only')
        config.pop('enable_debug_output', False)
        input_timing_cache = config.pop('input_timing_cache', None)
        output_timing_cache = config.pop('output_timing_cache', None)
        has_attention = config.pop('has_attention', True)
        vanilla_attn_precision = config.pop('vanilla_attn_precision', "float32")

        if plugin_config is None:
            plugin_config = PluginConfig()
        if "plugin_config" in config.keys():
            plugin_config.update_from_dict(config["plugin_config"])

        dry_run = config.pop('dry_run', False)
        monitor_memory = config.pop('monitor_memory', False)

        ret = cls(strongly_typed=strongly_typed,
                  force_num_profiles=force_num_profiles,
                  profiling_verbosity=profiling_verbosity,
                  plugin_config=plugin_config,
                  input_timing_cache=input_timing_cache,
                  output_timing_cache=output_timing_cache,
                  has_attention=has_attention,
                  vanilla_attn_precision=vanilla_attn_precision,
                  dry_run=dry_run,
                  monitor_memory=monitor_memory)
        ret.update_from_dict(config)
        return ret

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output['plugin_config'] = output['plugin_config'].to_dict()
        output['lora_config'] = output['lora_config'].to_dict()
        del output['module_config']
        return output
