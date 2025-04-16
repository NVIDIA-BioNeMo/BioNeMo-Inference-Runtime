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
from typing import Any, Optional

from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.plugin import PluginConfig

from .model_config import PretrainedModuleConfig


@dataclass
class BuildModuleConfig:
    strongly_typed: bool = True
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
    def from_dict(cls, config, plugin_config=None, module_config=None):
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

        assert module_config is not None

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
                  monitor_memory=monitor_memory,
                  module_config=module_config)
        ret.update_from_dict(config)
        return ret

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output['plugin_config'] = output['plugin_config'].to_dict()
        output['module_config'] = output['module_config'].to_dict()
        output['lora_config'] = output['lora_config'].to_dict()

        return output
