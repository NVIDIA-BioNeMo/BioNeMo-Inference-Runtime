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

import json
from collections import OrderedDict
from typing import Any, Callable, Optional, Union

import torch
from pydantic import BaseModel, Field, field_serializer, field_validator
from tensorrt_llm._utils import str_dtype_to_torch, torch_dtype_to_str
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.plugin import PluginConfig
from tensorrt_llm.quantization import QuantAlgo

from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.version import __version__


class BackendType:
    TRT = "trt"
    TORCH = "torch"

    @classmethod
    def is_supported(cls, backend: str) -> bool:
        return backend in [cls.TRT, cls.TORCH]

    @staticmethod
    def from_str(backend: str) -> "BackendType":
        if backend == "trt":
            return BackendType.TRT
        elif backend == "torch":
            return BackendType.TORCH
        else:
            raise ValueError(f"Invalid backend: {backend}")


class BaseConfig(BaseModel):
    """
    Base configuration for all modules and use for both Torch and TensorRT backends, contains recursively settable fields for all sub-modules.
    It can propagate some common configurations to all sub-modules, such as dtype, mapping, etc.
    TODO: Support for reference fields
    """
    dtype: str = "float32"
    norm_epsilon: float = 1e-5
    mask_inf: float = 1e9
    skip_create_weights: bool = False
    mapping: Mapping = Mapping()
    disable_custom_all_reduce: bool = False
    triangle_attention_backend: str = "VANILLA"
    pairwise_attention_backend: str = "VANILLA"
    support_batch: bool = True
    backend: Union[str, BackendType] = BackendType.TORCH
    max_batch_size: int = 1
    max_seq_len: int = 2048
    min_seq_len: int = 4
    quant_algo: Optional[QuantAlgo] = None
    package_version: str = __version__

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    def copy_and_validate(self, **update):
        data = self.model_dump()
        data.update(update)
        return type(self).model_validate(data)

    @field_serializer("mapping")
    def serialize_mapping(self, mapping: Union[Mapping, dict]):
        # Convert the custom object to something serializable
        if isinstance(mapping, Mapping):
            return mapping.to_dict()
        return mapping

    @field_validator("mapping", mode="before")
    def parse_mapping(cls, v):
        if isinstance(v, dict):
            return Mapping.from_dict(v)
        return v

    @property
    def torch_dtype(self) -> torch.dtype:
        return str_dtype_to_torch(self.dtype)

    def _recursive_set(self, setter_func: Callable):
        for field_name in self.__dict__.keys():
            field = getattr(self, field_name)
            if isinstance(field, BaseConfig) or issubclass(
                    field.__class__, BaseConfig):
                field._recursive_set(setter_func)
        for field_name in self.__pydantic_extra__.keys():
            field = getattr(self, field_name)
            if isinstance(field, BaseConfig) or issubclass(
                    field.__class__, BaseConfig):
                field._recursive_set(setter_func)
        setter_func(self)

    def set_dtype(self, value: Union[str, torch.dtype]) -> None:
        """ Recursively set the dtype for all fields in the model """

        def setter(x):
            if isinstance(value, torch.dtype):
                x.dtype = torch_dtype_to_str(value)
            else:
                x.dtype = value

        self._recursive_set(setter)

    def set_mapping(self, value: Mapping):

        def setter(x):
            x.mapping = value

        self._recursive_set(setter)

    def set_norm_epsilon(self, value: float):

        def setter(x):
            x.norm_epsilon = value

        self._recursive_set(setter)

    def set_mask_inf(self, value: float):

        def setter(x):
            x.mask_inf = value

        self._recursive_set(setter)

    def set_skip_create_weights(self, value: bool):

        def setter(x):
            x.skip_create_weights = value

        self._recursive_set(setter)

    def set_disable_custom_all_reduce(self, value: bool):

        def setter(x):
            x.disable_custom_all_reduce = value

        self._recursive_set(setter)

    def set_triangle_attention_backend(self, value: str):

        def setter(x):
            x.triangle_attention_backend = value

        self._recursive_set(setter)

    def set_pairwise_attention_backend(self, value: str):

        def setter(x):
            x.pairwise_attention_backend = value

        self._recursive_set(setter)

    def set_backend(self, value: str):

        def setter(x):
            x.backend = BackendType.from_str(value)

        self._recursive_set(setter)

    def set_max_batch_size(self, value: int):

        def setter(x):
            x.max_batch_size = value

        self._recursive_set(setter)

    def set_max_seq_len(self, value: int):

        def setter(x):
            x.max_seq_len = value

        self._recursive_set(setter)

    def set_min_seq_len(self, value: int):

        def setter(x):
            x.min_seq_len = value

        self._recursive_set(setter)

    def set_rank(self, value: int):

        def setter(x):
            x.mapping.rank = value

        self._recursive_set(setter)

    def to_dict(self):
        # Support this function for compatibility with the TensorRT-LLM build() function
        output = self.model_dump()
        return output


class DimSpec(BaseModel):
    size: int = -1
    name: str = ""
    dynamic: bool = False
    min: int = -1
    max: int = -1


class BuildConfig(BaseModel):
    """ TensorRT engines building configurations """

    strongly_typed: bool = False
    weakly_dtype: Optional[str] = None
    force_num_profiles: Optional[int] = None
    profiling_verbosity: Optional[str] = 'layer_names_only'
    plugin_config: PluginConfig = Field(default_factory=PluginConfig)
    module_config: Optional[BaseConfig] = None
    input_timing_cache: Optional[str] = None
    output_timing_cache: Optional[str] = 'model.cache'
    dry_run: Optional[bool] = False
    monitor_memory: Optional[bool] = False
    enable_debug_output: Optional[bool] = False
    lora_config: LoraConfig = Field(
        default_factory=LoraConfig)  # Patch for save engine

    def get_optimization_profiles(self) -> list[Any]:
        raise NotImplementedError("Subclasses must implement this method")

    def get_input_shapes(self) -> OrderedDict[str, DimSpec]:
        raise NotImplementedError("Subclasses must implement this method")

    def get_output_shapes(self) -> OrderedDict[str, DimSpec]:
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def from_json_file(cls, config_file):
        # Support this function for compatibility with the TensorRT-LLM build() function
        with open(config_file) as f:
            config = json.load(f)
        return BuildConfig(**config)

    def to_dict(self):
        # Support this function for compatibility with the TensorRT-LLM build() function
        output = self.model_dump()
        del output['module_config']
        return output


def create_optimization_profiles(build_config: BuildConfig,
                                 seqlen_key_names: list[str] = ["seqlen"],
                                 align: int = 16) -> list[Any]:
    """ Create optimization profiles for the build config """
    input_shapes = build_config.get_input_shapes()

    if build_config.force_num_profiles == 0:
        return []

    mc = build_config.module_config
    min_seqlen = mc.min_seq_len - mc.min_seq_len % align
    max_seqlen = (mc.max_seq_len + align - 1) // align * align
    assert (max_seqlen - min_seqlen) % build_config.force_num_profiles == 0
    step = (max_seqlen - min_seqlen) // build_config.force_num_profiles

    min_max_seqlens = []

    for i in range(build_config.force_num_profiles):
        min_max_seqlens.append(
            (min_seqlen + i * step, min_seqlen + (i + 1) * step))

    profiles = []
    for rmin, rmax in min_max_seqlens:
        profile = {}
        for k, v in input_shapes.items():
            min_shape = []
            opt_shape = []
            max_shape = []

            for spec in v:
                if spec.name in seqlen_key_names:
                    min_shape.append(rmin)
                    opt_shape.append(rmax)
                    max_shape.append(rmax)
                elif spec.name == "multiplicity":
                    min_shape.append(1)
                    opt_shape.append(mc.multiplicity)
                    max_shape.append(mc.multiplicity)
                elif spec.name == "batch_size":
                    min_shape.append(1)
                    opt_shape.append(mc.max_batch_size)
                    max_shape.append(mc.max_batch_size)
                else:
                    min_shape.append(spec.size)
                    opt_shape.append(spec.size)
                    max_shape.append(spec.size)
            profile[k] = (min_shape, opt_shape, max_shape)
        profiles.append(profile)
    return profiles


def print_model_tree(model: BaseModel, indent: int = 0):
    """ Print the model tree for debugging purposes """
    prefix = "  " * indent
    for name, value in model:
        if isinstance(value, BaseModel):
            print(f"{prefix}{name}:")
            print_model_tree(value, indent + 1)
        else:
            print(f"{prefix}{name}: {value}")
