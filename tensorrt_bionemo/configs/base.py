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
from pathlib import Path
from typing import Callable, Generic, Optional, TypeVar, Union

import dill
import transformers
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.logger import logger

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

    def set_dtype(self, value: str):
        if value is None:
            value = "float32"
        self.torch_dtype = str_dtype_to_torch(value)
        self.dtype = value


class TorchLoadWeightsMetadata:

    def __init__(self, load_weights_fn: Callable, load_weights_fn_kwargs: dict,
                 compile: bool):
        self._load_weights_fn = load_weights_fn
        self._load_weights_fn_kwargs = load_weights_fn_kwargs
        self._compile = compile

    def dump(self, file_path: Union[str, Path]):
        with open(file_path, 'wb') as f:
            dill.dump(
                {
                    "load_weights_fn": self._load_weights_fn,
                    "load_weights_fn_kwargs": self._load_weights_fn_kwargs,
                    "compile": self._compile
                }, f)

    @classmethod
    def load(cls, file_path: Union[str, Path]):
        with open(file_path, 'rb') as f:
            return cls(**dill.load(f))

    @property
    def load_weights_fn(self):
        return self._load_weights_fn

    @property
    def load_weights_fn_kwargs(self):
        return self._load_weights_fn_kwargs

    @property
    def compile(self):
        return self._compile
