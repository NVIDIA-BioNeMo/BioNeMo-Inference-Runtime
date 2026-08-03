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

from typing import Any, Callable, Literal, Optional, Union

import torch
from pydantic import BaseModel, Field, SkipValidation, model_validator

from tensorrt_bionemo.utils import str_dtype_to_torch, torch_dtype_to_str
from tensorrt_bionemo.version import __version__


class BackendType:
    TORCH = "torch"

    @classmethod
    def is_supported(cls, backend: str) -> bool:
        return backend in [cls.TORCH]

    @staticmethod
    def from_str(backend: str) -> "BackendType":
        if backend == "torch":
            return BackendType.TORCH
        else:
            raise ValueError(f"Invalid backend: {backend}")


class BaseConfig(BaseModel):
    """
    Base configuration for all modules, containing recursively settable
    fields for all sub-modules.
    It can propagate some common configurations to all sub-modules, such as dtype, etc.

    Note: fields are propagated by value. Reference fields (one sub-module
    config pointing at another's field) are not supported.
    """
    dtype: str = "float32"
    norm_epsilon: float = 1e-5
    mask_inf: float = 1e9
    skip_create_weights: bool = False
    triangle_attention_backend: str = "VANILLA"
    pairwise_attention_backend: str = "SDPA"
    support_batch: bool = True
    backend: Union[str, BackendType] = BackendType.TORCH
    # Opt-in CUDA-graph compilation for this module. Holds a
    # ``None | GraphOptimizationConfig`` (typed ``Any`` to avoid a config <->
    # graph_optimization import cycle). ``None`` means "leave eager"; a config
    # marks this module for wrapping by ``trtbnm_apply_graph_optimization`` /
    # ``OptimizedModuleSetterMixin.optimize`` when ``backend == TORCH``.
    graph_optimization_config: Optional[Any] = None
    # This function is used to determine if the module needs to fallback to the torch backend based on the input arguments
    need_fallback: Optional[Callable] = Field(exclude=True, default=None)
    max_batch_size: int = 1
    max_seq_len: int = 2048
    min_seq_len: int = 4
    package_version: str = __version__

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    def copy_and_validate(self, **update):
        data = self.model_dump()
        data.update(update)
        return type(self).model_validate(data)

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

    def to_dict(self):
        # Return the config as a plain, JSON-serialisable dict.
        output = self.model_dump()
        return output


def print_model_tree(model: BaseModel, indent: int = 0):
    """ Print the model tree for debugging purposes """
    prefix = "  " * indent
    for name, value in model:
        if isinstance(value, BaseModel):
            print(f"{prefix}{name}:")
            print_model_tree(value, indent + 1)
        else:
            print(f"{prefix}{name}: {value}")


class AcceleratedConfig(BaseModel):
    checkpoint: Optional[str] = None
    backend: Optional[str] = None
    default: Optional[BaseConfig] = None
    warmup: bool = False
    compile: bool = False
    need_fallback: Optional[Callable[..., bool]] = None

    class Config:
        arbitrary_types_allowed = True


class PostProcessorConfig(BaseModel):

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True


Device = Literal["auto", "cuda", "cpu"]


class DeviceConfig(BaseModel):
    device: SkipValidation[Device | torch.device | None] = "auto"
    device_type: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def auto_set_device(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device_type = "cuda"
            else:
                self.device_type = "cpu"
        else:
            # Device type is assigned explicitly
            if isinstance(self.device, str):
                self.device_type = self.device
            elif isinstance(self.device, torch.device):
                self.device_type = self.device.type
        self.device = torch.device(self.device_type)
        return self


class EngineConfig(BaseModel):
    name: Optional[str] = None
    model: Optional[BaseConfig] = None
    device: Optional[DeviceConfig] = None
    accelerated: Optional[dict[str, AcceleratedConfig]] = None
    postprocessor: Optional[PostProcessorConfig] = None
    profile_inference: bool = False
