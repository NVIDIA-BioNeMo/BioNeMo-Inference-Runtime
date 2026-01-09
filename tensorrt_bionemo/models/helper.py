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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import torch.nn as nn
from tensorrt_llm_lite.logger import logger

from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.runtime import BackendType, BaseContextMemoryManager


@dataclass
class AcceleratedConfig:
    checkpoint: str = None
    backend: BackendType = None
    default: BaseConfig = None
    warmup: bool = False
    compile: bool = False
    need_fallback: Optional[Callable[..., bool]] = None


class AcceleratedModules(ABC):

    def __init__(self, configs: dict[str, AcceleratedConfig] = {}):
        """
        This class is used to store the checkpoints and module configs for the accelerated modules.
        Args:
            configs: A dictionary of AcceleratedConfig for the accelerated modules.
        """
        self._configs = {}
        for k, v in configs.items():
            if k not in self.get_supported_modules().keys():
                logger.warning(f"Unknown module: {k}")
            else:
                self._configs[k] = v

    def get_module_config(self,
                          module_name: str) -> Optional[AcceleratedConfig]:
        return self._configs.get(module_name, None)

    def get_module_backend(self, module_name: str) -> Optional[BackendType]:
        return self._configs.get(module_name, None).backend

    def get_module_checkpoint(self, module_name: str) -> Optional[str]:
        return self._configs.get(module_name, None).checkpoint

    def get_module_names(self) -> list[str]:
        return list(self._configs.keys())

    def get_default_module_config(self,
                                  module_name: str) -> Optional[BaseConfig]:
        return self._configs.get(module_name, None).default

    def get_module_need_fallback(self, module_name: str) -> Optional[Callable]:
        return self._configs.get(module_name, None).need_fallback

    @abstractmethod
    def get_supported_modules(self) -> dict[str, nn.Module]:
        raise NotImplementedError("Subclass must implement this method")


class OptimizedModuleSetterMixin:

    def optimize(self,
                 accelerated_modules: AcceleratedModules,
                 context_memory_allocator: Optional[
                     BaseContextMemoryManager] = None,
                 **kwargs) -> nn.Module:
        """
        This function is used to build the optimized version of Boltz1 model from the original.
        Args:
            accelerated_modules: A dictionary of modules to be accelerated.
            context_memory_allocator: The context memory allocator to be used for each module.
        Returns:
            The optimized model.
        """
        supported_modules = accelerated_modules.get_supported_modules()

        for module_name, (cls_, setter_func) in supported_modules.items():
            backend = accelerated_modules.get_module_backend(module_name)
            if backend != BackendType.TRT:
                # Only support for TRT backend for now
                continue
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                module_name)
            opt_m = cls_.load_weights(
                checkpoint_dir=checkpoint_dir,
                context_memory_allocator=context_memory_allocator,
                **kwargs)
            org = setter_func(self, opt_m)
            opt_m.set_fallback_module(org)
            opt_m.config.need_fallback = accelerated_modules.get_module_need_fallback(
                module_name)
        return self
