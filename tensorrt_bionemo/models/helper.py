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

import torch
from tensorrt_llm.logger import logger

from tensorrt_bionemo.config import PretrainedModuleConfig
from tensorrt_bionemo.runtime import (BackendBuilder, BackendType,
                                      BaseContextMemoryManager)


@dataclass
class AcceleratedConfig:
    checkpoint: str = None
    backend: BackendType = None
    default: PretrainedModuleConfig = None


class AcceleratedModules(ABC):

    def __init__(self, configs: dict[str, AcceleratedConfig] = {}):
        """
        This class is used to store the checkpoints and module configs for the accelerated modules.
        Args:
            configs: A dictionary of AcceleratedConfig for the accelerated modules.
        """
        self._configs = {}
        for k, v in configs.items():
            if k not in self.get_supported_module_names():
                logger.warning(f"Unknown module: {k}")
            else:
                self._configs[k] = v

    def get_module_backend(self, module_name: str):
        return self._configs.get(module_name, None).backend

    def get_module_checkpoint(self, module_name: str):
        return self._configs.get(module_name, None).checkpoint

    def get_module_names(self):
        return self._configs.keys()

    def get_default_module_config(self, module_name: str):
        return self._configs.get(module_name, None).default

    @abstractmethod
    def get_supported_module_names(self):
        raise NotImplementedError("Subclass must implement this method")


def build_optimized_module(
        state_dict: dict,
        module_name: str,
        backend_builder: BackendBuilder,
        checkpoint_dir: str = None,
        backend: BackendType = BackendType.TORCH,
        default_config: PretrainedModuleConfig = None,
        convert_weights_func: Callable = None,
        convert_weights_func_kwargs: dict = {},
        context_memory_allocator: Optional[BaseContextMemoryManager] = None,
        compile: bool = False,
        device: torch.device = None):
    if checkpoint_dir is not None:
        return backend_builder.build(
            checkpoint_dir=checkpoint_dir,
            backend=backend,
            context_memory_allocator=context_memory_allocator)
    if backend == BackendType.TORCH:
        config = default_config
        weights = convert_weights_func(**convert_weights_func_kwargs)
        module = backend_builder.build(checkpoint_dir=None,
                                       config=config,
                                       backend=BackendType.TORCH,
                                       compile=compile,
                                       weights=weights).to(device).eval()
        logger.info(f"Using default Torch Backend for {module_name}")
    else:
        raise ValueError(
            f"`{module_name}`: Default accelerated module config is not provided for {backend} backend"
        )
    return module
