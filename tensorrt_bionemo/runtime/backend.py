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
from abc import ABC, abstractmethod
from typing import Any, Union
from pathlib import Path
import torch.nn as nn
from tensorrt_llm_lite.logger import logger

from tensorrt_bionemo.configs import BackendType, BaseConfig

from .allocator import BaseContextMemoryManager, SimpleContextMemoryManager


class BackendBase(nn.Module, ABC):
    CONFIG_CLASS = None

    def __init__(self,
                 config: BaseConfig,
                 context_memory_allocator: BaseContextMemoryManager = None):
        """ BackendBase is the base class for all backends.
        It provides the basic functionality for all backends.
        Args:
            config(BaseConfig): The configuration for the backend.
            context_memory_allocator(BaseContextMemoryManager): The context memory allocator to use. If None, the default allocator will be used.
        """
        super().__init__()
        self._config = config
        self._context_memory_allocator = context_memory_allocator
        if self._context_memory_allocator is None:
            self._context_memory_allocator = SimpleContextMemoryManager()
        self._loaded_by_manager = False
        self._world_size = 1
        self._runtime_rank = 0
        self._checkpoint_dir = None
        self._fallback_module = None

    def set_fallback_module(self, fallback_module: nn.Module):
        self._fallback_module = fallback_module

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: BaseConfig):
        self._config = config

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @checkpoint_dir.setter
    def checkpoint_dir(self, checkpoint_dir: str):
        self._checkpoint_dir = checkpoint_dir

    @property
    def world_size(self):
        return self._world_size

    @world_size.setter
    def world_size(self, world_size: int):
        self._world_size = world_size

    @property
    def runtime_rank(self):
        return self._runtime_rank

    @runtime_rank.setter
    def runtime_rank(self, runtime_rank: int):
        self._runtime_rank = runtime_rank

    def reset(self):
        """
        This method is used to reset the cache or something else for an backend implementation.
        """

    def warmup(self):
        """
        This method is used to warmup the backend implementation (i.e. torch.compile).
        """

    @classmethod
    def load_weights(cls,
                     checkpoint_dir: Union[str, Path] = None,
                     context_memory_allocator: BaseContextMemoryManager = None,
                     **kwargs):
        """
        Load weights into the backend.
        Args:
            checkpoint_dir(str): The directory to load the checkpoint from.
            world_size(int): Reversed for parallelism
            rank(int): Reversed for parallelism
            **kwargs: Additional arguments to pass to the load_weights_fn.
        """
        assert cls.CONFIG_CLASS is not None, "CONFIG_CLASS must be set for the backend: {cls.__name__}"
        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)
        backend_dir = checkpoint_dir / str(BackendType.TRT)
        if backend_dir.exists():
            # Build from trtbnm-build
            config_path = backend_dir / "config.json"
            with open(config_path, "r") as f:
                config_dict = json.load(f)["pretrained_config"]
        else:
            # Build for testing purposes
            backend_dir = checkpoint_dir
            config_path = checkpoint_dir / "config.json"
            with open(config_path, "r") as f:
                config_dict = json.load(f)

        _config = cls.CONFIG_CLASS.model_validate(config_dict)

        if "stream" in kwargs:
            _stream = kwargs["stream"]
        else:
            _stream = None

        module = cls(config=_config,
                     context_memory_allocator=context_memory_allocator)
        module.checkpoint_dir = backend_dir
        assert context_memory_allocator is not None, "Context memory allocator is not set"
        context_memory_allocator.add_handle(module, stream=_stream)

        return module

    @abstractmethod
    def forward_udf(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Subclass must implement this method")

    def forward(self, *args, **kwargs) -> Any:
        if self.config.need_fallback is not None:
            if self.config.need_fallback(*args, **kwargs):
                logger.info(
                    f"Fallback to torch backend for {self.__class__.__name__}")
                return self._fallback_module(*args, **kwargs)
        else:
            logger.info(f"No fallback needed for {self.__class__.__name__}")
        return self.forward_udf(*args, **kwargs)
