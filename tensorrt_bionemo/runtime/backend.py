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
from abc import ABC
from pathlib import Path
from typing import Callable, Optional

import tensorrt_llm
import torch
import torch.nn as nn

from tensorrt_bionemo.configs import (PretrainedModuleConfig,
                                      TorchLoadWeightsMetadata)

from .allocator import BaseContextMemoryManager, SimpleContextMemoryManager


class BackendType:
    TRT = "trt"
    TORCH = "torch"

    @classmethod
    def is_supported(cls, backend: str) -> bool:
        return backend in [cls.TRT, cls.TORCH]


class BackendBase(nn.Module):
    IMPL_CLASS = None

    def __init__(self,
                 config: PretrainedModuleConfig,
                 load_weights_fn: Optional[Callable] = None,
                 load_weights_fn_kwargs: dict = {},
                 impl: nn.Module = None,
                 context_memory_allocator: BaseContextMemoryManager = None):
        """ BackendBase is the base class for all backends.
        It provides the basic functionality for all backends.
        Args:
            config(PretrainedModuleConfig): The configuration for the backend.
            load_weights_fn(Optional[Callable]): The function to load the weights.
            impl(nn.Module): The implementation of the backend. If None, the implementation will be created by the IMPL_CLASS.
            context_memory_allocator(BaseContextMemoryManager): The context memory allocator to use. If None, the default allocator will be used.
        """
        super().__init__()
        self._config = config
        self._load_weights_fn = load_weights_fn
        self._module = impl
        self._load_weights_fn_kwargs = load_weights_fn_kwargs
        self._context_memory_allocator = context_memory_allocator
        if self._context_memory_allocator is None:
            self._context_memory_allocator = SimpleContextMemoryManager()

    @property
    def config(self):
        return self._config

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def world_size(self):
        return self._world_size

    @property
    def runtime_rank(self):
        return self._runtime_rank

    def reset(self):
        """
        This method is used to reset the cache or something else for an backend implementation.
        """

    def load_weights(self,
                     checkpoint_dir: str,
                     world_size: int = 1,
                     rank: int = 0,
                     weights: dict = None,
                     compile: bool = True,
                     **kwargs):
        """
        TODO: add caching cudagraphs support here
        Load weights into the backend.
        Args:
            checkpoint_dir(str): The directory to load the checkpoint from.
            world_size(int): The number of processes to use.
            rank(int): The rank of the process.
            weights(dict): The weights to load into the module.
            compile(bool): Whether to compile the module.
            **kwargs: Additional arguments to pass to the load_weights_fn.
        """
        if self._module is not None:
            return
        if self.IMPL_CLASS is not None:
            assert issubclass(
                self.IMPL_CLASS,
                nn.Module), "IMPL_CLASS must be a subclass of nn.Module"
            self._checkpoint_dir = checkpoint_dir
            self._world_size = world_size
            self._runtime_rank = rank
            self._module = self.IMPL_CLASS(self.config)
            if self._load_weights_fn_kwargs is not None:
                kwargs.update(self._load_weights_fn_kwargs)
            if self._load_weights_fn is not None:
                self._load_weights_fn(self._module,
                                      checkpoint_dir=checkpoint_dir,
                                      world_size=world_size,
                                      rank=rank,
                                      weights=weights,
                                      **kwargs)
            self._module.cuda()
            self._module.eval()
            # TODO: Whether use torch.compile() or not, checking chunking configurations
            mode = None
            if compile:
                if self.config.mapping.world_size > 1:
                    mode = "max-autotune-no-cudagraphs"
                self._module = torch.compile(self._module,
                                             fullgraph=True,
                                             dynamic=True,
                                             mode=mode)
        else:
            raise NotImplementedError(
                "load_weights is not implemented for this backend")


class BackendBuilder(ABC):
    BACKEND_CLASSES = {}
    CONFIG_CLASS = None

    # @overload
    @classmethod
    def build(cls,
              checkpoint_dir: str,
              backend: str,
              with_torch_load_fn: bool = False,
              context_memory_allocator: BaseContextMemoryManager = None,
              **kwargs) -> nn.Module:
        """
        Build the backend module from the checkpoint directory.
        Args:
            checkpoint_dir(str): The directory to load the checkpoint from.
            backend(str): The backend to use.
            with_torch_load_fn(bool): Whether to use the torch load function.
        """
        backend_checkpoint_dir = Path(checkpoint_dir) / backend
        if backend not in cls.BACKEND_CLASSES:
            raise ValueError(f"Invalid backend: {backend}")
        config_path = backend_checkpoint_dir / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)["pretrained_config"]
        config_dict["backend"] = backend
        load_weights_fn = None
        load_weights_fn_kwargs = {}
        rank = tensorrt_llm.mpi_rank()
        assert cls.CONFIG_CLASS is not None, f"CONFIG_CLASS must be set for the backend builder: {cls.__name__}"
        config = cls.CONFIG_CLASS.from_dict(config_dict)
        compile = True
        if backend == BackendType.TORCH or with_torch_load_fn:
            torch_checkpoint_dir = Path(checkpoint_dir) / BackendType.TORCH
            torch_load_weights_metadata = TorchLoadWeightsMetadata.load(
                torch_checkpoint_dir / f"rank{rank}.pkl")
            load_weights_fn = torch_load_weights_metadata.load_weights_fn
            load_weights_fn_kwargs = torch_load_weights_metadata.load_weights_fn_kwargs
            compile = torch_load_weights_metadata.compile if torch_load_weights_metadata.compile is not None else True

        if backend == BackendType.TORCH:
            module = cls.BACKEND_CLASSES[backend](config, load_weights_fn,
                                                  load_weights_fn_kwargs,
                                                  context_memory_allocator=context_memory_allocator)
            torch_load_weights_fn = None
        else:
            module = cls.BACKEND_CLASSES[backend](config, context_memory_allocator=context_memory_allocator)
            torch_load_weights_fn = load_weights_fn

        world_size = config.mapping.world_size
        module.load_weights(checkpoint_dir=backend_checkpoint_dir,
                            world_size=world_size,
                            rank=rank,
                            compile=compile,
                            torch_load_weights_fn=torch_load_weights_fn,
                            **kwargs)
        return module
