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

import tensorrt_llm
import torch
import torch.nn as nn
from tensorrt_llm.logger import logger

from tensorrt_bionemo.config import PretrainedModuleConfig

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
                 impl: nn.Module = None,
                 context_memory_allocator: BaseContextMemoryManager = None):
        """ BackendBase is the base class for all backends.
        It provides the basic functionality for all backends.
        Args:
            config(PretrainedModuleConfig): The configuration for the backend.
            impl(nn.Module): The implementation of the backend.
                             If None, the implementation will be created by the IMPL_CLASS.
                             This is to use for debugging purposes.
            context_memory_allocator(BaseContextMemoryManager): The context memory allocator to use. If None, the default allocator will be used.
        """
        super().__init__()
        self._config = config
        self._module = impl
        self._context_memory_allocator = context_memory_allocator
        if self._context_memory_allocator is None:
            self._context_memory_allocator = SimpleContextMemoryManager()
        self._loaded_by_manager = False

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
                     checkpoint_dir: str = None,
                     world_size: int = 1,
                     rank: int = 0,
                     weights: dict = None,
                     compile: bool = True,
                     loaded_by_manager: bool = False,
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
        self._checkpoint_dir = checkpoint_dir
        self._world_size = world_size
        self._runtime_rank = rank
        if "stream" in kwargs:
            self._stream = kwargs["stream"]
        else:
            self._stream = None
        self._loaded_by_manager = loaded_by_manager
        if self._loaded_by_manager:
            self._context_memory_allocator.add_handle(self, stream=self._stream)
            return

        if self._module is not None:
            self._module.load_weights(weights=weights)
        elif self.IMPL_CLASS is not None:
            assert issubclass(
                self.IMPL_CLASS,
                nn.Module), "IMPL_CLASS must be a subclass of nn.Module"
            self._module = self.IMPL_CLASS(self.config)
            self._module.load_weights(weights=weights)
        else:
            raise NotImplementedError(
                "IMPL_CLASS is not implemented and self._module is None")
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


class BackendBuilder(ABC):
    BACKEND_CLASSES = {}
    CONFIG_CLASS = None

    # @overload
    @classmethod
    def build(cls,
              checkpoint_dir: str,
              backend: str,
              context_memory_allocator: BaseContextMemoryManager = None,
              compile: bool = True,
              weights: dict = None,
              config: PretrainedModuleConfig = None,
              **kwargs) -> nn.Module:
        """
        Build the backend module from the checkpoint directory.
        Args:
            checkpoint_dir(str): The directory to load the checkpoint from.
            backend(str): The backend to use.
            context_memory_allocator(BaseContextMemoryManager): The context memory allocator to use. If None, the default allocator will be used.
            compile(bool): Whether to compile the module.
            weights(dict): The weights to load into the module. If None, the weights will be loaded from the checkpoint directory.
            **kwargs: Additional arguments to pass to the load_weights method.
        """
        backend_checkpoint_dir = None
        rank = tensorrt_llm.mpi_rank()

        if checkpoint_dir is not None:
            backend_checkpoint_dir = Path(checkpoint_dir) / backend
            if backend not in cls.BACKEND_CLASSES:
                raise ValueError(f"Invalid backend: {backend}")
            config_path = backend_checkpoint_dir / "config.json"
            with open(config_path, "r") as f:
                config_dict = json.load(f)["pretrained_config"]
            config_dict["backend"] = backend
            assert cls.CONFIG_CLASS is not None, f"CONFIG_CLASS must be set for the backend builder: {cls.__name__}"
            config = cls.CONFIG_CLASS.from_dict(config_dict)

        assert config is not None, f"config must be provided for the backend builder: {cls.__name__}"

        if backend == BackendType.TORCH:
            if weights is None:
                weights = torch.load(backend_checkpoint_dir / f"weights.pt")
            module = cls.BACKEND_CLASSES[backend](config)
            loaded_by_manager = False
        else:
            module = cls.BACKEND_CLASSES[backend](
                config, context_memory_allocator=context_memory_allocator)
            loaded_by_manager = True
            # still try to load the torch backend if any.
            # This is for the case that the TRT backend is also use the Torch backend inside.
            # For example, the Boltz1 TRT TokenTransformer is using along with the Torch backend.
            try:
                torch_backend_dir = Path(checkpoint_dir) / BackendType.TORCH
                weights = torch.load(torch_backend_dir / f"weights.pt")
            except Exception as e:
                logger.warning(
                    f"Torch backend weights are not available along with the TRT backend: {e}"
                )

        world_size = config.mapping.world_size
        module.load_weights(checkpoint_dir=backend_checkpoint_dir,
                            world_size=world_size,
                            rank=rank,
                            weights=weights,
                            compile=compile,
                            loaded_by_manager=loaded_by_manager,
                            **kwargs)
        return module
