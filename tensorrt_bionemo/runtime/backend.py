# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import inspect
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import torch
import torch.nn as nn
from tensorrt_llm_lite.logger import logger

from tensorrt_bionemo.configs import BackendType, BaseConfig

from .allocator import BaseContextMemoryManager, SimpleContextMemoryManager


class FallbackStrategy:
    """Base class for backend-specific fallback strategies.

    Subclass and override :meth:`should_fallback` for custom logic.
    The default implementation never triggers a fallback.
    """

    def bind(self, backend: 'BackendBase') -> 'FallbackStrategy':
        """Bind to a concrete backend instance (called once per backend)."""
        return self

    def should_fallback(self, *args, **kwargs) -> bool:
        """Return ``True`` when the current call should fall back to torch eager."""
        return False


class TRTFallbackStrategy(FallbackStrategy):
    """Fallback when inputs exceed TRT engine optimization profile limits.

    Inspects the TRT engine's optimization profiles at runtime and checks
    whether each input tensor's shape fits within at least one profile.
    If any tensor falls outside every profile, the call is routed to the
    PyTorch fallback module instead.

    Handles batch-dimension mismatches transparently: when the tensor's
    ``ndim`` differs from the profile's expected ``ndim`` by exactly one, a
    leading dimension of size 1 is added or removed before matching.
    """

    def __init__(self):
        self._backend = None
        self._param_names: list[str] | None = None
        self._allocator: BaseContextMemoryManager | None = None
        self._opt_profile_map: dict | None = None

    def bind(self, backend: 'BackendBase') -> 'TRTFallbackStrategy':
        if self._backend is backend:
            return self
        self._backend = backend
        self._allocator = backend._context_memory_allocator

        sig = inspect.signature(backend.forward_udf)
        self._param_names = [
            name for name, param in sig.parameters.items()
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]

        self._opt_profile_map = None
        return self

    def _ensure_profiles(self) -> dict | None:
        """Lazily resolve and cache the optimization profile map."""
        if self._opt_profile_map is not None:
            return self._opt_profile_map

        handle = self._allocator.get_deserialized_handles().get(self._backend)
        if handle is None:
            return None

        if not getattr(handle, '_opt_profile_map', None):
            self._allocator.build_opt_profile_map(handle)

        self._opt_profile_map = handle._opt_profile_map or {}
        return self._opt_profile_map

    def _shape_fits_profiles(self, tensor_name: str,
                             tensor_shape: tuple) -> bool:
        profiles = self._opt_profile_map.get(tensor_name)
        if not profiles:
            return True

        for p in profiles:
            if self._allocator._shape_matches_profile(tensor_shape,
                                                      p['min_shape'],
                                                      p['max_shape']):
                return True

        expected_ndim = len(profiles[0]['min_shape'])
        if len(tensor_shape) == expected_ndim - 1:
            adjusted = (1, ) + tensor_shape
            for p in profiles:
                if self._allocator._shape_matches_profile(
                        adjusted, p['min_shape'], p['max_shape']):
                    return True
        elif len(tensor_shape) == expected_ndim + 1:
            adjusted = tensor_shape[1:]
            for p in profiles:
                if self._allocator._shape_matches_profile(
                        adjusted, p['min_shape'], p['max_shape']):
                    return True

        return False

    def should_fallback(self, *args, **kwargs) -> bool:
        if self._backend is None:
            return False

        profiles = self._ensure_profiles()
        if not profiles:
            return False

        tensor_inputs: dict[str, torch.Tensor] = {}
        for name, arg in zip(self._param_names, args):
            if isinstance(arg, torch.Tensor):
                tensor_inputs[name] = arg
        for name, val in kwargs.items():
            if isinstance(val, torch.Tensor):
                tensor_inputs[name] = val

        for tensor_name, tensor in tensor_inputs.items():
            if not self._shape_fits_profiles(tensor_name, tuple(tensor.shape)):
                return True

        return False


class TorchFallbackStrategy(FallbackStrategy):
    """Fallback strategy for torch eager / torch.compile backends.

    The base torch eager backend never needs fallback (it *is* the fallback
    target).  Override :meth:`should_fallback` for torch.compile error
    recovery or input-guard based fallback.
    """


FALLBACK_STRATEGIES: dict[str, type[FallbackStrategy]] = {
    BackendType.TRT: TRTFallbackStrategy,
    BackendType.TORCH: TorchFallbackStrategy,
}


class AutoFallback:
    """Backend-aware fallback dispatcher.

    Selects the right :class:`FallbackStrategy` from
    :data:`FALLBACK_STRATEGIES` based on the backend type
    (``"trt"``, ``"torch"``, etc.).

    ``AutoFallback`` is the **default** fallback used by
    :meth:`BackendBase.forward` when ``config.need_fallback`` is ``None``.
    This means every backend automatically gets the appropriate fallback
    behaviour without any explicit configuration:

    * **TRT backends** — falls back to torch eager when input shapes exceed
      engine optimization profiles (:class:`TRTFallbackStrategy`).
    * **Torch backends** — never triggers fallback since torch eager is
      already the fallback target (:class:`TorchFallbackStrategy`).

    To override per-module, pass a custom callable or strategy via
    ``AcceleratedConfig.need_fallback``.  To register a strategy for a
    new backend::

        FALLBACK_STRATEGIES["my_backend"] = MyFallbackStrategy
    """

    def __init__(self):
        self._backend = None
        self._strategy: FallbackStrategy | None = None

    def bind(self, backend: 'BackendBase') -> 'AutoFallback':
        """Bind to a backend, selecting the appropriate strategy."""
        if self._backend is backend:
            return self
        self._backend = backend
        backend_type = getattr(backend.config, 'backend', None)
        strategy_cls = FALLBACK_STRATEGIES.get(backend_type)
        if strategy_cls is not None:
            self._strategy = strategy_cls()
            self._strategy.bind(backend)
        else:
            self._strategy = None
        return self

    def __call__(self, *args, **kwargs) -> bool:
        """Return ``True`` if fallback is needed for the current inputs."""
        if self._strategy is None:
            return False
        return self._strategy.should_fallback(*args, **kwargs)


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

    _default_auto_fallback = AutoFallback()

    def forward(self, *args, **kwargs) -> Any:
        need_fallback = self.config.need_fallback
        if need_fallback is None:
            need_fallback = self._default_auto_fallback
        if isinstance(need_fallback, AutoFallback):
            need_fallback.bind(self)
        if need_fallback(*args, **kwargs):
            logger.info(
                f"Fallback to torch eager for {self.__class__.__name__}")
            return self._fallback_module(*args, **kwargs)
        return self.forward_udf(*args, **kwargs)
