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
from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn

from tensorrt_bionemo.configs import BackendType, BaseConfig
from tensorrt_bionemo.logger import logger


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


class TorchFallbackStrategy(FallbackStrategy):
    """Fallback strategy for torch eager / torch.compile backends.

    The base torch eager backend never needs fallback (it *is* the fallback
    target).  Override :meth:`should_fallback` for torch.compile error
    recovery or input-guard based fallback.
    """


FALLBACK_STRATEGIES: dict[str, type[FallbackStrategy]] = {
    BackendType.TORCH: TorchFallbackStrategy,
}


class AutoFallback:
    """Backend-aware fallback dispatcher.

    Selects the right :class:`FallbackStrategy` from
    :data:`FALLBACK_STRATEGIES` based on the backend type (``"torch"``, etc.).

    ``AutoFallback`` is the **default** fallback used by
    :meth:`BackendBase.forward` when ``config.need_fallback`` is ``None``.

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

    def __init__(self, config: BaseConfig):
        """ BackendBase is the base class for all backends.
        It provides the basic functionality for all backends.
        Args:
            config(BaseConfig): The configuration for the backend.
        """
        super().__init__()
        self._config = config
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
