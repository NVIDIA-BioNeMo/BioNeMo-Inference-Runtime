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

from tensorrt_bionemo.configs import AcceleratedConfig, BaseConfig
from tensorrt_bionemo.runtime import BackendType, BaseContextMemoryManager
from tensorrt_bionemo._torch.graph_optimization.config_schema import \
    GraphOptimizationMode


@dataclass
class ModuleSpec:
    """Describes a single optimizable module.

    Attributes:
        getter: ``(model) -> nn.Module`` — retrieves the original module.
        setter: ``(model, optimized) -> None`` — installs the replacement.
        trt_cls: TRT wrapper class (has ``load_weights``).  ``None`` if TRT
            is not supported for this module.
        compiled_cls: ``CompilableModule`` subclass.  ``None`` if
            torch.compile is not supported for this module.
    """
    getter: Callable
    setter: Callable
    trt_cls: Optional[type] = None
    compiled_cls: Optional[type] = None
    graph_optimization_cls: Optional[type] = None


class ModuleRegistry(ABC):

    def __init__(self, configs: dict[str, AcceleratedConfig | dict] = {}):
        """
        This class is used to store the checkpoints and module configs for the accelerated modules.
        Args:
            configs: A dictionary of AcceleratedConfig for the accelerated modules.
        """
        self._configs = {}
        all_known: dict[str, ModuleSpec] = self.get_accelerated_modules()
        for k, v in configs.items():
            if k not in all_known:
                logger.warning(f"Unknown module: {k}")
            else:
                if isinstance(v, dict):
                    v = AcceleratedConfig(**v)
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
    def get_accelerated_modules(self) -> dict[str, ModuleSpec]:
        """Return ``{name: ModuleSpec(...)}`` for every optimizable module.

        Each :class:`ModuleSpec` carries getter/setter lambdas plus the
        per-backend wrapper classes (``trt_cls`` and/or ``compiled_cls``).
        """
        raise NotImplementedError("Subclass must implement this method")


class OptimizedModuleSetterMixin(ABC):

    @abstractmethod
    def get_optimized_modules(
            self,
            accelerated_configs: dict[str,
                                      AcceleratedConfig]) -> ModuleRegistry:
        raise NotImplementedError("Subclass must implement this method")

    def optimize(self,
                 accelerated_configs: dict[str, AcceleratedConfig],
                 context_memory_allocator: Optional[
                     BaseContextMemoryManager] = None,
                 **kwargs) -> nn.Module:
        """Build the optimized version of the model from the original.

        Supports TRT engine backends and torch backends (optionally with
        ``torch.compile`` when ``compile=True``).

        Args:
            accelerated_configs: A dictionary of modules to be accelerated.
                Each key is a module name (e.g. ``"evoformer"``) and the value
                is an :class:`AcceleratedConfig` whose ``backend`` field
                selects ``"trt"`` or ``"torch"``.  Set ``compile=True`` on
                torch-backend configs to enable ``torch.compile``.
            context_memory_allocator: The context memory allocator to be
                used for TRT modules.
        Returns:
            The optimized model (self, modified in-place).
        """
        optimized_modules = self.get_optimized_modules(accelerated_configs)
        module_specs = optimized_modules.get_accelerated_modules()

        for module_name in optimized_modules.get_module_names():
            acc_config = optimized_modules.get_module_config(module_name)
            if acc_config is None:
                continue
            spec = module_specs.get(module_name)
            if spec is None:
                continue
            backend = acc_config.backend

            # ── TRT engine path ─────────────────────────────────────────
            if backend == BackendType.TRT and spec.trt_cls is not None:
                org = spec.getter(self)
                checkpoint_dir = optimized_modules.get_module_checkpoint(
                    module_name)
                opt_m = spec.trt_cls.load_weights(
                    checkpoint_dir=checkpoint_dir,
                    context_memory_allocator=context_memory_allocator,
                    **kwargs)
                spec.setter(self, opt_m)
                opt_m.set_fallback_module(org)
                opt_m.config.need_fallback = (
                    optimized_modules.get_module_need_fallback(module_name))

            # ── torch + CUDA-graph path ─────────────────────────────────
            # Mirrors the TRT path (req 4.1): the eager module is replaced by a
            # graph-compilation tracker that keeps the original as its eager
            # ``inner_module`` / fallback.
            elif backend == BackendType.TORCH and spec.graph_optimization_cls is not None:
                org = spec.getter(self)
                graph_config = getattr(acc_config.default, "graph_optimization_config",
                                       None) if acc_config.default else None
                if graph_config is not None and graph_config.graph_optimization_mode != GraphOptimizationMode.NO_OPTIMIZATION:
                    opt_m = spec.graph_optimization_cls(config=graph_config,
                                                       inner_module=org)
                    opt_m.set_fallback_module(org)
                    spec.setter(self, opt_m)

        return self
