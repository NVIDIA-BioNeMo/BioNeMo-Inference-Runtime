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

from tensorrt_bionemo.configs import AcceleratedConfig, BackendType, BaseConfig
from tensorrt_bionemo._torch.graph_optimization.config import \
    GraphOptimizationMode
from tensorrt_bionemo._torch.graph_optimization.decorator import \
    GRAPH_OPT_DEFAULT_ATTR
from tensorrt_bionemo.logger import logger


@dataclass
class ModuleSpec:
    """Describes a single optimizable module.

    Attributes:
        getter: ``(model) -> nn.Module`` — retrieves the original module.
        setter: ``(model, optimized) -> None`` — installs the replacement.
        compiled_cls: ``CompilableModule`` subclass.  ``None`` if
            torch.compile is not supported for this module.
    """
    getter: Callable
    setter: Callable
    compiled_cls: Optional[type] = None
    graph_optimization_cls: Optional[type] = None


class ModuleRegistry(ABC):

    def __init__(self, configs: dict[str, AcceleratedConfig | dict] = {}):
        """
        This class is used to store the checkpoints and module configs for the accelerated modules.
        Args:
            configs: A dictionary of AcceleratedConfig for the accelerated modules.
        """
        all_known: dict[str, ModuleSpec] = self.get_accelerated_modules()
        self._configs = self._select_module_configs(all_known, configs)

    def _select_module_configs(
        self, all_known: dict[str, ModuleSpec],
        configs: dict[str, AcceleratedConfig | dict]
    ) -> dict[str, AcceleratedConfig]:
        """Validate requested module configs against the known registry.

        Unknown modules are warned about and skipped. A child module and its
        parent can't both be accelerated (the child lives inside the parent),
        so a requested child is skipped when its parent is *also* requested;
        requesting the child on its own stays valid.

        +--------------------------------------+----------------------------+
        | Requested                            | Result                     |
        +--------------------------------------+----------------------------+
        | token_transformer alone              | kept (was broken)          |
        | diffusion_module + token_transformer | child dropped, parent wins |
        | token_transformer + pairformer       | both kept                  |
        +--------------------------------------+----------------------------+
        """
        requested = {k: all_known[k] for k in configs if k in all_known}
        to_drop = self._child_module_names(requested)
        selected: dict[str, AcceleratedConfig] = {}
        for k, v in configs.items():
            if k not in all_known:
                logger.warning(f"Unknown module: {k}")
            elif k in to_drop:
                logger.warning(
                    f"Module '{k}' is nested inside another requested module; "
                    f"skipping it in favour of its parent.")
            else:
                if isinstance(v, dict):
                    v = AcceleratedConfig(**v)
                elif not isinstance(v, AcceleratedConfig):
                    raise TypeError(
                        f"Config for module '{k}' must be AcceleratedConfig or dict, "
                        f"got {type(v).__name__}"
                    )

                selected[k] = v
        return selected

    @staticmethod
    def _child_module_names(specs: dict[str, ModuleSpec]) -> set[str]:
        """Names in ``specs`` whose module is nested inside another spec's.

        Each spec's :attr:`ModuleSpec.getter` resolves to a module path (e.g.
        ``sample_diffusion.diffusion_module``). A name is a *child* of another
        when the other's path is a strict prefix of its own — accelerating both
        a module and its submodule is contradictory. Specs whose getter isn't a
        plain attribute chain (path ``None``) are never a child nor a parent.
        """
        paths = {name: _module_path(spec) for name, spec in specs.items()}
        children: set[str] = set()
        for name, path in paths.items():
            if path is None:
                continue
            for other_name, other_path in paths.items():
                if other_name == name or not other_path:
                    continue
                is_child = (len(other_path) < len(path)
                            and path[:len(other_path)] == other_path)
                if is_child:
                    children.add(name)
                    break
        return children

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
        per-backend wrapper classes (``compiled_cls`` and/or ``graph_optimization_cls``).
        """
        raise NotImplementedError("Subclass must implement this method")


class _ModulePathTracer:
    """Records the attribute chain a :attr:`ModuleSpec.getter` walks.

    Passed as the fake ``model`` to a getter: every attribute access returns a
    child tracer carrying the accumulated path, so ``lambda mod: mod.a.b.c``
    yields the path ``("a", "b", "c")``. Used to detect when one module spec
    resolves to a submodule of another (a strict prefix relationship).
    """

    def __init__(self, path: tuple[str, ...] = ()):
        # Stored in __dict__ directly so it doesn't route through __getattr__.
        self.__dict__["_path"] = path

    def __getattr__(self, name: str) -> "_ModulePathTracer":
        return _ModulePathTracer(self._path + (name, ))


def _module_path(spec: ModuleSpec) -> Optional[tuple[str, ...]]:
    """Return the attribute path ``spec.getter`` walks, or ``None``.

    ``None`` means the getter is not a plain attribute chain (it indexes,
    calls, etc.), so its nesting relative to other specs can't be determined.
    """
    try:
        traced = spec.getter(_ModulePathTracer())
    except Exception:
        return None
    return getattr(traced, "_path", None) or None


class DiscoveredModuleRegistry(ModuleRegistry):
    """A :class:`ModuleRegistry` built by discovery instead of hand-written specs.

    Replaces the per-model registry subclasses (item 1.2.1): every module in the
    model decorated with ``@support_graph_optimization`` becomes a candidate,
    keyed by its **qualified module path** (item 1.2.3, the canonical identity).
    Generic ``get_submodule`` / ``set_submodule`` replace the handwritten
    getter/setter lambdas. Config keys may be a qualified path directly, or one
    of the optional ``role_aliases`` (e.g. ``"structure_pairformer"``) mapping a
    friendly name to a path.

    Args:
        model: The model to discover decorated submodules on.
        configs: ``{key: AcceleratedConfig}`` requested for acceleration.
        role_aliases: ``{role_name: qualified_path}`` friendly-name aliases.
        graph_optimization_cls: Tracker class installed for a graph-optimized
            module (passed in so this module need not import it).
    """

    def __init__(
        self,
        model: nn.Module,
        configs: dict[str, "AcceleratedConfig | dict"] = {},
        role_aliases: Optional[dict[str, str]] = None,
        graph_optimization_cls: Optional[type] = None,
    ):
        self._model = model
        self._role_aliases = dict(role_aliases or {})
        self._graph_optimization_cls = graph_optimization_cls
        # name (path or alias) -> qualified path, filled by get_accelerated_modules.
        self._name_to_path: dict[str, str] = {}
        super().__init__(configs)

    @staticmethod
    def _is_decorated(module: nn.Module) -> bool:
        return getattr(module, GRAPH_OPT_DEFAULT_ATTR, None) is not None

    def _make_spec(self, path: str) -> ModuleSpec:
        return ModuleSpec(
            getter=lambda m, _p=path: m.get_submodule(_p),
            setter=lambda m, opt, _p=path: m.set_submodule(_p, opt),
            graph_optimization_cls=self._graph_optimization_cls,
        )

    def _select_module_configs(self, all_known, configs):
        """Reject a configured key that resolves to no decorated module (1.2.5).

        A requested key must be either a discovered qualified path or a known
        role alias. Anything else is a configuration error (e.g. a typo or a
        module that isn't decorated / present in this model) and is raised rather
        than silently skipped — a missing target must never be treated as
        "nothing to accelerate here".

        When the model declares ``role_aliases``, that map additionally acts as
        a **whitelist**: a module may be graph-optimized only if its qualified
        path is one of the alias *values* (the approved paths). A key is accepted
        either as an alias name or as its resolved path — both land on an
        approved value-path — while any other decorated-but-discoverable path
        (e.g. Protenix's known-NaN ``trunk.pairformer_stack``) is refused. Models
        with no aliases (e.g. OpenFold2) keep the unrestricted, path-based
        behaviour.
        """
        unknown = [k for k in configs if k not in all_known]
        if unknown:
            raise ValueError(
                "graph optimization requested for module(s) with no decorated, "
                f"discoverable target: {sorted(unknown)}. Known roles/paths: "
                f"{sorted(all_known)}")
        if self._role_aliases:
            approved = set(self._role_aliases.values())
            blocked = [
                k for k in configs
                if self._name_to_path.get(k) not in approved
            ]
            if blocked:
                raise ValueError(
                    "graph optimization requested for non-whitelisted "
                    f"module(s): {sorted(blocked)}. A module may be cuda-graphed "
                    "only if its path is a value in GRAPH_OPT_ENABLED_MODULES. "
                    f"Approved paths: {sorted(approved)}")
        return super()._select_module_configs(all_known, configs)

    def get_accelerated_modules(self) -> dict[str, ModuleSpec]:
        """Discover decorated submodules, keyed by qualified path (plus aliases).

        Uses ``remove_duplicate=False`` so a module reachable at several paths
        (e.g. OpenFold3's ``diffusion_module`` also lives at
        ``sample_diffusion.diffusion_module``) is discovered at *every* path —
        the wrapper must be installed on the exact path the forward calls, which
        is the one the role alias names.
        """
        specs: dict[str, ModuleSpec] = {}
        self._name_to_path = {}
        for path, module in self._model.named_modules(remove_duplicate=False):
            if path and self._is_decorated(module):
                specs[path] = self._make_spec(path)
                self._name_to_path[path] = path
        # Friendly role aliases resolve to their target path (authoritative:
        # it is the forward-reachable path from the model's own alias map).
        for role, path in self._role_aliases.items():
            try:
                target = self._model.get_submodule(path)
            except AttributeError:
                continue
            if self._is_decorated(target):
                specs[role] = self._make_spec(path)
                self._name_to_path[role] = path
        return specs

    def _child_module_names(self, specs: dict[str, ModuleSpec]) -> set[str]:
        """Detect conflicts directly from qualified paths (item 1.2.2), using the
        discovered ``name -> path`` map. A name is dropped when it is a strict
        *descendant* of another configured path, or when it resolves to the
        *same* path as an already-kept name (e.g. a role alias and its canonical
        path both configured) — so the same submodule is never wrapped twice."""
        paths = {name: self._name_to_path.get(name) for name in specs}
        children: set[str] = set()
        seen_paths: set[str] = set()
        for name, path in paths.items():
            if not path:
                continue
            # Same-path duplicate: an already-kept name targets this exact path.
            if path in seen_paths:
                children.add(name)
                continue
            parts = path.split(".")
            is_child = False
            for other_name, other_path in paths.items():
                if other_name == name or not other_path:
                    continue
                other_parts = other_path.split(".")
                if (len(other_parts) < len(parts)
                        and parts[:len(other_parts)] == other_parts):
                    is_child = True
                    break
            if is_child:
                children.add(name)
            else:
                seen_paths.add(path)
        return children


class OptimizedModuleSetterMixin(ABC):

    @abstractmethod
    def get_optimized_modules(
            self,
            accelerated_configs: dict[str,
                                      AcceleratedConfig]) -> ModuleRegistry:
        raise NotImplementedError("Subclass must implement this method")

    def optimize(self,
                 accelerated_configs: dict[str, AcceleratedConfig],
                 **kwargs) -> nn.Module:
        """Build the optimized version of the model from the original.

        Supports torch backends (optionally with ``torch.compile`` when
        ``compile=True``). For the CUDA-graph path, each matched module's tracker
        is configured from ``acc_config.default.graph_optimization_config`` when
        the model provides one, else from the module's decorator-declared
        ``graph_opt_default`` (set by ``@support_graph_optimization``).

        Args:
            accelerated_configs: A dictionary of modules to be accelerated.
                Each key is a module name (e.g. ``"evoformer"``) and the value
                is an :class:`AcceleratedConfig` whose ``backend`` field
                selects ``"torch"``.  Set ``compile=True`` on torch-backend
                configs to enable ``torch.compile``.
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

            # ── torch + CUDA-graph path ─────────────────────────────────
            # The eager module is replaced by a graph-compilation tracker that
            # keeps the original as its eager ``inner_module`` / fallback.
            if backend == BackendType.TORCH and spec.graph_optimization_cls is not None:
                org = spec.getter(self)
                graph_config = getattr(acc_config.default,
                                        "graph_optimization_config",
                                        None) if acc_config.default else None
                # Fall back to the module's decorator-declared default
                # (``graph_opt_default``, set by @support_graph_optimization) when
                # the model config supplies no explicit graph-optimization config,
                # so the decorator's declared defaults (mode / input-key method /
                # verify / routing) are what the tracker actually runs with.
                if graph_config is None:
                    graph_config = getattr(org, GRAPH_OPT_DEFAULT_ATTR, None)

                if graph_config is not None:
                    graph_mode = getattr(graph_config, "graph_optimization_mode", None)
                    if graph_mode != GraphOptimizationMode.NO_OPTIMIZATION:
                        opt_m = spec.graph_optimization_cls(config=graph_config,
                                                        inner_module=org)
                        spec.setter(self, opt_m)

        return self
