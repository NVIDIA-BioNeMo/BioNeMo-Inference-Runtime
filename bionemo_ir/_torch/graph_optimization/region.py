# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Model-owned CUDA graph regions: capture a repeated computation without replacing its modules."""

from __future__ import annotations

import inspect
import re
import weakref
from collections.abc import Callable, Hashable
from typing import Any

import torch
from lru import LRU
from torch import nn

from bionemo_ir._torch.graph_optimization.config import (
    CUDAGraphOptimizationConfig,
    GraphOptimizationMode,
    InputRoutingConfig,
    acceptance_max_by_name,
)
from bionemo_ir._torch.graph_optimization.cuda_graph.memory import container_device
from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import CUDAGraphOptimizationTracker, graph_eligible

# A positional input routes by its parameter's name; one past the named parameters, which only
# ``*args`` takes, routes as ``arg{i}`` (``GraphOptimizationTracker._positional_param_names``).
_POSITIONAL_NAME = re.compile(r"^arg(\d+)$")


def _validate_routing_against_signature(sig: inspect.Signature, routing: InputRoutingConfig | None, owner: str) -> None:
    """Reject routed names that reach no input of ``sig``: a parameter, ``**kwargs`` or an ``*args`` position."""
    param_names = set(sig.parameters)
    kinds = [p.kind for p in sig.parameters.values()]
    has_var_keyword = inspect.Parameter.VAR_KEYWORD in kinds
    has_var_positional = inspect.Parameter.VAR_POSITIONAL in kinds
    num_named_positional = sum(
        kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) for kind in kinds
    )

    def _check(name: str, kind: str) -> None:
        if name in param_names or has_var_keyword:
            return
        positional = _POSITIONAL_NAME.match(name)
        if positional is not None and has_var_positional and int(positional.group(1)) >= num_named_positional:
            return
        raise ValueError(
            f"{owner}: {kind} {name!r} is not a parameter of forward {tuple(n for n in param_names if n != 'self')}"
        )

    named_dim_ties = routing.named_dim_ties if routing is not None else ()
    workspace_kwargs = routing.internal_workspace_kwargs if routing is not None else ()
    stable_kwargs = routing.stable_input_kwargs if routing is not None else ()
    state_kwargs = routing.state_input_kwargs if routing is not None else ()
    borrowed_kwargs = routing.borrowed_input_kwargs if routing is not None else ()
    static_args = routing.static_args if routing is not None else ()
    for dim in named_dim_ties:
        for tensor_name, _axes in dim.input_dims:
            _check(tensor_name, "input-tie tensor")
    for name in workspace_kwargs:
        _check(name, "workspace kwarg")
    for name in stable_kwargs:
        _check(name, "stable kwarg")
    for name in state_kwargs:
        _check(name, "state kwarg")
    for name in borrowed_kwargs:
        _check(name, "borrowed kwarg")
    for name in static_args:
        _check(name, "static arg")


class _OwnerCall(nn.Module):
    """Call ``owner.<name>`` for the tracker without registering the owner's weights here."""

    def __init__(self, owner: nn.Module, name: str) -> None:
        super().__init__()
        if not callable(getattr(owner, name, None)):
            raise AttributeError(f"{type(owner).__name__} has no callable {name!r} for its graph region to run")
        self._owner = weakref.ref(owner)
        self._name = name

    def target(self) -> Any:
        owner = self._owner()
        if owner is None:
            raise RuntimeError("The graph region's owner no longer exists")
        return getattr(owner, self._name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.target()(*args, **kwargs)

    def __getstate__(self) -> dict[str, Any]:
        # Copy or pickle the owner itself, so a deep-copied model's region calls the copy.
        return {**self.__dict__, "_owner": self._owner()}

    def __setstate__(self, state: dict[str, Any]) -> None:
        owner = state.pop("_owner")
        super().__setstate__(state)
        self._owner = weakref.ref(owner)


class GraphRegion(nn.Module):
    """Capture and replay one repeated computation of its owner.

    The owner creates the region with the name of the method or submodule it repeats and calls the
    region in its place. A region with a policy captures by default; it runs eagerly while
    ``enabled`` is false, and also whenever gradients are on, the inputs are not on CUDA, or an
    enclosing :func:`eager_graphs` scope or graph capture owns the call. The computation's modules
    keep their places and state-dict paths; loading weights or moving the model drops the captured
    graphs.

    Args:
        owner: Module that owns the computation.
        name: Attribute of ``owner`` to call: a method or a submodule.
        policy: CUDA-graph policy, usually the owner config's ``graph_optimization_config``.
            Without one the region always runs eagerly.
    """

    def __init__(self, owner: nn.Module, name: str, policy: CUDAGraphOptimizationConfig | None) -> None:
        super().__init__()
        self.enabled = policy is not None
        self.policy = policy
        self.call = _OwnerCall(owner, name)
        self.tracker: CUDAGraphOptimizationTracker | None = None
        # Caller-owned buffers for borrowed inputs, per caller key, most recently used last.
        self._borrowed: LRU | None = None

    def enable(self, policy: CUDAGraphOptimizationConfig | None = None) -> None:
        """Capture from the next eligible call, under ``policy`` when given."""
        if policy is not None and policy is not self.policy:
            self.policy = policy
            self.reset()
        if self.policy is None:
            raise ValueError("This graph region has no CUDA-graph policy")
        self.enabled = True

    def reset(self) -> None:
        """Release the captured graphs and the borrowed buffers, and forget the signatures that fell back to eager."""
        if self.tracker is not None:
            self.tracker.reset()
        self.tracker = None
        self._borrowed = None

    def captures(self, **dims: int) -> bool:
        """Whether a call made now would take the graph path, given its named dimensions such as ``num_tokens``.

        The region must be enabled under an optimizing policy, outside gradients, training, an ``eager_graphs`` scope
        and any capture, and each named dimension within the policy's acceptance limit.
        """
        policy = self.policy
        if (
            not self.enabled
            or policy is None
            or policy.graph_optimization_mode == GraphOptimizationMode.NO_OPTIMIZATION
        ):
            return False
        if not graph_eligible(self.training):
            return False
        routing = policy.input_routing_config
        limits = acceptance_max_by_name(routing) if routing is not None else {}
        return all(limits.get(name) is None or value <= limits[name] for name, value in dims.items())

    def borrowed_buffers(self, key: Hashable, make: Callable[[], tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
        """The caller-owned buffers for ``key``, made by ``make`` on first use, to pass as borrowed inputs.

        The caller writes them in place between calls, and a captured graph reads them where they lie. The region
        keeps those of as many keys as it keeps graphs, the most recently used, and drops them all on
        :meth:`reset`.
        """
        capacity = self.policy.num_graphs_max_for_this_module
        if self._borrowed is None:
            self._borrowed = LRU(capacity)
        elif self._borrowed.get_size() != capacity:
            self._borrowed.set_size(capacity)
        buffers = self._borrowed.get(key)
        if buffers is None:
            buffers = make()
            self._borrowed[key] = buffers
        return buffers

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        policy = self.policy
        if (
            not self.enabled
            or policy is None
            or policy.graph_optimization_mode == GraphOptimizationMode.NO_OPTIMIZATION
        ):
            return self.call(*args, **kwargs)
        device = container_device((args, kwargs))
        if device is None or device.type != "cuda":
            return self.call(*args, **kwargs)
        if self.tracker is None:
            target = self.call.target()
            signature = inspect.signature(target.forward if isinstance(target, nn.Module) else target)
            _validate_routing_against_signature(signature, policy.input_routing_config, "GraphRegion")
            tracker = CUDAGraphOptimizationTracker(config=policy, inner_module=self.call)
            tracker._forward_positional_names = [
                parameter.name
                for parameter in signature.parameters.values()
                if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            tracker.train(self.training)
            self.tracker = tracker
        return self.tracker(*args, **kwargs)
