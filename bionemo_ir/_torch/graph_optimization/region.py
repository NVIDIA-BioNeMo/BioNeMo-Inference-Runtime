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
import weakref
from typing import Any

from torch import nn

from bionemo_ir._torch.graph_optimization.config import CUDAGraphOptimizationConfig, GraphOptimizationMode
from bionemo_ir._torch.graph_optimization.cuda_graph.memory import container_device
from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import CUDAGraphOptimizationTracker
from bionemo_ir._torch.graph_optimization.decorator import _validate_routing_against_signature


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

    def enable(self, policy: CUDAGraphOptimizationConfig | None = None) -> None:
        """Capture from the next eligible call, under ``policy`` when given."""
        if policy is not None and policy is not self.policy:
            self.policy = policy
            self.reset()
        if self.policy is None:
            raise ValueError("This graph region has no CUDA-graph policy")
        self.enabled = True

    def reset(self) -> None:
        """Release the captured graphs and forget the signatures that fell back to eager."""
        if self.tracker is not None:
            self.tracker.reset()
        self.tracker = None

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
