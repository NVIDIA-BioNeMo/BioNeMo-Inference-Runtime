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
"""Graph-optimization declarations and forward-call normalization.

The decorator stores a class-level default config; runtime discovery and
tracking live in the model setter and tracker modules.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Sequence
from typing import Any

from tensorrt_bionemo._torch.graph_optimization.config import (  # noqa: F401
    CUDAGraphOptimizationConfig,
    GraphOptimizationConfig,
    GraphOptimizationMode,
    InputAcceptanceDimSpec,
    InputKeyMethod,
    InputRoutingConfig,
    NamedDimTies,
    PaddedDimSpec,
)

GRAPH_OPT_DEFAULT_ATTR = "graph_opt_default"

# Positional ``arg{i}`` ties cannot be validated by parameter name.
_POSITIONAL_NAME = re.compile(r"^arg\d+$")


def support_graph_optimization(
    *,
    named_dims: Sequence[NamedDimTies],
    graph_optimization_mode: GraphOptimizationMode,
    input_key_method: InputKeyMethod,
    static_args: Sequence[str] = (),
    workspace_kwargs: Sequence[str] = (),
    verify_capture: bool = False,
    input_acceptance_dim_spec: InputAcceptanceDimSpec | None = None,
    padded_dim_spec: PaddedDimSpec | None = None,
):
    """Attach a default :class:`GraphOptimizationConfig` to a module class.

    Args:
        named_dims: Named input/output dimension ties.
        graph_optimization_mode: Backend to apply.
        input_key_method: Graph-cache key strategy.
        static_args: ``forward`` arguments omitted from dynamic keys.
        workspace_kwargs: Graph-owned scratch kwargs omitted from keys and copies.
        verify_capture: Compare a new capture with eager execution.
        input_acceptance_dim_spec: Optional input-length limit.
        padded_dim_spec: Optional shape-bucketing rule.

    Raises:
        ValueError: If the optimization mode is unsupported.
    """
    input_routing_config = InputRoutingConfig(
        named_dim_ties=list(named_dims),
        padded_dims=[padded_dim_spec] if padded_dim_spec is not None else [],
        input_acceptance_dims=([input_acceptance_dim_spec] if input_acceptance_dim_spec is not None else []),
        static_args=list(static_args),
        internal_workspace_kwargs=list(workspace_kwargs),
    )
    if graph_optimization_mode == GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH:
        graph_opt_default = CUDAGraphOptimizationConfig(
            graph_optimization_mode=graph_optimization_mode,
            input_key_method=input_key_method,
            input_routing_config=input_routing_config,
            verify_capture=verify_capture,
        )
    elif graph_optimization_mode == GraphOptimizationMode.NO_OPTIMIZATION:
        graph_opt_default = GraphOptimizationConfig(
            graph_optimization_mode=graph_optimization_mode,
            input_key_method=input_key_method,
            input_routing_config=input_routing_config,
        )
    else:
        raise ValueError(
            f"unsupported graph_optimization_mode {graph_optimization_mode!r}; "
            f"expected {GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH} or "
            f"{GraphOptimizationMode.NO_OPTIMIZATION}"
        )

    def _decorate(cls: type) -> type:
        setattr(cls, GRAPH_OPT_DEFAULT_ATTR, graph_opt_default)
        # Reject misspelled routing names at decoration time.
        validate_spec_against_forward(cls)
        return cls

    return _decorate


def validate_spec_against_forward(cls_or_instance: type | object) -> None:
    """Validate routed names against ``forward`` or its ``**kwargs``."""
    config = getattr(cls_or_instance, GRAPH_OPT_DEFAULT_ATTR, None)
    if not isinstance(config, GraphOptimizationConfig):
        target = getattr(cls_or_instance, "__name__", type(cls_or_instance).__name__)
        raise ValueError(f"{target!r} is not decorated with @support_graph_optimization (no graph_opt_default found)")
    routing = config.input_routing_config
    forward = cls_or_instance.forward
    sig = inspect.signature(forward)
    param_names = set(sig.parameters)
    has_var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    owner = getattr(cls_or_instance, "__name__", type(cls_or_instance).__name__)

    def _check(name: str, kind: str) -> None:
        if _POSITIONAL_NAME.match(name):
            return
        if name in param_names or has_var_keyword:
            return
        raise ValueError(
            f"@support_graph_optimization on {owner}: {kind} {name!r} is not a "
            f"parameter of forward "
            f"{tuple(n for n in param_names if n != 'self')}"
        )

    named_dim_ties = routing.named_dim_ties if routing is not None else ()
    workspace_kwargs = routing.internal_workspace_kwargs if routing is not None else ()
    static_args = routing.static_args if routing is not None else ()
    for dim in named_dim_ties:
        for tensor_name, _axes in dim.input_dims:
            _check(tensor_name, "input-tie tensor")
    for name in workspace_kwargs:
        _check(name, "workspace kwarg")
    for name in static_args:
        _check(name, "static arg")


def bind_forward_args(
    forward: Callable,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Map passed arguments to parameter names without applying defaults.

    Variadic positional values use indexed names; ``**kwargs`` values are
    flattened into the result.
    """
    sig = inspect.signature(forward)
    bound = sig.bind_partial(*args, **kwargs)
    normalized: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        kind = sig.parameters[name].kind
        if kind is inspect.Parameter.VAR_POSITIONAL:
            for i, item in enumerate(value):
                normalized[f"{name}{i}"] = item
        elif kind is inspect.Parameter.VAR_KEYWORD:
            normalized.update(value)
        else:
            normalized[name] = value
    return normalized
