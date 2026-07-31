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
"""``support_graph_optimization`` class decorator.

The decorator declares a module class's default graph-optimization config on the
class itself, assembled from:

* **Signature-derived tie points** (:class:`NamedDimTies`) — which ``forward``
  input/output axes carry a named dimension (e.g. ``num_tokens``), which kwargs
  are graph-internal workspaces, and which args are static. These depend only on
  the module's ``forward`` signature, so they are identical for every instance.
* **Per-dim input management** — an optional :class:`InputAcceptanceDimSpec`
  (acceptance limit) and/or :class:`PaddedDimSpec` (shape-bucket / padding) the
  tracker applies by default.

These are packaged into a single :class:`GraphOptimizationConfig` stored on the
class as ``graph_opt_default`` (:data:`GRAPH_OPT_DEFAULT_ATTR`). A model-side
config may read from it (typically its ``input_routing_config``); otherwise the
optimize-module setter uses ``graph_opt_default`` directly as the tracker's
config whenever the model supplies no explicit ``graph_optimization_config``.

Only spec declaration lives here. Discovery over ``model.named_modules()`` lives
in ``models/optimize_module_setter.py`` (``DiscoveredModuleRegistry``). The
tracker does its own positional-to-parameter-name normalization for a live call
(``_positional_param_names``); ``bind_forward_args`` below is a standalone
utility providing that same ``inspect.Signature``-based normalization.
"""
from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Dict, Optional, Sequence, Type, Union

from tensorrt_bionemo._torch.graph_optimization.config import (  # noqa: F401
    CUDAGraphOptimizationConfig, GraphOptimizationConfig, GraphOptimizationMode,
    InputAcceptanceDimSpec, InputKeyMethod, InputRoutingConfig, NamedDimTies,
    PaddedDimSpec)

# Attribute the decorator stashes its ``GraphOptimizationConfig`` default under
# on the decorated class.
GRAPH_OPT_DEFAULT_ATTR = "graph_opt_default"

# A tie-point tensor named ``arg{i}`` references a positional argument by index
# rather than by parameter name, so it cannot be validated against the forward
# signature by name (and ``Signature.bind`` is what removes the need for it).
_POSITIONAL_NAME = re.compile(r"^arg\d+$")


def support_graph_optimization(
    *,
    named_dims: Sequence[NamedDimTies],
    graph_optimization_mode: GraphOptimizationMode,
    input_key_method: InputKeyMethod,
    static_args: Sequence[str] = (),
    workspace_kwargs: Sequence[str] = (),
    verify_capture: bool = False,
    input_acceptance_dim_spec: Optional[InputAcceptanceDimSpec] = None,
    padded_dim_spec: Optional[PaddedDimSpec] = None,
):
    """Attach a default graph-optimization config to a module class.

    Assembles the arguments into a single :class:`GraphOptimizationConfig` — a
    :class:`CUDAGraphOptimizationConfig` for ``CUDA_GRAPH_VIA_TORCH`` or a base
    :class:`GraphOptimizationConfig` for ``NO_OPTIMIZATION`` — and stores it on
    the class as :data:`GRAPH_OPT_DEFAULT_ATTR` (``graph_opt_default``). The tie
    points and the two per-dim specs become the config's
    :class:`InputRoutingConfig`.

    Args:
        named_dims: signature-derived tie points — which ``forward`` input/output
            axes carry each named dimension. Identical for every instance of the
            class. Become the routing config's ``named_dim_ties``.
        graph_optimization_mode: backend to apply (see
            :class:`GraphOptimizationMode`).
        input_key_method: how a call's input is reduced to a graph-cache key
            (see :class:`InputKeyMethod`).
        static_args: names of ``forward`` args declared static for a given input
            key (so they need not participate in it). Signature-derived.
        workspace_kwargs: names of ``forward`` kwargs carrying graph-internal
            scratch (excluded from input-key derivation). Signature-derived.
        verify_capture: when True, each captured graph is replayed and compared
            against a fresh eager run at capture time (reverting a mismatching key
            permanently to eager).
        input_acceptance_dim_spec: optional acceptance limit for a named
            dimension; ``None`` accepts all lengths for it.
        padded_dim_spec: optional shape-bucket (padding) spec for a named
            dimension; ``None`` disables padding for it.

    Raises:
        ValueError: if ``graph_optimization_mode`` is neither
            ``CUDA_GRAPH_VIA_TORCH`` nor ``NO_OPTIMIZATION``.
    """
    input_routing_config = InputRoutingConfig(
        named_dim_ties=list(named_dims),
        padded_dims=[padded_dim_spec] if padded_dim_spec is not None else [],
        input_acceptance_dims=(
            [input_acceptance_dim_spec]
            if input_acceptance_dim_spec is not None else []),
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
            f"{GraphOptimizationMode.NO_OPTIMIZATION}")

    def _decorate(cls: Type) -> Type:
        setattr(cls, GRAPH_OPT_DEFAULT_ATTR, graph_opt_default)
        # (1.1.8) Fail loudly at decoration time if a declared name is not a
        # real forward parameter — a typo'd tie point is a config bug, not a
        # silently-ignored no-op.
        validate_spec_against_forward(cls)
        return cls

    return _decorate


def validate_spec_against_forward(cls_or_instance: Union[Type, object]) -> None:
    """Validate declared names against ``forward``'s signature (item 1.1.8).

    Every input-tie tensor name (except positional ``arg{i}`` references),
    workspace kwarg, and static arg must be a real parameter of ``forward`` —
    unless ``forward`` declares ``**kwargs``, which can absorb it. Raises
    :class:`ValueError` naming the offending declaration otherwise.
    """
    config = getattr(cls_or_instance, GRAPH_OPT_DEFAULT_ATTR, None)
    if not isinstance(config, GraphOptimizationConfig):
        target = getattr(cls_or_instance, "__name__",
                         type(cls_or_instance).__name__)
        raise ValueError(
            f"{target!r} is not decorated with @support_graph_optimization "
            "(no graph_opt_default found)")
    routing = config.input_routing_config
    forward = getattr(cls_or_instance, "forward")
    sig = inspect.signature(forward)
    param_names = set(sig.parameters)
    has_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values())
    owner = getattr(cls_or_instance, "__name__",
                    type(cls_or_instance).__name__)

    def _check(name: str, kind: str) -> None:
        if _POSITIONAL_NAME.match(name):
            return
        if name in param_names or has_var_keyword:
            return
        raise ValueError(
            f"@support_graph_optimization on {owner}: {kind} {name!r} is not a "
            f"parameter of forward "
            f"{tuple(n for n in param_names if n != 'self')}")

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
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize a live call to ``{parameter_name: value}`` (item 1.1.8).

    Uses :meth:`inspect.Signature.bind_partial` to map positional arguments onto
    their parameter names, so routing keyed by name is independent of whether a
    caller passed an argument positionally or by keyword (e.g. boltz-2 passes the
    pairformer's ``s``/``z`` positionally where OpenFold3 passes them by keyword).
    Only what the caller actually passed is returned — defaults are not applied.
    ``*args`` overflow is exposed as ``arg{i}`` keys and ``**kwargs`` overflow is
    flattened in, matching the tracker's naming.
    """
    sig = inspect.signature(forward)
    bound = sig.bind_partial(*args, **kwargs)
    normalized: Dict[str, Any] = {}
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
