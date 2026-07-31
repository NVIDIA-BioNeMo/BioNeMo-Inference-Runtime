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
"""Input-key derivation and shape-bucket padding shared by graph trackers.

:class:`GraphOptimizationTracker` is the ``BackendBase`` subclass that wraps an
eager module and derives a per-call *input key* from the input tensor shapes,
walking nested dict/list/tuple containers. It also provides the shape-bucket
``pad_input`` / ``unpad_output`` helpers used when the module runs at bucketed
(padded) shapes so one captured graph can serve a range of live shapes. The concrete CUDA-graph warmup/capture/replay tracker
(:class:`CUDAGraphOptimizationTracker`) and its per-key state live in
:mod:`tensorrt_bionemo._torch.graph_optimization.cuda_graph.runtime`.
"""
import abc
import inspect
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from tensorrt_bionemo.runtime.backend import BackendBase
from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.graph_optimization.config import (
    GraphOptimizationConfig, acceptance_max_by_name, bucket_lengths_by_name,
    input_acceptance_assignments, input_padded_assignments,
    output_padded_assignments)


# (Req 4.1) A ``TensorContainer`` is what the wrapped module is called with: a tensor,
# or a tuple/dict nesting tensors, non-tensor leaves, and further containers.  
# Contains at least one Tensor
TensorContainer = Tensor | tuple | dict

# (Req 4.2) Maps a tensor's dotted path within a container to a 1-D int tensor of
# its dim lengths, so a dim length is a direct index (no per-call shape logic).
TensorContainerShapes = dict[str, Tensor]


class GraphOptimizationTracker(BackendBase):
    """Module wrapper that derives input keys and tracks graph state.

    Inherits from :class:`tensorrt_bionemo.runtime.backend.BackendBase` (req 2.1)
    so the eager-fallback plumbing (``set_fallback_module`` / ``_fallback_module``)
    and the backend config surface are in place for memory management and the
    revert-to-eager strategy. Provides input-key derivation shared by concrete
    trackers: a key is the join of a shape-encoded subkey per input tensor,
    walked recursively through nested dicts/lists/tuples. The separators below
    build that string; concrete subclasses implement the state-management policy.

    Class attributes:
        SEP_FOR_ARGS: Separator joining the per-tensor subkeys into one key.
        SEP_BW_NAME_AND_SHAPE: Separator between a tensor's path and its shape.
        SEP_FOR_DIMS: Separator between the dims within a tensor's shape.
        GRAPH_INTERNAL_WORKSPACE_KWARGS: Names of keyword arguments that are
            graph-*internal scratch* rather than real inputs (see below).
    """
    SEP_FOR_ARGS = "|"
    SEP_BW_NAME_AND_SHAPE = "+"
    SEP_FOR_DIMS = "-"
    SEP_BW_ARG_AND_DIM = "."
    SEP_BW_ARG_AND_TYPE = ","

    # Keyword arguments that carry a graph-internal *workspace*, not stable inputs.
    #
    # The Boltz-2 / OpenFold3 score model threads a single ``buffers``
    # (``PreallocatedBuffers``) dict through the atom encoder -> token
    # transformer -> atom decoder. ``ensure_buffer`` lazily (re)allocates entries
    # in it *as a side effect* of each forward, and the encoder and token
    # transformer reuse the same buffer names (e.g. ``pw_attn_output``) at
    # different shapes. So at the wrapped module's entry the dict holds the
    # caller's (encoder-shaped) scratch, but the captured forward mutates it to
    # the module's own (token-shaped) scratch.
    #
    # That makes ``buffers`` unusable under the "clone the kwargs at capture,
    # copy the live kwargs into the static buffers on every replay" contract:
    # the live dict never matches the captured one (different shapes, or
    # different keys), which previously crashed in ``_copy_tensors_into``.
    #
    # These kwargs are therefore excluded from the input key *and* from the
    # per-replay static-buffer copy: the captured graph allocates and reuses this
    # scratch internally (in the graph mempool), so there is nothing to key on or
    # copy in. They are still passed through verbatim during warmup/capture so
    # the wrapped module runs exactly as it does eagerly.
    GRAPH_INTERNAL_WORKSPACE_KWARGS: frozenset = frozenset({"buffers"})

    def _effective_workspace_kwargs(self) -> frozenset:
        """Workspace kwargs to exclude from graph inputs.

        Prefers the per-module ``internal_workspace_kwargs`` declared on the
        :class:`InputRoutingConfig` (populated from the module's
        ``@support_graph_optimization`` decorator); falls back to the built-in
        :attr:`GRAPH_INTERNAL_WORKSPACE_KWARGS` when the config declares none, so
        configs predating the decorator still exclude ``buffers``.
        """
        cfg = self.graph_optimization_config.input_routing_config
        if cfg is not None and cfg.internal_workspace_kwargs:
            return frozenset(cfg.internal_workspace_kwargs)
        return self.GRAPH_INTERNAL_WORKSPACE_KWARGS

    def _positional_param_names(self, num_positional: int) -> list[str]:
        """Root names for the first ``num_positional`` positional arguments.

        (1.1.8) Maps each positional argument to the corresponding ``forward``
        parameter name (via :func:`inspect.signature`), so routing is
        independent of whether a caller passed an argument positionally or by
        keyword — e.g. boltz-2 passes the pairformer's ``s``/``z`` positionally
        where OpenFold3 passes them by keyword, yet both key on ``s``/``z``.
        Falls back to ``arg{i}`` when the signature is unavailable or a position
        overflows the declared positional parameters (``*args``). Cached.
        """
        names = self._forward_positional_names
        if names is None:
            names = []
            inner = self.inner_module
            if inner is not None:
                try:
                    sig = inspect.signature(inner.forward)
                    names = [
                        p.name for p in sig.parameters.values()
                        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    ]
                except (TypeError, ValueError):
                    names = []
            self._forward_positional_names = names
        return [names[i] if i < len(names) else f"arg{i}"
                for i in range(num_positional)]

    def _graph_input_kwargs(self, kwargs: dict) -> dict:
        """Return ``kwargs`` without the graph-internal workspace entries.

        Used for both input-key derivation and the per-replay copy so that
        side-effect-populated workspace dicts (see
        :meth:`_effective_workspace_kwargs`) are never treated as graph inputs.
        """
        workspace = self._effective_workspace_kwargs()
        if not workspace:
            return kwargs
        return {
            k: v
            for k, v in kwargs.items()
            if k not in workspace
        }

    def __init__(self,
                 graph_optimization_config: GraphOptimizationConfig,
                 inner_module: nn.Module | None = None) -> None:
        """Store the optimization config and the wrapped (inner) module.

        Args:
            graph_optimization_config: The graph-optimization configuration. Also used as this
                backend's ``config`` (it is a ``BaseConfig`` subclass), so the
                wrapper exposes ``config.backend`` etc.
            inner_module: The eager module being graph-optimized. This is the
                module the tracker warms up, captures, and falls back to.
        """
        super().__init__(graph_optimization_config)
        self.inner_module = inner_module
        # (1.2.4) One-shot guard: input tie points are validated against the
        # first representative call, then never again (hot path).
        self._input_ties_validated = False
        # (1.1.8) Cached positional-parameter names of the wrapped forward, used
        # to normalize positional args to their parameter names (lazy).
        self._forward_positional_names: list[str] | None = None

    @property
    def graph_optimization_config(self) -> GraphOptimizationConfig:
        return self._config
    
    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped eager ``inner_module``.

        The wrapper replaces the eager module in the model tree, so callers that
        read the original module's attributes (e.g. ``dtype``, ``version``) must
        still find them. ``nn.Module.__getattr__`` is tried first (parameters /
        buffers / submodules, including ``inner_module`` itself); only genuinely
        missing names fall through to the wrapped module.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules")
            inner = modules.get("inner_module") if modules else None
            if inner is not None:
                return getattr(inner, name)
            raise

    def input_accepted(self, tensor_container_shapes: TensorContainerShapes) -> bool:
        """Whether this call's inputs are within the configured acceptance range.

        Reads the rules off the module's :class:`InputRoutingConfig` snapshot:
        for every ``(tensor_name, dim_idx) -> dim_name`` input assignment whose
        ``dim_name`` carries an ``input_acceptance_dims`` limit, the length of
        that tensor axis — looked up in ``tensor_container_shapes`` (the
        ``path -> dim-lengths`` map from :meth:`_extract_tensor_container_shapes`)
        under the key ``f"{tensor_name}_shape"`` — must not exceed that limit (an
        inclusive maximum). Returns ``True`` when all such axes pass (and when no
        config / no rule applies), ``False`` as soon as one exceeds its limit.

        Because the axis is resolved by walk path, a tensor nested inside a
        container input (e.g. a token-axis feature riding inside a ``batch`` dict,
        keyed ``batch.residue_index``) is checked exactly like a top-level input:
        ``_extract_tensor_container_shapes`` records both under their dotted
        paths. Rebuilt from serializable data, so it survives a deserialized
        config with no closure involved.

        Args:
            tensor_container_shapes: The ``path -> dim-lengths`` map for this call
                (from :meth:`_extract_tensor_container_shapes`); acceptance is
                decided purely from it, not from the live tensors.
        """
        cfg = self.graph_optimization_config.input_routing_config
        if cfg is None:
            return True
        acceptance_max = acceptance_max_by_name(cfg)
        for tensor_name, dim_idx, dim_name in input_acceptance_assignments(cfg):
            dim_len_max = acceptance_max.get(dim_name)
            if dim_len_max is None:
                continue  # this dim has no acceptance rule
            shape = tensor_container_shapes.get(f"{tensor_name}_shape")
            if shape is None:
                continue  # tensor absent from this call
            ndim = int(shape.numel())
            if not (-ndim <= dim_idx < ndim):
                continue  # axis not present on this tensor
            if int(shape[dim_idx]) > dim_len_max:
                return False
        return True

    def validate_input_ties(
            self, tensor_container_shapes: TensorContainerShapes) -> None:
        """Validate configured input tie points against the first call (1.2.4).

        Every ``(tensor_name, dim_idx)`` an :class:`InputRoutingConfig` ties to a
        named dimension — for acceptance *and* for padding — must resolve to a
        real input tensor axis on the first representative call. A tie to a
        tensor that isn't present, or to an axis out of that tensor's range, is a
        configuration error (item 1.2.5: "not as acceptance") and is raised —
        rather than being silently ignored, which would let a mis-tied routing
        rule pass unenforced. Runs once (guarded by ``_input_ties_validated``).

        Args:
            tensor_container_shapes: the ``path -> dim-lengths`` map for this call
                (from :meth:`_extract_tensor_container_shapes`).
        """
        if self._input_ties_validated:
            return
        cfg = self.graph_optimization_config.input_routing_config
        if cfg is not None:
            available = sorted(
                name[:-len("_shape")] for name in tensor_container_shapes
                if name.endswith("_shape"))
            ties = (input_acceptance_assignments(cfg)
                    + input_padded_assignments(cfg))
            for tensor_name, dim_idx, dim_name in ties:
                shape = tensor_container_shapes.get(f"{tensor_name}_shape")
                if shape is None:
                    raise ValueError(
                        f"input routing ties dim {dim_name!r} to input "
                        f"{tensor_name!r} (axis {dim_idx}), but no such tensor is "
                        f"present in the call; available inputs: {available}")
                ndim = int(shape.numel())
                if not (-ndim <= dim_idx < ndim):
                    raise ValueError(
                        f"input routing ties dim {dim_name!r} to axis {dim_idx} "
                        f"of input {tensor_name!r}, but that tensor has only "
                        f"{ndim} dimension(s)")
        self._input_ties_validated = True

    def input_key_for_this_call(self, *args, **kwargs) -> str:
        """Return the graph-cache key for this call's positional/keyword inputs.

        (Req 4.6) The key joins the call's *input-metadata* — the tensor
        metadata (path/dtype/shape of every input tensor) and the non-tensor
        metadata (path/type of every non-tensor leaf), from
        :meth:`_extract_container_metadata`. So calls whose tensors share dtype
        and shape and whose non-tensor leaves share type map to one captured
        graph.
        """
        key = self.SEP_FOR_ARGS.join(
            self._extract_input_metadata(*args, **kwargs))
        return key
    
    def _collect_subkeys(self, value: Any, path: str, subkeys: list[str]) -> None:
        """Recursively walk ``value``, appending a shape-encoded subkey to
        ``subkeys`` for every tensor found.

        ``path`` is the dotted name of the current position (e.g.
        ``feature_dict.ref_pos`` or ``coords.0``) so tensors at different
        nesting locations produce distinct keys.
        """
        if isinstance(value, torch.Tensor):
            subkeys.append(
                self.SEP_BW_NAME_AND_SHAPE.join(
                    [path, self.SEP_FOR_DIMS.join([str(dim) for dim in value.shape])]
                )
            )
        elif isinstance(value, dict):
            for k, v in value.items():
                child = f"{path}.{k}" if path else str(k)
                self._collect_subkeys(v, child, subkeys)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                child = f"{path}.{i}" if path else str(i)
                self._collect_subkeys(v, child, subkeys)
        # Non-tensor leaves (None, scalars, callables, ...) are ignored.

    # ------------------------------------------------------------------
    # (Req 4) TensorContainer metadata extraction. Every derived string/shape map is
    # produced by a single recursive walk over the call's args/kwargs, mirroring
    # ``_collect_subkeys``: positional arg ``i`` roots at path ``arg{i}``,
    # keyword ``k`` roots at ``k``, and nested dict/list/tuple children append
    # their key/index with ``SEP_BW_ARG_AND_DIM``.
    # ------------------------------------------------------------------
    def _walk_container_leaves(self, value: TensorContainer, path: str, on_leaf) -> None:
        """Recurse a ``TensorContainer`` invoking ``on_leaf(path, leaf)`` for every
        non-container leaf (both tensor and non-tensor); the caller keeps the
        kind it cares about."""
        if isinstance(value, torch.Tensor):
            on_leaf(path, value)
        elif isinstance(value, dict):
            for k, v in value.items():
                child = f"{path}{self.SEP_BW_ARG_AND_DIM}{k}" if path else str(k)
                self._walk_container_leaves(v, child, on_leaf)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                child = f"{path}{self.SEP_BW_ARG_AND_DIM}{i}" if path else str(i)
                self._walk_container_leaves(v, child, on_leaf)
        else:
            on_leaf(path, value)

    def _walk_call(self, args: tuple, kwargs: dict, on_leaf) -> None:
        """Walk every leaf of a call's positional args (rooted ``arg{i}``) then
        keyword args (rooted ``k``), excluding graph-internal workspace kwargs
        (see :attr:`GRAPH_INTERNAL_WORKSPACE_KWARGS`)."""
        roots = self._positional_param_names(len(args))
        for i, v in enumerate(args):
            self._walk_container_leaves(v, roots[i], on_leaf)
        for k, v in self._graph_input_kwargs(kwargs).items():
            self._walk_container_leaves(v, k, on_leaf)

    def _extract_tensor_container_shapes(self, args: tuple, kwargs: dict) -> TensorContainerShapes:
        """(Req 4.2) Map each input tensor's path to a 1-D int tensor of its dim
        lengths, so shape-bucket padding can read a dim length by direct index.
        Output shape tensors on same device as input tensor."""
        shapes: TensorContainerShapes = {}

        def on_leaf(path: str, leaf: Any) -> None:
            if isinstance(leaf, torch.Tensor):
                shapes[f"{path}_shape"] = torch.tensor(tuple(leaf.shape), dtype=torch.int32, device=leaf.device)

        self._walk_call(args, kwargs, on_leaf)
        return shapes

    @staticmethod
    def _tensor_container_shapes_to_cpu(tensor_container_shapes: TensorContainerShapes):
        return {
            name: shape.cpu() for name, shape in tensor_container_shapes.items()}

    def _extract_input_tensor_metadata(self, *args, **kwargs) -> str:
        """(Req 4.3) Deterministic string over every input tensor's path, dtype
        and shape, in walk order."""
        parts: list[str] = []

        def on_leaf(path: str, leaf: Any) -> None:
            if isinstance(leaf, torch.Tensor):
                dims = self.SEP_FOR_DIMS.join(str(d) for d in leaf.shape)
                parts.append(
                    f"{path}{self.SEP_BW_NAME_AND_SHAPE}{leaf.dtype}"
                    f"{self.SEP_BW_NAME_AND_SHAPE}{dims}")

        self._walk_call(args, kwargs, on_leaf)
        return self.SEP_FOR_ARGS.join(parts)

    def _extract_input_nontensor_metadata(self, *args, **kwargs) -> str:
        """(Req 4.4) Deterministic string over every non-tensor input leaf's path
        and type, in walk order."""
        parts: list[str] = []

        def on_leaf(path: str, leaf: Any) -> None:
            if isinstance(leaf, (bool, int, float, str)):
                parts.append(f"{path}{self.SEP_BW_NAME_AND_SHAPE}{leaf}")
            elif isinstance(leaf, AttentionMetadata):
                parts.append(f"{path}{self.SEP_BW_ARG_AND_TYPE}{type(leaf)}")
            elif not isinstance(leaf, Tensor):
                parts.append(f"{path}{self.SEP_BW_ARG_AND_TYPE}{type(leaf)}")

        self._walk_call(args, kwargs, on_leaf)
        return self.SEP_FOR_ARGS.join(parts)

    def _extract_input_metadata(self, *args, **kwargs) -> tuple[str, str]:
        """(Req 4.5) The call's ``(tensor-metadata, non-tensor-metadata)`` pair.

        Serves as *input-metadata* (Req 4.7) when applied to the forward inputs
        and as *output-metadata* (Req 4.8) when applied to the forward output.
        """
        tensor_metadata = self._extract_input_tensor_metadata(*args, **kwargs)
        nontensor_metadata = self._extract_input_nontensor_metadata(
            *args, **kwargs)
        return tensor_metadata, nontensor_metadata

    @abc.abstractmethod
    def update_graph_state_by_key(self, key: str) -> None:
        """Advance this call's per-key state for ``key``. Implemented by subclasses."""
        ...

    # ------------------------------------------------------------------
    # (Req 5.5-5.7) Shape-bucket padding. When ``input_key_method`` is
    # BUCKETED_SHAPES the wrapped module always runs at bucketed
    # (padded) shapes so one captured graph serves a range of live shapes:
    # ``pad_input`` grows the flagged input dims up to their bucket length before
    # capture/replay, and ``unpad_output`` truncates the tied output dims back to
    # the live shapes afterwards. Both are no-ops without a shape-bucket config.
    #
    # The rules come from ``config.input_routing_config`` (a ``InputRoutingConfig``,
    # produced by :class:`InputRoutingConfigFactory`), expanded from its named-dim
    # tie points + per-dim specs by the ``config`` helpers:
    #   - ``input_padded_assignments``   : (tensor_name, dim_idx, dim_name)
    #   - ``output_padded_assignments``  : (out_idx, dim_idx, dim_name)
    #   - ``bucket_lengths_by_name``     : dim_name -> sorted int bucket lengths
    # ``tensor_name`` matches the walk path used by ``_extract_container_*``
    # (positional arg ``i`` -> ``arg{i}``; keyword ``k`` -> ``k``).
    # ------------------------------------------------------------------
    def _map_container_tensors(self, value: TensorContainer, path: str, fn) -> TensorContainer:
        """Return a structural copy of ``value`` with ``fn(path, tensor)`` applied
        to every tensor leaf; dict/list/tuple nesting and non-tensor leaves are
        preserved. Paths are built exactly as in :meth:`_walk_container_leaves`."""
        if isinstance(value, torch.Tensor):
            return fn(path, value)
        if isinstance(value, dict):
            return {
                k: self._map_container_tensors(
                    v, f"{path}{self.SEP_BW_ARG_AND_DIM}{k}" if path else str(k), fn)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            mapped = [
                self._map_container_tensors(
                    v, f"{path}{self.SEP_BW_ARG_AND_DIM}{i}" if path else str(i), fn)
                for i, v in enumerate(value)
            ]
            return tuple(mapped) if isinstance(value, tuple) else mapped
        return value

    @staticmethod
    def _bucket_length(dim_len_values, length: int) -> int:
        """Lowest bucket boundary >= ``length`` (Req 6.1). Raises if the live
        length exceeds every configured boundary (no bucket can hold it)."""
        candidates = [int(v) for v in dim_len_values if int(v) >= int(length)]
        if not candidates:
            raise ValueError(
                f"dim length {int(length)} exceeds the largest configured "
                f"bucket {max(int(v) for v in dim_len_values)}")
        return min(candidates)

    def _pad_tensor_dims_to(self, t: Tensor, targets: dict) -> Tensor:
        """0-pad ``t`` so each ``dim_idx`` in ``targets`` reaches an explicit
        target length. ``targets`` maps ``dim_idx -> target_len``; ``dim_idx``
        may be negative, counting from the end (``-1`` is the last dimension)."""
        rank = t.dim()
        pad = [0] * (2 * rank)  # F.pad order: (last_dim_L, last_dim_R, ...)
        changed = False
        for dim_idx, target in targets.items():
            axis = dim_idx + rank if dim_idx < 0 else dim_idx
            if not 0 <= axis < rank:
                raise ValueError(
                    f"padded dim {dim_idx} out of range for tensor of rank {rank}")
            extra = int(target) - t.shape[axis]
            if extra < 0:
                raise ValueError(
                    f"pad target {int(target)} < current length {t.shape[axis]} "
                    f"on dim {dim_idx}")
            if extra:
                pad[2 * (rank - 1 - axis) + 1] = extra
                changed = True
        return F.pad(t, pad, mode="constant", value=0) if changed else t

    def _padded_input_targets(self) -> dict:
        """``tensor_name -> {dim_idx: dim_len_values}`` from the bucket config,
        or ``{}`` when no config / method is active."""
        cfg = self.graph_optimization_config.input_routing_config
        if cfg is None:
            return {}
        targets: dict = {}
        bucket_lengths = bucket_lengths_by_name(cfg)
        for tensor_name, dim_idx, dim_name in input_padded_assignments(cfg):
            targets.setdefault(tensor_name, {})[dim_idx] = bucket_lengths[dim_name]
        return targets

    def _output_tied_live_lengths(
            self, input_tensor_shapes: TensorContainerShapes) -> dict:
        """``out_idx -> {out_dim_idx: (dim_name, live_len)}`` for output dims tied
        to a padded input dim, reading each tied dim's live length from
        ``input_tensor_shapes``. Used by :meth:`unpad_output` to truncate each
        such output dim back to ``live_len``. ``{}`` when no config."""
        cfg = self.graph_optimization_config.input_routing_config
        if cfg is None:
            return {}
        live_len_by_dim_name: dict = {}
        for tensor_name, dim_idx, dim_name in input_padded_assignments(cfg):
            shape = input_tensor_shapes.get(f"{tensor_name}_shape")
            if shape is not None:
                live_len_by_dim_name.setdefault(dim_name, int(shape[dim_idx]))
        targets: dict = {}
        for out_idx, out_dim_idx, dim_name in output_padded_assignments(cfg):
            if dim_name in live_len_by_dim_name:
                targets.setdefault(out_idx, {})[out_dim_idx] = (
                    dim_name, live_len_by_dim_name[dim_name])
        return targets

    def pad_input(
        self, 
        args: tuple,
        kwargs: dict,
        input_tensor_shapes: TensorContainerShapes,
        ) -> TensorContainer:
        """(Req 5.5) Return ``(padded_args, padded_kwargs)`` with every input
        tensor 0-padded on the dims flagged in the shape-bucket config, up to the
        lowest bucket length >= the live length read from ``input_tensor_shapes``.
        No-op (identity) without a config. Graph-internal workspace kwargs are
        passed through untouched."""
        targets_by_path = self._padded_input_targets()
        if not targets_by_path:
            return args, kwargs

        def fn(path: str, t: Tensor) -> Tensor:
            dims = targets_by_path.get(path)
            if not dims:
                return t
            shape = input_tensor_shapes.get(f"{path}_shape")
            pad_targets = {
                dim_idx: self._bucket_length(
                    dim_len_values,
                    int(shape[dim_idx]) if shape is not None else t.shape[dim_idx])
                for dim_idx, dim_len_values in dims.items()
            }
            return self._pad_tensor_dims_to(t, pad_targets)

        roots = self._positional_param_names(len(args))
        padded_args = tuple(
            self._map_container_tensors(v, roots[i], fn)
            for i, v in enumerate(args))
        workspace = self._effective_workspace_kwargs()
        padded_kwargs = {
            k: (v if k in workspace
                else self._map_container_tensors(v, k, fn))
            for k, v in kwargs.items()
        }
        return padded_args, padded_kwargs

    def unpad_output(
        self, 
        args: tuple,
        input_tensor_shapes: TensorContainerShapes) -> TensorContainer:
        """(Req 5.7) Truncate output tensors whose dims are tied to a padded
        input dim back to the live input length.

        For each ``(out_idx, out_dim_idx, dim_name)`` from the config's padded
        output ties (``output_padded_assignments``), the target length is the
        live length of any input dim assigned the same ``dim_name`` (read from
        ``input_tensor_shapes``). No-op without a config.
        """
        if not isinstance(args, tuple):
            raise ValueError(f"expected args tuple, got {type(args)}")
        output = args[0] if len(args) == 1 else args
        out_targets = self._output_tied_live_lengths(input_tensor_shapes)
        if not out_targets:
            return output

        def truncate_at(idx: int, t):
            dims = out_targets.get(idx)
            if not dims or not isinstance(t, torch.Tensor):
                return t
            target = list(t.shape)
            for d, (_dim_name, live_len) in dims.items():
                target[d] = min(target[d], live_len)
            return t[tuple(slice(0, n) for n in target)]

        if isinstance(output, torch.Tensor):
            return truncate_at(0, output)
        if isinstance(output, (list, tuple)):
            mapped = [truncate_at(i, t) for i, t in enumerate(output)]
            return tuple(mapped) if isinstance(output, tuple) else mapped
        return output
    
