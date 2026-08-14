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
"""Shared graph-cache keying, input routing, and shape-bucket padding."""

import abc
import inspect
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bionemo_ir._torch.attention_backend import AttentionMetadata
from bionemo_ir._torch.graph_optimization.config import (
    GraphOptimizationConfig,
    acceptance_max_by_name,
    bucket_lengths_by_name,
    input_acceptance_assignments,
    input_padded_assignments,
    output_padded_assignments,
)
from bionemo_ir.runtime.backend import BackendBase

# A call container with at least one tensor.
TensorContainer = Tensor | tuple | dict

# Dotted tensor path to its 1-D shape tensor.
TensorContainerShapes = dict[str, Tensor]
TensorContainerHostShapes = dict[str, tuple[int, ...]]


class GraphOptimizationTracker(BackendBase):
    """Base wrapper for graph-cache keying, routing, and eager fallback."""

    SEP_FOR_ARGS = "|"
    SEP_BW_NAME_AND_SHAPE = "+"
    SEP_FOR_DIMS = "-"
    SEP_BW_ARG_AND_DIM = "."
    SEP_BW_ARG_AND_TYPE = ","

    # ``buffers`` is mutable scratch shared across score-model stages. Capture
    # changes its keys and shapes, so it cannot satisfy the static copy contract.
    # Pass it during warmup/capture, but exclude it from keys and replay copies;
    # the captured graph owns and reuses its workspace.
    GRAPH_INTERNAL_WORKSPACE_KWARGS: frozenset = frozenset({"buffers"})

    def _effective_workspace_kwargs(self) -> frozenset:
        """Return configured workspace kwargs or the built-in fallback."""
        cfg = self.graph_optimization_config.input_routing_config
        if cfg is not None and cfg.internal_workspace_kwargs:
            return frozenset(cfg.internal_workspace_kwargs)
        return self.GRAPH_INTERNAL_WORKSPACE_KWARGS

    def _positional_param_names(self, num_positional: int) -> list[str]:
        """Return cached parameter names, falling back to ``arg{i}``."""
        names = self._forward_positional_names
        if names is None:
            names = []
            inner = self.inner_module
            if inner is not None:
                try:
                    sig = inspect.signature(inner.forward)
                    names = [
                        p.name
                        for p in sig.parameters.values()
                        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    ]
                except (TypeError, ValueError):
                    names = []
            self._forward_positional_names = names
        return [names[i] if i < len(names) else f"arg{i}" for i in range(num_positional)]

    def _graph_input_kwargs(self, kwargs: dict) -> dict:
        """Remove graph-owned workspaces from key and replay inputs."""
        workspace = self._effective_workspace_kwargs()
        if not workspace:
            return kwargs
        return {k: v for k, v in kwargs.items() if k not in workspace}

    def __init__(
        self, graph_optimization_config: GraphOptimizationConfig, inner_module: nn.Module | None = None
    ) -> None:
        """Initialize the wrapper.

        Args:
            graph_optimization_config: Backend configuration.
            inner_module: Wrapped eager module.
        """
        super().__init__(graph_optimization_config)
        self.inner_module = inner_module
        # Validate ties once, outside the hot path.
        self._input_ties_validated = False
        # Lazily cached positional parameter names.
        self._forward_positional_names: list[str] | None = None

    @property
    def graph_optimization_config(self) -> GraphOptimizationConfig:
        return self._config

    def __getattr__(self, name: str) -> Any:
        """Resolve module attributes before delegating to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules")
            inner = modules.get("inner_module") if modules else None
            if inner is not None:
                return getattr(inner, name)
            raise

    def input_accepted(self, tensor_container_shapes: TensorContainerHostShapes) -> bool:
        """Return whether all configured axes are within inclusive limits."""
        cfg = self.graph_optimization_config.input_routing_config
        if cfg is None:
            return True
        acceptance_max = acceptance_max_by_name(cfg)
        for tensor_name, dim_idx, dim_name in input_acceptance_assignments(cfg):
            dim_len_max = acceptance_max.get(dim_name)
            if dim_len_max is None:
                continue
            shape = tensor_container_shapes.get(f"{tensor_name}_shape")
            if shape is None:
                continue
            ndim = len(shape)
            if not (-ndim <= dim_idx < ndim):
                continue
            if int(shape[dim_idx]) > dim_len_max:
                return False
        return True

    def validate_input_ties(self, tensor_container_shapes: TensorContainerHostShapes) -> None:
        """Validate that configured ties resolve on the first call."""
        if self._input_ties_validated:
            return
        cfg = self.graph_optimization_config.input_routing_config
        if cfg is not None:
            available = sorted(name[: -len("_shape")] for name in tensor_container_shapes if name.endswith("_shape"))
            ties = input_acceptance_assignments(cfg) + input_padded_assignments(cfg)
            for tensor_name, dim_idx, dim_name in ties:
                shape = tensor_container_shapes.get(f"{tensor_name}_shape")
                if shape is None:
                    raise ValueError(
                        f"input routing ties dim {dim_name!r} to input "
                        f"{tensor_name!r} (axis {dim_idx}), but no such tensor is "
                        f"present in the call; available inputs: {available}"
                    )
                ndim = len(shape)
                if not (-ndim <= dim_idx < ndim):
                    raise ValueError(
                        f"input routing ties dim {dim_name!r} to axis {dim_idx} "
                        f"of input {tensor_name!r}, but that tensor has only "
                        f"{ndim} dimension(s)"
                    )
        self._input_ties_validated = True

    def input_key_for_this_call(self, *args, **kwargs) -> str:
        """Return a cache key from tensor metadata and non-tensor values/types."""
        key = self.SEP_FOR_ARGS.join(self._extract_input_metadata(*args, **kwargs))
        return key

    def _collect_subkeys(self, value: Any, path: str, subkeys: list[str]) -> None:
        """Append a path-and-shape subkey for every tensor."""
        if isinstance(value, torch.Tensor):
            subkeys.append(
                self.SEP_BW_NAME_AND_SHAPE.join([path, self.SEP_FOR_DIMS.join([str(dim) for dim in value.shape])])
            )
        elif isinstance(value, dict):
            for k, v in value.items():
                child = f"{path}.{k}" if path else str(k)
                self._collect_subkeys(v, child, subkeys)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                child = f"{path}.{i}" if path else str(i)
                self._collect_subkeys(v, child, subkeys)

    def _walk_container_leaves(self, value: TensorContainer, path: str, on_leaf) -> None:
        """Call ``on_leaf(path, leaf)`` for each nested leaf."""
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
        """Walk call leaves, excluding graph-owned workspace kwargs."""
        roots = self._positional_param_names(len(args))
        for i, v in enumerate(args):
            self._walk_container_leaves(v, roots[i], on_leaf)
        for k, v in self._graph_input_kwargs(kwargs).items():
            self._walk_container_leaves(v, k, on_leaf)

    def _extract_tensor_container_shape_maps(
        self, args: tuple, kwargs: dict
    ) -> tuple[TensorContainerShapes, TensorContainerHostShapes]:
        """Map tensor paths to device tensors and zero-copy host metadata."""
        shapes_device: TensorContainerShapes = {}
        shapes_host: TensorContainerHostShapes = {}

        def on_leaf(path: str, leaf: Any) -> None:
            if isinstance(leaf, torch.Tensor):
                key = f"{path}_shape"
                shape = tuple(leaf.shape)
                shapes_device[key] = torch.tensor(shape, dtype=torch.int32).to(leaf.device, non_blocking=True)
                shapes_host[key] = shape

        self._walk_call(args, kwargs, on_leaf)
        return shapes_device, shapes_host

    def _extract_tensor_container_shapes(self, args: tuple, kwargs: dict) -> TensorContainerShapes:
        """Map tensor paths to shape tensors on the input devices."""
        shapes_device, _ = self._extract_tensor_container_shape_maps(args, kwargs)
        return shapes_device

    def _extract_input_tensor_metadata(self, *args, **kwargs) -> str:
        """Serialize tensor paths, dtypes, and shapes in walk order."""
        parts: list[str] = []

        def on_leaf(path: str, leaf: Any) -> None:
            if isinstance(leaf, torch.Tensor):
                dims = self.SEP_FOR_DIMS.join(str(d) for d in leaf.shape)
                parts.append(f"{path}{self.SEP_BW_NAME_AND_SHAPE}{leaf.dtype}{self.SEP_BW_NAME_AND_SHAPE}{dims}")

        self._walk_call(args, kwargs, on_leaf)
        return self.SEP_FOR_ARGS.join(parts)

    def _extract_input_nontensor_metadata(self, *args, **kwargs) -> str:
        """Serialize non-tensor paths and values/types in walk order."""
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
        """Return tensor and non-tensor metadata strings."""
        tensor_metadata = self._extract_input_tensor_metadata(*args, **kwargs)
        nontensor_metadata = self._extract_input_nontensor_metadata(*args, **kwargs)
        return tensor_metadata, nontensor_metadata

    @abc.abstractmethod
    def update_graph_state_by_key(self, key: str) -> None:
        """Advance this call's per-key state for ``key``. Implemented by subclasses."""
        ...

    # Bucketed mode pads tied inputs for capture and restores live output sizes.
    # Routing assignments use the same dotted paths as metadata extraction.
    def _map_container_tensors(self, value: TensorContainer, path: str, fn) -> TensorContainer:
        """Map tensor leaves while preserving paths and container structure."""
        if isinstance(value, torch.Tensor):
            return fn(path, value)
        if isinstance(value, dict):
            return {
                k: self._map_container_tensors(v, f"{path}{self.SEP_BW_ARG_AND_DIM}{k}" if path else str(k), fn)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            mapped = [
                self._map_container_tensors(v, f"{path}{self.SEP_BW_ARG_AND_DIM}{i}" if path else str(i), fn)
                for i, v in enumerate(value)
            ]
            return tuple(mapped) if isinstance(value, tuple) else mapped
        return value

    @staticmethod
    def _bucket_length(dim_len_values, length: int) -> int:
        """Return the smallest bucket that covers ``length``."""
        candidates = [int(v) for v in dim_len_values if int(v) >= int(length)]
        if not candidates:
            raise ValueError(
                f"dim length {int(length)} exceeds the largest configured bucket {max(int(v) for v in dim_len_values)}"
            )
        return min(candidates)

    def _pad_tensor_dims_to(self, t: Tensor, targets: dict) -> Tensor:
        """Right-pad selected dimensions to explicit lengths."""
        rank = t.dim()
        pad = [0] * (2 * rank)  # F.pad orders dimensions from last to first.
        changed = False
        for dim_idx, target in targets.items():
            axis = dim_idx + rank if dim_idx < 0 else dim_idx
            if not 0 <= axis < rank:
                raise ValueError(f"padded dim {dim_idx} out of range for tensor of rank {rank}")
            extra = int(target) - t.shape[axis]
            if extra < 0:
                raise ValueError(f"pad target {int(target)} < current length {t.shape[axis]} on dim {dim_idx}")
            if extra:
                pad[2 * (rank - 1 - axis) + 1] = extra
                changed = True
        return F.pad(t, pad, mode="constant", value=0) if changed else t

    def _padded_input_targets(self) -> dict:
        """Return bucket boundaries by input path and dimension."""
        cfg = self.graph_optimization_config.input_routing_config
        if cfg is None:
            return {}
        targets: dict = {}
        bucket_lengths = bucket_lengths_by_name(cfg)
        for tensor_name, dim_idx, dim_name in input_padded_assignments(cfg):
            targets.setdefault(tensor_name, {})[dim_idx] = bucket_lengths[dim_name]
        return targets

    def _output_tied_live_lengths(self, input_tensor_shapes: TensorContainerHostShapes) -> dict:
        """Return live lengths for output dimensions tied to padded inputs."""
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
                targets.setdefault(out_idx, {})[out_dim_idx] = (dim_name, live_len_by_dim_name[dim_name])
        return targets

    def pad_input(
        self,
        args: tuple,
        kwargs: dict,
        input_tensor_shapes: TensorContainerHostShapes,
    ) -> TensorContainer:
        """Pad configured input dimensions to their smallest covering buckets.

        Graph-owned workspace kwargs pass through unchanged.
        """
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
                    dim_len_values, int(shape[dim_idx]) if shape is not None else t.shape[dim_idx]
                )
                for dim_idx, dim_len_values in dims.items()
            }
            return self._pad_tensor_dims_to(t, pad_targets)

        roots = self._positional_param_names(len(args))
        padded_args = tuple(self._map_container_tensors(v, roots[i], fn) for i, v in enumerate(args))
        workspace = self._effective_workspace_kwargs()
        padded_kwargs = {k: (v if k in workspace else self._map_container_tensors(v, k, fn)) for k, v in kwargs.items()}
        return padded_args, padded_kwargs

    def unpad_output(self, args: tuple, input_tensor_shapes: TensorContainerHostShapes) -> TensorContainer:
        """Restore tied output dimensions to their live input lengths."""
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
