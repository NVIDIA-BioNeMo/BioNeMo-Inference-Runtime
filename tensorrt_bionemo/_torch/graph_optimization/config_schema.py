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
import enum

from tensorrt_bionemo.configs.base import BaseConfig


class GraphOptimizationMode(enum.Enum):
    """Which graph-optimization backend to apply when wrapping a module.

    Members:
        NO_OPTIMIZATION: Leave the module untouched
        CUDA_GRAPHS_VIA_TORCH: Wrap the module to drive per-input-key CUDA-graph
            warmup, capture, and replay via ``torch.cuda.graph``.
    """
    NO_OPTIMIZATION = "no_optimization"
    CUDA_GRAPHS_VIA_TORCH = "cuda_graphs_via_torch"
    # ToDo: TORCH_COMPILE = "torch_compile"


class InputKeyMethod(enum.Enum):
    """How to derive the per-call cache key that selects a captured graph.

    Each distinct key maps to its own ``CUDAGraphState`` (and thus its own
    captured graph / static buffers) in the tracker's LRU cache.

    Members:
        EXACT: Key on the exact shapes of the input tensors, so a separate graph
            is captured for every distinct input-shape signature.
    """
    EXACT = "exact"
    # ToDo:BUCKETED_SHAPES_METHOD_1 = "bucketed_shapes_method_1"


class BaseGraphOptimizationConfig(BaseConfig):
    """Base config describing how a module should be graph-optimized.

    Rooted in :class:`tensorrt_bionemo.configs.BaseConfig` so it composes with the
    rest of the config system (a module's ``config.graph_optimization_config`` holds
    one of these, and the wrapper carries it as its own ``config``). Concrete
    subclasses (e.g. :class:`CUDAGraphOptimizationConfig`) add framework-specific
    options.

    Attributes:
        graph_optimization_mode: Backend to apply; ``NO_OPTIMIZATION`` leaves
            the module eager. See :class:`GraphOptimizationMode`.
        input_key_method: How a call's input is reduced to a graph-cache key.
            See :class:`InputKeyMethod`.
        num_graphs_max_for_this_module: Capacity of the per-module LRU cache of
            captured graphs; the least-recently-used graph is evicted (and its
            buffers freed) once this many distinct input keys are live.
    """
    graph_optimization_mode: GraphOptimizationMode = GraphOptimizationMode.NO_OPTIMIZATION
    input_key_method: InputKeyMethod = InputKeyMethod.EXACT
    num_graphs_max_for_this_module: int = 1


class CUDAGraphOptimizationConfig(BaseGraphOptimizationConfig):
    """Config for the ``CUDA_GRAPHS_VIA_TORCH`` path.

    Adds the warmup-schedule thresholds the tracker uses to advance an input
    key's state machine (warmup -> kernels compiled -> allocator ready ->
    captured) before capturing its CUDA graph, plus an optional post-capture
    correctness check.

    Attributes:
        num_calls_for_kernel_compilation: Number of prior calls for a key
            before kernels are considered compiled/autotuned.
        num_calls_for_memory_allocator: Number of prior calls for a key
            before the caching allocator is considered primed and the graph is
            captured on the next call.
        verify_capture: When True, immediately after capturing a key's graph the
            tracker replays it and compares the result against a fresh eager run;
            a mismatch reverts that key permanently to eager. Adds one extra eager
            forward at capture time, so it is off by default.
    """
    num_calls_for_kernel_compilation: int = 1
    num_calls_for_memory_allocator: int = 3
    verify_capture: bool = False
