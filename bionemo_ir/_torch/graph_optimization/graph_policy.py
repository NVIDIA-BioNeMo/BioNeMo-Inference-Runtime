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

"""Construct graph execution policies from model-owned routing declarations."""

from __future__ import annotations

import gc
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .config import (
    CUDAGraphOptimizationConfig,
    GraphOptimizationMode,
    InputAcceptanceDimSpec,
    InputKeyMethod,
    InputRoutingConfig,
    NamedDimTies,
)

if TYPE_CHECKING:
    from .region import GraphRegion


def _release_unused_graphs(regions: Sequence[GraphRegion], *, num_tokens: int) -> None:
    """Release ineligible region caches before allocating an eager request.

    Args:
        regions: Persistent regions owned by the model serving the request.
        num_tokens: Token count before construction of large model activations.

    Cached graphs cannot serve inputs above their configured token limit.
    Drop their live storage first, then return wholly unused allocator segments
    to CUDA. In-range caches and permanent per-key fallback decisions survive.
    """
    released = 0
    for region in regions:
        policy = region.policy
        routing = policy.input_routing_config if policy is not None else None
        if routing is None or region.tracker is None:
            continue
        maximum = next((spec.dim_len_max for spec in routing.input_acceptance_dims if spec.name == "num_tokens"), None)
        if maximum is not None and num_tokens > maximum and region.tracker.graph_state_by_key:
            if torch.cuda.is_current_stream_capturing():
                return
            released += region.tracker._evict_all_keys()
    if released:
        gc.collect()
        torch.cuda.empty_cache()


def exact_graph_config(
    *,
    named_dims: Sequence[NamedDimTies],
    max_tokens: int,
    repeated: bool = False,
    static_args: Sequence[str] = (),
    workspace_kwargs: Sequence[str] = (),
    stable_kwargs: Sequence[str] = (),
) -> CUDAGraphOptimizationConfig:
    """Build an exact policy with model-declared dimensions and acceptance.

    Args:
        named_dims: Input/output axes defined by the model's region signature.
        max_tokens: Inclusive token limit; larger inputs run eagerly.
        repeated: Verify and prepare the first eligible repeated-region call.
        static_args: Arguments whose structural metadata enters routing keys.
        workspace_kwargs: Scratch arguments owned by each graph state.
        stable_kwargs: Read-only inputs that stay fixed during a rollout.
    """
    return CUDAGraphOptimizationConfig(
        graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
        input_key_method=InputKeyMethod.EXACT,
        input_routing_config=InputRoutingConfig(
            named_dim_ties=list(named_dims),
            input_acceptance_dims=[InputAcceptanceDimSpec(name="num_tokens", dim_len_max=max_tokens)],
            static_args=list(static_args),
            internal_workspace_kwargs=list(workspace_kwargs),
            stable_input_kwargs=list(stable_kwargs),
        ),
        num_calls_for_kernel_compilation=1,
        num_calls_for_memory_allocator=1 if repeated else 3,
        capture_on_first_call=True,
        verify_capture=repeated,
        num_graphs_max_for_this_module=4 if repeated else 1,
        graph_cache_budget_bytes=(4 << 30) if repeated else None,
    )


def trunk_graph_config() -> CUDAGraphOptimizationConfig:
    """Exact policy for one trunk recycle that maps ``(s, z, s_init, z_init, ...)`` to ``(s, z)``."""
    return exact_graph_config(
        named_dims=(
            NamedDimTies(
                name="num_tokens",
                input_dims=(("s", (-2,)), ("z", (-2, -3)), ("s_init", (-2,)), ("z_init", (-2, -3))),
                output_dims=((0, (-2,)), (1, (-2, -3))),
            ),
        ),
        max_tokens=1024,
        repeated=True,
    )


def pairformer_graph_config() -> CUDAGraphOptimizationConfig:
    """Exact policy for a Pairformer stack called once per sample, such as a confidence head's."""
    return exact_graph_config(
        named_dims=(
            NamedDimTies(
                name="num_tokens",
                input_dims=(("s", (-2,)), ("z", (-2, -3)), ("mask", (-1,)), ("pair_mask", (-1, -2))),
                output_dims=((0, (-2,)), (1, (-2, -3))),
            ),
        ),
        max_tokens=1024,
        repeated=True,
        static_args=("mask", "pair_mask"),
        workspace_kwargs=("buffers",),
    )
