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
"""Checkpoint validation and stable per-request beam pruning."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def validate_step_checkpoints(step_checkpoints: Sequence[int], num_steps: int) -> tuple[int, ...]:
    """Require a ``0``-started, strictly increasing schedule that ends at ``num_steps``.

    Args:
        step_checkpoints: Inclusive step indices, typically ``[0, ..., num_steps]``.
        num_steps: Number of integration intervals in the rollout plan.

    Returns:
        The checkpoints as a tuple of ints.

    Raises:
        ValueError: If the checkpoints cannot drive a beam over ``num_steps``.
    """
    checkpoints = tuple(int(step) for step in step_checkpoints)
    if len(checkpoints) < 2:
        raise ValueError(f"step_checkpoints must have at least 2 entries (start and end), got {step_checkpoints}")
    if checkpoints[0] != 0:
        raise ValueError(
            f"step_checkpoints must begin at 0 (the beam is initialized at the start of the schedule), "
            f"got {checkpoints[0]}"
        )
    if any(later <= earlier for earlier, later in zip(checkpoints, checkpoints[1:], strict=False)):
        raise ValueError(f"step_checkpoints must be strictly increasing; got {step_checkpoints}")
    if checkpoints[-1] != num_steps:
        raise ValueError(
            f"step_checkpoints[-1]={checkpoints[-1]} must equal num_steps={num_steps}. "
            "The last checkpoint must coincide with the end of the schedule so surviving "
            "beam states are fully denoised before they are returned."
        )
    return checkpoints


def select_beam_indices(
    rewards: torch.Tensor,
    n_branch: int,
    beam_width: int,
    num_requests: int,
) -> torch.Tensor:
    """Return interleaved survivor indices for a branch-major candidate tensor.

    ``rewards`` is ``[n_branch * beam_width * num_requests]`` laid out as
    branch-major, then replica, then request. Selection is independent per
    request. ``argsort(..., stable=True, descending=True)`` makes equal
    rewards keep the lower candidate index.

    Args:
        rewards: One scalar per branched candidate, same layout as the stacked
            product-space states.
        n_branch: Branches spawned from each survivor.
        beam_width: Survivors kept per request.
        num_requests: Original batch size.

    Returns:
        Indices ``[beam_width * num_requests]`` in replica-major interleaved
        order ``[req0_rep0, req1_rep0, ..., req0_rep1, ...]``.

    Raises:
        ValueError: If ``rewards`` has the wrong length, or every candidate at
            a request is ``-inf``.
    """
    expected = n_branch * beam_width * num_requests
    if rewards.numel() != expected:
        raise ValueError(f"rewards must contain {expected} values, got {rewards.numel()}")
    if n_branch < 1 or beam_width < 1 or num_requests < 1:
        raise ValueError("n_branch, beam_width, and num_requests must be positive")

    flat = rewards.reshape(-1)
    negative_inf = torch.isinf(flat) & (flat < 0)
    if bool(negative_inf.all()):
        raise ValueError(
            "Every candidate scored -inf, so the prune would rank nothing: a weighted metric "
            "key is missing, non-finite, or not produced for this checkpoint"
        )

    reshaped = flat.view(n_branch, beam_width, num_requests)
    per_request = reshaped.permute(2, 0, 1).reshape(num_requests, n_branch * beam_width)
    request_invalid = torch.isinf(per_request) & (per_request < 0)
    if bool(request_invalid.all(dim=1).any()):
        bad = torch.where(request_invalid.all(dim=1))[0].tolist()
        raise ValueError(f"Every candidate for request(s) {bad} scored -inf, so the prune would rank nothing")
    top_k = torch.argsort(per_request, dim=1, descending=True, stable=True)[:, :beam_width]
    replica_indices = top_k % beam_width
    branch_indices = top_k // beam_width
    request_indices = torch.arange(num_requests, device=rewards.device).unsqueeze(1).expand(-1, beam_width)
    global_indices = branch_indices * num_requests * beam_width + replica_indices * num_requests + request_indices
    return global_indices.transpose(0, 1).reshape(-1)
