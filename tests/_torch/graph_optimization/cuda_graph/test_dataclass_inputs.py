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
"""Tensors held in a dataclass input are keyed, copied, and released like those in a dict.

The RFProteína trunk hands its attention masks and triangle-multiplication row lengths to a graph region
as dataclasses. A replay that skipped them kept the masks of the call that captured the graph, so a request
whose true length differed from the capture's, but rounded up to the same padded shape, lost its last tokens.
"""

import dataclasses

import pytest
import torch
import torch.nn as nn

from bionemo_ir._torch.graph_optimization.config import (
    CUDAGraphOptimizationConfig,
    InputRoutingConfig,
)
from bionemo_ir._torch.graph_optimization.cuda_graph.memory import container_device, tensor_bytes
from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import (
    CUDAGraphOptimizationTracker,
    CUDAGraphPreparationState,
)
from bionemo_ir._torch.graph_optimization.tensor_copy_utils import (
    _assert_equal_but_distinct,
    _clone_tensors,
    _copy_tensors_into,
    _delete_tensors_in_container,
    _tensor_leaves,
)

NUM_CALLS_TO_CAPTURE = 4  # warmup thresholds (1, 3) + capture on the next call


@dataclasses.dataclass(frozen=True)
class _Lengths:
    """A frozen record of per-row valid lengths, as the trunk's masks and metadata are."""

    valid: torch.Tensor
    inner: dict[str, torch.Tensor] | None = None
    pad_multiple: int = -1


def _same_values(left: _Lengths, right: _Lengths) -> bool:
    return all(torch.equal(a, b) for a, b in zip(_tensor_leaves(left), _tensor_leaves(right), strict=True))


def _lengths(valid: list[bool], device: str = "cpu", pad_multiple: int = -1) -> _Lengths:
    mask = torch.tensor([valid], dtype=torch.bool, device=device)
    return _Lengths(valid=mask, inner={"count": mask.sum(-1, keepdim=True)}, pad_multiple=pad_multiple)


def test_clone_of_a_frozen_dataclass_is_independent_and_keeps_its_type() -> None:
    original = _lengths([True, True, False])
    clone = _clone_tensors(original)
    assert type(clone) is _Lengths and clone is not original
    _assert_equal_but_distinct(original, clone)
    assert clone.pad_multiple == original.pad_multiple
    assert _same_values(original, clone)
    clone.inner["count"].add_(1)
    assert not _same_values(original, clone)


def test_copy_into_a_dataclass_updates_its_tensors_in_place() -> None:
    static, live = _lengths([True, True, False]), _lengths([True, True, True])
    pointers = [leaf.data_ptr() for leaf in _tensor_leaves(static)]
    _copy_tensors_into(static, live)
    assert [leaf.data_ptr() for leaf in _tensor_leaves(static)] == pointers
    assert _same_values(static, live)


def test_copy_into_a_dataclass_rejects_another_type() -> None:
    with pytest.raises(AssertionError, match="structure mismatch"):
        _copy_tensors_into(_lengths([True]), {"valid": torch.ones(1, 1, dtype=torch.bool)})


def test_leaves_bytes_and_device_see_through_a_dataclass() -> None:
    value = _lengths([True, False, True])
    assert len(_tensor_leaves(value)) == 2
    assert tensor_bytes(value) == tensor_bytes([value.valid, value.inner])
    assert container_device(value) == value.valid.device


def test_release_drops_the_tensors_a_dataclass_holds() -> None:
    value = _lengths([True, False])
    _delete_tensors_in_container(value)
    assert value.valid is None and value.inner is None and value.pad_multiple == -1


class _Masked(nn.Module):
    def forward(self, x: torch.Tensor, lengths: _Lengths) -> torch.Tensor:
        return (x * lengths.valid.unsqueeze(-1)).sum(-2) + lengths.inner["count"]


def _tracker(stable: bool) -> CUDAGraphOptimizationTracker:
    routing = InputRoutingConfig(stable_input_kwargs=["lengths"] if stable else [])
    return CUDAGraphOptimizationTracker(
        CUDAGraphOptimizationConfig(input_routing_config=routing), inner_module=_Masked().cuda()
    ).eval()


def _reference(x: torch.Tensor, lengths: _Lengths) -> torch.Tensor:
    return _Masked()(x, lengths)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-graph tests require CUDA")
@pytest.mark.parametrize("stable", [False, True])
def test_replay_uses_the_dataclass_of_the_live_call_not_the_capturing_one(stable: bool) -> None:
    tracker = _tracker(stable)
    x = torch.arange(1, 9, dtype=torch.float32, device="cuda").reshape(1, 4, 2)
    captured_with = _lengths([True, True, True, False], "cuda")  # three valid tokens of four
    with torch.no_grad():
        for _ in range(NUM_CALLS_TO_CAPTURE):
            tracker(x, lengths=captured_with)
        key = tracker.input_key_for_this_call(x, lengths=captured_with)
        assert tracker.graph_state_by_key[key].preparation_state is CUDAGraphPreparationState.GRAPH_CAPTURED
        # The same padded shape with a longer true length: the last token is now valid.
        for valid in ([True, True, True, True], [True, False, False, False]):
            live = _lengths(valid, "cuda")
            assert tracker.input_key_for_this_call(x, lengths=live) == key
            torch.testing.assert_close(tracker(x, lengths=live), _reference(x, live))
    assert tracker.execution_counts["replay"] >= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-graph tests require CUDA")
def test_a_non_tensor_field_that_differs_keys_another_graph() -> None:
    tracker = _tracker(stable=False)
    x = torch.ones(1, 4, 2, device="cuda")
    keys = {
        tracker.input_key_for_this_call(x, lengths=_lengths([True] * 4, "cuda", pad_multiple=multiple))
        for multiple in (-1, 16)
    }
    assert len(keys) == 2
