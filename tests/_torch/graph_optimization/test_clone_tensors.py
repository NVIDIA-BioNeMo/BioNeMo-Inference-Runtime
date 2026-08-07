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
"""Unit tests for ``GraphOptimizationTracker._clone_tensors``.

``_clone_tensors`` recursively walks an arbitrary nesting of dicts / lists /
tuples and replaces every ``torch.Tensor`` with an independent ``clone()``,
preserving container structure and passing non-tensor leaves through
unchanged. These tests assert that for every cloned tensor the clone:

  * lives at a *different* memory address (distinct storage / ``data_ptr``),
  * holds the *same* values,
  * has the *same* dtype, and
  * is on the *same* device.
"""

import pytest
import torch
import torch.nn as nn

from tensorrt_bionemo._torch.graph_optimization.config import CUDAGraphOptimizationConfig
from tensorrt_bionemo._torch.graph_optimization.cuda_graph.runtime import CUDAGraphOptimizationTracker
from tensorrt_bionemo._torch.graph_optimization.tensor_copy_utils import (
    _clone_tensors,
    _copy_tensors_into,
    _delete_tensors_in_container,
    _tensor_containers_are_byte_equal,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture
def tracker() -> CUDAGraphOptimizationTracker:
    return CUDAGraphOptimizationTracker(CUDAGraphOptimizationConfig(), nn.Identity())


def _assert_is_independent_clone(original, clone) -> None:
    """Recursively walk ``original`` and ``clone`` in lock-step, asserting at
    every tensor leaf that the clone is an independent copy (different address;
    same values, dtype, device) and that container structure is preserved.
    """
    if isinstance(original, torch.Tensor):
        assert isinstance(clone, torch.Tensor)
        # Different address: a clone has its own storage.
        assert clone.data_ptr() != original.data_ptr()
        assert clone is not original
        # Same values / dtype / device.
        assert clone.dtype == original.dtype
        assert clone.device == original.device
        assert clone.shape == original.shape
        assert torch.equal(clone, original)
    elif isinstance(original, dict):
        assert isinstance(clone, dict)
        assert clone.keys() == original.keys()
        for k in original:
            _assert_is_independent_clone(original[k], clone[k])
    elif isinstance(original, (list, tuple)):
        assert type(clone) is type(original)
        assert len(clone) == len(original)
        for o, c in zip(original, clone, strict=True):
            _assert_is_independent_clone(o, c)
    else:
        # Non-tensor leaves are passed through unchanged (same object).
        assert clone is original


@pytest.mark.parametrize("device", DEVICES)
def test_clone_single_tensor(tracker: CUDAGraphOptimizationTracker, device: str):
    original = torch.randn(2, 3, device=device)
    clone = _clone_tensors(original)
    _assert_is_independent_clone(original, clone)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64])
def test_clone_preserves_dtype(tracker: CUDAGraphOptimizationTracker, device: str, dtype: torch.dtype):
    original = torch.ones(4, 5, dtype=dtype, device=device)
    clone = _clone_tensors(original)
    _assert_is_independent_clone(original, clone)
    assert clone.dtype == dtype


@pytest.mark.parametrize("device", DEVICES)
def test_clone_nested_structure(tracker: CUDAGraphOptimizationTracker, device: str):
    original = {
        "s_trunk": torch.randn(1, 5, device=device),
        "feature_dict": {
            "ref_pos": torch.randn(1, 5, 3, device=device),
            "mol_type": torch.zeros(1, 5, dtype=torch.int64, device=device),
        },
        "coords": [torch.arange(4, device=device), (torch.ones(2, device=device), torch.zeros(3, device=device))],
    }
    clone = _clone_tensors(original)
    _assert_is_independent_clone(original, clone)


@pytest.mark.parametrize("device", DEVICES)
def test_clone_args_tuple(tracker: CUDAGraphOptimizationTracker, device: str):
    # Mirrors how forward() clones the positional ``args`` tuple.
    args = (torch.randn(2, 3, device=device), [torch.randn(4, device=device)])
    clone = _clone_tensors(args)
    _assert_is_independent_clone(args, clone)


def test_clone_passes_through_non_tensors(tracker: CUDAGraphOptimizationTracker):
    original = {"flag": "x", "n": 7, "none": None, "fn": len}
    clone = _clone_tensors(original)
    _assert_is_independent_clone(original, clone)


@pytest.mark.parametrize("device", DEVICES)
def test_clone_mutation_does_not_affect_original(tracker: CUDAGraphOptimizationTracker, device: str):
    original = torch.zeros(3, 4, device=device)
    clone = _clone_tensors(original)
    clone.add_(1.0)
    # Original must be untouched: clone is backed by independent storage.
    assert torch.equal(original, torch.zeros(3, 4, device=device))
    assert torch.equal(clone, torch.ones(3, 4, device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_clone_preserves_container_type(tracker: CUDAGraphOptimizationTracker, device: str):
    list_in = [torch.randn(2, device=device)]
    tuple_in = (torch.randn(2, device=device),)
    assert isinstance(_clone_tensors(list_in), list)
    assert isinstance(_clone_tensors(tuple_in), tuple)


# ---------------------------------------------------------------------------
# _copy_tensors_into
# ---------------------------------------------------------------------------
# ``_copy_tensors_into(dest, src)`` walks ``dest`` and ``src`` in lock-step and
# writes each source tensor's values into the corresponding destination tensor
# in-place (``dest.copy_(src)``). After the copy each ``dest`` tensor must hold
# the *same* values, dtype, and device as ``src`` while keeping its *own*
# storage (a *different* address from ``src``).


def _assert_copied_in_place(dest, src) -> None:
    """Recursively assert that ``dest`` received ``src``'s values in-place:
    identical values / dtype / device, but a different address (distinct
    storage) from ``src``.
    """
    if isinstance(dest, torch.Tensor):
        assert isinstance(src, torch.Tensor)
        # Different address: the copy is in-place, dest keeps its own storage.
        assert dest.data_ptr() != src.data_ptr()
        assert dest is not src
        # Identical values / dtype / device.
        assert dest.dtype == src.dtype
        assert dest.device == src.device
        assert dest.shape == src.shape
        assert torch.equal(dest, src)
    elif isinstance(dest, dict):
        assert dest.keys() == src.keys()
        for k in dest:
            _assert_copied_in_place(dest[k], src[k])
    elif isinstance(dest, (list, tuple)):
        assert len(dest) == len(src)
        for d, s in zip(dest, src, strict=True):
            _assert_copied_in_place(d, s)


@pytest.mark.parametrize("device", DEVICES)
def test_copy_single_tensor(tracker: CUDAGraphOptimizationTracker, device: str):
    dest = torch.zeros(2, 3, device=device)
    src = torch.randn(2, 3, device=device)
    _copy_tensors_into(dest, src)
    _assert_copied_in_place(dest, src)


@pytest.mark.parametrize("device", DEVICES)
def test_copy_preserves_dest_address(tracker: CUDAGraphOptimizationTracker, device: str):
    # The whole point of the in-place copy: dest's storage address is the same
    # before and after (so a captured CUDA graph keeps reading from it).
    dest = torch.zeros(4, 5, device=device)
    addr_before = dest.data_ptr()
    src = torch.randn(4, 5, device=device)
    _copy_tensors_into(dest, src)
    assert dest.data_ptr() == addr_before
    _assert_copied_in_place(dest, src)


@pytest.mark.parametrize("device", DEVICES)
def test_copy_nested_structure(tracker: CUDAGraphOptimizationTracker, device: str):
    def _make(fill):
        return {
            "s_trunk": torch.full((1, 5), fill, device=device),
            "feature_dict": {
                "ref_pos": torch.full((1, 5, 3), fill, device=device),
                "mol_type": torch.full((1, 5), int(fill), dtype=torch.int64, device=device),
            },
            "coords": [torch.full((4,), fill, device=device), (torch.full((2,), fill, device=device),)],
        }

    dest = _make(0.0)
    src = _make(3.0)
    # Record nested addresses to confirm they survive the in-place copy.
    addr_before = dest["feature_dict"]["ref_pos"].data_ptr()
    _copy_tensors_into(dest, src)
    assert dest["feature_dict"]["ref_pos"].data_ptr() == addr_before
    _assert_copied_in_place(dest, src)


@pytest.mark.parametrize("device", DEVICES)
def test_copy_args_and_kwargs_shapes(tracker: CUDAGraphOptimizationTracker, device: str):
    # Mirrors how copy_args_into_static feeds args (tuple) / kwargs (dict).
    dest_args = (torch.zeros(2, 3, device=device), [torch.zeros(4, device=device)])
    src_args = (torch.randn(2, 3, device=device), [torch.randn(4, device=device)])
    _copy_tensors_into(dest_args, src_args)
    _assert_copied_in_place(dest_args, src_args)

    dest_kwargs = {"x": torch.zeros(5, device=device)}
    src_kwargs = {"x": torch.randn(5, device=device)}
    _copy_tensors_into(dest_kwargs, src_kwargs)
    _assert_copied_in_place(dest_kwargs, src_kwargs)


@pytest.mark.parametrize("device", DEVICES)
def test_copy_does_not_alias_source(tracker: CUDAGraphOptimizationTracker, device: str):
    # After copying, mutating src must NOT change dest (independent storage).
    dest = torch.zeros(3, 4, device=device)
    src = torch.ones(3, 4, device=device)
    _copy_tensors_into(dest, src)
    src.add_(1.0)
    assert torch.equal(dest, torch.ones(3, 4, device=device))


# ---------------------------------------------------------------------------
# _delete_tensors_in_container
# ---------------------------------------------------------------------------
# ``_delete_tensors_in_container(value)`` recursively walks an arbitrary nesting
# of dict / list / tuple and drops the tensors it holds by emptying the mutable
# containers that reference them (``dict.clear()`` / ``list.clear()``). Tuples
# are immutable so they keep their elements, but any mutable container nested
# *inside* a tuple is still cleared. Non-tensor leaves are left untouched.


@pytest.mark.parametrize("device", DEVICES)
def test_delete_clears_dict(device: str):
    container = {"a": torch.randn(2, 3, device=device), "b": torch.randn(4, device=device)}
    _delete_tensors_in_container(container)
    assert container == {}


@pytest.mark.parametrize("device", DEVICES)
def test_delete_clears_list(device: str):
    container = [torch.randn(2, device=device), torch.randn(3, device=device)]
    _delete_tensors_in_container(container)
    assert container == []


@pytest.mark.parametrize("device", DEVICES)
def test_delete_clears_nested_containers(device: str):
    inner_dict = {"ref_pos": torch.randn(1, 5, 3, device=device)}
    inner_list = [torch.arange(4, device=device)]
    container = {
        "s_trunk": torch.randn(1, 5, device=device),
        "feature_dict": inner_dict,
        "coords": inner_list,
    }
    _delete_tensors_in_container(container)
    # Both the outer dict and every nested mutable container are emptied.
    assert container == {}
    assert inner_dict == {}
    assert inner_list == []


@pytest.mark.parametrize("device", DEVICES)
def test_delete_tuple_not_cleared_but_inner_mutables_are(device: str):
    # A tuple is immutable: it keeps its length/elements, but a mutable
    # container nested inside it is still cleared.
    inner_dict = {"x": torch.randn(2, device=device)}
    container = (torch.randn(2, device=device), inner_dict)
    _delete_tensors_in_container(container)
    assert len(container) == 2  # tuple itself is untouched
    assert inner_dict == {}  # nested dict is cleared


@pytest.mark.parametrize("device", DEVICES)
def test_delete_bare_tensor_does_not_raise(device: str):
    # A bare top-level tensor has no container to clear; the call must be a
    # no-op that returns None without raising.
    tensor = torch.randn(3, 4, device=device)
    assert _delete_tensors_in_container(tensor) is None


def test_delete_clears_container_with_non_tensor_leaves():
    # Containers are emptied regardless of leaf type; bare non-tensor leaves
    # passed at the top level are left untouched (no matching branch).
    container = {"flag": "x", "n": 7, "none": None}
    _delete_tensors_in_container(container)
    assert container == {}

    scalar = 7
    assert _delete_tensors_in_container(scalar) is None


# ---------------------------------------------------------------------------
# _tensor_containers_are_byte_equal
# ---------------------------------------------------------------------------
# ``_tensor_containers_are_byte_equal(left, right)`` returns True iff both args
# are matching *tensor-containers* (a tensor, or a dict / list / tuple whose
# every leaf is itself a tensor-container) with identical structure (same
# container type, length, dict keys) and byte-equal tensors at every leaf.
# Comparison is raw-byte (via ``_tensors_byte_equal``): same dtype / shape /
# device required, NaNs with the same bit pattern compare equal, and +0.0 vs
# -0.0 compare unequal. Any non-tensor leaf makes the object a non-container,
# which yields False (stricter than the clone/copy helpers, which pass
# non-tensors through).


@pytest.mark.parametrize("device", DEVICES)
def test_byte_equal_identical_single_tensor(device: str):
    a = torch.randn(2, 3, device=device)
    # A clone is a distinct object with identical bytes -> equal.
    assert _tensor_containers_are_byte_equal(a, a.clone())


@pytest.mark.parametrize("device", DEVICES)
def test_byte_equal_bf16_random_values(device: str):
    # bfloat16 is 2 bytes wide; this exercises the raw uint8 reinterpret on a
    # non-default-width float dtype carrying random (non-trivial) bit patterns.
    a = torch.randn(8, 16, device=device).to(torch.bfloat16)
    # A clone has identical bytes -> equal.
    assert _tensor_containers_are_byte_equal(a, a.clone())
    # Perturbing a single bf16 element makes it byte-unequal.
    b = a.clone()
    b[0, 0] = a[0, 0] + torch.tensor(1.0, dtype=torch.bfloat16, device=device)
    assert not _tensor_containers_are_byte_equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_byte_equal_differing_values_is_false(device: str):
    a = torch.zeros(2, 3, device=device)
    b = torch.zeros(2, 3, device=device)
    b[0, 0] = 1.0
    assert not _tensor_containers_are_byte_equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_byte_equal_differing_dtype_is_false(device: str):
    a = torch.ones(4, dtype=torch.float32, device=device)
    b = torch.ones(4, dtype=torch.float16, device=device)
    assert not _tensor_containers_are_byte_equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_byte_equal_differing_shape_is_false(device: str):
    a = torch.ones(2, 3, device=device)
    b = torch.ones(3, 2, device=device)
    assert not _tensor_containers_are_byte_equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_byte_equal_empty_tensors_same_meta_is_true(device: str):
    # numel() == 0 short-circuits to True once dtype/shape/device match.
    a = torch.empty(0, dtype=torch.int64, device=device)
    b = torch.empty(0, dtype=torch.int64, device=device)
    assert _tensor_containers_are_byte_equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_byte_equal_matching_nan_bit_patterns_is_true(device: str):
    # Raw-byte comparison: identical NaN bit patterns compare equal even though
    # ``nan != nan`` under value equality.
    a = torch.full((4,), float("nan"), device=device)
    b = torch.full((4,), float("nan"), device=device)
    assert _tensor_containers_are_byte_equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_byte_equal_signed_zero_is_false(device: str):
    # +0.0 and -0.0 are value-equal but have different bit patterns.
    a = torch.zeros(3, device=device)  # +0.0
    b = torch.full((3,), -0.0, device=device)  # -0.0
    assert torch.equal(a, b)  # value-equal ...
    assert not _tensor_containers_are_byte_equal(a, b)  # ... but byte-unequal


@pytest.mark.parametrize("device", DEVICES)
def test_byte_equal_matching_nested_structure_is_true(device: str):
    def _make():
        return {
            "s_trunk": torch.ones(1, 5, device=device),
            "feature_dict": {
                "ref_pos": torch.arange(15, device=device).reshape(1, 5, 3).float(),
                "mol_type": torch.zeros(1, 5, dtype=torch.int64, device=device),
            },
            "coords": [torch.arange(4, device=device), (torch.ones(2, device=device),)],
        }

    assert _tensor_containers_are_byte_equal(_make(), _make())


@pytest.mark.parametrize("device", DEVICES)
def test_byte_equal_one_differing_leaf_is_false(device: str):
    left = {"a": torch.ones(2, device=device), "b": [torch.zeros(3, device=device)]}
    right = {"a": torch.ones(2, device=device), "b": [torch.zeros(3, device=device)]}
    right["b"][0][1] = 9.0  # perturb a single nested leaf
    assert not _tensor_containers_are_byte_equal(left, right)


@pytest.mark.parametrize("device", DEVICES)
def test_byte_equal_differing_dict_keys_is_false(device: str):
    left = {"a": torch.ones(2, device=device)}
    right = {"b": torch.ones(2, device=device)}
    assert not _tensor_containers_are_byte_equal(left, right)


@pytest.mark.parametrize("device", DEVICES)
def test_byte_equal_differing_length_is_false(device: str):
    left = [torch.ones(2, device=device)]
    right = [torch.ones(2, device=device), torch.ones(2, device=device)]
    assert not _tensor_containers_are_byte_equal(left, right)


@pytest.mark.parametrize("device", DEVICES)
def test_byte_equal_list_vs_tuple_is_false(device: str):
    # list / tuple are distinct container types even with identical contents.
    left = [torch.ones(2, device=device)]
    right = (torch.ones(2, device=device),)
    assert not _tensor_containers_are_byte_equal(left, right)


@pytest.mark.parametrize("device", DEVICES)
def test_byte_equal_mismatched_container_kinds_is_false(device: str):
    # A tensor on one side, a container on the other.
    tensor = torch.ones(2, device=device)
    assert not _tensor_containers_are_byte_equal(tensor, [tensor])
    assert not _tensor_containers_are_byte_equal([tensor], tensor)
    assert not _tensor_containers_are_byte_equal({"x": tensor}, [tensor])


def test_byte_equal_non_tensor_leaf_is_false():
    # Any non-tensor leaf makes an object a non-tensor-container -> False,
    # even when the two sides are otherwise structurally identical / equal.
    assert not _tensor_containers_are_byte_equal({"n": 7}, {"n": 7})
    assert not _tensor_containers_are_byte_equal([None], [None])
    assert not _tensor_containers_are_byte_equal(("x",), ("x",))


def test_byte_equal_bare_non_containers_is_false():
    # Bare scalars / strings / None are not tensor-containers.
    assert not _tensor_containers_are_byte_equal(7, 7)
    assert not _tensor_containers_are_byte_equal("a", "a")
    assert not _tensor_containers_are_byte_equal(None, None)
