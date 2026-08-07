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
"""Pure, stateless helpers for recursively walking tensor trees (arbitrary
nestings of dict / list / tuple containing tensors): clone every tensor, copy
values in-place, and assert two trees are value-equal but memory-distinct.
"""

from typing import Any

import torch


# ---------------------------------------------------------------------------
# Tensor-tree helpers (pure functions: stateless recursive walks over
# arbitrary nestings of dict / list / tuple containing tensors).
# ---------------------------------------------------------------------------
def _clone_tensors(value: Any) -> Any:
    """Recursively walk ``value``, returning a copy in which every tensor is
    cloned and all container structure (dict / list / tuple) is preserved.
    Non-tensor leaves are returned unchanged.
    """
    if isinstance(value, torch.Tensor):
        return value.clone()
    elif isinstance(value, dict):
        return {k: _clone_tensors(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return type(value)(_clone_tensors(v) for v in value)
    # Non-tensor leaves (None, scalars, callables, ...) are passed through.
    return value


def _copy_tensors_into(dest: Any, src: Any) -> None:
    """Recursively walk ``dest`` and ``src`` in lock-step, copying every source
    tensor's values into the corresponding destination tensor with
    ``dest.copy_(src)`` (in-place, preserving ``dest``'s storage/address).

    ``dest`` and ``src`` must have matching structure (the same nesting, keys,
    lengths, and tensor shapes/dtypes). Non-tensor leaves are left as-is.
    """
    if isinstance(dest, torch.Tensor):
        assert isinstance(src, torch.Tensor), (
            f"structure mismatch: static buffer is a tensor but live input is {type(src)}"
        )
        dest.copy_(src)
    elif isinstance(dest, dict):
        assert dest.keys() == src.keys(), "structure mismatch: static buffer and live input have different keys"
        for k in dest:
            _copy_tensors_into(dest[k], src[k])
    elif isinstance(dest, (list, tuple)):
        assert len(dest) == len(src), "structure mismatch: static buffer and live input have different lengths"
        for d, s in zip(dest, src, strict=True):
            _copy_tensors_into(d, s)
    # Non-tensor leaves: nothing to copy.


def _assert_equal_but_distinct(original: Any, clone: Any) -> None:
    """Recursively assert that ``clone`` mirrors ``original`` in value but not
    in memory: every tensor is element-wise equal yet backed by a different
    storage (distinct ``data_ptr`` and a distinct Python object). Container
    structure is walked in lock-step; non-tensor leaves are not checked.
    """
    if isinstance(original, torch.Tensor):
        assert isinstance(clone, torch.Tensor), f"expected a tensor clone, got {type(clone)}"
        assert torch.equal(original, clone), "cloned tensor value differs from the original"
        assert original is not clone, "cloned tensor is the same object as the original"
        assert original.data_ptr() != clone.data_ptr(), "cloned tensor shares storage (same address) with the original"
    elif isinstance(original, dict):
        assert original.keys() == clone.keys()
        for k in original:
            _assert_equal_but_distinct(original[k], clone[k])
    elif isinstance(original, (list, tuple)):
        assert len(original) == len(clone)
        for o, c in zip(original, clone, strict=True):
            _assert_equal_but_distinct(o, c)


def _tensors_byte_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return True iff two tensors are byte-for-byte identical: same dtype, shape,
    and device, and the same underlying bytes.

    Unlike ``torch.equal`` this is a raw-byte comparison, so two NaNs with the
    same bit pattern compare equal while ``+0.0`` and ``-0.0`` compare unequal.
    Tensors on different devices are treated as not equal.
    """
    if left.dtype != right.dtype or left.shape != right.shape or left.device != right.device:
        return False
    if left.numel() == 0:
        return True
    left_bytes = left.contiguous().flatten().view(torch.uint8)
    right_bytes = right.contiguous().flatten().view(torch.uint8)
    return torch.equal(left_bytes, right_bytes)


def _tensor_containers_are_byte_equal(left: Any, right: Any) -> bool:
    """Return True iff ``left`` and ``right`` are matching *tensor-containers* whose
    tensors are byte-equal at every matching path.

    A *tensor-container* is either a ``torch.Tensor`` or a ``dict`` / ``list`` /
    ``tuple`` whose every value is itself a tensor-container. To return ``True``
    both arguments must:

      * be tensor-containers (any leaf that is not a tensor or one of these
        containers makes its object non-tensor-container -> ``False``; note this is
        stricter than ``_assert_equal_but_distinct``, which ignores non-tensor
        leaves), and
      * have identical container structure (same container type, length, and
        dict keys), and
      * hold byte-equal tensors at every tensor leaf (see
        :func:`_tensors_byte_equal`).
    """
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and _tensors_byte_equal(left, right)
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            return False
        return all(_tensor_containers_are_byte_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_tensor_containers_are_byte_equal(lhs, rhs) for lhs, rhs in zip(left, right, strict=True))
    # Not a tensor-container (scalar, str, None, callable, ...).
    return False


def _delete_tensors_in_container(value: Any) -> None:
    """Recursively walk ``value`` and delete every tensor it contains, dropping
    the container's references so the tensors' storage can be reclaimed.

    ``value`` is a tensor or an arbitrary nesting of dict / list / tuple whose
    values are such objects. Children are processed first, then mutable
    containers (dict / list) are emptied; tuples are immutable, so their tensor
    elements are released only when the tuple itself is dropped. Non-tensor
    leaves (None, scalars, callables, ...) are left untouched.
    """
    if isinstance(value, torch.Tensor):
        del value
    elif isinstance(value, dict):
        for v in value.values():
            _delete_tensors_in_container(v)
        value.clear()
    elif isinstance(value, (list, tuple)):
        for v in value:
            _delete_tensors_in_container(v)
        if isinstance(value, list):
            value.clear()
