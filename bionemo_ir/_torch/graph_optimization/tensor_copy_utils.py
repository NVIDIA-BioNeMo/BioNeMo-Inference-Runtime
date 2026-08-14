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
"""Clone, copy, compare, and release tensors in nested containers."""

from typing import Any

import torch


def _clone_tensors(value: Any) -> Any:
    """Clone every tensor while preserving container structure."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    elif isinstance(value, dict):
        return {k: _clone_tensors(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return type(value)(_clone_tensors(v) for v in value)
    return value


def _copy_tensors_into(dest: Any, src: Any) -> None:
    """Copy tensor leaves in place, preserving destination addresses.

    Source and destination container structures must match.
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


def _assert_equal_but_distinct(original: Any, clone: Any) -> None:
    """Assert that tensor leaves are equal but use distinct storage."""
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
    """Return whether tensors have identical metadata and bytes.

    Raw comparison treats matching NaN bits as equal and distinguishes signed
    zero.
    """
    if left.dtype != right.dtype or left.shape != right.shape or left.device != right.device:
        return False
    if left.numel() == 0:
        return True
    left_bytes = left.contiguous().flatten().view(torch.uint8)
    right_bytes = right.contiguous().flatten().view(torch.uint8)
    return torch.equal(left_bytes, right_bytes)


def _tensor_containers_are_byte_equal(left: Any, right: Any) -> bool:
    """Return whether matching tensor-only containers are byte-equal.

    Container types, lengths, and keys must match; non-tensor leaves fail.
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
    return False


def _delete_tensors_in_container(value: Any) -> None:
    """Release tensor references and clear mutable containers recursively."""
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
