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
"""Tests for the container-tensor-shape extraction helpers."""

import torch

from tensorrt_bionemo._torch.graph_optimization.config import CUDAGraphOptimizationConfig
from tensorrt_bionemo._torch.graph_optimization.cuda_graph.runtime import CUDAGraphOptimizationTracker


def test_extract_container_tensor_shapes_single_tensor():
    """A lone positional tensor produces a one-entry ``{path -> dim-lengths}``
    map: it roots at walk path ``arg0`` and its value is a 1-D int tensor of the
    tensor's shape."""
    tracker = CUDAGraphOptimizationTracker(CUDAGraphOptimizationConfig(), inner_module=None)

    x = torch.zeros(1, 1, 95, 768, device="cuda")
    shapes = tracker._extract_tensor_container_shapes((x,), {})

    assert isinstance(shapes, dict)
    assert len(shapes) == 1

    ((key, value),) = shapes.items()
    assert key == "arg0_shape"
    assert isinstance(value, torch.Tensor)
    assert torch.equal(value, torch.tensor([1, 1, 95, 768], device="cuda"))


def test_extract_container_shape_maps_include_host_metadata():
    tracker = CUDAGraphOptimizationTracker(CUDAGraphOptimizationConfig(), inner_module=None)
    x = torch.zeros(1, 95, 768, device="cuda")

    shapes_device, shapes_host = tracker._extract_tensor_container_shape_maps((x,), {})

    assert shapes_device["arg0_shape"].device == x.device
    assert shapes_host == {"arg0_shape": (1, 95, 768)}


def test_extract_container_tensor_shapes_single_tensor_in_tuple():
    """A single positional tuple holding one tensor produces a one-entry
    ``{path -> dim-lengths}`` map: the tuple roots at walk path ``arg0`` and its
    element at index 0 is appended, so the tensor's path is ``arg0`` and its
    value is a 1-D int tensor of the tensor's shape."""
    tracker = CUDAGraphOptimizationTracker(CUDAGraphOptimizationConfig(), inner_module=None)

    x = torch.zeros(1, 1, 95, 768, device="cuda")
    args = (x,)
    shapes = tracker._extract_tensor_container_shapes(args, {})

    assert isinstance(shapes, dict)
    assert len(shapes) == 1

    ((key, value),) = shapes.items()
    assert key == "arg0_shape"
    assert isinstance(value, torch.Tensor)
    assert torch.equal(value, torch.tensor([1, 1, 95, 768], device="cuda"))
