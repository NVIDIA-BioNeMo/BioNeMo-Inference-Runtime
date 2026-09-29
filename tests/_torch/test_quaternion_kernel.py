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
"""Verify fused quaternion arithmetic and dispatch."""

from unittest.mock import patch

import pytest
import torch

import bionemo_ir._torch.layers.random_augmentation as augmentation
from bionemo_ir._torch.layers.random_augmentation import _quaternion_components_to_matrix, quaternion_to_matrix
from bionemo_ir.dsl_kernels.triton.quaternion_rotation import _quaternion_rotation_kernel


def reference(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    return _quaternion_components_to_matrix(r, i, j, k, 2.0 / (quaternions * quaternions).sum(-1))


@pytest.mark.parametrize("shape", [(4,), (0, 4), (1, 4), (5, 4), (2, 3, 4), (2048, 4)])
def test_quaternion_kernel_matches(shape: tuple[int, ...]) -> None:
    torch.manual_seed(19)
    quaternions = torch.randn(shape, device="cuda")
    expected = reference(quaternions)
    with patch.object(augmentation, "quaternion_matrix", wraps=augmentation.quaternion_matrix) as fused:
        actual = quaternion_to_matrix(quaternions)
    fused.assert_called_once_with(quaternions)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    identity = torch.eye(3, device="cuda").expand_as(actual)
    torch.testing.assert_close(actual @ actual.transpose(-1, -2), identity, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_quaternion_dtype_fallback(dtype: torch.dtype) -> None:
    q = torch.randn(5, 4, device="cuda", dtype=dtype)
    torch.testing.assert_close(quaternion_to_matrix(q), reference(q), atol=0, rtol=0)


def test_quaternion_views_fallback() -> None:
    noncontiguous = torch.randn(4, 5, device="cuda").T
    unaligned = torch.randn(21, device="cuda")[1:].reshape(5, 4)
    for q in (noncontiguous, unaligned):
        torch.testing.assert_close(quaternion_to_matrix(q), reference(q), atol=0, rtol=0)


def test_quaternion_nonfinite() -> None:
    q = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [float("inf"), 1.0, 0.0, 0.0], [float("nan"), 1.0, 0.0, 0.0]], device="cuda"
    )
    torch.testing.assert_close(quaternion_to_matrix(q), reference(q), atol=0, rtol=0, equal_nan=True)


def test_quaternion_graph_stream() -> None:
    q = torch.randn(5, 4, device="cuda")
    quaternion_to_matrix(q)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        output = quaternion_to_matrix(q)
    torch.cuda.current_stream().wait_stream(side)
    torch.testing.assert_close(output, reference(q), atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = quaternion_to_matrix(q)
    q.add_(0.1)
    graph.replay()
    torch.testing.assert_close(output, reference(q), atol=0, rtol=0)


def test_quaternion_launcher_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    kernel = _quaternion_rotation_kernel(torch.cuda.current_device()).kernel
    monkeypatch.setattr(kernel, "_driver", None)
    for rows in (1, 16, 127, 128, 129, 2049):
        q = torch.randn(rows, 4, device="cuda")
        torch.testing.assert_close(quaternion_to_matrix(q), reference(q), atol=0, rtol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_quaternion_device_cache() -> None:
    for device in (1, 0, 1):
        q = torch.randn(129, 4, device=f"cuda:{device}")
        with torch.cuda.device(1 - device):
            actual = quaternion_to_matrix(q)
            assert torch.cuda.current_device() == 1 - device
        torch.testing.assert_close(actual, reference(q), atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(0, 4), (2, 0, 4), (127, 4), (128, 4), (129, 4)])
def test_quaternion_inference(shape: tuple[int, ...]) -> None:
    with torch.inference_mode():
        q = torch.randn(shape, device="cuda")
        torch.testing.assert_close(quaternion_to_matrix(q), reference(q), atol=0, rtol=0)


@pytest.mark.parametrize("scale", [1e-20, 1e-18, 1e18, 1e20])
def test_quaternion_extreme_scales(scale: float) -> None:
    q = torch.randn(129, 4, device="cuda") * scale
    torch.testing.assert_close(quaternion_to_matrix(q), reference(q), atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("normalize", [False, True])
def test_augmentation_rng(monkeypatch: pytest.MonkeyPatch, normalize: bool) -> None:
    generator = torch.Generator(device="cuda").manual_seed(17)
    x = torch.randn(2, 3, 7, 3, device="cuda")
    mask = torch.randint(0, 2, (2, 7), device="cuda")
    state = generator.get_state()
    actual = augmentation.centre_random_augmentation(
        x, mask, generator=generator, normalize_quaternions_first=normalize
    )
    after = generator.get_state()
    generator.set_state(state)
    monkeypatch.setattr(augmentation, "quaternion_to_matrix", reference)
    expected = augmentation.centre_random_augmentation(
        x, mask, generator=generator, normalize_quaternions_first=normalize
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(generator.get_state(), after)
