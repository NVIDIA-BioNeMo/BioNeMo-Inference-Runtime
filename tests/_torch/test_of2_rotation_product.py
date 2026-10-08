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

"""Check rotation fusion broadcasting and arithmetic."""

import pytest
import torch

from bionemo_ir._torch.modules.openfold2.utils import rigid_utils
from bionemo_ir.dsl_kernels.triton.rotation_product import rotation_product


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("layout", ["scalar", "contiguous", "broadcast", "transpose", "offset", "wide_stride", "empty"])
def test_rotation_product(matrix: bool, device: str, layout: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    tail = (3, 3) if matrix else (3,)
    a = torch.randn(2, 7, 3, 3, device=device)
    b = torch.randn(2, 7, *tail, device=device)
    if layout == "scalar":
        a, b = a[0, 0], b[0, 0]
    elif layout == "broadcast":
        a, b = a[:, :1], b[:1]
    elif layout == "transpose":
        a, b = a.transpose(0, 1).transpose(-1, -2), b.transpose(0, 1)
    elif layout == "offset":
        a, b = a[:, 1:], b[:, 1:]
    elif layout == "wide_stride":
        a = a[:1].as_strided((1, 7, 3, 3), (2**32, 9, 3, 1))
        b = b[:1].as_strided((1, 7, *tail), (2**32, *b.stride()[1:]))
    elif layout == "empty":
        a, b = a[:, :0], b[:, :0]
    function = rigid_utils.rot_matmul if matrix else rigid_utils.rot_vec_mul
    with torch.inference_mode():
        if device == "cuda" and layout != "empty":
            assert rotation_product(a, b, matrix=matrix) is not None
        actual = function(a, b)
        monkeypatch.setattr(rigid_utils, "rotation_product", lambda *args, **kwargs: None)
        expected = function(a, b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("matrix", [False, True])
def test_rotation_graph(matrix: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    tail = (3, 3) if matrix else (3,)
    a = torch.randn(13, 3, 3, device="cuda")
    b = torch.randn(13, *tail, device="cuda")
    function = rigid_utils.rot_matmul if matrix else rigid_utils.rot_vec_mul
    with torch.inference_mode():
        function(a, b)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = function(a, b)
        a.add_(0.25)
        b.add_(0.125)
        graph.replay()
        monkeypatch.setattr(rigid_utils, "rotation_product", lambda *args, **kwargs: None)
        expected = function(a, b)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("matrix", [False, True])
def test_rotation_cold_graph(matrix: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from bionemo_ir.dsl_kernels.triton import rotation_product as module

    monkeypatch.setattr(module, "_KERNELS", {})
    monkeypatch.setattr(module, "_LAYOUTS", {})
    tail = (3, 3) if matrix else (3,)
    a = torch.randn(29, 3, 3, device="cuda")
    b = torch.randn(29, *tail, device="cuda")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    function = rigid_utils.rot_matmul if matrix else rigid_utils.rot_vec_mul
    with torch.inference_mode():
        with torch.cuda.graph(graph):
            assert rotation_product(a, b, matrix=matrix) is None
            actual = function(a, b)
        a.add_(0.25)
        b.add_(0.125)
        graph.replay()
        expected = function(a, b)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("matrix", [False, True])
def test_rotation_autocast(matrix: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    tail = (3, 3) if matrix else (3,)
    a = torch.randn(17, 3, 3, device="cuda")
    b = torch.randn(17, *tail, device="cuda")
    function = rigid_utils.rot_matmul if matrix else rigid_utils.rot_vec_mul
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = function(a, b)
        monkeypatch.setattr(rigid_utils, "rotation_product", lambda *args, **kwargs: None)
        expected = function(a, b)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("backend", ["driver", "compiled"])
def test_rotation_stream(matrix: bool, backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from bionemo_ir.dsl_kernels.triton import rotation_product as module

    tail = (3, 3) if matrix else (3,)
    a = torch.randn(37, 3, 3, device="cuda")
    b = torch.randn(37, *tail, device="cuda")
    function = rigid_utils.rot_matmul if matrix else rigid_utils.rot_vec_mul
    with torch.inference_mode():
        function(a, b)
        kernel = module._KERNELS[(a.device.index, 1, matrix)].kernel
        if backend == "driver":
            if kernel.driver is None:
                pytest.skip("CUDA driver launcher unavailable")
        else:
            monkeypatch.setattr(kernel, "_driver", None)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            a.add_(0.25)
            b.add_(0.125)
            actual = function(a, b)
        stream.synchronize()
        monkeypatch.setattr(rigid_utils, "rotation_product", lambda *args, **kwargs: None)
        expected = function(a, b)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("matrix", [False, True])
def test_rotation_cache_capacity(matrix: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from bionemo_ir.dsl_kernels.triton import rotation_product as module

    monkeypatch.setattr(module, "_LAYOUTS", {})
    monkeypatch.setattr(module, "_MAX_LAYOUTS", 2)
    tail = (3, 3) if matrix else (3,)
    a = torch.randn(11, 3, 3, device="cuda")
    b = torch.randn(11, *tail, device="cuda")
    function = rigid_utils.rot_matmul if matrix else rigid_utils.rot_vec_mul
    with torch.inference_mode():
        assert rotation_product(a, b, matrix=matrix) is not None
        retained = tuple(module._LAYOUTS.values())
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = function(a, b)
        for rows in range(12, 20):
            left = torch.randn(rows, 3, 3, device="cuda")
            right = torch.randn(rows, *tail, device="cuda")
            fused = rotation_product(left, right, matrix=matrix)
            assert (fused is not None) == (rows == 12)
        assert len(module._LAYOUTS) == 2
        assert next(iter(module._LAYOUTS.values())) is retained[0]
        assert rotation_product(a, b, matrix=matrix) is not None
        fallback = function(left, right)
        a.add_(0.25)
        b.add_(0.125)
        graph.replay()
        monkeypatch.setattr(rigid_utils, "rotation_product", lambda *args, **kwargs: None)
        torch.testing.assert_close(fallback, function(left, right), rtol=0, atol=0)
        torch.testing.assert_close(actual, function(a, b), rtol=0, atol=0)
