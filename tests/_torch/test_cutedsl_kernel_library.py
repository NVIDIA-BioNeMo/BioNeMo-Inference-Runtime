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

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_bionemo._torch import _cutedsl_kernel_library as library_runtime


class _LibraryExecutable(library_runtime.CuTeDSLKernelLibraryExecutable):
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return "launched"


def test_populate_compiled_cache_from_library(monkeypatch):
    family = object()
    library = SimpleNamespace(example=family)
    monkeypatch.setattr(library_runtime, "_kernel_library", library)

    cache = {}
    factory_calls = []

    def factory(loaded_library, loaded_family):
        factory_calls.append((loaded_library, loaded_family))
        return _LibraryExecutable()

    executable = library_runtime.populate_compiled_cache_from_library(cache, ("variant",), "example", factory)
    cached = library_runtime.populate_compiled_cache_from_library(cache, ("variant",), "example", factory)

    assert cached is executable
    assert cache[("variant",)] is executable
    assert factory_calls == [(library, family)]


def test_launch_library_executable_does_not_require_tvm_ffi():
    executable = _LibraryExecutable()

    result = library_runtime.launch_compiled_kernel(executable, 1, 2, 3)

    assert result == "launched"
    assert executable.calls == [(1, 2, 3)]


@pytest.mark.parametrize("qkv_packed", [False, True], ids=["separate", "packed"])
def test_triangle_attention_uses_library_when_source_is_missing(monkeypatch, qkv_packed):
    try:
        importlib.import_module("tensorrt_bionemo.libs._cutedsl_kernels")
    except ImportError:
        pytest.skip("_cutedsl_kernels extension is not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    major, minor = torch.cuda.get_device_capability()
    sm_version = major * 10 + minor
    if sm_version not in (80, 86, 89, 90):
        pytest.skip(f"no directly launchable D32 CUBIN for SM{sm_version}")

    from tensorrt_bionemo._torch.attention_backend import (
        AttentionMetadata,
        TriangleAttentionCuTeLeftMask,
        TriangleAttentionCuTeLeftMaskMetadata,
        VanillaTriangleAttention,
    )
    from tensorrt_bionemo._torch.attention_backend.triangle_attention import _config as triangle_config
    from tests._torch import make_left_aligned_mask

    def missing_source(_implementation):
        raise ModuleNotFoundError("CuTeDSL kernel source removed")

    monkeypatch.setattr(triangle_config, "resolve_implementation", missing_source)
    monkeypatch.setattr(library_runtime, "_kernel_library", None)
    TriangleAttentionCuTeLeftMask._compiled_cache.clear()

    torch.manual_seed(42)
    B, I, J, H, D = 1, 4, 13, 2, 32
    dtype = torch.bfloat16
    device = torch.device("cuda")
    if qkv_packed:
        qkv = torch.randn(B, I, J, 3, H * D, dtype=dtype, device=device)
        q = qkv[..., 0, :]
        k = qkv[..., 1, :]
        v = qkv[..., 2, :]
    else:
        q = torch.randn(B, I, J, H * D, dtype=dtype, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
    pair_bias = torch.randn(B, H, J, J, dtype=dtype, device=device)
    binary_mask = make_left_aligned_mask(B, I, J, dtype=torch.float32, device=device)
    actual_s_kv = binary_mask.sum(dim=-1).to(torch.int32)
    additive_mask = (1.0 - binary_mask) * -1e9

    vanilla = VanillaTriangleAttention(0, H, D, num_kv_heads=H)
    expected = vanilla.forward(
        q,
        k,
        v,
        biases=[additive_mask.unsqueeze(-2).unsqueeze(-2), pair_bias],
        metadata=AttentionMetadata(),
    )

    backend = TriangleAttentionCuTeLeftMask(0, H, D, num_kv_heads=H)
    metadata = TriangleAttentionCuTeLeftMaskMetadata()
    metadata.qkv_packed = qkv_packed
    actual = backend.forward(
        q,
        k,
        v,
        biases=[actual_s_kv, pair_bias],
        metadata=metadata,
    )

    assert len(TriangleAttentionCuTeLeftMask._compiled_cache) == 1
    assert isinstance(
        next(iter(TriangleAttentionCuTeLeftMask._compiled_cache.values())),
        library_runtime.CuTeDSLKernelLibraryExecutable,
    )
    torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-2)
