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
"""Unit tests for the shared precompiled-kernel runtime."""

from __future__ import annotations

import importlib
from importlib.machinery import EXTENSION_SUFFIXES
from types import SimpleNamespace

import pytest
import torch

from bionemo_ir._torch import _cutedsl_kernel_library as library_runtime


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


def test_missing_family_reports_unavailable(monkeypatch):
    monkeypatch.setattr(library_runtime, "_kernel_library", SimpleNamespace())

    with pytest.raises(library_runtime.CuTeDSLKernelLibraryUnavailable):
        library_runtime.populate_compiled_cache_from_library({}, ("variant",), "absent", lambda *_: None)


@pytest.mark.parametrize(
    ("extension", "sources"),
    [(True, True), (True, False), (False, True)],
    ids=["both", "wheel", "checkout"],
)
def test_require_kernel_backend_accepts_either_path(monkeypatch, extension, sources):
    monkeypatch.setattr(library_runtime, "_extension_installed", lambda: extension)
    monkeypatch.setattr(library_runtime, "_kernel_sources_installed", lambda: sources)

    library_runtime.require_kernel_backend()


def test_require_kernel_backend_rejects_a_build_with_neither(monkeypatch):
    monkeypatch.setattr(library_runtime, "_extension_installed", lambda: False)
    monkeypatch.setattr(library_runtime, "_kernel_sources_installed", lambda: False)

    with pytest.raises(library_runtime.CuTeDSLKernelLibraryUnavailable, match="No CuTeDSL kernel backend"):
        library_runtime.require_kernel_backend()


def test_kernel_sources_probe_ignores_the_package_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(library_runtime, "_KERNEL_SOURCE_DIR", tmp_path)
    (tmp_path / "__init__.py").touch()
    assert not library_runtime._kernel_sources_installed()

    # Any non-marker filename does; naming a real private kernel here would put
    # that name in a file the public sync publishes.
    (tmp_path / "kernel.py").touch()
    assert library_runtime._kernel_sources_installed()


def test_extension_probe_matches_the_built_filename(monkeypatch, tmp_path):
    """The probe must name the extension exactly as ``setup.py`` writes it."""
    monkeypatch.setattr(library_runtime, "_KERNEL_LIBRARY_DIR", tmp_path)
    assert not library_runtime._extension_installed()

    (tmp_path / f"{library_runtime._KERNEL_LIBRARY_STEM}{EXTENSION_SUFFIXES[0]}").touch()
    assert library_runtime._extension_installed()


def test_launch_library_executable_does_not_require_tvm_ffi():
    executable = _LibraryExecutable()

    result = library_runtime.launch_compiled_kernel(executable, 1, 2, 3)

    assert result == "launched"
    assert executable.calls == [(1, 2, 3)]


@pytest.mark.parametrize("qkv_packed", [False, True], ids=["separate", "packed"])
def test_triangle_attention_uses_library_when_source_is_missing(monkeypatch, qkv_packed):
    try:
        importlib.import_module("bionemo_ir.libs._cutedsl_kernels")
    except ImportError:
        pytest.skip("_cutedsl_kernels extension is not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    major, minor = torch.cuda.get_device_capability()
    sm_version = major * 10 + minor
    if sm_version not in (80, 86, 89, 90):
        pytest.skip(f"no directly launchable D32 CUBIN for SM{sm_version}")

    from bionemo_ir._torch.attention_backend import (
        AttentionMetadata,
        TriangleAttentionCuTeLeftMask,
        TriangleAttentionCuTeLeftMaskMetadata,
        VanillaTriangleAttention,
    )
    from bionemo_ir._torch.attention_backend.triangle_attention import _config as triangle_config
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


def test_tensor_views_reject_the_wrong_rank():
    library = SimpleNamespace(
        Tensor1View=lambda *args: args,
        Tensor2View=lambda *args: args,
        Tensor3View=lambda *args: args,
        Tensor4View=lambda *args: args,
    )
    rank2 = torch.zeros(2, 3)
    rank3 = torch.zeros(2, 3, 4)

    with pytest.raises(ValueError):
        library_runtime.tensor_s1_d0(library, rank2)
    with pytest.raises(ValueError):
        library_runtime.tensor_s2_d1(library, rank3)
    with pytest.raises(ValueError):
        library_runtime.tensor_s3_d2(library, rank2)
    with pytest.raises(ValueError):
        library_runtime.tensor_s4_d3(library, rank2)


def test_tensor_views_carry_shapes_strides_and_device():
    library = SimpleNamespace(
        Tensor1View=lambda data, shape, strides, device: ("1d", shape, device),
        Tensor2View=lambda data, shape, strides, device: ("2d", shape, strides, device),
        Tensor3View=lambda data, shape, strides, device: ("3d", shape, strides, device),
        Tensor4View=lambda data, shape, strides, device: ("4d", shape, strides, device),
    )
    tensor2 = torch.zeros(2, 3)
    transposed_tensor2 = torch.zeros(3, 2).T
    tensor4 = torch.zeros(2, 3, 4, 5)
    cpu = -1  # get_device() on a CPU tensor, matching the library's UNKNOWN_DEVICE

    assert library_runtime.tensor_s1_d0(library, torch.zeros(7)) == ("1d", (7,), cpu)
    assert library_runtime.tensor_s2_d1(library, tensor2) == ("2d", (2, 3), (3,), cpu)
    assert library_runtime.tensor_s2_d1(library, transposed_tensor2, dynamic_stride_dim=1) == (
        "2d",
        (2, 3),
        (2,),
        cpu,
    )
    assert library_runtime.tensor_s3_d2(library, tensor4) == (
        "3d",
        (2, 3, 4),
        tensor4.stride()[:2],
        cpu,
    )
    assert library_runtime.tensor_s4_d3(library, tensor4) == (
        "4d",
        (2, 3, 4, 5),
        tensor4.stride()[:3],
        cpu,
    )


@pytest.mark.parametrize("dynamic_stride_dim", [-1, 2])
def test_tensor_s2_d1_rejects_invalid_dynamic_stride_dimension(dynamic_stride_dim):
    library = SimpleNamespace(Tensor2View=lambda *args: args)

    with pytest.raises(ValueError, match="dynamic_stride_dim"):
        library_runtime.tensor_s2_d1(library, torch.zeros(2, 3), dynamic_stride_dim)


def test_tensor_s2_d1_requires_the_other_dimension_to_be_contiguous():
    library = SimpleNamespace(Tensor2View=lambda *args: args)
    row_major = torch.zeros(2, 3)
    column_major = torch.zeros(3, 2).T

    with pytest.raises(ValueError, match="stride for dimension 0"):
        library_runtime.tensor_s2_d1(library, row_major, dynamic_stride_dim=1)
    with pytest.raises(ValueError, match="stride for dimension 1"):
        library_runtime.tensor_s2_d1(library, column_major, dynamic_stride_dim=0)
