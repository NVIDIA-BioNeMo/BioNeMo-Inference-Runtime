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

from bionemo_ir._torch.utils.kernel import _cutedsl_kernel_library as library_runtime
from tests._torch import require_cubin_library


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
        kernel_library = importlib.import_module("bionemo_ir.libs._cutedsl_kernels")
    except ImportError:
        pytest.skip("_cutedsl_kernels extension is not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    major, minor = torch.cuda.get_device_capability()
    sm_version = major * 10 + minor
    if sm_version not in (80, 86, 89, 90, 100, 103):
        pytest.skip(f"no directly launchable D32 CUBIN for SM{sm_version}")

    from bionemo_ir._torch.attention_backend import (
        AttentionMetadata,
        TriangleAttentionCuTeLeftMask,
        TriangleAttentionCuTeLeftMaskMetadata,
        VanillaTriangleAttention,
    )
    from bionemo_ir._torch.attention_backend.triangle_attention import _config as triangle_config
    from tests._torch import make_left_aligned_mask

    B, I, J, H, D = 1, 4, 13, 2, 32
    bucket = triangle_config.get_nearest_bucket(sm_version, D, int(round((I * J) ** 0.5)))
    try:
        config = kernel_library.triangle_attention.make_kernel_config(
            sm_version,
            D,
            bucket,
            kernel_library.triangle_attention.DType.BFLOAT16,
            qkv_packed,
        )
    except (RuntimeError, TypeError, ValueError):
        pytest.skip(f"the installed extension has no D32 CUBIN for SM{sm_version}")
    if not config.spec.supports_direct_launch:
        pytest.skip(f"the installed SM{sm_version} D32 CUBIN has no direct launcher")

    def missing_source(_implementation):
        raise ModuleNotFoundError("CuTeDSL kernel source removed")

    monkeypatch.setattr(triangle_config, "resolve_implementation", missing_source)
    monkeypatch.setattr(library_runtime, "_kernel_library", None)
    TriangleAttentionCuTeLeftMask._compiled_cache.clear()

    torch.manual_seed(42)
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


@pytest.mark.parametrize("operand", ["q", "k", "v", "output"])
def test_triangle_cubin_rejects_wrong_static_head_dim(operand):
    library = pytest.importorskip("bionemo_ir.libs._cutedsl_kernels")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    sm_version = major * 10 + minor
    if sm_version not in (80, 86, 89, 90):
        pytest.skip(f"no directly launchable D32 CUBIN for SM{sm_version}")

    from bionemo_ir._torch.attention_backend.triangle_attention import _cubin as triangle_cubin

    batch, i_dim, seqlen, heads, head_dim = 1, 2, 13, 2, 32
    batch_times_i = batch * i_dim
    dtype = torch.bfloat16
    executable = triangle_cubin.TriangleAttentionCubinExecutable(
        library,
        library.triangle_attention,
        sm_version,
        head_dim,
        seqlen,
        dtype,
        False,
    )
    qkv = torch.zeros(batch_times_i, seqlen, heads, head_dim, dtype=dtype, device="cuda")
    args = {
        "q": qkv,
        "k": qkv,
        "v": qkv,
        "bias": torch.zeros(batch, heads, seqlen, seqlen, dtype=dtype, device="cuda"),
        "actual_s_kv": torch.full((batch_times_i,), seqlen, dtype=torch.int32, device="cuda"),
        "output": torch.zeros_like(qkv),
        "lse": torch.zeros(batch_times_i, seqlen, heads, dtype=torch.float32, device="cuda"),
    }
    tensor = args[operand]
    args[operand] = torch.zeros(*tensor.shape[:-1], head_dim + 1, dtype=dtype, device="cuda")

    with pytest.raises(ValueError, match=rf"{operand} static tail"):
        executable(
            args["q"],
            args["k"],
            args["v"],
            args["bias"],
            args["actual_s_kv"],
            args["output"],
            args["lse"],
            1.0,
            1.0,
            i_dim,
        )


def _triangle_backend(head_dim: int = 32):
    from bionemo_ir._torch.attention_backend.triangle_attention import cutedsl as triangle_cutedsl

    backend = object.__new__(triangle_cutedsl.TriangleAttentionCuTeLeftMask)
    backend.layer_idx = 0
    backend.num_heads = 2
    backend.num_kv_heads = 2
    backend.head_dim = head_dim
    backend._sm_version = 100
    backend._last_executable = None
    backend._last_variant = None
    backend._last_force_cubin = None
    return backend, triangle_cutedsl


def _triangle_inputs(head_dim: int = 32):
    batch_size, i_dim, seqlen, num_heads = 1, 2, 5, 2
    shape = (batch_size, i_dim, seqlen, num_heads, head_dim)
    q = torch.randn(shape, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    actual_s_kv = torch.full((batch_size, i_dim), seqlen, dtype=torch.int32)
    bias = torch.randn(batch_size, num_heads, seqlen, seqlen, dtype=q.dtype)
    return q, k, v, actual_s_kv, bias


@pytest.mark.parametrize("operand", ["K", "V", "pair_bias"])
def test_triangle_attention_rejects_mixed_input_dtypes(operand):
    backend, _ = _triangle_backend()
    q, k, v, actual_s_kv, bias = _triangle_inputs()
    replacements = {
        "K": k.float(),
        "V": v.float(),
        "pair_bias": bias.float(),
    }
    k = replacements["K"] if operand == "K" else k
    v = replacements["V"] if operand == "V" else v
    bias = replacements["pair_bias"] if operand == "pair_bias" else bias

    with pytest.raises(TypeError, match=operand):
        backend._prepare_launch_inputs(q, k, v, actual_s_kv, bias, False, None, None)


def test_triangle_attention_reuses_pre_padded_pair_bias():
    backend, _ = _triangle_backend()
    q, k, v, actual_s_kv, bias = _triangle_inputs()
    padded_bias = torch.zeros(*bias.shape[:-1], 8, dtype=bias.dtype)
    padded_bias[..., : bias.shape[-1]] = bias

    launch_inputs = backend._prepare_launch_inputs(q, k, v, actual_s_kv, padded_bias, False, None, None)

    assert launch_inputs.bias.shape == padded_bias.shape
    assert launch_inputs.bias.data_ptr() == padded_bias.data_ptr()


def test_triangle_attention_rejects_static_inner_stride_mismatches():
    backend, _ = _triangle_backend()
    q, k, v, actual_s_kv, bias = _triangle_inputs()
    q = torch.randn(1, 2, 5, 32, 2, dtype=q.dtype).transpose(-1, -2)

    with pytest.raises(ValueError, match="Q inner strides"):
        backend._prepare_launch_inputs(q, k, v, actual_s_kv, bias, False, None, None)

    q = torch.randn_like(k)
    output = torch.empty(2, 5, 32, 2, dtype=q.dtype).transpose(-1, -2)
    with pytest.raises(ValueError, match="output inner strides"):
        backend._prepare_launch_inputs(q, k, v, actual_s_kv, bias, False, output, None)

    output_lse = torch.empty(2, 2, 5, 1).permute(0, 2, 1, 3)
    with pytest.raises(ValueError, match="output_lse inner strides"):
        backend._prepare_launch_inputs(q, k, v, actual_s_kv, bias, False, None, output_lse)


def test_triangle_attention_rejects_shared_qkv_stride_mismatches():
    backend, _ = _triangle_backend()
    q, _, v, actual_s_kv, bias = _triangle_inputs()
    k = torch.randn(1, 2, 6, 2, 32, dtype=q.dtype)[:, :, :5]

    with pytest.raises(ValueError, match="K outer strides must match Q"):
        backend._prepare_launch_inputs(q, k, v, actual_s_kv, bias, False, None, None)


def test_triangle_attention_writes_unpadded_output_buffer(monkeypatch):
    backend, triangle_cutedsl = _triangle_backend(head_dim=31)
    q, k, v, actual_s_kv, bias = _triangle_inputs(head_dim=31)
    output = torch.full((2, 5, 2, 31), torch.nan, dtype=q.dtype)

    monkeypatch.setattr(backend, "_get_executable", lambda *args: object())

    def launch_stub(executable, *args):
        del executable
        args[5].fill_(3)

    monkeypatch.setattr(triangle_cutedsl, "launch_compiled_kernel", launch_stub)
    result = backend.forward(q, k, v, [actual_s_kv, bias], output=output)

    assert result.reshape_as(output).data_ptr() == output.data_ptr()
    torch.testing.assert_close(output, torch.full_like(output, 3))


def test_triangle_attention_cache_separates_source_and_forced_cubin(monkeypatch):
    backend, triangle_cutedsl = _triangle_backend()
    variant = triangle_cutedsl._TriangleAttentionVariant(torch.bfloat16, 32, 0, False)
    config = SimpleNamespace(arch="sm100", cache_identity=("fixed-sm100",))
    source_executable = object()
    cubin_executable = object()
    cache = {
        backend._source_cache_key(variant, config): source_executable,
        backend._cubin_cache_key(variant): cubin_executable,
    }
    monkeypatch.setattr(triangle_cutedsl.TriangleAttentionCuTeLeftMask, "_compiled_cache", cache)
    monkeypatch.setattr(backend, "_resolve_source_kernel", lambda *_: (config, object()))
    monkeypatch.setattr(
        triangle_cutedsl,
        "load_source_module",
        lambda _: SimpleNamespace(compile_triangle_attention_source=lambda *args: None),
    )

    monkeypatch.setenv(triangle_cutedsl.FORCE_CUBIN_ENV, "1")
    assert backend._get_executable(variant, object(), 8, 1.0, 1) is cubin_executable

    monkeypatch.delenv(triangle_cutedsl.FORCE_CUBIN_ENV)
    assert backend._get_executable(variant, object(), 8, 1.0, 1) is source_executable


def test_triangle_attention_forward_rechecks_forced_mode(monkeypatch):
    backend, triangle_cutedsl = _triangle_backend()
    q, k, v, actual_s_kv, bias = _triangle_inputs()
    source_executable = object()
    cubin_executable = object()
    launches = []

    def resolve_executable(*args):
        del args
        return cubin_executable if backend.force_cubin() else source_executable

    def launch_stub(executable, *args):
        launches.append(executable)
        args[5].zero_()

    monkeypatch.setattr(backend, "_get_executable", resolve_executable)
    monkeypatch.setattr(triangle_cutedsl, "launch_compiled_kernel", launch_stub)
    monkeypatch.delenv(triangle_cutedsl.FORCE_CUBIN_ENV, raising=False)
    backend.forward(q, k, v, [actual_s_kv, bias])
    monkeypatch.setenv(triangle_cutedsl.FORCE_CUBIN_ENV, "1")
    backend.forward(q, k, v, [actual_s_kv, bias])

    assert launches == [source_executable, cubin_executable]


def test_triangle_attention_disk_cache_key_includes_source_config():
    backend, triangle_cutedsl = _triangle_backend()
    variant = triangle_cutedsl._TriangleAttentionVariant(torch.bfloat16, 32, 0, False)
    first = SimpleNamespace(arch="sm100", cache_identity=("first",))
    second = SimpleNamespace(arch="sm100", cache_identity=("second",))

    assert backend._disk_cache_key(variant, first) != backend._disk_cache_key(variant, second)


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
        library_runtime.tensor_s3_d2_static(library, rank3)
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
    assert library_runtime.tensor_s3_d2_static(library, tensor4) == (
        "4d",
        (2, 3, 4, 5),
        tensor4.stride()[:3],
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


def test_failed_extension_import_is_not_retried(monkeypatch):
    """A retry re-runs the extension's nanobind init, which aborts the process.

    Python drops a module whose init raised from ``sys.modules``, so the second
    ``import_module`` re-enters ``PyInit__cutedsl_kernels``. nanobind refuses
    the duplicate enum registration with ``Fatal Python error: Aborted`` rather
    than an exception, taking the whole pytest-xdist worker with it.
    """
    monkeypatch.setattr(library_runtime, "_kernel_library", None)
    monkeypatch.setattr(library_runtime, "_kernel_library_error", None)
    attempts = []

    def failing_import(name):
        attempts.append(name)
        raise ImportError(name)

    monkeypatch.setattr(library_runtime.importlib, "import_module", failing_import)

    for _ in range(3):
        with pytest.raises(library_runtime.CuTeDSLKernelLibraryUnavailable):
            library_runtime._load_kernel_library()

    assert attempts == [library_runtime._KERNEL_LIBRARY_MODULE]


def test_dlopen_failure_is_reported_as_unavailable(monkeypatch):
    """A failed dlopen raises OSError, not ImportError -- and is not retried."""
    monkeypatch.setattr(library_runtime, "_kernel_library", None)
    monkeypatch.setattr(library_runtime, "_kernel_library_error", None)
    attempts = []

    def failing_import(name):
        attempts.append(name)
        raise OSError("libcuda.so.1: cannot open shared object file")

    monkeypatch.setattr(library_runtime.importlib, "import_module", failing_import)

    for _ in range(3):
        with pytest.raises(library_runtime.CuTeDSLKernelLibraryUnavailable):
            library_runtime._load_kernel_library()

    assert attempts == [library_runtime._KERNEL_LIBRARY_MODULE]


def test_require_cubin_library_shares_the_runtime_import_cache(monkeypatch):
    """Two import sites would re-enter PyInit and abort the worker."""
    monkeypatch.setattr(library_runtime, "_kernel_library", None)
    monkeypatch.setattr(library_runtime, "_kernel_library_error", None)
    attempts = []

    def failing_import(name):
        attempts.append(name)
        raise ImportError(name)

    monkeypatch.setattr(library_runtime.importlib, "import_module", failing_import)

    with pytest.raises(library_runtime.CuTeDSLKernelLibraryUnavailable):
        library_runtime._load_kernel_library()
    with pytest.raises(pytest.fail.Exception):
        require_cubin_library()

    assert attempts == [library_runtime._KERNEL_LIBRARY_MODULE]
