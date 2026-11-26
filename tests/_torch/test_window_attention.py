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
"""Windowed atom attention: kernel selection, source and CUBIN numerics, and graph replay."""

import contextlib
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bionemo_ir._torch.attention_backend.window_attention import _config as window_config
from bionemo_ir._torch.attention_backend.window_attention import cutedsl as window_cutedsl
from bionemo_ir._torch.attention_backend.window_attention import (
    get_window_attention_op,
    window_attention,
)
from bionemo_ir._torch.attention_backend.window_attention import ops as window_ops
from bionemo_ir._torch.attention_backend.window_attention._config import config_kernel_abi, dtype_kernel_abi
from bionemo_ir._torch.utils.kernel import CuTeDSLKernelLibraryExecutable, KernelSourceUnavailable
from tests._torch import (
    SM_VERSION,
    cutedsl_test_modes,
    library_runtime,
    require_cubin_family,
    run_cutedsl_test_mode,
    source_module_available,
)

_W, _NK, _H, _D = 32, 128, 4, 32
_PAD = (_NK - _W) // 2
_SOURCE_MODULE = "bionemo_ir._torch.attention_backend.window_attention._source"
MODES = cutedsl_test_modes(_SOURCE_MODULE)
#: The kernel each configured GPU runs.
_KERNEL_ABI_BY_SM = {80: "sm80", 86: "sm80", 89: "sm80", 90: "sm90", 100: "sm80", 103: "sm80", 107: "sm107"}


@pytest.mark.parametrize("symlink", [False, True])
def test_configs_follow_the_package_import_path(tmp_path: Path, symlink: bool) -> None:
    """A sparse package keeps its configs beside the imported module, even with symlinked code."""
    original = tmp_path / "original"
    package = tmp_path / "package"
    original.mkdir()
    package.mkdir()
    source = original / "_config.py"
    source.write_text(Path(window_config.__file__).read_text())
    imported = package / "_config.py"
    if symlink:
        imported.symlink_to(source)
    else:
        imported.write_text(source.read_text())
    (package / "configs").symlink_to(window_config.CONFIGS_DIR, target_is_directory=True)
    config = runpy.run_path(str(imported))
    assert config["CONFIGS_DIR"] == package / "configs"
    for sm, abi in _KERNEL_ABI_BY_SM.items():
        assert config["config_kernel_abi"](sm, _D, _W, _NK) == abi
    assert config["config_kernel_abi"](107, 64, _W, _NK) is None


@pytest.fixture
def selector(monkeypatch: pytest.MonkeyPatch):
    """Resolve the getter for a patched SM on any host, CUDA or not, without compiling or loading a kernel."""
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    monkeypatch.setattr(window_ops.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(window_ops.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(window_ops.torch.cuda, "device", lambda index: contextlib.nullcontext())
    monkeypatch.setattr(window_cutedsl.WindowAttentionCuTe, "executable", lambda self: object())
    monkeypatch.setattr(window_ops, "_ops", {})

    def resolve(sm: int, dtype: torch.dtype = torch.bfloat16, shape: tuple[int, int, int] = (_D, _W, _NK)):
        monkeypatch.setattr(window_ops, "_sm_version", lambda index: sm)
        window_ops._ops.clear()
        return get_window_attention_op(dtype, *shape)

    return resolve


@pytest.fixture
def fresh_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve from nothing: no getter entry, no compiled or loaded kernel, no loaded extension."""
    monkeypatch.setattr(window_ops, "_ops", {})
    monkeypatch.setattr(window_cutedsl.WindowAttentionCuTe, "_compiled_cache", {})
    monkeypatch.setattr(library_runtime, "_kernel_library", None)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("sm", sorted(_KERNEL_ABI_BY_SM))
def test_each_configured_gpu_is_served(selector, sm: int, dtype: torch.dtype) -> None:
    """SM90 runs the WGMMA kernel and every other configured GPU the warp-MMA one."""
    assert selector(sm, dtype) is not None
    assert config_kernel_abi(sm, _D, _W, _NK) == _KERNEL_ABI_BY_SM[sm]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sm107_selects_the_source_shape_without_aliasing_its_device(selector, dtype: torch.dtype) -> None:
    op = selector(107, dtype)
    assert op is not None
    assert op._sm_version == 107
    assert config_kernel_abi(107, _D, _W, _NK) == "sm107"
    assert selector(107, dtype, (64, _W, _NK)) is None


def test_the_sm107_kernel_runs_bf16_and_leaves_fp16_to_the_warp_mma_kernel() -> None:
    assert dtype_kernel_abi("sm107", "bf16") == "sm107"
    assert dtype_kernel_abi("sm107", "fp16") == "sm80"
    for abi in ("sm80", "sm90"):
        assert dtype_kernel_abi(abi, "bf16") == dtype_kernel_abi(abi, "fp16") == abi


@pytest.mark.parametrize(("dtype", "kernel_abi"), [("bf16", "sm107"), ("fp16", "sm80")])
def test_sm107_source_compiles_its_dtypes_kernel_through_the_family_path(
    monkeypatch: pytest.MonkeyPatch, fresh_caches: None, dtype: str, kernel_abi: str
) -> None:
    """The SM107 config's BF16 call compiles the tcgen05 kernel for sm_107a, keyed by the SM count its persistent
    grid bakes in; FP16 compiles the warp-MMA kernel for the device, keyed as before."""
    checks, kernels, options, keys = [], [], [], []
    executable = object()

    def make_kernel(abi: str, *shape: int, num_sms: int | None = None) -> object:
        kernels.append((abi, *shape, num_sms))
        return object()

    def compile_source(compile_kernel, kernel, dtype_str, *rest) -> object:
        return compile_kernel(kernel)

    source = SimpleNamespace(
        can_implement=lambda *args: checks.append(args) or True,
        make_kernel=make_kernel,
        compile_options=lambda abi, base: base + (" --gpu-arch sm_107a" if abi == "sm107" else ""),
        compile_window_attention_source=compile_source,
    )
    monkeypatch.setattr(window_cutedsl, "load_source_module", lambda package: source)
    monkeypatch.setattr(
        window_cutedsl.torch.cuda, "get_device_properties", lambda index: SimpleNamespace(multi_processor_count=212)
    )
    monkeypatch.setattr(window_cutedsl.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(window_cutedsl.WindowAttentionCuTe, "load_from_cache", lambda self, key: keys.append(key))
    monkeypatch.setattr(window_cutedsl.WindowAttentionCuTe, "save_to_cache", lambda self, key, value: keys.append(key))
    monkeypatch.setattr(
        window_cutedsl.WindowAttentionCuTe,
        "compile",
        lambda self, *args, **kwargs: options.append(kwargs) or executable,
    )
    op = window_cutedsl.WindowAttentionCuTe(107, dtype, _D, _W, _NK)
    assert op._load_or_compile_source((107, dtype, _D, _W, _NK)) is executable
    assert checks == [(kernel_abi, dtype, _D, _W, _NK, 107)]
    assert kernels == [(kernel_abi, _D, _W, _NK, 212 if kernel_abi == "sm107" else None)]
    arch = " --gpu-arch sm_107a" if kernel_abi == "sm107" else ""
    assert options == [{"options": "--enable-tvm-ffi" + arch}]
    grid = (212,) if kernel_abi == "sm107" else ()
    assert keys == [("window_attention_cute", 2, 107, kernel_abi, dtype, _D, _W, _NK, *grid)] * 2


@pytest.mark.parametrize("force_cubin", [False, True])
def test_sm107_without_sources_needs_its_own_cubin(
    monkeypatch: pytest.MonkeyPatch, fresh_caches: None, force_cubin: bool
) -> None:
    """A source alias never permits an SM100 image to launch on Rubin."""
    monkeypatch.setattr(window_ops.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(window_ops.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(window_ops.torch.cuda, "device", lambda index: contextlib.nullcontext())
    monkeypatch.setattr(window_ops, "_sm_version", lambda index: 107)
    if force_cubin:
        monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    else:
        monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)

    def missing_source(package: str) -> None:
        raise KernelSourceUnavailable(f"{package} has no sources", name=f"{package}._source")

    targets = []

    def missing_cubin(library, launcher, target_sm: int, *shape) -> None:
        targets.append(target_sm)
        raise library_runtime.CuTeDSLKernelVariantUnavailable("no native Rubin image")

    def populate(cache, key, family, factory):
        assert key[0] == 107 and family == "window_attention"
        return factory(None, None)

    monkeypatch.setattr(window_cutedsl, "load_source_module", missing_source)
    monkeypatch.setattr(window_cutedsl, "WindowAttentionCubinExecutable", missing_cubin)
    monkeypatch.setattr(window_cutedsl, "populate_compiled_cache_from_library", populate)
    if force_cubin:
        with pytest.raises(RuntimeError, match="CUTEDSL_FORCE_CUBIN.*SM107"):
            get_window_attention_op(torch.bfloat16, _D, _W, _NK)
    else:
        assert get_window_attention_op(torch.bfloat16, _D, _W, _NK) is None
    assert targets == [107]


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("sm", sorted(_KERNEL_ABI_BY_SM))
def test_each_configured_kernel_fits_its_gpu(sm: int, dtype: str) -> None:
    source = pytest.importorskip(_SOURCE_MODULE)
    assert source.can_implement(dtype_kernel_abi(_KERNEL_ABI_BY_SM[sm], dtype), dtype, _D, _W, _NK, sm)


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_sm107_resource_check_enforces_its_hard_cap(dtype: str) -> None:
    source = pytest.importorskip(_SOURCE_MODULE)
    assert source.dynamic_smem_bytes("sm80", dtype, _D, _W, _NK) == 26624
    assert source.dynamic_smem_bytes("sm80", dtype, 384, _W, _NK) == 229376
    assert source.can_implement("sm80", dtype, 384, _W, _NK, 107)
    # 303104 fits the compiler's incorrect 334848-byte table entry, but not
    # the device's 232448-byte cap. A larger test would miss that regression.
    assert source.dynamic_smem_bytes("sm80", dtype, 512, _W, _NK) == 303104
    assert not source.can_implement("sm80", dtype, 512, _W, _NK, 107)


@pytest.mark.parametrize("sm", [70, 75, 120, 121])
def test_gpus_without_a_config_decline(selector, sm: int) -> None:
    """GPUs before SM80 and the SM12x consumer parts have no config: the model keeps its attention backend."""
    assert selector(sm) is None


@pytest.mark.parametrize("shape", [(64, 32, 128), (32, 16, 128), (32, 32, 64)])
def test_other_window_shapes_decline(selector, shape: tuple[int, int, int]) -> None:
    assert selector(90, shape=shape) is None


def test_fp32_declines(selector) -> None:
    assert selector(90, torch.float32) is None


def test_selection_follows_the_tensors_device(selector, monkeypatch: pytest.MonkeyPatch) -> None:
    """The SM is read for the tensors' own device: a device before SM80 declines."""
    monkeypatch.setattr(window_ops, "_sm_version", lambda index: {0: 90, 1: 75}[index])
    assert get_window_attention_op(torch.bfloat16, _D, _W, _NK, 0) is not None
    assert get_window_attention_op(torch.bfloat16, _D, _W, _NK, 1) is None


def _requires_kernel() -> None:
    if not torch.cuda.is_available() or config_kernel_abi(SM_VERSION, _D, _W, _NK) is None:
        pytest.skip("no windowed attention kernel for this device")


def _cubin_family_shipped() -> bool:
    try:
        return hasattr(library_runtime._load_kernel_library(), "window_attention")
    except library_runtime.CuTeDSLKernelLibraryError:
        return False


def _cubin_variant_shipped() -> bool:
    """Whether the kernel library embeds this device's bf16 window image.

    Only a missing library, family or image counts as not shipped; any other launcher error, such as an image with
    incompatible launch metadata, fails the calling test.
    """
    if not _cubin_family_shipped():
        return False
    launcher = library_runtime._load_kernel_library().window_attention
    try:
        launcher.make_kernel_config(SM_VERSION, _D, _W, _NK, launcher.DType.BFLOAT16)
    except ValueError as error:
        if str(error).startswith("No embedded window_attention CUBIN"):
            return False
        raise
    return True


def test_forced_cubin_serves_the_packaged_kernel(monkeypatch: pytest.MonkeyPatch, fresh_caches: None) -> None:
    _requires_kernel()
    require_cubin_family("window_attention")
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    if SM_VERSION == 107 and not _cubin_variant_shipped():
        with pytest.raises(RuntimeError, match=f"CUTEDSL_FORCE_CUBIN.*SM{SM_VERSION}"):
            get_window_attention_op(torch.bfloat16, _D, _W, _NK)
        return
    op = get_window_attention_op(torch.bfloat16, _D, _W, _NK)
    assert op is not None
    assert isinstance(op.executable(), CuTeDSLKernelLibraryExecutable)


def test_forced_cubin_without_a_packaged_kernel_raises(monkeypatch: pytest.MonkeyPatch, fresh_caches: None) -> None:
    """A missing CUBIN under CUTEDSL_FORCE_CUBIN is an error, not a quiet return to the attention backend."""
    _requires_kernel()
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")

    def missing(*_args: object) -> object:
        raise library_runtime.CuTeDSLKernelLibraryUnavailable("no window_attention family")

    monkeypatch.setattr(window_cutedsl, "populate_compiled_cache_from_library", missing)
    with pytest.raises(RuntimeError, match="CUTEDSL_FORCE_CUBIN"):
        get_window_attention_op(torch.bfloat16, _D, _W, _NK)


def test_only_a_missing_window_image_counts_as_not_shipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A present but broken image fails the CUBIN tests instead of skipping them as not yet shipped."""

    def broken(*_args: object) -> object:
        raise ValueError("window_attention CUBIN has incompatible launch metadata")

    def missing(*_args: object) -> object:
        raise ValueError("No embedded window_attention CUBIN for SM107, head_dim=32, n_queries=32, n_keys=128, bf16")

    launcher = SimpleNamespace(DType=SimpleNamespace(BFLOAT16=object()), make_kernel_config=broken)
    monkeypatch.setattr(library_runtime, "_load_kernel_library", lambda: SimpleNamespace(window_attention=launcher))
    with pytest.raises(ValueError, match="incompatible launch metadata"):
        _cubin_variant_shipped()
    launcher.make_kernel_config = missing
    assert not _cubin_variant_shipped()
    launcher.make_kernel_config = lambda *_args: object()
    assert _cubin_variant_shipped()
    monkeypatch.setattr(library_runtime, "_load_kernel_library", lambda: SimpleNamespace())
    assert not _cubin_variant_shipped()


def test_a_build_without_kernel_sources_serves_the_cubin_or_declines(
    monkeypatch: pytest.MonkeyPatch, fresh_caches: None
) -> None:
    """Without the sources the packaged CUBIN serves; without it too, the getter declines and the model keeps SDPA."""
    _requires_kernel()
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)

    def missing(package: str):
        raise KernelSourceUnavailable(f"{package} has no sources", name=f"{package}._source")

    monkeypatch.setattr(window_cutedsl, "load_source_module", missing)
    op = get_window_attention_op(torch.bfloat16, _D, _W, _NK)
    if _cubin_family_shipped() and (SM_VERSION != 107 or _cubin_variant_shipped()):
        assert op is not None
        assert isinstance(op.executable(), CuTeDSLKernelLibraryExecutable)
    else:
        assert op is None


def _inputs(samples: int, atoms: int, bias_samples: int, dtype: torch.dtype, column_blocks: bool):
    n = (atoms + _W - 1) // _W * _W
    windows = n // _W
    if column_blocks:  # column blocks of one projection, as the attention layer passes them
        proj = torch.randn(samples, n, 4 * _H * _D, device="cuda", dtype=dtype)
        q, k, v = (proj[..., i * _H * _D : (i + 1) * _H * _D].view(samples, n, _H, _D) for i in range(3))
    else:
        q, k, v = (torch.randn(samples, n, _H, _D, device="cuda", dtype=dtype) for _ in range(3))
    keys = torch.arange(windows, device="cuda")[:, None] * _W - _PAD + torch.arange(_NK, device="cuda")[None, :]
    masked = ((keys < 0) | (keys >= atoms)).float() * -1e9
    bias = torch.randn(bias_samples, windows, _H, _W, _NK, device="cuda") + masked[None, :, None, None, :]
    return q, k, v, bias.to(dtype), n, windows


def _reference(q, k, v, bias, n: int, windows: int) -> torch.Tensor:
    """fp32 attention over gathered key windows; rows outside ``[0, n)`` are zero vectors, and each batch item's
    consecutive samples share its bias."""
    samples = q.shape[0]
    keys = torch.arange(windows, device=q.device)[:, None] * _W - _PAD + torch.arange(_NK, device=q.device)[None, :]
    valid = ((keys >= 0) & (keys < n))[None, :, :, None, None].float()
    kw = k.float()[:, keys.clamp(0, n - 1)] * valid
    vw = v.float()[:, keys.clamp(0, n - 1)] * valid
    qw = q.float().view(samples, windows, _W, _H, _D)
    sample_bias = bias.float().repeat_interleave(samples // bias.shape[0], dim=0)
    logits = torch.einsum("stqhd,stkhd->sthqk", qw, kw) * _D**-0.5 + sample_bias
    out = torch.einsum("sthqk,stkhd->stqhd", torch.softmax(logits, dim=-1), vw)
    return out.reshape(samples, n, _H, _D)


def _run(mode: str, monkeypatch: pytest.MonkeyPatch, operation):
    """Run ``operation`` twice through exactly the ``mode`` path; the mode holds for the rest of the test."""
    if mode == "cubin":
        if SM_VERSION == 107 and not _cubin_variant_shipped():
            pytest.skip("this build has no window attention image for the current SM")
        require_cubin_family("window_attention")
    return run_cutedsl_test_mode(mode, monkeypatch, window_cutedsl.WindowAttentionCuTe, window_cutedsl, operation)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("samples", "atoms", "bias_samples", "column_blocks"),
    [
        (1, 37, 1, True),
        (3, 200, 1, True),
        (3, 200, 3, True),
        (5, 919, 1, False),
        (2, 300, 2, False),
        (6, 200, 2, True),
        (6, 919, 3, False),
        (7, 100, 1, True),
    ],
)
def test_window_attention_matches_fp32_reference(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    samples: int,
    atoms: int,
    bias_samples: int,
    column_blocks: bool,
    dtype: torch.dtype,
) -> None:
    """Every samples-per-item count, power of two or not, picks its own item's bias."""
    _requires_kernel()
    torch.manual_seed(0)
    q, k, v, bias, n, windows = _inputs(samples, atoms, bias_samples, dtype, column_blocks)
    out = _run(mode, monkeypatch, lambda: window_attention(q, k, v, bias, _W, _NK))
    assert out.shape == (samples, n, _H, _D) and out.dtype == dtype and out.is_contiguous()
    torch.testing.assert_close(out.float(), _reference(q, k, v, bias, n, windows), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a_key_half_masked_to_minus_inf_matches_the_reference(
    mode: str, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    """A -inf mask over a whole 64-key half, which the SM90 kernel softmaxes on its own, leaves the other half."""
    _requires_kernel()
    torch.manual_seed(0)
    q, k, v, bias, n, windows = _inputs(2, 6 * _W, 1, dtype, True)
    halves = bias.float().unflatten(-1, (2, _NK // 2))
    # Even windows mask their last 64 keys and odd ones their first 64; every query keeps keys inside [0, n).
    halves[:, 0::2, ..., 1, :] = float("-inf")
    halves[:, 1::2, ..., 0, :] = float("-inf")
    bias = halves.flatten(-2).to(dtype)
    out = _run(mode, monkeypatch, lambda: window_attention(q, k, v, bias, _W, _NK))
    torch.testing.assert_close(out.float(), _reference(q, k, v, bias, n, windows), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("mode", MODES)
def test_window_attention_graph_replay_matches_eager(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _requires_kernel()
    torch.manual_seed(0)
    q, k, v, bias, _, _ = _inputs(5, 919, 1, torch.bfloat16, True)
    eager = _run(mode, monkeypatch, lambda: window_attention(q, k, v, bias, _W, _NK))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        window_attention(q, k, v, bias, _W, _NK)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graphed = window_attention(q, k, v, bias, _W, _NK)
    graph.replay()
    torch.testing.assert_close(graphed, eager, atol=0, rtol=0)
    q.copy_(torch.randn_like(q))
    graph.replay()
    torch.testing.assert_close(graphed, window_attention(q, k, v, bias, _W, _NK), atol=0, rtol=0)


@pytest.mark.parametrize("mode", MODES)
def test_offsets_past_two_to_the_31_elements(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Strides are 64-bit: a second sample and bias item 2^31 elements past the first read the right rows."""
    _requires_kernel()
    stride = 2**31 + 64
    n, windows = 2 * _W, 2
    rows, item = n * _H * _D, windows * _H * _W * _NK
    if torch.cuda.mem_get_info()[0] < 2 * (stride + 3 * rows + item) + (1 << 30):
        pytest.skip("needs about 5 GiB of free device memory")
    torch.manual_seed(3)
    buffer = torch.empty(stride + 3 * rows + item, device="cuda", dtype=torch.bfloat16)
    sample_shape, sample_strides = (2, n, _H, _D), (stride, _H * _D, _D, 1)
    q, k, v = (buffer.as_strided(sample_shape, sample_strides, offset * rows) for offset in range(3))
    bias = buffer.as_strided((2, windows, _H, _W, _NK), (stride, _H * _W * _NK, _W * _NK, _NK, 1), 3 * rows)
    for x in (q, k, v, bias):
        x.copy_(torch.randn(x.shape, device="cuda", dtype=torch.bfloat16))

    out = _run(mode, monkeypatch, lambda: window_attention(q, k, v, bias, _W, _NK))
    contiguous = window_attention(q.contiguous(), k.contiguous(), v.contiguous(), bias.contiguous(), _W, _NK)
    torch.testing.assert_close(out, contiguous, atol=0, rtol=0)


@pytest.mark.parametrize("mode", MODES)
def test_batch_items_with_several_samples_share_their_items_bias(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """B batch items with S samples each (q has B*S sequences) take one bias per item in one launch: equal, bit for
    bit, to running each item alone with its bias shared by its samples."""
    _requires_kernel()
    torch.manual_seed(5)
    items, per, n = 2, 3, 6 * _W
    q, k, v = (torch.randn(items * per, n, _H, _D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    bias = torch.randn(items, n // _W, _H, _W, _NK, device="cuda", dtype=torch.bfloat16)
    launches = 0
    launch = window_cutedsl.launch_compiled_kernel

    def counted(*args: object) -> object:
        nonlocal launches
        launches += 1
        return launch(*args)

    monkeypatch.setattr(window_cutedsl, "launch_compiled_kernel", counted)
    out = _run(mode, monkeypatch, lambda: window_attention(q, k, v, bias, _W, _NK))
    assert launches == 2  # the mode check runs the call twice
    for item in range(items):
        rows = slice(item * per, (item + 1) * per)
        alone = window_attention(q[rows], k[rows], v[rows], bias[item : item + 1], _W, _NK)
        torch.testing.assert_close(out[rows], alone, atol=0, rtol=0)


def test_cubin_launcher_checks_the_layout_it_is_handed(monkeypatch: pytest.MonkeyPatch, fresh_caches: None) -> None:
    """The C++ launcher re-checks shapes and strides: raw pointers reach the device only in the layout compiled for."""
    _requires_kernel()
    if SM_VERSION == 107 and not _cubin_variant_shipped():
        pytest.skip("this build has no window attention image for the current SM")
    require_cubin_family("window_attention")
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    executable = get_window_attention_op(torch.bfloat16, _D, _W, _NK).executable()
    q, k, v, bias, n, _ = _inputs(3, 64, 1, torch.bfloat16, False)
    out = torch.empty_like(q)
    scale = _D**-0.5
    with pytest.raises(ValueError, match="divide the samples"):
        executable(q, k, v, bias.expand(2, -1, -1, -1, -1).contiguous(), out, scale, scale)
    with pytest.raises(ValueError, match="non-positive stride"):
        executable(q, k, v, bias.expand(3, -1, -1, -1, -1), out, scale, scale)
    with pytest.raises(ValueError, match="multiples of 8"):
        executable(q, k, v, bias.as_strided(bias.shape, (1, *bias.stride()[1:])), out, scale, scale)


def test_a_broadcast_bias_takes_the_contiguous_copy() -> None:
    """One item's bias expanded over the batch has a zero stride, which no kernel build reads in place."""
    _requires_kernel()
    torch.manual_seed(1)
    q, k, v, bias, _, _ = _inputs(4, 64, 1, torch.bfloat16, False)
    expected = window_attention(q, k, v, bias.expand(4, -1, -1, -1, -1).contiguous(), _W, _NK)
    actual = window_attention(q, k, v, bias.expand(4, -1, -1, -1, -1), _W, _NK)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fully_masked_fp16_windows_stay_finite() -> None:
    """In fp16 a -1e9 mask is -inf; a key window that is all masked must not turn the output into NaN."""
    _requires_kernel()
    q, k, v, bias, _, _ = _inputs(1, 64, 1, torch.float16, False)
    bias = bias.float()
    bias[:, 0] = -1e9
    out = window_attention(q, k, v, bias, _W, _NK)
    assert torch.isfinite(out).all()


def test_window_attention_takes_the_bias_to_the_queries_device() -> None:
    """A bias on another device is moved to the queries' device, as the attention backend does."""
    _requires_kernel()
    q, k, v, bias, _, _ = _inputs(2, 64, 1, torch.bfloat16, False)
    expected = window_attention(q, k, v, bias, _W, _NK)
    torch.testing.assert_close(window_attention(q, k, v, bias.cpu(), _W, _NK), expected, atol=0, rtol=0)


@pytest.mark.skipif(not source_module_available(_SOURCE_MODULE), reason="needs the kernel sources")
def test_the_source_compile_resolves_eagerly(monkeypatch: pytest.MonkeyPatch, fresh_caches: None) -> None:
    """The getter compiles, so neither the forward pass nor a CUDA graph capture does."""
    _requires_kernel()
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    get_window_attention_op(torch.bfloat16, _D, _W, _NK)
    assert window_cutedsl.WindowAttentionCuTe._compiled_cache


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("mask", [-1e9, -1e30, -1e37])
def test_logits_all_at_a_large_negative_mask_average_the_values(
    mode: str, monkeypatch: pytest.MonkeyPatch, mask: float
) -> None:
    """With q = k = 0 every logit is its bias, a large negative mask everywhere: the softmax is uniform over the window,
    and the row max must cancel exactly, or exp2 of its rounding residual overflows to inf and the row turns NaN. The
    masks reach bias / scale near FP32's limit (-1e37 / 32**-0.5 is -5.7e37); one past it is out of contract.

    Only the SM107 kernel subtracts the max before scaling; the warp-MMA and WGMMA kernels still fold it into an FMA.
    """
    _requires_kernel()
    if SM_VERSION != 107:
        pytest.skip("only the SM107 kernel cancels the row max exactly")
    torch.manual_seed(0)
    q, k, v, bias, n, windows = _inputs(2, 300, 1, torch.bfloat16, False)
    q, k = torch.zeros_like(q), torch.zeros_like(k)
    bias = torch.full_like(bias, mask)
    out = _run(mode, monkeypatch, lambda: window_attention(q, k, v, bias, _W, _NK))
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out.float(), _reference(q, k, v, bias, n, windows), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("mode", MODES)
def test_ctas_that_run_many_steps_match_the_reference(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """576 steps (24 groups of 4 windows, 4 heads, 6 samples in 2 bias items) on a persistent grid of 2 CTAs per SM: on
    SM107 some CTAs run two steps, and a CTA can start inside a bias item's run of samples."""
    _requires_kernel()
    torch.manual_seed(0)
    q, k, v, bias, n, windows = _inputs(6, 3000, 2, torch.bfloat16, True)
    out = _run(mode, monkeypatch, lambda: window_attention(q, k, v, bias, _W, _NK))
    torch.testing.assert_close(out.float(), _reference(q, k, v, bias, n, windows), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(("samples", "atoms", "bias_samples"), [(6, 300, 2), (5, 919, 1), (3, 200, 3)])
def test_sm107_ctas_through_many_steps_match_the_reference(
    monkeypatch: pytest.MonkeyPatch, fresh_caches: None, samples: int, atoms: int, bias_samples: int
) -> None:
    """A source build for a 5-SM GPU runs 10 persistent CTAs, so each runs several steps: both shared-memory stages and
    their barrier phases, bias reuse within a batch item's samples, the next item's bias prefetch, and the next step's
    QK^T issued under the epilogue. With 72, 160 and 24 steps over 10 CTAs, some CTAs start inside a batch item's run
    of samples."""
    _requires_kernel()
    if SM_VERSION != 107 or not source_module_available(_SOURCE_MODULE):
        pytest.skip("the persistent SM107 kernel's source build")
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    real_properties = torch.cuda.get_device_properties

    class _FiveSMs:
        """The device's real properties, but for a 5-SM GPU; everything else reads through."""

        multi_processor_count = 5

        def __init__(self, index: int) -> None:
            self._properties = real_properties(index)

        def __getattr__(self, name: str) -> object:
            return getattr(self._properties, name)

    monkeypatch.setattr(torch.cuda, "get_device_properties", _FiveSMs)
    torch.manual_seed(0)
    q, k, v, bias, n, windows = _inputs(samples, atoms, bias_samples, torch.bfloat16, False)
    out = window_attention(q, k, v, bias, _W, _NK)
    torch.testing.assert_close(out.float(), _reference(q, k, v, bias, n, windows), atol=2e-2, rtol=2e-2)


def test_an_empty_call_returns_an_empty_output() -> None:
    _requires_kernel()
    q = torch.empty(0, _W, _H, _D, device="cuda", dtype=torch.bfloat16)
    bias = torch.empty(1, 1, _H, _W, _NK, device="cuda", dtype=torch.bfloat16)
    out = window_attention(q, q, q, bias, _W, _NK)
    assert out.shape == (0, _W, _H, _D)
