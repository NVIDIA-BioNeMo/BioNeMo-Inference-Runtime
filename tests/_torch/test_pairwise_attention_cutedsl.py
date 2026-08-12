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
"""Source/CUBIN parity for the CuTeDSL pairwise-attention backend.

``TRTBNM_TEST_CUTEDSL_MODES`` selects which implementations run. A private
checkout defaults to ``source``; a source-free public build defaults to
``cubin``; private CI sets ``source,cubin`` so both are covered.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorrt_bionemo._torch import _cutedsl_kernel_library as library_runtime
from tensorrt_bionemo._torch.attention_backend import (
    AttentionMetadata,
    PairwiseAttentionCuTeLeftMask,
    PairwiseAttentionCuTeLeftMaskMetadata,
    VanillaPairwiseAttention,
)
from tensorrt_bionemo._torch.attention_backend.pairwise_attention import _PW_CONFIGS_DIR
from tensorrt_bionemo._torch.attention_backend.pairwise_attention import _config as pw_config
from tensorrt_bionemo._torch.attention_backend.pairwise_attention import _cubin as pw_cubin
from tensorrt_bionemo._torch.attention_backend.pairwise_attention import cutedsl as pw_cutedsl
from tests._torch import SM_VERSION, cutedsl_test_modes, skip_cutedsl

pytestmark = skip_cutedsl

FORCE_CUBIN_ENV = pw_cutedsl.FORCE_CUBIN_ENV
_SOURCE_MODULE = "tensorrt_bionemo.dsl_kernels.cute.sm80_attn_pb_left_mask"
_MODES = cutedsl_test_modes(_SOURCE_MODULE)
_HEAD_DIMS = (32, 48, 64)


def _tuned_head_dims() -> tuple[int, ...]:
    """Head dimensions the current device actually ships a config for."""
    available = []
    for head_dim in _HEAD_DIMS:
        if (Path(_PW_CONFIGS_DIR) / f"D{head_dim}_sm{SM_VERSION}.json").is_file():
            available.append(head_dim)
    return tuple(available)


def _tuned_anchors(head_dim: int) -> tuple[int, ...]:
    path = Path(_PW_CONFIGS_DIR) / f"D{head_dim}_sm{SM_VERSION}.json"
    with path.open() as handle:
        configs = json.load(handle)["configs"]
    return tuple(sorted(int(key.removeprefix("S=")) for key in configs))


def _make_inputs(batch, mult, seqlen_q, seqlen_kv, num_heads, head_dim, dtype, kv_packed, seed=7):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    batch_flat = batch * mult
    if kv_packed:
        # A packed QKV projection leaves q/k/v as non-contiguous views.
        packed = torch.randn(batch_flat, seqlen_q, 3, num_heads * head_dim, dtype=dtype, device=device)
        q, k, v = packed[:, :, 0, :], packed[:, :, 1, :], packed[:, :, 2, :]
        seqlen_kv = seqlen_q
    else:
        q = torch.randn(batch_flat, seqlen_q, num_heads * head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_flat, seqlen_kv, num_heads * head_dim, dtype=dtype, device=device)
        v = torch.randn_like(k)
    pair_bias = torch.randn(batch, num_heads, seqlen_q, seqlen_kv, dtype=dtype, device=device)
    actual_s_kv = torch.randint(1, seqlen_kv + 1, (batch,), dtype=torch.int32, device=device)
    return q, k, v, pair_bias, actual_s_kv


def _run(mode, monkeypatch, q, k, v, pair_bias, actual_s_kv, num_heads, head_dim, kv_packed):
    """Run one backend instance through exactly one implementation path."""
    if mode == "cubin":

        def missing_source(_implementation):
            raise ModuleNotFoundError("CuTeDSL kernel source removed")

        monkeypatch.setattr(pw_config, "resolve_implementation", missing_source)
    PairwiseAttentionCuTeLeftMask._compiled_cache.clear()

    backend = PairwiseAttentionCuTeLeftMask(0, num_heads, head_dim, num_kv_heads=num_heads)
    metadata = PairwiseAttentionCuTeLeftMaskMetadata()
    metadata.kv_packed = kv_packed
    out = backend.forward(q, k, v, biases=[actual_s_kv, pair_bias], metadata=metadata)

    cached = list(PairwiseAttentionCuTeLeftMask._compiled_cache.values())
    assert len(cached) == 1
    is_library = isinstance(cached[0], library_runtime.CuTeDSLKernelLibraryExecutable)
    assert is_library == (mode == "cubin"), f"{mode} mode selected the wrong executable (library={is_library})"
    return out


# bfloat16 keeps ~3 decimal digits, so a flash-attention rescale can land a
# whole ULP away from the fp32 reference on individual elements.
_ATOL = {torch.float16: 2e-2, torch.bfloat16: 6e-2}


def _reference(q, k, v, pair_bias, actual_s_kv, num_heads, head_dim, seqlen_kv):
    positions = torch.arange(seqlen_kv, device=q.device)
    keep = positions[None, :] < actual_s_kv[:, None]
    additive = torch.where(keep, 0.0, -1e9).to(pair_bias.dtype)
    mult = q.shape[0] // actual_s_kv.shape[0]
    vanilla = VanillaPairwiseAttention(0, num_heads, head_dim, num_kv_heads=num_heads)
    return vanilla.forward(
        q,
        k,
        v,
        biases=[
            additive.repeat_interleave(mult, dim=0)[:, None, None, :],
            pair_bias.repeat_interleave(mult, dim=0),
        ],
        metadata=AttentionMetadata(),
    )


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_packed", [False, True], ids=["separate", "packed"])
@pytest.mark.parametrize("head_dim", _HEAD_DIMS)
def test_pairwise_attention_matches_reference(monkeypatch, mode, dtype, kv_packed, head_dim):
    if head_dim not in _tuned_head_dims():
        pytest.skip(f"no tuned D{head_dim} config for SM{SM_VERSION}")
    num_heads, batch, mult, seqlen = 4, 2, 2, 96
    inputs = _make_inputs(batch, mult, seqlen, seqlen, num_heads, head_dim, dtype, kv_packed)
    actual = _run(mode, monkeypatch, *inputs, num_heads, head_dim, kv_packed)
    expected = _reference(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], num_heads, head_dim, seqlen)
    torch.testing.assert_close(actual.float(), expected.float(), atol=_ATOL[dtype], rtol=1e-2)


@pytest.mark.skipif(
    tuple(_MODES) != ("source", "cubin") and tuple(_MODES) != ("cubin", "source"),
    reason="both implementations are required for an equivalence check",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_packed", [False, True], ids=["separate", "packed"])
@pytest.mark.parametrize("head_dim", _HEAD_DIMS)
def test_source_and_cubin_agree_bitwise(monkeypatch, dtype, kv_packed, head_dim):
    """Both paths run the same machine code, so any difference is a launcher bug."""
    if head_dim not in _tuned_head_dims():
        pytest.skip(f"no tuned D{head_dim} config for SM{SM_VERSION}")
    num_heads, batch, mult = 4, 2, 2
    for anchor in _tuned_anchors(head_dim):
        # Land squarely inside each tuned anchor, and use a sequence length
        # that is not a multiple of the tile so the masked tail is exercised.
        seqlen = 96 if anchor == 0 else anchor + 33
        inputs = _make_inputs(batch, mult, seqlen, seqlen, num_heads, head_dim, dtype, kv_packed)
        with monkeypatch.context() as source_patch:
            source = _run("source", source_patch, *inputs, num_heads, head_dim, kv_packed)
        with monkeypatch.context() as cubin_patch:
            cubin = _run("cubin", cubin_patch, *inputs, num_heads, head_dim, kv_packed)
        assert torch.equal(source, cubin), (
            f"D{head_dim} anchor S={anchor} seqlen={seqlen}: "
            f"max|diff|={(source.float() - cubin.float()).abs().max().item()}"
        )


@pytest.mark.parametrize("mode", _MODES)
def test_rectangular_and_broadcast_batch(monkeypatch, mode):
    """Sq != Sk with mult > 1: the bias and actual_s_kv broadcast over mult."""
    head_dim = _tuned_head_dims()[0]
    num_heads, batch, mult, seqlen_q, seqlen_kv = 2, 3, 4, 64, 80
    inputs = _make_inputs(batch, mult, seqlen_q, seqlen_kv, num_heads, head_dim, torch.bfloat16, False)
    actual = _run(mode, monkeypatch, *inputs, num_heads, head_dim, False)
    expected = _reference(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], num_heads, head_dim, seqlen_kv)
    assert actual.shape == (batch * mult, seqlen_q, num_heads, head_dim)
    torch.testing.assert_close(actual.float(), expected.float(), atol=_ATOL[torch.bfloat16], rtol=1e-2)


def test_unavailable_cubin_variant_raises(monkeypatch):
    """A missing CUBIN must fail loudly rather than silently degrade."""
    pytest.importorskip("tensorrt_bionemo.libs._cutedsl_kernels")

    def missing_source(_implementation):
        raise ModuleNotFoundError("CuTeDSL kernel source removed")

    monkeypatch.setattr(pw_config, "resolve_implementation", missing_source)
    PairwiseAttentionCuTeLeftMask._compiled_cache.clear()

    unshipped_head_dim = 96
    backend = PairwiseAttentionCuTeLeftMask(0, 2, unshipped_head_dim, num_kv_heads=2)
    inputs = _make_inputs(1, 1, 32, 32, 2, unshipped_head_dim, torch.bfloat16, False)
    metadata = PairwiseAttentionCuTeLeftMaskMetadata()
    metadata.kv_packed = False
    with pytest.raises((RuntimeError, ValueError)):
        backend.forward(inputs[0], inputs[1], inputs[2], biases=[inputs[4], inputs[3]], metadata=metadata)


# CUTEDSL_FORCE_CUBIN is the process-wide switch, distinct from the per-test
# TRTBNM_TEST_CUTEDSL_MODES plumbing. The two must not be combined -- source mode
# asserts it never reaches the library -- so these tests stay unparametrised.


def _forced_backend(head_dim, num_heads=4):
    """A backend plus one call's worth of inputs at a shipped head dim."""
    backend = PairwiseAttentionCuTeLeftMask(0, num_heads, head_dim, num_kv_heads=num_heads)
    metadata = PairwiseAttentionCuTeLeftMaskMetadata()
    metadata.kv_packed = False
    inputs = _make_inputs(2, 1, 96, 96, num_heads, head_dim, torch.bfloat16, False)
    return backend, metadata, inputs


def test_force_cubin_takes_the_library_path_with_sources_present(monkeypatch):
    """The flag must reach the CUBINs without deleting the private sources."""
    pytest.importorskip("tensorrt_bionemo.libs._cutedsl_kernels")
    monkeypatch.setenv(FORCE_CUBIN_ENV, "1")
    PairwiseAttentionCuTeLeftMask._compiled_cache.clear()

    backend, metadata, inputs = _forced_backend(_tuned_head_dims()[0])
    backend.forward(inputs[0], inputs[1], inputs[2], biases=[inputs[4], inputs[3]], metadata=metadata)

    cached = list(PairwiseAttentionCuTeLeftMask._compiled_cache.values())
    assert len(cached) == 1
    assert isinstance(cached[0], library_runtime.CuTeDSLKernelLibraryExecutable)


def test_force_cubin_error_names_the_flag(monkeypatch):
    """A forced run must not blame absent sources when the CUBIN is missing."""
    pytest.importorskip("tensorrt_bionemo.libs._cutedsl_kernels")

    def unavailable(*_args, **_kwargs):
        raise library_runtime.CuTeDSLKernelVariantUnavailable("no such variant")

    monkeypatch.setenv(FORCE_CUBIN_ENV, "1")
    monkeypatch.setattr(pw_cutedsl, "populate_compiled_cache_from_library", unavailable)
    PairwiseAttentionCuTeLeftMask._compiled_cache.clear()

    backend, metadata, inputs = _forced_backend(_tuned_head_dims()[0])
    with pytest.raises(RuntimeError, match=FORCE_CUBIN_ENV):
        backend.forward(inputs[0], inputs[1], inputs[2], biases=[inputs[4], inputs[3]], metadata=metadata)


def _cubin_launch_args(head_dim, num_heads=4, batch=2, seqlen=96):
    """One valid launch through the CUBIN adapter, ready to be perturbed."""
    library = pytest.importorskip("tensorrt_bionemo.libs._cutedsl_kernels")
    executable = pw_cubin.PairwiseAttentionCubinExecutable(
        library, library.pairwise_attention, SM_VERSION, head_dim, 0, torch.float16, False
    )
    device = torch.device("cuda")
    qkv = torch.zeros(batch, seqlen, num_heads, head_dim, dtype=torch.float16, device=device)
    args = {
        "q": qkv,
        "k": qkv,
        "v": qkv,
        "bias": torch.zeros(batch, num_heads, seqlen, seqlen, dtype=torch.float16, device=device),
        "actual_s_kv": torch.full((batch,), seqlen, dtype=torch.int32, device=device),
        "output": torch.zeros_like(qkv),
        "lse": torch.zeros(batch, seqlen, num_heads, 1, dtype=torch.float32, device=device),
    }
    return executable, args


def _launch(executable, args):
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
        1,
    )


def test_cubin_launch_rejects_an_operand_on_another_device():
    """Descriptors carry bare addresses, so a foreign pointer must not launch."""
    executable, args = _cubin_launch_args(_tuned_head_dims()[0])
    args["bias"] = args["bias"].cpu()
    with pytest.raises(ValueError, match="no CUDA device"):
        _launch(executable, args)


def test_cubin_launch_rejects_a_broadcast_bias():
    """``expand`` leaves a zero stride the kernel cannot honour."""
    executable, args = _cubin_launch_args(_tuned_head_dims()[0])
    bias = args["bias"]
    args["bias"] = bias[:1].expand(bias.shape)
    with pytest.raises(ValueError, match="non-positive stride"):
        _launch(executable, args)
