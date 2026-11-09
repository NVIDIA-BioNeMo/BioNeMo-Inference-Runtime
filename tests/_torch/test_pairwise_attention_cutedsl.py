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

``BIOIR_TEST_CUTEDSL_MODES`` selects which implementations run. A build
carrying kernel sources defaults to ``source``; a source-free build defaults to
``cubin``; setting ``source,cubin`` covers both.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bionemo_ir._torch.attention_backend import (
    AttentionMetadata,
    PairwiseAttentionCuTeLeftMask,
    PairwiseAttentionCuTeLeftMaskMetadata,
    VanillaPairwiseAttention,
)
from bionemo_ir._torch.attention_backend.pairwise_attention import _PW_CONFIGS_DIR
from bionemo_ir._torch.attention_backend.pairwise_attention import _config as pw_config
from bionemo_ir._torch.attention_backend.pairwise_attention import _cubin as pw_cubin
from bionemo_ir._torch.attention_backend.pairwise_attention import cutedsl as pw_cutedsl
from bionemo_ir._torch.utils.kernel import _cutedsl_kernel_library as library_runtime
from tests._torch import (
    SM_VERSION,
    cutedsl_test_modes,
    require_cubin_library,
    skip_cutedsl_for,
    source_module_available,
)

pytestmark = skip_cutedsl_for("pairwise_attention")

FORCE_CUBIN_ENV = pw_cutedsl.FORCE_CUBIN_ENV
_SOURCE_MODULE = "bionemo_ir._torch.attention_backend.pairwise_attention._source"
_BLACKWELL = SM_VERSION in (100, 103)
_MODES = cutedsl_test_modes(_SOURCE_MODULE)
_HEAD_DIMS = (32, 48, 64)
loud_on_missing_cubin = pytest.mark.skipif(
    _BLACKWELL, reason=f"SM{SM_VERSION} runs on SDPA when the kernel library has no CUBIN for it"
)


@functools.cache
def _library_misses_a_tuned_image() -> bool:
    """Whether some anchor has no current image; an SM100 image from before the mult-inner ABI counts as missing."""
    launcher = require_cubin_library().pairwise_attention
    try:
        for head_dim in _tuned_head_dims():
            for anchor in _tuned_anchors(head_dim):
                config = launcher.make_kernel_config(SM_VERSION, head_dim, anchor, launcher.DType.BFLOAT16, False)
                abi = config.cubin.launch_abi
                if abi.startswith("pairwise_attention_sm100_") and "_mult_inner_" not in abi:
                    return True
    except IndexError:
        return True
    return False


def _require_blackwell_cubins() -> None:
    """Skip a CUBIN run on SM100/SM103 when the kernel library predates their images: it runs SDPA there."""
    if _BLACKWELL and _library_misses_a_tuned_image():
        pytest.skip(f"this kernel library has no pairwise-attention CUBIN for every SM{SM_VERSION} anchor yet")


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
    return tuple(
        sorted(
            int(key.removeprefix("S="))
            for key, params in configs.items()
            if pw_config._general_tile(params) is not None
        )
    )


def _anchor_seqlen(anchor: int) -> int:
    """A sequence length served by the tuned anchor, not a multiple of any tile, so the masked tail runs."""
    return 96 if anchor == 0 else anchor + 1


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


def _run(mode, monkeypatch, q, k, v, pair_bias, actual_s_kv, num_heads, head_dim, kv_packed, output_lse=None):
    """Run one backend instance through exactly one implementation path."""
    if mode == "cubin":
        _require_blackwell_cubins()

        def missing_source(_implementation):
            raise ModuleNotFoundError("CuTeDSL kernel source removed")

        monkeypatch.setattr(pw_config, "resolve_implementation", missing_source)
    PairwiseAttentionCuTeLeftMask._compiled_cache.clear()

    backend = PairwiseAttentionCuTeLeftMask(0, num_heads, head_dim, num_kv_heads=num_heads)
    metadata = PairwiseAttentionCuTeLeftMaskMetadata()
    metadata.kv_packed = kv_packed
    out = backend.forward(q, k, v, biases=[actual_s_kv, pair_bias], metadata=metadata, output_lse=output_lse)

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
    num_heads, batch, mult = 4, 2, 2
    # Every anchor, since an anchor can name its own kernel (SM100 runs two).
    for anchor in _tuned_anchors(head_dim):
        seqlen = _anchor_seqlen(anchor)
        inputs = _make_inputs(batch, mult, seqlen, seqlen, num_heads, head_dim, dtype, kv_packed)
        with monkeypatch.context() as patch:
            actual = _run(mode, patch, *inputs, num_heads, head_dim, kv_packed)
        expected = _reference(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], num_heads, head_dim, seqlen)
        torch.testing.assert_close(actual.float(), expected.float(), atol=_ATOL[dtype], rtol=1e-2)


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("head_dim", _HEAD_DIMS)
def test_lse_is_written_only_when_requested(monkeypatch, mode, head_dim):
    """A null LSE pointer skips the store; a buffer receives the reference LSE."""
    if head_dim not in _tuned_head_dims():
        pytest.skip(f"no tuned D{head_dim} config for SM{SM_VERSION}")
    num_heads, batch, mult = 4, 2, 2
    for anchor in _tuned_anchors(head_dim):
        seqlen = _anchor_seqlen(anchor)
        q, k, v, pair_bias, actual_s_kv = _make_inputs(
            batch, mult, seqlen, seqlen, num_heads, head_dim, torch.bfloat16, False
        )
        without_lse = _run(mode, monkeypatch, q, k, v, pair_bias, actual_s_kv, num_heads, head_dim, False).clone()
        lse = torch.full((batch * mult, seqlen, num_heads, 1), torch.nan, device="cuda")
        with_lse = _run(mode, monkeypatch, q, k, v, pair_bias, actual_s_kv, num_heads, head_dim, False, output_lse=lse)
        assert torch.equal(with_lse, without_lse)

        q_heads, k_heads = (t.float().unflatten(-1, (num_heads, head_dim)) for t in (q, k))
        scores = torch.einsum("bqhd,bkhd->bhqk", q_heads, k_heads) * head_dim**-0.5
        scores = scores + pair_bias.float().repeat_interleave(mult, dim=0)
        keep = torch.arange(seqlen, device="cuda") < actual_s_kv.repeat_interleave(mult)[:, None]
        expected = torch.logsumexp(scores.masked_fill(~keep[:, None, None, :], -torch.inf), dim=-1)
        torch.testing.assert_close(lse[..., 0], expected.transpose(1, 2), atol=1e-2, rtol=1e-3)


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
        seqlen = _anchor_seqlen(anchor)
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
@pytest.mark.parametrize(
    ("head_dim", "seqlens"),
    [(32, (64, 80)), (32, (320, 264)), (48, (600, 900)), (48, (1100, 520))],
    ids=["D32-short", "D32-long", "D48-wide", "D48-tall"],
)
def test_rectangular_and_broadcast_batch(
    monkeypatch: pytest.MonkeyPatch, mode: str, head_dim: int, seqlens: tuple[int, int]
) -> None:
    """Sq != Sk with mult > 1: the bias and actual_s_kv broadcast over mult.

    The D=48 cases sit past the SM100 crossover, so on SM100 and SM103 they run the Blackwell kernel.
    """
    if head_dim not in _tuned_head_dims():
        pytest.skip(f"no tuned D{head_dim} config for SM{SM_VERSION}")
    seqlen_q, seqlen_kv = seqlens
    num_heads, batch, mult = 2, 3, 4
    inputs = _make_inputs(batch, mult, seqlen_q, seqlen_kv, num_heads, head_dim, torch.bfloat16, False)
    actual = _run(mode, monkeypatch, *inputs, num_heads, head_dim, False)
    expected = _reference(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], num_heads, head_dim, seqlen_kv)
    assert actual.shape == (batch * mult, seqlen_q, num_heads, head_dim)
    torch.testing.assert_close(actual.float(), expected.float(), atol=_ATOL[torch.bfloat16], rtol=1e-2)
    if _BLACKWELL and head_dim == 48:
        ((_, variant, _),) = PairwiseAttentionCuTeLeftMask._compiled_cache
        assert variant.kernel_abi == "sm100"


@pytest.mark.parametrize("sm", [100, 103])
@pytest.mark.parametrize("head_dim", _HEAD_DIMS)
def test_blackwell_anchors_name_their_kernel(sm: int, head_dim: int) -> None:
    """Short sequences take the Ampere kernel; long ones the Blackwell kernel."""
    short = pw_config.get_nearest_bucket(sm, head_dim, 128)
    long = pw_config.get_nearest_bucket(sm, head_dim, 2048)
    assert pw_config.get_bucket_kernel_abi(sm, head_dim, short) == "sm80"
    assert pw_config.get_bucket_kernel_abi(sm, head_dim, long) == "sm100"


@pytest.mark.skipif(not _BLACKWELL or "source" not in _MODES, reason="requires the Blackwell source kernel")
@pytest.mark.parametrize("head_dim", _HEAD_DIMS)
def test_blackwell_reuses_kernel_across_multiplicities(monkeypatch: pytest.MonkeyPatch, head_dim: int) -> None:
    """Static and dynamic scheduling broadcast each batch's mask over all samples."""
    monkeypatch.delenv(FORCE_CUBIN_ENV, raising=False)
    PairwiseAttentionCuTeLeftMask._compiled_cache.clear()
    num_heads, batch = 4, 2
    backend = PairwiseAttentionCuTeLeftMask(0, num_heads, head_dim, num_kv_heads=num_heads)
    metadata = PairwiseAttentionCuTeLeftMaskMetadata()
    metadata.kv_packed = True
    for mult in (5, 10, 20):
        # Lengths past every Ampere anchor, so the Blackwell kernel runs.
        for seqlen in (1537, 2049):
            q, k, v, pair_bias, _ = _make_inputs(batch, mult, seqlen, seqlen, num_heads, head_dim, torch.bfloat16, True)
            actual_s_kv = torch.tensor([seqlen, seqlen // 2 + 1], dtype=torch.int32, device="cuda")
            actual = backend.forward(q, k, v, biases=[actual_s_kv, pair_bias], metadata=metadata)
            assert backend._last_abi == "sm100"
            expected = _reference(q, k, v, pair_bias, actual_s_kv, num_heads, head_dim, seqlen)
            torch.testing.assert_close(actual.float(), expected.float(), atol=_ATOL[torch.bfloat16], rtol=1e-2)


def test_blackwell_disk_cache_key_names_the_kernel_and_tile() -> None:
    """A retuned SM100 anchor must not reload the kernel an older tile compiled; SM80-SM90 keys keep their form."""
    backend = PairwiseAttentionCuTeLeftMask(0, 4, 48, num_kv_heads=4)
    variant = pw_cutedsl._PairwiseAttentionVariant(torch.bfloat16, 48, 0, True, "sm80")
    first = SimpleNamespace(cache_identity=(64, 64, 128, 3, True))
    second = SimpleNamespace(cache_identity=(64, 128, 128, 3, True))
    backend._sm_version = 100
    assert backend._disk_cache_key(variant, first) != backend._disk_cache_key(variant, second)
    backend._sm_version = 90
    assert backend._disk_cache_key(variant, first) == backend._disk_cache_key(variant, second)


@pytest.mark.skipif("source" not in _MODES, reason="builds the source kernel object")
def test_blackwell_kernel_skips_the_tail_split() -> None:
    """A pairwise call has few work items, so a serialized single-tile tail launch costs more than it saves."""
    kernel = pw_config.get_kernel_config(100, 48, 1142).kernel_factory(48)
    assert kernel.tail_split is False
    assert kernel.dynamic_scheduler is True


def test_anchor_kernel_abi_is_validated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundle = {"kernel_abi": "sm100", "configs": {"S=0": {"kernel_abi": "sm70"}}}
    (tmp_path / "D48_sm100.json").write_text(json.dumps(bundle))
    monkeypatch.setattr(pw_config, "_PW_CONFIGS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="sm70"):
        pw_config.get_bucket_kernel_abi(100, 48, 0)


def _tuned_calls(head_dim: int) -> dict[tuple, tuple[int, int, tuple]]:
    """One (mult, seqlen, key) per measured kernel family and SplitN tile mode, from the shipped table."""
    calls: dict[tuple, tuple[int, int, tuple]] = {}
    for variant in pw_config._tuned_table(SM_VERSION, head_dim).variants:
        mult = variant.mult_ranges[0][0]
        # The anchor itself keeps full SplitN tiles; a nearby length masks the last one.
        for seqlen in (variant.anchor, variant.anchor - 1):
            key = pw_config.tuned_key(mult, head_dim, seqlen, SM_VERSION)
            if key is not None:
                calls.setdefault((key[0], key[0] == "splitn" and key[4]), (mult, seqlen, key))
    return calls


@pytest.mark.skipif(
    not _BLACKWELL or "source" not in _MODES, reason="the measured kernels run from source on SM100/SM103"
)
@pytest.mark.parametrize("head_dim", pw_config.TUNED_HEAD_DIMS)
def test_measured_kernels_match_the_reference(monkeypatch: pytest.MonkeyPatch, head_dim: int) -> None:
    """16-head, batch-1, packed BF16 calls run the measured kernels: every family, padded keys included."""
    monkeypatch.setattr(PairwiseAttentionCuTeLeftMask, "_compiled_cache", {})
    calls = _tuned_calls(head_dim)
    assert {family for family, _ in calls} == {
        variant.key[0] for variant in pw_config._tuned_table(SM_VERSION, head_dim).variants
    }
    num_heads = 16
    for mult, seqlen, key in calls.values():
        q, k, v, pair_bias, actual_s_kv = _make_inputs(
            1, mult, seqlen, seqlen, num_heads, head_dim, torch.bfloat16, True
        )
        backend = PairwiseAttentionCuTeLeftMask(0, num_heads, head_dim, num_kv_heads=num_heads)
        metadata = PairwiseAttentionCuTeLeftMaskMetadata()
        metadata.kv_packed = True
        actual = backend.forward(q, k, v, biases=[actual_s_kv, pair_bias], metadata=metadata)
        assert any(entry[0] == "tuned" and entry[5] == key for entry in PairwiseAttentionCuTeLeftMask._compiled_cache)
        expected = _reference(q, k, v, pair_bias, actual_s_kv, num_heads, head_dim, seqlen)
        torch.testing.assert_close(actual.float(), expected.float(), atol=_ATOL[torch.bfloat16], rtol=1e-2)


@pytest.mark.skipif(not _BLACKWELL, reason="the Blackwell pairwise kernel runs on SM100 and SM103")
@pytest.mark.parametrize(
    ("head_dim", "seqlen"),
    [
        (48, 959),
        (32, 1189),
        # SM103 alone sends this D=48 size to the Blackwell kernel (its S=240 anchor).
        pytest.param(48, 241, marks=pytest.mark.skipif(SM_VERSION != 103, reason="an SM103-only Blackwell window")),
    ],
)
def test_blackwell_kernel_serves_the_diffusion_token_shape(
    head_dim: int, seqlen: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five samples share each bias, the token count is not a multiple of 8, and two batches mask
    different key counts; the bias's padding columns hold large finite values the mask must hide.
    Every selected mode runs the Blackwell kernel, and with both selected they agree bitwise."""
    num_heads, batch, mult = 4, 2, 5
    q, k, v, pair_bias, _ = _make_inputs(batch, mult, seqlen, seqlen, num_heads, head_dim, torch.bfloat16, True)
    masked = 259 if seqlen > 518 else seqlen // 2
    actual_s_kv = torch.tensor([seqlen, seqlen - masked], dtype=torch.int32, device="cuda")
    padded_bias = torch.full(
        (batch, num_heads, seqlen, (seqlen + 7) // 8 * 8), 1e4, dtype=torch.bfloat16, device="cuda"
    )
    padded_bias[..., :seqlen] = pair_bias
    expected = _reference(q, k, v, pair_bias, actual_s_kv, num_heads, head_dim, seqlen)
    outputs = []
    for mode in _MODES:
        with monkeypatch.context() as patch:
            actual = _run(mode, patch, q, k, v, padded_bias, actual_s_kv, num_heads, head_dim, True)
        ((_, variant, _),) = PairwiseAttentionCuTeLeftMask._compiled_cache
        assert variant.kernel_abi == "sm100"
        if mode == "cubin":
            (executable,) = PairwiseAttentionCuTeLeftMask._compiled_cache.values()
            assert type(executable._config.spec).__name__ == "KernelSpecSM100"
        torch.testing.assert_close(actual.float(), expected.float(), atol=_ATOL[torch.bfloat16], rtol=1e-2)
        outputs.append(actual)
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])


@pytest.mark.skipif(not _BLACKWELL, reason="only SM100 and SM103 fall back to SDPA on a library without their CUBINs")
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("cause", ["force_cubin", "no_sources"])
@pytest.mark.parametrize(("batch", "mult", "head_dim", "seqlen"), [(1, 5, 48, 959), (2, 5, 48, 301), (2, 1, 24, 200)])
def test_call_without_a_kernel_takes_sdpa(
    monkeypatch: pytest.MonkeyPatch, graph: bool, cause: str, batch: int, mult: int, head_dim: int, seqlen: int
) -> None:
    """Model tests set CUTEDSL_FORCE_CUBIN after building their layers, and a source-free build has no kernel to
    compile. A kernel library built before the SM100 and SM103 CUBINs shipped cannot load one either, so the call
    runs on SDPA instead of raising: eagerly and under graph capture."""
    num_heads = 4
    backend = PairwiseAttentionCuTeLeftMask(0, num_heads, head_dim, num_kv_heads=num_heads)
    q, k, v, pair_bias, actual_s_kv = _make_inputs(
        batch, mult, seqlen, seqlen, num_heads, head_dim, torch.bfloat16, True
    )

    def library_without_this_sm(*_args, **_kwargs):
        raise pw_cubin.PairwiseAttentionCubinMissing(f"no pairwise-attention CUBIN for SM{SM_VERSION}")

    monkeypatch.setattr(pw_cutedsl, "populate_compiled_cache_from_library", library_without_this_sm)
    if cause == "force_cubin":
        monkeypatch.setenv(FORCE_CUBIN_ENV, "1")
    else:

        def missing_source(_package: str) -> None:
            raise ModuleNotFoundError("CuTeDSL kernel source removed")

        monkeypatch.setattr(pw_cutedsl, "load_source_module", missing_source)
    monkeypatch.setattr(PairwiseAttentionCuTeLeftMask, "_compiled_cache", {})
    metadata = PairwiseAttentionCuTeLeftMaskMetadata()

    def call() -> torch.Tensor:
        return backend.forward(q, k, v, biases=[actual_s_kv, pair_bias], metadata=metadata)

    actual = call()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            actual = call()
        captured.replay()
        torch.cuda.synchronize()
    assert not PairwiseAttentionCuTeLeftMask._compiled_cache
    expected = _reference(q, k, v, pair_bias, actual_s_kv, num_heads, head_dim, seqlen)
    torch.testing.assert_close(actual.float(), expected.float(), atol=_ATOL[torch.bfloat16], rtol=1e-2)


@pytest.mark.parametrize("route", [*_MODES, "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [24, 48])
def test_an_empty_prefix_attends_to_nothing(
    monkeypatch: pytest.MonkeyPatch, route: str, dtype: torch.dtype, head_dim: int
) -> None:
    """A batch whose key prefix is empty gets zero output rows, as SDPA defines a fully masked row, on every route.

    Its samples share the call with a batch whose prefix is not empty, which still matches the reference. A
    finite mask value fails this in float16, where it is too small to swamp the scores.
    """
    if route == "sdpa" and not _BLACKWELL:
        pytest.skip("only SM100 and SM103 fall back to SDPA on a library without their CUBINs")
    num_heads, batch, mult, seqlen = 4, 2, 2, 200
    q, k, v, pair_bias, actual_s_kv = _make_inputs(batch, mult, seqlen, seqlen, num_heads, head_dim, dtype, False)
    actual_s_kv = torch.tensor([0, 77], dtype=torch.int32, device=q.device)
    if route == "sdpa":

        def library_without_this_sm(*_args, **_kwargs):
            raise pw_cubin.PairwiseAttentionCubinMissing(f"no pairwise-attention CUBIN for SM{SM_VERSION}")

        monkeypatch.setattr(pw_cutedsl, "populate_compiled_cache_from_library", library_without_this_sm)
        monkeypatch.setenv(FORCE_CUBIN_ENV, "1")
        monkeypatch.setattr(PairwiseAttentionCuTeLeftMask, "_compiled_cache", {})
        backend = PairwiseAttentionCuTeLeftMask(0, num_heads, head_dim, num_kv_heads=num_heads)
        actual = backend.forward(
            q, k, v, biases=[actual_s_kv, pair_bias], metadata=PairwiseAttentionCuTeLeftMaskMetadata()
        )
    else:
        actual = _run(route, monkeypatch, q, k, v, pair_bias, actual_s_kv, num_heads, head_dim, False)
    torch.testing.assert_close(actual[:mult], torch.zeros_like(actual[:mult]), atol=0, rtol=0)
    expected = _reference(q, k, v, pair_bias, actual_s_kv, num_heads, head_dim, seqlen)
    torch.testing.assert_close(actual[mult:].float(), expected[mult:].float(), atol=_ATOL[dtype], rtol=1e-2)


@pytest.mark.skipif(not _BLACKWELL, reason="only SM100 and SM103 fall back to SDPA on a library without their CUBINs")
@pytest.mark.parametrize(
    "fault",
    [
        library_runtime.CuTeDSLKernelLibraryUnavailable("the kernel library failed to load"),
        library_runtime.CuTeDSLKernelVariantUnavailable("the CUBIN config cannot launch"),
    ],
    ids=["library", "config"],
)
def test_a_broken_kernel_library_raises_instead_of_taking_sdpa(
    monkeypatch: pytest.MonkeyPatch, fault: Exception
) -> None:
    """Only a library without the image falls back to SDPA: one that fails to load, or whose image cannot launch,
    is a broken installation and must not degrade silently."""

    def broken_library(*_args, **_kwargs):
        raise fault

    monkeypatch.setattr(pw_cutedsl, "populate_compiled_cache_from_library", broken_library)
    monkeypatch.setenv(FORCE_CUBIN_ENV, "1")
    monkeypatch.setattr(PairwiseAttentionCuTeLeftMask, "_compiled_cache", {})
    backend, metadata, inputs = _forced_backend(48)
    with pytest.raises(RuntimeError, match=FORCE_CUBIN_ENV):
        backend.forward(inputs[0], inputs[1], inputs[2], biases=[inputs[4], inputs[3]], metadata=metadata)


@pytest.mark.parametrize(
    ("raised", "missing"), [(IndexError, True), (ValueError, False), (TypeError, False), (RuntimeError, False)]
)
def test_the_cubin_adapter_tells_a_missing_image_from_a_bad_config(raised: type[Exception], missing: bool) -> None:
    """The launcher reports a registry miss as IndexError; any other config failure is not a missing image."""

    def make_kernel_config(*_args: object) -> None:
        raise raised("make_kernel_config failed")

    launcher = SimpleNamespace(DType=SimpleNamespace(FLOAT16=0, BFLOAT16=1), make_kernel_config=make_kernel_config)
    with pytest.raises(library_runtime.CuTeDSLKernelVariantUnavailable) as error:
        pw_cubin.PairwiseAttentionCubinExecutable(None, launcher, 100, 48, 959, torch.bfloat16, True)
    assert isinstance(error.value, pw_cubin.PairwiseAttentionCubinMissing) is missing


def test_the_launcher_reports_a_missing_image_as_index_error() -> None:
    """The registry miss the SDPA fallback keys on comes from the C++ lookup, not from argument checks."""
    launcher = require_cubin_library().pairwise_attention
    with pytest.raises(IndexError, match="No embedded pairwise-attention CUBIN"):
        launcher.make_kernel_config(100, 96, 128, launcher.DType.BFLOAT16, False)


@loud_on_missing_cubin
def test_unavailable_cubin_variant_raises(monkeypatch):
    """A missing CUBIN must fail loudly rather than silently degrade."""
    require_cubin_library()

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
# BIOIR_TEST_CUTEDSL_MODES plumbing. The two must not be combined -- source mode
# asserts it never reaches the library -- so these tests stay unparametrised.


def _forced_backend(head_dim, num_heads=4):
    """A backend plus one call's worth of inputs at a shipped head dim."""
    backend = PairwiseAttentionCuTeLeftMask(0, num_heads, head_dim, num_kv_heads=num_heads)
    metadata = PairwiseAttentionCuTeLeftMaskMetadata()
    metadata.kv_packed = False
    inputs = _make_inputs(2, 1, 96, 96, num_heads, head_dim, torch.bfloat16, False)
    return backend, metadata, inputs


def test_force_cubin_takes_the_library_path_with_sources_present(monkeypatch):
    """The flag must reach the CUBINs without deleting the kernel sources."""
    _require_blackwell_cubins()
    monkeypatch.setenv(FORCE_CUBIN_ENV, "1")
    PairwiseAttentionCuTeLeftMask._compiled_cache.clear()

    backend, metadata, inputs = _forced_backend(_tuned_head_dims()[0])
    backend.forward(inputs[0], inputs[1], inputs[2], biases=[inputs[4], inputs[3]], metadata=metadata)

    cached = list(PairwiseAttentionCuTeLeftMask._compiled_cache.values())
    assert len(cached) == 1
    assert isinstance(cached[0], library_runtime.CuTeDSLKernelLibraryExecutable)


@pytest.mark.skipif(not source_module_available(_SOURCE_MODULE), reason="the CuTeDSL kernel sources are not installed")
@pytest.mark.parametrize("library", ["installed", "without_this_sm"])
def test_forcing_cubins_after_a_source_call_loads_the_cubin(monkeypatch: pytest.MonkeyPatch, library: str) -> None:
    """Model tests set CUTEDSL_FORCE_CUBIN after their layers ran: the next call loads the CUBIN, or on SM100/SM103
    runs a pre-mult-inner CUBIN or SDPA from a library without current images, rather than reuse the source
    executable. Clearing the switch, caches kept, runs the source executable again."""
    if library == "without_this_sm" and not _BLACKWELL:
        pytest.skip("only SM100 and SM103 fall back to SDPA on a library without their CUBINs")
    monkeypatch.delenv(FORCE_CUBIN_ENV, raising=False)
    monkeypatch.setattr(PairwiseAttentionCuTeLeftMask, "_compiled_cache", {})
    head_dim = _tuned_head_dims()[0]
    backend, metadata, (q, k, v, pair_bias, actual_s_kv) = _forced_backend(head_dim)

    def call() -> torch.Tensor:
        return backend.forward(q, k, v, biases=[actual_s_kv, pair_bias], metadata=metadata).clone()

    expected = call()
    source_executable = backend._last_executable
    assert source_executable is not None
    assert not isinstance(source_executable, library_runtime.CuTeDSLKernelLibraryExecutable)
    lookups = []
    populate = pw_cutedsl.populate_compiled_cache_from_library

    def counting_populate(*args, **kwargs):
        lookups.append(args)
        if library == "without_this_sm":
            raise pw_cubin.PairwiseAttentionCubinMissing(f"no pairwise-attention CUBIN for SM{SM_VERSION}")
        return populate(*args, **kwargs)

    if library == "installed":
        require_cubin_library()
    monkeypatch.setattr(pw_cutedsl, "populate_compiled_cache_from_library", counting_populate)
    monkeypatch.setenv(FORCE_CUBIN_ENV, "1")
    forced = call()
    assert len(lookups) == 1
    if library == "installed" and not (_BLACKWELL and _library_misses_a_tuned_image()):
        assert isinstance(backend._last_executable, library_runtime.CuTeDSLKernelLibraryExecutable)
        assert torch.equal(forced, expected)
    else:
        # A library without the image runs SDPA; one with an SM100 image from
        # before the mult-inner ABI runs that earlier kernel. Neither matches bitwise.
        if library == "without_this_sm":
            assert backend._last_executable is None
        reference = _reference(q, k, v, pair_bias, actual_s_kv, 4, head_dim, q.shape[1])
        torch.testing.assert_close(forced.float(), reference.float(), atol=_ATOL[torch.bfloat16], rtol=1e-2)
    monkeypatch.delenv(FORCE_CUBIN_ENV)
    assert torch.equal(call(), expected)
    assert backend._last_executable is source_executable
    assert len(lookups) == 1


@loud_on_missing_cubin
def test_force_cubin_error_names_the_flag(monkeypatch):
    """A forced run must not blame absent sources when the CUBIN is missing."""
    require_cubin_library()

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
    _require_blackwell_cubins()
    library = require_cubin_library()
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


@pytest.mark.parametrize("mode", _MODES)
def test_one_layers_slice_of_a_shared_bias_goes_in_without_a_copy(monkeypatch, mode):
    """At batch > 1, one layer's heads of a bias projected for two layers are not contiguous. The kernel reads them
    where they lie and gives the contiguous copy's output, bit for bit."""
    head_dim = _tuned_head_dims()[0]
    num_heads, batch, seqlen = 4, 3, 96
    q, k, v, _, actual_s_kv = _make_inputs(batch, 1, seqlen, seqlen, num_heads, head_dim, torch.bfloat16, False)
    shared = torch.randn(batch, 2 * num_heads, seqlen, seqlen, dtype=torch.bfloat16, device="cuda")
    layer_bias = shared.narrow(1, num_heads, num_heads)
    assert not layer_bias.is_contiguous()
    assert pw_cutedsl._kernel_bias(layer_bias, num_heads, seqlen, seqlen).data_ptr() == layer_bias.data_ptr()
    # A layout the kernel cannot read is still copied.
    transposed = shared.transpose(-1, -2).narrow(1, 0, num_heads)
    assert pw_cutedsl._kernel_bias(transposed, num_heads, seqlen, seqlen).is_contiguous()

    in_place = _run(mode, monkeypatch, q, k, v, layer_bias, actual_s_kv, num_heads, head_dim, False)
    copied = _run(mode, monkeypatch, q, k, v, layer_bias.contiguous(), actual_s_kv, num_heads, head_dim, False)
    assert torch.equal(in_place, copied)


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


@pytest.mark.parametrize("operand", ["q", "k", "v", "output"])
def test_cubin_launch_rejects_wrong_static_head_dim(operand):
    executable, args = _cubin_launch_args(_tuned_head_dims()[0])
    tensor = args[operand]
    args[operand] = torch.zeros(*tensor.shape[:-1], tensor.shape[-1] + 1, dtype=tensor.dtype, device=tensor.device)

    with pytest.raises(ValueError, match=rf"{operand} static tail"):
        _launch(executable, args)
