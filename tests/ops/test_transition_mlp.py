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

import json

import pytest
import torch

from bionemo_ir._torch.custom_ops.transition_mlp import (
    TransitionMlpVariant,
    get_transition_mlp_op,
    shipped_variants,
    transition_mlp_reference,
)
from bionemo_ir._torch.custom_ops.transition_mlp import _config as transition_config
from bionemo_ir._torch.custom_ops.transition_mlp import cutedsl as transition_cutedsl
from bionemo_ir._torch.custom_ops.transition_mlp import ops as transition_ops
from bionemo_ir._torch.custom_ops.transition_mlp._config import (
    ANY_SIZE,
    buckets,
    bundle_variants,
    config_key,
    get_tile_params,
    nearest_bucket,
    parse_config_key,
    pseudo_seqlen,
)
from bionemo_ir._torch.custom_ops.transition_mlp.cutedsl import TransitionMlpCuTe, TransitionMlpOp
from bionemo_ir._torch.layers.transition import MSATransition, PairTransition, Transition
from bionemo_ir._torch.utils.kernel import launch_compiled_kernel
from tests._torch import cutedsl_test_modes, require_cubin_library, run_cutedsl_test_mode, skip_if_not_sm90

_CUTEDSL_MODES = cutedsl_test_modes("bionemo_ir._torch.custom_ops.transition_mlp._source")
_SM90_VARIANTS = sorted(shipped_variants(90))
# SM versions whose configs run the SM80 kernel; each tunes its own tile per variant.
_SM8X = (80, 86, 89)
# Shared memory per block each SM8x config may use: 163 KB on SM80 (A100, A30), 99 KB on SM86 and SM89.
_SM8X_SMEM_BYTES = {80: 163 * 1024, 86: 99 * 1024, 89: 99 * 1024}

# (batch, tokens) whose batch * tokens**2 rows leave a partial final 128-row tile: 25 and 121
# rows fit in one tile (the second warpgroup gets none, then some), and tails of 2, 65 and 27 rows.
_NON_ALIGNED_TOKENS = [(1, 5), (1, 11), (2, 33), (1, 97), (3, 131)]
# Output rows past the end, filled with NaN; the kernel must leave them untouched.
_GUARD_ROWS = 256


def _variant_id(variant: TransitionMlpVariant) -> str:
    return (
        f"{variant.activation}-bias{int(variant.has_bias)}-mask{int(variant.has_mask)}"
        f"-res{int(variant.has_residual)}-{variant.width}x{variant.hidden}"
    )


def _sm90_images() -> list[tuple[TransitionMlpVariant, int]]:
    """``(variant, bucket)`` once for every distinct tile the SM90 configs ship a variant with."""
    cases = []
    for variant in _SM90_VARIANTS:
        seen = set()
        for bucket in buckets(90, variant):
            tile = tuple(sorted(get_tile_params(90, variant, bucket).items()))
            if tile not in seen:
                seen.add(tile)
                cases.append((variant, bucket))
    return cases


_SM90_IMAGES = _sm90_images()
_SM90_IMAGE_IDS = [f"{_variant_id(variant)}-S{bucket}" for variant, bucket in _SM90_IMAGES]


def _operands(variant: TransitionMlpVariant, batch: int, tokens: int, seed: int = 0) -> tuple:
    """Pair features, released-scale weights and, under their flags, a residual and a mask padding the last tokens."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    width, hidden = variant.width, variant.hidden
    w1_rows = 2 * hidden if variant.activation == "silu_gate" else hidden

    def randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, device="cuda", generator=generator) * scale).to(torch.bfloat16)

    x = randn(batch, tokens, tokens, width)
    w1, w2 = randn(w1_rows, width, scale=width**-0.5), randn(width, hidden, scale=hidden**-0.5)
    b1 = randn(w1_rows, scale=0.1) if variant.has_bias else None
    b2 = randn(width, scale=0.1) if variant.has_bias else None
    residual = randn(batch, tokens, tokens, width) if variant.has_residual else None
    mask = None
    if variant.has_mask:
        valid = torch.arange(tokens, device="cuda") < tokens - min(7, tokens // 4)
        mask = (valid[:, None] & valid[None, :]).expand(batch, -1, -1).unsqueeze(-1).to(torch.bfloat16)
    return x, w1, b1, w2, b2, residual, mask


def _float(operands: tuple) -> tuple:
    return tuple(None if t is None else t.float() for t in operands)


def _max_abs_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    """Max absolute error with NaN counted as infinite; Python's ``max`` would drop it."""
    return (value.float() - reference).abs().nan_to_num(nan=float("inf")).max().item()


def _launch_into_guarded_output(op, operands: tuple, bucket: int | None = None) -> torch.Tensor:
    """Launch the compiled kernel into a NaN buffer, returning the output rows plus the guard rows.

    ``bucket`` forces that bucket's tile; by default the call takes the one its rows select.
    """
    x, w1, b1, w2, b2, residual, mask = operands
    width = op.variant.width
    rows = x.numel() // width
    buffer = torch.full((rows + _GUARD_ROWS, width), float("nan"), device="cuda", dtype=torch.bfloat16)
    if bucket is None:
        bucket = op.backend.bucket(op.variant, rows)
    executable = op.backend._get_or_compile(torch.bfloat16, op.variant, bucket)
    launch_compiled_kernel(
        executable,
        x.reshape(rows, width),
        w1,
        b1,
        w2,
        b2,
        None if residual is None else residual.reshape(rows, width),
        None if mask is None else mask.reshape(rows),
        buffer[:rows],
    )
    return buffer


def _check_guarded_output(buffer: torch.Tensor, operands: tuple, variant: TransitionMlpVariant) -> None:
    """Every row matches fp32 as closely as the unfused bf16 path does; rows past the end stay NaN."""
    rows = operands[0].numel() // variant.width
    reference = transition_mlp_reference(*_float(operands), activation=variant.activation).reshape(rows, -1)
    # The unfused bf16 path rounds the hidden activation and the output as the kernel does.
    bf16 = transition_mlp_reference(*operands, activation=variant.activation).reshape(rows, -1)
    assert torch.isnan(buffer[rows:]).all(), "the kernel wrote past the last row"
    assert not torch.isnan(buffer[:rows]).any(), "the kernel left rows unwritten"
    assert _max_abs_error(buffer[:rows], reference) <= 2 * _max_abs_error(bf16, reference)


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize("variant", _SM90_VARIANTS, ids=_variant_id)
@pytest.mark.parametrize(("batch", "tokens"), _NON_ALIGNED_TOKENS)
def test_matches_reference_on_non_aligned_tokens(mode, variant, batch, tokens, monkeypatch):
    """Every row, including a partial final tile, matches fp32; rows past the end stay untouched."""
    skip_if_not_sm90()
    op = get_transition_mlp_op(torch.bfloat16, **variant._asdict())
    assert op is not None
    operands = _operands(variant, batch, tokens)

    with torch.inference_mode():
        buffer = run_cutedsl_test_mode(
            mode, monkeypatch, TransitionMlpCuTe, transition_cutedsl, lambda: _launch_into_guarded_output(op, operands)
        )
        _check_guarded_output(buffer, operands, variant)


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize(("variant", "bucket"), _SM90_IMAGES, ids=_SM90_IMAGE_IDS)
def test_every_shipped_sm90_tile_matches_reference(mode, variant, bucket, monkeypatch):
    """Each distinct tile a bucket ships meets the same bar, forced on a map with a partial last tile."""
    skip_if_not_sm90()
    op = get_transition_mlp_op(torch.bfloat16, **variant._asdict())
    assert op is not None
    operands = _operands(variant, 3, 131)

    with torch.inference_mode():
        buffer = run_cutedsl_test_mode(
            mode,
            monkeypatch,
            TransitionMlpCuTe,
            transition_cutedsl,
            lambda: _launch_into_guarded_output(op, operands, bucket),
        )
        _check_guarded_output(buffer, operands, variant)


def test_calls_take_the_bucket_nearest_their_pseudo_sequence_length():
    """``round(sqrt(rows))`` names the bucket, ``sqrt(I * J)`` for a pair map; ties go to the smaller anchor."""
    assert pseudo_seqlen(300 * 300) == 300
    anchors = (64, 256, 384)
    assert nearest_bucket(anchors, 320) == 256
    assert nearest_bucket(anchors, 321) == 384
    assert nearest_bucket(anchors, 1) == 64
    assert nearest_bucket(anchors, 10_000) == 384
    assert nearest_bucket((ANY_SIZE,), 10_000) == ANY_SIZE


def test_residual_specific_entries_win_over_entries_for_both_states():
    configs = {
        "act=relu|bias=1|mask=1": {"weight_stages": 4},
        "S=256|act=relu|bias=1|mask=1|res=1": {"weight_stages": 8},
        "S=1024|act=relu|bias=1|mask=1|res=1": {"weight_stages": 12},
    }
    plain = TransitionMlpVariant("relu", True, True, False, 64, 256)
    resolved = bundle_variants(configs, plain.width, plain.hidden)
    assert set(resolved) == {plain, plain._replace(has_residual=True)}
    assert {bucket: tile for bucket, (_, tile) in resolved[plain].items()} == {ANY_SIZE: {"weight_stages": 4}}
    residual = resolved[plain._replace(has_residual=True)]
    assert {bucket: tile["weight_stages"] for bucket, (_, tile) in residual.items()} == {256: 8, 1024: 12}


@pytest.mark.parametrize(
    "key",
    ["act=relu|bias=1|mask=1", "S=256|act=silu_gate|bias=0|mask=0", "S=64|act=relu|bias=1|mask=0|res=1"],
)
def test_config_keys_round_trip(key):
    parsed = parse_config_key(key)
    assert config_key(*parsed[1:4], bucket=parsed.bucket, has_residual=parsed.has_residual) == key


@pytest.mark.parametrize("key", ["S=0|act=relu|bias=1|mask=1", "S=064|act=relu|bias=1|mask=1", "act=relu|bias=1"])
def test_malformed_config_keys_are_rejected(key):
    with pytest.raises(ValueError, match="invalid transition MLP config key"):
        parse_config_key(key)


def _sm8x_tiles() -> list[tuple[int, TransitionMlpVariant, int]]:
    """``(sm, variant, bucket)`` once for every distinct tile an SM8x config ships a variant with, lowest SM first."""
    cases, seen = [], set()
    for sm in _SM8X:
        for variant in sorted(shipped_variants(sm)):
            for bucket in buckets(sm, variant):
                tile = (variant, tuple(sorted(get_tile_params(sm, variant, bucket).items())))
                if tile not in seen:
                    seen.add(tile)
                    cases.append((sm, variant, bucket))
    return cases


_SM8X_TILES = _sm8x_tiles()
_SM8X_TILE_IDS = [f"sm{sm}-{_variant_id(variant)}-S{bucket}" for sm, variant, bucket in _SM8X_TILES]


@pytest.mark.parametrize(("sm", "variant", "bucket"), _SM8X_TILES, ids=_SM8X_TILE_IDS)
@pytest.mark.parametrize(("batch", "tokens"), _NON_ALIGNED_TOKENS)
def test_sm80_kernel_matches_reference_on_non_aligned_tokens(sm, variant, bucket, batch, tokens, monkeypatch):
    """The SM80 kernel, compiled for this GPU from each SM8x config's tile, meets the same bar."""
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("the SM80 kernel needs an SM80 or newer GPU")
    try:
        transition_cutedsl.load_source_module(transition_cutedsl.__package__)
    except ImportError:
        pytest.skip("the SM80 kernel runs from source, which this build strips")
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    op = TransitionMlpOp(TransitionMlpCuTe(sm_version=sm), variant)
    operands = _operands(variant, batch, tokens)
    with torch.inference_mode():
        _check_guarded_output(_launch_into_guarded_output(op, operands, bucket), operands, variant)


@pytest.mark.parametrize("sm", _SM8X)
def test_sm8x_configs_fit_their_shared_memory(sm):
    """Every SM8x config fits its SM's shared memory per block, and names only the SM80 kernel's tile knobs."""
    try:
        source = transition_cutedsl.load_source_module(transition_cutedsl.__package__)
    except ImportError:
        pytest.skip("kernel sources are stripped from this build")

    variants = shipped_variants(sm)
    assert variants == shipped_variants(80)
    for variant in variants:
        for bucket in buckets(sm, variant):
            tile = get_tile_params(sm, variant, bucket)
            assert set(tile) == {"tile_m", "num_warps", "stages"}
            kernel = source.make_kernel(variant, tile, kernel_abi="sm80")
            smem = kernel.dynamic_smem_bytes(
                variant.width,
                variant.hidden,
                variant.activation,
                variant.has_bias,
                tile_m=tile["tile_m"],
                stages=tile["stages"],
            )
            assert smem <= _SM8X_SMEM_BYTES[sm], (sm, variant, bucket, smem)


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize("in_place", [False, True])
def test_op_returns_output_shaped_like_residual(mode, in_place, monkeypatch):
    """The op flattens leading dimensions, and may write its output over the residual."""
    skip_if_not_sm90()
    variant = TransitionMlpVariant("silu_gate", False, False, True, 128, 512)
    op = get_transition_mlp_op(torch.bfloat16, **variant._asdict())
    x, w1, b1, w2, b2, residual, mask = _operands(variant, 2, 45)
    reference = transition_mlp_reference(*_float((x, w1, b1, w2, b2, residual, mask)), activation=variant.activation)

    with torch.inference_mode():
        target = torch.empty_like(residual)

        def run():
            # The mode runner calls twice; an in-place call needs a fresh residual each time.
            if in_place:
                return op(x, w1, b1, w2, b2, target.copy_(residual), mask, out=target)
            return op(x, w1, b1, w2, b2, residual, mask)

        output = run_cutedsl_test_mode(mode, monkeypatch, TransitionMlpCuTe, transition_cutedsl, run)

    assert output.shape == residual.shape and output.dtype == torch.bfloat16
    assert (output.data_ptr() == target.data_ptr()) == in_place
    torch.testing.assert_close(output.float(), reference, atol=5e-2, rtol=2e-2)


def test_op_declines_calls_it_cannot_run():
    """Misaligned rows and mismatched operands return ``None`` so callers keep their own path."""
    skip_if_not_sm90()
    variant = TransitionMlpVariant("relu", True, True, True, 256, 512)
    op = get_transition_mlp_op(torch.bfloat16, **variant._asdict())
    no_residual_op = get_transition_mlp_op(torch.bfloat16, **variant._replace(has_residual=False)._asdict())
    x, w1, b1, w2, b2, residual, mask = _operands(variant, 1, 16)
    flat = torch.empty(x.numel() + 1, device="cuda", dtype=torch.bfloat16)
    misaligned = flat[1:].view_as(x).copy_(x)

    assert op(misaligned, w1, b1, w2, b2, residual, mask) is None
    assert op(x, w1, b1, w2, b2, residual, mask[..., :1, :, :].expand(1, 16, 16, 2)) is None
    assert op(x, w1.float(), b1, w2, b2, residual, mask) is None
    assert op(x, w1, None, w2, None, residual, mask) is None
    assert op(x, w1, b1, w2, b2, residual, None) is None
    assert op(x, w1, b1, w2, b2, None, mask) is None
    assert no_residual_op(x, w1, b1, w2, b2, residual, mask) is None
    assert no_residual_op(misaligned, w1, b1, w2, b2, None, mask) is None
    cpu = [t.cpu() for t in (x, w1, b1, w2, b2, residual, mask)]
    assert not op.accepts(cpu[5], cpu[6], *cpu[1:5])
    assert op(*cpu) is None
    assert not op.accepts(residual, mask.cpu(), w1, b1, w2, b2)
    assert op(x, w1, b1, w2, b2, residual, mask.cpu()) is None


@pytest.mark.parametrize(
    "variant",
    [
        TransitionMlpVariant("relu", True, True, True, 384, 768),
        TransitionMlpVariant("relu", True, True, False, 256, 768),
        TransitionMlpVariant("relu", False, True, True, 256, 512),
        TransitionMlpVariant("silu_gate", False, False, False, 256, 512),
    ],
    ids=_variant_id,
)
def test_unshipped_variants_have_no_op(variant):
    assert get_transition_mlp_op(torch.bfloat16, **variant._asdict()) is None
    shipped = TransitionMlpVariant("relu", True, True, True, 256, 512)
    assert get_transition_mlp_op(torch.float16, **shipped._asdict()) is None


def test_shipped_variants_skip_bundles_the_runtime_rejects(tmp_path, monkeypatch):
    """A bundle for another kernel ABI, one naming a kernel variant, and an SM without an ABI ship nothing."""
    monkeypatch.delenv("BIOIR_TUNED_CONFIG_FOLDER", raising=False)
    monkeypatch.setattr(transition_config, "CONFIGS_DIR", tmp_path)
    configs = {"act=relu|bias=1|mask=1": {"tile_m": 64, "num_warps": 4, "stages": 4}}
    for name, header in (
        ("W128_H512_sm86.json", {"kernel_abi": "sm80"}),
        ("W64_H128_sm86.json", {"kernel_abi": "sm90"}),
        ("W64_H256_sm86.json", {"kernel_abi": "sm80", "kernel_variant": "other"}),
        ("W64_H128_sm100.json", {"kernel_abi": "sm100"}),
    ):
        (tmp_path / name).write_text(json.dumps({**header, "configs": configs}))

    shipped = {TransitionMlpVariant("relu", True, True, residual, 128, 512) for residual in (False, True)}
    assert shipped_variants(86) == shipped
    assert shipped_variants(100) == frozenset()
    for width, hidden in ((128, 512), (64, 128), (64, 256)):
        variant = TransitionMlpVariant("relu", True, True, True, width, hidden)
        assert (get_tile_params(86, variant) is not None) == (variant in shipped)


def test_op_backend_follows_the_current_device_sm(monkeypatch):
    """Each SM gets its own backend, so a later call on another GPU reads that GPU's configs."""
    monkeypatch.setattr(transition_ops, "_transition_mlp_cute_instances", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    variant = TransitionMlpVariant("relu", True, True, True, 256, 512)
    backends = {}
    for capability in ((9, 0), (8, 6), (9, 0)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_, c=capability: c)
        sm = capability[0] * 10 + capability[1]
        op = get_transition_mlp_op(torch.bfloat16, **variant._asdict())
        backend = transition_ops._transition_mlp_cute_instances[sm]
        assert backends.setdefault(sm, backend) is backend and backend._sm_version == sm
        assert op is None or op.backend is backend
    assert set(backends) == {90, 86}
    assert get_transition_mlp_op(torch.float16, **variant._asdict()) is None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_transition_mlp_op(torch.bfloat16, **variant._asdict()) is None


@pytest.mark.parametrize(("variant", "bucket"), _SM90_IMAGES, ids=_SM90_IMAGE_IDS)
def test_cubin_launch_matches_source_launch_bitwise(variant, bucket, monkeypatch):
    """The direct CUBIN launch and CuTe's host launcher run the same code on the same parameters.

    Any difference in a TMA descriptor, atom tag, coordinate or scalar shows up as unequal bits.
    """
    skip_if_not_sm90()
    library = require_cubin_library()
    if not hasattr(library, "transition_mlp"):
        pytest.skip("this extension has no transition_mlp family")
    try:
        sources = transition_cutedsl.load_source_module(transition_cutedsl.__package__)
    except ImportError:
        pytest.skip("kernel sources are stripped from this build")
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    monkeypatch.setattr(TransitionMlpCuTe, "_compiled_cache", {})
    backend = TransitionMlpCuTe()
    source = backend._get_or_compile(torch.bfloat16, variant, bucket)
    cubin = backend._load_cubin_executable(("cubin", *variant, bucket), torch.bfloat16, variant, bucket)
    assert not isinstance(source, transition_cutedsl.CuTeDSLKernelLibraryExecutable)
    assert isinstance(cubin, transition_cutedsl.CuTeDSLKernelLibraryExecutable)
    # Tiles that differ only in pipeline depth compute the same bits, so check the image is this bucket's.
    tile = get_tile_params(90, variant, bucket)
    kernel = sources.make_kernel(variant, tile, kernel_abi="sm90")
    assert cubin._config.spec.bucket == bucket
    assert cubin._config.dynamic_smem_bytes == kernel.dynamic_smem_bytes(
        variant.width, variant.hidden, variant.activation, variant.has_bias, tile["weight_stages"], tile["x_stages"]
    )

    for batch, tokens in ((1, 11), (3, 131)):
        x, w1, b1, w2, b2, residual, mask = _operands(variant, batch, tokens, seed=batch)
        rows = x.numel() // variant.width
        outputs = []
        for executable in (source, cubin):
            output = torch.full((rows, variant.width), float("nan"), device="cuda", dtype=torch.bfloat16)
            launch_compiled_kernel(
                executable,
                x.reshape(rows, -1),
                w1,
                b1,
                w2,
                b2,
                None if residual is None else residual.reshape(rows, -1),
                None if mask is None else mask.reshape(rows),
                output,
            )
            outputs.append(output)
        assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))


def test_forced_cubins_select_the_op_only_when_its_family_ships(monkeypatch):
    """Forcing CUBINs never fails a layer: without a packaged image it keeps the unfused path."""
    skip_if_not_sm90()
    library = require_cubin_library()
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    monkeypatch.setattr(TransitionMlpCuTe, "_compiled_cache", {})

    op = get_transition_mlp_op(torch.bfloat16, width=256, hidden=512)
    assert (op is not None) == hasattr(library, "transition_mlp")

    layer = PairTransition(c_z=256, n=2, dtype=torch.bfloat16, enable_cudnn_graph=True).cuda()
    z = torch.randn(1, 9, 9, 256, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 9, 9, 1, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        assert layer(z, mask=mask, residual=True).shape == z.shape


def _init(layer: torch.nn.Module) -> torch.nn.Module:
    generator = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.copy_(torch.randn(parameter.shape, device="cuda", generator=generator) * 0.05)
    return layer


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize(("c_z", "n"), [(256, 2), (128, 4), (64, 2)])
@pytest.mark.parametrize("residual", [False, True])
def test_pair_transition_runs_fused(mode, c_z, n, residual, monkeypatch):
    """Pair transitions at C=256, 128 and 64 match their unfused path on a non-aligned pair map."""
    skip_if_not_sm90()
    layer = _init(PairTransition(c_z=c_z, n=n, dtype=torch.bfloat16, enable_cudnn_graph=True).cuda())
    assert layer._fused_mlp_ops[residual] is not None
    z = torch.randn(2, 37, 37, c_z, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(2, 37, 37, 1, device="cuda") > 0.2).to(torch.bfloat16)

    with torch.inference_mode():
        fused = run_cutedsl_test_mode(
            mode, monkeypatch, TransitionMlpCuTe, transition_cutedsl, lambda: layer(z, mask=mask, residual=residual)
        )
        unfused = layer._forward_impl(z, mask, residual=residual)

    torch.testing.assert_close(fused, unfused, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize("c_m", [256, 64])
@pytest.mark.parametrize("residual", [False, True])
def test_msa_transition_runs_fused(mode, c_m, residual, monkeypatch):
    """OpenFold2's MSA and extra-MSA transitions match their unfused path, with or without ``m + update``."""
    skip_if_not_sm90()
    layer = _init(MSATransition(c_m=c_m, n=4, dtype=torch.bfloat16).cuda())
    assert layer._fused_mlp_ops[residual] is not None
    m = torch.randn(1, 13, 41, c_m, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(1, 13, 41, device="cuda") > 0.2).to(torch.bfloat16)

    with torch.inference_mode():
        fused = run_cutedsl_test_mode(
            mode, monkeypatch, TransitionMlpCuTe, transition_cutedsl, lambda: layer(m, mask, residual=residual)
        )
        update = layer._forward_impl(m, mask)
        unfused = m + update if residual else update

    torch.testing.assert_close(fused, unfused, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize(("dim", "factor", "masked"), [(128, 4, False), (64, 4, False), (64, 2, True)])
def test_swiglu_transition_runs_its_update_fused(mode, dim, factor, masked, monkeypatch):
    """Without a residual, SwiGLU transitions return only the update, from the fused op."""
    skip_if_not_sm90()
    layer = _init(Transition(dim, dim * factor, dtype=torch.bfloat16).cuda())
    assert layer._fused_mlp_ops[(masked, False)] is not None
    x = torch.randn(1, 29, 29, dim, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(1, 29, 29, device="cuda") > 0.2).to(torch.bfloat16) if masked else None

    with torch.inference_mode():
        fused = run_cutedsl_test_mode(mode, monkeypatch, TransitionMlpCuTe, transition_cutedsl, lambda: layer(x, mask))
        unfused = layer._forward_impl(x, mask)

    torch.testing.assert_close(fused, unfused, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize(("dim", "factor", "masked"), [(128, 4, False), (64, 4, False), (64, 2, True)])
@pytest.mark.parametrize("inplace", [False, True])
def test_swiglu_transition_fuses_its_residual(mode, dim, factor, masked, inplace, monkeypatch):
    """Boltz and OpenFold2-SwiGLU transitions match ``x + update``, in place when asked."""
    skip_if_not_sm90()
    layer = _init(Transition(dim, dim * factor, dtype=torch.bfloat16).cuda())
    assert layer._fused_mlp_ops[(masked, True)] is not None
    x = torch.randn(1, 29, 29, dim, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(1, 29, 29, device="cuda") > 0.2).to(torch.bfloat16) if masked else None

    with torch.inference_mode():
        expected = x + layer._forward_impl(x, mask)
        target = x.clone()
        fused = run_cutedsl_test_mode(
            mode,
            monkeypatch,
            TransitionMlpCuTe,
            transition_cutedsl,
            lambda: layer(target.copy_(x), mask, residual=True, inplace=inplace),
        )

    assert (fused.data_ptr() == target.data_ptr()) == inplace
    torch.testing.assert_close(fused, expected, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
def test_fused_transitions_replay_under_cuda_graph_capture(mode, monkeypatch):
    """Captured fused calls, including an in-place one, replay new inputs like eager calls."""
    skip_if_not_sm90()
    pair = _init(PairTransition(c_z=128, n=4, dtype=torch.bfloat16).cuda())
    msa = _init(MSATransition(c_m=64, n=4, dtype=torch.bfloat16).cuda())
    swiglu = _init(Transition(128, 512, dtype=torch.bfloat16).cuda())
    z = torch.randn(1, 23, 23, 128, device="cuda", dtype=torch.bfloat16)
    m = torch.randn(1, 5, 23, 64, device="cuda", dtype=torch.bfloat16)
    pair_mask = (torch.rand(1, 23, 23, 1, device="cuda") > 0.2).to(torch.bfloat16)
    msa_mask = (torch.rand(1, 5, 23, device="cuda") > 0.2).to(torch.bfloat16)
    static_z, static_m = z.clone(), m.clone()

    def step():
        z_out = pair(static_z, mask=pair_mask, residual=True)
        m_out = static_m + msa(static_m, msa_mask)
        return swiglu(z_out, residual=True, inplace=True), m_out

    with torch.inference_mode():
        # Compile and warm up in the requested mode before capture.
        run_cutedsl_test_mode(mode, monkeypatch, TransitionMlpCuTe, transition_cutedsl, lambda: step()[0])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_z, captured_m = step()
        z_new, m_new = torch.randn_like(z), torch.randn_like(m)
        static_z.copy_(z_new)
        static_m.copy_(m_new)
        graph.replay()
        pair_out = pair._forward_impl(z_new, pair_mask, residual=True)
        expected_z = pair_out + swiglu._forward_impl(pair_out, None)
        expected_m = m_new + msa._forward_impl(m_new, msa_mask)

    torch.testing.assert_close(captured_z, expected_z, atol=6e-2, rtol=3e-2)
    torch.testing.assert_close(captured_m, expected_m, atol=5e-2, rtol=2e-2)
