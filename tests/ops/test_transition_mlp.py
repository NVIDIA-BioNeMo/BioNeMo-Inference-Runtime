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
from bionemo_ir._torch.custom_ops.transition_mlp import _cubin as transition_cubin
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
from bionemo_ir._torch.layers.transition import ConditionedTransitionBlock, MSATransition, PairTransition, Transition
from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryUnavailable,
    CuTeDSLKernelVariantUnavailable,
    launch_compiled_kernel,
)
from tests._torch import (
    SM_VERSION,
    cutedsl_test_modes,
    init_module_weights,
    require_cubin_library,
    run_cutedsl_test_mode,
    skip_if_not_sm90,
    skip_if_not_sm100_family,
)

_CUTEDSL_MODES = cutedsl_test_modes("bionemo_ir._torch.custom_ops.transition_mlp._source")
_SM90_VARIANTS = sorted(shipped_variants(90))
_SM100_VARIANTS = sorted(shipped_variants(100))
# SM versions whose configs run the SM80 kernel; each tunes its own tile per variant.
_SM8X = (80, 86, 89)
# Shared memory per block each SM8x config may use: 163 KB on SM80 (A100, A30), 99 KB on SM86 and SM89.
_SM8X_SMEM_BYTES = {80: 163 * 1024, 86: 99 * 1024, 89: 99 * 1024}
# Shared memory per block an SM100 or SM103 config may use: the 227 KiB opt-in less the 1 KiB the runtime reserves.
_SM100_SMEM_BYTES = 226 * 1024

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
    w1_rows = transition_config.w1_rows(variant)

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
        bucket = op.backend._plan(op.variant, rows)[0]
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
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(variant, marks=pytest.mark.xdist_group(f"transition-sm90-{_variant_id(variant)}"))
        for variant in _SM90_VARIANTS
    ],
    ids=_variant_id,
)
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
@pytest.mark.parametrize(
    ("variant", "bucket"),
    [
        pytest.param(variant, bucket, marks=pytest.mark.xdist_group(f"transition-sm90-{_variant_id(variant)}"))
        for variant, bucket in _SM90_IMAGES
    ],
    ids=_SM90_IMAGE_IDS,
)
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
    resolved = bundle_variants(configs, plain.width, plain.hidden, "sm90")
    assert set(resolved) == {plain, plain._replace(has_residual=True)}
    assert {bucket: entry.tile_params for bucket, entry in resolved[plain].items()} == {ANY_SIZE: {"weight_stages": 4}}
    residual = resolved[plain._replace(has_residual=True)]
    assert {bucket: entry.tile_params["weight_stages"] for bucket, entry in residual.items()} == {256: 8, 1024: 12}


def test_an_entry_naming_a_kernel_abi_runs_that_kernel():
    """The bundle's ABI runs every entry that names none, so few rows can take the SM80 kernel on SM90."""
    configs = {
        "S=16|act=silu_gate|bias=0|mask=1|res=0": {"kernel_abi": "sm80", "tile_m": 32, "num_warps": 2, "stages": 4},
        "S=128|act=silu_gate|bias=0|mask=1|res=0": {"weight_stages": 5, "x_stages": 2},
    }
    ((variant, entries),) = bundle_variants(configs, 256, 512, "sm90").items()
    assert variant == TransitionMlpVariant("silu_gate", False, True, False, 256, 512)
    assert {bucket: (entry.kernel_abi, entry.tile_params) for bucket, entry in entries.items()} == {
        16: ("sm80", {"tile_m": 32, "num_warps": 2, "stages": 4}),
        128: ("sm90", {"weight_stages": 5, "x_stages": 2}),
    }


def test_an_entry_naming_a_kernel_abi_its_sm_cannot_run_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("BIOIR_TUNED_CONFIG_FOLDER", raising=False)
    monkeypatch.setattr(transition_config, "CONFIGS_DIR", tmp_path)
    configs = {"S=16|act=relu|bias=1|mask=1": {"kernel_abi": "sm90", "weight_stages": 4}}
    (tmp_path / "W64_H256_sm86.json").write_text(json.dumps({"kernel_abi": "sm80", "configs": configs}))
    with pytest.raises(ValueError, match="which SM86 cannot run"):
        get_tile_params(86, TransitionMlpVariant("relu", True, True, True, 64, 256))


@pytest.mark.parametrize(
    "key",
    [
        "act=relu|bias=1|mask=1",
        "S=256|act=silu_gate|bias=0|mask=0",
        "S=64|act=relu|bias=1|mask=0|res=1",
        "act=silu_gate_3way|bias=0|mask=0|res=0",
    ],
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


@pytest.mark.parametrize(
    ("sm", "variant", "bucket"),
    [
        pytest.param(
            sm,
            variant,
            bucket,
            marks=pytest.mark.xdist_group(f"transition-sm{sm}-{_variant_id(variant)}-S{bucket}"),
        )
        for sm, variant, bucket in _SM8X_TILES
    ],
    ids=_SM8X_TILE_IDS,
)
@pytest.mark.parametrize(("batch", "tokens"), _NON_ALIGNED_TOKENS)
def test_sm80_kernel_matches_reference_on_non_aligned_tokens(sm, variant, bucket, batch, tokens, monkeypatch):
    """The SM80 kernel, compiled for this GPU from each SM8x config's tile, meets the same bar."""
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("the SM80 kernel needs an SM80 or newer GPU")
    try:
        source = transition_cutedsl.load_source_module(transition_cutedsl.__package__)
    except ImportError:
        pytest.skip("the SM80 kernel runs from source, which this build strips")
    tile = get_tile_params(sm, variant, bucket)
    kernel = source.make_kernel(variant, tile, kernel_abi="sm80")
    smem = kernel.dynamic_smem_bytes(
        variant.width,
        variant.hidden,
        variant.activation,
        variant.has_bias,
        tile_m=tile["tile_m"],
        stages=tile["stages"],
    )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    capacity = properties.shared_memory_per_block_optin
    if smem > capacity:
        # Cross-SM coverage includes A100 tiles that cannot launch on A6000 or L40S.
        assert (properties.major, properties.minor) != (sm // 10, sm % 10)
        pytest.skip(f"SM{sm} tile needs {smem} B shared memory; this GPU allows {capacity} B")
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    op = TransitionMlpOp(TransitionMlpCuTe(sm_version=sm), variant)
    operands = _operands(variant, batch, tokens)
    with torch.inference_mode():
        _check_guarded_output(_launch_into_guarded_output(op, operands, bucket), operands, variant)


@pytest.mark.parametrize("kernel_abi", ["sm90", "sm80"])
def test_three_way_kernels_take_every_operand(kernel_abi, tmp_path, monkeypatch):
    """A 3-way SwiGLU biases its second value from the last third of ``b1`` and composes with the mask and residual.

    The shipped 3-way entry takes none of these operands, so a private bundle declares the variant. SM90 runs on
    four weight stages, the fewest a 3-way chunk fits in: its three W1 tiles and the previous chunk's W2 tile.
    """
    sm = int(kernel_abi.removeprefix("sm"))
    if kernel_abi == "sm90":
        skip_if_not_sm90()
    elif torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("the SM80 kernel needs an SM80 or newer GPU")
    try:
        transition_cutedsl.load_source_module(transition_cutedsl.__package__)
    except ImportError:
        pytest.skip("an unshipped variant runs only from source, which this build strips")
    monkeypatch.delenv("BIOIR_TUNED_CONFIG_FOLDER", raising=False)
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    monkeypatch.setattr(transition_config, "CONFIGS_DIR", tmp_path)
    monkeypatch.setattr(TransitionMlpCuTe, "_compiled_cache", {})
    variant = TransitionMlpVariant("silu_gate_3way", True, True, True, 64, 256)
    tile = {"weight_stages": 4} if kernel_abi == "sm90" else {"tile_m": 64, "num_warps": 4, "stages": 3}
    configs = {config_key(*variant[:3], has_residual=True): tile}
    (tmp_path / f"W64_H256_sm{sm}.json").write_text(json.dumps({"kernel_abi": kernel_abi, "configs": configs}))

    op = TransitionMlpOp(TransitionMlpCuTe(sm_version=sm), variant)
    with torch.inference_mode():
        for batch, tokens in _NON_ALIGNED_TOKENS:
            operands = _operands(variant, batch, tokens)
            _check_guarded_output(_launch_into_guarded_output(op, operands), operands, variant)


@pytest.mark.parametrize(
    ("variant", "tile", "message"),
    [
        # The epilogue waits for both warpgroups' residual tiles, two column tiles each, before it frees one.
        (TransitionMlpVariant("relu", True, True, True, 384, 768), {"weight_stages": 3}, "at least 4 weight stages"),
        (TransitionMlpVariant("silu_gate", False, True, False, 384, 768), {"weight_stages": 5}, "ReLU only"),
    ],
    ids=["residual-on-three-stages", "silu-gate"],
)
def test_sm90_kernel_rejects_c384_configs_it_cannot_run(variant, tile, message):
    skip_if_not_sm90()
    try:
        source = transition_cutedsl.load_source_module(transition_cutedsl.__package__)
    except ImportError:
        pytest.skip("kernel sources are stripped from this build")
    import cutlass

    with pytest.raises(ValueError, match=message):
        source.compile_transition_mlp_source(
            TransitionMlpCuTe(sm_version=90).compile, cutlass.BFloat16, variant, tile, "sm90"
        )


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
    variant = TransitionMlpVariant("relu", True, True, True, 256, 512)
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


def test_three_way_op_declines_a_two_way_projection():
    """The 3-way op reads three blocks of W1 rows, so a 2-way block's two blocks keep the caller's own path."""
    if SM_VERSION not in (90, 100, 103):
        pytest.skip(f"the 3-way SwiGLU ships on SM90, SM100 and SM103 (current SM{SM_VERSION})")
    variant = TransitionMlpVariant("silu_gate_3way", False, False, False, 128, 256)
    op = get_transition_mlp_op(torch.bfloat16, **variant._asdict())
    assert op is not None
    x, w1, _, w2, *_ = _operands(variant, 1, 16)
    two_way = w1[: 2 * variant.hidden]
    assert op.accepts(x, None, w1, None, w2, None)
    assert not op.accepts(x, None, two_way, None, w2, None)
    assert op(x, two_way, None, w2, None, None, None) is None


@pytest.mark.parametrize(
    "variant",
    [
        # SM90 ships 384x768 for the pair transition, which adds the residual in the kernel.
        TransitionMlpVariant("relu", True, True, False, 384, 768),
        TransitionMlpVariant("relu", True, True, False, 256, 768),
        TransitionMlpVariant("relu", False, True, True, 256, 512),
        TransitionMlpVariant("silu_gate", False, False, False, 256, 768),
        TransitionMlpVariant("silu_gate_3way", False, False, True, 128, 256),
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
        ("W64_H128_sm120.json", {"kernel_abi": "sm100"}),
    ):
        (tmp_path / name).write_text(json.dumps({**header, "configs": configs}))

    shipped = {TransitionMlpVariant("relu", True, True, residual, 128, 512) for residual in (False, True)}
    assert shipped_variants(86) == shipped
    assert shipped_variants(120) == frozenset()
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
    kernel_abi = transition_config.kernel_abi(90, variant, bucket)
    kernel = sources.make_kernel(variant, tile, kernel_abi=kernel_abi)
    shape = (variant.width, variant.hidden, variant.activation, variant.has_bias)
    if kernel_abi == "sm90":
        smem_bytes = kernel.dynamic_smem_bytes(*shape, tile["weight_stages"], tile["x_stages"])
    else:
        smem_bytes = kernel.dynamic_smem_bytes(*shape, tile_m=tile["tile_m"], stages=tile["stages"])
    assert cubin._config.spec.bucket == bucket
    assert cubin._config.dynamic_smem_bytes == smem_bytes

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


@pytest.mark.parametrize("sm", _SM8X)
def test_sm8x_cubins_match_source_for_every_shipped_tile(sm, monkeypatch):
    """Exercise the actual SM8x device ABI, including optional slots and dynamic row strides."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (sm // 10, sm % 10):
        pytest.skip(f"requires SM{sm}")
    require_cubin_library()
    try:
        sources = transition_cutedsl.load_source_module(transition_cutedsl.__package__)
    except ImportError:
        pytest.skip("kernel sources are stripped from this build")
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    monkeypatch.setattr(TransitionMlpCuTe, "_compiled_cache", {})
    backend = TransitionMlpCuTe()
    for variant in sorted(shipped_variants(sm)):
        seen = set()
        for bucket in buckets(sm, variant):
            tile = get_tile_params(sm, variant, bucket)
            identity = tuple(sorted(tile.items()))
            if identity in seen:
                continue
            seen.add(identity)
            source = backend._get_or_compile(torch.bfloat16, variant, bucket)
            cubin = backend._load_cubin_executable(("cubin", *variant, bucket), torch.bfloat16, variant, bucket)
            assert isinstance(cubin, transition_cutedsl.CuTeDSLKernelLibraryExecutable)
            kernel = sources.make_kernel(variant, tile, kernel_abi="sm80")
            assert cubin._config.spec.bucket == bucket
            assert cubin._config.dynamic_smem_bytes == kernel.dynamic_smem_bytes(
                variant.width,
                variant.hidden,
                variant.activation,
                variant.has_bias,
                tile_m=tile["tile_m"],
                stages=tile["stages"],
            )
            x, w1, b1, w2, b2, residual, mask = _operands(variant, 1, 11, seed=1)
            rows = x.numel() // variant.width
            strided_x = torch.empty((rows, variant.width + 8), device="cuda", dtype=torch.bfloat16)[:, : variant.width]
            strided_x.copy_(x.reshape(rows, -1))
            results = []
            for executable in (source, cubin):
                storage = torch.full((rows + 1, variant.width + 8), float("nan"), device="cuda", dtype=torch.bfloat16)
                output = storage[:rows, : variant.width]
                r = None
                if residual is not None:
                    output.copy_(residual.reshape(rows, -1))
                    r = output  # Verify supported in-place residual aliasing.
                launch_compiled_kernel(
                    executable,
                    strided_x,
                    w1,
                    b1,
                    w2,
                    b2,
                    r,
                    None if mask is None else mask.reshape(rows),
                    output,
                )
                assert torch.isnan(storage[rows:]).all()
                assert torch.isnan(storage[:, variant.width :]).all()
                results.append(output.clone())
            assert torch.equal(results[0].view(torch.int16), results[1].view(torch.int16)), (sm, variant, bucket)


@pytest.mark.parametrize("three_way", [False, True])
def test_sm8x_gated_cubin_preserves_bf16_rounding_boundaries(three_way, monkeypatch):
    """First projections between BF16 values must round before the SiLU and value products."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((8, 0), (8, 6), (8, 9)):
        pytest.skip("requires an SM8x GPU")
    require_cubin_library()
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    monkeypatch.setattr(TransitionMlpCuTe, "_compiled_cache", {})
    variant = (
        TransitionMlpVariant("silu_gate_3way", False, False, False, 128, 256)
        if three_way
        else TransitionMlpVariant("silu_gate", False, True, False, 64, 128)
    )
    op = get_transition_mlp_op(torch.bfloat16, **variant._asdict())
    assert op is not None
    x = torch.ones((121, variant.width), device="cuda", dtype=torch.bfloat16)
    w1 = torch.zeros((transition_config.w1_rows(variant), variant.width), device="cuda", dtype=x.dtype)
    w1[:, 0] = 1.0
    w1[:, 1] = 3.0 / 1024.0
    w2 = torch.zeros((variant.width, variant.hidden), device="cuda", dtype=x.dtype)
    w2[:, 0] = 1.0
    mask = torch.ones(121, device="cuda", dtype=x.dtype) if variant.has_mask else None
    with torch.inference_mode():
        output = op(x, w1, None, w2, None, None, mask)
        expected = transition_mlp_reference(x, w1, None, w2, None, None, mask, activation=variant.activation)
    assert output is not None
    assert torch.equal(output, expected)


@pytest.mark.parametrize("residual", [False, True])
def test_small_sm8x_transition_uses_fused_output(residual):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((8, 0), (8, 6), (8, 9)):
        pytest.skip("requires an SM8x GPU")
    layer = _init(Transition(128, 512, dtype=torch.bfloat16).cuda())
    assert layer._fused_mlp_ops[(False, residual)] is not None
    x = torch.randn((1, 42, 42, 128), device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = layer._forward_fused(x, None, residual=residual, inplace=False)
        assert expected is not None
        output = layer(x, residual=residual)
    assert torch.equal(output, expected)


@pytest.mark.parametrize("using_silu", [False, True])
def test_small_sm8x_conditioned_transition_uses_fused_output(using_silu):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((8, 0), (8, 6), (8, 9)):
        pytest.skip("requires an SM8x GPU")
    layer = _init(ConditionedTransitionBlock(128, 128, dtype=torch.bfloat16, using_silu=using_silu).cuda())
    assert layer._fused_mlp_op is not None
    a = torch.randn((1, 121, 128), device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        output = layer._swiglu_mlp(a)
        expected = layer._fused_mlp_op(a, layer.fused_swl_a_to_b.weight, None, layer.b_to_a.weight, None, None, None)
        assert expected is not None
    assert torch.equal(output, expected)


def test_large_sm8x_conditioned_transition_uses_fused_bf16_projection():
    """An available full MLP takes precedence over the dual GEMM's FP32 gate."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((8, 0), (8, 6), (8, 9)):
        pytest.skip("requires an SM8x GPU")
    layer = ConditionedTransitionBlock(128, 128, dtype=torch.bfloat16, using_silu=True).cuda()
    op = layer._fused_mlp_op
    assert op is not None and layer._dual_gemm_silu_op is not None
    a = torch.ones((1, 4096, 128), device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        weight = layer.fused_swl_a_to_b.weight
        weight.zero_()
        weight[:, 0] = 1.0
        weight[:, 1] = 3.0 / 1024.0
        layer.b_to_a.weight.zero_()
        layer.b_to_a.weight[:, 0] = 1.0
    with torch.inference_mode():
        projection = layer._dual_gemm_silu_op(a, weight[layer.dim_inner :], weight[: layer.dim_inner], gate="silu")
        expected = layer.b_to_a(projection)
        rounded = op(a, weight, None, layer.b_to_a.weight, None, None, None)
        output = layer._swiglu_mlp(a)
    assert rounded is not None and not torch.equal(rounded, expected)
    assert torch.equal(output, rounded)


def test_large_sm86_transition_uses_fused_output():
    """An available SM86 variant runs above the former row cutoff."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 6):
        pytest.skip("requires SM86")
    layer = _init(Transition(128, 512, dtype=torch.bfloat16).cuda())
    op = layer._fused_mlp_ops[(False, True)]
    assert op is not None
    x = torch.randn((1, 897, 897, 128), device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = layer._forward_fused(x, None, residual=True, inplace=False)
        assert expected is not None
        output = layer(x, residual=True)
    assert torch.equal(output, expected)


def _init(layer: torch.nn.Module) -> torch.nn.Module:
    generator = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.copy_(torch.randn(parameter.shape, device="cuda", generator=generator) * 0.05)
    return layer


def _on_transition_mlp(layer: torch.nn.Module) -> torch.nn.Module:
    """Keep ``layer`` off the LayerNorm-fused kernel, on this op: the path for maps that kernel declines."""
    layer._w_norm_op = None
    return layer


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize("residual", [False, True])
def test_pair_transition_runs_fused(mode, residual, monkeypatch):
    """A pair transition at C = 256, H = 512 matches its unfused path on a non-aligned pair map.

    SM90 ships the ReLU transition MLP at that shape only: every other ReLU shape a layer builds takes the
    LayerNorm-fused kernel there instead.
    """
    skip_if_not_sm90()
    c_z, n = 256, 2
    layer = _on_transition_mlp(
        _init(PairTransition(c_z=c_z, n=n, dtype=torch.bfloat16, enable_cudnn_graph=True).cuda())
    )
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
@pytest.mark.parametrize("residual", [False, True])
def test_msa_transition_runs_fused(mode, residual, monkeypatch):
    """An MSA transition at C = 256, H = 512 matches its unfused path, with or without ``m + update``.

    The MSA transitions at 256 -> 1024 and 64 -> 256 take the LayerNorm-fused kernel on SM90, which ships the ReLU
    transition MLP at 256 -> 512 only.
    """
    skip_if_not_sm90()
    c_m = 256
    layer = _on_transition_mlp(_init(MSATransition(c_m=c_m, n=2, dtype=torch.bfloat16).cuda()))
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
@pytest.mark.parametrize(("dim", "factor", "masked"), [(128, 2, False), (128, 2, True)])
def test_swiglu_transition_runs_its_update_fused(mode, dim, factor, masked, monkeypatch):
    """Without a residual, SwiGLU transitions return only the update, from the fused op.

    At C = 128, H = 256, the one SwiGLU shape SM90 still ships the transition MLP for: the transitions that skip their
    LayerNorm call it with a mask, and ConditionedTransitionBlock without one.
    """
    skip_if_not_sm90()
    layer = _on_transition_mlp(_init(Transition(dim, dim * factor, dtype=torch.bfloat16).cuda()))
    assert layer._fused_mlp_ops[(masked, False)] is not None
    x = torch.randn(1, 29, 29, dim, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(1, 29, 29, device="cuda") > 0.2).to(torch.bfloat16) if masked else None

    with torch.inference_mode():
        fused = run_cutedsl_test_mode(mode, monkeypatch, TransitionMlpCuTe, transition_cutedsl, lambda: layer(x, mask))
        unfused = layer._forward_impl(x, mask)

    torch.testing.assert_close(fused, unfused, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize(
    ("dim", "factor", "masked"), [(128, 4, False), (64, 4, False), (64, 2, True), (256, 4, False), (64, 2, False)]
)
@pytest.mark.parametrize("inplace", [False, True])
def test_swiglu_transition_fuses_its_residual(mode, dim, factor, masked, inplace, monkeypatch):
    """Boltz, OpenFold2-SwiGLU and Protenix transitions match ``x + update``, in place when asked.

    SM90 runs these residual calls as the LayerNorm-fused kernel and ships no transition MLP for them, so on SM90 each
    case skips.
    """
    skip_if_not_sm90()
    layer = _on_transition_mlp(_init(Transition(dim, dim * factor, dtype=torch.bfloat16).cuda()))
    if layer._fused_mlp_ops[(masked, True)] is None:
        pytest.skip(f"this GPU ships no SwiGLU transition MLP for C={dim}, H={dim * factor} with the residual")
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
    """Captured fused calls replay new inputs like eager calls.

    On the shapes where SM90 still ships the transition MLP: the ReLU pair and MSA transitions at C = 256, H = 512, and
    the SwiGLU update at C = 128, H = 256.
    """
    skip_if_not_sm90()
    pair = _on_transition_mlp(_init(PairTransition(c_z=256, n=2, dtype=torch.bfloat16).cuda()))
    msa = _on_transition_mlp(_init(MSATransition(c_m=256, n=2, dtype=torch.bfloat16).cuda()))
    swiglu = _on_transition_mlp(_init(Transition(128, 256, dtype=torch.bfloat16).cuda()))
    z = torch.randn(1, 23, 23, 256, device="cuda", dtype=torch.bfloat16)
    m = torch.randn(1, 5, 23, 256, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(1, 23, 23, 128, device="cuda", dtype=torch.bfloat16)
    pair_mask = (torch.rand(1, 23, 23, 1, device="cuda") > 0.2).to(torch.bfloat16)
    msa_mask = (torch.rand(1, 5, 23, device="cuda") > 0.2).to(torch.bfloat16)
    static_z, static_m, static_x = z.clone(), m.clone(), x.clone()

    def step():
        z_out = pair(static_z, mask=pair_mask, residual=True)
        m_out = static_m + msa(static_m, msa_mask)
        return z_out, m_out, swiglu(static_x)

    with torch.inference_mode():
        # Compile and warm up in the requested mode before capture.
        run_cutedsl_test_mode(mode, monkeypatch, TransitionMlpCuTe, transition_cutedsl, lambda: step()[0])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_z, captured_m, captured_x = step()
        z_new, m_new, x_new = torch.randn_like(z), torch.randn_like(m), torch.randn_like(x)
        static_z.copy_(z_new)
        static_m.copy_(m_new)
        static_x.copy_(x_new)
        graph.replay()
        expected_z = pair._forward_impl(z_new, pair_mask, residual=True)
        expected_m = m_new + msa._forward_impl(m_new, msa_mask)
        expected_x = swiglu._forward_impl(x_new, None)

    torch.testing.assert_close(captured_z, expected_z, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(captured_m, expected_m, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(captured_x, expected_x, atol=5e-2, rtol=2e-2)


def _require_sm100() -> None:
    """Skip unless this GPU runs the SM100 launch ABI."""
    skip_if_not_sm100_family()


def _require_sm100_cubins(*sms: int) -> None:
    """Skip unless the kernel library carries SM100-kernel images for ``sms``: one linked from packs that predate
    them keeps these SMs on the layers' own path."""
    specs = require_cubin_library().transition_mlp.kernel_specs()
    if missing := sorted(set(sms) - {spec.target_sm for spec in specs if spec.kernel_sm == 100}):
        pytest.skip(f"this kernel library has no transition MLP CUBIN for SM{', SM'.join(map(str, missing))} yet")


def test_sm100_ships_the_model_swiglu_shapes_on_both_family_members():
    """SM100 and SM103 ship the same SwiGLU variants: every mask and residual state at the widths the pair, MSA and
    template transitions and the atom transformer use, and the 3-way atom block's MLP, on the SM100 kernel at every
    bucket. SM103's configs copy SM100's until a B300 retunes them."""
    widths = {(64, 128), (64, 256), (128, 256), (128, 512), (256, 512), (256, 1024)}
    expected = {
        TransitionMlpVariant("silu_gate", False, mask, residual, width, hidden)
        for width, hidden in widths
        for mask in (False, True)
        for residual in (False, True)
    }
    expected.add(TransitionMlpVariant("silu_gate_3way", False, False, False, 128, 256))
    assert set(_SM100_VARIANTS) == expected
    assert shipped_variants(103) == shipped_variants(100)
    for sm in (100, 103):
        for variant in _SM100_VARIANTS:
            for bucket in buckets(sm, variant):
                assert transition_config.kernel_abi(sm, variant, bucket) == "sm100", (sm, variant, bucket)


def test_sm100_configs_fit_shared_memory():
    """Every SM100 and SM103 tile fits a CTA's shared memory and names only the SM100 kernel's knobs."""
    try:
        source = transition_cutedsl.load_source_module(transition_cutedsl.__package__)
    except ImportError:
        pytest.skip("kernel sources are stripped from this build")

    for sm in (100, 103):
        for variant in _SM100_VARIANTS:
            for bucket in buckets(sm, variant):
                tile = get_tile_params(sm, variant, bucket)
                knobs = {"x_stages", "w1_stages", "w2_stages", "gemm2_lag", "first_use_loads"}
                assert set(tile) <= knobs, (sm, variant, tile)
                kernel = source.make_kernel(variant, tile, kernel_abi="sm100")
                smem = kernel.dynamic_smem_bytes(variant.width, variant.hidden, **tile, activation=variant.activation)
                assert smem <= _SM100_SMEM_BYTES, (sm, variant, smem)


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(variant, marks=pytest.mark.xdist_group(f"transition-sm100-{_variant_id(variant)}"))
        for variant in _SM100_VARIANTS
    ],
    ids=_variant_id,
)
@pytest.mark.parametrize(("batch", "tokens"), _NON_ALIGNED_TOKENS)
@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
def test_sm100_kernel_matches_reference_on_non_aligned_tokens(mode, variant, batch, tokens, monkeypatch):
    """Every row, including a partial last 256-row pair tile and pairs that loop over several tiles,
    matches fp32 as closely as the unfused bf16 path; rows past the end stay untouched."""
    _require_sm100()
    if mode == "cubin":
        _require_sm100_cubins(SM_VERSION)
    op = get_transition_mlp_op(torch.bfloat16, **variant._asdict())
    assert op is not None
    operands = _operands(variant, batch, tokens)
    with torch.inference_mode():
        output = run_cutedsl_test_mode(
            mode,
            monkeypatch,
            TransitionMlpCuTe,
            transition_cutedsl,
            lambda: _launch_into_guarded_output(op, operands),
        )
        _check_guarded_output(output, operands, variant)
        if mode == "cubin":
            assert all(ex._config.cubin.kernel_sm == 100 for ex in TransitionMlpCuTe._compiled_cache.values())


def test_sm100_kernel_matches_reference_at_a_large_pair():
    """The largest benchmark complex's padded pair (1600 tokens, 2.56 M rows) through the Pairformer transition
    (C = 256, H = 1024, residual written in place): every row of the fused output stays as close to fp32 as the
    unfused bf16 path. The reference runs in row chunks to bound its memory."""
    _require_sm100()
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 16 * 2**30:
        pytest.skip("needs about 16 GiB of free device memory")
    variant = TransitionMlpVariant("silu_gate", False, False, True, 256, 1024)
    op = get_transition_mlp_op(torch.bfloat16, **variant._asdict())
    x, w1, b1, w2, b2, residual, mask = _operands(variant, 1, 1600)
    rows = x.numel() // variant.width
    with torch.inference_mode():
        target = residual.clone()
        output = op(x, w1, b1, w2, b2, target, mask, out=target)
        assert output is not None and output.data_ptr() == target.data_ptr()
        x_rows, residual_rows, output_rows = (t.reshape(rows, variant.width) for t in (x, residual, output))
        fused_error = unfused_error = 0.0
        chunk = 1 << 18
        for start in range(0, rows, chunk):
            part = slice(start, start + chunk)
            reference = transition_mlp_reference(
                x_rows[part].float(),
                w1.float(),
                None,
                w2.float(),
                None,
                residual_rows[part].float(),
                None,
                activation=variant.activation,
            )
            unfused = transition_mlp_reference(
                x_rows[part], w1, None, w2, None, residual_rows[part], None, activation=variant.activation
            )
            fused_error = max(fused_error, _max_abs_error(output_rows[part], reference))
            unfused_error = max(unfused_error, _max_abs_error(unfused, reference))
    assert fused_error <= 2 * unfused_error


@pytest.mark.parametrize(("width", "hidden"), [(128, 512), (256, 1024)])
def test_sm100_op_writes_over_its_residual(width, hidden):
    """The in-place call the Pairformer makes: the output overwrites the residual it adds."""
    _require_sm100()
    variant = TransitionMlpVariant("silu_gate", False, True, True, width, hidden)
    op = get_transition_mlp_op(torch.bfloat16, **variant._asdict())
    x, w1, b1, w2, b2, residual, mask = _operands(variant, 1, 300)
    reference = transition_mlp_reference(*_float((x, w1, b1, w2, b2, residual, mask)), activation=variant.activation)
    bf16 = transition_mlp_reference(x, w1, b1, w2, b2, residual, mask, activation=variant.activation)
    with torch.inference_mode():
        target = residual.clone()
        output = op(x, w1, b1, w2, b2, target, mask, out=target)
    assert output is not None and output.data_ptr() == target.data_ptr()
    assert _max_abs_error(output, reference) <= 2 * _max_abs_error(bf16, reference)


def test_sm100_forced_cubin_binds_the_fused_op(monkeypatch):
    """Layers bind the SM100 op when packaged kernels are forced."""
    skip_if_not_sm100_family()
    _require_sm100_cubins(SM_VERSION)
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    for width, hidden in ((64, 128), (64, 256), (128, 256), (128, 512), (256, 512), (256, 1024)):
        layer = Transition(width, hidden, dtype=torch.bfloat16).cuda()
        for (masked, residual), op in layer._fused_mlp_ops.items():
            variant = TransitionMlpVariant("silu_gate", False, masked, residual, width, hidden)
            assert (op is not None) == (variant in _SM100_VARIANTS)
        x = torch.randn(1, 9, 9, width, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            assert layer(x, None, residual=True).shape == x.shape


@pytest.mark.parametrize(("width", "hidden"), [(128, 512), (256, 1024)])
def test_sm100_layer_uses_cubins_when_forced_after_build(monkeypatch, width, hidden):
    """An existing layer switches to its packaged kernel with identical output."""
    _require_sm100()
    _require_sm100_cubins(SM_VERSION)
    torch.manual_seed(0)
    layer = Transition(width, hidden, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(0.0, parameter.shape[-1] ** -0.5 if parameter.ndim > 1 else 0.1)
    ops = layer._fused_mlp_ops
    assert all(op is not None for op in ops.values())
    # 129 tokens: above every SM's small-call cutoff, so the fused path runs the kernel.
    x = torch.randn(1, 129, 129, width, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(1, 129, 129, 1, device="cuda") > 0.2).to(torch.bfloat16)
    with torch.inference_mode():
        fused = layer(x.clone(), mask, residual=True, inplace=True)
        monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
        forced = layer(x.clone(), mask, residual=True, inplace=True)
        rebuilt = Transition(width, hidden, dtype=torch.bfloat16).cuda()
        monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
        again = layer(x.clone(), mask, residual=True, inplace=True)
    assert torch.equal(forced, fused)
    assert all(op is not None for op in rebuilt._fused_mlp_ops.values())
    assert torch.equal(again, fused)


def test_sm100_layer_keeps_its_path_on_a_library_without_sm100_cubins(monkeypatch):
    """A kernel library built before the SM100 and SM103 CUBINs shipped has no image for them. Under
    ``CUTEDSL_FORCE_CUBIN`` a layer bound before it runs its unfused path rather than failing, and its fused
    kernel again once the switch is cleared."""
    _require_sm100()
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    torch.manual_seed(0)
    layer = _on_transition_mlp(init_module_weights(Transition(128, 512, dtype=torch.bfloat16).cuda()))
    ops = layer._fused_mlp_ops
    assert all(op is not None for op in ops.values())
    x = torch.randn(1, 129, 129, 128, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(1, 129, 129, 1, device="cuda") > 0.2).to(torch.bfloat16)

    def library_without_sm100(*_args, **_kwargs):
        raise transition_cubin.TransitionMlpCubinMissing("no SM100 transition MLP CUBIN")

    with torch.inference_mode():
        fused = layer(x.clone(), mask, residual=True, inplace=True)
        layer._fused_mlp_ops = dict.fromkeys(ops)
        unfused = layer(x.clone(), mask, residual=True, inplace=True)
        layer._fused_mlp_ops = ops
        with monkeypatch.context() as patch:
            patch.setattr(transition_cutedsl, "populate_compiled_cache_from_library", library_without_sm100)
            patch.setattr(TransitionMlpCuTe, "_compiled_cache", {})
            patch.setenv("CUTEDSL_FORCE_CUBIN", "1")
            forced = layer(x.clone(), mask, residual=True, inplace=True)
            assert Transition(128, 512, dtype=torch.bfloat16).cuda()._fused_mlp_ops == dict.fromkeys(ops)
        again = layer(x.clone(), mask, residual=True, inplace=True)
    assert torch.isfinite(unfused).all()
    assert torch.equal(forced, unfused)
    assert torch.equal(again, fused)


@pytest.mark.parametrize(
    "fault",
    [
        CuTeDSLKernelLibraryUnavailable("the kernel library failed to load"),
        CuTeDSLKernelVariantUnavailable("the CUBIN config cannot launch"),
    ],
    ids=["library", "config"],
)
def test_sm100_a_broken_kernel_library_raises_under_forced_cubins(monkeypatch, fault):
    """Only a library without the image keeps the caller's path: one that fails to load, or whose image cannot
    launch, is a broken installation and must not degrade silently."""
    _require_sm100()
    layer = _on_transition_mlp(Transition(128, 512, dtype=torch.bfloat16).cuda())
    assert all(op is not None for op in layer._fused_mlp_ops.values())
    # 129 tokens: above every SM's small-call cutoff, so the fused path reaches the kernel.
    x = torch.randn(1, 129, 129, 128, device="cuda", dtype=torch.bfloat16)

    def broken_library(*_args, **_kwargs):
        raise fault

    monkeypatch.setattr(transition_cutedsl, "populate_compiled_cache_from_library", broken_library)
    monkeypatch.setattr(TransitionMlpCuTe, "_compiled_cache", {})
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    with torch.inference_mode(), pytest.raises(RuntimeError, match="CUTEDSL_FORCE_CUBIN"):
        layer(x, None, residual=True)


@pytest.mark.parametrize(
    ("raised", "missing"), [(IndexError, True), (ValueError, False), (TypeError, False), (RuntimeError, False)]
)
def test_the_cubin_adapter_tells_a_missing_image_from_a_bad_config(raised, missing):
    """The launcher reports a registry miss as IndexError; any other config failure is not a missing image."""

    class Launcher:
        class DType:
            BFLOAT16 = 0

        def make_kernel_config(self, *_args):
            raise raised("make_kernel_config failed")

    variant = TransitionMlpVariant("silu_gate", False, True, True, 128, 512)
    with pytest.raises(CuTeDSLKernelVariantUnavailable) as error:
        transition_cubin.TransitionMlpCubinExecutable(None, Launcher(), 100, torch.bfloat16, variant, 960)
    assert isinstance(error.value, transition_cubin.TransitionMlpCubinMissing) is missing


def test_the_launcher_reports_a_missing_image_as_index_error():
    """The registry miss the layer fallback keys on comes from the C++ lookup, not from argument checks."""
    _require_sm100()
    launcher = require_cubin_library().transition_mlp
    with pytest.raises(IndexError, match="No embedded transition MLP CUBIN"):
        launcher.make_kernel_config(100, launcher.DType.BFLOAT16, True, False, False, False, False, 48, 96, 1)


# (128, 256) unmasked is Boltz's pair conditioning (``PairwiseConditioning``), which ships no masked variant.
@pytest.mark.parametrize(
    ("width", "hidden", "masked"),
    [(128, 512, False), (128, 512, True), (256, 1024, False), (256, 1024, True), (128, 256, False)],
)
def test_sm100_transition_layer_takes_the_fused_path(width, hidden, masked):
    """``Transition`` picks the fused op on SM100 and matches its own unfused path, in place."""
    _require_sm100()
    torch.manual_seed(0)
    layer = Transition(width, hidden, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(0.0, parameter.shape[-1] ** -0.5 if parameter.ndim > 1 else 0.1)
    assert all(layer._fused_mlp_ops[(masked, residual)] is not None for residual in (False, True))
    # 129 tokens: above every SM's small-call cutoff, so the fused path runs the kernel.
    x = torch.randn(1, 129, 129, width, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(1, 129, 129, 1, device="cuda") > 0.2).to(torch.bfloat16) if masked else None
    with torch.inference_mode():
        fused = layer(x.clone(), mask, residual=True, inplace=True)
        ops = layer._fused_mlp_ops
        layer._fused_mlp_ops = dict.fromkeys(ops)
        unfused = layer(x.clone(), mask, residual=True, inplace=True)
        layer._fused_mlp_ops = ops
    update = (unfused.float() - x.float()).abs().max().item()
    assert (fused.float() - unfused.float()).abs().max().item() <= 0.05 * max(update, 1.0)


def test_sm100_op_declines_an_output_off_its_32_byte_alignment():
    """The SM100 kernel stores rows in 32-byte vectors, so an output 16 bytes off that alignment keeps
    the caller's path instead of faulting."""
    _require_sm100()
    variant = TransitionMlpVariant("silu_gate", False, False, True, 128, 512)
    op = get_transition_mlp_op(torch.bfloat16, **variant._asdict())
    # 65 tokens: above every SM's small-call cutoff for this variant.
    x, w1, b1, w2, b2, residual, mask = _operands(variant, 1, 65)
    storage = torch.empty(residual.numel() + 16, device="cuda", dtype=torch.bfloat16)
    assert storage.data_ptr() % 32 == 0
    shifted = storage[8 : 8 + residual.numel()].view_as(residual)
    with torch.inference_mode():
        assert op(x, w1, b1, w2, b2, residual, mask, out=shifted) is None
        aligned = storage[16 : 16 + residual.numel()].view_as(residual)
        assert op(x, w1, b1, w2, b2, residual, mask, out=aligned) is not None
        torch.cuda.synchronize()


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(("width", "hidden"), [(256, 512), (256, 1024), (128, 512)])
def test_sm100_small_calls_keep_the_layer_path_where_it_is_faster(width, hidden, masked):
    """At C = 256 the kernel streams every weight through a few CTA pairs, so calls below the config's
    ``min_seqlen`` (set per mask variant) keep the unfused path; C = 128 serves every size. SM103 copies SM100's
    cutoffs."""
    _require_sm100()
    variant = TransitionMlpVariant("silu_gate", False, masked, True, width, hidden)
    # The smallest bucket's entry serves the small calls.
    assert (transition_config.min_seqlen(100, variant, buckets(100, variant)[0]) > 0) == (width == 256)
    if width == 256:
        assert transition_config.min_seqlen(103, variant, buckets(103, variant)[0]) > 0
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    threshold = transition_config.min_seqlen(sm, variant, buckets(sm, variant)[0])
    small, large = (threshold - 1, threshold) if threshold else (8, 64)
    op = get_transition_mlp_op(torch.bfloat16, **variant._asdict())
    for tokens, expected in ((small, threshold == 0), (large, True)):
        x, w1, b1, w2, b2, residual, mask = _operands(variant, 1, tokens)
        assert op.accepts(residual, mask, w1, b1, w2, b2) == expected, tokens


def _atom_block(seed: int = 0, using_silu: bool = True) -> ConditionedTransitionBlock:
    """The diffusion atom transformer's conditioned transition (Protenix-v2, OpenFold3): C = 128, 2-way SwiGLU, or
    with ``using_silu=False`` the 3-way SwiGLU."""
    torch.manual_seed(seed)
    layer = ConditionedTransitionBlock(128, 128, dtype=torch.bfloat16, using_silu=using_silu).cuda()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(0.0, parameter.shape[-1] ** -0.5 if parameter.ndim > 1 else 0.1)
    return layer


def _dual_gemm_mlp(layer: ConditionedTransitionBlock, a: torch.Tensor) -> torch.Tensor:
    """The block's MLP on the path it takes without the fused op: the dual GEMM's SiLU gate, then ``b_to_a``."""
    op = layer._fused_mlp_op
    layer._fused_mlp_op = None
    try:
        return layer._swiglu_mlp(a)
    finally:
        layer._fused_mlp_op = op


# Atom windows of 32 rows over 5 diffusion samples (the 5SBJ, 7QSJ, 7WR3_A_C and 8IC7_A_B calls): one 256-row
# tile per CTA pair ending in a partial one (7 and 91 windows), one pair looping over three tiles while the
# rest run two (238), and pairs running three or four (421).
@pytest.mark.parametrize("windows", [7, 91, 238, 421])
def test_sm100_atom_conditioned_transition_runs_the_fused_mlp(windows):
    """The atom transformer's SwiGLU MLP takes the fused op on SM100 and matches fp32 as closely as the dual GEMM
    path it replaces, which rounds the hidden activation to bf16 the same way."""
    _require_sm100()
    layer = _atom_block()
    assert layer._fused_mlp_op is not None and layer._dual_gemm_silu_op is not None
    a = torch.randn(1, 5, windows, 32, 128, device="cuda", dtype=torch.bfloat16)
    weight, projection = layer.fused_swl_a_to_b.weight, layer.b_to_a.weight
    with torch.inference_mode():
        assert layer._fused_mlp_op.accepts(a, None, weight, None, projection, None)
        fused = layer._swiglu_mlp(a)
        dual_gemm = _dual_gemm_mlp(layer, a)
    reference = transition_mlp_reference(
        a.float(), weight.float(), None, projection.float(), None, None, None, activation="silu_gate"
    )
    assert fused.shape == a.shape
    assert _max_abs_error(fused, reference) <= 2 * _max_abs_error(dual_gemm, reference)


def test_sm100_atom_conditioned_transition_block_matches_its_dual_gemm_path():
    """The whole block (AdaLN, MLP, output gate, residual) agrees with the block on its dual GEMM path."""
    _require_sm100()
    layer = _atom_block()
    a = torch.randn(1, 5, 29, 32, 128, device="cuda", dtype=torch.bfloat16)
    s = torch.randn(1, 1, 29, 32, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        fused = layer(a, s, residual=a)
        op = layer._fused_mlp_op
        layer._fused_mlp_op = None
        dual_gemm = layer(a, s, residual=a)
        layer._fused_mlp_op = op
    update = (dual_gemm.float() - a.float()).abs().max().item()
    assert (fused.float() - dual_gemm.float()).abs().max().item() <= 0.05 * max(update, 1.0)


def test_sm100_atom_mlp_op_runs_under_forced_cubins(monkeypatch):
    """Forcing packaged kernels keeps the atom MLP on the fused path."""
    _require_sm100()
    _require_sm100_cubins(SM_VERSION)
    layer = _atom_block()
    op = layer._fused_mlp_op
    a = torch.randn(1, 5, 20, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = (layer.fused_swl_a_to_b.weight, None, layer.b_to_a.weight, None)
    projections = []

    def dual_gemm(x, weight_a, weight_b, gate):
        # The dual GEMM ships CUBINs of its own; this test only checks that the block reaches it.
        projections.append(gate)
        return torch.zeros(*x.shape[:-1], layer.dim_inner, device=x.device, dtype=x.dtype)

    layer._dual_gemm_silu_op = dual_gemm
    with torch.inference_mode():
        assert op(a, *weights, None, None) is not None
        layer._swiglu_mlp(a)
        assert projections == []
        monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
        assert op(a, *weights, None, None) is not None
        output = layer._swiglu_mlp(a)
    assert projections == []
    assert output.shape == a.shape
    assert get_transition_mlp_op(torch.bfloat16, **op.variant._asdict()) is not None


def test_sm103_images_resolve_for_every_sm103_call():
    """SM103 runs the SM100 kernel's images, built for sm_103a. Every SM103 variant and bucket, masked and unmasked
    calls and the 3-way SwiGLU, resolves through the C++ lookup to an SM103-target image of kernel SM100; a masked call
    and its unmasked twin get one image. A registry lookup, so it needs no SM103 GPU."""
    if SM_VERSION == 0:
        pytest.skip("loading the kernel library needs the CUDA driver")
    _require_sm100_cubins(103)
    launcher = require_cubin_library().transition_mlp
    variants = shipped_variants(103)
    assert {v.has_mask for v in variants} == {False, True} and any(v.activation == "silu_gate_3way" for v in variants)
    images = {}
    for variant in variants:
        for bucket in buckets(103, variant):
            config = launcher.make_kernel_config(
                103,
                launcher.DType.BFLOAT16,
                True,
                variant.activation == "silu_gate_3way",
                False,
                variant.has_mask,
                variant.has_residual,
                variant.width,
                variant.hidden,
                bucket,
            )
            assert (config.spec.target_sm, config.spec.kernel_sm) == (103, 100), (variant, bucket)
            assert config.spec.has_mask == variant.has_mask and config.spec.bucket == bucket
            images.setdefault((variant._replace(has_mask=False), bucket), set()).add(config.variant_id)
    assert all(len(ids) == 1 for ids in images.values())
    specs = [spec for spec in launcher.kernel_specs() if spec.target_sm == 103]
    assert {spec.kernel_sm for spec in specs} == {100} and len(specs) == len(images)


def test_sm103_launch_passes_every_check_before_the_device_check():
    """An SM103 image launched on another SM100-family GPU gets through the CUBIN adapter and every image and operand
    check of the C++ launcher, and stops at the device check, for every SM103 variant: masked and unmasked calls and
    the 3-way SwiGLU. On an SM103 GPU the CUBIN-mode reference tests launch these images instead."""
    _require_sm100()
    if SM_VERSION == 103:
        pytest.skip("the CUBIN-mode reference tests launch the SM103 images on this GPU")
    _require_sm100_cubins(103)
    library = require_cubin_library()
    for variant in shipped_variants(103):
        bucket = buckets(103, variant)[0]
        executable = transition_cubin.TransitionMlpCubinExecutable(
            library, library.transition_mlp, 103, torch.bfloat16, variant, bucket
        )
        x, w1, b1, w2, b2, residual, mask = _operands(variant, 1, 17)
        rows, width = x.numel() // variant.width, variant.width
        out = torch.empty(rows, width, device="cuda", dtype=torch.bfloat16)
        residual = None if residual is None else residual.reshape(rows, width)
        mask = None if mask is None else mask.reshape(rows)
        with pytest.raises(ValueError, match=f"selects SM103 but current device is SM{SM_VERSION}"):
            executable(x.reshape(rows, width), w1, b1, w2, b2, residual, mask, out)


def test_sm100_library_covers_every_tuned_config():
    """Every SM100 and SM103 variant and bucket resolves to an SM100-kernel image, and the masked and unmasked calls of
    a bucket resolve to the same one: the kernel takes the mask as an address. The config records the call's state."""
    _require_sm100()
    _require_sm100_cubins(100, 103)
    launcher = require_cubin_library().transition_mlp
    images = {}
    for sm in (100, 103):
        for variant in shipped_variants(sm):
            for bucket in buckets(sm, variant):
                config = launcher.make_kernel_config(
                    sm,
                    launcher.DType.BFLOAT16,
                    True,
                    variant.activation == "silu_gate_3way",
                    False,
                    variant.has_mask,
                    variant.has_residual,
                    variant.width,
                    variant.hidden,
                    bucket,
                )
                assert config.spec.kernel_sm == 100 and config.spec.has_mask == variant.has_mask
                images.setdefault((sm, variant._replace(has_mask=False), bucket), set()).add(config.variant_id)
    assert all(len(ids) == 1 for ids in images.values())
    # One image per variant without its mask state and bucket; each records no compile-time mask.
    specs = [spec for spec in launcher.kernel_specs() if spec.kernel_sm == 100]
    assert not any(spec.has_mask for spec in specs)
    keys = [(s.target_sm, s.width, s.hidden, s.bucket, s.is_three_way, s.has_residual) for s in specs]
    assert len(keys) == len(set(keys)) == len(images)


def _unfused_mlp(layer: ConditionedTransitionBlock, a: torch.Tensor) -> torch.Tensor:
    """The block's MLP without the fused op: the projection, the Triton SwiGLU, then ``b_to_a``."""
    op = layer._fused_mlp_op
    layer._fused_mlp_op = None
    try:
        return layer._swiglu_mlp(a)
    finally:
        layer._fused_mlp_op = op


# The 5SBJ, 7QSJ, 7WR3_A_C and 8IC7_A_B atom windows (see the 2-way test above).
@pytest.mark.parametrize("windows", [7, 91, 238, 421])
@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
def test_sm100_three_way_atom_block_runs_the_fused_mlp(mode, windows, monkeypatch):
    """The 3-way atom block's SwiGLU MLP takes the fused op on SM100 and matches fp32 as closely as its unfused path,
    which rounds the hidden activation to bf16 the same way."""
    _require_sm100()
    if mode == "cubin":
        _require_sm100_cubins(SM_VERSION)
    layer = _atom_block(using_silu=False)
    assert layer._fused_mlp_op is not None and layer._fused_mlp_op.variant.activation == "silu_gate_3way"
    a = torch.randn(1, 5, windows, 32, 128, device="cuda", dtype=torch.bfloat16)
    weight, projection = layer.fused_swl_a_to_b.weight, layer.b_to_a.weight
    with torch.inference_mode():
        assert layer._fused_mlp_op.accepts(a, None, weight, None, projection, None)
        fused = run_cutedsl_test_mode(
            mode,
            monkeypatch,
            TransitionMlpCuTe,
            transition_cutedsl,
            lambda: layer._fused_mlp_op(a, weight, None, projection, None, None, None),
        )
        unfused = _unfused_mlp(layer, a)
    reference = transition_mlp_reference(
        a.float(), weight.float(), None, projection.float(), None, None, None, activation="silu_gate_3way"
    )
    assert fused.shape == a.shape
    assert _max_abs_error(fused, reference) <= 2 * _max_abs_error(unfused, reference)


def test_sm100_three_way_atom_block_matches_its_unfused_path():
    """The whole 3-way block (AdaLN, MLP, output gate, residual) agrees with the block on its unfused MLP."""
    _require_sm100()
    layer = _atom_block(using_silu=False)
    a = torch.randn(1, 5, 29, 32, 128, device="cuda", dtype=torch.bfloat16)
    s = torch.randn(1, 1, 29, 32, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        fused = layer(a, s, residual=a)
        op = layer._fused_mlp_op
        layer._fused_mlp_op = None
        unfused = layer(a, s, residual=a)
        layer._fused_mlp_op = op
    update = (unfused.float() - a.float()).abs().max().item()
    assert (fused.float() - unfused.float()).abs().max().item() <= 0.05 * max(update, 1.0)


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
def test_sm100_masked_and_three_way_mlps_replay_under_cuda_graph_capture(mode, monkeypatch):
    """The atom-row calls the SM100 kernel gained, captured with the unmasked call that shares the masked call's image:
    the C = 128, H = 256 SwiGLU update with and without a row mask (a partial last tile) and the 3-way atom block's
    MLP. Replays of new inputs match the layers' unfused paths."""
    _require_sm100()
    if mode == "cubin":
        _require_sm100_cubins(SM_VERSION)
    torch.manual_seed(0)
    swiglu = _init(Transition(128, 256, dtype=torch.bfloat16, normalize=False).cuda())
    assert all(op is not None for op in swiglu._fused_mlp_ops.values())
    block = _atom_block(using_silu=False)
    x = torch.randn(1, 4100, 128, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(1, 4100, device="cuda") > 0.3).to(torch.bfloat16)
    a = torch.randn(1, 5, 23, 32, 128, device="cuda", dtype=torch.bfloat16)
    static_x, static_a = x.clone(), a.clone()

    def step():
        return swiglu(static_x, mask), swiglu(static_x), block._swiglu_mlp(static_a)

    with torch.inference_mode():
        run_cutedsl_test_mode(mode, monkeypatch, TransitionMlpCuTe, transition_cutedsl, lambda: step()[0])
        step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            masked, unmasked, three_way = step()
        x_new, a_new = torch.randn_like(x), torch.randn_like(a)
        static_x.copy_(x_new)
        static_a.copy_(a_new)
        graph.replay()
        torch.cuda.synchronize()
        expected_masked = swiglu._forward_impl(x_new, mask)
        expected_unmasked = swiglu._forward_impl(x_new, None)
        expected_three_way = _unfused_mlp(block, a_new)
    torch.testing.assert_close(masked, expected_masked, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(unmasked, expected_unmasked, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(three_way, expected_three_way, atol=5e-2, rtol=2e-2)
    assert (masked.float().abs().sum(-1) == 0).equal((mask == 0).expand_as(masked[..., 0]))
