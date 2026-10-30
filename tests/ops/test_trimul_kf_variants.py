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
"""SM90 TriMul KF per-op variants from source: K1's a/b layouts, K2 tiles on dense and padded planes, K3 variants."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bionemo_ir._torch.custom_ops import trimul_kf_k1 as k1
from bionemo_ir._torch.custom_ops import trimul_kf_k2 as k2
from bionemo_ir._torch.custom_ops import trimul_kf_k3 as k3
from bionemo_ir._torch.custom_ops.trimul_kf_k1 import _config as k1_config
from bionemo_ir._torch.custom_ops.trimul_kf_k1 import _cubin as k1_cubin
from bionemo_ir._torch.custom_ops.trimul_kf_k1 import cutedsl as k1_cutedsl
from bionemo_ir._torch.custom_ops.trimul_kf_k2 import _config as k2_config
from bionemo_ir._torch.custom_ops.trimul_kf_k2 import ops as k2_ops
from bionemo_ir._torch.custom_ops.trimul_kf_k3 import _config as k3_config
from tests._torch import skip_if_not_sm90, source_module_available

from .test_trimul_kf import _ATOL, _EPS, _k1_reference, _k2_reference, _problem, _reference, _widths

_K2 = k2_config.TrimulKFK2Tile
_requires_source = pytest.mark.skipif(
    not all(source_module_available(f"bionemo_ir._torch.custom_ops.trimul_kf_k{op}._source") for op in (1, 2, 3)),
    reason="forces trimul KF kernels that only compile from source",
)


def _folds(problem: dict) -> tuple[k1.TrimulKFInputFold, k3.TrimulKFOutputFold]:
    fold_in = k1.fold_input_weights(
        problem["norm_in_weight"],
        problem["norm_in_bias"],
        problem["p_in_weight"],
        problem["g_in_weight"],
        problem["p_in_bias"],
        problem["g_in_bias"],
        centred=k1.get_trimul_kf_k1_op(torch.bfloat16, *_widths(problem)).centred,
    )
    fold_out = k3.fold_output_weights(
        problem["norm_out_weight"],
        problem["norm_out_bias"],
        problem["norm_in_weight"],
        problem["norm_in_bias"],
        problem["p_out_weight"],
        problem["g_out_weight"],
        problem["p_out_bias"],
        problem["g_out_bias"],
    )
    return fold_in, fold_out


def _k1(problem: dict, layout: str | None = None) -> k1.TrimulKFK1Output:
    """K1's own selection for the problem, or that variant forced onto ``layout``."""
    x = problem["x"]
    op1 = k1.get_trimul_kf_k1_op(torch.bfloat16, *_widths(problem))
    fold_in, _ = _folds(problem)
    if layout is None:
        return op1(x, problem["seqlen"], fold_in, _EPS)
    selection = op1.select(x.shape[1])._replace(ab_layout=layout)
    w_in, w_gate_in = (fold_in.interleaved, None) if selection.interleaved else (fold_in.proj, fold_in.gate)
    return op1.backend.run(x, problem["seqlen"], w_in, w_gate_in, fold_in.vec, _EPS, selection)


def _chain(
    problem: dict,
    outgoing: bool,
    layout: str | None = None,
    tile=None,
    k3_variant: str | None = None,
    pitched: bool | None = None,
):
    """The chain as the node runs it, with K1's layout, K2's tile, K3's variant or K2's product pitch optionally
    forced: by default K2 writes the product at K1's pitch where K3's rule reads a pitch."""
    x = problem["x"]
    (C, D), N = _widths(problem), x.shape[1]
    a, b, stats = _k1(problem, layout)
    op3 = k3.get_trimul_kf_k3_op(torch.bfloat16, C, D, True)
    selection = op3.select(N) if k3_variant is None else op3.select(N)._replace(kernel_variant=k3_variant)
    pitched = selection.reads_pitch if pitched is None else pitched
    op2 = k2.get_trimul_kf_k2_op(torch.bfloat16, D, outgoing)
    prod_pitch = a.stride(2) if pitched else N
    prod = op2(a, b, pitched=pitched) if tile is None else op2.backend.run(a, b, outgoing, tile, prod_pitch)
    _, fold_out = _folds(problem)
    if k3_variant is None:
        return op3(prod, x, fold_out, stats, _EPS, residual=True, actual_seqlen=problem["out_seqlen"])
    out = torch.empty_like(x)
    return op3.backend.run(prod, x, *fold_out.weights(selection), stats, problem["out_seqlen"], _EPS, selection, out)


# --- selection (no GPU work) --------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", sorted(k1_config.shipped_shapes(90)))
def test_k1_layout_comes_from_the_token_count_alone(shape):
    """K1 pads exactly where its rule says, on variants with a padded flavour."""
    C, D = shape
    rules = k1_config.layout_rules(90, C, D)
    for n in range(8, 4097, 8):
        selection = k1_config.select(90, C, D, n)
        assert selection.ab_layout == k1_config.ab_layout(rules[selection.bucket], n), n
        P = selection.ab_pitch(n)
        assert P % 8 == 0 and P >= n, n
        if selection.ab_layout == "padded":
            assert P % k1_config.AB_PAD_ALIGN == 0 and P > n, n
            assert selection.kernel_variant in k1_config.PADDED_VARIANTS


def test_k1_writes_padded_planes_only_where_this_build_can(monkeypatch):
    """The padded flavour compiles from source, and runs from a packaged CUBIN only where one ships."""
    from bionemo_ir._torch.custom_ops.trimul_kf_k1.cutedsl import TrimulKFK1CuTe
    from bionemo_ir._torch.utils.kernel import CuTeDSLKernelLibraryError
    from bionemo_ir._torch.utils.kernel._cutedsl_kernel_library import _load_kernel_library

    # supports() caches the CUBIN executables it probes; keep them out of the source path's cache.
    monkeypatch.setattr(TrimulKFK1CuTe, "_compiled_cache", {})
    backend = TrimulKFK1CuTe(90)
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    assert backend.supports(256, 256, "K1_2", "padded") and backend.supports(256, 256, "K1_2", "dense")
    try:
        specs = _load_kernel_library().trimul_kf_k1.kernel_specs()
    except (CuTeDSLKernelLibraryError, AttributeError):
        specs = []
    padded_images = {(spec.C, spec.D, f"K1_{spec.kernel_variant}") for spec in specs if spec.padded}
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    for C, D in sorted(k1_config.shipped_shapes(90)):
        for variant in k1_config.KERNEL_VARIANTS:
            assert backend.supports(C, D, variant, "padded") == ((C, D, variant) in padded_images), (C, D, variant)
            assert backend.supports(C, D, variant, "dense")


def test_k1_layout_rules():
    """dense never pads; pad_n_mod_16_8 pads every N % 16 == 8 its anchor covers, at any size."""
    assert [n for n in range(8, 1281, 8) if k1_config.ab_layout("pad_n_mod_16_8", n) == "padded"] == [
        n for n in range(8, 1281, 8) if n % 16 == 8
    ]
    assert all(k1_config.ab_layout("dense", n) == "dense" for n in range(8, 1281, 8))
    assert k1_config.ab_layout("pad_n_mod_16_8", 2904) == "padded"
    rules = k1_config.parse_layout_rules(
        {"S=128": {"kernel_variant": "K1_0"}, "S=256": {"kernel_variant": "K1_1", "ab_layout": "pad_n_mod_16_8"}}
    )
    assert rules == {128: "dense", 256: "pad_n_mod_16_8"}
    for bad in (
        {"kernel_variant": "K1_1", "ab_layout": "strided"},
        {"kernel_variant": "K1_1", "ab_layout": "pad"},
        {"kernel_variant": "K1_1", "pad": True},
        {"kernel_variant": "K1_1", "ab_layout": "pad_n_mod_16_8", "ab_max_waste": 0.05},
    ):
        with pytest.raises(ValueError):
            k1_config.parse_configs({"S=256": bad})


#: Where each shipped K1 bundle starts padding N % 16 == 8 token counts: its anchors set it (a dense anchor at S and a
#: padding one at 2 M - S pad from M up). The D=256 bundles (C=256 and C=384) pad from 496: from N = 504 the padded
#: planes (and, where N % 64 >= 56, the pitched product K2 writes on them) speed the whole chain up over both
#: directions; at N = 488 and below, K2's dynamic schedule runs the dense planes faster.
_K1_PAD_FROM = {(32, 32): 512, (64, 64): 512, (128, 128): 512, (256, 256): 496, (384, 256): 496}


def test_shipped_k1_bundles_pad_where_their_anchors_say():
    """Each shipped bundle pads exactly the N % 16 == 8 token counts from its own threshold up."""
    assert set(_K1_PAD_FROM) == set(k1_config.shipped_shapes(90))
    for (C, D), start in _K1_PAD_FROM.items():
        padded = [n for n in range(8, 4097, 8) if k1_config.select(90, C, D, n).ab_layout == "padded"]
        assert padded == [n for n in range(start, 4097, 8) if n % 16 == 8], (C, D)


def test_k2_tile_lists_take_the_tile_that_pads_n_least():
    """An anchor naming several tiles runs the one with the fewest padded columns, ties going to the wider."""
    narrow = {"kernel_variant": "K2_1", "tile_n": 192}
    wide = {"kernel_variant": "K2_0", "tile_n": 256, "cluster_m": 2, "defer_kmin": 4, "split_epi": True}
    tiles = k2_config.parse_configs({"S=320": [narrow, wide]})[320]
    assert tiles == (_K2("K2_0", 256, 2, 4, True), _K2("K2_1", 192))
    assert [n for n in range(232, 1281, 8) if k2_config.least_padding(tiles, n).tile_n == 192] == [
        *range(264, 385, 8),
        *range(520, 577, 8),
        *range(776, 961, 8),
        *range(1032, 1153, 8),
    ]
    assert set(k2_config.shipped_tiles([tiles])) == {*tiles, _K2("K2_1", 128)}
    for bad in ([wide], [wide, wide], [wide, {**wide, "split_epi": False}], [[narrow, wide]], []):
        with pytest.raises(ValueError):
            k2_config.parse_configs({"S=320": bad})


#: Distinct K2 tiles a bundle may run per direction (outgoing, incoming) over all token counts: the CUBIN budget. D=256
#: adds cluster_m 1 tiles for 513-988 tokens (RFD4 and Protenix): two more outgoing, one more incoming.
_K2_TILE_BUDGET = {256: (6, 5)}


@pytest.mark.parametrize("D", sorted(k2_config.shipped_widths(90)))
def test_k2_tables_stay_within_the_cubin_budget(D):
    """Every shipped bundle runs at most four distinct tiles per direction over all token counts (CUBIN budget); D=256
    at most six outgoing and five incoming."""
    for outgoing, budget in zip((True, False), _K2_TILE_BUDGET.get(D, (4, 4)), strict=True):
        assert len({k2_config.select(90, D, n, outgoing)[1] for n in range(8, 4097, 8)}) <= budget


#: D=256's K2 tile for every token count from 449 to 1100, per direction, as (first, last, tile). The single-CTA tiles
#: run outgoing at 513-672 (224), 769-832 (208) and 833-896 (224), incoming at 769-896 (224); incoming 897-952 moved
#: from the single-CTA 192 to the 256 pair; every other count runs the tile it ran before.
_D256_K2_MAP = {
    True: [
        (449, 512, _K2("K2_0", 256, 2, 4, False, True)),
        (513, 672, _K2("K2_0", 224, 1, 3, True, True)),
        (673, 768, _K2("K2_0", 256, 2, 4, False, True)),
        (769, 832, _K2("K2_0", 208, 1, 3, True, True)),
        (833, 896, _K2("K2_0", 224, 1, 3, True, True)),
        (897, 1024, _K2("K2_0", 256, 2, 4, False, True)),
        (1025, 1100, _K2("K2_0", 224, 2, 3, True, True)),
    ],
    False: [
        (449, 512, _K2("K2_0", 256, 2, 4, True, True)),
        (513, 576, _K2("K2_0", 192, 1, 3, True, True)),
        (577, 768, _K2("K2_0", 256, 2, 4, True, True)),
        (769, 896, _K2("K2_0", 224, 1, 3, True, True)),
        (897, 1100, _K2("K2_0", 256, 2, 4, True, True)),
    ],
}


@pytest.mark.parametrize("outgoing", [True, False], ids=["out", "in"])
def test_k2_d256_runs_the_mapped_tile_at_every_token_count(outgoing):
    """Every token count from 449 to 1100 selects D=256's mapped K2 tile, both edges of every run included. The
    S=776 anchor hands over to S=1200 between 988 and 989 tokens inside a run in both directions; the tail-anchor test
    pins that hand-over."""
    for first, last, tile in _D256_K2_MAP[outgoing]:
        for n in range(first, last + 1):
            assert k2_config.select(90, 256, n, outgoing)[1] == tile, n


def test_k2_d256_tail_anchor_keeps_the_large_n_tiles_where_it_took_over():
    """D=256's S=776 anchor serves 741-988 tokens; from 953 tokens, which S=1200 served before, it runs the tiles
    S=1200 would run, so only 741-952 tokens change."""
    for outgoing in (True, False):
        entries = k2_config.anchors(90, 256, outgoing)
        assert k2_config.select(90, 256, 741, outgoing)[0] == 776
        assert k2_config.select(90, 256, 989, outgoing)[0] == 1200
        for n in range(953, 989):
            shipped = k2_config.select(90, 256, n, outgoing)[1]
            assert shipped == k2_config.runtime_tile(k2_config.least_padding(entries[1200], n), n)


def test_launches_keep_tiles_in_one_batch_and_coordinates_in_int32():
    """A kernel tiling a launch's rows flat shares a launch across batches unless a 128-row tile would straddle two,
    or the rows would pass int32 coordinates. K1 tiles each batch on its own: only int32 coordinates split it."""
    from bionemo_ir._torch.custom_ops.trimul_kf_k1.cutedsl import MAX_LAUNCH_ROWS, batch_chunks, launch_chunks

    assert batch_chunks(4, 3008) == launch_chunks(4, 3008) == ((0, 4),)  # flat extents past 2^31 stay in one launch
    assert batch_chunks(2, 136) == ((0, 1), (1, 2))  # 136 * 136 rows end mid-tile
    assert launch_chunks(2, 136) == ((0, 2),)
    assert launch_chunks(4, 504, 512) == ((0, 4),)
    assert batch_chunks(3, 32768) == launch_chunks(3, 32768) == ((0, 1), (1, 2), (2, 3))  # 2^30 rows a batch
    # 32760 * 32760 rows end mid-tile; two batches' whole tiles still fit int32
    assert launch_chunks(3, 32760) == ((0, 2), (2, 3))
    assert 46336 * 46336 <= MAX_LAUNCH_ROWS
    for n, pitch in ((46344, None), (46336, 46400)):
        for chunks in (batch_chunks, launch_chunks):
            with pytest.raises(ValueError):
                chunks(1, n, pitch)


def test_padded_pitch_rounds_rows_to_128_bytes():
    assert [k1.ab_pitch(n, "padded") for n in (128, 136, 520, 824, 832, 1032)] == [128, 192, 576, 832, 832, 1088]
    assert all(k1.ab_pitch(n) == n for n in (8, 136, 824))
    with pytest.raises(ValueError):
        k1.ab_pitch(128, "strided")


# --- on the GPU ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("B", "N", "C", "D", "outgoing", "lengths"),
    [
        (1, 8, 64, 64, True, "full"),
        (2, 16, 128, 128, False, "random"),
        (3, 24, 256, 256, True, "tail"),
        (3, 24, 384, 256, False, "tail"),
        (1, 40, 32, 32, False, "random"),
        (2, 264, 128, 128, True, "random"),
        (2, 264, 256, 256, False, "tail"),
        (2, 264, 384, 256, True, "random"),
        (1, 520, 256, 256, True, "random"),
        (1, 776, 128, 128, False, "random"),
    ],
    ids=lambda v: str(v),
)
def test_chain_matches_reference_at_small_n_and_batches(B, N, C, D, outgoing, lengths):
    """Every op on its own selection, small N and B > 1 included."""
    skip_if_not_sm90()
    problem = _problem(B, N, C, True, lengths, D=D)
    out = _chain(problem, outgoing)
    torch.testing.assert_close(out.float(), _reference(problem, outgoing, True), atol=_ATOL, rtol=0)


@pytest.mark.parametrize(
    ("B", "N", "C", "D", "outgoing"),
    [
        (1, 264, 256, 256, True),
        (2, 136, 128, 128, False),
        (1, 520, 128, 128, True),
        (2, 200, 64, 64, False),
        (1, 824, 256, 256, False),
        (2, 392, 384, 256, False),
    ],
    ids=lambda v: str(v),
)
@_requires_source
def test_padded_layout_is_bitwise_the_dense_one(B, N, C, D, outgoing):
    """K1's padded planes hold the dense values (a, b, stats) and the chain output does not change, bit for bit."""
    skip_if_not_sm90()
    problem = _problem(B, N, C, True, "random", D=D)
    dense, padded = _k1(problem, "dense"), _k1(problem, "padded")
    P = k1.ab_pitch(N, "padded")
    assert dense.a.is_contiguous() and padded.a.stride() == (D * P * P, P * P, P, 1)
    assert padded.b.stride() == padded.a.stride()
    assert torch.equal(dense.a, padded.a) and torch.equal(dense.b, padded.b)
    assert (dense.stats is None) == (padded.stats is None)
    if dense.stats is not None:  # rows past N * N pad the buffer to whole 128-row tiles; neither layout writes them
        assert torch.equal(dense.stats[:, : N * N], padded.stats[:, : N * N])
    out_dense, out_padded = _chain(problem, outgoing, "dense"), _chain(problem, outgoing, "padded")
    assert torch.equal(out_dense, out_padded)
    torch.testing.assert_close(out_padded.float(), _reference(problem, outgoing, True), atol=_ATOL, rtol=0)


# (B, N, C, D): K1_2 dense and padded at C = 384 (N * N % 128 == 64 for all three N), K1_0 and K1_1 dense, K1_1
# padded, and a batch whose rows end on a tile.
_ONE_LAUNCH_CASES = [
    (4, 104, 384, 256),
    (3, 200, 384, 256),
    (2, 504, 384, 256),
    (2, 136, 128, 128),
    (3, 200, 64, 64),
    (2, 520, 64, 64),
    (2, 128, 384, 256),
]


@pytest.mark.parametrize(("B", "N", "C", "D"), _ONE_LAUNCH_CASES, ids=lambda v: str(v))
@_requires_source
def test_k1_one_launch_matches_per_batch_launches_bitwise(B, N, C, D, monkeypatch):
    """K1 tiles each batch on its own, so one launch over B batches writes every batch's a, b and row statistics
    exactly as a launch over that batch alone, whether or not its N * N rows end on a 128-row tile."""
    skip_if_not_sm90()
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    problem = _problem(B, N, C, True, "random", D=D)
    op1 = k1.get_trimul_kf_k1_op(torch.bfloat16, C, D)
    fold_in, _ = _folds(problem)
    x, seqlen = problem["x"], problem["seqlen"]
    launches = []
    launch = k1_cutedsl.launch_compiled_kernel
    monkeypatch.setattr(k1_cutedsl, "launch_compiled_kernel", lambda *args: launches.append(args[11]) or launch(*args))
    out = op1(x, seqlen, fold_in, _EPS)
    assert launches == [B]
    for b in range(B):
        one = op1(x[b : b + 1], seqlen[b : b + 1], fold_in, _EPS)
        assert torch.equal(out.a[b], one.a[0]) and torch.equal(out.b[b], one.b[0]), b
        assert (out.stats is None) == (one.stats is None)
        if out.stats is not None:  # K3 reads each batch's first N * N rows
            assert torch.equal(out.stats[b, : N * N], one.stats[0, : N * N]), b


# (B, N, C, D): K1_2's dense and padded planes at C = 384, where CTAs split tiles between their runs of
# (tile, channel group) units and a store warp takes each unit's output from the MMA warpgroups in turn.
_K1_REPEAT_CASES = [(2, 200, 384, 256), (1, 1016, 384, 256)]


@pytest.mark.parametrize(("B", "N", "C", "D"), _K1_REPEAT_CASES, ids=lambda v: str(v))
@_requires_source
def test_k1_repeated_launches_write_the_same_bits(B, N, C, D, monkeypatch):
    """Every launch of K1 writes the same a, b and row statistics: each unit's output is stored once, after its
    warpgroup wrote it and before the next unit overwrites the buffer, whatever the timing between them."""
    skip_if_not_sm90()
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    problem = _problem(B, N, C, True, "tail", D=D)
    op1 = k1.get_trimul_kf_k1_op(torch.bfloat16, C, D)
    fold_in, _ = _folds(problem)
    x, seqlen = problem["x"], problem["seqlen"]
    first = op1(x, seqlen, fold_in, _EPS)
    for launch in range(8):
        again = op1(x, seqlen, fold_in, _EPS)
        assert torch.equal(again.a, first.a) and torch.equal(again.b, first.b), launch
        assert torch.equal(again.stats[:, : N * N], first.stats[:, : N * N]), launch


@pytest.mark.parametrize(("B", "N"), [(2, 200), (2, 504)])
@_requires_source
def test_k1_2_on_its_centred_fold_tracks_the_reference_as_closely_as_a_mean_term_k1(B, N, monkeypatch):
    """K1_2 reads centred weights and applies no LayerNorm mean term: its a and b are as close to fp32 as those of
    K1_0, which applies the mean term to the plain fold, and masked positions stay exactly zero; dense planes
    (N = 200) and padded (N = 504)."""
    skip_if_not_sm90()
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    C, D = 384, 256
    problem = _problem(B, N, C, True, "random", D=D)
    x, seqlen = problem["x"], problem["seqlen"]
    op1 = k1.get_trimul_kf_k1_op(torch.bfloat16, C, D)
    selection = op1.select(N)
    assert selection.kernel_variant == "K1_2" and selection.centred
    names = ("norm_in_weight", "norm_in_bias", "p_in_weight", "g_in_weight", "p_in_bias", "g_in_bias")
    weights = [problem[name] for name in names]
    a, b, _ = op1(x, seqlen, k1.fold_input_weights(*weights, centred=True), _EPS)
    plain = k1.fold_input_weights(*weights)
    mean_term = selection._replace(kernel_variant="K1_0", ab_layout="dense")
    ma, mb, _ = op1.backend.run(x, seqlen, plain.proj, plain.gate, plain.vec, _EPS, mean_term)
    ra, rb = _k1_reference(problem)
    masked = ~(torch.arange(N, device="cuda") < seqlen.reshape(B, 1, N, 1)).expand(B, D, N, N)
    for got, mean_term_got, ref in ((a, ma, ra), (b, mb, rb)):
        torch.testing.assert_close(got.float(), ref, atol=_ATOL, rtol=1e-2)
        assert (got.float() - ref).abs().mean() <= 1.05 * (mean_term_got.float() - ref).abs().mean()
        assert not got[masked].any()


class _RecordingK1Cubin(k1_cubin.TrimulKFK1CubinExecutable):
    """A K1 CUBIN executable with the given launch ABI that records each launch's batch count instead of launching."""

    def __init__(self, launch_abi: str) -> None:
        config = SimpleNamespace(cubin=SimpleNamespace(launch_abi=launch_abi))
        launcher = SimpleNamespace(make_kernel_config=lambda *args, **kwargs: config, DType=SimpleNamespace(BFLOAT16=0))
        super().__init__(None, launcher, 90, 384, 256, "K1_2")
        self.batches: list[int] = []

    def __call__(self, *args: object) -> None:
        self.batches.append(args[10])


@pytest.mark.parametrize(
    ("launch_abi", "batches"), [("trimul_kf_k1_sm90_v3", [1, 1]), ("trimul_kf_k1_sm90_v4", [2])], ids=["v3", "v4"]
)
def test_k1_images_that_tile_launch_rows_flat_keep_one_launch_per_batch(launch_abi, batches, monkeypatch):
    """A CUBIN image built before K1 tiled each batch on its own still gets one launch per batch where a tile would
    straddle two; current images take every batch in one launch."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    executable = _RecordingK1Cubin(launch_abi)
    assert executable.flat_tiles == (launch_abi == "trimul_kf_k1_sm90_v3")
    backend = k1_cutedsl.TrimulKFK1CuTe(90)
    monkeypatch.setattr(backend, "_get_or_compile", lambda *args, **kwargs: executable)
    B, N, C, D = 2, 104, 384, 256
    x = torch.zeros(B, N, N, C, device="cuda", dtype=torch.bfloat16)
    seqlen = torch.full((B, N), N, device="cuda", dtype=torch.int32)
    w_in = torch.zeros(4 * D, C, device="cuda", dtype=torch.bfloat16)
    vec = torch.zeros(8 * D, device="cuda")
    backend.run(x, seqlen, w_in, None, vec, _EPS, k1_config.TrimulKFK1Selection(128, "K1_2"))
    assert executable.batches == batches


@pytest.mark.parametrize("outgoing", [True, False], ids=["out", "in"])
def test_k2_reads_dense_and_padded_planes(outgoing):
    """K2 takes the pitch off the strides: dense and padded planes give one product; mismatched layouts are refused."""
    skip_if_not_sm90()
    C, N, P = 64, 136, 192
    op2 = k2.get_trimul_kf_k2_op(torch.bfloat16, C, outgoing)
    planes = torch.randn(2, 2, C, P, P, device="cuda").to(torch.bfloat16)
    a, b = planes[..., :N, :N].unbind(0)
    assert op2.accepts(a, b)
    prod = op2(a, b)
    assert torch.equal(prod, op2(a.contiguous(), b.contiguous()))
    torch.testing.assert_close(prod.float(), _k2_reference(a, b, outgoing), atol=_ATOL, rtol=1e-2)
    assert not op2.accepts(a, b.contiguous())  # one pitch for both operands
    odd = torch.empty(2, C, N, N + 4, device="cuda", dtype=torch.bfloat16)[..., :N]
    assert not op2.accepts(odd, odd)  # rows 16-byte aligned


_OFF_128_TILES = [
    _K2("K2_0", 192, 2, 4, True),
    _K2("K2_0", 192, 2, 8, True),
    _K2("K2_0", 192, 2, 8, False),
    _K2("K2_0", 192, 1, 4, False),
    _K2("K2_0", 208, 2, 8, False),
    _K2("K2_0", 224, 2, 8, False),
    _K2("K2_2", 144, 2, 8, False),
    _K2("K2_2", 160, 2, 8, False),
    _K2("K2_2", 208, 2, 8, False),
    _K2("K2_2", 192, 2, 8, False),
]


@pytest.mark.parametrize("tile", _OFF_128_TILES, ids=str)
@pytest.mark.parametrize("outgoing", [True, False], ids=["out", "in"])
@_requires_source
def test_k2_tiles_off_128_columns_keep_their_store_ring_at_small_n(tile, outgoing):
    """A deferred-store tile whose store count is not a multiple of the ring (tile_n % 128 != 0) must not
    refill a buffer its previous bulk store still reads."""
    skip_if_not_sm90()
    D, N = 256, 520
    generator = torch.Generator(device="cuda").manual_seed(0)
    a, b = (torch.randn(1, D, N, N, device="cuda", generator=generator).to(torch.bfloat16) for _ in range(2))
    op2 = k2.get_trimul_kf_k2_op(torch.bfloat16, D, outgoing)
    reference_tile = _K2("K2_0", 256, 2, 8, False)
    expected = op2.backend.run(a, b, outgoing, reference_tile)
    torch.testing.assert_close(expected.float(), _k2_reference(a, b, outgoing), atol=_ATOL, rtol=1e-2)
    for _ in range(3):
        assert torch.equal(op2.backend.run(a, b, outgoing, tile), expected)


@pytest.mark.parametrize(
    "tile",
    [
        _K2("K2_0", 256, 2, 4, False, True),
        _K2("K2_0", 224, 2, 3, True, True),
        _K2("K2_2", 192, 2, 4, False, True),
        _K2("K2_2", 192, 2, 3, True, True),
        _K2("K2_0", 224, 1, 3, True, True),
        _K2("K2_0", 208, 1, 3, True, True),
    ],
    ids=str,
)
@pytest.mark.parametrize("outgoing", [True, False], ids=["out", "in"])
@_requires_source
def test_k2_dynamic_schedule_matches_the_static_one(tile, outgoing):
    """A dynamic tile claims its units from a counter: its product equals the static schedule's bitwise, launch
    after launch and under CUDA-graph replay, and every launch leaves the counter at zero."""
    skip_if_not_sm90()
    B, D, N = 2, 256, 384  # several units per cluster; both column tiles of a K2_2 pair hold columns
    generator = torch.Generator(device="cuda").manual_seed(0)
    a, b = (torch.randn(B, D, N, N, device="cuda", generator=generator).to(torch.bfloat16) for _ in range(2))
    op2 = k2.get_trimul_kf_k2_op(torch.bfloat16, D, outgoing)
    expected = op2.backend.run(a, b, outgoing, tile._replace(dynamic=False))
    torch.testing.assert_close(expected.float(), _k2_reference(a, b, outgoing), atol=_ATOL, rtol=1e-2)
    counter = torch.zeros(k2_config.COUNTER_WORDS, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError):
        op2.backend.run(a, b, outgoing, tile)
    for _ in range(3):
        assert torch.equal(op2.backend.run(a, b, outgoing, tile, None, counter), expected)
    assert not counter.any()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        prod = op2.backend.run(a, b, outgoing, tile, None, counter)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(prod, expected)
    assert not counter.any()


@pytest.mark.parametrize("outgoing", [True, False], ids=["out", "in"])
def test_k2_graphs_captured_on_two_streams_replay_concurrently(outgoing):
    """Each capture owns its dynamic schedule's counter: graphs captured on two streams replay concurrently and give
    the eager product bit for bit, and neither shares a counter with the other or with eager launches, which share
    their stream's."""
    skip_if_not_sm90()
    B, D, N = 2, 256, 384
    generator = torch.Generator(device="cuda").manual_seed(0)
    a, b = (torch.randn(B, D, N, N, device="cuda", generator=generator).to(torch.bfloat16) for _ in range(2))
    op2 = k2.get_trimul_kf_k2_op(torch.bfloat16, D, outgoing)
    assert op2.select(N).dynamic
    expected = op2(a, b)
    eager = k2_ops.work_counter(a.device)
    assert k2_ops.work_counter(a.device) is eager
    streams = [torch.cuda.Stream() for _ in range(2)]
    graphs, products, counters = [], [], []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            products.append([op2(a, b) for _ in range(3)])
            counters.append(k2_ops.work_counter(a.device))
        graphs.append(graph)
    assert counters[0] is not counters[1] and all(counter is not eager for counter in counters)
    for _ in range(4):
        # poison every product, so a unit a replay misses cannot keep an earlier replay's value
        for prod in (prod for prods in products for prod in prods):
            prod.fill_(float("nan"))
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        for graph, stream in zip(graphs, streams, strict=True):
            with torch.cuda.stream(stream):
                graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(prod, expected) for prods in products for prod in prods)
    assert not any(counter.any() for counter in (eager, *counters))


@pytest.mark.parametrize("shared_pool", [False, True], ids=["own-pools", "shared-pool"])
def test_k2_keeps_one_capture_counter_per_stream_and_every_graph_replays(shared_pool):
    """A capture's counter is its graph's intermediate: the op lets go of it once the stream captures again, so
    repeated captures hold one counter per stream, and every graph still gives the eager product bit for bit."""
    skip_if_not_sm90()
    B, D, N = 1, 256, 384
    generator = torch.Generator(device="cuda").manual_seed(1)
    a, b = (torch.randn(B, D, N, N, device="cuda", generator=generator).to(torch.bfloat16) for _ in range(2))
    op2 = k2.get_trimul_kf_k2_op(torch.bfloat16, D, True)
    assert op2.select(N).dynamic
    expected = op2(a, b)
    stream = torch.cuda.Stream()
    pool = torch.cuda.graph_pool_handle() if shared_pool else None
    graphs, products = [], []
    for _ in range(4):
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            products.append([op2(a, b) for _ in range(2)])
        graphs.append(graph)
    held = [key for key in k2_ops._work_counters if len(key) > 2 and key[:2] == (a.device.index, stream.cuda_stream)]
    assert len(held) == 1
    # one graph at a time, in capture order, as graphs sharing a pool must replay
    for graph, prods in zip(graphs, products, strict=True):
        for prod in prods:
            prod.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(prod, expected) for prod in prods)


@pytest.mark.parametrize(("N", "C", "D"), [(264, 256, 256), (520, 128, 128), (392, 384, 256)])
@_requires_source
def test_k3_variants_stay_in_the_error_class(N, C, D):
    """Every K3 variant pairing with K1's selection matches the reference as well as the shipped one."""
    skip_if_not_sm90()
    problem = _problem(1, N, C, True, "random", D=D)
    reference = _reference(problem, True, True)
    writes_stats = k1_config.select(90, C, D, N).writes_stats
    errors = {}
    for variant in k3_config.KERNEL_VARIANTS:
        if (variant in k3_config.STATS_VARIANTS) != writes_stats or not k3_config.supports(variant, C, D):
            continue
        out = _chain(problem, True, k3_variant=variant).float()
        torch.testing.assert_close(out, reference, atol=_ATOL, rtol=0)
        errors[variant] = ((out - reference).norm() / reference.norm()).item()
    shipped = k3_config.select(90, C, D, N).kernel_variant
    assert len(errors) >= 2 and all(error <= 1.25 * errors[shipped] for error in errors.values()), errors


def test_k3_4_runs_every_n_of_c384_d256():
    """One output-side kernel for C=384, D=256: a single anchor names K3_4, so every N selects it."""
    assert k3_config.anchors(90, 384, 256) == {128: "K3_4"}
    assert {k3_config.select(90, 384, 256, n).kernel_variant for n in range(1, 4097)} == {"K3_4"}


def test_centred_output_weights_keep_the_layernorm_mean_out():
    """K3_4's weights: rows centred to a near-zero sum (so the input mean drops out), permuted within 128-row tiles."""
    C, D = 384, 256
    generator = torch.Generator().manual_seed(0)
    w_out = (torch.randn(C, D, generator=generator) * D**-0.5).to(torch.bfloat16)
    w_gate = (torch.randn(C, C, generator=generator) * C**-0.5).to(torch.bfloat16)
    fold = k3.TrimulKFOutputFold(w_out, w_gate, torch.zeros(4 * C))
    order = k3.ops._centred_row_order(C, w_out.device)
    assert sorted(order.tolist()) == list(range(C))
    for centred, folded in zip(fold.centred, (w_out, w_gate), strict=True):
        exact = folded.double() - folded.double().mean(-1, keepdim=True)
        rows = centred.double()[order.argsort()]
        assert (rows.sum(-1).abs() / rows.abs().sum(-1)).max() < 1e-5
        # within 1.5 bf16 steps of the exact centred value (the step of the larger magnitude, across a binade edge)
        step = torch.exp2(torch.floor(torch.log2(torch.maximum(rows.abs(), exact.abs()).clamp_min(1e-30))) - 7)
        assert ((rows - exact).abs() <= 1.5 * step).all()
    selection = k3_config.TrimulKFK3Selection(128, "K3_4")
    assert fold.weights(selection)[:2] == fold.centred and selection.centred and selection.one_launch


@pytest.mark.parametrize(("B", "N"), [(3, 104), (2, 136), (1, 64), (4, 8)], ids=lambda v: str(v))
@_requires_source
def test_shipped_k3_shares_one_launch_across_batches(B, N):
    """The shipped C=384, D=256 K3 runs every batch in one launch, its 128-row slabs straddling batch items where
    N * N % 128 != 0."""
    skip_if_not_sm90()
    variant = k3_config.select(90, 384, 256, N).kernel_variant
    assert variant in k3_config.ONE_LAUNCH_VARIANTS
    problem = _problem(B, N, 384, True, "random", D=256)
    out = _chain(problem, True, k3_variant=variant).float()
    torch.testing.assert_close(out, _reference(problem, True, True), atol=_ATOL, rtol=0)


@pytest.mark.parametrize("layout", ["dense", "padded"])
@pytest.mark.parametrize(
    ("B", "N", "lengths"), [(4, 200, (200, 7, 130, 0)), (2, 104, (100, 60)), (3, 136, (136, 0, 50))], ids=str
)
@_requires_source
def test_shipped_k3_skips_all_masked_slabs_bit_for_bit(B, N, lengths, layout):
    """The shipped C=384, D=256 K3 skips the loads and GEMMs of slabs whose rows are all masked (padding rows, short
    batch items padded to the batch's N) and still writes the full path's output, bit for bit: exact zeros there.
    On K1's padded planes K2 writes a pitched product and K3 skips per-row slabs."""
    skip_if_not_sm90()
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream

    from bionemo_ir._torch.custom_ops.trimul_kf_k3 import _source
    from bionemo_ir.dsl_kernels.cute import sm90_trimul_kf_k3

    backend = k3.get_trimul_kf_k3_op(torch.bfloat16, 384, 256, True).backend
    if backend.force_cubin():
        pytest.skip("compares two source builds of the shipped K3")
    variant = k3_config.select(90, 384, 256, N).kernel_variant
    kernel_cls = getattr(sm90_trimul_kf_k3, f"Sm90TrimulKF_{variant}")
    problem = _problem(B, N, 384, True, "full", D=256)
    out_seqlen = torch.zeros(B, N, dtype=torch.int32, device="cuda")
    for b, length in enumerate(lengths):
        out_seqlen[b, :length] = length  # rows i >= length are padding: every one of their slots is masked
    problem["out_seqlen"] = out_seqlen
    assert kernel_cls.skip_masked_slabs
    pitched = layout == "padded"
    assert not pitched or variant in k3_config.PITCHED_VARIANTS
    skipped = _chain(problem, True, layout, k3_variant=variant, pitched=pitched)
    torch.testing.assert_close(skipped.float(), _reference(problem, True, True), atol=_ATOL, rtol=0)
    mask = (torch.arange(N, device="cuda") < out_seqlen.reshape(B, N, 1)).reshape(B, N, N, 1).expand_as(skipped)
    assert bool((skipped[~mask] == 0).all())
    # the same kernel with the skip off, swapped into the backend for one call
    full = type(f"{kernel_cls.__name__}_full", (kernel_cls,), {"skip_masked_slabs": False})
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    executable = cute.compile(
        full(384, 256, num_sms, True, **({"pitched": True} if pitched else {})),
        *_source.make_fake_args(384, 256, variant, True),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        *_source.make_fake_pitch_args(pitched),
        options="--enable-tvm-ffi",
    )
    key = backend._key(384, 256, variant, True, pitched)
    cache = type(backend)._compiled_cache
    shipped = cache.get(key)
    cache[key] = executable
    try:
        unskipped = _chain(problem, True, layout, k3_variant=variant, pitched=pitched)
    finally:
        cache[key] = shipped
    assert torch.equal(skipped, unskipped)


@pytest.mark.parametrize("switch", ["split_ring", "wg_offset", "p_inflight"])
@pytest.mark.parametrize("layout", ["dense", "padded"])
@pytest.mark.parametrize(
    ("B", "N", "lengths"), [(1, 400, (400,)), (4, 200, (200, 7, 130, 0)), (3, 136, (136, 0, 50))], ids=str
)
@_requires_source
def test_shipped_k3_schedule_switches_write_the_same_bits(B, N, lengths, layout, switch):
    """The shipped C=384, D=256 K3's schedule switches change when work is issued, not what is computed: ring
    barriers per consumer warpgroup, the second warpgroup's half-slab offset and one P-stage WGMMA group in flight
    each move waits, not arithmetic. The output is bit for bit the one the same kernel writes with the switch
    flipped, on dense and on K1's padded planes (per-row slabs), all-masked slabs included."""
    skip_if_not_sm90()
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream

    from bionemo_ir._torch.custom_ops.trimul_kf_k3 import _source
    from bionemo_ir.dsl_kernels.cute import sm90_trimul_kf_k3

    backend = k3.get_trimul_kf_k3_op(torch.bfloat16, 384, 256, True).backend
    if backend.force_cubin():
        pytest.skip("compares two source builds of the shipped K3")
    variant = k3_config.select(90, 384, 256, N).kernel_variant
    kernel_cls = getattr(sm90_trimul_kf_k3, f"Sm90TrimulKF_{variant}")
    problem = _problem(B, N, 384, True, "full", D=256)
    problem["out_seqlen"] = _out_seqlen(B, N, lengths)
    pitched = layout == "padded"
    shipped_out = _chain(problem, True, layout, k3_variant=variant, pitched=pitched)
    torch.testing.assert_close(shipped_out.float(), _reference(problem, True, True), atol=_ATOL, rtol=0)
    flipped = {switch: not getattr(kernel_cls, switch)}
    plain = type(f"{kernel_cls.__name__}_flipped", (kernel_cls,), flipped)
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    executable = cute.compile(
        plain(384, 256, num_sms, True, **({"pitched": True} if pitched else {})),
        *_source.make_fake_args(384, 256, variant, True),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        *_source.make_fake_pitch_args(pitched),
        options="--enable-tvm-ffi",
    )
    key = backend._key(384, 256, variant, True, pitched)
    cache = type(backend)._compiled_cache
    shipped = cache.get(key)
    cache[key] = executable
    try:
        flipped_out = _chain(problem, True, layout, k3_variant=variant, pitched=pitched)
    finally:
        cache[key] = shipped
    assert torch.equal(shipped_out, flipped_out)


def _out_seqlen(B: int, N: int, lengths: tuple) -> torch.Tensor:
    """Output prefix lengths: rows i >= lengths[b] of batch item b are padding, every one of their slots masked."""
    out_seqlen = torch.zeros(B, N, dtype=torch.int32, device="cuda")
    for b, length in enumerate(lengths):
        out_seqlen[b, :length] = length
    return out_seqlen


@pytest.mark.parametrize(
    ("B", "N", "lengths"), [(1, 400, "random"), (2, 304, "tail"), (3, 136, (136, 0, 50)), (2, 104, (100, 60))], ids=str
)
@_requires_source
def test_k3_4_packed_output_step_stays_in_the_error_class(B, N, lengths):
    """K3_4's packed bf16 output step (gate and ``rstd_p o + b_o`` rounded to bf16, one ``fma.rn.bf16x2`` with the
    residual) matches the fp32 reference within the KF tolerance and within 1.25x the error of K3_3, which reads
    the same statistics through an fp32 output step; its masked rows, all-masked slabs included, are exact zeros."""
    skip_if_not_sm90()
    if k3.get_trimul_kf_k3_op(torch.bfloat16, 384, 256, True).backend.force_cubin():
        pytest.skip("compares against a source build of K3_3, which ships no C=384 CUBIN")
    problem = _problem(B, N, 384, True, lengths if isinstance(lengths, str) else "full", D=256)
    if not isinstance(lengths, str):
        problem["out_seqlen"] = _out_seqlen(B, N, lengths)
    reference = _reference(problem, True, True)
    outputs = {variant: _chain(problem, True, k3_variant=variant) for variant in ("K3_4", "K3_3")}
    errors = {}
    for variant, out in outputs.items():
        torch.testing.assert_close(out.float(), reference, atol=_ATOL, rtol=0)
        errors[variant] = ((out.float() - reference).norm() / reference.norm()).item()
    assert errors["K3_4"] <= 1.25 * errors["K3_3"], errors
    mask = (torch.arange(N, device="cuda") < problem["out_seqlen"].reshape(B, N, 1)).reshape(B, N, N, 1)
    assert bool((outputs["K3_4"][~mask.expand_as(outputs["K3_4"])] == 0).all())
