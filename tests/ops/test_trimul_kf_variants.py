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

import pytest
import torch

from bionemo_ir._torch.custom_ops import trimul_kf_k1 as k1
from bionemo_ir._torch.custom_ops import trimul_kf_k2 as k2
from bionemo_ir._torch.custom_ops import trimul_kf_k3 as k3
from bionemo_ir._torch.custom_ops.trimul_kf_k1 import _config as k1_config
from bionemo_ir._torch.custom_ops.trimul_kf_k2 import _config as k2_config
from bionemo_ir._torch.custom_ops.trimul_kf_k3 import _config as k3_config
from tests._torch import skip_if_not_sm90, source_module_available

from .test_trimul_kf import _ATOL, _EPS, _k2_reference, _problem, _reference

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
    C = x.shape[-1]
    op1 = k1.get_trimul_kf_k1_op(torch.bfloat16, C, C)
    fold_in, _ = _folds(problem)
    if layout is None:
        return op1(x, problem["seqlen"], fold_in, _EPS)
    selection = op1.select(x.shape[1])._replace(ab_layout=layout)
    w_in, w_gate_in = (fold_in.interleaved, None) if selection.interleaved else (fold_in.proj, fold_in.gate)
    return op1.backend.run(x, problem["seqlen"], w_in, w_gate_in, fold_in.vec, _EPS, selection)


def _chain(problem: dict, outgoing: bool, layout: str | None = None, tile=None, k3_variant: str | None = None):
    """The chain as the node runs it, with K1's layout, K2's tile or K3's variant optionally forced."""
    x = problem["x"]
    C, N = x.shape[-1], x.shape[1]
    a, b, stats = _k1(problem, layout)
    op2 = k2.get_trimul_kf_k2_op(torch.bfloat16, C, outgoing)
    prod = op2(a, b) if tile is None else op2.backend.run(a, b, outgoing, tile)
    op3 = k3.get_trimul_kf_k3_op(torch.bfloat16, C, C, True)
    _, fold_out = _folds(problem)
    if k3_variant is None:
        return op3(prod, x, fold_out, stats, _EPS, residual=True, actual_seqlen=problem["out_seqlen"])
    selection = op3.select(N)._replace(kernel_variant=k3_variant)
    out = torch.empty_like(x)
    return op3.backend.run(
        prod, x, fold_out.w_out, fold_out.w_gate, fold_out.vec, stats, problem["out_seqlen"], _EPS, selection, out
    )


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
    """dense never pads; pad_n_mod_16_8 pads N % 16 == 8 from AB_PAD_MIN_N tokens up, at any size."""
    assert k1_config.AB_PAD_MIN_N % k1_config.AB_PAD_ALIGN == 0
    assert [n for n in range(8, 1281, 8) if k1_config.ab_layout("pad_n_mod_16_8", n) == "padded"] == [
        n for n in range(k1_config.AB_PAD_MIN_N, 1281, 8) if n % 16 == 8
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


@pytest.mark.parametrize("D", sorted(k2_config.shipped_widths(90)))
def test_k2_tables_run_at_most_four_tiles_per_direction(D):
    """Every shipped bundle runs at most four distinct tiles per direction over all token counts (CUBIN budget)."""
    for outgoing in (True, False):
        assert len({k2_config.select(90, D, n, outgoing)[1] for n in range(8, 4097, 8)}) <= 4


def test_launches_keep_tiles_in_one_batch_and_coordinates_in_int32():
    """Batches share a launch unless a 128-row tile would straddle two, or the rows would pass int32 coordinates."""
    from bionemo_ir._torch.custom_ops.trimul_kf_k1.cutedsl import MAX_LAUNCH_ROWS, batch_chunks

    assert batch_chunks(4, 3008) == ((0, 4),)  # flat extents past 2^31 stay in one launch
    assert batch_chunks(2, 136) == ((0, 1), (1, 2))  # 136 * 136 rows end mid-tile
    assert batch_chunks(3, 32768) == ((0, 1), (1, 2), (2, 3))  # 2^30 rows a batch
    assert 46336 * 46336 <= MAX_LAUNCH_ROWS
    for n, pitch in ((46344, None), (46336, 46400)):
        with pytest.raises(ValueError):
            batch_chunks(1, n, pitch)


def test_padded_pitch_rounds_rows_to_128_bytes():
    assert [k1.ab_pitch(n, "padded") for n in (128, 136, 520, 824, 832, 1032)] == [128, 192, 576, 832, 832, 1088]
    assert all(k1.ab_pitch(n) == n for n in (8, 136, 824))
    with pytest.raises(ValueError):
        k1.ab_pitch(128, "strided")


# --- on the GPU ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("B", "N", "C", "outgoing", "lengths"),
    [
        (1, 8, 64, True, "full"),
        (2, 16, 128, False, "random"),
        (3, 24, 256, True, "tail"),
        (1, 40, 32, False, "random"),
        (2, 264, 128, True, "random"),
        (2, 264, 256, False, "tail"),
        (1, 520, 256, True, "random"),
        (1, 776, 128, False, "random"),
    ],
    ids=lambda v: str(v),
)
def test_chain_matches_reference_at_small_n_and_batches(B, N, C, outgoing, lengths):
    """Every op on its own selection, small N and B > 1 included."""
    skip_if_not_sm90()
    problem = _problem(B, N, C, True, lengths)
    out = _chain(problem, outgoing)
    torch.testing.assert_close(out.float(), _reference(problem, outgoing, True), atol=_ATOL, rtol=0)


@pytest.mark.parametrize(
    ("B", "N", "C", "outgoing"),
    [(1, 264, 256, True), (2, 136, 128, False), (1, 520, 128, True), (2, 200, 64, False), (1, 824, 256, False)],
    ids=lambda v: str(v),
)
@_requires_source
def test_padded_layout_is_bitwise_the_dense_one(B, N, C, outgoing):
    """K1's padded planes hold the dense values (a, b, stats) and the chain output does not change, bit for bit."""
    skip_if_not_sm90()
    problem = _problem(B, N, C, True, "random")
    dense, padded = _k1(problem, "dense"), _k1(problem, "padded")
    P = k1.ab_pitch(N, "padded")
    assert dense.a.is_contiguous() and padded.a.stride() == (C * P * P, P * P, P, 1)
    assert padded.b.stride() == padded.a.stride()
    assert torch.equal(dense.a, padded.a) and torch.equal(dense.b, padded.b)
    assert (dense.stats is None) == (padded.stats is None)
    if dense.stats is not None:  # rows past N * N pad the buffer to whole 128-row tiles; neither layout writes them
        assert torch.equal(dense.stats[:, : N * N], padded.stats[:, : N * N])
    out_dense, out_padded = _chain(problem, outgoing, "dense"), _chain(problem, outgoing, "padded")
    assert torch.equal(out_dense, out_padded)
    torch.testing.assert_close(out_padded.float(), _reference(problem, outgoing, True), atol=_ATOL, rtol=0)


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


@pytest.mark.parametrize(("N", "C"), [(264, 256), (520, 128)])
@_requires_source
def test_k3_variants_stay_in_the_error_class(N, C):
    """Every K3 variant pairing with K1's selection matches the reference as well as the shipped one."""
    skip_if_not_sm90()
    problem = _problem(1, N, C, True, "random")
    reference = _reference(problem, True, True)
    writes_stats = k1_config.select(90, C, C, N).writes_stats
    errors = {}
    for variant in k3_config.KERNEL_VARIANTS:
        if (variant in k3_config.STATS_VARIANTS) != writes_stats:
            continue
        out = _chain(problem, True, k3_variant=variant).float()
        torch.testing.assert_close(out, reference, atol=_ATOL, rtol=0)
        errors[variant] = ((out - reference).norm() / reference.norm()).item()
    shipped = k3_config.select(90, C, C, N).kernel_variant
    assert len(errors) >= 2 and all(error <= 1.25 * errors[shipped] for error in errors.values()), errors
