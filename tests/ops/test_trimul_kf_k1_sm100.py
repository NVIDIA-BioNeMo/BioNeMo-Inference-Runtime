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
"""SM100/SM103 TriMul KF K1: the tcgen05 kernels against the fp32 reference, and their configs and selection."""

from __future__ import annotations

import json

import pytest
import torch

from bionemo_ir._torch.custom_ops import trimul_kf_k1 as k1
from bionemo_ir._torch.custom_ops.trimul_kf_k1 import _config as k1_config
from bionemo_ir._torch.custom_ops.trimul_kf_k1 import cutedsl as k1_cutedsl
from bionemo_ir._torch.custom_ops.trimul_kf_k1 import ops as k1_ops
from tests._torch import skip_if_not_sm100_family, source_module_available

from .test_trimul_kf import _ATOL, _EPS, _k1_reference, _problem

_SOURCE = "bionemo_ir._torch.custom_ops.trimul_kf_k1._source"
_requires_source = pytest.mark.skipif(
    not source_module_available(_SOURCE), reason="the SM100 K1 kernels compile from source only"
)
_SHAPES = [(32, 32), (64, 64), (128, 128), (256, 256), (384, 256)]
_BLACKWELL = (100, 103)
_Tile = k1_config.TrimulKFK1Tile

# (B, N, C, D, has_bias, lengths). Tiny N, at B = 1 and 2, on every shape: fewer tiles than CTAs, CTA
# pairs whose second tile lies past the plane, and N * N below one 128-position tile (N = 8). Then the
# tile switch points, four epilogue groups (C = 64 from 208, C = 128 from 72), N * N % 128 != 0 with
# several batches, padded planes (N % 16 == 8 from 512) and the large-N tiles.
_TINY_CASES = [
    (B, N, C, D, (B + N // 8) % 2 == 0, "random" if (N // 8 + C) % 2 else "tail")
    for C, D in _SHAPES
    for N in (8, 16, 24, 40, 64)
    for B in (1, 2)
]
_CASES = [
    *_TINY_CASES,
    (1, 72, 64, 64, True, "random"),
    (1, 72, 128, 128, False, "tail"),
    (1, 72, 384, 256, True, "random"),
    (1, 128, 32, 32, True, "tail"),
    (2, 136, 32, 32, False, "random"),
    (3, 168, 64, 64, True, "random"),
    (2, 216, 64, 64, True, "random"),
    (2, 136, 128, 128, True, "random"),
    (2, 200, 384, 256, False, "random"),
    (1, 264, 256, 256, False, "tail"),
    (1, 320, 128, 128, True, "random"),
    (1, 328, 128, 128, True, "tail"),
    (2, 392, 256, 256, True, "random"),
    (1, 520, 128, 128, True, "tail"),
    (2, 552, 32, 32, True, "random"),
    (1, 520, 384, 256, True, "random"),
    (1, 1016, 256, 256, True, "tail"),
    (1, 1024, 64, 64, False, "random"),
    (1, 1032, 64, 64, True, "random"),
    (1, 1536, 64, 64, True, "tail"),
    (1, 2056, 32, 32, False, "random"),
]


def _case_id(case) -> str:
    B, N, C, D, has_bias, lengths = case
    return f"B{B}-N{N}-C{C}-D{D}-bias{int(has_bias)}-{lengths}"


def _fold(problem: dict) -> k1.TrimulKFInputFold:
    return k1.fold_input_weights(
        problem["norm_in_weight"],
        problem["norm_in_bias"],
        problem["p_in_weight"],
        problem["g_in_weight"],
        problem["p_in_bias"],
        problem["g_in_bias"],
    )


def _op(C: int, D: int, monkeypatch: pytest.MonkeyPatch) -> k1.TrimulKFK1Op:
    """K1 from source on this SM100/SM103 GPU."""
    skip_if_not_sm100_family()
    if not source_module_available(_SOURCE):
        pytest.skip("the SM100 K1 kernels compile from source only")
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    op = k1.get_trimul_kf_k1_op(torch.bfloat16, C, D)
    assert op is not None
    return op


# --- kernels against the fp32 reference ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_k1_matches_reference(case, monkeypatch):
    """a and b match fp32 within the chain's tolerance, are exactly zero on masked positions, and K1 writes no
    statistics for the SM100 K3."""
    B, N, C, D, has_bias, lengths = case
    op = _op(C, D, monkeypatch)
    problem = _problem(B, N, C, has_bias, lengths, seed=N + C, D=D)
    selection = op.select(N)
    a, b, stats = op(problem["x"], problem["seqlen"], _fold(problem), _EPS)
    assert stats is None
    assert a.shape == b.shape == (B, D, N, N) and a.dtype == b.dtype == torch.bfloat16
    assert a.stride() == b.stride() and a.stride(2) == selection.ab_pitch(N)
    a_ref, b_ref = _k1_reference(problem)
    torch.testing.assert_close(a.float(), a_ref, atol=_ATOL, rtol=1e-2)
    torch.testing.assert_close(b.float(), b_ref, atol=_ATOL, rtol=1e-2)
    masked = ~(torch.arange(N, device=a.device) < problem["seqlen"].reshape(B, N, 1))
    assert not a.permute(0, 2, 3, 1)[masked].any() and not b.permute(0, 2, 3, 1)[masked].any()


@pytest.mark.parametrize(
    ("B", "N", "C", "D"),
    [(1, 16, 32, 32), (2, 40, 256, 256), (1, 136, 64, 64), (2, 168, 128, 128), (1, 520, 32, 32), (1, 552, 384, 256)],
    ids=lambda v: str(v),
)
def test_graph_replay_matches_eager(B, N, C, D, monkeypatch):
    """A captured K1 replayed on new x equals an eager call on that x, bit for bit."""
    op = _op(C, D, monkeypatch)
    problem = _problem(B, N, C, True, "random", D=D)
    fold = _fold(problem)
    static_x = problem["x"].clone()
    seqlen = problem["seqlen"]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(static_x, seqlen, fold, _EPS)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        a, b, _ = op(static_x, seqlen, fold, _EPS)
    for seed in range(1, 4):
        x = _problem(B, N, C, True, "random", seed=seed, D=D)["x"]
        static_x.copy_(x)
        graph.replay()
        a_eager, b_eager, _ = op(x, seqlen, fold, _EPS)
        assert torch.equal(a, a_eager) and torch.equal(b, b_eager), seed


def test_k1_rejects_unaligned_tokens(monkeypatch):
    """N must stay a multiple of 8 on SM100 too; the node then keeps its regular path."""
    op = _op(32, 32, monkeypatch)
    problem = _problem(1, 128, 32, False, "full")
    x, seqlen = problem["x"][:, :124, :124].contiguous(), problem["seqlen"][:, :124].contiguous()
    assert not op.accepts(x, seqlen)
    with pytest.raises(ValueError, match="multiple of 8"):
        op(x, seqlen, _fold(problem), _EPS)


def test_force_cubin_ships_no_k1_on_this_gpu(monkeypatch):
    """No SM100 CUBIN ships, so CUTEDSL_FORCE_CUBIN=1 leaves every shape on the caller's path."""
    _op(32, 32, monkeypatch)
    monkeypatch.setattr(k1_ops, "_trimul_kf_k1_instances", {})
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    for C, D in _SHAPES:
        assert k1.get_trimul_kf_k1_op(torch.bfloat16, C, D) is None, (C, D)
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN")
    for C, D in _SHAPES:
        assert k1.get_trimul_kf_k1_op(torch.bfloat16, C, D) is not None, (C, D)


# --- configs and selection (no GPU work) ---------------------------------------------------------------------------


def test_sm100_and_sm103_ship_one_bundle_per_shape():
    """Every shape ships on both SMs from identical ``sm100`` bundles."""
    assert k1_config.shipped_shapes(100) == k1_config.shipped_shapes(103) == set(_SHAPES)
    assert k1_config.KERNEL_ABIS[100] == k1_config.KERNEL_ABIS[103] == "sm100"
    for C, D in _SHAPES:
        bundles = [json.loads((k1_config.CONFIGS_DIR / f"C{C}_D{D}_sm{sm}.json").read_text()) for sm in _BLACKWELL]
        assert bundles[0] == bundles[1] and bundles[0]["kernel_abi"] == "sm100", (C, D)
        assert k1_config.anchors(100, C, D) == k1_config.anchors(103, C, D)
        assert k1_config.layout_rules(100, C, D) == k1_config.layout_rules(103, C, D)
        assert k1_config.tiles(100, C, D) == k1_config.tiles(103, C, D)


@pytest.mark.parametrize("sm", _BLACKWELL)
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda shape: f"C{shape[0]}-D{shape[1]}")
def test_sm100_selection_follows_the_measured_dispatch(sm, shape):
    """K1_B1 at C = 32 and K1_B0 elsewhere, tiles switching where the interleaved measurements put them, the
    pad_n_mod_16_8 layout from 512 tokens (from C = 128's wider tile, past 320; at D = 256 past 288 where padding
    adds 8 columns), separate folds and no statistics, at every token count. Four epilogue groups run C = 64 past
    200 tokens and up to 512, and C = 128 past 64 and up to 1032; two run every other tile."""
    C, D = shape
    pad_from = 321 if C == 128 else 513
    for n in range(8, 4097, 8):
        selection = k1_config.select(sm, C, D, n)
        padded = n % 16 == 8 and (n >= pad_from or (D == 256 and n > 288 and n % 64 == 56))
        assert selection.ab_layout == ("padded" if padded else "dense"), n
        assert not selection.interleaved and not selection.writes_stats, n
        if C == 32:
            assert (selection.kernel_variant, selection.tile) == ("K1_B1", None), n
            continue
        tile = selection.tile
        assert selection.kernel_variant == "K1_B0" and tile is not None, n
        assert tile.split == (2 if n <= 64 else 1), n
        if C == 64:
            assert (tile.ctas, tile.w_slots, tile.s_stages) == (1, 0, 4), n
            expected = (64, 16, 1) if n <= 64 else (128, 16, 1) if n <= 1024 else (64, 32, 2)
            assert (tile.bnc, tile.epi_n, tile.store_warps) == expected, n
            assert tile.epi_groups == (4 if 200 < n <= 512 else 2), n
        elif C == 128:
            assert (tile.ctas, tile.w_slots, tile.s_stages, tile.epi_n, tile.store_warps) == (2, 0, 2, 16, 1), n
            assert tile.bnc == (128 if n > 320 else 64), n
            assert tile.epi_groups == (4 if 64 < n <= 1032 else 2), n
        else:
            assert tile == _Tile(128, 16, 2, 4, 2, tile.split, 1, 2), n


@_requires_source
@pytest.mark.parametrize("sm", _BLACKWELL)
def test_sm100_anchors_build_kernels_that_fit(sm):
    """Every shipped anchor's kernel, dense and padded, tiles its shape and fits in shared memory."""
    from bionemo_ir._torch.custom_ops.trimul_kf_k1 import _source

    for C, D in _SHAPES:
        rules = k1_config.layout_rules(sm, C, D)
        tiles = k1_config.tiles(sm, C, D)
        for bucket, variant in k1_config.anchors(sm, C, D).items():
            layouts = ("dense", "padded") if rules[bucket] != "dense" else ("dense",)
            for layout in layouts:
                kernel = _source.make_kernel(variant, C, D, 148, layout, tiles[bucket])
                assert kernel.padded == (layout == "padded") and kernel.x_slots >= 2, (C, D, bucket, layout)
        with pytest.raises(ValueError):
            _source.make_kernel("K1_B1", C, D, 148, "dense", _Tile(64, 16, 1, 0, 4, 1, 1, 2))


@pytest.mark.parametrize("sm", _BLACKWELL)
def test_force_cubin_never_ships_sm100(sm, monkeypatch):
    """SM100 and SM103 run from source only: forcing CUBINs, or a source-free build, ships no shape."""
    backend = k1_cutedsl.TrimulKFK1CuTe(sm)
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    for C, D in _SHAPES:
        assert not backend.ships(C, D), (C, D)
        assert not backend.supports(C, D, "K1_B0", "padded") and backend.supports(C, D, "K1_B0", "dense")
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN")
    assert all(backend.ships(C, D) for C, D in _SHAPES) == source_module_available(_SOURCE)


@pytest.mark.parametrize("sm", _BLACKWELL)
def test_get_op_follows_ships_on_blackwell(sm, monkeypatch):
    """The op exists on SM100/SM103 exactly when its kernels compile from source."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: divmod(sm, 10))
    monkeypatch.setattr(k1_ops, "_trimul_kf_k1_instances", {})
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    assert k1.get_trimul_kf_k1_op(torch.bfloat16, 128, 128) is None
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN")
    op = k1.get_trimul_kf_k1_op(torch.bfloat16, 128, 128)
    assert (op is not None) == source_module_available(_SOURCE)
    assert k1.get_trimul_kf_k1_op(torch.bfloat16, 96, 96) is None


_K1_B0 = {
    "kernel_variant": "K1_B0",
    "bnc": 64,
    "epi_n": 16,
    "ctas": 2,
    "w_slots": 0,
    "s_stages": 2,
    "split": 1,
    "epi_groups": 2,
}


@pytest.mark.parametrize(
    ("entry", "kernel_abi"),
    [
        ({**_K1_B0, "store_warps": 1}, "sm90"),  # an SM100 variant in an SM90 bundle
        ({"kernel_variant": "K1_0"}, "sm100"),  # and the other way round
        (_K1_B0, "sm100"),  # a tile field missing
        ({**_K1_B0, "store_warps": 1, "tile_n": 128}, "sm100"),  # an unknown field
        ({**_K1_B0, "store_warps": 3}, "sm100"),
        ({**_K1_B0, "store_warps": 1, "ctas": 4}, "sm100"),
        ({**_K1_B0, "store_warps": 1, "split": 0}, "sm100"),
        ({**_K1_B0, "store_warps": 1, "epi_n": 24}, "sm100"),
        ({**_K1_B0, "store_warps": True}, "sm100"),
        ({**_K1_B0, "store_warps": 1, "epi_groups": 3}, "sm100"),
        ({**_K1_B0, "store_warps": 1, "epi_n": 64, "epi_groups": 4}, "sm100"),  # one subtile cannot be shared
        ({"kernel_variant": "K1_B1", "bnc": 64}, "sm100"),  # K1_B1 takes no tile
        ({"kernel_variant": "K1_B1", "ab_layout": "pad"}, "sm100"),
        ({"kernel_variant": "K1_B1"}, "sm110"),
    ],
)
def test_malformed_sm100_configs_are_rejected(entry, kernel_abi):
    with pytest.raises(ValueError):
        k1_config.parse_configs({"S=128": entry}, "test", kernel_abi)


def test_sm100_entries_parse_into_tiles():
    configs = {
        "S=32": {**_K1_B0, "store_warps": 1},
        "S=96": {**_K1_B0, "bnc": 128, "store_warps": 2, "epi_groups": 4, "ab_layout": "pad_n_mod_16_8"},
    }
    assert k1_config.parse_configs(configs, "test", "sm100") == {32: "K1_B0", 96: "K1_B0"}
    assert k1_config.parse_layout_rules(configs, "test", "sm100") == {32: "dense", 96: "pad_n_mod_16_8"}
    assert k1_config.parse_tiles(configs, "test", "sm100") == {
        32: _Tile(64, 16, 2, 0, 2, 1, 1, 2),
        96: _Tile(128, 16, 2, 0, 2, 1, 2, 4),
    }
    assert k1_config.parse_tiles({"S=128": {"kernel_variant": "K1_B1"}}, "test", "sm100") == {128: None}
