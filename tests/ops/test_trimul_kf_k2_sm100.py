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
"""SM100/SM103 TriMul KF K2 from source: shipped tiles against the fp32 reference, graph replay, and selection.

No SM100 CUBIN ships, so under ``CUTEDSL_FORCE_CUBIN=1`` or without kernel sources these GPUs get no K2.
"""

from __future__ import annotations

import json

import pytest
import torch

from bionemo_ir._torch.custom_ops import trimul_kf_k2 as k2
from bionemo_ir._torch.custom_ops.trimul_kf_k2 import _config as k2_config
from bionemo_ir._torch.custom_ops.trimul_kf_k2 import cutedsl as k2_cutedsl
from tests._torch import skip_if_not_sm100_family, source_module_available

from .test_trimul_kf import _ATOL, _k1_reference, _k2_reference, _problem

_T = k2_config.TrimulKFK2Sm100Tile
_SM100_FAMILY = (100, 103)
_WIDTHS = (32, 64, 128, 256)
_requires_source = pytest.mark.skipif(
    not source_module_available("bionemo_ir._torch.custom_ops.trimul_kf_k2._source"),
    reason="the SM100 K2 compiles only from source",
)
#: K1 pads plane rows to whole 128-byte lines.
_PAD = 64


def _k1_planes(B: int, N: int, C: int, D: int, lengths: str, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """K1's fp32 reference ``a``, ``b`` of a ``(C, D)`` problem as dense bf16 ``[B, D, N, N]`` planes."""
    a, b = _k1_reference(_problem(B, N, C, True, lengths, seed=seed, D=D))
    return a.to(torch.bfloat16).contiguous(), b.to(torch.bfloat16).contiguous()


def _masked_planes(B: int, N: int, D: int, lengths: str, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Random bf16 ``a``, ``b`` ``[B, D, N, N]`` with K1's mask ``j < seqlen[b, i]``, without K1's reference cost."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    seqlen = torch.full((B, N), N, device="cuda")
    if lengths == "tail":
        seqlen[:, N - 5 :] = 0
        seqlen[:, : N - 5] = N - 5
    elif lengths == "random":
        seqlen = torch.randint(0, N + 1, (B, N), device="cuda", generator=generator)
    mask = (torch.arange(N, device="cuda") < seqlen[..., None]).unsqueeze(1)
    a, b = ((torch.randn(B, D, N, N, device="cuda", generator=generator) * 0.5 * mask).to(torch.bfloat16) for _ in "ab")
    return a, b


def _padded(t: torch.Tensor, pitch: int) -> torch.Tensor:
    """``t`` ``[B, D, N, N]`` as a view of ``pitch x pitch`` planes, the way K1 writes padded planes."""
    B, D, N, _ = t.shape
    planes = torch.zeros(B, D, pitch, pitch, dtype=t.dtype, device=t.device)
    view = planes[..., :N, :N]
    view.copy_(t)
    return view


def _layouts(a: torch.Tensor, b: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Dense planes and planes padded to whole lines."""
    pitch = -(-a.shape[-1] // _PAD) * _PAD
    return [(a, b), (_padded(a, pitch), _padded(b, pitch))]


def _op(D: int, outgoing: bool) -> k2.TrimulKFK2Op:
    op = k2.get_trimul_kf_k2_op(torch.bfloat16, D, outgoing)
    assert op is not None
    return op


def _assert_matches(prod: torch.Tensor, reference: torch.Tensor) -> None:
    assert prod.shape == reference.shape and prod.dtype == torch.bfloat16 and prod.is_contiguous()
    torch.testing.assert_close(prod.float(), reference, atol=_ATOL, rtol=1e-2)


# --- selection and configs (no GPU work) ---------------------------------------------------------------------------


def _shipped(sm: int, D: int, outgoing: bool) -> set:
    return {k2_config.select(sm, D, n, outgoing)[1] for n in range(8, 4097, 8)}


def test_sm100_and_sm103_ship_identical_bundles_for_every_width():
    assert k2_config.KERNEL_ABIS[100] == k2_config.KERNEL_ABIS[103] == "sm100"
    assert k2_config.shipped_widths(100) == k2_config.shipped_widths(103) == frozenset(_WIDTHS)
    for D in _WIDTHS:
        bundles = [json.loads((k2_config.CONFIGS_DIR / f"D{D}_sm{sm}.json").read_text()) for sm in _SM100_FAMILY]
        assert bundles[0] == bundles[1] and bundles[0]["kernel_abi"] == "sm100"
        for outgoing in (True, False):
            assert k2_config.anchors(100, D, outgoing) == k2_config.anchors(103, D, outgoing)
            assert all(isinstance(tile, _T) for tile in _shipped(100, D, outgoing))
    assert k2_config.shipped_widths(90) == frozenset(_WIDTHS)
    assert all(isinstance(tile, k2_config.TrimulKFK2Tile) for tile in _shipped(90, 128, True))


@pytest.mark.parametrize("D", _WIDTHS)
def test_sm100_dispatch_runs_k2_b0_below_512_then_pair_tiles_and_incoming_tails(D):
    """K2_B0 below N = 512 and, from D = 64, up to N = 544; K2_B1 elsewhere, on columns 192 wide up to N = 576 (incoming
    below D = 128: 544) and outgoing 224 wide up to N = 672; K2_B2's tails only incoming, past N = 2592, just past
    256 k."""
    for outgoing in (True, False):
        for n in range(8, 4097, 8):
            _, tile = k2_config.select(100, D, n, outgoing)
            if n < 512 or (D >= 64 and 520 <= n <= 544):
                assert tile.kernel_variant == "K2_B0", (n, tile)
            elif not outgoing and n >= 2600 and 0 < n % 256 <= 32:
                assert tile == _T("K2_B2", tail_w=32), (n, tile)
            else:
                assert tile.kernel_variant == "K2_B1" and tile.tile_m == 256, (n, tile)
                assert tile.clc == (n >= (1032 if D == 32 else 1024)), (n, tile)
                if 512 < n <= 672:
                    # Three narrower column tiles instead of three 256 wide: 576 or 672 columns, not 768. At D = 256,
                    # N = 640 (whole 128-byte rows) the 224-wide tile did not beat the 256-wide one.
                    narrow = n <= (576 if outgoing or D >= 128 else 544)
                    wide = not outgoing or (D == 256 and n == 640)
                    assert tile.tile_n == (192 if narrow else 256 if wide else 224), (n, tile)
                else:
                    assert tile.tile_n == 256 or (outgoing and tile.padded(n) < -(-n // 256) * 256), (n, tile)
        # Each direction compiles a handful of images, fallbacks included (23 across both, shared by every width).
        assert len(k2_config.shipped_tiles(list(k2_config.anchors(100, D, outgoing).values()))) <= 12


@pytest.mark.parametrize("D", _WIDTHS)
def test_sm100_paired_ctas_run_alone_on_an_odd_number_of_column_tiles(D):
    """Anchors sit at multiples of 8, so the anchor of a K2_B0 tile on paired CTAs also takes token counts with an
    odd number of its column tiles (N = 385 at the 384 anchor). The pair would compute a whole column tile past N
    there, so the tile runs unpaired instead, and the bundle ships that image."""
    for outgoing in (True, False):
        shipped = set(k2_config.shipped_tiles(list(k2_config.anchors(100, D, outgoing).values())))
        for n in range(1, 4097):
            _, tile = k2_config.select(100, D, n, outgoing)
            assert tile in shipped, (n, tile)
            if tile.kernel_variant == "K2_B0" and tile.cluster_n > 1:
                assert -(-n // tile.tile_n) % tile.cluster_n == 0, (n, tile)
    if D >= 64:
        assert k2_config.select(100, D, 384, True)[1] == _T("K2_B0", 128, 192, cluster_n=2)
        assert k2_config.select(100, D, 385, True)[1] == _T("K2_B0", 128, 192)
        assert k2_config.select(100, D, 321, False)[1] == _T("K2_B0", 128, 160)
    assert k2_config.fallback(_T("K2_B1", tile_n=224, cluster_n=2, clc=True)) is None


def test_sm100_tile_lists_take_the_tile_that_computes_fewest_columns():
    pair = _T("K2_B1", tile_n=256, cluster_n=1, clc=True)
    quad = _T("K2_B1", tile_n=224, cluster_n=2, clc=True)
    tails = _T("K2_B2", tail_w=32)
    # 224 x 2 pads N to 448 columns: it wins only where that beats 256 (N = 3096: 3136 < 3328).
    assert [k2_config.least_padding((pair, quad), n) for n in (1024, 1104, 1800, 2104, 3096)] == [
        pair,
        pair,
        pair,
        quad,
        quad,
    ]
    # Tail tiles 32 wide cover N = 256 k + r, 0 < r <= 32; otherwise the plain tile wins the tie.
    assert [k2_config.least_padding((pair, tails), n) for n in (2824, 3072, 3096, 3112, 3200)] == [
        tails,
        pair,
        tails,
        pair,
        pair,
    ]
    assert tails.padded(3096) == 3104 and tails.padded(3072) == 3072 and tails.padded(200) == 256
    assert k2_config.fallback(quad) is None and k2_config.runtime_tile(quad, 1104) == quad
    assert set(k2_config.shipped_tiles([(pair, quad), tails])) == {pair, quad, tails}


@pytest.mark.parametrize(
    "entry",
    [
        {"kernel_variant": "K2_B0", "tile_m": 64, "tile_n": 64},
        {"kernel_variant": "K2_B0", "tile_m": 128, "tile_n": 48},
        {"kernel_variant": "K2_B0", "tile_m": 128},
        {"kernel_variant": "K2_B0", "tile_m": 256, "tile_n": 256, "cluster_n": 2},
        {"kernel_variant": "K2_B0", "tile_m": 128, "tile_n": 192, "cluster_n": 4},
        {"kernel_variant": "K2_B1", "tile_n": 256, "cluster_n": 4, "clc": True},
        {"kernel_variant": "K2_B1", "tile_n": 256, "cluster_n": 1, "clc": 1},
        {"kernel_variant": "K2_B1", "tile_n": 256, "cluster_n": 1, "clc": True, "tile_m": 256},
        {"kernel_variant": "K2_B2", "tail_w": 16},
        {"kernel_variant": "K2_0", "tile_n": 256, "cluster_m": 2, "defer_kmin": 0, "split_epi": False},
        {"kernel_variant": "K2_B9"},
    ],
)
def test_malformed_sm100_tiles_are_rejected(entry):
    with pytest.raises(ValueError):
        k2_config.parse_tile(entry, "test", "sm100")


def test_sm100_anchors_keep_tails_incoming_and_lists_distinct():
    tails = {"kernel_variant": "K2_B2", "tail_w": 32}
    pair = {"kernel_variant": "K2_B1", "tile_n": 256, "cluster_n": 1, "clc": True}
    out, inc = k2_config.parse_entry({"out": pair, "in": [pair, tails]}, "ok", "sm100")
    assert out == (_T("K2_B1", tile_n=256, cluster_n=1, clc=True),) and len(inc) == 2
    for bad in (tails, [pair, tails], {"out": [pair, tails], "in": pair}, [pair, {**pair, "clc": False}]):
        with pytest.raises(ValueError):
            k2_config.parse_entry(bad, "bad", "sm100")
    # SM90 bundles keep their own schema.
    with pytest.raises(ValueError):
        k2_config.parse_tile(pair, "test", "sm90")


@pytest.mark.parametrize("sm", _SM100_FAMILY)
def test_force_cubin_or_a_source_free_build_ships_no_sm100_k2(sm, monkeypatch):
    """No SM100 CUBIN ships: ``CUTEDSL_FORCE_CUBIN=1`` or missing sources leave the layer on its regular path."""
    monkeypatch.setattr(k2_cutedsl.TrimulKFK2CuTe, "_compiled_cache", {})
    backend = k2_cutedsl.TrimulKFK2CuTe(sm)
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    assert not any(backend.ships(D, outgoing) for D in _WIDTHS for outgoing in (True, False))
    with pytest.raises(RuntimeError, match="No trimul KF K2 CUBIN ships"):
        backend._get_or_compile(True, _T("K2_B0", 128, 64))
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN")

    def missing(package: str):
        raise ImportError(package)

    monkeypatch.setattr(k2_cutedsl, "load_source_module", missing)
    assert not backend.ships(128, True)
    assert not k2_cutedsl.TrimulKFK2CuTe(sm).ships(96, True)


@_requires_source
def test_source_builds_ship_sm100_k2_and_name_the_sm100_kernels(monkeypatch):
    from bionemo_ir._torch.custom_ops.trimul_kf_k2 import _source

    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    assert all(k2_cutedsl.TrimulKFK2CuTe(100).ships(D, o) for D in _WIDTHS for o in (True, False))
    assert _source.source_implementation("sm100", None, "K2_B2").endswith("sm100_trimul_kf_k2.Sm100TrimulKF_K2_2")
    with pytest.raises(ValueError):
        _source.source_implementation("sm100", None, "K2_0")


# --- on the GPU ----------------------------------------------------------------------------------------------------


def test_force_cubin_turns_the_op_off_on_this_gpu(monkeypatch):
    skip_if_not_sm100_family()
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    assert all(k2.get_trimul_kf_k2_op(torch.bfloat16, D, o) is None for D in _WIDTHS for o in (True, False))


@pytest.mark.parametrize("outgoing", [True, False], ids=["out", "in"])
@pytest.mark.parametrize("D", [32, 256])
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("N", [8, 16, 24, 40, 64])
@_requires_source
def test_tiny_planes_match_reference(N, B, D, outgoing, monkeypatch):
    """Planes smaller than one tile (TMA boxes past the tensor, fewer tiles than SMs) are exact; graphed trunks run
    the KF chain at N = 16."""
    skip_if_not_sm100_family()
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    a, b = _k1_planes(B, N, D, D, "full" if B == 1 else "random", seed=N)
    reference = _k2_reference(a, b, outgoing)
    op = _op(D, outgoing)
    for aa, bb in _layouts(a, b):
        _assert_matches(op(aa, bb), reference)


# (B, N, C, D, lengths): every K2_B0 tile in its N range (1-SM CTAs alone and in pairs along N), K2_B1 with a static
# split (256, 224 and 192 wide) and with CLC (256 and 224 x 2 tiles), K2_B2's tail tiles; masked rows, B > 1,
# N % 16 == 8 (dense rows below N = 512) and every shipped (C, D).
_CASES = [
    (1, 128, 32, 32, "tail"),
    (2, 136, 128, 128, "random"),
    (3, 168, 64, 64, "random"),
    (1, 264, 256, 256, "tail"),
    (1, 320, 32, 32, "random"),
    (2, 328, 128, 128, "random"),
    (1, 392, 384, 256, "tail"),
    (1, 504, 64, 64, "random"),
    (1, 520, 128, 128, "tail"),
    (1, 568, 128, 128, "random"),
    (1, 568, 384, 256, "tail"),
    (1, 600, 256, 256, "random"),
    (1, 616, 32, 32, "tail"),
    (2, 776, 32, 32, "random"),
    (1, 1016, 384, 256, "tail"),
    (1, 1032, 64, 64, "random"),
    (1, 1320, 32, 32, "random"),
]
# Large N, on synthetic masked planes: incoming tails at r = 8 and 24, outgoing 256 and 224 x 2.
_LARGE_CASES = [(1, 2824, 32, "tail"), (1, 3096, 128, "random")]


def _case_id(case) -> str:
    return "-".join(str(v) for v in case)


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
@_requires_source
def test_shipped_tiles_match_reference(case, monkeypatch):
    skip_if_not_sm100_family()
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    B, N, C, D, lengths = case
    a, b = _k1_planes(B, N, C, D, lengths, seed=N)
    for outgoing in (True, False):
        reference = _k2_reference(a, b, outgoing)
        op = _op(D, outgoing)
        for aa, bb in _layouts(a, b):
            _assert_matches(op(aa, bb), reference)


@pytest.mark.parametrize("case", _LARGE_CASES, ids=_case_id)
@_requires_source
def test_large_n_tiles_match_reference(case, monkeypatch):
    skip_if_not_sm100_family()
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    B, N, D, lengths = case
    a, b = _masked_planes(B, N, D, lengths, seed=N)
    pair, quad = _T("K2_B1", clc=True), _T("K2_B1", tile_n=224, cluster_n=2, clc=True)
    for outgoing in (True, False):
        op = _op(D, outgoing)
        assert op.select(N) == (k2_config.least_padding((pair, quad), N) if outgoing else _T("K2_B2", tail_w=32))
        reference = _k2_reference(a, b, outgoing)
        for aa, bb in _layouts(a, b):
            _assert_matches(op(aa, bb), reference)
        del reference
        torch.cuda.empty_cache()


@pytest.mark.parametrize("outgoing", [True, False], ids=["out", "in"])
@pytest.mark.parametrize("N", [8, 40, 136, 264, 520])
@_requires_source
def test_every_shipped_tile_handles_any_plane_size(N, outgoing, monkeypatch):
    """Every tile a bundle ships, run at sizes its anchor never sends it: planes below one tile, partial K blocks,
    fewer tiles than SMs or clusters, and K2_B2 with and without room for tails (N = 264: 256 + 8)."""
    skip_if_not_sm100_family()
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    D = 32
    a, b = _k1_planes(2, N, D, D, "random", seed=N)
    reference = _k2_reference(a, b, outgoing)
    op = _op(D, outgoing)
    entries = [entry for width in _WIDTHS for entry in k2_config.anchors(100, width, outgoing).values()]
    for tile in k2_config.shipped_tiles(entries):
        for aa, bb in _layouts(a, b):
            prod = op.backend.run(aa, bb, outgoing, tile)
            torch.testing.assert_close(
                prod.float(), reference, atol=_ATOL, rtol=1e-2, msg=lambda m, t=tile: f"{t}: {m}"
            )


@pytest.mark.parametrize("outgoing", [True, False], ids=["out", "in"])
@_requires_source
def test_graph_replay_matches_eager(outgoing, monkeypatch):
    """Captured calls of every kernel kind replay the eager products bit for bit on fresh inputs."""
    skip_if_not_sm100_family()
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    shapes = [(2, 16, 32), (1, 200, 64), (1, 520, 128), (1, 1032, 32), (1, 2824, 32)]
    statics, inputs = [], []
    for B, N, D in shapes:
        a, b = _masked_planes(B, N, D, "random", seed=1)
        pitch = -(-N // _PAD) * _PAD if N >= 512 else N
        statics.append((_op(D, outgoing), _padded(a, pitch), _padded(b, pitch)))
        inputs.append([_masked_planes(B, N, D, "random", seed=seed) for seed in (2, 3)])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for op, a, b in statics:
            op(a, b)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = [op(a, b) for op, a, b in statics]
    for round_ in range(2):
        for (_, a, b), fresh in zip(statics, inputs, strict=True):
            a.copy_(fresh[round_][0])
            b.copy_(fresh[round_][1])
        graph.replay()
        torch.cuda.synchronize()
        for (op, a, b), out in zip(statics, outs, strict=True):
            assert torch.equal(out, op(a.contiguous(), b.contiguous()))


def _prod_rows(a: torch.Tensor, b: torch.Tensor, outgoing: bool, rows: list[int]) -> torch.Tensor:
    """fp32 product rows ``[D, len(rows), N]`` of the last batch, 32 planes at a time."""
    B, D, N, _ = a.shape
    out = torch.empty((D, len(rows), N), device=a.device)
    for d in range(0, D, 32):
        planes = b[B - 1, d : d + 32].float()
        if outgoing:
            out[d : d + 32] = torch.einsum("drk,djk->drj", a[B - 1, d : d + 32, rows].float(), planes)
        else:
            out[d : d + 32] = torch.einsum("dkr,dkj->drj", a[B - 1, d : d + 32, :, rows].float(), planes)
    return out


# One launch past 2^31 elements: B = 2 puts the batch stride of N = 2904 at 128 channels past it (K2_B1); N = 3096 at
# 256 channels passes it in one batch, incoming on K2_B2's tails.
@pytest.mark.parametrize(("B", "N", "D", "outgoing"), [(2, 2904, 128, True), (1, 3096, 256, False)], ids=str)
@_requires_source
def test_past_int32_offsets_matches_reference_rows(B, N, D, outgoing, monkeypatch):
    skip_if_not_sm100_family()
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    torch.cuda.empty_cache()
    pitch = -(-N // _PAD) * _PAD
    if torch.cuda.mem_get_info()[0] < 2 * B * D * (2 * pitch * pitch + N * N) * 5 // 4:
        pytest.skip("needs more free GPU memory")
    generator = torch.Generator(device="cuda").manual_seed(0)
    a, b = (
        torch.empty(B, D, pitch, pitch, device="cuda", dtype=torch.bfloat16)[..., :N, :N].normal_(generator=generator)
        for _ in "ab"
    )
    prod = _op(D, outgoing)(a, b)
    assert prod.numel() > 2**31 and a.untyped_storage().nbytes() // 2 > 2**31
    rows = [0, N // 2, N - 1]
    torch.testing.assert_close(prod[B - 1, :, rows].float(), _prod_rows(a, b, outgoing, rows), atol=_ATOL, rtol=1e-2)


@pytest.mark.parametrize("outgoing", [True, False], ids=["out", "in"])
@_requires_source
def test_k2_reads_dense_and_padded_planes_and_refuses_others(outgoing, monkeypatch):
    """The pitch comes off the strides, as on SM90; a/b with different strides or unaligned rows are refused."""
    skip_if_not_sm100_family()
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    D, N, P = 64, 136, 192
    op = _op(D, outgoing)
    planes = torch.randn(2, 2, D, P, P, device="cuda").to(torch.bfloat16)
    a, b = planes[..., :N, :N].unbind(0)
    assert op.accepts(a, b)
    prod = op(a, b)
    assert torch.equal(prod, op(a.contiguous(), b.contiguous()))
    _assert_matches(prod, _k2_reference(a, b, outgoing))
    assert not op.accepts(a, b.contiguous())
    odd = torch.empty(2, D, N, N + 4, device="cuda", dtype=torch.bfloat16)[..., :N]
    assert not op.accepts(odd, odd)
    assert not op.accepts(a[:, :, :124, :124], b[:, :, :124, :124])
