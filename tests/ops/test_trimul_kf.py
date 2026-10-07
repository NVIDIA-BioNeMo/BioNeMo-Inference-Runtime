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
"""SM90 TriMul KF custom ops: the K1 -> K2 -> K3 chain against the fp32 reference, from source and CUBINs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bionemo_ir import dsl_kernels
from bionemo_ir._torch.custom_ops import trimul_kf_k1 as k1
from bionemo_ir._torch.custom_ops import trimul_kf_k2 as k2
from bionemo_ir._torch.custom_ops import trimul_kf_k3 as k3
from bionemo_ir._torch.custom_ops.trimul_kf_k1 import _config as k1_config
from bionemo_ir._torch.custom_ops.trimul_kf_k1 import cutedsl as k1_cutedsl
from bionemo_ir._torch.custom_ops.trimul_kf_k2 import _config as k2_config
from bionemo_ir._torch.custom_ops.trimul_kf_k2 import cutedsl as k2_cutedsl
from bionemo_ir._torch.custom_ops.trimul_kf_k3 import _config as k3_config
from bionemo_ir._torch.custom_ops.trimul_kf_k3 import cutedsl as k3_cutedsl
from bionemo_ir._torch.utils.kernel import CuTeDSLKernelLibraryExecutable
from tests._torch import cutedsl_test_modes, require_cubin_library, skip_if_not_sm90

_CUTEDSL_MODES = cutedsl_test_modes("bionemo_ir._torch.custom_ops.trimul_kf_k1._source")
_BACKENDS = (
    (k1_cutedsl.TrimulKFK1CuTe, k1_cutedsl),
    (k2_cutedsl.TrimulKFK2CuTe, k2_cutedsl),
    (k3_cutedsl.TrimulKFK3CuTe, k3_cutedsl),
)
_MODE_CACHES: dict[tuple[type, str], dict] = {}
_EPS = 1e-5
# bf16 a, b and product between fp32 stages.
_ATOL = 8e-2

# (B, N, C, D, outgoing, has_bias, residual, lengths), selecting every K1 variant and its padded a/b
# layout (N = 520, 552, 1016), every K2 tile the D = 128 and 256 bundles run, K2_1's odd-cluster fallback
# (N = 392 at C = 64), and the K3 variants the configs select, with and without the residual.
# N * N % 128 != 0 at N = 136, 168, 200 and 376, where batches run one launch each.
_CASES = [
    (1, 128, 64, 64, True, False, False, "full"),
    (1, 128, 32, 32, False, True, True, "tail"),
    (1, 256, 32, 32, True, True, True, "random"),
    (1, 256, 128, 128, False, False, True, "random"),
    (1, 384, 128, 128, False, True, True, "tail"),
    (1, 392, 128, 128, True, False, True, "random"),
    (1, 512, 64, 64, True, True, True, "random"),
    (2, 136, 128, 128, False, True, True, "random"),
    (3, 168, 64, 64, True, True, False, "random"),
    (1, 128, 256, 256, False, True, True, "full"),
    (1, 256, 256, 256, True, True, True, "random"),
    (1, 384, 256, 256, False, False, True, "tail"),
    (2, 392, 256, 256, True, True, True, "random"),
    (1, 1024, 256, 256, False, True, True, "tail"),
    (1, 264, 256, 256, True, False, True, "tail"),
    (2, 552, 128, 128, False, True, True, "random"),
    (1, 392, 64, 64, False, False, True, "tail"),
    (1, 128, 384, 256, True, False, True, "full"),
    (1, 136, 384, 256, True, True, True, "tail"),
    (1, 168, 384, 256, False, True, False, "random"),
    (2, 200, 384, 256, False, False, True, "random"),
    (1, 256, 384, 256, True, True, False, "random"),
    (1, 264, 384, 256, True, False, True, "tail"),
    (2, 376, 384, 256, False, True, True, "random"),
    (1, 400, 384, 256, False, True, True, "tail"),
    (1, 520, 384, 256, False, True, True, "random"),
    (1, 1016, 384, 256, True, True, True, "tail"),
    (1, 264, 128, 128, True, True, True, "random"),
    (1, 520, 128, 128, True, False, True, "tail"),
]


def _case_id(case) -> str:
    B, N, C, D, outgoing, has_bias, residual, lengths = case
    widths = f"C{C}" if C == D else f"C{C}-D{D}"
    return f"B{B}-N{N}-{widths}-{'out' if outgoing else 'in'}-bias{int(has_bias)}-res{int(residual)}-{lengths}"


def _reject_cubin(*_args, **_kwargs):
    raise AssertionError("source test mode unexpectedly fell back to the CUBIN library")


def _use_mode(mode: str, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Route all three ops through exactly one of the source and CUBIN paths."""
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    caches = []
    for backend, module in _BACKENDS:
        cache = _MODE_CACHES.setdefault((backend, mode), {})
        monkeypatch.setattr(backend, "_compiled_cache", cache)
        caches.append(cache)
        if mode == "source":
            monkeypatch.setattr(module, "populate_compiled_cache_from_library", _reject_cubin)
    if mode == "cubin":
        require_cubin_library()
        monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    return caches


def _assert_dispatched(caches: list[dict], mode: str) -> None:
    for cache in caches:
        assert cache
        assert all(isinstance(value, CuTeDSLKernelLibraryExecutable) == (mode == "cubin") for value in cache.values())


def _problem(
    B: int, N: int, C: int, has_bias: bool, lengths: str, seed: int = 0, D: int | None = None
) -> dict[str, torch.Tensor | None]:
    """A chain problem of width ``C`` and hidden width ``D`` (``C`` unless given)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, device="cuda", generator=generator) * scale).to(torch.bfloat16)

    D = C if D is None else D
    seqlen = torch.full((B, N), N, device="cuda", dtype=torch.int32)
    out_seqlen = seqlen.clone()
    if lengths == "tail":
        seqlen[:, : N - 5] = N - 5
        seqlen[:, N - 5 :] = 0
        out_seqlen = seqlen.clone()
    elif lengths == "random":
        seqlen = torch.randint(0, N + 1, (B, N), device="cuda", dtype=torch.int32, generator=generator)
        out_seqlen = torch.randint(0, N + 1, (B, N), device="cuda", dtype=torch.int32, generator=generator)
    problem = {
        "x": rnd(B, N, N, C),
        "seqlen": seqlen,
        "out_seqlen": out_seqlen,
        "norm_in_weight": rnd(C, scale=0.1) + 1,
        "norm_in_bias": rnd(C, scale=0.1),
        "norm_out_weight": rnd(D, scale=0.1) + 1,
        "norm_out_bias": rnd(D, scale=0.1),
        "p_in_weight": rnd(2 * D, C, scale=C**-0.5),
        "g_in_weight": rnd(2 * D, C, scale=C**-0.5),
        "p_out_weight": rnd(C, D, scale=D**-0.5),
        "g_out_weight": rnd(C, C, scale=C**-0.5),
    }
    biases = ("p_in_bias", "g_in_bias", "p_out_bias", "g_out_bias")
    sizes = (2 * D, 2 * D, C, C)
    for name, size in zip(biases, sizes, strict=True):
        problem[name] = rnd(size, scale=0.1) if has_bias else None
    return problem


def _widths(problem: dict) -> tuple[int, int]:
    """The problem's ``(C, D)``."""
    return problem["x"].shape[-1], problem["p_in_weight"].shape[0] // 2


def _run_chain(problem: dict, outgoing: bool, residual: bool) -> torch.Tensor:
    return _run_chain_stages(problem, outgoing, residual)[-1]


def _run_chain_stages(problem: dict, outgoing: bool, residual: bool) -> tuple[torch.Tensor | None, ...]:
    """``a``, ``b``, ``stats``, the product and the output of the op chain."""
    x = problem["x"]
    C, D = _widths(problem)
    op1 = k1.get_trimul_kf_k1_op(torch.bfloat16, C, D)
    op2 = k2.get_trimul_kf_k2_op(torch.bfloat16, D, outgoing)
    op3 = k3.get_trimul_kf_k3_op(torch.bfloat16, C, D, residual)
    assert op1 is not None and op2 is not None and op3 is not None
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
    a, b, stats = op1(x, problem["seqlen"], fold_in, _EPS)
    prod = op2(a, b)
    out = op3(
        prod,
        x,
        fold_out,
        stats,
        _EPS,
        residual=residual,
        actual_seqlen=problem["out_seqlen"] if residual else None,
    )
    return a, b, stats, prod, out


def _k1_reference(problem: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 channel-major ``a``, ``b`` ``[B, D, N, N]``: ``mask * (n @ W_p.T + b_p) * sigmoid(n @ W_g.T + b_g)``.

    ``n = LayerNorm(x)``, and ``mask[b, i, j] = j < seqlen[b, i]``.
    """
    x = problem["x"].float()
    B, N, _, C = x.shape
    normed = torch.nn.functional.layer_norm(
        x, (C,), problem["norm_in_weight"].float(), problem["norm_in_bias"].float(), _EPS
    )
    proj = torch.nn.functional.linear(normed, problem["p_in_weight"].float(), _float(problem["p_in_bias"]))
    gate = torch.nn.functional.linear(normed, problem["g_in_weight"].float(), _float(problem["g_in_bias"]))
    mask = torch.arange(N, device=x.device) < problem["seqlen"].reshape(B, N, 1)
    out = (proj * torch.sigmoid(gate) * mask[..., None]).permute(0, 3, 1, 2)
    D = out.shape[1] // 2
    return out[:, :D], out[:, D:]


def _k2_reference(a: torch.Tensor, b: torch.Tensor, outgoing: bool) -> torch.Tensor:
    """fp32 ``prod`` ``[B, D, N, N]``: ``a[i, k] b[j, k]`` summed over ``k`` outgoing, ``a[k, i] b[k, j]`` incoming."""
    return torch.einsum("bdik,bdjk->bdij" if outgoing else "bdki,bdkj->bdij", a.float(), b.float())


def _k3_reference(prod: torch.Tensor, problem: dict, residual: bool) -> torch.Tensor:
    """fp32 ``[B, N, N, C]``: ``(LN_out(P) @ W_out.T + b_out) * sigmoid(LN_in(x) @ W_g.T + b_g)``.

    ``P`` is ``prod`` channel-last. The fused residual gives ``(x + update) * mask``,
    ``mask[b, i, j] = j < out_seqlen[b, i]``.
    """
    x = problem["x"].float()
    B, N, _, C = x.shape
    products = torch.nn.functional.layer_norm(
        prod.float().permute(0, 2, 3, 1),
        (prod.shape[1],),
        problem["norm_out_weight"].float(),
        problem["norm_out_bias"].float(),
        _EPS,
    )
    normed = torch.nn.functional.layer_norm(
        x, (C,), problem["norm_in_weight"].float(), problem["norm_in_bias"].float(), _EPS
    )
    update = torch.nn.functional.linear(
        products, problem["p_out_weight"].float(), _float(problem["p_out_bias"])
    ) * torch.sigmoid(
        torch.nn.functional.linear(normed, problem["g_out_weight"].float(), _float(problem["g_out_bias"]))
    )
    if not residual:
        return update
    mask = torch.arange(N, device=x.device) < problem["out_seqlen"].reshape(B, N, 1)
    return (x + update) * mask[..., None]


def _float(bias: torch.Tensor | None) -> torch.Tensor | None:
    return None if bias is None else bias.float()


def _reference(problem: dict, outgoing: bool, residual: bool) -> torch.Tensor:
    a, b = _k1_reference(problem)
    return _k3_reference(_k2_reference(a, b, outgoing), problem, residual)


def _skip_without_room(B: int, N: int, C: int, D: int) -> None:
    """Skip a case the free GPU memory cannot hold.

    The fp32 reference is the largest term, and the widest cases need several
    GiB. Tests sharing one device run out before the kernels are at fault, so
    size the requirement from the case rather than failing on allocation.
    """
    torch.cuda.empty_cache()
    P = k1_config.ab_pitch(N, k1_config.ab_layout("pad_n_mod_16_8", N))
    footprint = B * (
        4 * N * N * C  # bf16 x and out
        + 4 * D * P * P  # bf16 a and b
        + 2 * D * N * N  # bf16 product
        + 8 * N * N * C  # fp32 reference and one intermediate
    )
    needed = footprint + footprint // 4
    if torch.cuda.mem_get_info()[0] < needed:
        pytest.skip(f"needs {needed / 2**30:.1f} GiB of free GPU memory")


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_chain_matches_reference(mode, case, monkeypatch):
    skip_if_not_sm90()
    B, N, C, D, outgoing, has_bias, residual, lengths = case
    _skip_without_room(B, N, C, D)
    caches = _use_mode(mode, monkeypatch)
    problem = _problem(B, N, C, has_bias, lengths, D=D)
    out = _run_chain(problem, outgoing, residual)
    _assert_dispatched(caches, mode)
    reference = _reference(problem, outgoing, residual)
    assert out.shape == problem["x"].shape and out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), reference, atol=_ATOL, rtol=0)


# Dense and padded K1 planes at C = 128, 256 and 384, K3 with and without the residual, and each K2 tile
# list's widths.
_BITWISE_CASES = [
    _CASES[0],
    _CASES[4],
    _CASES[7],
    _CASES[9],
    _CASES[11],
    _CASES[13],
    _CASES[14],
    _CASES[15],
    _CASES[17],
    _CASES[19],
    _CASES[21],
    _CASES[25],
    _CASES[27],
    _CASES[28],
    (1, 520, 256, 256, True, True, True, "random"),
]


@pytest.mark.parametrize("case", _BITWISE_CASES, ids=_case_id)
def test_cubin_chain_matches_source_chain_bitwise(case, monkeypatch):
    """Both paths run the same machine code, so their outputs agree bit for bit."""
    skip_if_not_sm90()
    if set(_CUTEDSL_MODES) != {"source", "cubin"}:
        pytest.skip("needs BIOIR_TEST_CUTEDSL_MODES=source,cubin")
    B, N, C, D, outgoing, has_bias, residual, lengths = case
    problem = _problem(B, N, C, has_bias, lengths, D=D)
    outputs = []
    for mode in ("source", "cubin"):
        with monkeypatch.context() as patch:
            caches = _use_mode(mode, patch)
            outputs.append(_run_chain(problem, outgoing, residual))
            _assert_dispatched(caches, mode)
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
def test_graph_replay_feeds_each_k1_the_x_its_k3_wrote(mode, monkeypatch):
    """tri_mul_out's K3 writes the x tri_mul_in's K1 reads; replay must not let K1 read it early.

    The node folds its weights once, so in the trunk K1 follows the previous K3 directly. At N = 64
    K3 leaves SMs free, and a K1 overlapping it would read the previous replay's x.
    """
    skip_if_not_sm90()
    caches = _use_mode(mode, monkeypatch)
    problem = _problem(1, 64, 256, True, "full")
    C = problem["x"].shape[-1]
    op1 = k1.get_trimul_kf_k1_op(torch.bfloat16, C, C)
    op2 = {outgoing: k2.get_trimul_kf_k2_op(torch.bfloat16, C, outgoing) for outgoing in (True, False)}
    op3 = k3.get_trimul_kf_k3_op(torch.bfloat16, C, C, True)
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

    def out_then_in(x: torch.Tensor, sync: bool) -> torch.Tensor:
        """tri_mul_out then tri_mul_in on ``x``; ``sync`` finishes each chain before the next starts."""
        for outgoing in (True, False):
            a, b, stats = op1(x, problem["seqlen"], fold_in, _EPS)
            prod = op2[outgoing](a, b)
            x = op3(prod, x, fold_out, stats, _EPS, residual=True, actual_seqlen=problem["out_seqlen"])
            if sync:
                torch.cuda.synchronize()
        return x

    static = problem["x"].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out_then_in(static, sync=False)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = out_then_in(static, sync=False)
    _assert_dispatched(caches, mode)
    for seed in range(1, 4):
        x = _problem(1, 64, C, True, "full", seed=seed)["x"]
        static.copy_(x)
        graph.replay()
        torch.testing.assert_close(out, out_then_in(x, sync=True), rtol=0, atol=0)


def _rows_reference(problem: dict, a: torch.Tensor, b: torch.Tensor, prod: torch.Tensor, outgoing: bool, rows):
    """fp32 ``a``, ``b``, product and output rows of the last batch, each from its stage's own inputs."""
    B, N, _, C = problem["x"].shape
    x = problem["x"][B - 1, rows].float()
    normed_in = torch.nn.functional.layer_norm(
        x, (C,), problem["norm_in_weight"].float(), problem["norm_in_bias"].float(), _EPS
    )
    proj = torch.nn.functional.linear(normed_in, problem["p_in_weight"].float(), _float(problem["p_in_bias"]))
    gate = torch.nn.functional.linear(normed_in, problem["g_in_weight"].float(), _float(problem["g_in_bias"]))
    mask = torch.arange(N, device=x.device) < problem["seqlen"][B - 1, rows, None]
    ab = (proj * torch.sigmoid(gate) * mask[..., None]).permute(2, 0, 1)
    D = ab.shape[0] // 2
    prod_rows = torch.empty((D, len(rows), N), device=x.device)
    for d in range(0, D, 32):  # b in fp32, 32 planes at a time
        planes = b[B - 1, d : d + 32].float()
        if outgoing:
            prod_rows[d : d + 32] = torch.einsum("drk,djk->drj", a[B - 1, d : d + 32, rows].float(), planes)
        else:
            prod_rows[d : d + 32] = torch.einsum("dkr,dkj->drj", a[B - 1, d : d + 32, :, rows].float(), planes)
    products = torch.nn.functional.layer_norm(
        prod[B - 1, :, rows].float().permute(1, 2, 0),
        (D,),
        problem["norm_out_weight"].float(),
        problem["norm_out_bias"].float(),
        _EPS,
    )
    update = torch.nn.functional.linear(
        products, problem["p_out_weight"].float(), _float(problem["p_out_bias"])
    ) * torch.sigmoid(
        torch.nn.functional.linear(normed_in, problem["g_out_weight"].float(), _float(problem["g_out_bias"]))
    )
    out_mask = torch.arange(N, device=x.device) < problem["out_seqlen"][B - 1, rows, None]
    return ab[:D], ab[D:], prod_rows, (x + update) * out_mask[..., None]


# One launch per op past 2^31 elements: N = 2904 pads a/b to P = 2944; B = 2 at N = 3008 shares a launch; and B = 2 at
# N = 2912, C = D = 256 puts the batch stride D * N * N past 2^31.
_LARGE_CASES = [(1, 2904, 256), (2, 3008, 128), (2, 2912, 256)]


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize("outgoing", [True, False], ids=["out", "in"])
@pytest.mark.parametrize(("B", "N", "C"), _LARGE_CASES, ids=lambda v: str(v))
def test_chain_past_int32_offsets_matches_reference_rows(mode, B, N, C, outgoing, monkeypatch):
    """Every stage's rows of the last batch, the last planes included, match fp32 once offsets pass 2^31."""
    skip_if_not_sm90()
    torch.cuda.empty_cache()
    P = k1.ab_pitch(N, "padded")
    footprint = 2 * B * (2 * N * N * C + 2 * C * P * P + C * N * N)  # bf16 x and out, a and b, product
    if torch.cuda.mem_get_info()[0] < max(32 * 2**30, footprint + footprint // 4):
        pytest.skip(f"needs {max(32 * 2**30, footprint + footprint // 4) >> 30} GiB of free GPU memory")
    caches = _use_mode(mode, monkeypatch)
    problem = _problem(B, N, C, True, "random")
    a, b, stats, prod, out = _run_chain_stages(problem, outgoing, True)
    _assert_dispatched(caches, mode)
    assert min(a.numel(), prod.numel(), out.numel()) > 2**31
    rows = [0, N // 2, N - 1]
    a_ref, b_ref, prod_ref, out_ref = _rows_reference(problem, a, b, prod, outgoing, rows)
    torch.testing.assert_close(a[B - 1, :, rows].float(), a_ref, atol=_ATOL, rtol=1e-2)
    torch.testing.assert_close(b[B - 1, :, rows].float(), b_ref, atol=_ATOL, rtol=1e-2)
    torch.testing.assert_close(prod[B - 1, :, rows].float(), prod_ref, atol=_ATOL, rtol=1e-2)
    torch.testing.assert_close(out[B - 1, rows].float(), out_ref, atol=_ATOL, rtol=0)


@pytest.mark.parametrize("shape", sorted(k1_config.shipped_shapes(90)))
def test_k1_hands_row_statistics_exactly_to_the_k3_that_reads_them(shape):
    """For every token count, K1 writes statistics exactly when the K3 it pairs with reads them."""
    C, D = shape
    assert shape in k3_config.shipped_shapes(90)
    for n in range(8, 4097, 8):
        assert k1_config.select(90, C, D, n).writes_stats == k3_config.select(90, C, D, n).reads_stats, n


@pytest.mark.parametrize("D", sorted(k2_config.shipped_widths(90)))
def test_k2_tiles_fill_their_clusters_and_deferred_stores(D):
    """Every token count runs a tile whose clusters fill and whose deferred stores have enough k-blocks."""
    for outgoing in (True, False):
        for n in range(8, 4097, 8):
            _, tile = k2_config.select(90, D, n, outgoing)
            assert -(-n // tile.tile_n) % tile.cluster_n == 0, (n, tile)
            assert tile.defer_kmin <= -(-n // k2_config.TILE_K), (n, tile)
    paired = k2_config.TrimulKFK2Tile("K2_1", 192)
    assert k2_config.runtime_tile(paired, 392) == k2_config.TrimulKFK2Tile("K2_1", 128)
    assert k2_config.runtime_tile(paired, 384) == paired


def test_shipped_shapes():
    shapes = {(32, 32), (64, 64), (128, 128), (256, 256), (384, 256)}
    assert k1_config.shipped_shapes(90) == shapes == k3_config.shipped_shapes(90)
    assert k2_config.shipped_widths(90) == {32, 64, 128, 256}
    assert k1_config.shipped_shapes(80) == frozenset()


def test_input_fold_interleaves_eight_row_blocks():
    C, D = 32, 16
    weight = torch.randn(2 * D, C).to(torch.bfloat16)
    gate = torch.randn(2 * D, C).to(torch.bfloat16)
    fold = k1.fold_input_weights(torch.ones(C), torch.zeros(C), weight, gate)
    for r in range(2 * D):
        assert torch.equal(fold.interleaved[(r // 8) * 16 + r % 8], fold.proj[r])
        assert torch.equal(fold.interleaved[(r // 8) * 16 + r % 8 + 8], fold.gate[r])
    assert fold.C == C and fold.D == D and fold.vec.shape == (8 * D,)


@pytest.mark.parametrize("config", [{"S=128": {"kernel_variant": "K1_9"}}, {"N=128": {"kernel_variant": "K1_0"}}, {}])
def test_malformed_k1_configs_are_rejected(config):
    with pytest.raises(ValueError):
        k1_config.parse_configs(config)


@pytest.mark.parametrize(
    "entry",
    [
        {"kernel_variant": "K2_1", "tile_n": 256},
        {"kernel_variant": "K2_0", "tile_n": 128},
        {"kernel_variant": "K2_0", "tile_n": 128, "cluster_m": 1, "defer_kmin": 0, "split_epi": True},
        {"kernel_variant": "K2_0", "tile_n": 136, "cluster_m": 2, "defer_kmin": 0, "split_epi": False},
    ],
)
def test_malformed_k2_tiles_are_rejected(entry):
    with pytest.raises(ValueError):
        k2_config.parse_tile(entry, "test")


def test_k2_anchors_name_one_tile_or_one_per_direction():
    shared = {"kernel_variant": "K2_1", "tile_n": 192}
    outgoing = {"kernel_variant": "K2_2", "tile_n": 208, "cluster_m": 2, "defer_kmin": 8, "split_epi": False}
    incoming = {"kernel_variant": "K2_0", "tile_n": 256, "cluster_m": 2, "defer_kmin": 8, "split_epi": True}
    configs = {"S=264": shared, "S=832": {"out": outgoing, "in": incoming}}
    out, inc = k2_config.parse_configs(configs, "o", True), k2_config.parse_configs(configs, "i", False)
    assert out[264] == inc[264] == (k2_config.TrimulKFK2Tile("K2_1", 192),)
    assert out[832] == (k2_config.parse_tile(outgoing, "o"),) and inc[832] == (k2_config.parse_tile(incoming, "i"),)
    for bad in ({"out": outgoing}, {"out": outgoing, "in": incoming, "other": shared}):
        with pytest.raises(ValueError):
            k2_config.parse_configs({"S=832": bad})


def test_ops_reject_misuse():
    skip_if_not_sm90()
    C = 64
    problem = _problem(1, 128, C, False, "full")
    op1 = k1.get_trimul_kf_k1_op(torch.bfloat16, C, C)
    op3 = k3.get_trimul_kf_k3_op(torch.bfloat16, C, C, True)
    fold_in = k1.fold_input_weights(
        problem["norm_in_weight"], problem["norm_in_bias"], problem["p_in_weight"], problem["g_in_weight"]
    )
    fold_out = k3.fold_output_weights(
        problem["norm_out_weight"],
        problem["norm_out_bias"],
        problem["norm_in_weight"],
        problem["norm_in_bias"],
        problem["p_out_weight"],
        problem["g_out_weight"],
    )
    x = problem["x"]
    assert not op1.accepts(x[:, :124, :124].contiguous(), problem["seqlen"][:, :124].contiguous())
    with pytest.raises(ValueError, match="multiple of 8"):
        op1(x[:, :124, :124].contiguous(), problem["seqlen"][:, :124].contiguous(), fold_in, _EPS)
    prod = torch.zeros(1, C, 128, 128, device="cuda", dtype=torch.bfloat16)
    stats = torch.zeros(1, 128 * 128, 2, device="cuda")
    with pytest.raises(ValueError, match="row statistics"):
        op3(prod, x, fold_out, stats, _EPS, residual=True, actual_seqlen=problem["out_seqlen"])
    with pytest.raises(ValueError, match="actual_seqlen exactly with residual"):
        op3(prod, x, fold_out, None, _EPS, residual=True)
    with pytest.raises(ValueError, match="must not alias x"):
        op3(prod, x, fold_out, None, _EPS, residual=False, out=x)
    # A partial overlap is aliasing too; a neighbouring range in the same buffer is not.
    buffer = torch.empty(2 * x.numel(), device="cuda", dtype=x.dtype)
    x_in_buffer = buffer[: x.numel()].view_as(x).copy_(x)
    with pytest.raises(ValueError, match="must not alias x"):
        op3(
            prod,
            x_in_buffer,
            fold_out,
            None,
            _EPS,
            residual=False,
            out=buffer[x.numel() // 2 :][: x.numel()].view_as(x),
        )
    op3(prod, x_in_buffer, fold_out, None, _EPS, residual=False, out=buffer[x.numel() :].view_as(x))
    assert k1.get_trimul_kf_k1_op(torch.float16, C, C) is None
    assert k1.get_trimul_kf_k1_op(torch.bfloat16, 96, 96) is None


@pytest.mark.parametrize(("B", "N"), [(1, 0), (0, 128)], ids=["no-tokens", "no-batch"])
def test_ops_reject_empty_problems(B, N):
    """An empty problem has nothing to launch; each op turns it away before its batch chunking."""
    skip_if_not_sm90()
    C = 64
    op1 = k1.get_trimul_kf_k1_op(torch.bfloat16, C, C)
    op2 = k2.get_trimul_kf_k2_op(torch.bfloat16, C, True)
    op3 = k3.get_trimul_kf_k3_op(torch.bfloat16, C, C, True)
    x = torch.empty(B, N, N, C, device="cuda", dtype=torch.bfloat16)
    seqlen = torch.empty(B, N, device="cuda", dtype=torch.int32)
    channel_major = torch.empty(B, C, N, N, device="cuda", dtype=torch.bfloat16)

    assert not op1.accepts(x, seqlen)
    assert not op2.accepts(channel_major, channel_major)
    assert not op3.accepts(channel_major, x)
    with pytest.raises(ValueError, match="non-empty"):
        op2(channel_major, channel_major)


def test_source_kernels_never_request_programmatic_dependent_launch():
    """The traced launches must not ask for PDL, which is unsafe under graph capture.

    PDL is correct under plain stream ordering, but a capturing stream records
    nodes rather than launching them and the attribute does not survive into
    the graph: ``griddepcontrol.wait`` is then left with no trigger behind it,
    so a replay can read a predecessor's output while it is still being
    written. The resulting mismatch is a race -- it surfaced on a 14-layer
    trunk yet not on an 8-layer one -- so no output comparison detects it
    reliably and this guards the invariant directly instead.

    ``use_pdl`` is a trace-time constant here, so this path cannot tell a
    capture from a plain launch. The CUBIN launcher decides per launch and
    keeps PDL everywhere except capture.
    """
    sources = Path(dsl_kernels.__file__).parent / "cute"
    checked = 0
    for name in ("sm90_trimul_kf_k1", "sm90_trimul_kf_k2", "sm90_trimul_kf_k3"):
        text = (sources / f"{name}.py").read_text()
        assert "use_pdl=False," in text, f"{name} should launch without PDL"
        assert "use_pdl=True" not in text, f"{name} requests PDL, which breaks CUDA graph replay"
        checked += text.count("use_pdl=")
    assert checked == 8, f"expected the chain's 8 launch sites, found {checked}"
