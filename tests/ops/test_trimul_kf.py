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

import pytest
import torch

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

# (B, N, C=D, outgoing, has_bias, residual, lengths), selecting every K1 variant, every K2 tile and
# its odd-cluster fallback (N = 392), and both K3 variants. N * N % 128 != 0 at N = 136 and 168,
# where batches run one launch each.
_CASES = [
    (1, 128, 64, True, False, False, "full"),
    (1, 128, 32, False, True, True, "tail"),
    (1, 256, 32, True, True, True, "random"),
    (1, 256, 128, False, False, True, "random"),
    (1, 384, 128, False, True, True, "tail"),
    (1, 392, 128, True, False, True, "random"),
    (1, 512, 64, True, True, True, "random"),
    (2, 136, 128, False, True, True, "random"),
    (3, 168, 64, True, True, False, "random"),
    (1, 128, 256, False, True, True, "full"),
    (1, 256, 256, True, True, True, "random"),
    (1, 384, 256, False, False, True, "tail"),
    (2, 392, 256, True, True, True, "random"),
    (1, 1024, 256, False, True, True, "tail"),
]


def _case_id(case) -> str:
    B, N, C, outgoing, has_bias, residual, lengths = case
    return f"B{B}-N{N}-C{C}-{'out' if outgoing else 'in'}-bias{int(has_bias)}-res{int(residual)}-{lengths}"


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


def _problem(B: int, N: int, C: int, has_bias: bool, lengths: str, seed: int = 0) -> dict[str, torch.Tensor | None]:
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, device="cuda", generator=generator) * scale).to(torch.bfloat16)

    D = C
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


def _run_chain(problem: dict, outgoing: bool, residual: bool) -> torch.Tensor:
    x = problem["x"]
    C = x.shape[-1]
    op1 = k1.get_trimul_kf_k1_op(torch.bfloat16, C, C)
    op2 = k2.get_trimul_kf_k2_op(torch.bfloat16, C, outgoing)
    op3 = k3.get_trimul_kf_k3_op(torch.bfloat16, C, C, residual)
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
    return op3(
        prod,
        x,
        fold_out,
        stats,
        _EPS,
        residual=residual,
        actual_seqlen=problem["out_seqlen"] if residual else None,
    )


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


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_chain_matches_reference(mode, case, monkeypatch):
    skip_if_not_sm90()
    B, N, C, outgoing, has_bias, residual, lengths = case
    caches = _use_mode(mode, monkeypatch)
    problem = _problem(B, N, C, has_bias, lengths)
    out = _run_chain(problem, outgoing, residual)
    _assert_dispatched(caches, mode)
    reference = _reference(problem, outgoing, residual)
    assert out.shape == problem["x"].shape and out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), reference, atol=_ATOL, rtol=0)


@pytest.mark.parametrize("case", [_CASES[0], _CASES[4], _CASES[7], _CASES[11]], ids=_case_id)
def test_cubin_chain_matches_source_chain_bitwise(case, monkeypatch):
    """Both paths run the same machine code, so their outputs agree bit for bit."""
    skip_if_not_sm90()
    if set(_CUTEDSL_MODES) != {"source", "cubin"}:
        pytest.skip("needs BIOIR_TEST_CUTEDSL_MODES=source,cubin")
    B, N, C, outgoing, has_bias, residual, lengths = case
    problem = _problem(B, N, C, has_bias, lengths)
    outputs = []
    for mode in ("source", "cubin"):
        with monkeypatch.context() as patch:
            caches = _use_mode(mode, patch)
            outputs.append(_run_chain(problem, outgoing, residual))
            _assert_dispatched(caches, mode)
    assert torch.equal(outputs[0], outputs[1])


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
    for n in range(8, 4097, 8):
        _, tile = k2_config.select(90, D, n)
        assert -(-n // tile.tile_n) % tile.cluster_n == 0, (n, tile)
        assert tile.defer_kmin <= -(-n // k2_config.TILE_K), (n, tile)
    _, odd = k2_config.select(90, D, 392)
    assert odd == k2_config.TrimulKFK2Tile("K2_1", 128)


def test_shipped_shapes_cover_dims_up_to_256():
    shapes = {(32, 32), (64, 64), (128, 128), (256, 256)}
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
    ],
)
def test_malformed_k2_tiles_are_rejected(entry):
    with pytest.raises(ValueError):
        k2_config.parse_tile(entry, "test")


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
