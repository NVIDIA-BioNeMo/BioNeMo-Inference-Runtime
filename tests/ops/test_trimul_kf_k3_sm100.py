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
"""SM100/SM103 TriMul KF K3: every shipped ``(C, D)`` against the fp32 reference, graph replay, the CUBINs against
the source, and configs."""

from __future__ import annotations

import json

import pytest
import torch

from bionemo_ir._torch.custom_ops import trimul_kf_k3 as k3
from bionemo_ir._torch.custom_ops.trimul_kf_k3 import _config as k3_config
from bionemo_ir._torch.custom_ops.trimul_kf_k3.cutedsl import TrimulKFK3CuTe
from tests._torch import skip_if_not_sm100_family, source_module_available

from .test_trimul_kf import _ATOL, _EPS, _float, _k1_reference, _k2_reference, _k3_reference, _problem, library_carries

_VARIANTS = {(32, 32): "K3_B0", (64, 64): "K3_B0", (128, 128): "K3_B0", (256, 256): "K3_B1", (384, 256): "K3_B2"}
# (variant, largest token count) a width runs below its _VARIANTS entry: K3_B0 with the two-tile x ring
_SMALL_N_VARIANTS = {(128, 128): ("K3_B3", 192)}
_requires_source = pytest.mark.skipif(
    not source_module_available("bionemo_ir._torch.custom_ops.trimul_kf_k3._source"),
    reason="needs the K3 kernel sources",
)

# (B, N, C, D, lengths, has_bias): every shipped pair with masked rows, B > 1, N % 16 == 8 (N * N not
# a multiple of 128, so batches end in partial tiles) and N * N not a multiple of 256 (half a 2-CTA tile).
_CASES = [
    (1, 128, 32, 32, "tail", True),
    (2, 136, 32, 32, "random", True),
    (1, 264, 32, 32, "random", False),
    (1, 128, 64, 64, "full", False),
    (3, 168, 64, 64, "random", True),
    (1, 392, 64, 64, "tail", True),
    (1, 256, 128, 128, "random", False),
    (2, 136, 128, 128, "random", True),
    (1, 520, 128, 128, "tail", True),
    (1, 128, 256, 256, "full", True),
    (1, 264, 256, 256, "tail", False),
    (2, 392, 256, 256, "random", True),
    (1, 104, 384, 256, "random", False),
    (1, 136, 384, 256, "tail", True),
    (2, 200, 384, 256, "random", False),
    (1, 520, 384, 256, "random", True),
]
# Fewer rows than one tile, or than one 2-CTA pair's tile, and fewer tiles than CTAs: N * N = 64 ... 4096.
_TINY_CASES = [(B, N, C, D) for (C, D) in _VARIANTS for N in (8, 16, 24, 40, 64) for B in (1, 2)]


def _case_id(case) -> str:
    B, N, C, D, lengths, has_bias = case
    return f"B{B}-N{N}-C{C}-D{D}-{lengths}-bias{int(has_bias)}"


def _fold(problem: dict) -> k3.TrimulKFOutputFold:
    return k3.fold_output_weights(
        problem["norm_out_weight"],
        problem["norm_out_bias"],
        problem["norm_in_weight"],
        problem["norm_in_bias"],
        problem["p_out_weight"],
        problem["g_out_weight"],
        problem["p_out_bias"],
        problem["g_out_bias"],
    )


def _k3_problem(B: int, N: int, C: int, D: int, lengths: str, has_bias: bool, seed: int = 0):
    """A K3 problem: the chain problem, its fp32-reference product in bf16 and the output fold."""
    problem = _problem(B, N, C, has_bias, lengths, seed=seed, D=D)
    a, b = _k1_reference(problem)
    prod = _k2_reference(a.to(torch.bfloat16), b.to(torch.bfloat16), True).to(torch.bfloat16).contiguous()
    return problem, prod, _fold(problem)


def _op(C: int, D: int, residual: bool) -> k3.TrimulKFK3Op:
    op = k3.get_trimul_kf_k3_op(torch.bfloat16, C, D, residual)
    assert op is not None and op.select(512).kernel_variant == _VARIANTS[(C, D)]
    return op


def _run(op, prod, problem, fold, residual: bool, out: torch.Tensor | None = None) -> torch.Tensor:
    seqlen = problem["out_seqlen"] if residual else None
    return op(prod, problem["x"], fold, None, _EPS, residual=residual, actual_seqlen=seqlen, out=out)


def _skip_unless_sm100_source() -> None:
    skip_if_not_sm100_family()
    if not source_module_available("bionemo_ir._torch.custom_ops.trimul_kf_k3._source"):
        pytest.skip("needs the K3 kernel sources")


@pytest.mark.parametrize("residual", [True, False], ids=["res", "nores"])
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_k3_matches_reference(case, residual):
    _skip_unless_sm100_source()
    B, N, C, D, lengths, has_bias = case
    problem, prod, fold = _k3_problem(B, N, C, D, lengths, has_bias)
    out = _run(_op(C, D, residual), prod, problem, fold, residual)
    assert out.shape == problem["x"].shape and out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), _k3_reference(prod, problem, residual), atol=_ATOL, rtol=0)


@pytest.mark.parametrize("residual", [True, False], ids=["res", "nores"])
@pytest.mark.parametrize(("B", "N", "C", "D"), _TINY_CASES, ids=lambda v: str(v))
def test_k3_tiny_token_counts_match_reference(B, N, C, D, residual):
    """Tiles mostly past N * N, 2-CTA pairs with an empty half and grids smaller than the SM count."""
    _skip_unless_sm100_source()
    problem, prod, fold = _k3_problem(B, N, C, D, "random", True, seed=N)
    out = _run(_op(C, D, residual), prod, problem, fold, residual)
    torch.testing.assert_close(out.float(), _k3_reference(prod, problem, residual), atol=_ATOL, rtol=0)


def _reference_rows(prod, x, out_seqlen, problem, residual: bool, i0: int, i1: int) -> torch.Tensor:
    """``_k3_reference`` on rows ``i0:i1`` of every batch; K3 is row-local."""
    N, C, D = x.shape[1], x.shape[-1], prod.shape[1]
    xr = x[:, i0:i1].float()
    products = torch.nn.functional.layer_norm(
        prod[:, :, i0:i1].float().permute(0, 2, 3, 1),
        (D,),
        problem["norm_out_weight"].float(),
        problem["norm_out_bias"].float(),
        _EPS,
    )
    normed = torch.nn.functional.layer_norm(
        xr, (C,), problem["norm_in_weight"].float(), problem["norm_in_bias"].float(), _EPS
    )
    update = torch.nn.functional.linear(
        products, problem["p_out_weight"].float(), _float(problem["p_out_bias"])
    ) * torch.sigmoid(
        torch.nn.functional.linear(normed, problem["g_out_weight"].float(), _float(problem["g_out_bias"]))
    )
    if not residual:
        return update
    mask = torch.arange(N, device=x.device) < out_seqlen[:, i0:i1, None]
    return (xr + update) * mask[..., None]


@pytest.mark.parametrize("residual", [True, False], ids=["res", "nores"])
@pytest.mark.parametrize("lengths", ["tail", "random"])
@pytest.mark.parametrize(("N", "C"), [(1016, 256), (1016, 384), (2056, 256), (2056, 384)], ids=lambda v: str(v))
def test_k3_large_n_masked_repeats_are_bit_identical(N, C, lengths, residual):
    """Many tiles per CTA pair, rings wrapping hundreds of times: every repeat matches the first bit for bit.

    A pipeline race corrupts rows in a few late tiles of a run, differently every run, so the repeats
    must agree exactly and each row must match the fp32 reference.
    """
    _skip_unless_sm100_source()
    D = 256
    torch.cuda.empty_cache()
    needed = N * N * (8 * C + 2 * D) + 16 * 256 * N * (C + D)  # x, two outputs, product, reference rows
    if torch.cuda.mem_get_info()[0] < needed + needed // 4:
        pytest.skip(f"needs {needed / 2**30:.1f} GiB of free GPU memory")
    problem = _problem(1, 8, C, True, "full", seed=C, D=D)
    fold = _fold(problem)
    generator = torch.Generator(device="cuda").manual_seed(N + C)
    x = torch.randn(1, N, N, C, generator=generator, device="cuda").to(torch.bfloat16)
    prod = (torch.randn(1, D, N, N, generator=generator, device="cuda") * 4).to(torch.bfloat16)
    if lengths == "tail":
        seqlen = torch.full((1, N), N - 5, dtype=torch.int32, device="cuda")
        seqlen[:, N - 5 :] = 0
    else:
        seqlen = torch.randint(0, N + 1, (1, N), generator=generator, device="cuda", dtype=torch.int32)
    op = _op(C, D, residual)
    first = op(prod, x, fold, None, _EPS, residual=residual, actual_seqlen=seqlen if residual else None)
    for i0 in range(0, N, 256):
        i1 = min(N, i0 + 256)
        reference = _reference_rows(prod, x, seqlen, problem, residual, i0, i1)
        torch.testing.assert_close(first[:, i0:i1].float(), reference, atol=_ATOL, rtol=0)
        del reference
    for _ in range(3):
        again = op(prod, x, fold, None, _EPS, residual=residual, actual_seqlen=seqlen if residual else None)
        assert torch.equal(again, first)
        del again


@pytest.mark.parametrize("residual", [True, False], ids=["res", "nores"])
@pytest.mark.parametrize(("C", "D"), sorted(_VARIANTS))
@pytest.mark.parametrize("N", [16, 136])
def test_k3_graph_replay_matches_eager(N, C, D, residual):
    """A captured launch replays on new inputs into the caller's ``out`` bit for bit like eager calls.

    N = 16 is the token count the graphed RFD4 trunk test runs the KF chain at.
    """
    _skip_unless_sm100_source()
    op = _op(C, D, residual)
    problem, prod, fold = _k3_problem(2, N, C, D, "random", True)
    static = {"x": problem["x"].clone(), "prod": prod.clone(), "out_seqlen": problem["out_seqlen"].clone()}
    out = torch.empty_like(static["x"])

    def step() -> torch.Tensor:
        return _run(op, static["prod"], static, fold, residual, out=out)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = step()
    assert captured is out
    for seed in range(1, 4):
        fresh, fresh_prod, _ = _k3_problem(2, N, C, D, "random", True, seed=seed)
        static["x"].copy_(fresh["x"])
        static["prod"].copy_(fresh_prod)
        static["out_seqlen"].copy_(fresh["out_seqlen"])
        graph.replay()
        torch.cuda.synchronize()
        expected = _run(op, fresh_prod, fresh, fold, residual)
        assert torch.equal(out, expected)


def test_k3_writes_the_given_out_and_takes_no_row_statistics():
    _skip_unless_sm100_source()
    C, D = 64, 64
    op = _op(C, D, True)
    problem, prod, fold = _k3_problem(1, 128, C, D, "random", True)
    assert not op.select(128).reads_stats
    out = torch.full_like(problem["x"], float("nan"))
    assert _run(op, prod, problem, fold, True, out=out) is out
    assert torch.equal(out, _run(op, prod, problem, fold, True))
    stats = torch.zeros(1, 128 * 128, 2, device="cuda")
    with pytest.raises(ValueError, match="row statistics"):
        op(prod, problem["x"], fold, stats, _EPS, residual=True, actual_seqlen=problem["out_seqlen"])
    with pytest.raises(ValueError, match="must not alias x"):
        _run(op, prod, problem, fold, False, out=problem["x"])


def test_k3_ships_from_the_library_on_sm100_under_force_cubin(monkeypatch):
    """Under CUTEDSL_FORCE_CUBIN=1 the op exists exactly when the CUBIN library carries this SM's images; otherwise
    the layer keeps its regular path."""
    skip_if_not_sm100_family()
    from tests._torch import SM_VERSION

    monkeypatch.setattr(TrimulKFK3CuTe, "_compiled_cache", {})
    carried = library_carries("trimul_kf_k3", SM_VERSION)
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    for C, D in _VARIANTS:
        for residual in (True, False):
            assert (k3.get_trimul_kf_k3_op(torch.bfloat16, C, D, residual) is not None) == carried


# (B, N, C, D): every variant, K3_B3 at small N, partial tiles and partial 2-CTA tiles, several batches.
_CUBIN_CASES = [
    (2, 136, 32, 32),
    (1, 392, 64, 64),
    (2, 136, 128, 128),
    (1, 520, 128, 128),
    (1, 264, 256, 256),
    (2, 200, 384, 256),
    (1, 520, 384, 256),
]


@pytest.mark.parametrize("residual", [True, False], ids=["res", "nores"])
@pytest.mark.parametrize(("B", "N", "C", "D"), _CUBIN_CASES, ids=lambda v: str(v))
def test_cubin_matches_source_on_this_gpu(B, N, C, D, residual, monkeypatch):
    """The packaged CUBIN writes the source kernel's output bit for bit."""
    _skip_unless_sm100_source()
    from tests._torch import SM_VERSION

    if not library_carries("trimul_kf_k3", SM_VERSION):
        pytest.skip(f"this build's CUBIN library carries no SM{SM_VERSION} K3 images")
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    problem, prod, fold = _k3_problem(B, N, C, D, "random", True, seed=N)
    op = k3.get_trimul_kf_k3_op(torch.bfloat16, C, D, residual)
    outputs = []
    for force in (False, True):
        if force:
            monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
        cache: dict = {}
        monkeypatch.setattr(TrimulKFK3CuTe, "_compiled_cache", cache)
        outputs.append(_run(op, prod, problem, fold, residual).clone())
        assert (
            all(type(executable).__name__ == "TrimulKFK3Sm100CubinExecutable" for executable in cache.values()) == force
        )
    assert torch.equal(outputs[0], outputs[1])


def test_cubin_launcher_rejects_an_output_overlapping_x(monkeypatch):
    """The CUBIN launcher, which callers can reach without the op's checks, refuses an output sharing any bytes with
    x, and writes the neighbouring range of the same buffer as the op does."""
    skip_if_not_sm100_family()
    from tests._torch import SM_VERSION

    if not library_carries("trimul_kf_k3", SM_VERSION):
        pytest.skip(f"this build's CUBIN library carries no SM{SM_VERSION} K3 images")
    C, D, N = 64, 64, 128
    problem, prod, fold = _k3_problem(1, N, C, D, "random", True)
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    monkeypatch.setattr(TrimulKFK3CuTe, "_compiled_cache", {})
    backend = TrimulKFK3CuTe(SM_VERSION)
    selection = backend.select(C, D, N)
    executable = backend._get_or_compile(C, D, selection.kernel_variant, False)
    w_out, w_gate, vec = fold.weights(selection)
    x = problem["x"]
    elements = x.numel()
    buffer = torch.empty(2 * elements, device="cuda", dtype=x.dtype)
    x_in_buffer = buffer[:elements].copy_(x.view(-1))

    def launch(output: torch.Tensor) -> None:
        executable(prod.view(-1), x_in_buffer, w_out.view(-1), w_gate.view(-1), vec, None, output, N, 1, _EPS)

    with pytest.raises(ValueError, match="must not alias x"):
        launch(buffer[elements // 2 : elements // 2 + elements])
    launch(buffer[elements:])
    torch.cuda.synchronize()
    assert torch.equal(
        buffer[elements:].view_as(x),
        _run(k3.get_trimul_kf_k3_op(torch.bfloat16, C, D, False), prod, problem, fold, False),
    )


# ---------------------------------------------------------------------------------------------- CPU


@pytest.mark.parametrize("sm", [100, 103])
def test_sm100_configs_select_the_tuned_variant_for_every_token_count(sm):
    assert k3_config.KERNEL_ABIS[sm] == "sm100"
    assert k3_config.shipped_shapes(sm) == set(_VARIANTS)
    for (C, D), variant in _VARIANTS.items():
        assert (C, D) in k3_config.SM100_VARIANT_SHAPES[variant]
        small_variant, small_n = _SMALL_N_VARIANTS.get((C, D), (variant, 0))
        assert (C, D) in k3_config.SM100_VARIANT_SHAPES[small_variant]
        for n in range(8, 4097, 8):
            selection = k3_config.select(sm, C, D, n)
            expected = small_variant if n <= small_n else variant
            assert selection.kernel_variant == expected and not selection.reads_stats, n


def test_sm100_and_sm103_bundles_are_identical():
    for C, D in _VARIANTS:
        bundles = [json.loads((k3_config.CONFIGS_DIR / f"C{C}_D{D}_sm{sm}.json").read_text()) for sm in (100, 103)]
        assert bundles[0] == bundles[1] and bundles[0]["kernel_abi"] == "sm100"


def test_sm90_selection_is_unchanged():
    """The parser the ABIs share still reads every SM90 bundle's anchors and product layout rules as written."""
    assert k3_config.shipped_shapes(90) == set(_VARIANTS)
    for C, D in _VARIANTS:
        configs = json.loads((k3_config.CONFIGS_DIR / f"C{C}_D{D}_sm90.json").read_text())["configs"]
        buckets = {int(key.removeprefix("S=")): entry for key, entry in configs.items()}
        assert k3_config.anchors(90, C, D) == {bucket: entry["kernel_variant"] for bucket, entry in buckets.items()}
        rules = {bucket: entry.get("prod_layout", "dense") for bucket, entry in buckets.items()}
        assert k3_config.layout_rules(90, C, D) == rules


@pytest.mark.parametrize(
    ("kernel_abi", "variant"), [("sm100", "K3_0"), ("sm100", "K3_4"), ("sm100", "K3_B4"), ("sm90", "K3_B0")]
)
def test_variants_outside_their_abi_are_rejected(kernel_abi, variant):
    with pytest.raises(ValueError, match="unknown"):
        k3_config.parse_configs({"S=128": {"kernel_variant": variant}}, "test", kernel_abi)


def test_sm100_variant_for_another_width_is_rejected(tmp_path, monkeypatch):
    bundle = {"kernel_abi": "sm100", "configs": {"S=128": {"kernel_variant": "K3_B1"}}}
    (tmp_path / "C384_D256_sm100.json").write_text(json.dumps(bundle))
    monkeypatch.setattr(k3_config, "CONFIGS_DIR", tmp_path)
    with pytest.raises(ValueError, match="does not run C=384, D=256"):
        k3_config.anchors(100, 384, 256)


@pytest.mark.parametrize("sm", [100, 103])
def test_force_cubin_ships_sm100_from_the_library(sm, monkeypatch):
    backend = TrimulKFK3CuTe(sm)
    monkeypatch.setattr(TrimulKFK3CuTe, "_compiled_cache", {})
    carried = library_carries("trimul_kf_k3", sm)
    monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
    for C, D in _VARIANTS:
        for residual in (True, False):
            assert backend.ships(C, D, residual) == carried
    # an image of a variant the shape's configs never name does not ship
    with pytest.raises(RuntimeError, match="no CUBIN is available"):
        backend._get_or_compile(256, 256, "K3_B0", True)


@_requires_source
@pytest.mark.parametrize("sm", [100, 103])
def test_sm100_ships_from_source(sm, monkeypatch):
    monkeypatch.delenv("CUTEDSL_FORCE_CUBIN", raising=False)
    backend = TrimulKFK3CuTe(sm)
    assert all(backend.ships(C, D, residual) for C, D in _VARIANTS for residual in (True, False))
    assert not backend.ships(96, 96, True)


@_requires_source
@pytest.mark.parametrize(
    ("variant", "C", "D"),
    [(variant, C, D) for variant, shapes in sorted(k3_config.SM100_VARIANT_SHAPES.items()) for C, D in sorted(shapes)],
)
def test_sm100_source_builds_every_variant(variant, C, D):
    from bionemo_ir._torch.custom_ops.trimul_kf_k3 import _source as k3_source

    assert k3_config.variant_abi(variant) == "sm100" and k3_source.ptxas_options(variant) == ""
    implementation = k3_source.source_implementation("sm100", None, variant)
    assert implementation.endswith(f".Sm100TrimulKF_K3_{variant.removeprefix('K3_B')}")
    for residual in (True, False):
        kernel = k3_source.make_kernel(variant, C, D, residual, 148)
        assert (kernel.C, kernel.D, kernel.residual) == (C, D, residual)
        assert len(k3_source.make_fake_args(C, D, variant, residual)) == 10
