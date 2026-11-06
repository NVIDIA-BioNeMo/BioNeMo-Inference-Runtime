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
from dataclasses import dataclass, field

import pytest
import torch

from bionemo_ir._torch.custom_ops.gated_sigmoid import (
    GatedSigmoidCuTe,
    _classify_m_range,
    _invoke_vanilla_gated_sigmoid,
    get_gated_sigmoid_op,
)
from bionemo_ir._torch.custom_ops.gated_sigmoid import _config as gated_config
from bionemo_ir._torch.custom_ops.gated_sigmoid import cutedsl as gated_cutedsl
from bionemo_ir._torch.custom_ops.gated_sigmoid import ops as gated_ops
from bionemo_ir._torch.custom_ops.gated_sigmoid._cubin import GatedSigmoidCubinExecutable
from bionemo_ir._torch.utils.kernel import load_kernel_configs
from tests._torch import (
    _CUTEDSL_MODE_CACHES,
    SM_VERSION,
    cutedsl_test_modes,
    require_cubin_library,
    run_cutedsl_test_mode,
    skip_if_no_cutedsl,
)

_CUTEDSL_MODES = cutedsl_test_modes("bionemo_ir._torch.custom_ops.gated_sigmoid._source")


def _skip_cubin_without_an_image(mode: str) -> None:
    """CUBIN mode needs an image for this SM. The Blackwell packs are not published yet."""
    if mode == "cubin" and SM_VERSION not in gated_cutedsl.published_cubin_sms():
        pytest.skip(f"the gated-sigmoid CUBIN pack has no image for SM{SM_VERSION}")


_SUPPORTED_SMS = tuple(sorted(gated_ops._SUPPORTED_SM))
_BLACKWELL_SMS = (100, 103, 120, 121)


@pytest.fixture
def no_fallback(monkeypatch):
    """Fail a call that would run PyTorch instead of the kernel."""

    def reject(*_args, **_kwargs):
        raise AssertionError("unexpected vanilla fallback")

    monkeypatch.setattr(gated_cutedsl, "_invoke_vanilla_gated_sigmoid", reject)


@pytest.fixture
def source_free_selection(monkeypatch):
    """Fail configuration selection that imports a private kernel."""

    def reject(_implementation):
        raise AssertionError("configuration selection imported a private kernel")

    monkeypatch.setattr(gated_config, "resolve_implementation", reject)


def _broadcast_problem(
    dtype, K, N, outer, mult, inner, *, has_bias=True, has_residual=False, has_mask=False, seed=0
) -> dict:
    """Operands of one call whose ``outer * inner`` gate rows each serve ``mult`` output rows."""
    torch.manual_seed(seed)
    s = torch.randn(outer, 1, inner, K, device="cuda", dtype=dtype)
    weight = (torch.randn(N, K, device="cuda") / K**0.5).to(dtype)
    bias = torch.randn(N, device="cuda", dtype=dtype) if has_bias else None
    mha_out = torch.randn(outer, mult, inner, N, device="cuda", dtype=dtype)
    residual = torch.randn_like(mha_out) if has_residual else None
    mask = (torch.rand(outer, mult, inner, device="cuda") > 0.3).to(dtype) if has_mask else None
    return {"s": s, "weight": weight, "mha_out": mha_out, "bias": bias, "residual": residual, "mask": mask}


def _assert_matches_pytorch(result: torch.Tensor, problem: dict) -> None:
    def fp32(name):
        return None if problem[name] is None else problem[name].float()

    reference = _invoke_vanilla_gated_sigmoid(
        fp32("s"), fp32("weight"), fp32("mha_out"), fp32("bias"), residual=fp32("residual"), mask=problem["mask"]
    )
    torch.testing.assert_close(result.float(), reference, atol=0.05, rtol=0.02)


def _ref_gated_sigmoid(
    s: torch.Tensor,
    weight: torch.Tensor,
    mha_out: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference implementation in fp32 for tight tolerance checks."""
    gate = torch.nn.functional.linear(s.float(), weight.float(), bias.float() if bias is not None else None)
    return (gate.sigmoid() * mha_out.float()).to(s.dtype)


# ---------------------------------------------------------------------------
# Parametrized M values
# ---------------------------------------------------------------------------

# Odd M values exercise CTA-tile predicate logic (partial tiles).
# CTA tiles by range: short=64x64, medium=64x128, long=128x64.

SHORT_M_VALUES = [
    1,  # single row
    33,  # odd, < one CTA row tile
    65,  # one full tile + 1 row
    127,  # just under 2 tiles (64*2=128)
    511,  # large odd
    999,  # odd, near boundary
    1024,  # exact boundary
]

MEDIUM_M_VALUES = [
    1025,  # boundary + 1
    1279,  # odd
    1537,  # odd
    2047,  # boundary - 1
    2048,  # exact boundary
]

LONG_M_VALUES = [
    2049,  # boundary + 1
    2561,  # odd
    3001,  # odd
    3455,  # odd, near upper benchmark limit
]

# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    K: int = 128
    N_out: int = 128
    M_values: list[int] = field(default_factory=lambda: [256])
    has_bias: bool = False
    dtype: torch.dtype = torch.bfloat16
    atol: float = 0.05
    rtol: float = 1e-2


# ---------------------------------------------------------------------------
# CuTe kernel tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "bias"])
@pytest.mark.parametrize("M", [65, 513, 1025, 2049], ids=["tiny", "short", "medium", "long"])
def test_gated_sigmoid_source_and_cubin(mode, dtype, has_bias, M, monkeypatch):
    _skip_cubin_without_an_image(mode)
    if SM_VERSION not in _SUPPORTED_SMS:
        pytest.skip(f"gated-sigmoid CUBINs do not target SM{SM_VERSION}")
    torch.manual_seed(13)
    K, N = 128, 256
    s = torch.randn(M, K, device="cuda", dtype=dtype)
    weight = torch.randn(N, K, device="cuda", dtype=dtype)
    mha_out = torch.randn(M, N, device="cuda", dtype=dtype)
    bias = torch.randn(N, device="cuda", dtype=dtype) if has_bias else None

    result = run_cutedsl_test_mode(
        mode,
        monkeypatch,
        GatedSigmoidCuTe,
        gated_cutedsl,
        lambda: GatedSigmoidCuTe()(s, weight, mha_out, bias),
    )
    reference = _invoke_vanilla_gated_sigmoid(s, weight, mha_out, bias)
    torch.testing.assert_close(result, reference, atol=0.05, rtol=0.02)


def test_gated_sigmoid_force_cubin_ignores_warmed_source(monkeypatch):
    backend = GatedSigmoidCuTe()
    resolved = gated_config.resolve(backend._sm_version, 128, 256, 65)
    key = (backend._sm_version, torch.bfloat16, True, False, False, 128, 256) + backend._tile_signature(resolved)
    cached_source, cubin = object(), object()
    monkeypatch.setattr(GatedSigmoidCuTe, "_compiled_cache", {key: cached_source})
    monkeypatch.setattr(backend, "_load_cubin_executable", lambda *_args, **_kwargs: cubin)
    executable = backend._get_or_compile(torch.bfloat16, True, False, False, 128, 256, 65, resolved, force_cubin=True)
    assert executable is cubin


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize("has_mask", [False, True], ids=["nomask", "mask"])
@pytest.mark.parametrize("has_residual", [False, True], ids=["nores", "res"])
@pytest.mark.parametrize(
    "K,N",
    # Every tuned shape runs the SM90 kernel on Hopper and the SM80 one elsewhere.
    [(384, 768), (256, 768), (768, 768), (128, 128)],
    ids=["K384_N768", "K256_N768", "K768_N768", "K128_N128"],
)
@pytest.mark.parametrize(
    "outer,mult,inner,bucket",
    # s has outer * inner rows, which pick the M bucket.
    [(1, 5, 65, "tiny"), (1, 1, 300, "short"), (2, 1, 700, "medium"), (1, 3, 2100, "long")],
    ids=["tiny_mult5", "short_mult1", "medium_mult1", "long_mult3"],
)
def test_gated_sigmoid_residual_mask(
    mode, has_mask, has_residual, K, N, outer, mult, inner, bucket, no_fallback, monkeypatch
):
    _skip_cubin_without_an_image(mode)
    if SM_VERSION not in _SUPPORTED_SMS:
        pytest.skip(f"gated-sigmoid CUBINs do not target SM{SM_VERSION}")
    assert _classify_m_range(outer * inner) == bucket
    problem = _broadcast_problem(
        torch.bfloat16, K, N, outer, mult, inner, has_residual=has_residual, has_mask=has_mask, seed=7
    )
    result = run_cutedsl_test_mode(
        mode, monkeypatch, GatedSigmoidCuTe, gated_cutedsl, lambda: GatedSigmoidCuTe()(**problem)
    )
    _assert_matches_pytorch(result, problem)


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize("K,N", [(384, 768), (128, 128)], ids=["K384_N768", "K128_N128"])
def test_gated_sigmoid_residual_inplace(mode, K, N, monkeypatch):
    """Writing the sum over the residual matches the out-of-place result exactly."""
    _skip_cubin_without_an_image(mode)
    if SM_VERSION not in _SUPPORTED_SMS:
        pytest.skip(f"gated-sigmoid CUBINs do not target SM{SM_VERSION}")
    torch.manual_seed(11)
    dtype = torch.bfloat16
    s = torch.randn(2, 1, 333, K, device="cuda", dtype=dtype)
    weight = (torch.randn(N, K, device="cuda") / K**0.5).to(dtype)
    mha_out = torch.randn(2, 5, 333, N, device="cuda", dtype=dtype)
    residual = torch.randn_like(mha_out)
    mask = (torch.rand(2, 5, 333, device="cuda") > 0.3).to(dtype)

    def operation() -> torch.Tensor:
        expected = GatedSigmoidCuTe()(s, weight, mha_out, None, residual=residual, mask=mask)
        inplace = residual.clone()
        out = GatedSigmoidCuTe()(s, weight, mha_out, None, output=inplace, residual=inplace, mask=mask)
        assert out.data_ptr() == inplace.data_ptr()
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        return out

    run_cutedsl_test_mode(mode, monkeypatch, GatedSigmoidCuTe, gated_cutedsl, operation)


def test_gated_sigmoid_sm90_config_selects_sm90_kernel(source_free_selection):
    for K in (384, 768):
        config = gated_config.get_kernel_config(90, K, 768, 65)
        assert config.kernel_abi == "sm90"
        assert config.tile_params == gated_config.get_tile_params(90, K, 768, 65)
    assert gated_config.get_kernel_config(80, 384, 768, 65).kernel_abi == "sm80"


@pytest.mark.parametrize("sm", _BLACKWELL_SMS)
def test_gated_sigmoid_blackwell_config_selects_sm100_kernel(sm, source_free_selection):
    for K, N in ((64, 64), (384, 768), (999, 1234)):
        for M in (65, 1500, 4096):
            abi = gated_config.get_kernel_config(sm, K, N, M).kernel_abi
            assert abi == gated_config.get_kernel_abi(sm, K, N, M)
            # SM120 and SM121 have no tensor memory; SM100 and SM103 may run the TMA source where a bucket says so.
            assert abi in (("sm80", "sm100") if sm in (100, 103) else ("sm80",))


@pytest.mark.parametrize("sm", _BLACKWELL_SMS)
def test_gated_sigmoid_blackwell_tiles_fit_with_and_without_residual(sm):
    """Every tile a Blackwell SM can resolve fits its shared memory for both residual flags."""
    cutlass = pytest.importorskip("cutlass")
    cutlass_utils = pytest.importorskip("cutlass.utils")
    source = pytest.importorskip("bionemo_ir._torch.custom_ops.gated_sigmoid._source")
    for K, N in ((64, 64), (384, 768), (999, 1234)):
        for M in (65, 1500, 4096):
            tile = gated_config.get_tile_params(sm, K, N, M)
            abi = gated_config.get_kernel_abi(sm, K, N, M)
            kernel_cls = gated_config.resolve_implementation(source.source_implementation(abi))
            limit = int(cutlass_utils.get_smem_capacity_in_bytes(f"sm_{sm}"))
            for has_residual in (False, True):
                smem = kernel_cls.dynamic_smem_bytes(
                    cutlass.BFloat16,
                    tile["m_block_size"],
                    tile["n_block_size"],
                    tile["k_block_size"],
                    tile["num_stages"],
                    has_residual=has_residual,
                )
                assert smem <= limit, f"SM{sm} K={K} N={N} M={M} residual={has_residual}: {tile} needs {smem} B"


@pytest.mark.skipif(SM_VERSION not in _BLACKWELL_SMS, reason="Blackwell-only kernel")
@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("has_bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize("has_mask", [False, True], ids=["nomask", "mask"])
@pytest.mark.parametrize("has_residual", [False, True], ids=["nores", "res"])
def test_gated_sigmoid_blackwell_matches_pytorch(
    mode, dtype, has_bias, has_mask, has_residual, no_fallback, monkeypatch
):
    """The shipped kernel of every operand flag matches PyTorch with a broadcast gate (``mult > 1``)."""
    _skip_cubin_without_an_image(mode)
    problem = _broadcast_problem(
        dtype, 128, 256, 2, 3, 333, has_bias=has_bias, has_residual=has_residual, has_mask=has_mask, seed=5
    )
    result = run_cutedsl_test_mode(
        mode, monkeypatch, GatedSigmoidCuTe, gated_cutedsl, lambda: GatedSigmoidCuTe()(**problem)
    )
    _assert_matches_pytorch(result, problem)

    if has_residual:
        # The residual add is compiled in, not skipped: the same call without it differs by exactly the residual.
        without = GatedSigmoidCuTe()(**{**problem, "residual": None})
        torch.testing.assert_close(result.float() - without.float(), problem["residual"].float(), atol=0.05, rtol=0.02)


@pytest.mark.parametrize("sm", [90, 100, 120])
def test_gated_sigmoid_disk_cache_key_follows_the_tile(sm, monkeypatch):
    """A retuned schedule must not load a kernel cached under an older one."""
    op = GatedSigmoidCuTe()
    op._sm_version = sm
    keys: list[tuple] = []
    sentinel = object()

    def load_from_cache(key):
        keys.append(key)
        return sentinel

    monkeypatch.setattr(op, "load_from_cache", load_from_cache)
    monkeypatch.setattr(GatedSigmoidCuTe, "_compiled_cache", {})
    monkeypatch.setattr(op, "_resolve_source_kernel", lambda *_a: (None, None, "sm80"))

    def run(tile):
        # A cold process each time: only the disk cache can answer.
        GatedSigmoidCuTe._compiled_cache.clear()
        resolved = gated_config.ResolvedTile("sm80", tile, 0)
        assert op._get_or_compile(torch.bfloat16, True, False, False, 128, 256, 64, resolved, False) is sentinel

    base = {"m_block_size": 64, "n_block_size": 64, "k_block_size": 32, "num_stages": 3, "atom_layout_mnk": [2, 2, 1]}
    run(base)
    run(dict(base, n_block_size=128))
    run(dict(base, atom_layout_mnk=[1, 2, 1]))
    run(dict(base))
    assert len({*keys}) == 3
    assert keys[0] == keys[3]


def test_gated_sigmoid_config_selection_is_source_free(source_free_selection):
    config = gated_config.get_kernel_config(90, 128, 256, 65)
    expected = gated_config.get_tile_params(90, 128, 256, 65)
    assert gated_config._params_from_json(config.tile_params) == expected


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(M_values=SHORT_M_VALUES, has_bias=True, dtype=torch.bfloat16),
        Scenario(M_values=SHORT_M_VALUES, has_bias=False, dtype=torch.bfloat16),
        Scenario(M_values=MEDIUM_M_VALUES, has_bias=True, dtype=torch.bfloat16),
        Scenario(M_values=MEDIUM_M_VALUES, has_bias=False, dtype=torch.bfloat16),
        Scenario(M_values=LONG_M_VALUES, has_bias=True, dtype=torch.bfloat16),
        Scenario(M_values=LONG_M_VALUES, has_bias=False, dtype=torch.bfloat16),
        Scenario(M_values=SHORT_M_VALUES, has_bias=True, dtype=torch.float16),
        Scenario(M_values=SHORT_M_VALUES, has_bias=False, dtype=torch.float16),
        Scenario(M_values=MEDIUM_M_VALUES, has_bias=True, dtype=torch.float16),
        Scenario(M_values=MEDIUM_M_VALUES, has_bias=False, dtype=torch.float16),
        Scenario(M_values=LONG_M_VALUES, has_bias=True, dtype=torch.float16),
        Scenario(M_values=LONG_M_VALUES, has_bias=False, dtype=torch.float16),
    ],
    ids=[
        "short_bias_bf16",
        "short_nobias_bf16",
        "medium_bias_bf16",
        "medium_nobias_bf16",
        "long_bias_bf16",
        "long_nobias_bf16",
        "short_bias_fp16",
        "short_nobias_fp16",
        "medium_bias_fp16",
        "medium_nobias_fp16",
        "long_bias_fp16",
        "long_nobias_fp16",
    ],
)
def test_gated_sigmoid_cute_2d(sc: Scenario):
    """Test CuTeDSL kernel with 2-D [M, K] inputs across M-ranges."""
    skip_if_no_cutedsl("gated_sigmoid")
    torch.manual_seed(42)
    cute_op = GatedSigmoidCuTe()

    W = torch.randn(sc.N_out, sc.K, dtype=sc.dtype, device="cuda")
    bias = torch.randn(sc.N_out, dtype=sc.dtype, device="cuda") if sc.has_bias else None

    for M in sc.M_values:
        s = torch.randn(M, sc.K, dtype=sc.dtype, device="cuda")
        mha = torch.randn(M, sc.N_out, dtype=sc.dtype, device="cuda")

        ref = _ref_gated_sigmoid(s, W, mha, bias)
        out = cute_op(s, W, mha, bias)

        torch.testing.assert_close(
            out, ref, atol=sc.atol, rtol=sc.rtol, msg=lambda m, M=M: f"M={M}, range={_classify_m_range(M)}: {m}"
        )


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(M_values=[33, 127, 511], has_bias=True, dtype=torch.bfloat16, N_out=64, K=64),
        Scenario(M_values=[33, 127, 511], has_bias=False, dtype=torch.bfloat16, N_out=64, K=64),
        Scenario(M_values=[33, 127, 511], has_bias=True, dtype=torch.bfloat16, N_out=256, K=128),
        Scenario(M_values=[33, 127, 511], has_bias=False, dtype=torch.bfloat16, N_out=256, K=128),
        Scenario(M_values=[33, 127, 511], has_bias=True, dtype=torch.bfloat16, N_out=768, K=384),
        Scenario(M_values=[33, 127, 511], has_bias=False, dtype=torch.bfloat16, N_out=768, K=384),
    ],
    ids=[
        "N64_K64_bias",
        "N64_K64_nobias",
        "N256_K128_bias",
        "N256_K128_nobias",
        "N768_K384_bias",
        "N768_K384_nobias",
    ],
)
def test_gated_sigmoid_cute_varied_kn(sc: Scenario):
    """Test with non-default K and N_out dimensions."""
    skip_if_no_cutedsl("gated_sigmoid")
    torch.manual_seed(42)
    cute_op = GatedSigmoidCuTe()

    W = torch.randn(sc.N_out, sc.K, dtype=sc.dtype, device="cuda")
    bias = torch.randn(sc.N_out, dtype=sc.dtype, device="cuda") if sc.has_bias else None

    for M in sc.M_values:
        s = torch.randn(M, sc.K, dtype=sc.dtype, device="cuda")
        mha = torch.randn(M, sc.N_out, dtype=sc.dtype, device="cuda")

        ref = _ref_gated_sigmoid(s, W, mha, bias)
        out = cute_op(s, W, mha, bias)

        torch.testing.assert_close(
            out, ref, atol=sc.atol, rtol=sc.rtol, msg=lambda m, M=M: f"M={M}, K={sc.K}, N={sc.N_out}: {m}"
        )


@pytest.mark.parametrize("has_bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize(
    "shape",
    [
        (4, 255, 128),  # 3-D, odd seq
        (2, 3, 171, 128),  # 4-D, odd seq (M = 2*3*171 = 1026 → medium)
        (1, 1, 65, 128),  # 4-D, M=65 (short, odd)
        (2, 1025, 128),  # 3-D, M=2050 (long, even)
    ],
    ids=["3d_M1020", "4d_M1026", "4d_M65", "3d_M2050"],
)
def test_gated_sigmoid_cute_batched(shape, has_bias):
    """Test with batched (3-D/4-D) inputs; kernel flattens leading dims."""
    skip_if_no_cutedsl("gated_sigmoid")
    torch.manual_seed(42)
    dtype = torch.bfloat16
    K = shape[-1]
    N_out = 128

    cute_op = GatedSigmoidCuTe()
    W = torch.randn(N_out, K, dtype=dtype, device="cuda")
    bias = torch.randn(N_out, dtype=dtype, device="cuda") if has_bias else None

    s = torch.randn(*shape, dtype=dtype, device="cuda")
    mha = torch.randn(*shape[:-1], N_out, dtype=dtype, device="cuda")

    ref = _ref_gated_sigmoid(s, W, mha, bias)
    out = cute_op(s, W, mha, bias)

    torch.testing.assert_close(out, ref, atol=0.05, rtol=1e-2)


# ---------------------------------------------------------------------------
# Broadcasting tests (gate `s` shared across a multiplicity dim of `mha_out`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize(
    "B,S,I,K,N_out",
    [
        (2, 4, 130, 128, 128),  # I not a multiple of any tile dim
        (1, 3, 65, 128, 128),  # single batch, odd inner
        (3, 1, 200, 128, 128),  # mult==1 via a size-1 broadcast dim
        (2, 5, 1, 128, 128),  # inner==1
        (4, 2, 33, 128, 128),  # many small batches, odd inner
        (1, 5, 76, 384, 768),  # OF3 token diffusion: K != N
        (1, 5, 76, 768, 768),  # OF3 token diffusion: K == N
    ],
    ids=["B2S4I130", "B1S3I65", "B3S1I200", "B2S5I1", "B4S2I33", "B1S5I76_K384_N768", "B1S5I76_K768_N768"],
)
def test_gated_sigmoid_cute_broadcast(B, S, I, K, N_out, has_bias):
    """Gate `s` is [B, 1, I, K], shared across S samples of mha_out [B, S, I, N]."""
    skip_if_no_cutedsl("gated_sigmoid")
    torch.manual_seed(42)
    dtype = torch.bfloat16

    cute_op = GatedSigmoidCuTe()
    W = torch.randn(N_out, K, dtype=dtype, device="cuda")
    bias = torch.randn(N_out, dtype=dtype, device="cuda") if has_bias else None

    s = torch.randn(B, 1, I, K, dtype=dtype, device="cuda")
    mha = torch.randn(B, S, I, N_out, dtype=dtype, device="cuda")

    ref = _ref_gated_sigmoid(s, W, mha, bias)
    out = cute_op(s, W, mha, bias)

    assert out.shape == mha.shape
    torch.testing.assert_close(
        out, ref, atol=0.05, rtol=1e-2, msg=lambda m: f"B={B}, S={S}, I={I}, K={K}, N={N_out}, bias={has_bias}: {m}"
    )


@pytest.mark.parametrize("has_bias", [True, False], ids=["bias", "nobias"])
def test_gated_sigmoid_cute_broadcast_3d(has_bias):
    """Broadcast with the multiplicity as the leading dim: s [1, I, K]."""
    skip_if_no_cutedsl("gated_sigmoid")
    torch.manual_seed(0)
    dtype = torch.bfloat16
    K, N_out, S, I = 256, 128, 6, 257

    cute_op = GatedSigmoidCuTe()
    W = torch.randn(N_out, K, dtype=dtype, device="cuda")
    bias = torch.randn(N_out, dtype=dtype, device="cuda") if has_bias else None

    s = torch.randn(1, I, K, dtype=dtype, device="cuda")
    mha = torch.randn(S, I, N_out, dtype=dtype, device="cuda")

    ref = _ref_gated_sigmoid(s, W, mha, bias)
    out = cute_op(s, W, mha, bias)

    assert out.shape == mha.shape
    torch.testing.assert_close(out, ref, atol=0.05, rtol=1e-2)


def test_gated_sigmoid_broadcast_matches_nonbroadcast():
    """Broadcasting S samples == expanding `s` and running the dense kernel."""
    skip_if_no_cutedsl("gated_sigmoid")
    torch.manual_seed(123)
    dtype = torch.bfloat16
    K, N_out, B, S, I = 128, 128, 2, 4, 70

    cute_op = GatedSigmoidCuTe()
    W = torch.randn(N_out, K, dtype=dtype, device="cuda")
    bias = torch.randn(N_out, dtype=dtype, device="cuda")

    s = torch.randn(B, 1, I, K, dtype=dtype, device="cuda")
    mha = torch.randn(B, S, I, N_out, dtype=dtype, device="cuda")

    out_bcast = cute_op(s, W, mha, bias)
    # Materialize the broadcast and run the dense (mult==1) path.
    s_dense = s.expand(B, S, I, K).contiguous()
    out_dense = cute_op(s_dense, W, mha, bias)

    torch.testing.assert_close(out_bcast, out_dense, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Vanilla fallback tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize("M", [1, 33, 127, 512, 1025, 2049])
def test_vanilla_gated_sigmoid(M, has_bias):
    """Test vanilla PyTorch fallback against fp32 reference."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    K, N_out = 128, 128

    W = torch.randn(N_out, K, dtype=dtype, device="cuda")
    bias = torch.randn(N_out, dtype=dtype, device="cuda") if has_bias else None
    s = torch.randn(M, K, dtype=dtype, device="cuda")
    mha = torch.randn(M, N_out, dtype=dtype, device="cuda")

    ref = _ref_gated_sigmoid(s, W, mha, bias)
    out = _invoke_vanilla_gated_sigmoid(s, W, mha, bias)

    torch.testing.assert_close(out, ref, atol=0.05, rtol=1e-2)


# ---------------------------------------------------------------------------
# get_gated_sigmoid_op selector test
# ---------------------------------------------------------------------------


def test_get_gated_sigmoid_op_selector():
    """Verify that get_gated_sigmoid_op returns the correct backend."""
    op_bf16 = get_gated_sigmoid_op(torch.bfloat16, N=128, K=128)
    op_fp16 = get_gated_sigmoid_op(torch.float16, N=128, K=128)
    op_fp32 = get_gated_sigmoid_op(torch.float32, N=128, K=128)

    if SM_VERSION in _SUPPORTED_SMS:
        assert isinstance(op_bf16, GatedSigmoidCuTe)
        assert isinstance(op_fp16, GatedSigmoidCuTe)
    else:
        assert op_bf16 is _invoke_vanilla_gated_sigmoid
        assert op_fp16 is _invoke_vanilla_gated_sigmoid

    assert op_fp32 is _invoke_vanilla_gated_sigmoid
    assert get_gated_sigmoid_op(torch.bfloat16, N=128, K=64) is _invoke_vanilla_gated_sigmoid
    # A 768-wide output gated from a 256-wide input ships tuned tiles.
    assert get_gated_sigmoid_op(torch.bfloat16, N=768, K=256) is op_bf16


def test_get_gated_sigmoid_op_runs():
    """End-to-end: get_gated_sigmoid_op returns a callable that produces correct results."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    K, N_out, M = 128, 128, 513

    op = get_gated_sigmoid_op(dtype, N=N_out, K=K)
    W = torch.randn(N_out, K, dtype=dtype, device="cuda")
    bias = torch.randn(N_out, dtype=dtype, device="cuda")
    s = torch.randn(M, K, dtype=dtype, device="cuda")
    mha = torch.randn(M, N_out, dtype=dtype, device="cuda")

    ref = _ref_gated_sigmoid(s, W, mha, bias)
    out = op(s, W, mha, bias)

    torch.testing.assert_close(out, ref, atol=0.05, rtol=1e-2)


def test_sm80_cubin_adapter_selects_atom_layout():
    """The Python adapter must pass every tile-selection axis to C++."""
    library = require_cubin_library()
    launcher = library.gated_sigmoid
    tile_params = gated_config.get_tile_params(80, K=128, N=128, M=1)

    executable = GatedSigmoidCubinExecutable(
        library,
        launcher,
        target_sm=80,
        m_bucket=0,
        dtype=torch.bfloat16,
        has_bias=True,
        has_residual=False,
        has_mask=False,
        kernel_abi="sm80",
        tile_params=tile_params,
    )

    assert executable._config.spec.target_sm == 80
    assert tuple(executable._config.spec.atom_layout_mnk) == tuple(tile_params["atom_layout_mnk"])


# ---------------------------------------------------------------------------
# M-range classification unit test
# ---------------------------------------------------------------------------


def test_classify_m_range():
    assert _classify_m_range(1) == "tiny"
    assert _classify_m_range(256) == "tiny"
    assert _classify_m_range(257) == "short"
    assert _classify_m_range(768) == "short"
    assert _classify_m_range(769) == "short_hi"
    assert _classify_m_range(1024) == "short_hi"
    assert _classify_m_range(1025) == "medium"
    assert _classify_m_range(2048) == "medium"
    assert _classify_m_range(2049) == "long"
    assert _classify_m_range(10000) == "long"


# ---------------------------------------------------------------------------
# In-place gated sigmoid tests (output aliases mha_out)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize("M", [33, 127, 390, 1025, 2049])
def test_gated_sigmoid_cute_inplace_2d(M, has_bias):
    """Verify in-place (output=mha_out) produces the same result as out-of-place."""
    skip_if_no_cutedsl("gated_sigmoid")
    torch.manual_seed(42)
    K, N_out = 768, 768
    dtype = torch.bfloat16

    cute_op = GatedSigmoidCuTe()
    W = torch.randn(N_out, K, dtype=dtype, device="cuda")
    bias = torch.randn(N_out, dtype=dtype, device="cuda") if has_bias else None

    s = torch.randn(M, K, dtype=dtype, device="cuda")
    mha = torch.randn(M, N_out, dtype=dtype, device="cuda")
    mha_copy = mha.clone()

    ref = cute_op(s, W, mha, bias)
    out = cute_op(s, W, mha_copy, bias, output=mha_copy)

    assert out.data_ptr() == mha_copy.data_ptr(), "in-place should reuse mha_out memory"
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0, msg=lambda m: f"M={M}, in-place != out-of-place: {m}")


@pytest.mark.parametrize("has_bias", [True, False], ids=["bias", "nobias"])
@pytest.mark.parametrize(
    "shape_s,shape_mha",
    [
        ((4, 390, 768), (4, 390, 768)),
        ((2, 3, 171, 768), (2, 3, 171, 768)),
    ],
    ids=["3d_DiT", "4d_batched"],
)
def test_gated_sigmoid_cute_inplace_batched(shape_s, shape_mha, has_bias, monkeypatch):
    """In-place with a batched gate and flat output must stay on the fused path."""
    skip_if_no_cutedsl("gated_sigmoid")
    torch.manual_seed(42)
    K = shape_s[-1]
    N_out = shape_mha[-1]
    dtype = torch.bfloat16

    cute_op = GatedSigmoidCuTe()
    W = torch.randn(N_out, K, dtype=dtype, device="cuda")
    bias = torch.randn(N_out, dtype=dtype, device="cuda") if has_bias else None

    s = torch.randn(*shape_s, dtype=dtype, device="cuda")
    mha = torch.randn(*shape_mha, dtype=dtype, device="cuda")

    ref = _ref_gated_sigmoid(s, W, mha, bias)

    def reject_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected vanilla fallback")

    monkeypatch.setattr(gated_cutedsl, "_invoke_vanilla_gated_sigmoid", reject_fallback)
    mha_flat = mha.reshape(-1, N_out)
    out = cute_op(s, W, mha_flat, bias, output=mha_flat)

    assert out.data_ptr() == mha_flat.data_ptr(), "in-place should reuse mha_out memory"
    out_restored = out.view(shape_mha)
    torch.testing.assert_close(out_restored, ref, atol=0.07, rtol=1e-2, msg=lambda m: f"shapes={shape_s}: {m}")


@pytest.mark.parametrize("noncontiguous", ["mha", "output"])
def test_gated_sigmoid_falls_back_for_unsupported_layout(noncontiguous, monkeypatch):
    torch.manual_seed(61)
    rows, K, N_out = 6, 64, 64
    dtype = torch.bfloat16
    s = torch.randn(rows, K, dtype=dtype, device="cuda")
    weight = torch.randn(N_out, K, dtype=dtype, device="cuda")
    contiguous_mha = torch.randn(rows, N_out, dtype=dtype, device="cuda")
    mha = contiguous_mha if noncontiguous == "output" else contiguous_mha.t().contiguous().t()
    output = None
    if noncontiguous == "output":
        output = torch.empty(N_out, rows, dtype=dtype, device="cuda").t()

    calls = 0
    original_fallback = _invoke_vanilla_gated_sigmoid

    def record_fallback(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_fallback(*args, **kwargs)

    monkeypatch.setattr(gated_cutedsl, "_invoke_vanilla_gated_sigmoid", record_fallback)
    result = GatedSigmoidCuTe()(s, weight, mha, output=output)
    expected = original_fallback(s, weight, mha)

    assert calls == 1
    if output is not None:
        assert result.data_ptr() == output.data_ptr()
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


# --- per-bucket kernel selection: the persistent TMA source ----------------------------------------------------------

_MMA_SYNC_TILE = {
    "m_block_size": 64,
    "n_block_size": 128,
    "k_block_size": 64,
    "num_stages": 3,
    "atom_layout_mnk": [2, 2, 1],
    "raster_factor": 1,
}
_TMA_TILE = {
    "kernel_abi": "sm100",
    "m_block_size": 128,
    "n_block_size": 64,
    "k_block_size": 64,
    "num_stages": 6,
    "atom_layout_mnk": [1, 1, 1],
    "raster_factor": 1,
}


def _drop_gated_sigmoid_mode_caches() -> None:
    """Drop executables compiled against an earlier tile so a retuned config cannot reuse them."""
    stale = [key for key in _CUTEDSL_MODE_CACHES if key[0] is GatedSigmoidCuTe]
    for key in stale:
        del _CUTEDSL_MODE_CACHES[key]


@pytest.fixture
def tuned_configs(tmp_path, request, monkeypatch):
    """Point the config loader at ``tmp_path``; ``write(name, bundle)`` adds a file there."""
    _drop_gated_sigmoid_mode_caches()
    monkeypatch.setenv("BIOIR_TUNED_CONFIG_FOLDER", str(tmp_path))
    load_kernel_configs.cache_clear()
    request.addfinalizer(_drop_gated_sigmoid_mode_caches)
    request.addfinalizer(load_kernel_configs.cache_clear)

    def write(name: str, bundle: dict) -> None:
        (tmp_path / name).write_text(json.dumps(bundle))

    return write


def test_gated_sigmoid_short_bucket_inherits_the_tiny_entry_of_an_older_file(tuned_configs):
    """A file without a ``257`` key runs its ``0`` entry for every ``M <= 1024``, and CUBINs index that same key."""
    tuned_configs(
        "K384_N768_sm100.json",
        {"kernel_abi": "sm80", "configs": {"0": _MMA_SYNC_TILE, "1025": _MMA_SYNC_TILE, "2049": _TMA_TILE}},
    )
    assert [gated_config.get_m_bucket(M, 100, 384, 768) for M in (64, 256, 257, 1024, 1025, 2049)] == [
        0,
        0,
        0,
        0,
        1025,
        2049,
    ]
    assert gated_config.get_tile_params(100, 384, 768, 300) == gated_config.get_tile_params(100, 384, 768, 64)


def test_gated_sigmoid_tiny_and_short_buckets_can_differ(tuned_configs):
    tuned_configs(
        "K384_N768_sm100.json",
        {
            "kernel_abi": "sm80",
            "configs": {"0": _MMA_SYNC_TILE, "257": _TMA_TILE, "1025": _TMA_TILE, "2049": _TMA_TILE},
        },
    )
    assert [gated_config.get_kernel_abi(100, 384, 768, M) for M in (64, 256, 257, 1024)] == [
        "sm80",
        "sm80",
        "sm100",
        "sm100",
    ]
    assert [gated_config.get_m_bucket(M, 100, 384, 768) for M in (256, 257)] == [0, 257]


def test_gated_sigmoid_short_hi_bucket_can_differ_from_short_and_inherits_it_otherwise(tuned_configs):
    tuned_configs(
        "K384_N768_sm100.json",
        {
            "kernel_abi": "sm80",
            "configs": {"0": _MMA_SYNC_TILE, "257": _TMA_TILE, "769": _MMA_SYNC_TILE, "1025": _TMA_TILE},
        },
    )
    tuned_configs(
        "K256_N768_sm100.json",
        {"kernel_abi": "sm80", "configs": {"0": _MMA_SYNC_TILE, "257": _TMA_TILE, "1025": _TMA_TILE}},
    )
    assert [gated_config.get_m_bucket(M, 100, 384, 768) for M in (768, 769, 1024, 1025)] == [257, 769, 769, 1025]
    assert [gated_config.get_kernel_abi(100, 384, 768, M) for M in (768, 769)] == ["sm100", "sm80"]
    # A file without a ``769`` key keeps running its ``257`` entry up to 1024 rows.
    assert [gated_config.get_m_bucket(M, 100, 256, 768) for M in (768, 769, 1024)] == [257, 257, 257]


def test_gated_sigmoid_bucket_names_its_own_kernel(tuned_configs, source_free_selection):
    """The long bucket runs the TMA kernel while the others keep the file's mma.sync kernel."""

    tuned_configs(
        "K384_N768_sm100.json",
        {"kernel_abi": "sm80", "configs": {"0": _MMA_SYNC_TILE, "1025": _MMA_SYNC_TILE, "2049": _TMA_TILE}},
    )
    assert gated_config.get_kernel_abi(100, 384, 768) == "sm80"
    assert [gated_config.get_kernel_abi(100, 384, 768, M) for M in (65, 1024, 1025, 2048, 2049, 8192)] == [
        "sm80",
        "sm80",
        "sm80",
        "sm80",
        "sm100",
        "sm100",
    ]

    config = gated_config.get_kernel_config(100, 384, 768, 4096)
    assert config.kernel_abi == "sm100"
    expected = {key: value for key, value in _TMA_TILE.items() if key != "kernel_abi"}
    assert config.tile_params == expected
    # The tile the registry and the kernel factory see never carries the ABI selector.
    assert gated_config.get_tile_params(100, 384, 768, 4096) == {**expected, "atom_layout_mnk": (1, 1, 1)}
    assert gated_config.get_kernel_config(100, 384, 768, 1500).kernel_abi == "sm80"
    assert gated_config.get_kernel_config(100, 384, 768, 1500).tile_params["atom_layout_mnk"] == [2, 2, 1]


def test_gated_sigmoid_file_level_tma_leaves_unlisted_buckets_to_the_default_source(tuned_configs):
    tuned_configs("K128_N256_sm103.json", {"kernel_abi": "sm100", "configs": {"0": _TMA_TILE}})
    assert gated_config.get_kernel_abi(103, 128, 256, 64) == "sm100"
    # A bucket the file does not list resolves to the heuristic tile, which belongs to the SM's default source.
    assert gated_config.get_kernel_abi(103, 128, 256, 4096) == "sm80"
    assert gated_config.get_kernel_config(103, 128, 256, 4096).tile_params["atom_layout_mnk"] == [2, 2, 1]


def test_gated_sigmoid_unknown_bucket_kernel_is_refused(tuned_configs):
    bucket = {**_MMA_SYNC_TILE, "kernel_abi": "x"}
    tuned_configs("K384_N768_sm100.json", {"kernel_abi": "sm80", "configs": {"0": bucket}})
    with pytest.raises(ValueError, match="Unsupported"):
        gated_config.get_kernel_abi(100, 384, 768, 65)


_TCGEN05_ONLY = pytest.mark.skipif(
    SM_VERSION not in gated_config._TCGEN05_SMS, reason="tcgen05 needs tensor memory: SM100 or SM103"
)
_FROM_SOURCE = pytest.mark.skipif("source" not in _CUTEDSL_MODES, reason="compiles the kernel from source")


# --- the persistent TMA source: file-level selection and numerics --------------------------------------------------


def test_gated_sigmoid_file_level_tma_runs_every_bucket_it_lists(tuned_configs, source_free_selection):
    short = {**_TMA_TILE, "n_block_size": 32, "k_block_size": 128, "num_stages": 4}
    tuned_configs(
        "K768_N768_sm100.json",
        {"kernel_abi": "sm100", "configs": {"0": short, "1025": _TMA_TILE, "2049": _TMA_TILE}},
    )
    assert [gated_config.get_kernel_abi(100, 768, 768, M) for M in (65, 1024, 1025, 2048, 2049, 8192)] == ["sm100"] * 6
    config = gated_config.get_kernel_config(100, 768, 768, 512)
    assert config.kernel_abi == "sm100"
    assert config.tile_params["n_block_size"] == 32 and config.tile_params["k_block_size"] == 128
    assert gated_config.get_kernel_config(100, 768, 768, 4096).tile_params["n_block_size"] == 64


@pytest.mark.parametrize("sm", [120, 121])
def test_gated_sigmoid_tma_bucket_is_refused_without_tensor_memory(tuned_configs, sm):
    tuned_configs(f"K768_N768_sm{sm}.json", {"kernel_abi": "sm80", "configs": {"0": _TMA_TILE}})
    with pytest.raises(ValueError, match="runs on SM"):
        gated_config.get_kernel_abi(sm, 768, 768, 65)


def _run_tma(problem: dict, monkeypatch) -> None:
    """Run ``problem`` from source, check that every call took the TMA kernel, and compare with PyTorch."""
    op = GatedSigmoidCuTe()
    result = run_cutedsl_test_mode("source", monkeypatch, GatedSigmoidCuTe, gated_cutedsl, lambda: op(**problem))
    assert {abi for _, abi in op._launches.values()} == {"sm100"}
    _assert_matches_pytorch(result, problem)


def _check_tma_against_pytorch(
    tile, dtype, has_bias, has_mask, has_residual, K, N, inner, outer, mult, tuned_configs, monkeypatch
):
    """Run the op with every bucket of ``K{K}_N{N}`` on a TMA ``tile`` and compare with PyTorch."""
    n, k, stages = tile
    bucket = {**_TMA_TILE, "n_block_size": n, "k_block_size": k, "num_stages": stages}
    tuned_configs(
        f"K{K}_N{N}_sm{SM_VERSION}.json",
        {"kernel_abi": "sm100", "configs": {"0": bucket, "1025": bucket, "2049": bucket}},
    )
    problem = _broadcast_problem(
        dtype, K, N, outer, mult, inner, has_bias=has_bias, has_residual=has_residual, has_mask=has_mask, seed=5
    )
    _run_tma(problem, monkeypatch)


@_TCGEN05_ONLY
@_FROM_SOURCE
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("has_bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize("has_mask", [False, True], ids=["nomask", "mask"])
@pytest.mark.parametrize("has_residual", [False, True], ids=["nores", "res"])
def test_gated_sigmoid_tma_matches_pytorch(
    dtype, has_bias, has_mask, has_residual, tuned_configs, no_fallback, monkeypatch
):
    """Every operand flag, with a gate row count that is not a multiple of the 128-row tile and ``mult > 1``."""
    _check_tma_against_pytorch(
        (64, 64, 6), dtype, has_bias, has_mask, has_residual, 128, 256, 333, 2, 3, tuned_configs, monkeypatch
    )


@_TCGEN05_ONLY
@_FROM_SOURCE
@pytest.mark.parametrize("tile", [(32, 128, 4), (64, 128, 3), (128, 64, 4), (32, 32, 6)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_gated_sigmoid_tma_tile_shapes_match_pytorch(tile, dtype, tuned_configs, no_fallback, monkeypatch):
    """Each tile width and K block, including one of two swizzle atoms, and a K that is not a multiple of the block."""
    _check_tma_against_pytorch(tile, dtype, True, True, True, 192, 200, 65, 3, 2, tuned_configs, monkeypatch)


@_TCGEN05_ONLY
@_FROM_SOURCE
def test_gated_sigmoid_tma_persistent_grid_covers_more_tiles_than_sms(tuned_configs, no_fallback, monkeypatch):
    """More work items than SMs: each CTA loops, and the accumulator stages alternate."""
    _check_tma_against_pytorch(
        (64, 64, 6), torch.bfloat16, True, False, True, 256, 768, 1024, 2, 1, tuned_configs, monkeypatch
    )


@_TCGEN05_ONLY
@_FROM_SOURCE
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("has_residual", [False, True], ids=["nores", "res"])
@pytest.mark.parametrize(
    "rows,mult", [(512, 1), (2048, 1), (4096, 1), (1024, 3)], ids=["short", "medium", "long", "mult3"]
)
def test_gated_sigmoid_shipped_768x768_config_runs_the_tma_kernel(
    rows, mult, has_residual, dtype, no_fallback, monkeypatch
):
    """The 768x768 files ship the TMA kernel in every bucket: the op runs it and matches PyTorch."""
    _run_tma(_broadcast_problem(dtype, 768, 768, 1, mult, rows, has_residual=has_residual, seed=9), monkeypatch)
