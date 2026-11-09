# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
from dataclasses import dataclass

import pytest
import torch
from test_utils.boltz.create_and_load_weights import (
    create_outer_product_mean_weights,
    load_outer_product_mean_weights_torch,
)
from test_utils.boltz.ref_layers import RefOuterProductMean

from bionemo_ir._torch.custom_ops import outer_product_mean as opm_ops
from bionemo_ir._torch.custom_ops.outer_product_mean import (
    OuterProductMeanCuTe,
    _select_opm_config_bucket,
    config_identity,
    get_outer_product_mean_op,
    select_opm_config,
)
from bionemo_ir._torch.custom_ops.outer_product_mean import _cubin as opm_cubin
from bionemo_ir._torch.custom_ops.outer_product_mean import cutedsl as opm_cutedsl
from bionemo_ir._torch.custom_ops.outer_product_mean._config import (
    _OPM_CONFIGS_DIR,
    Sm90KernelConfig,
    Sm100KernelConfig,
    sm100_default_config,
    sm100_fit,
)
from bionemo_ir._torch.custom_ops.outer_product_mean.ops import _invoke_triton_opm, _invoke_vanilla_opm
from bionemo_ir._torch.layers.outer_product_mean import OuterProductMean
from bionemo_ir._torch.utils import ChunkPolicy
from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelLibraryUnavailable,
    CuTeDSLKernelVariantUnavailable,
)
from bionemo_ir.dsl_kernels.cute_cache import FORCE_CUBIN_ENV
from bionemo_ir.utils import str_dtype_to_torch
from tests._torch import (
    SM_VERSION,
    cutedsl_test_modes,
    require_cubin_family,
    require_cubin_library,
    run_cutedsl_test_mode,
    skip_if_no_cutedsl,
    source_module_available,
)

_CUTEDSL_SM = (80, 86, 89, 90, 100, 103)
_SOURCE_MODULE = "bionemo_ir._torch.custom_ops.outer_product_mean._source"
_CUTEDSL_MODES = cutedsl_test_modes(_SOURCE_MODULE)


@dataclass(kw_only=True, frozen=True)
class Scenario:
    torch_dtype: str = "float32"
    # Output token-rows per chunk via a registry-style ChunkPolicy. Row-chunking is numerically
    # identical, so the chunked output must still match the ref.
    policy_chunk: int | None = None
    n_seq: int = 32
    n_res: int = 64
    norm_before_output: bool = True


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(),
        # fp16 / bf16 single-GPU with c_hidden==32, c_out==128 routes through the
        # SM80 fused CuTe custom op; fp32 / chunked stay on the eager path.
        Scenario(torch_dtype="bfloat16"),
        Scenario(torch_dtype="float16"),
        # Larger N exercises the kernel's hierarchical N-then-S config selection.
        Scenario(torch_dtype="bfloat16", n_res=128),
        # norm-after-output epilogue (the kernel divides the proj-o accumulator).
        Scenario(torch_dtype="bfloat16", norm_before_output=False),
        # fp32 cannot use the fused kernel, so these exercise the auto-chunk fallback.
        Scenario(policy_chunk=16),
        Scenario(policy_chunk=12, n_res=65),
    ],
    ids=[
        "float32",
        "bfloat16",
        "float16",
        "bfloat16_n128",
        "bfloat16_nb_false",
        "policy_chunk16",
        "policy_chunk_partial",
    ],
)
def test_outer_product_mean(sc: Scenario):
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device("cuda")

    ref_m = RefOuterProductMean.load_weights()
    ref_m = ref_m.to(device)
    # The reference loads norm_before_output=True; override to also cover the
    # norm-after-output path (weights are identical, only the division moves).
    ref_m.norm_before_output = sc.norm_before_output

    weights_and_biases = create_outer_product_mean_weights(from_ref=ref_m)

    # A low min_size makes the policy trip at the test's token count, so `policy_chunk` scenarios
    # exercise the registry-driven output-row chunking.
    chunk_policy = ChunkPolicy(chunk_size=sc.policy_chunk, min_size=1) if sc.policy_chunk is not None else None
    outer_product_mean = OuterProductMean(
        c_in=ref_m.c_in,
        c_hidden=ref_m.c_hidden,
        c_out=ref_m.c_out,
        norm_before_output=sc.norm_before_output,
        dtype=dtype,
        chunk_policy=chunk_policy,
    )
    load_outer_product_mean_weights_torch(outer_product_mean, weights_and_biases, dtype=dtype)
    outer_product_mean.to(device)

    # On SM80+ the half-precision, non-chunked path must resolve to a fused
    # custom op (guards against a silent fall-back to eager).
    if dtype in (torch.float16, torch.bfloat16) and SM_VERSION >= 80:
        assert outer_product_mean._opm_eligible
        assert (
            get_outer_product_mean_op(
                dtype,
                C=outer_product_mean.c_hidden,
                D=outer_product_mean.c_hidden,
                C_z=outer_product_mean.c_out,
            )
            is not _invoke_vanilla_opm
        )

    m = torch.randn(bs, sc.n_seq, sc.n_res, ref_m.c_in, dtype=torch.float32).cuda()
    mask = torch.randint(0, 2, (bs, sc.n_seq, sc.n_res), dtype=torch.float32).to(device)

    ref_output_float = ref_m(m, mask)
    m = m.to(dtype)
    mask = mask.to(dtype)
    ref_m = ref_m.to(dtype)

    ref_output = ref_m(m, mask)
    output = outer_product_mean.forward(m, mask)

    assert ref_output.shape == output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(ref_output, output, atol=1e-3, rtol=1e-4)
    else:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output.float() - ref_output_float))

        assert abs(diff0_max - diff1_max) / torch.min(diff0_max, diff1_max) <= 0.5
        assert abs(diff0_mean - diff1_mean) <= 0.2


@pytest.mark.parametrize("rows", [8, 16, 40, 7], ids=["r8", "r16", "r40_all", "r7_partial"])
def test_outer_product_mean_chunk_matches_dense(rows: int):
    """Output token-row chunking (registry policy) matches the dense path.

    Same instance dense vs chunked (no golden weights needed); each output row ``i`` depends only on
    ``a[:, :, i]``, so slicing the output token dim and concatenating is numerically identical.
    """
    torch.manual_seed(0)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")

    opm = OuterProductMean(c_in=32, c_hidden=8, c_out=16, dtype=torch.float32).to(device)
    opm.eval()
    # Constructed weights are zero-initialized (production loads them); give them real values so
    # the dense-vs-chunked comparison is meaningful rather than 0 == 0.
    opm.load_state_dict({name: torch.randn_like(value) * 0.1 for name, value in opm.state_dict().items()})
    m = torch.randn(1, 6, 40, 32, device=device)  # N=40 output rows
    mask = torch.randint(0, 2, (1, 6, 40), dtype=torch.float32, device=device)

    # Registry default (memory-scaled min_size) not tripped at N=40 -> dense.
    dense = opm(m, mask)

    # Registry-style policy chunking over output token-rows (rows=7 hits a partial tail).
    opm.chunk_policy = ChunkPolicy(chunk_size=rows, min_size=1)
    policy = opm(m, mask)

    torch.testing.assert_close(policy, dense, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Fused SM80 CuTe custom op: kernel path vs the eager path (self-contained --
# random weights, no downloaded checkpoints), so CI exercises the kernel even
# without hub access. Confirms the dispatch, the proj_o weight layout, and the
# norm-before / norm-after epilogues all match the reference eager math within
# bf16/fp16 tolerance.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "bias"])
@pytest.mark.parametrize("norm_before", [False, True], ids=["norm_after", "norm_before"])
@pytest.mark.parametrize("C_z", [128, 256], ids=["cz128", "cz256"])
def test_outer_product_mean_source_and_cubin(mode, dtype, has_bias, norm_before, C_z, monkeypatch):
    _require_cutedsl_mode(mode)
    torch.manual_seed(7)
    B, S, I, J, C, D = 1, 31, 17, 19, 32, 32
    a = torch.randn(B, S, I, C, device="cuda", dtype=dtype).mul_(0.2)
    b = torch.randn(B, S, J, D, device="cuda", dtype=dtype).mul_(0.2)
    num_mask = torch.randint(1, S + 1, (B, I, J), device="cuda", dtype=torch.int32).float()
    weight = torch.randn(C_z, C * D, device="cuda", dtype=dtype).mul_(0.1)
    bias = torch.randn(C_z, device="cuda", dtype=dtype).mul_(0.1) if has_bias else None

    result = run_cutedsl_test_mode(
        mode,
        monkeypatch,
        OuterProductMeanCuTe,
        opm_cutedsl,
        lambda: OuterProductMeanCuTe()(a, b, num_mask, weight, bias, norm_before),
    )
    reference = _invoke_vanilla_opm(a, b, num_mask, weight, bias, norm_before)
    torch.testing.assert_close(result, reference, atol=0.08, rtol=0.02)


# (dtype, B, S, I, J, C_z, has_bias, norm_before): every axis off its tile multiple and I != J,
# with sqrt(I * J) near each tuned N bucket so every shipped tile -- paired tiles, the C split,
# and j-strip rasterization included -- runs partial tiles. Paired tiles see an odd tile count.
_UNALIGNED_CASES = [
    (torch.bfloat16, 1, 255, 117, 133, 128, True, True),
    (torch.bfloat16, 2, 511, 245, 259, 128, False, True),
    (torch.bfloat16, 1, 500, 515, 513, 128, True, False),
    (torch.bfloat16, 1, 1000, 1001, 1003, 128, True, True),
    (torch.bfloat16, 1, 999, 1301, 1299, 128, False, False),
    (torch.bfloat16, 1, 599, 821, 827, 256, True, True),
    (torch.bfloat16, 2, 33, 260, 251, 256, True, False),
    (torch.float16, 1, 300, 517, 515, 128, True, True),
    (torch.float16, 1, 97, 263, 257, 256, False, True),
]


def _unaligned_operands(dtype, B, S, I, J, C_z, has_bias):
    torch.manual_seed(I + J + S)
    a = torch.randn(B, S, I, 32, device="cuda", dtype=dtype)
    b = torch.randn(B, S, J, 32, device="cuda", dtype=dtype)
    # Counts near S keep every output on one scale, so the tolerance catches a wrong tile.
    num_mask = torch.randint(S // 2 + 1, S + 2, (B, I, J), device="cuda").float()
    weight = torch.randn(C_z, 1024, device="cuda", dtype=dtype).mul_(0.05)
    bias = torch.randn(C_z, device="cuda", dtype=dtype).mul_(0.1) if has_bias else None
    return a, b, num_mask, weight, bias


def _assert_matches_fp32_reference(result, a, b, num_mask, weight, bias, norm_before):
    # Row blocks bound the reference's FP32 [B, rows, J, 1024] intermediate.
    reference = torch.cat(
        [
            _invoke_vanilla_opm(
                a[:, :, i : i + 256].float(), b.float(), num_mask[:, i : i + 256], weight.float(), bias, norm_before
            )
            for i in range(0, a.shape[2], 256)
        ],
        dim=1,
    )
    assert result.shape == reference.shape and result.dtype == a.dtype
    torch.testing.assert_close(result.float(), reference, rtol=0.02, atol=0.02 * reference.abs().median().item())


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize(
    "dtype,B,S,I,J,C_z,has_bias,norm_before",
    _UNALIGNED_CASES,
    ids=[f"{str(c[0]).removeprefix('torch.')}-B{c[1]}-S{c[2]}-I{c[3]}-J{c[4]}-cz{c[5]}" for c in _UNALIGNED_CASES],
)
def test_outer_product_mean_cute_unaligned(mode, dtype, B, S, I, J, C_z, has_bias, norm_before, monkeypatch):
    _require_cutedsl_mode(mode)
    a, b, num_mask, weight, bias = _unaligned_operands(dtype, B, S, I, J, C_z, has_bias)
    result = run_cutedsl_test_mode(
        mode,
        monkeypatch,
        OuterProductMeanCuTe,
        opm_cutedsl,
        lambda: OuterProductMeanCuTe()(a, b, num_mask, weight, bias, norm_before),
    )
    _assert_matches_fp32_reference(result, a, b, num_mask, weight, bias, norm_before)


@pytest.mark.skipif(SM_VERSION not in (100, 103), reason="the SM100 kernel runs on SM100 and SM103")
@pytest.mark.parametrize("norm_before", [False, True], ids=["norm_after", "norm_before"])
@pytest.mark.parametrize("C_z", [128, 256], ids=["cz128", "cz256"])
def test_outer_product_mean_sm100_empty_msa(C_z, norm_before):
    """An empty MSA (S = 0) leaves only bias and normalization; the SM100 kernel needs a sequence row."""
    a, b, num_mask, weight, bias = _unaligned_operands(torch.bfloat16, 1, 0, 40, 24, C_z, True)
    result = OuterProductMeanCuTe()(a, b, num_mask, weight, bias, norm_before)
    _assert_matches_fp32_reference(result, a, b, num_mask, weight, bias, norm_before)


@pytest.mark.parametrize("sm", [86, 89])
@pytest.mark.parametrize("C_z", [128, 256], ids=["cz128", "cz256"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_outer_product_mean_smem_constrained_tiles(sm, C_z, dtype):
    """The SM86/89 tiles, including the C_z = 256 default, run exactly through the source path."""
    if SM_VERSION not in _CUTEDSL_SM or "source" not in _CUTEDSL_MODES:
        pytest.skip("the SM86/89 tiles compile for this GPU only through the CuTeDSL source path")
    backend = OuterProductMeanCuTe()
    # Select that SM's tile; the source path compiles it for the local GPU.
    backend._sm_version = sm
    a, b, num_mask, weight, bias = _unaligned_operands(dtype, 2, 97, 263, 257, C_z, True)
    _assert_matches_fp32_reference(backend(a, b, num_mask, weight, bias, False), a, b, num_mask, weight, bias, False)


def test_outer_product_mean_force_cubin_ignores_warmed_source(monkeypatch):
    backend = OuterProductMeanCuTe()
    config = object()
    key = (90, ("config",), True, True)
    cached_source, cubin = object(), object()
    monkeypatch.setattr(OuterProductMeanCuTe, "_compiled_cache", {key: cached_source})
    monkeypatch.setattr(backend, "force_cubin", lambda: True)
    monkeypatch.setattr(backend, "_load_cubin_executable", lambda *_args, **_kwargs: cubin)
    assert backend._get_or_compile(config, torch.bfloat16, True, True, key) is cubin


@pytest.mark.parametrize(
    ("operand", "tail"),
    [("a", 32), ("b", 32), ("output", 128)],
)
def test_outer_product_mean_cubin_rejects_wrong_static_tail(operand, tail):
    library = require_cubin_library()
    _require_cutedsl_mode("cubin")

    sm100 = SM_VERSION in (100, 103)
    if sm100 and operand == "a":
        pytest.skip("the SM100 kernel takes a relaid, with no static channel tail")

    batch, sequence, rows, columns = 1, 31, 17, 19
    dtype = torch.float16
    config = select_opm_config(SM_VERSION, rows, columns, sequence, True, False, "fp16")
    executable = opm_cubin.OuterProductMeanCubinExecutable(
        library,
        library.outer_product_mean,
        SM_VERSION,
        dtype,
        False,
        True,
        config_identity(config),
    )
    args = {
        "a": torch.zeros(batch, sequence, rows, 32, dtype=dtype, device="cuda"),
        "b": torch.zeros(batch, sequence, columns, 32, dtype=dtype, device="cuda"),
        "num_mask": torch.ones(batch, rows, columns, dtype=torch.float32, device="cuda"),
        "weight": torch.zeros(128, 32 * 32, dtype=dtype, device="cuda"),
        "output": torch.zeros(batch, rows, columns, 128, dtype=dtype, device="cuda"),
    }
    tensor = args[operand]
    args[operand] = torch.zeros(*tensor.shape[:-1], tail + 1, dtype=tensor.dtype, device=tensor.device)
    if sm100:
        args["a"] = opm_cutedsl._relayout_a(args["a"], config.cc)

    with pytest.raises(ValueError, match=rf"{operand} static tail"):
        executable(
            args["a"],
            args["b"],
            args["num_mask"],
            args["weight"],
            None,
            args["output"],
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("norm_before_output", [True, False], ids=["nb1", "nb0"])
@pytest.mark.parametrize(
    "n_seq,n_res",
    [(128, 64), (256, 128), (100, 130), (127, 63), (31, 129)],
    ids=["s128n64", "s256n128", "s100n130", "s127n63", "s31n129"],
)
@pytest.mark.parametrize("c_out", [128, 256], ids=["cz128", "cz256"])
def test_outer_product_mean_fused_matches_eager(dtype, norm_before_output, n_seq, n_res, c_out):
    skip_if_no_cutedsl("outer_product_mean")
    torch.manual_seed(0)
    c_in, c_hidden = 128, 32

    layer = OuterProductMean(
        c_in=c_in, c_hidden=c_hidden, c_out=c_out, norm_before_output=norm_before_output, dtype=dtype
    ).cuda()
    layer.load_state_dict({name: torch.randn_like(value) * 0.3 for name, value in layer.state_dict().items()})

    # Kernel-eligible (single-GPU, c_hidden==32, c_out in {128, 256}): the layer runs a fused backend.
    assert layer._opm_eligible

    m = torch.randn(1, n_seq, n_res, c_in, dtype=dtype, device="cuda")
    mask = (torch.rand(1, n_seq, n_res, device="cuda") < 0.9).to(dtype)

    out_kernel = layer(m, mask)
    layer._opm_eligible = False  # force the original eager path
    out_eager = layer(m, mask)

    assert out_kernel.shape == out_eager.shape == (1, n_res, n_res, c_out)
    diff = (out_kernel.float() - out_eager.float()).abs()
    rel_l2 = (diff.norm() / out_eager.float().norm().clamp_min(1e-6)).item()
    assert rel_l2 < 2e-2, (
        f"kernel vs eager rel_l2={rel_l2:.3e} (dtype={dtype}, nb={norm_before_output}, S={n_seq}, N={n_res})"
    )


def test_outer_product_mean_op_selector():
    """The selector returns a fused op on SM80+ for fp16/bf16 and the vanilla
    fallback for fp32 and unsupported dims."""
    op_bf16 = get_outer_product_mean_op(torch.bfloat16, C=32, D=32, C_z=128)
    op_fp32 = get_outer_product_mean_op(torch.float32, C=32, D=32, C_z=128)
    if SM_VERSION >= 80:
        assert op_bf16 is not _invoke_vanilla_opm
        assert get_outer_product_mean_op(torch.bfloat16, C=32, D=32, C_z=256) is not _invoke_vanilla_opm
    assert op_fp32 is _invoke_vanilla_opm
    assert get_outer_product_mean_op(torch.bfloat16, C=64, D=32, C_z=128) is _invoke_vanilla_opm
    assert get_outer_product_mean_op(torch.bfloat16, C=32, D=32, C_z=192) is _invoke_vanilla_opm


@pytest.mark.parametrize("force_cubin", [False, True], ids=["auto", "force_cubin"])
@pytest.mark.parametrize("sm", [100, 103])
def test_outer_product_mean_blackwell_selection(monkeypatch, sm, force_cubin):
    """SM100/103 select the tcgen05 CuTe kernel, which runs from source or from its CUBINs alike."""
    # Public backend selection lives in ops.py; patch it where it is defined.
    monkeypatch.setattr(opm_ops.ops, "get_sm_version", lambda: sm)
    monkeypatch.setenv(FORCE_CUBIN_ENV, "1" if force_cubin else "0")
    for dtype in (torch.bfloat16, torch.float16):
        for C_z in (128, 256):
            assert opm_ops.get_outer_product_mean_op(dtype, C=32, D=32, C_z=C_z) is opm_ops.ops._invoke_cute_opm


def _route_cute_call(monkeypatch, sm: int, S: int, C_z: int, side: int = 1) -> str:
    """Which backend ``_invoke_cute_opm`` runs for one ``side x side`` call at depth ``S`` on ``sm`` (backends
    stubbed)."""
    monkeypatch.setattr(opm_ops.ops, "get_sm_version", lambda: sm)
    monkeypatch.setattr(opm_ops.ops, "_invoke_triton_opm", lambda *args: "triton")
    monkeypatch.setattr(opm_ops.ops, "_get_cute_opm", lambda: lambda *args: "cute")
    a = torch.empty(1, S, side, 32, device="meta")
    return opm_ops.ops._invoke_cute_opm(a, a, None, torch.empty(C_z, 1024, device="meta"), None, False)


@pytest.mark.parametrize("sm", [100, 103])
def test_outer_product_mean_sm100_forced_cubin_at_call_time(monkeypatch, sm):
    """A layer built before CUBINs are forced still runs the CuTe op on SM100/103: the CUBINs serve its call."""
    monkeypatch.setenv(FORCE_CUBIN_ENV, "1")
    for C_z in (128, 256):
        assert _route_cute_call(monkeypatch, sm, 173, C_z) == "cute"


@pytest.mark.parametrize("force_cubin", [False, True], ids=["auto", "force_cubin"])
@pytest.mark.parametrize("C_z,sm103_max_s", [(128, 1536), (256, 1024)], ids=["cz128", "cz256"])
def test_outer_product_mean_sm100_deep_msa_takes_triton(monkeypatch, C_z, sm103_max_s, force_cubin):
    """Past its measured crossover depth the SM100 kernel hands the call to Triton in either mode: on SM100 at a
    depth set by the pair side, on SM103 per C_z. Other SMs have no cap."""
    monkeypatch.setenv(FORCE_CUBIN_ENV, "1" if force_cubin else "0")
    for sm, side, max_s in ((100, 320, 16384), (100, 319, 8192), (103, 384, sm103_max_s), (103, 256, sm103_max_s)):
        assert _route_cute_call(monkeypatch, sm, max_s, C_z, side) == "cute"
        assert _route_cute_call(monkeypatch, sm, max_s + 1, C_z, side) == "triton"
    assert _route_cute_call(monkeypatch, 90, 16384, C_z, 256) == "cute"


def test_outer_product_mean_config_selects_n_bucket_before_s():
    """Changing S must not make a fixed N jump to another tuned N bucket."""
    # For N=1024, S=1600 is closer to the N=1024/S=2048 variant than S=1024.
    # A sqrt(N*S) selector would instead pick the N=1536/S=1024 anchor.
    config = select_opm_config(sm_version=80, I=1024, J=1024, S=1600, norm_before=True, has_bias=True, dtype_str="bf16")

    assert (config.TILE_I, config.TILE_J) == (8, 4)
    assert config.atom_layout_s == (4, 2, 1)
    assert config.atom_layout_o == (1, 8, 1)

    selected, variants = _select_opm_config_bucket(
        sm_version=80, I=1024, J=1024, S=1600, norm_before=True, has_bias=True, dtype_str="bf16"
    )
    assert selected.config_key() == config.config_key()
    # N=1024 has three distinct tuned S variants; the backend precompiles all.
    assert len({variant.config_key() for variant in variants}) == 3


@pytest.mark.parametrize("C_z", [128, 256], ids=["cz128", "cz256"])
def test_outer_product_mean_sm90_config_selection(C_z):
    """SM90 bf16 selects the warp-specialized kernel for its C_z; fp16 keeps the SM80 kernel."""
    config = select_opm_config(90, 824, 824, 600, True, True, "bf16", C_z=C_z)
    assert isinstance(config, Sm90KernelConfig)
    assert config.cz == C_z
    assert not isinstance(select_opm_config(90, 824, 824, 600, True, True, "fp16", C_z=C_z), Sm90KernelConfig)


@pytest.mark.parametrize("sm", [100, 103])
@pytest.mark.parametrize("dtype_str", ["bf16", "fp16"])
@pytest.mark.parametrize("C_z", [128, 256], ids=["cz128", "cz256"])
def test_outer_product_mean_sm100_config_selection(sm, dtype_str, C_z):
    """SM100/103 select the tcgen05 tile for their C_z; the bf16-tuned tiles serve fp16 and every flag."""
    for I, S, norm_before, has_bias in ((960, 173, False, True), (37, 5, True, False), (4000, 2000, True, True)):
        config, variants = _select_opm_config_bucket(sm, I, I, S, norm_before, has_bias, dtype_str, C_z=C_z)
        assert isinstance(config, Sm100KernelConfig)
        assert (config.C_z, config.ab_dtype) == (C_z, dtype_str)
        assert all(isinstance(v, Sm100KernelConfig) and v.ab_dtype == dtype_str for v in variants)


@pytest.mark.parametrize("C_z", [128, 256], ids=["cz128", "cz256"])
def test_outer_product_mean_sm100_resident_b_fits_s(C_z):
    """A tile that keeps b resident serves S up to its capacity; past it the streaming default takes over."""
    resident = Sm100KernelConfig(C_z=C_z, tile_s=64, b_res=3)
    assert sm100_fit(resident, 192) is resident
    assert sm100_fit(resident, 193) == sm100_default_config("bf16", C_z)
    assert sm100_default_config("bf16", C_z).b_res == 0
    for S in (5, 192, 193, 2000):
        config, variants = _select_opm_config_bucket(100, 960, 960, S, False, True, "bf16", C_z=C_z)
        for c in (config, *variants):
            assert not c.b_res or -(-S // c.tile_s) <= c.b_res


@pytest.mark.parametrize("sm", [100, 103])
def test_outer_product_mean_sm100_selects_nearest_tile_that_holds_s(sm):
    """Between bucket capacities the selector takes the nearest-S tile that still holds S, not the default."""
    # C_z = 256, N = 960 ships S = 173 (b resident up to 192) and S = 597 (up to 640).
    for S in (193, 300, 640):
        config = select_opm_config(sm, 960, 960, S, False, True, "bf16", C_z=256)
        assert config.b_res * config.tile_s == 640, S
    assert select_opm_config(sm, 960, 960, 173, False, True, "bf16", C_z=256).b_res * 64 == 192
    # No resident tile holds S = 641, nor S = 300 at C_z = 128 (capacities 128 and 192), so b streams: on SM100
    # with the shipped streaming tile (cc = 8, half relayout buffers), on SM103 with the default one.
    for S, C_z in ((641, 256), (300, 128)):
        config = select_opm_config(sm, 960, 960, S, False, True, "bf16", C_z=C_z)
        if sm == 100:
            assert (config.b_res, config.cc, config.z_split) == (0, 8, 2), (S, C_z)
        else:
            assert config == sm100_default_config("bf16", C_z), (S, C_z)


@pytest.mark.parametrize("sm", [100, 103])
@pytest.mark.parametrize("C_z", [128, 256], ids=["cz128", "cz256"])
def test_outer_product_mean_a_chunks_cover_every_selectable_tile(monkeypatch, sm, C_z):
    """Every tile the SM100 selector can pick takes ``a`` in a chunk that ``outer_product_mean_a_chunks`` lists, so a
    producer that prepares those layouts serves every call. Other SMs and dtypes take ``a`` plain only."""
    monkeypatch.setattr(opm_ops.ops, "get_sm_version", lambda: sm)
    chunks = opm_ops.outer_product_mean_a_chunks(C_z, torch.bfloat16)
    for side in (16, 300, 960, 1144, 2000):
        for S in (1, 101, 173, 597, 641, 1024, 4096, 16384):
            assert select_opm_config(sm, side, side, S, False, True, "bf16", C_z=C_z).cc in chunks, (side, S)
    assert opm_ops.outer_product_mean_a_chunks(C_z, torch.float32) == ()
    monkeypatch.setattr(opm_ops.ops, "get_sm_version", lambda: 90)
    assert opm_ops.outer_product_mean_a_chunks(C_z, torch.bfloat16) == ()


def _sm100_configs(sms: tuple[int, ...] = (100, 103)) -> list[Sm100KernelConfig]:
    """Every tile the interface can select on ``sms``: the bundles' entries and the per-C_z defaults."""
    import json

    configs = {}
    for sm in sms:
        with open(os.path.join(_OPM_CONFIGS_DIR, f"sm{sm}.json")) as f:
            bundle = json.load(f)
        assert bundle["kernel_abi"] == "sm100"
        for raw in bundle["configs"].values():
            config = Sm100KernelConfig.from_dict(raw)
            configs[config.config_key()] = config
    for C_z in (128, 256):
        config = sm100_default_config("bf16", C_z)
        configs[config.config_key()] = config
    return list(configs.values())


def _require_sm100():
    if SM_VERSION not in (100, 103):
        pytest.skip("the SM100 OPM kernel runs on SM100/103")


def _require_cutedsl_mode(mode: str):
    if SM_VERSION not in _CUTEDSL_SM:
        pytest.skip(f"OPM CUBINs do not target SM{SM_VERSION}")
    if mode == "cubin" and SM_VERSION in (100, 103):
        _require_sm100_cubins(SM_VERSION)


def _require_sm100_cubins(*sms: int):
    """Skip unless the kernel library carries SM100-kernel images for ``sms``: one linked from packs that predate
    them runs these SMs on Triton."""
    require_cubin_family("outer_product_mean")
    shipped = {spec.target_sm for spec in require_cubin_library().outer_product_mean.kernel_specs() if spec.is_sm100}
    if missing := sorted(set(sms) - shipped):
        pytest.skip(f"this kernel library has no outer-product-mean CUBIN for SM{', SM'.join(map(str, missing))} yet")


def _runs_from_source() -> bool:
    return not OuterProductMeanCuTe.force_cubin() and source_module_available(_SOURCE_MODULE)


# (dtype, B, S, I, J, has_bias, norm_before): S off the ring tile and below one MMA K step, I off the
# pair's i tile and J off its j tile, I != J, a batch loop, and a single pair.
_SM100_CASES = [
    (torch.bfloat16, 1, 173, 117, 133, True, False),
    (torch.bfloat16, 2, 70, 65, 9, False, True),
    (torch.bfloat16, 1, 1, 33, 31, True, True),
    (torch.bfloat16, 1, 300, 129, 128, True, False),
    (torch.float16, 1, 97, 63, 70, True, True),
    (torch.float16, 3, 13, 1, 1, False, False),
]


@pytest.mark.parametrize("C_z", [128, 256], ids=["cz128", "cz256"])
@pytest.mark.parametrize(
    "dtype,B,S,I,J,has_bias,norm_before",
    _SM100_CASES,
    ids=[f"{str(c[0]).removeprefix('torch.')}-B{c[1]}-S{c[2]}-I{c[3]}-J{c[4]}" for c in _SM100_CASES],
)
def test_outer_product_mean_sm100_matches_reference(dtype, B, S, I, J, has_bias, norm_before, C_z):
    """The public backend on SM100/103 against the FP32 reference, partial tiles on every axis."""
    _require_sm100()
    a, b, num_mask, weight, bias = _unaligned_operands(dtype, B, S, I, J, C_z, has_bias)
    result = OuterProductMeanCuTe()(a, b, num_mask, weight, bias, norm_before)
    _assert_matches_fp32_reference(result, a, b, num_mask, weight, bias, norm_before)


@pytest.mark.parametrize("config", _sm100_configs(), ids=lambda c: "-".join(str(v) for v in c.config_key()[4:]))
def test_outer_product_mean_sm100_every_tile(config, monkeypatch):
    """Each selectable SM100 tile, forced through the backend, at a shape with partial pair tiles."""
    _require_sm100()
    if not _runs_from_source():
        _require_sm100_cubins(SM_VERSION)
    if not _runs_from_source() and config not in _sm100_configs((SM_VERSION,)):
        other_sm = 103 if SM_VERSION == 100 else 100
        pytest.skip(f"SM{SM_VERSION} ships no CUBIN for this SM{other_sm} tile")
    monkeypatch.setattr(opm_cutedsl, "_select_opm_config_bucket", lambda *args, **kwargs: (config, (config,)))
    # S at the resident b capacity (its last box full) and inside it (zero fill past S).
    for S in sorted({config.b_res * config.tile_s or 150, 150 if config.b_res * config.tile_s >= 150 else 1}):
        a, b, num_mask, weight, bias = _unaligned_operands(torch.bfloat16, 1, S, 200, 75, config.C_z, True)
        result = OuterProductMeanCuTe()(a, b, num_mask, weight, bias, False)
        _assert_matches_fp32_reference(result, a, b, num_mask, weight, bias, False)
        assert OuterProductMeanCuTe._compiled_cache.get((SM_VERSION, config.config_key(), True, False)) is not None


# Half relayout buffers (z_split = 2) with buffer counts, chunk widths and output paths the shipped tiles do not
# use: two buffers and register stores, staged stores beside a deeper W ring, and cc = 4 halves over a resident b.
_SM100_SPLIT_TILES = [
    Sm100KernelConfig(C_z=128, cc=8, ab_stages=4, w_stages=2, z_stages=2, z_split=2, epi_cols=0),
    Sm100KernelConfig(C_z=256, cc=8, ab_stages=3, w_stages=4, z_stages=1, z_split=2, raster=8, epi_cols=32),
    Sm100KernelConfig(C_z=256, cc=4, ab_stages=4, w_stages=2, z_stages=1, z_split=2, b_res=5, epi_cols=32),
]


@pytest.mark.parametrize(
    "config", _SM100_SPLIT_TILES, ids=lambda c: f"cz{c.C_z}-cc{c.cc}-z{c.z_stages}-e{c.epi_cols}-b{c.b_res}"
)
def test_outer_product_mean_sm100_split_relayout(config, monkeypatch):
    """Half relayout buffers where every cluster runs several pair tiles, so each tile's last O-GEMM halves run
    inside the next tile's first S-GEMM: the same bits as Triton (same rounding points), near the FP32 reference."""
    _require_sm100()
    if not _runs_from_source():
        pytest.skip("these tiles ship no CUBIN; they run from source")
    monkeypatch.setattr(opm_cutedsl, "_select_opm_config_bucket", lambda *args, **kwargs: (config, (config,)))
    # 2 x 7 x 10 pair tiles at cc = 8 (2 x 4 x 19 at cc = 4) over at most 74 clusters; S off the ring tile.
    a, b, num_mask, weight, bias = _unaligned_operands(torch.bfloat16, 2, 150, 200, 75, config.C_z, True)
    for norm_before in (False, True):
        result = OuterProductMeanCuTe()(a, b, num_mask, weight, bias, norm_before)
        assert torch.equal(result, _invoke_triton_opm(a, b, num_mask, weight, bias, norm_before))
        _assert_matches_fp32_reference(result, a, b, num_mask, weight, bias, norm_before)


def test_outer_product_mean_sm100_capture_safe():
    """The SM100 path (relayout plus kernel) replays from a CUDA graph with fresh operand values."""
    _require_sm100()
    a, b, num_mask, weight, bias = _unaligned_operands(torch.bfloat16, 1, 64, 96, 96, 256, True)
    backend = OuterProductMeanCuTe()
    backend(a, b, num_mask, weight, bias, False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = backend(a, b, num_mask, weight, bias, False)
    a.copy_(torch.randn_like(a))
    graph.replay()
    _assert_matches_fp32_reference(out, a, b, num_mask, weight, bias, False)


@pytest.mark.parametrize("C_z", [128, 256], ids=["cz128", "cz256"])
def test_outer_product_mean_sm100_layer_runs_with_cubins_forced_later(monkeypatch, C_z):
    """A layer built on the CuTe op runs the SM100 CUBIN once CUTEDSL_FORCE_CUBIN=1 is set (as model tests do),
    bitwise equal to the kernel compiled from source."""
    _require_sm100()
    if not source_module_available(_SOURCE_MODULE):
        pytest.skip("compares against the CuTeDSL source kernel")
    _require_sm100_cubins(SM_VERSION)
    monkeypatch.delenv(FORCE_CUBIN_ENV, raising=False)
    cache: dict = {}
    monkeypatch.setattr(OuterProductMeanCuTe, "_compiled_cache", cache)
    layer = OuterProductMean(c_in=64, c_hidden=32, c_out=C_z, norm_before_output=False, dtype=torch.bfloat16).cuda()
    assert layer._opm_op is opm_ops.ops._invoke_cute_opm
    a, b, num_mask, weight, bias = _unaligned_operands(torch.bfloat16, 1, 40, 48, 48, C_z, True)
    source = layer._opm_op(a, b, num_mask, weight, bias, False)
    _assert_matches_fp32_reference(source, a, b, num_mask, weight, bias, False)
    assert not any(isinstance(executable, CuTeDSLKernelLibraryExecutable) for executable in cache.values())
    monkeypatch.setenv(FORCE_CUBIN_ENV, "1")
    forced = layer._opm_op(a, b, num_mask, weight, bias, False)
    assert cache and all(
        isinstance(executable, CuTeDSLKernelLibraryExecutable) and executable._config.spec.is_sm100
        for executable in cache.values()
    )
    assert torch.equal(forced, source)


def _sequence_last_reference(a: torch.Tensor, cc: int) -> torch.Tensor:
    """``a [B, S, I, C]`` as ``[B, C / cc, ceil(I / T), cc, T, S]`` (``T = 128 / cc``), pad tokens zero."""
    B, S, I, C = a.shape
    T = 128 // cc
    i_pad = -(-I // T) * T
    padded = torch.nn.functional.pad(a, (0, 0, 0, i_pad - I))
    return padded.reshape(B, S, i_pad // T, T, C // cc, cc).permute(0, 4, 2, 5, 3, 1)


@pytest.mark.parametrize("cc", [4, 8])
@pytest.mark.parametrize("shape", [(1, 1, 1, 32), (2, 37, 65, 32), (1, 173, 960, 32), (3, 64, 7, 32)])
def test_relayout_a_writes_the_sm100_layout(cc: int, shape: tuple[int, ...]) -> None:
    """Rows of ``S`` start every 64 elements, a chunk's ``cc`` channels of ``T`` tokens are consecutive rows,
    and ``_plain_a`` undoes it."""
    a = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    a_t = opm_cutedsl._relayout_a(a, cc)
    expected = _sequence_last_reference(a, cc)
    T = 128 // cc
    # Pad tokens are never written; compare the real ones.
    real = torch.arange(a_t.shape[2] * T, device="cuda").view(a_t.shape[2], T) < shape[2]
    assert torch.equal(a_t.permute(0, 1, 3, 5, 2, 4)[..., real], expected.permute(0, 1, 3, 5, 2, 4)[..., real])
    pitch = -(-shape[1] // 64) * 64
    rows = a_t.shape[2] * a_t.shape[3] * a_t.shape[4]
    assert a_t.stride()[-2:] == (pitch, 1) and a_t.stride(1) == rows * pitch and a_t.data_ptr() % 16 == 0
    assert torch.equal(opm_cutedsl._plain_a(a_t, shape[2]), a)


def test_relayout_a_reads_strided_input() -> None:
    base = torch.randn(1, 40, 33, 32, device="cuda", dtype=torch.bfloat16)
    a = base[:, 3:, 1:]  # a non-contiguous view at an odd element offset
    assert torch.equal(opm_cutedsl._plain_a(opm_cutedsl._relayout_a(a, 4), a.shape[2]), a)


@pytest.mark.parametrize("mode", _CUTEDSL_MODES)
@pytest.mark.parametrize("C_z", [128, 256], ids=["cz128", "cz256"])
def test_outer_product_mean_layer_prologue_feeds_the_kernel_in_place(mode, C_z, monkeypatch):
    """On every SM the fused op serves, the layer's prologue writes ``a`` and ``b`` in the layout that SM's
    kernel loads: the op launches on the prologue's own tensors, with no copy or relayout in between."""
    _require_cutedsl_mode(mode)
    from bionemo_ir.dsl_kernels.triton.opm_prologue import OpmPrologue

    produced, launched = [], []
    prologue = OpmPrologue.__call__
    launch = opm_cutedsl.launch_compiled_kernel

    def record_prologue(self, *args, **kwargs):
        outputs = prologue(self, *args, **kwargs)
        produced.append(outputs)
        return outputs

    def record_launch(executable, a, b, *rest):
        launched.append((a, b))
        return launch(executable, a, b, *rest)

    def no_relayout(*_args, **_kwargs):
        raise AssertionError("the op relaid the prologue's a")

    monkeypatch.setattr(OpmPrologue, "__call__", record_prologue)
    monkeypatch.setattr(opm_cutedsl, "launch_compiled_kernel", record_launch)
    monkeypatch.setattr(opm_cutedsl, "_relayout_a", no_relayout)
    torch.manual_seed(0)
    layer = OuterProductMean(c_in=64, c_hidden=32, c_out=C_z, norm_before_output=False, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(0, 0.1)
        m = torch.randn(1, 40, 133, 64, device="cuda", dtype=torch.bfloat16)
        mask = (torch.rand(1, 40, 133, device="cuda") > 0.1).to(torch.bfloat16)
        run_cutedsl_test_mode(mode, monkeypatch, OuterProductMeanCuTe, opm_cutedsl, lambda: layer(m, mask))

    assert len(produced) == len(launched) == 2
    for (a, b), (launched_a, launched_b) in zip(produced, launched, strict=True):
        for made, used in ((a, launched_a), (b, launched_b)):
            assert used.data_ptr() == made.data_ptr()
            assert (used.shape, used.stride()) == (made.shape, made.stride())
        # SM100/SM103 take a sequence-last; every other SM reads a and b as [B, S, I, C].
        assert a.dim() == (6 if SM_VERSION in (100, 103) else 4)
        assert b.is_contiguous() and (a.dim() == 6 or a.is_contiguous())


@pytest.mark.parametrize("C_z", [128, 256], ids=["cz128", "cz256"])
def test_outer_product_mean_sm100_layer_prologue_feeds_the_cubin(monkeypatch, C_z):
    """The layer's fused prologue writes ``a`` sequence-last; the forced CUBIN takes it bitwise like the source."""
    _require_sm100()
    if not source_module_available(_SOURCE_MODULE):
        pytest.skip("compares against the CuTeDSL source kernel")
    _require_sm100_cubins(SM_VERSION)
    monkeypatch.delenv(FORCE_CUBIN_ENV, raising=False)
    monkeypatch.setattr(OuterProductMeanCuTe, "_compiled_cache", {})
    torch.manual_seed(0)
    layer = OuterProductMean(c_in=64, c_hidden=32, c_out=C_z, norm_before_output=False, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(0, 0.1)
        m = torch.randn(1, 40, 133, 64, device="cuda", dtype=torch.bfloat16)
        mask = (torch.rand(1, 40, 133, device="cuda") > 0.1).to(torch.bfloat16)
        assert opm_ops.outer_product_mean_a_chunk(40, 133, 133, C_z, False, True, torch.bfloat16) is not None
        source = layer(m, mask)
        monkeypatch.setenv(FORCE_CUBIN_ENV, "1")
        forced = layer(m, mask)
    assert torch.equal(forced, source)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("C_z", [128, 256], ids=["cz128", "cz256"])
def test_outer_product_mean_sm100_modes_run_the_blackwell_kernel(monkeypatch, dtype, C_z):
    """Every selected mode runs the tcgen05 kernel, the CUBIN through its SM100 launch ABI, and with both
    selected they agree bitwise."""
    _require_sm100()
    if "cubin" in _CUTEDSL_MODES:
        _require_sm100_cubins(SM_VERSION)
    a, b, num_mask, weight, bias = _unaligned_operands(dtype, 2, 173, 117, 133, C_z, True)
    outputs = []
    for mode in _CUTEDSL_MODES:
        with monkeypatch.context() as patch:
            result = run_cutedsl_test_mode(
                mode,
                patch,
                OuterProductMeanCuTe,
                opm_cutedsl,
                lambda: OuterProductMeanCuTe()(a, b, num_mask, weight, bias, False),
            )
            executables = list(OuterProductMeanCuTe._compiled_cache.values())
        if mode == "cubin":
            assert all(executable._config.spec.is_sm100 for executable in executables)
            assert all(executable._config.cubin.kernel_sm == 100 for executable in executables)
        _assert_matches_fp32_reference(result, a, b, num_mask, weight, bias, False)
        outputs.append(result)
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])


def test_every_sm100_tile_has_a_shipped_cubin():
    """Each SM100 and SM103 bundle tile, in both dtypes, has a registry image with the SM100 launch ABI."""
    _require_sm100_cubins(100, 103)
    launcher = require_cubin_library().outer_product_mean
    for sm in (100, 103):
        for config in _sm100_configs((sm,)):
            for dtype, library_dtype in (("bf16", launcher.DType.BFLOAT16), ("fp16", launcher.DType.FLOAT16)):
                identity = config_identity(Sm100KernelConfig.from_dict(dict(config.to_dict(), ab_dtype=dtype)))
                for has_bias in (False, True):
                    for norm_before in (False, True):
                        kernel = launcher.make_kernel_config(sm, library_dtype, has_bias, norm_before, identity)
                        assert kernel.spec.is_sm100 and kernel.spec.target_sm == sm
                        assert kernel.cubin.kernel_sm == 100


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("cause", ["force_cubin", "no_sources"])
def test_outer_product_mean_sm100_library_without_its_cubin_takes_triton(monkeypatch, graph, cause):
    """A kernel library built before the SM100 and SM103 CUBINs shipped cannot load one, so a call that cannot
    compile the kernel runs on Triton instead of raising: eagerly and under graph capture."""
    _require_sm100()

    def library_without_this_sm(*_args, **_kwargs):
        raise opm_cubin.OuterProductMeanCubinMissing(f"no outer-product-mean CUBIN for SM{SM_VERSION}")

    def no_sources(*_args, **_kwargs):
        raise ImportError("no CuTeDSL outer-product-mean sources")

    monkeypatch.setattr(opm_cutedsl, "populate_compiled_cache_from_library", library_without_this_sm)
    monkeypatch.setattr(OuterProductMeanCuTe, "_compiled_cache", {})
    monkeypatch.setattr(OuterProductMeanCuTe, "_triton_fallback_keys", set())
    if cause == "force_cubin":
        monkeypatch.setenv(FORCE_CUBIN_ENV, "1")
    else:
        monkeypatch.delenv(FORCE_CUBIN_ENV, raising=False)
        monkeypatch.setattr(OuterProductMeanCuTe, "_resolve_source", staticmethod(no_sources))
    a, b, num_mask, weight, bias = _unaligned_operands(torch.bfloat16, 1, 17, 9, 9, 128, False)
    backend = OuterProductMeanCuTe()
    # Triton takes a plain, so a producer is not asked for the sequence-last layout.
    assert backend.a_chunk(17, 9, 9, 128, False, False, torch.bfloat16) is None
    expected = _invoke_triton_opm(a, b, num_mask, weight, bias, False)
    if graph:
        backend(a, b, num_mask, weight, bias, False)
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            actual = backend(a, b, num_mask, weight, bias, False)
        captured.replay()
    else:
        actual = backend(a, b, num_mask, weight, bias, False)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("cause", ["force_cubin", "no_sources"])
@pytest.mark.parametrize(
    "fault",
    [
        CuTeDSLKernelLibraryUnavailable("the kernel library failed to load"),
        CuTeDSLKernelVariantUnavailable("the CUBIN config cannot launch"),
    ],
    ids=["library", "config"],
)
def test_outer_product_mean_sm100_broken_library_raises_instead_of_taking_triton(monkeypatch, fault, cause):
    """Only a library without the image falls back to Triton: one that fails to load, or whose image cannot
    launch, is a broken installation and must not degrade silently."""
    _require_sm100()

    def broken_library(*_args, **_kwargs):
        raise fault

    def no_sources(*_args, **_kwargs):
        raise ImportError("no CuTeDSL outer-product-mean sources")

    monkeypatch.setattr(opm_cutedsl, "populate_compiled_cache_from_library", broken_library)
    monkeypatch.setattr(OuterProductMeanCuTe, "_compiled_cache", {})
    monkeypatch.setattr(OuterProductMeanCuTe, "_triton_fallback_keys", set())
    if cause == "force_cubin":
        monkeypatch.setenv(FORCE_CUBIN_ENV, "1")
        raised = pytest.raises(RuntimeError, match=FORCE_CUBIN_ENV)
    else:
        monkeypatch.delenv(FORCE_CUBIN_ENV, raising=False)
        monkeypatch.setattr(OuterProductMeanCuTe, "_resolve_source", staticmethod(no_sources))
        raised = pytest.raises(type(fault), match=str(fault))
    a, b, num_mask, weight, bias = _unaligned_operands(torch.bfloat16, 1, 17, 9, 9, 128, False)
    with raised:
        OuterProductMeanCuTe()(a, b, num_mask, weight, bias, False)


def test_outer_product_mean_sm100_library_miss_is_looked_up_once(monkeypatch):
    """A call that fell back to Triton on a library without its image does not search the library again."""
    _require_sm100()
    lookups = []

    def library_without_this_sm(*_args, **_kwargs):
        lookups.append(1)
        raise opm_cubin.OuterProductMeanCubinMissing(f"no outer-product-mean CUBIN for SM{SM_VERSION}")

    monkeypatch.setattr(opm_cutedsl, "populate_compiled_cache_from_library", library_without_this_sm)
    monkeypatch.setattr(OuterProductMeanCuTe, "_compiled_cache", {})
    monkeypatch.setattr(OuterProductMeanCuTe, "_triton_fallback_keys", set())
    monkeypatch.setenv(FORCE_CUBIN_ENV, "1")
    a, b, num_mask, weight, bias = _unaligned_operands(torch.bfloat16, 1, 17, 9, 9, 128, False)
    backend = OuterProductMeanCuTe()
    first = backend(a, b, num_mask, weight, bias, False)
    misses = len(lookups)
    assert misses
    assert torch.equal(backend(a, b, num_mask, weight, bias, False), first)
    assert len(lookups) == misses


@pytest.mark.parametrize(
    ("raised", "missing"), [(IndexError, True), (ValueError, False), (TypeError, False), (RuntimeError, False)]
)
def test_the_cubin_adapter_tells_a_missing_image_from_a_bad_config(raised, missing):
    """The launcher reports a registry miss as IndexError; any other config failure is not a missing image."""

    class Launcher:
        class DType:
            FLOAT16, BFLOAT16 = 0, 1

        def make_kernel_config(self, *_args):
            raise raised("make_kernel_config failed")

    with pytest.raises(CuTeDSLKernelVariantUnavailable) as error:
        opm_cubin.OuterProductMeanCubinExecutable(None, Launcher(), 100, torch.bfloat16, False, False, "tile")
    assert isinstance(error.value, opm_cubin.OuterProductMeanCubinMissing) is missing


def test_the_launcher_reports_a_missing_image_as_index_error():
    """The registry miss the Triton fallback keys on comes from the C++ lookup, not from argument checks."""
    require_cubin_family("outer_product_mean")
    launcher = require_cubin_library().outer_product_mean
    with pytest.raises(IndexError, match="No embedded outer-product-mean CUBIN"):
        launcher.make_kernel_config(100, launcher.DType.BFLOAT16, False, False, "no-such-tile")


@pytest.mark.parametrize("B,I,J", [(1, 0, 3), (0, 4, 4), (1, 4, 0)])
def test_outer_product_mean_sm100_empty_inputs(B, I, J):
    """Empty batch or token axes return an empty output without launching."""
    _require_sm100()
    a = torch.randn(B, 5, I, 32, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(B, 5, J, 32, device="cuda", dtype=torch.bfloat16)
    num_mask = torch.ones(B, I, J, device="cuda")
    weight = torch.randn(128, 1024, device="cuda", dtype=torch.bfloat16)
    assert OuterProductMeanCuTe()(a, b, num_mask, weight, None, False).shape == (B, I, J, 128)
    assert opm_ops.ops._invoke_cute_opm(a, b, num_mask, weight, None, False).shape == (B, I, J, 128)


def test_outer_product_mean_unaligned_operand_bases():
    """Contiguous views that start off a 16-byte boundary still run on every SM the CuTe op serves: the kernels
    declare 16-byte bases, and TMA (SM90, SM100) requires them."""
    if SM_VERSION not in _CUTEDSL_SM:
        pytest.skip(f"the CuTe OPM does not run on SM{SM_VERSION}")
    a, b, num_mask, weight, bias = _unaligned_operands(torch.bfloat16, 1, 37, 70, 33, 128, True)

    def shifted(t):
        flat = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)[1:]
        return flat.view(t.shape).copy_(t)

    a, b, weight, bias = shifted(a), shifted(b), shifted(weight), shifted(bias)
    assert all(t.data_ptr() % 16 for t in (a, b, weight, bias))
    result = OuterProductMeanCuTe()(a, b, num_mask, weight, bias, False)
    _assert_matches_fp32_reference(result, a, b, num_mask, weight, bias, False)


def test_in_place_copies_only_operands_off_the_kernel_contract():
    """Operands the kernels read as they lie pass through; a narrowed, misaligned or channel-strided one is copied."""
    base = torch.randn(2, 40, 33, 32, device="cuda", dtype=torch.bfloat16)
    for conforming in (base, base[:, 8:], base[:, :, 1:]):
        assert opm_cutedsl._in_place(conforming) is conforming
    misaligned = torch.empty(base.numel() + 1, device="cuda", dtype=base.dtype)[1:].view(base.shape).copy_(base)
    for bad in (base[..., :16], misaligned, base.transpose(1, 2)):
        fixed = opm_cutedsl._in_place(bad)
        assert fixed is not bad and fixed.is_contiguous() and fixed.data_ptr() % 16 == 0 and torch.equal(fixed, bad)
    # An fp32 count with an odd I * J is dense over (I, J); its batch stride need not be a 16-byte multiple.
    counts = torch.rand(3, 5, 7, device="cuda")
    assert opm_cutedsl._in_place(counts, align=4) is counts


@pytest.mark.parametrize("sm", [100, 103])
@pytest.mark.parametrize("depth", [0, 17])
def test_outer_product_mean_unsupported_direct_width(monkeypatch: pytest.MonkeyPatch, sm: int, depth: int) -> None:
    """Direct internal routing rejects an unsupported output width without a KeyError."""
    assert _route_cute_call(monkeypatch, sm, depth, 192) == "triton"
