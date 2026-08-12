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

from tensorrt_bionemo._torch.auto_chunk import ChunkPolicy
from tensorrt_bionemo._torch.custom_ops import outer_product_mean as opm_ops
from tensorrt_bionemo._torch.custom_ops.outer_product_mean import (
    OuterProductMeanCuTe,
    _select_opm_config_bucket,
    get_outer_product_mean_op,
    select_opm_config,
)
from tensorrt_bionemo._torch.custom_ops.outer_product_mean import cutedsl as opm_cutedsl
from tensorrt_bionemo._torch.custom_ops.outer_product_mean.ops import _invoke_vanilla_opm
from tensorrt_bionemo._torch.layers.outer_product_mean import OuterProductMean
from tensorrt_bionemo.utils import str_dtype_to_torch
from tests._torch import SM_VERSION, cutedsl_test_modes, run_cutedsl_test_mode, skip_if_no_cutedsl

_CUTEDSL_SM = (80, 86, 89, 90, 100, 103)
_CUTEDSL_MODES = cutedsl_test_modes("tensorrt_bionemo.dsl_kernels.cute.sm80_opm")


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

    # On a CuTeDSL-capable GPU the half-precision, non-chunked path must resolve
    # to the fused custom op (guards against a silent fall-back to eager).
    if dtype in (torch.float16, torch.bfloat16) and SM_VERSION in _CUTEDSL_SM:
        assert outer_product_mean._opm_eligible
        assert isinstance(
            get_outer_product_mean_op(
                dtype,
                C=outer_product_mean.c_hidden,
                D=outer_product_mean.c_hidden,
                C_z=outer_product_mean.c_out,
            ),
            OuterProductMeanCuTe,
        )

    m = torch.randn(bs, sc.n_seq, sc.n_res, ref_m.c_in, dtype=torch.float32).cuda()
    mask = torch.randint(0, 2, (bs, sc.n_seq, sc.n_res), dtype=torch.float32).to(device)

    with torch.inference_mode():
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
    with torch.no_grad():
        for p in opm.parameters():
            p.normal_(mean=0.0, std=0.1)
    m = torch.randn(1, 6, 40, 32, device=device)  # N=40 output rows
    mask = torch.randint(0, 2, (1, 6, 40), dtype=torch.float32, device=device)

    with torch.inference_mode():
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
def test_outer_product_mean_source_and_cubin(mode, dtype, has_bias, norm_before, monkeypatch):
    if SM_VERSION not in _CUTEDSL_SM:
        pytest.skip(f"OPM CUBINs do not target SM{SM_VERSION}")
    torch.manual_seed(7)
    B, S, I, J, C, D, C_z = 1, 31, 17, 19, 32, 32, 128
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


def test_outer_product_mean_force_cubin_ignores_warmed_source(monkeypatch):
    backend = OuterProductMeanCuTe()
    config = object()
    key = (90, ("config",), True, True)
    cached_source, cubin = object(), object()
    monkeypatch.setattr(OuterProductMeanCuTe, "_compiled_cache", {key: cached_source})
    monkeypatch.setattr(backend, "force_cubin", lambda: True)
    monkeypatch.setattr(backend, "_load_cubin_executable", lambda *_args, **_kwargs: cubin)
    assert backend._get_or_compile(config, torch.bfloat16, True, True, key) is cubin


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("norm_before_output", [True, False], ids=["nb1", "nb0"])
@pytest.mark.parametrize(
    "n_seq,n_res",
    [(128, 64), (256, 128), (100, 130), (127, 63), (31, 129)],
    ids=["s128n64", "s256n128", "s100n130", "s127n63", "s31n129"],
)
def test_outer_product_mean_cute_matches_eager(dtype, norm_before_output, n_seq, n_res):
    skip_if_no_cutedsl()
    torch.manual_seed(0)
    c_in, c_hidden, c_out = 128, 32, 128

    layer = OuterProductMean(
        c_in=c_in, c_hidden=c_hidden, c_out=c_out, norm_before_output=norm_before_output, dtype=dtype
    ).cuda()
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(0, 0.3)

    # Kernel-eligible (single-GPU, c_hidden==32, c_out==128) and the op resolves
    # to the CuTe backend on this GPU.
    assert layer._opm_eligible
    assert isinstance(
        get_outer_product_mean_op(dtype, C=layer.c_hidden, D=layer.c_hidden, C_z=layer.c_out),
        OuterProductMeanCuTe,
    )

    m = torch.randn(1, n_seq, n_res, c_in, dtype=dtype, device="cuda")
    mask = (torch.rand(1, n_seq, n_res, device="cuda") < 0.9).to(dtype)

    with torch.inference_mode():
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
    """The selector returns the CuTe op on CuTeDSL GPUs for fp16/bf16 and the
    vanilla fallback for fp32 / unsupported hardware."""
    op_bf16 = get_outer_product_mean_op(torch.bfloat16, C=32, D=32, C_z=128)
    op_fp32 = get_outer_product_mean_op(torch.float32, C=32, D=32, C_z=128)
    if SM_VERSION in _CUTEDSL_SM:
        assert isinstance(op_bf16, OuterProductMeanCuTe)
    assert not isinstance(op_fp32, OuterProductMeanCuTe)
    assert get_outer_product_mean_op(torch.bfloat16, C=64, D=32, C_z=128) is _invoke_vanilla_opm


@pytest.mark.parametrize("sm", [100, 103])
def test_outer_product_mean_supports_blackwell(monkeypatch, sm):
    sentinel = object()
    # Public backend selection lives in ops.py; patch it where it is defined.
    monkeypatch.setattr(opm_ops.ops, "get_sm_version", lambda: sm)
    monkeypatch.setattr(opm_ops.ops, "_opm_cute_instance", sentinel)
    assert opm_ops.get_outer_product_mean_op(torch.bfloat16, C=32, D=32, C_z=128) is sentinel


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
