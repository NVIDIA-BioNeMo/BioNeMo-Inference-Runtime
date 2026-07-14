# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional

import pytest
import torch
from tensorrt_llm_lite._utils import str_dtype_to_torch
from test_utils.boltz.create_and_load_weights import (
    create_pair_weighted_averaging_weights,
    load_pair_weighted_averaging_weights_torch)
from test_utils.boltz.ref_layers import RefPairWeightedAveraging

from tensorrt_bionemo._torch.auto_chunk import ChunkPolicy
from tensorrt_bionemo._torch.custom_ops.pair_weighted_averaging import (
    PairWeightedAveragingCuTe, get_pair_weighted_averaging_op)
from tensorrt_bionemo._torch.layers.pair_averaging import PairWeightedAveraging
from tests._torch import SM_VERSION, skip_if_no_cutedsl

_CUTEDSL_SM = (80, 86, 89, 90)


@dataclass(kw_only=True, frozen=True)
class Scenario:
    torch_dtype: str = "float32"
    # Sequence(S)-rows per chunk via a registry-style ChunkPolicy (None -> dense path). Row-chunking
    # over S is numerically identical, so the chunked output must still match the golden ref.
    chunk: Optional[int] = None
    n_seq: int = 32
    n_res: int = 64


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(),
        # fp16 / bf16 single-GPU with the kernel's fixed dims (H==8, c_h==32,
        # c_m==64) routes through the SM80 fused PWA CuTe custom op; fp32 stays
        # on the eager path.
        Scenario(torch_dtype="bfloat16"),
        Scenario(torch_dtype="float16"),
        # Larger N exercises the kernel's pseudo-seqlen config selection.
        Scenario(torch_dtype="bfloat16", n_res=128),
        # fp32 cannot use the fused kernel, so these exercise the auto-chunk fallback.
        Scenario(chunk=1),
        Scenario(chunk=3, n_seq=8),
    ],
    ids=[
        "float32", "bfloat16", "float16", "bfloat16_n128", "chunk1_fp32",
        "chunk3_partial_fp32"
    ])
def test_pair_weighted_averaging(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device('cuda')

    ref_m = RefPairWeightedAveraging.load_weights()
    ref_m = ref_m.to(device)

    weights_and_biases = create_pair_weighted_averaging_weights(from_ref=ref_m)

    # A low min_size makes the policy trip at the test's token count, so `chunk` scenarios exercise
    # the head-chunked accumulate path; `chunk=None` leaves the registry default (dense here).
    chunk_policy = (ChunkPolicy(chunk_size=sc.chunk, min_size=1)
                    if sc.chunk is not None else None)
    pair_weighted_averaging = PairWeightedAveraging(c_m=ref_m.c_m,
                                                    c_z=ref_m.c_z,
                                                    c_h=ref_m.c_h,
                                                    num_heads=ref_m.num_heads,
                                                    dtype=dtype,
                                                    chunk_policy=chunk_policy)
    load_pair_weighted_averaging_weights_torch(pair_weighted_averaging,
                                               weights_and_biases,
                                               dtype=dtype)
    pair_weighted_averaging.to(device)

    # On a CuTeDSL-capable GPU the half-precision path (boltz-2 PWA is H=8,
    # c_h=32, c_m=64) must resolve to the fused custom op -- guards against a
    # silent fall-back to eager.
    if dtype in (torch.float16, torch.bfloat16) and SM_VERSION in _CUTEDSL_SM:
        assert pair_weighted_averaging._pwa_op_eligible
        assert isinstance(get_pair_weighted_averaging_op(dtype),
                          PairWeightedAveragingCuTe)

    m = torch.randn(bs, sc.n_seq, sc.n_res, ref_m.c_m,
                    dtype=torch.float32).cuda()
    z = torch.randn(bs, sc.n_res, sc.n_res, ref_m.c_z,
                    dtype=torch.float32).cuda()
    # 0/1 pair mask: the layer applies ``(1 - mask) * -inf`` with inf=1e9, so a
    # random *normal* mask overflows fp16 (-> +/-inf -> NaN softmax); a 0/1 mask
    # keeps the masked bias at 0 / -1e9 and is the realistic input anyway.
    mask = torch.randint(0, 2, (bs, sc.n_res, sc.n_res),
                         dtype=torch.float32).cuda()

    with torch.inference_mode():
        ref_output_float = ref_m(m, z, mask)
        m = m.to(dtype)
        z = z.to(dtype)
        mask = mask.to(dtype)
        ref_m = ref_m.to(dtype)

        ref_output = ref_m(m, z, mask)
        output = pair_weighted_averaging.forward(m, z, mask)

    assert ref_output.shape == output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(ref_output, output, atol=1e-3, rtol=1e-4)
    else:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(
            torch.abs(ref_output.float() - ref_output_float))

        assert abs(diff0_max - diff1_max) / torch.min(diff0_max,
                                                      diff1_max) <= 0.5
        assert abs(diff0_mean - diff1_mean) <= 0.2


@pytest.mark.parametrize("s_rows", [1, 2, 3, 5],
                         ids=["s1", "s2", "s3_partial", "s5_all"])
def test_pair_weighted_averaging_chunk_matches_dense(s_rows: int):
    """Sequence(S)-row chunking (registry policy) matches the dense path.

    Same instance dense vs chunked (no golden weights needed): each S-slice is independent (the
    attention mixes only the token dims), so the concatenated result is numerically identical.
    """
    torch.manual_seed(0)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device('cuda')

    pwa = PairWeightedAveraging(c_m=64,
                                c_z=32,
                                c_h=16,
                                num_heads=8,
                                dtype=torch.float32).to(device)
    pwa.eval()
    # Constructed weights are zero-initialized (production loads them); give them real values so
    # the dense-vs-chunked comparison is meaningful rather than 0 == 0.
    with torch.no_grad():
        for p in pwa.parameters():
            p.normal_(mean=0.0, std=0.1)
    m = torch.randn(1, 5, 48, 64, device=device)  # S=5
    z = torch.randn(1, 48, 48, 32, device=device)
    mask = torch.randint(0, 2, (1, 48, 48), dtype=torch.float32, device=device)

    with torch.inference_mode():
        # Registry default (memory-scaled min_size) does not trip at S=5 -> dense reference.
        dense = pwa(m, z, mask)

        pwa.chunk_policy = ChunkPolicy(chunk_size=s_rows,
                                       min_size=1)  # chunk over S
        chunked = pwa(m, z, mask)

    torch.testing.assert_close(chunked, dense, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Fused SM80 CuTe custom op: kernel path vs the eager path (self-contained --
# random weights, no downloaded checkpoints), so CI exercises the kernel even
# without hub access. Confirms the dispatch, the j-padding, the RAW-gate /
# in-kernel-sigmoid, and the proj_o weight layout all match the eager math
# within bf16/fp16 tolerance, on aligned and non-aligned token counts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16],
                         ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "n_seq,n_res",
    [(16, 128), (8, 100), (4, 256), (8, 130), (200, 128), (48, 33), (17, 127),
     (31, 129)],
    ids=[
        "s16n128", "s8n100", "s4n256", "s8n130", "s200n128", "s48n33",
        "s17n127", "s31n129"
    ],
)
def test_pwa_cute_matches_eager(dtype, n_seq, n_res):
    skip_if_no_cutedsl()
    torch.manual_seed(0)
    c_m, c_z, c_h, num_heads = 64, 128, 32, 8

    layer = PairWeightedAveraging(c_m=c_m,
                                  c_z=c_z,
                                  c_h=c_h,
                                  num_heads=num_heads,
                                  dtype=dtype).cuda()
    with torch.no_grad():
        for lin in (layer.fused_proj_m_g, layer.proj_z, layer.proj_o):
            lin.weight.normal_(0, 1.0 / lin.weight.shape[1]**0.5)

    # Kernel-eligible (single-GPU, H==8, c_h==32, c_m==64) and the op resolves
    # to the CuTe backend on this GPU.
    assert layer._pwa_op_eligible
    assert isinstance(get_pair_weighted_averaging_op(dtype),
                      PairWeightedAveragingCuTe)

    m = torch.randn(1, n_seq, n_res, c_m, dtype=dtype, device="cuda") * 0.5
    z = torch.randn(1, n_res, n_res, c_z, dtype=dtype, device="cuda") * 0.5
    mask = (torch.rand(1, n_res, n_res, device="cuda") < 0.9).to(dtype)

    with torch.inference_mode():
        out_kernel = layer(m, z, mask)
        layer._pwa_op_eligible = False  # force the original eager path
        out_eager = layer(m, z, mask)

    assert out_kernel.shape == out_eager.shape == (1, n_seq, n_res, c_m)
    diff = (out_kernel.float() - out_eager.float()).abs()
    rel_l2 = (diff.norm() / out_eager.float().norm().clamp_min(1e-6)).item()
    assert rel_l2 < 2e-2, (
        f"kernel vs eager rel_l2={rel_l2:.3e} (dtype={dtype}, "
        f"S={n_seq}, N={n_res})")


def test_pair_weighted_averaging_op_selector():
    """The selector returns the CuTe op on CuTeDSL GPUs for fp16/bf16 and the
    vanilla fallback for fp32 / unsupported hardware."""
    op_bf16 = get_pair_weighted_averaging_op(torch.bfloat16)
    op_fp32 = get_pair_weighted_averaging_op(torch.float32)
    if SM_VERSION in _CUTEDSL_SM:
        assert isinstance(op_bf16, PairWeightedAveragingCuTe)
    assert not isinstance(op_fp32, PairWeightedAveragingCuTe)
