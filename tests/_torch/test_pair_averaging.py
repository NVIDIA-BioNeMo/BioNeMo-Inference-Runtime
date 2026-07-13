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
from tensorrt_bionemo._torch.layers.pair_averaging import PairWeightedAveraging


@dataclass(kw_only=True, frozen=True)
class Scenario:
    torch_dtype: str = "float32"
    # Sequence(S)-rows per chunk via a registry-style ChunkPolicy (None -> dense path). Row-chunking
    # over S is numerically identical, so the chunked output must still match the golden ref.
    chunk: Optional[int] = None


@pytest.mark.parametrize("sc", [
    Scenario(),
    Scenario(torch_dtype="bfloat16"),
    Scenario(chunk=1),
    Scenario(chunk=3),
    Scenario(chunk=2, torch_dtype="bfloat16"),
],
                         ids=[
                             "dense_fp32", "dense_bf16", "chunk1_fp32",
                             "chunk3_partial_fp32", "chunk2_bf16"
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

    m = torch.randn(bs, 32, 64, ref_m.c_m, dtype=torch.float32).cuda()
    z = torch.randn(bs, 64, 64, ref_m.c_z, dtype=torch.float32).cuda()
    mask = torch.randn(bs, 64, 64, dtype=torch.float32).cuda()

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
