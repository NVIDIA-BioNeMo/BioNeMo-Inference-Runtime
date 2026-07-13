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
    create_outer_product_mean_weights, load_outer_product_mean_weights_torch)
from test_utils.boltz.ref_layers import RefOuterProductMean

from tensorrt_bionemo._torch.auto_chunk import ChunkPolicy
from tensorrt_bionemo._torch.layers.outer_product_mean import OuterProductMean


@dataclass(kw_only=True, frozen=True)
class Scenario:
    torch_dtype: str = "float32"
    # Output token-rows per chunk via a registry-style ChunkPolicy. Row-chunking is numerically
    # identical, so the chunked output must still match the ref.
    policy_chunk: Optional[int] = None


@pytest.mark.parametrize("sc", [
    Scenario(),
    Scenario(torch_dtype="bfloat16"),
    Scenario(policy_chunk=16),
    Scenario(policy_chunk=8),
    Scenario(policy_chunk=12, torch_dtype="bfloat16"),
],
                         ids=[
                             "float32", "bfloat16", "policy_chunk16",
                             "policy_chunk8", "policy_chunk_partial_bf16"
                         ])
def test_outer_product_mean(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device('cuda')

    ref_m = RefOuterProductMean.load_weights()
    ref_m = ref_m.to(device)

    weights_and_biases = create_outer_product_mean_weights(from_ref=ref_m)

    # A low min_size makes the policy trip at the test's token count, so `policy_chunk` scenarios
    # exercise the registry-driven output-row chunking.
    chunk_policy = (ChunkPolicy(chunk_size=sc.policy_chunk, min_size=1)
                    if sc.policy_chunk is not None else None)
    outer_product_mean = OuterProductMean(c_in=ref_m.c_in,
                                          c_hidden=ref_m.c_hidden,
                                          c_out=ref_m.c_out,
                                          dtype=dtype,
                                          chunk_policy=chunk_policy)
    load_outer_product_mean_weights_torch(outer_product_mean,
                                          weights_and_biases,
                                          dtype=dtype)
    outer_product_mean.to(device)

    m = torch.randn(bs, 32, 64, ref_m.c_in, dtype=torch.float32).cuda()
    mask = torch.randint(0, 2, (bs, 32, 64), dtype=torch.float32).to(device)

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
        diff1_mean = torch.mean(
            torch.abs(ref_output.float() - ref_output_float))

        assert abs(diff0_max - diff1_max) / torch.min(diff0_max,
                                                      diff1_max) <= 0.5
        assert abs(diff0_mean - diff1_mean) <= 0.2


@pytest.mark.parametrize("rows", [8, 16, 40, 7],
                         ids=["r8", "r16", "r40_all", "r7_partial"])
def test_outer_product_mean_chunk_matches_dense(rows: int):
    """Output token-row chunking (registry policy) matches the dense path.

    Same instance dense vs chunked (no golden weights needed); each output row ``i`` depends only on
    ``a[:, :, i]``, so slicing the output token dim and concatenating is numerically identical.
    """
    torch.manual_seed(0)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device('cuda')

    opm = OuterProductMean(c_in=32, c_hidden=8, c_out=16,
                           dtype=torch.float32).to(device)
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
