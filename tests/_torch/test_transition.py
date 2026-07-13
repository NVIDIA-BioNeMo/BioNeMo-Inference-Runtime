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

import pytest
import torch
from tensorrt_llm_lite._utils import str_dtype_to_torch
from test_utils.boltz.create_and_load_weights import (
    create_conditioned_transition_block_weights,
    load_conditioned_transition_block_weights_torch)
from test_utils.boltz.ref_layers import RefConditionedTransitionBlock

from tensorrt_bionemo._torch.auto_chunk import ChunkPolicy
from tensorrt_bionemo._torch.layers.transition import (
    ConditionedTransitionBlock, Transition)
from tensorrt_bionemo.mapping import Mapping


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dim_single: int = 768
    dim_single_cond: int = 768
    expansion_factor: int = 2
    torch_dtype: str = "float32"
    seq_len: int = 128


@pytest.mark.parametrize("sc", [
    Scenario(dim_single=768, dim_single_cond=768),
    Scenario(dim_single=768, dim_single_cond=768, torch_dtype="bfloat16"),
    Scenario(dim_single=768, dim_single_cond=768, seq_len=256),
    Scenario(dim_single=768,
             dim_single_cond=768,
             torch_dtype="bfloat16",
             seq_len=256),
])
def test_conditioned_transition_block(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device('cuda')

    ref_cond_trans = RefConditionedTransitionBlock.load_weights()
    ref_cond_trans = ref_cond_trans.to(device)

    weights_and_biases = create_conditioned_transition_block_weights(
        from_ref=ref_cond_trans)

    cond_trans = ConditionedTransitionBlock(
        dim_single=sc.dim_single,
        dim_single_cond=sc.dim_single_cond,
        expansion_factor=sc.expansion_factor,
        dtype=dtype)
    load_conditioned_transition_block_weights_torch(cond_trans,
                                                    weights_and_biases,
                                                    dtype=dtype)
    cond_trans.to(device)

    a = torch.randn(bs, sc.seq_len, sc.dim_single, dtype=torch.float32).cuda()
    s = torch.randn(bs, sc.seq_len, sc.dim_single_cond,
                    dtype=torch.float32).cuda()

    with torch.inference_mode():
        ref_output_float = ref_cond_trans(a, s)
        a = a.to(dtype)
        s = s.to(dtype)
        ref_cond_trans = ref_cond_trans.to(dtype)

        ref_output = ref_cond_trans(a, s)
        output = cond_trans.forward(a, s)

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
                                                      diff1_max) <= 1.0
        assert abs(diff0_mean - diff1_mean) <= 0.2


@pytest.mark.parametrize("mult", [3, 5], ids=["S3", "S5"])
def test_conditioned_transition_block_broadcast(mult: int):
    """Cond input `s` is shared (size-1 multiplicity dim) while `a` varies.

    Exercises the broadcast path of the fused gated-sigmoid output gate:
    ``a`` is ``[B, S, I, d]`` and ``s`` is ``[B, 1, I, d_cond]``.
    """
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs, seq_len, dim = 2, 96, 768
    dtype = torch.bfloat16
    device = torch.device('cuda')

    ref_cond_trans = RefConditionedTransitionBlock.load_weights().to(device)
    weights_and_biases = create_conditioned_transition_block_weights(
        from_ref=ref_cond_trans)

    cond_trans = ConditionedTransitionBlock(dim_single=dim,
                                            dim_single_cond=dim,
                                            expansion_factor=2,
                                            dtype=dtype)
    load_conditioned_transition_block_weights_torch(cond_trans,
                                                    weights_and_biases,
                                                    dtype=dtype)
    cond_trans.to(device)

    a = torch.randn(bs, mult, seq_len, dim, dtype=torch.float32).cuda()
    s = torch.randn(bs, 1, seq_len, dim, dtype=torch.float32).cuda()

    with torch.inference_mode():
        ref_output_float = ref_cond_trans(a, s)
        a = a.to(dtype)
        s = s.to(dtype)
        ref_cond_trans = ref_cond_trans.to(dtype)

        ref_output = ref_cond_trans(a, s)
        output = cond_trans.forward(a, s)

    assert output.shape == a.shape
    diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
    diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
    diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
    diff1_mean = torch.mean(torch.abs(ref_output.float() - ref_output_float))
    assert abs(diff0_max - diff1_max) / torch.min(diff0_max, diff1_max) <= 1.0
    assert abs(diff0_mean - diff1_mean) <= 0.2


@pytest.mark.parametrize("torch_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("chunk_rows", [32, 24], ids=["even", "partial"])
def test_transition_auto_chunk(torch_dtype: str, chunk_rows: int):
    """``Transition.auto_chunk_policy`` row-chunks the pair FFN identically to the dense path.

    The FFN is position-wise, so slicing dim=1 and concatenating is numerically identical (same
    instance dense vs chunked, no golden weights needed). Also checks the rank gate: a rank-3
    single-rep activation is left on the dense path (``min_rank=4``).
    """
    torch.manual_seed(0)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device('cuda')
    dtype = str_dtype_to_torch(torch_dtype)
    dim, hidden, n = 128, 512, 64

    policy = ChunkPolicy(chunk_size=chunk_rows, min_size=1, dim=1, min_rank=4)
    transition = Transition(dim,
                            hidden,
                            dtype=dtype,
                            mapping=Mapping(),
                            auto_chunk_policy=policy).to(device)
    transition.eval()
    # Constructed weights are zero-initialized (production loads them); give them real values so
    # the dense-vs-chunked comparison is meaningful rather than 0 == 0.
    with torch.no_grad():
        for p in transition.parameters():
            p.normal_(mean=0.0, std=0.1)

    tol = (dict(atol=1e-4, rtol=1e-4)
           if dtype == torch.float32 else dict(atol=3e-3, rtol=3e-3))

    # Rank-4 pair activation [B, N, N, dim]: policy trips -> chunked path must match dense.
    z = torch.randn(1, n, n, dim, dtype=dtype, device=device)
    with torch.inference_mode():
        dense = transition._forward_impl(z)
        chunked = transition(z)
    assert chunked.shape == dense.shape
    torch.testing.assert_close(chunked, dense, **tol)

    # Rank-3 single-rep activation [B, N, dim]: rank gate keeps it on the dense path.
    s = torch.randn(1, n, dim, dtype=dtype, device=device)
    assert not policy.should_chunk(s)
    with torch.inference_mode():
        torch.testing.assert_close(transition(s), transition._forward_impl(s),
                                   **tol)
