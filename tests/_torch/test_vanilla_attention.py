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

import pytest
import torch
from test_utils.boltz.ref_attn import plain_mha

from tensorrt_bionemo._torch.attention_backend.interface import \
    AttentionMetadata
from tensorrt_bionemo._torch.attention_backend.vanilla import (
    VanillaPairwiseAttention, VanillaTriangleAttention)


@pytest.mark.parametrize("seq_len", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vanilla_attention_for_triangle(seq_len, dtype):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    num_heads = 8
    head_dim = 32
    layer_idx = 0
    bs = 1

    q = torch.randn(bs, seq_len, seq_len, num_heads * head_dim).cuda().to(dtype)
    k = torch.randn(bs, seq_len, seq_len, num_heads * head_dim).cuda().to(dtype)
    v = torch.randn(bs, seq_len, seq_len, num_heads * head_dim).cuda().to(dtype)

    vanilla_attn = VanillaTriangleAttention(layer_idx,
                                            num_heads,
                                            head_dim,
                                            num_kv_heads=num_heads)

    biases = [
        torch.randn(bs, seq_len, 1, 1, seq_len).cuda().to(dtype),
        torch.randn(bs, 1, num_heads, seq_len, seq_len).cuda().to(dtype)
    ]
    metadata = AttentionMetadata()
    vanilla_out = vanilla_attn.forward(q,
                                       k,
                                       v,
                                       biases=[biases[0], biases[1].squeeze(1)],
                                       metadata=metadata)
    assert vanilla_out.shape == (bs, seq_len, seq_len, num_heads, head_dim)
    plain_out = plain_mha(q, k, v, num_heads, head_dim, biases)
    assert vanilla_out.shape == plain_out.shape
    if dtype == torch.float32:
        torch.testing.assert_close(vanilla_out, plain_out)
    elif dtype == torch.bfloat16:
        torch.testing.assert_close(vanilla_out, plain_out, atol=5e-2, rtol=1e-4)


@pytest.mark.parametrize("batch_size", [16, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vanilla_attention_for_pairwise(batch_size, dtype):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    num_heads = 8
    head_dim = 32
    layer_idx = 0
    q_size = 32
    kv_size = 128

    q = torch.randn(batch_size, q_size, num_heads * head_dim).to(dtype)
    k = torch.randn(batch_size, kv_size, num_heads * head_dim).to(dtype)
    v = torch.randn(batch_size, kv_size, num_heads * head_dim).to(dtype)

    vanilla_attn = VanillaPairwiseAttention(layer_idx,
                                            num_heads,
                                            head_dim,
                                            num_kv_heads=num_heads)

    biases = [
        torch.randn(batch_size, 1, 1, kv_size),
        torch.randn(batch_size, num_heads, q_size, kv_size)
    ]
    metadata = AttentionMetadata()
    vanilla_out = vanilla_attn.forward(q,
                                       k,
                                       v,
                                       biases=biases,
                                       metadata=metadata)
    assert vanilla_out.shape == (batch_size, q_size, num_heads, head_dim)
    plain_out = plain_mha(q, k, v, num_heads, head_dim, biases)
    assert vanilla_out.shape == plain_out.shape

    if dtype == torch.float32:
        torch.testing.assert_close(vanilla_out, plain_out)
    elif dtype == torch.bfloat16:
        torch.testing.assert_close(vanilla_out, plain_out, atol=5e-2, rtol=1e-4)
