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

import numpy as np
import pytest
import torch
from test_utils.ref_attn import plain_triangle_mha

from tensorrt_bionemo._torch.attention_backend.trifast import (
    TrifastAttention, TrifastAttentionMetadata)


@pytest.mark.parametrize("seq_len", [64, 128, 192, 256])
@pytest.mark.parametrize("i_factor", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_trifast_attention_for_triangle(seq_len, i_factor, dtype):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    num_heads = 8
    head_dim = 32
    layer_idx = 0
    bs = 1

    q = torch.randn(bs, seq_len // i_factor, seq_len,
                    num_heads * head_dim).cuda()
    k = torch.randn(bs, seq_len // i_factor, seq_len,
                    num_heads * head_dim).cuda()
    v = torch.randn(bs, seq_len // i_factor, seq_len,
                    num_heads * head_dim).cuda()
    # Note: trifast use the original mask, but plain_attn use original_mask*neg_inf(dtype) as mask bias
    original_mask = torch.randint(
        0, 2, (bs, seq_len // i_factor, 1, 1, seq_len)).cuda()

    biases = [
        original_mask.to(dtype) * torch.finfo(dtype).min,
        torch.randn(bs, num_heads, seq_len, seq_len).cuda()
    ]
    trifast_attn = TrifastAttention(layer_idx,
                                    num_heads,
                                    head_dim,
                                    num_kv_heads=num_heads)
    metadata = TrifastAttentionMetadata()
    metadata.closest_n = 2**int(np.ceil(np.log2(seq_len)))
    trifast_out = trifast_attn.forward(
        q.to(dtype),
        k.to(dtype),
        v.to(dtype),
        # biases=[original_mask.bool(), biases[1].to(dtype)],
        biases=[biases[0], biases[1].to(dtype)],
        metadata=metadata)
    plain_out = plain_triangle_mha(q, k, v, num_heads, head_dim,
                                   [biases[0].float(), biases[1].float()])

    assert trifast_out.shape == plain_out.shape
    if dtype == torch.float32:
        torch.testing.assert_close(trifast_out, plain_out)
    elif dtype == torch.bfloat16:
        torch.testing.assert_close(trifast_out.float(),
                                   plain_out,
                                   atol=5e-2,
                                   rtol=1e-4)
