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
from tensorrt_llm._utils import str_dtype_to_torch
from test_utils.boltz.create_and_load_weights import (
    create_triangle_attention_weights, load_triangle_attention_weights_torch)
from test_utils.boltz.ref_attn import RefTriangleAttention

from tensorrt_bionemo._torch.attention_backend import (AttentionType,
                                                       get_attention_backend)
from tensorrt_bionemo._torch.layers.attention import TriangleAttention
from tensorrt_bionemo.mapping import Mapping


@dataclass(kw_only=True, frozen=True)
class Scenario:
    backend: str
    seq_len: int = 16
    hidden_size: int = 128
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    gating: bool = True
    # FIXME: add chunk_size and chunk_dim back
    # chunk_size: int = None
    # chunk_dim: int = None
    torch_dtype: str = "float32"


@pytest.mark.parametrize("s", [
    Scenario(backend="VANILLA"),
    Scenario(backend="VANILLA", torch_dtype="bfloat16"),
    Scenario(backend="CUEQUIV"),
    Scenario(backend="CUEQUIV", torch_dtype="bfloat16"),
])
def test_triangle_attention_backend(s: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    metadata_cls = get_attention_backend(s.backend,
                                         AttentionType.TRIANGLE).Metadata
    bs = 1
    dtype = str_dtype_to_torch(s.torch_dtype)
    device = torch.device('cuda')

    ref_attn = RefTriangleAttention.load_weights(no_heads=s.num_attention_heads)
    ref_attn = ref_attn.to(device)

    weights_and_biases = create_triangle_attention_weights(from_ref=ref_attn)

    attn = TriangleAttention(layer_idx=0,
                             hidden_size=s.hidden_size,
                             head_dim=s.hidden_size // s.num_attention_heads,
                             num_attention_heads=s.num_attention_heads,
                             num_key_value_heads=s.num_key_value_heads,
                             gating=s.gating,
                             dtype=dtype)
    load_triangle_attention_weights_torch(attn, weights_and_biases, dtype=dtype)
    attn.to(device)
    attn_metadata = metadata_cls(mapping=Mapping())
    hidden_states = torch.randn(bs,
                                s.seq_len,
                                s.seq_len,
                                s.hidden_size,
                                dtype=torch.float32,
                                device=device)
    biases = [
        torch.randn(bs,
                    s.seq_len,
                    1,
                    1,
                    s.seq_len,
                    dtype=torch.float32,
                    device=device),
        torch.randn(bs,
                    s.num_attention_heads,
                    s.seq_len,
                    s.seq_len,
                    dtype=torch.float32,
                    device=device)
    ]

    with torch.inference_mode():
        ref_output_float = ref_attn(hidden_states, hidden_states, biases=biases)
        hidden_states = hidden_states.to(dtype)
        biases = [bias.to(dtype) for bias in biases]
        ref_attn = ref_attn.to(dtype)
        ref_output = ref_attn(hidden_states, hidden_states, biases=biases)
        output = attn(hidden_states, biases=biases, attn_metadata=attn_metadata)

    assert output.shape == ref_output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-3)
    else:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output.float() -
                                          ref_output_float))
        assert abs(diff0_max - diff1_max) / torch.min(diff0_max,
                                                      diff1_max) <= 0.6
        assert abs(diff0_mean - diff1_mean) <= 0.2
