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
from copy import deepcopy
from dataclasses import dataclass

import pytest
import torch
import transformers
from test_utils.ref_attn import RefPairwiseSelfAttention

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.model_config import ModelConfig
from tensorrt_bionemo._torch.modules.attention import SelfAttentionPairBias

_MOCK_MODEL_CONFIG = {
    "architectures": ["self-attention-pair-bias"],
    "torch_dtype": "float32",
}


@dataclass(kw_only=True, frozen=True)
class Scenario:
    backend: str
    seq_len: int = 16
    c_s: int = 384
    c_z: int = 128
    num_attention_heads: int = 16
    chunk_size: int = None
    chunk_dim: int = None
    torch_dtype: str = "float32"


@pytest.mark.parametrize("sc", [
    Scenario(backend="VANILLA"),
    Scenario(backend="VANILLA", torch_dtype="bfloat16"),
    Scenario(backend="VANILLA", torch_dtype="float16"),
])
def test_pairwise_attention_backend(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    metadata_cls = get_attention_backend(sc.backend).Metadata
    config_dict = deepcopy(_MOCK_MODEL_CONFIG)
    config_dict["torch_dtype"] = sc.torch_dtype
    model_config = ModelConfig(
        pretrained_config=transformers.PretrainedConfig.from_dict(config_dict),
        attn_backend=sc.backend,
    )
    dtype = model_config.pretrained_config.torch_dtype
    device = torch.device('cuda')

    ref_attn = RefPairwiseSelfAttention.load_weights(
        num_heads=sc.num_attention_heads)
    ref_attn.to(device)
    ref_attn = ref_attn

    q_proj_weights = [{
        "weight": ref_attn.proj_q.weight.data.to(dtype),
        "bias": ref_attn.proj_q.bias.data.to(dtype)
    }]
    k_weights = [{
        "weight": ref_attn.proj_k.weight.data.to(dtype),
        "bias": None
    }]
    v_weights = [{
        "weight": ref_attn.proj_v.weight.data.to(dtype),
        "bias": None
    }]
    o_proj_weights = [{
        "weight": ref_attn.proj_o.weight.data.to(dtype),
        "bias": None
    }]
    g_proj_weights = [{
        "weight": ref_attn.proj_g.weight.data.to(dtype),
        "bias": None
    }]

    z_1_proj_weights = [{
        "weight": ref_attn.proj_z[1].weight.data.to(dtype),
        "bias": None
    }]

    attn = SelfAttentionPairBias(layer_idx=0,
                                 c_s=sc.c_s,
                                 c_z=sc.c_z,
                                 num_heads=sc.num_attention_heads,
                                 dtype=dtype,
                                 config=model_config,
                                 initial_norm=True)
    if attn.norm_s:
        attn.norm_s.weight.data.copy_(ref_attn.norm_s.weight.data)
        attn.norm_s.bias.data.copy_(ref_attn.norm_s.bias.data)
    attn.proj_k.load_weights(k_weights)
    attn.proj_v.load_weights(v_weights)
    attn.proj_q.load_weights(q_proj_weights)
    attn.proj_o.load_weights(o_proj_weights)
    attn.proj_g.load_weights(g_proj_weights)
    attn.proj_z[0].weight.data.copy_(ref_attn.proj_z[0].weight.data)
    attn.proj_z[0].bias.data.copy_(ref_attn.proj_z[0].bias.data)
    attn.proj_z[1].load_weights(z_1_proj_weights)
    attn.to(device)

    attn_metadata = metadata_cls(chunk_size=sc.chunk_size,
                                 chunk_dim=sc.chunk_dim)
    s = torch.randn(1, sc.seq_len, sc.c_s).to(device)
    z = torch.randn(1, sc.seq_len, sc.seq_len, sc.c_z).to(device)
    mask = torch.randn(1, sc.seq_len).to(device)

    with torch.inference_mode():
        ref_output_float = ref_attn(s, z, mask)
        s = s.to(dtype)
        z = z.to(dtype)
        mask = mask.to(dtype)
        ref_attn = ref_attn.to(dtype)

        ref_output = ref_attn(s, z, mask)
        output = attn.forward(s, z, mask, attn_metadata)

    assert ref_output.shape == output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(ref_output, output, atol=1e-2, rtol=1e-3)
    else:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output.float() -
                                          ref_output_float))

        if dtype == torch.bfloat16:
            assert abs(diff0_max - diff1_max) <= 3
            assert abs(diff0_mean - diff1_mean) <= 0.2
        else:  # fp16 return NaN for ref
            assert diff0_max <= 0.2
            assert diff0_mean <= 0.01
