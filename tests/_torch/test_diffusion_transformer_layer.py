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
from test_utils.create_and_load_weights import (
    create_diffusion_transformer_layer_weights,
    load_diffusion_transformer_layer_weights_torch)
from test_utils.ref_layers import RefDiffusionTransformerLayer

from tensorrt_bionemo._torch.attention_backend.utils import \
    get_attention_backend
from tensorrt_bionemo._torch.layers.transformers import \
    DiffusionTransformerLayer


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dim: int = 768
    dim_single_cond: int = 768
    torch_dtype: str = "float32"
    seq_len: int = 128
    num_heads: int = 16
    dim_pairwise: int = 128


@pytest.mark.parametrize("sc", [
    Scenario(dim=768, dim_single_cond=768),
    Scenario(dim=768, dim_single_cond=768, torch_dtype="bfloat16"),
])
def test_diffusion_transformer_layer(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device('cuda')

    ref_module = RefDiffusionTransformerLayer.load_weights()
    ref_module = ref_module.to(device)

    weights_and_biases = create_diffusion_transformer_layer_weights(
        from_ref=ref_module)

    attn_pairwise_metadata_cls = get_attention_backend("VANILLA").Metadata
    module = DiffusionTransformerLayer(
        layer_idx=0,
        num_heads=ref_module.pair_bias_attn.num_heads,
        dim=sc.dim,
        dim_single_cond=sc.dim_single_cond,
        dim_pairwise=sc.dim_pairwise,
        with_pair_bias_cache=True,
        dtype=dtype)
    load_diffusion_transformer_layer_weights_torch(module,
                                                   weights_and_biases,
                                                   dtype=dtype)
    module.to(device)

    a = torch.randn(bs, sc.seq_len, sc.dim, dtype=torch.float32).cuda()
    s = torch.randn(bs, sc.seq_len, sc.dim_single_cond,
                    dtype=torch.float32).cuda()
    z = torch.randn(bs,
                    sc.seq_len,
                    sc.seq_len,
                    sc.dim_pairwise,
                    dtype=torch.float32).cuda()
    mask = torch.randn(bs, sc.seq_len, dtype=torch.float32).cuda()

    with torch.inference_mode():
        ref_output_float = ref_module(a, s, z, mask)
        a = a.to(dtype)
        s = s.to(dtype)
        z = z.to(dtype)
        mask = mask.to(dtype)
        ref_module = ref_module.to(dtype)

        ref_output = ref_module(a, s, z, mask)
        output = module.forward(
            a,
            s,
            z,
            mask,
            attn_metadata=attn_pairwise_metadata_cls(bias_cache={}))

    assert ref_output.shape == output.shape
    if dtype == torch.float32:
        torch.testing.assert_close(ref_output, output, atol=1e-3, rtol=1e-4)
    else:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output.float() - ref_output_float))
        diff0_mean = torch.mean(torch.abs(output.float() - ref_output_float))
        diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
        diff1_mean = torch.mean(torch.abs(ref_output.float() -
                                          ref_output_float))

        assert abs(diff0_max - diff1_max) / torch.min(diff0_max,
                                                      diff1_max) <= 0.5
        assert abs(diff0_mean - diff1_mean) <= 0.2
