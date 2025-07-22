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

import numpy as np
import pytest
import torch
from tensorrt_llm._utils import str_dtype_to_torch
from test_utils.create_and_load_weights import (
    create_pairformer_layer_weights, load_pairformer_layer_weights_torch)
from test_utils.ref_layers import RefPairformerLayer

from tensorrt_bionemo._torch.attention_backend import (AttentionType,
                                                       get_attention_backend)
from tensorrt_bionemo._torch.layers.transformers import PairformerLayerV1
from tensorrt_bionemo.mapping import Mapping


@dataclass(kw_only=True, frozen=True)
class Scenario:
    triangle_attn_backend: str
    pairwise_attn_backend: str
    seq_len: int = 32
    chunk_size: int = None
    torch_dtype: str = "float32"


@pytest.mark.parametrize("sc", [
    Scenario(triangle_attn_backend="VANILLA", pairwise_attn_backend="VANILLA"),
    Scenario(triangle_attn_backend="VANILLA",
             pairwise_attn_backend="VANILLA",
             torch_dtype="bfloat16"),
    Scenario(triangle_attn_backend="TRIFAST", pairwise_attn_backend="VANILLA"),
    Scenario(triangle_attn_backend="TRIFAST",
             pairwise_attn_backend="VANILLA",
             torch_dtype="bfloat16"),
    Scenario(triangle_attn_backend="CUEQUIV", pairwise_attn_backend="VANILLA"),
    Scenario(triangle_attn_backend="CUEQUIV",
             pairwise_attn_backend="VANILLA",
             torch_dtype="bfloat16"),
])
def test_pairformer_layer(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    pairwise_metadata_cls = get_attention_backend(
        sc.pairwise_attn_backend, AttentionType.PAIRWISE).Metadata
    triangle_metadata_cls = get_attention_backend(
        sc.triangle_attn_backend, AttentionType.TRIANGLE).Metadata
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device('cuda')

    ref_layer = RefPairformerLayer.load_weights()
    ref_layer = ref_layer.to(device)

    weights_and_biases = create_pairformer_layer_weights(from_ref=ref_layer)

    layer = PairformerLayerV1(
        layer_idx=0,
        token_s=ref_layer.token_s,
        token_z=ref_layer.token_z,
        num_heads=ref_layer.num_heads,
        pairwise_head_width=ref_layer.pairwise_head_width,
        pairwise_num_heads=ref_layer.pairwise_num_heads,
        dtype=dtype,
        triangle_attn_backend=sc.triangle_attn_backend,
        pairwise_attn_backend=sc.pairwise_attn_backend,
        skip_create_weights=False,
    )
    layer.to(device)
    load_pairformer_layer_weights_torch(layer, weights_and_biases, dtype)

    s = torch.rand(bs, sc.seq_len, ref_layer.token_s).to(device)
    z = torch.rand(bs, sc.seq_len, sc.seq_len, ref_layer.token_z).to(device)
    mask = torch.randn(bs, sc.seq_len).to(device)
    # pair_mask = torch.randn(bs, sc.seq_len, sc.seq_len).to(device)
    pair_mask = torch.randint(0,
                              2, (bs, sc.seq_len, sc.seq_len),
                              dtype=torch.float32).to(device)

    attn_metadatas = {
        "triangle_attn": triangle_metadata_cls(mapping=Mapping()),
        "pairwise_attn": pairwise_metadata_cls(mapping=Mapping()),
    }
    if sc.triangle_attn_backend == "TRIFAST":
        attn_metadatas["triangle_attn"].closest_n = 2**int(
            np.ceil(np.log2(sc.seq_len)))

    with torch.inference_mode():
        ref_s_float, ref_z_float = ref_layer(s, z, mask, pair_mask)
        s = s.to(dtype)
        z = z.to(dtype)
        mask = mask.to(dtype)
        pair_mask = pair_mask.to(dtype)
        # ref_layer = ref_layer.to(dtype)
        # cast all modules to dtype, except norm_out in tri_mul
        for name, module in ref_layer.named_modules():
            if name.startswith("tri_attn_start.") or \
                name.startswith("tri_attn_end.") or \
                name.startswith("transition_s.") or \
                name.startswith("transition_z.") or \
                name.startswith("attention."):
                module.to(dtype)
            elif name.startswith("tri_mul_out.") or name.startswith(
                    "tri_mul_in."):
                if not "norm_out" in name and not "p_out" in name and not "g_out" in name:
                    module.to(dtype)
                else:
                    module.float()

        ref_s, ref_z = ref_layer(s, z, mask, pair_mask)
        output_s, output_z = layer(s,
                                   z,
                                   mask,
                                   pair_mask,
                                   attn_metadatas=attn_metadatas)

    assert ref_s.shape == output_s.shape
    assert ref_z.shape == output_z.shape
    if dtype == torch.float32:
        torch.testing.assert_close(ref_s, output_s, atol=1e-3, rtol=1e-4)
        torch.testing.assert_close(ref_z, output_z, atol=1e-3, rtol=1e-4)
    else:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output_s.float() - ref_s_float))
        diff0_mean = torch.mean(torch.abs(output_s.float() - ref_s_float))
        diff1_max = torch.max(torch.abs(ref_s.float() - ref_s_float))
        diff1_mean = torch.mean(torch.abs(ref_s.float() - ref_s_float))
        assert abs(diff0_max - diff1_max) / torch.min(diff0_max,
                                                      diff1_max) <= 0.5
        assert abs(diff0_mean - diff1_mean) <= 0.2

        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output_z.float() - ref_z_float))
        diff0_mean = torch.mean(torch.abs(output_z.float() - ref_z_float))
        diff1_max = torch.max(torch.abs(ref_z.float() - ref_z_float))
        diff1_mean = torch.mean(torch.abs(ref_z.float() - ref_z_float))

        assert abs(diff0_max - diff1_max) / torch.min(diff0_max,
                                                      diff1_max) <= 0.5
        assert abs(diff0_mean - diff1_mean) <= 0.2
