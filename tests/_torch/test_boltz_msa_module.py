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
    create_msa_layer_weights, create_msa_module_weights,
    load_msa_layer_weights_torch, load_msa_module_weights_torch)
from test_utils.boltz.ref_layers import RefMSALayer, RefMSAModule

from tensorrt_bionemo._torch.attention_backend import (AttentionType,
                                                       get_attention_backend)
from tensorrt_bionemo._torch.modules.boltz.trunk import MSALayer, MSAModule
from tensorrt_bionemo.configs import MSAModuleConfig


@dataclass(kw_only=True, frozen=True)
class Scenario:
    torch_dtype: str = "float32"


@pytest.mark.parametrize("sc", [
    Scenario(),
    Scenario(torch_dtype="bfloat16"),
])
def test_msa_layer(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device('cuda')

    ref_mod = RefMSALayer.load_weights()
    ref_mod = ref_mod.to(device)

    weights_and_biases = create_msa_layer_weights(from_ref=ref_mod)

    msa_layer = MSALayer(msa_s=ref_mod.msa_s,
                         token_z=ref_mod.token_z,
                         pairwise_head_width=ref_mod.pairwise_head_width,
                         pairwise_num_heads=ref_mod.pairwise_num_heads,
                         dtype=dtype)
    load_msa_layer_weights_torch(msa_layer, weights_and_biases, dtype=dtype)
    msa_layer.to(device)

    z = torch.randn(bs, 64, 64, ref_mod.token_z, dtype=torch.float32).cuda()
    m = torch.randn(bs, 32, 64, ref_mod.msa_s, dtype=torch.float32).cuda()
    token_mask = torch.randint(0, 2, (bs, 64, 64),
                               dtype=torch.float32).to(device)
    msa_mask = torch.randint(0, 2, (bs, 32, 64),
                             dtype=torch.float32).to(device)

    triangle_metadata_cls = get_attention_backend(
        "VANILLA", AttentionType.TRIANGLE).Metadata
    with torch.inference_mode():
        ref_z_float, ref_m_float = ref_mod(z, m, token_mask, msa_mask)
        z = z.to(dtype)
        m = m.to(dtype)
        token_mask = token_mask.to(dtype)
        msa_mask = msa_mask.to(dtype)

        ref_mod = ref_mod.to(dtype)
        ref_z, ref_m = ref_mod(z, m, token_mask, msa_mask)
        output_z, output_m = msa_layer(z,
                                       m,
                                       token_mask,
                                       msa_mask,
                                       attn_metadata=triangle_metadata_cls())

    assert ref_z.shape == output_z.shape
    assert ref_m.shape == output_m.shape
    if dtype == torch.float32:
        torch.testing.assert_close(ref_z, output_z, atol=1e-3, rtol=1e-4)
        torch.testing.assert_close(ref_m, output_m, atol=1e-3, rtol=1e-4)
    else:
        # This is right way to check float16 and bfloat16 accuracy
        diff0_max = torch.max(torch.abs(output_m.float() -
                                        ref_m_float.float()))
        diff0_mean = torch.mean(
            torch.abs(output_m.float() - ref_m_float.float()))
        diff1_max = torch.max(torch.abs(ref_m.float() - ref_m_float.float()))
        diff1_mean = torch.mean(torch.abs(ref_m.float() - ref_m_float.float()))

        assert abs(diff0_max - diff1_max) / torch.min(diff0_max,
                                                      diff1_max) <= 0.5
        assert abs(diff0_mean - diff1_mean) <= 0.2

        diff0_max = torch.max(torch.abs(output_z.float() -
                                        ref_z_float.float()))
        diff0_mean = torch.mean(
            torch.abs(output_z.float() - ref_z_float.float()))
        diff1_max = torch.max(torch.abs(ref_z.float() - ref_z_float.float()))
        diff1_mean = torch.mean(torch.abs(ref_z.float() - ref_z_float.float()))

        assert abs(diff0_max - diff1_max) / torch.min(diff0_max,
                                                      diff1_max) <= 0.5
        assert abs(diff0_mean - diff1_mean) <= 0.2


@pytest.mark.parametrize("sc", [
    Scenario(),
])
def test_msa_module(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    dtype = str_dtype_to_torch(sc.torch_dtype)
    device = torch.device('cuda')

    ref_mod = RefMSAModule.load_weights()
    ref_mod = ref_mod.to(device)

    weights_and_biases = create_msa_module_weights(from_ref=ref_mod)

    config = MSAModuleConfig(architecture="msa_module",
                             msa_s=ref_mod.msa_s,
                             token_z=ref_mod.token_z,
                             token_s=ref_mod.token_s,
                             msa_blocks=ref_mod.msa_blocks,
                             num_tokens=ref_mod.num_tokens,
                             pairwise_head_width=ref_mod.pairwise_head_width,
                             pairwise_num_heads=ref_mod.pairwise_num_heads,
                             version="v2",
                             dtype=sc.torch_dtype)
    msa_module = MSAModule(config)
    load_msa_module_weights_torch(msa_module, weights_and_biases, dtype=dtype)
    msa_module.to(device)

    B = 1
    N = 117
    N_msa = 83
    token_z = ref_mod.token_z
    token_s = ref_mod.token_s

    z = torch.randn(B, N, N, token_z, dtype=torch.float32).cuda()
    emb = torch.randn(B, N, token_s, dtype=torch.float32).cuda()
    msa = torch.randint(0, 33, (B, N_msa, N), dtype=torch.int64).cuda()
    has_deletion = torch.randint(0, 2, (B, N_msa, N),
                                 dtype=torch.float32).cuda()
    deletion_value = torch.randn(B, N_msa, N, dtype=torch.float32).cuda()
    msa_paired = torch.randint(0, 2, (B, N_msa, N), dtype=torch.float32).cuda()
    msa_mask = torch.randint(0, 2, (B, N_msa, N), dtype=torch.float32).cuda()
    token_pad_mask = torch.randint(0, 2, (B, N), dtype=torch.float32).cuda()
    pair_mask = token_pad_mask[:, :, None] * token_pad_mask[:, None, :]

    triangle_metadata_cls = get_attention_backend(
        "VANILLA", AttentionType.TRIANGLE).Metadata
    with torch.inference_mode():
        ref_z_float = ref_mod(z, emb, msa, has_deletion, deletion_value,
                              msa_paired, msa_mask, token_pad_mask)
        z = z.to(dtype)
        emb = emb.to(dtype)
        has_deletion = has_deletion.to(dtype)
        deletion_value = deletion_value.to(dtype)
        msa_paired = msa_paired.to(dtype)
        msa_mask = msa_mask.to(dtype)
        token_pad_mask = token_pad_mask.to(dtype)

        ref_mod = ref_mod.to(dtype)
        ref_z = ref_mod(z, emb, msa, has_deletion, deletion_value, msa_paired,
                        msa_mask, token_pad_mask)
        output_z = msa_module(z, emb, msa, has_deletion, deletion_value,
                              msa_paired, msa_mask, pair_mask)

    assert ref_z.shape == output_z.shape
    torch.testing.assert_close(ref_z, output_z, atol=1e-3, rtol=1e-4)
