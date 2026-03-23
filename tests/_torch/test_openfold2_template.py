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
from test_utils.openfold.create_and_load_weights import (
    create_template_pair_stack_block_weights,
    create_template_pointwise_attention_weights,
    load_template_pair_stack_block_weights_torch,
    load_template_pointwise_attention_weights_torch)
from test_utils.openfold.ref_layers import (RefTemplatePairStackBlock,
                                            RefTemplatePointwiseAttention)

from tensorrt_bionemo._torch.modules.openfold2.template import (
    TemplatePairBlock, TemplatePointwiseAttention)
from tensorrt_bionemo.mapping import Mapping


@dataclass(kw_only=True, frozen=True)
class Scenario:
    batch_size: int = 1
    n_res: int = 64
    n_templ: int = 32
    dtype: str = "float32"
    triangle_attn_backend: str = "VANILLA"


@pytest.mark.parametrize(
    "sc", [
        Scenario(triangle_attn_backend="VANILLA"),
        Scenario(triangle_attn_backend="CUEQUIV"),
        Scenario(triangle_attn_backend="VANILLA", dtype="bfloat16"),
        Scenario(triangle_attn_backend="CUEQUIV", dtype="bfloat16"),
    ],
    ids=["vanilla", "cuequiv", "vanilla_bf16", "cuequiv_bf16"])
def test_template_pair_stack_block(sc: Scenario):
    torch.manual_seed(123)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    torch_dtype = str_dtype_to_torch(sc.dtype)
    device = torch.device('cuda')

    ref_module = RefTemplatePairStackBlock.load_weights()
    ref_module = ref_module.to(device)

    weights_and_biases = create_template_pair_stack_block_weights(
        from_ref=ref_module)
    t = torch.randn(bs,
                    sc.n_templ,
                    sc.n_res,
                    sc.n_res,
                    ref_module.c_t,
                    dtype=torch.float32).cuda()
    mask = torch.randint(0,
                         2, (bs, sc.n_templ, sc.n_res, sc.n_res),
                         dtype=torch.float32).cuda()

    module = TemplatePairBlock(
        local_layer_idx=0,
        c_t=ref_module.c_t,
        c_hidden_tri_att=ref_module.c_hidden_tri_att,
        c_hidden_tri_mul=ref_module.c_hidden_tri_mul,
        no_heads=ref_module.no_heads,
        pair_transition_n=ref_module.pair_transition_n,
        tri_mul_first=ref_module.tri_mul_first,
        dtype=torch_dtype,
        mapping=Mapping())

    load_template_pair_stack_block_weights_torch(module, weights_and_biases)
    module = module.to(device)
    module.eval()
    ref_module.eval()

    with torch.no_grad():
        t = t.flatten(0, 1)  # fused batch_dim and template_dim
        mask = mask.flatten(0, 1)  # fused batch_dim and template_dim
        ref_t = ref_module(t, mask)

        t = t.to(torch_dtype)
        mask = mask.to(torch_dtype)
        output_t = module(t, mask)

    if torch_dtype == torch.float32:
        torch.testing.assert_close(output_t.float(),
                                   ref_t.float(),
                                   atol=1e-3,
                                   rtol=1e-4)
    else:
        torch.testing.assert_close(output_t.float(),
                                   ref_t.float(),
                                   atol=1e-1,
                                   rtol=1e-1)


@pytest.mark.parametrize("sc", [Scenario(triangle_attn_backend="VANILLA")])
def test_template_pointwise_attention(sc: Scenario):
    torch.manual_seed(123)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    torch_dtype = str_dtype_to_torch(sc.dtype)
    device = torch.device('cuda')

    ref_module = RefTemplatePointwiseAttention.load_weights()
    ref_module = ref_module.to(device)
    ref_module.eval()

    weights_and_biases = create_template_pointwise_attention_weights(
        from_ref=ref_module)
    t = torch.randn(bs,
                    sc.n_templ,
                    sc.n_res,
                    sc.n_res,
                    ref_module.c_t,
                    dtype=torch.float32).cuda()
    z = torch.randn(bs,
                    sc.n_res,
                    sc.n_res,
                    ref_module.c_z,
                    dtype=torch.float32).cuda()
    template_mask = torch.randint(0, 2, (bs, sc.n_templ),
                                  dtype=torch.float32).cuda()

    module = TemplatePointwiseAttention(c_t=ref_module.c_t,
                                        c_z=ref_module.c_z,
                                        c_hidden=ref_module.c_hidden,
                                        no_heads=ref_module.no_heads,
                                        inf=ref_module.inf,
                                        dtype=torch_dtype,
                                        mapping=Mapping())

    load_template_pointwise_attention_weights_torch(module, weights_and_biases)
    module = module.to(device)
    module.eval()
    ref_module.eval()

    with torch.no_grad():
        ref_z = ref_module(t, z, template_mask)
        output_z = module(t, z, template_mask)

    torch.testing.assert_close(output_z.float(),
                               ref_z.float(),
                               atol=1e-3,
                               rtol=1e-4)
