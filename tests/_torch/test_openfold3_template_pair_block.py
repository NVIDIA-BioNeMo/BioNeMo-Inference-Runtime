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

# TODO: fix this after finish full openfold3 model
pytestmark = pytest.mark.skip(reason="openfold3")
from tensorrt_llm_lite._utils import str_dtype_to_torch
from tests.common.test_utils.openfold3.create_and_load_weights_from_of3oss import (
    create_template_pair_block_weights_from_of3oss_torch,
    load_template_pair_block_weights_from_of3oss_torch,
)
from test_utils.openfold3.ref_layers_from_oss import RefTemplatePairBlockFromOF3OSS

from tensorrt_bionemo._torch.modules.openfold2.template import \
    TemplatePairBlock
from tensorrt_bionemo.mapping import Mapping


@dataclass(kw_only=True, frozen=True)
class Scenario:
    batch_size: int = 1
    n_res: int = 63
    n_templ: int = 31
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
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    test_dtype = str_dtype_to_torch(sc.dtype)
    device = torch.device('cuda')
    
    c_in = 64
    
    # (1) Input: 
    #   - call RNG early in code block since impl of modules may change
    #   - input t, mask have float32 values.
    t_float = torch.randn(bs,
                    sc.n_templ,
                    sc.n_res,
                    sc.n_res,
                    c_in,
                    dtype=torch.float32).cuda() * 20.0
    mask_float = torch.randint(0,
                         10, (bs, sc.n_templ, sc.n_res, sc.n_res),
                         dtype=torch.float32).cuda()
    mask_float = (mask_float > 8).float()  # float32, 10% of values are 1

    # (2) Reference module creation
    #   - ref module stays in float32
    ref_module = RefTemplatePairBlockFromOF3OSS.load_weights()
    ref_module = ref_module.to(device)
    assert ref_module.tri_mul_out.c_z == c_in

    tri_mul_keys = ["p_in", "g_in", "p_out", "g_out"]
    tri_attn_keys = ["q", "k", "v", "g", "z", "o"]

    tri_mul_out_bias = {k: False for k in tri_mul_keys}
    tri_mul_in_bias = {k: False for k in tri_mul_keys}
    tri_attn_start_bias = {k: False for k in tri_attn_keys}
    tri_attn_end_bias = {k: False for k in tri_attn_keys}

    # (3) Test module creation
    #   - module weights and buffers changed to test_dtype
    module = TemplatePairBlock(
        local_layer_idx=0,
        c_t=ref_module.tri_mul_out.c_z,
        c_hidden_tri_att=ref_module.tri_att_start.c_hidden,
        c_hidden_tri_mul=ref_module.tri_mul_out.c_hidden,
        no_heads=ref_module.tri_att_start.no_heads,
        pair_transition_n=ref_module.pair_transition.n,
        tri_mul_first=ref_module.tri_mul_first,
        transition_type="swiglu",
        tri_mul_out_bias=tri_mul_out_bias,
        tri_mul_in_bias=tri_mul_in_bias,
        tri_attn_start_bias=tri_attn_start_bias,
        tri_attn_end_bias=tri_attn_end_bias,
        dtype=test_dtype,
        mapping=Mapping())

    weights_and_biases = create_template_pair_block_weights_from_of3oss_torch(
        from_ref=ref_module)
    load_template_pair_block_weights_from_of3oss_torch(module, 
                                                        weights_and_biases,
                                                        test_dtype)
    module = module.to(device)
    module.to(test_dtype).eval()
    ref_module.eval()

    # (4) Forward
    with torch.no_grad():
        t_float = t_float.flatten(0, 1)  # fused batch_dim and template_dim
        mask_float = mask_float.flatten(0, 1)  # fused batch_dim and template_dim
        
        t_test_dtype = t_float.to(test_dtype)
        mask_test_dtype = mask_float.to(test_dtype)
        
        output_test_dtype = module(t_test_dtype, mask_test_dtype)        
        ref_output_float = ref_module(t_float, mask_float)

        if test_dtype == torch.float32:
            torch.testing.assert_close(output_test_dtype.float(),
                                        ref_output_float.float(),
                                        atol=1e-3,
                                        rtol=1e-4)
            print(f"after test for test_dtype={test_dtype}")

        else:
            with torch.amp.autocast(device_type="cuda", dtype=test_dtype):
                ref_output = ref_module(t_float.to(test_dtype), mask_float.to(test_dtype))
            
            
                # This is right way to check float16 and bfloat16 accuracy
                diff0_max = torch.max(torch.abs(output_test_dtype.float() - ref_output_float))
                diff0_mean = torch.mean(torch.abs(output_test_dtype.float() - ref_output_float))
                        
                diff1_max = torch.max(torch.abs(ref_output.float() - ref_output_float))
                diff1_mean = torch.mean(torch.abs(ref_output.float() - ref_output_float))

                assert abs(diff0_max - diff1_max) / torch.min(diff0_max, diff1_max) <= 0.5
                assert abs(diff0_mean - diff1_mean) <= 0.2
                print(f"after test for test_dtype={test_dtype}")
