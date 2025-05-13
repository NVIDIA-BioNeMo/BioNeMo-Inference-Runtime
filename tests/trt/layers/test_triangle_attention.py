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
from collections import namedtuple

import pytest
import tensorrt_llm
import torch
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch
from test_utils.create_and_load_weights import *
from test_utils.ref_attn import RefTriangleAttention

import tensorrt_bionemo

TriAttnTestScenario = namedtuple(
    "TriAttnTestScenario",
    ["bs", "si", "sj", "hidden_size", "num_attention_heads", "dtype"])


@pytest.mark.parametrize("sc", [
    TriAttnTestScenario(bs=1,
                        si=5,
                        sj=5,
                        hidden_size=32,
                        num_attention_heads=16,
                        dtype="float32"),
    TriAttnTestScenario(bs=2,
                        si=6,
                        sj=12,
                        hidden_size=48,
                        num_attention_heads=8,
                        dtype="float32"),
    TriAttnTestScenario(bs=3,
                        si=100,
                        sj=200,
                        hidden_size=64,
                        num_attention_heads=8,
                        dtype="float32"),
])
def test_triangle_attention(sc: TriAttnTestScenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

    c_q = c_k = c_v = sc.hidden_size * sc.num_attention_heads

    mean = 0.0
    std_dev = 1 if sc.dtype == "float32" else 0.05
    torch_dtype = str_dtype_to_torch(sc.dtype)
    hidden_states = torch.empty(size=[sc.bs, sc.si, sc.sj, c_q],
                                dtype=torch_dtype,
                                device="cuda",
                                requires_grad=False)
    hidden_states.normal_(mean=mean, std=std_dev)

    mask_bias = torch.empty(size=[sc.bs, sc.si, 1, 1, sc.sj],
                            dtype=torch_dtype,
                            device="cuda",
                            requires_grad=False)
    mask_bias.normal_(mean=mean, std=std_dev)
    triangle_bias = torch.empty(
        size=[sc.bs, sc.num_attention_heads, sc.sj, sc.sj],
        dtype=torch_dtype,
        device="cuda",
        requires_grad=False)
    triangle_bias.normal_(mean=mean, std=std_dev)

    weights_and_biases = \
        create_triangle_attention_weights(c_q, c_k, c_v, torch_dtype)

    # construct trt network
    builder = tensorrt_llm.Builder()
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()
    with tensorrt_llm.net_guard(net):
        trt_hidden_states = Tensor(name='hidden_states',
                                   shape=hidden_states.shape,
                                   dtype=tensorrt_llm.str_dtype_to_trt(
                                       sc.dtype))
        trt_mask_bias = Tensor(name="mask_bias",
                               shape=mask_bias.shape,
                               dtype=tensorrt_llm.str_dtype_to_trt(sc.dtype))
        trt_triangle_bias = Tensor(name="triangle_bias",
                                   shape=triangle_bias.shape,
                                   dtype=tensorrt_llm.str_dtype_to_trt(
                                       sc.dtype))
        attn_layer = tensorrt_bionemo._trt.layers.TriangleAttention(
            hidden_size=c_q,
            num_attention_heads=sc.num_attention_heads,
            num_kv_heads=sc.num_attention_heads,
            local_layer_idx=0,
            gating=True)
        load_triangle_attention_weights_trt(attn_layer, weights_and_biases)

        input_tensor = trt_hidden_states
        attention_params = tensorrt_bionemo._trt.layers.attention.AttentionParams(
        )
        output = attn_layer(input_tensor,
                            biases=[trt_mask_bias, trt_triangle_bias],
                            attention_params=attention_params)
        output.mark_output("output", tensorrt_llm.str_dtype_to_trt(sc.dtype))

    builder_config = builder.create_builder_config(name="tri_attention",
                                                   precision=sc.dtype)

    # Build engine
    engine_buffer = builder.build_engine(net, builder_config)
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine_buffer)
    stream = torch.cuda.current_stream().cuda_stream

    # Verify result
    inputs = {
        'hidden_states': hidden_states,
        'mask_bias': mask_bias,
        'triangle_bias': triangle_bias
    }
    outputs = {
        'output':
        torch.empty(hidden_states.shape,
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(sc.dtype),
                    device="cuda")
    }
    session.run(inputs=inputs, outputs=outputs, stream=stream)
    torch.cuda.synchronize()

    ref_attn = RefTriangleAttention(c_q,
                                    c_k,
                                    c_v,
                                    sc.hidden_size,
                                    sc.num_attention_heads,
                                    gating=True)
    ref_attn.to("cuda", dtype=torch_dtype)

    load_triangle_attention_weights_ref_torch(ref_attn, weights_and_biases)

    with torch.inference_mode():
        ref_output = ref_attn(hidden_states, hidden_states,
                              [mask_bias, triangle_bias])

    trt_output = outputs['output']
    torch.testing.assert_close(trt_output, ref_output, atol=1e-4, rtol=1e-3)
