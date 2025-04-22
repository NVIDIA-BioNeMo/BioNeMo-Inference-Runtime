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
from test_utils.ref_attn import RefPairwiseSelfAttention

import tensorrt_bionemo

SelfPairwiseTestScenario = namedtuple("SelfPairwiseTestScenario", [
    "batch_size", "seq_len", "c_s", "c_z", "num_attention_heads",
    "vanilla_attn_precision", "dtype"
])


@pytest.mark.parametrize("sc", [
    SelfPairwiseTestScenario(batch_size=1,
                             seq_len=5,
                             c_s=384,
                             c_z=128,
                             num_attention_heads=16,
                             vanilla_attn_precision="float32",
                             dtype="float32"),
    SelfPairwiseTestScenario(batch_size=2,
                             seq_len=15,
                             c_s=96,
                             c_z=64,
                             num_attention_heads=8,
                             vanilla_attn_precision="float32",
                             dtype="float32"),
    SelfPairwiseTestScenario(batch_size=3,
                             seq_len=30,
                             c_s=384,
                             c_z=128,
                             num_attention_heads=32,
                             vanilla_attn_precision="float32",
                             dtype="float32"),
])
def test_self_pairwise_attention(sc: SelfPairwiseTestScenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

    mean = 0.0
    std_dev = 1 if sc.dtype == "float32" else 0.005
    torch_dtype = str_dtype_to_torch(sc.dtype)

    s = torch.empty(size=[sc.batch_size, sc.seq_len, sc.c_s],
                    dtype=torch_dtype,
                    device="cuda",
                    requires_grad=False)
    s.normal_(mean=mean, std=std_dev)
    z = torch.empty(size=[sc.batch_size, sc.seq_len, sc.seq_len, sc.c_z],
                    dtype=torch_dtype,
                    device="cuda",
                    requires_grad=False)
    z.normal_(mean=mean, std=std_dev)
    mask = torch.empty(size=[sc.batch_size, sc.seq_len],
                       dtype=torch_dtype,
                       device="cuda",
                       requires_grad=False)
    mask.normal_(mean=mean, std=std_dev)

    weights_and_biases = \
        create_self_pairwise_attention_weights(sc.c_s, sc.c_z, sc.num_attention_heads, torch_dtype)

    # construct trt network
    builder = tensorrt_llm.Builder()
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()
    with tensorrt_llm.net_guard(net):
        trt_s = Tensor(name='input_s',
                       shape=s.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(sc.dtype))
        trt_z = Tensor(name='input_z',
                       shape=z.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(sc.dtype))
        trt_mask = Tensor(name='mask',
                          shape=mask.shape,
                          dtype=tensorrt_llm.str_dtype_to_trt(sc.dtype))

        attn_layer = tensorrt_bionemo.layers.SelfAttentionPairBias(
            c_s=sc.c_s,
            c_z=sc.c_z,
            num_heads=sc.num_attention_heads,
            initial_norm=True,
            local_layer_idx=0)
        load_self_pairwise_attention_weights_trt(attn_layer, weights_and_biases)

        attention_params = tensorrt_bionemo.layers.attention.AttentionParams(
            vanilla_attn_precision=sc.vanilla_attn_precision)
        output = attn_layer(trt_s,
                            trt_z,
                            mask=trt_mask,
                            attention_params=attention_params)
        output.mark_output("output", tensorrt_llm.str_dtype_to_trt(sc.dtype))
    builder_config = builder.create_builder_config(
        name="self_pairwise_attention", precision=sc.dtype)

    # Build engine
    engine_buffer = builder.build_engine(net, builder_config)
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine_buffer)
    stream = torch.cuda.current_stream().cuda_stream

    # Verify results
    inputs = {'input_s': s, 'input_z': z, 'mask': mask}
    outputs = {
        'output':
        torch.empty(s.shape,
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(sc.dtype),
                    device="cuda")
    }
    session.run(inputs=inputs, outputs=outputs, stream=stream)
    torch.cuda.synchronize()

    # Verify result
    ref_attn = RefPairwiseSelfAttention(c_s=sc.c_s,
                                        c_z=sc.c_z,
                                        num_heads=sc.num_attention_heads,
                                        inf=1e6,
                                        initial_norm=True)
    ref_attn.to("cuda", dtype=torch_dtype)

    load_self_pairwise_attention_weights_ref_torch(ref_attn, weights_and_biases)

    with torch.inference_mode():
        ref_output = ref_attn(s, z, mask)

    trt_output = outputs['output']
    torch.testing.assert_close(trt_output, ref_output, atol=1e-3, rtol=1e-4)
