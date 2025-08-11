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
import tensorrt_llm
import torch
from tensorrt_llm import Tensor, str_dtype_to_torch, str_dtype_to_trt
from test_utils.boltz.create_and_load_weights import *
from test_utils.boltz.ref_layers import RefConditionedTransitionBlock

from tensorrt_bionemo._trt.layers.transition import ConditionedTransitionBlock


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dim_single: int = 768
    dim_single_cond: int = 768
    expansion_factor: int = 2
    dtype: str = "float32"
    seq_len: int = 128
    batch_size: int = 1


@pytest.mark.parametrize("sc", [
    Scenario(dim_single=768, dim_single_cond=768),
    Scenario(dim_single=768, dim_single_cond=768, seq_len=256),
])
def test_conditioned_transition_block(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

    bs = 1
    torch_dtype = str_dtype_to_torch(sc.dtype)
    trt_dtype = str_dtype_to_trt(sc.dtype)
    device = torch.device('cuda')

    ref_cond_trans = RefConditionedTransitionBlock.load_weights()
    ref_cond_trans = ref_cond_trans.to(device)

    weights_and_biases = create_conditioned_transition_block_weights(
        from_ref=ref_cond_trans)

    a = torch.randn(bs, sc.seq_len, sc.dim_single, dtype=torch.float32).cuda()
    s = torch.randn(bs, sc.seq_len, sc.dim_single_cond,
                    dtype=torch.float32).cuda()

    # construct trt network
    builder = tensorrt_llm.Builder()
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()
    with tensorrt_llm.net_guard(net):
        input_a = Tensor(name='input_a', shape=a.shape, dtype=trt_dtype)
        input_s = Tensor(name='input_s', shape=s.shape, dtype=trt_dtype)

        cond_trans_layer = ConditionedTransitionBlock(
            dim_single=sc.dim_single,
            dim_single_cond=sc.dim_single_cond,
            expansion_factor=sc.expansion_factor,
            dtype=sc.dtype)
        load_conditioned_transition_block_weights_trt(cond_trans_layer,
                                                      weights_and_biases)

        output = cond_trans_layer(input_a, input_s)
        output.mark_output("output", trt_dtype)
    builder_config = builder.create_builder_config(
        name="conditioned_transition_block", precision=sc.dtype)

    # Build engine
    engine_buffer = builder.build_engine(net, builder_config)
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine_buffer)
    stream = torch.cuda.current_stream().cuda_stream

    # Verify results
    inputs = {'input_a': a.to(torch_dtype), 'input_s': s.to(torch_dtype)}
    outputs = {'output': torch.empty(a.shape, dtype=torch_dtype, device="cuda")}
    session.run(inputs=inputs, outputs=outputs, stream=stream)

    with torch.inference_mode():
        ref_output = ref_cond_trans(a, s)

    trt_output = outputs['output']
    torch.cuda.synchronize()
    torch.testing.assert_close(trt_output, ref_output, atol=1e-3, rtol=1e-4)
