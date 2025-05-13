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

import pytest
import tensorrt as trt
import tensorrt_llm
import torch
from tensorrt_llm.functional import Tensor, matmul

from tensorrt_bionemo._trt.functional import chunk_loop


@pytest.mark.parametrize("chunk_size", [2, 4, 8])
@pytest.mark.parametrize("niters", [1, 2, 3])
def test_chunk_loop(chunk_size: int, niters: int):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    input_shape = [chunk_size * niters, 32, 128]
    x = torch.empty(size=input_shape,
                    dtype=torch.float32,
                    device="cuda",
                    requires_grad=False)
    y = torch.empty(size=[chunk_size * niters, 1, 128],
                    dtype=torch.float32,
                    device="cuda",
                    requires_grad=False)
    weight = torch.empty(size=[4, 128],
                         dtype=torch.float32,
                         device="cuda",
                         requires_grad=False)
    x.normal_(mean=0.0, std=1.0)
    y.normal_(mean=0.0, std=1.0)
    weight.normal_(mean=0.0, std=1.0)
    # construct trt network
    builder = tensorrt_llm.Builder()
    net = builder.create_network()

    with tensorrt_llm.net_guard(net):
        input_x = Tensor(name="x", shape=x.shape, dtype=trt.float32)
        input_y = Tensor(name="y", shape=y.shape, dtype=trt.float32)
        input_weight = Tensor(name="weight", shape=[4, 128], dtype=trt.float32)
        output = chunk_loop(
            [input_x, input_y],
            chunk_size=chunk_size,
            loop_body=lambda xy: matmul(xy[0], input_weight, transb=True) +
            matmul(xy[1], input_weight, transb=True))
        output.mark_output("output", tensorrt_llm.str_dtype_to_trt("float32"))

    # Build engine
    builder_config = builder.create_builder_config(name="chunk_loop",
                                                   precision="float32")
    # builder_config.trt_builder_config.add_optimization_profile(profile)
    engine_buffer = builder.build_engine(net, builder_config)
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine_buffer)
    stream = torch.cuda.current_stream().cuda_stream

    # Verify result
    inputs = {
        'x': x,
        'y': y,
        'weight': weight,
    }
    outputs = {
        'output':
        torch.empty([chunk_size * niters, 32, 4],
                    dtype=tensorrt_llm._utils.str_dtype_to_torch("float32"),
                    device="cuda")
    }
    session.run(inputs=inputs, outputs=outputs, stream=stream)
    torch.cuda.synchronize()

    trt_output = outputs['output']
    torch.testing.assert_close(
        trt_output,
        torch.matmul(x, weight.transpose(0, 1)) +
        torch.matmul(y, weight.transpose(0, 1)))
