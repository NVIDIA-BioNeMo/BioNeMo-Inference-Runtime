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

import pytest
import tensorrt_llm
import torch
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.functional import Tensor, concat, shape

from tensorrt_bionemo._trt.functional import dynamic_const_tensor


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("input_shape", [(1, 1), (1, 2, 3), (2, 3, 4, 5)])
def test_dynamic_const_tensor(dtype, input_shape):
    torch_dtype = str_dtype_to_torch(dtype)
    x = torch.randn(input_shape, dtype=torch_dtype).cuda()

    builder = tensorrt_llm.Builder()
    net = builder.create_network()

    with tensorrt_llm.net_guard(net):
        input_x = Tensor(name="x", shape=x.shape, dtype=dtype)

        x_shape = concat([shape(input_x, i) for i in range(input_x.ndim())])
        two_out = dynamic_const_tensor(x_shape, value=2.0, dtype=dtype)
        one_out = dynamic_const_tensor(x_shape, value=1.0, dtype=dtype)
        zero_out = dynamic_const_tensor(x_shape, value=0.0, dtype=dtype)

        two_out.mark_output("two_out", tensorrt_llm.str_dtype_to_trt(dtype))
        one_out.mark_output("one_out", tensorrt_llm.str_dtype_to_trt(dtype))
        zero_out.mark_output("zero_out", tensorrt_llm.str_dtype_to_trt(dtype))

    # Build engine
    builder_config = builder.create_builder_config(name="dynamic_const_tensor",
                                                   precision="float32")
    # builder_config.trt_builder_config.add_optimization_profile(profile)
    engine_buffer = builder.build_engine(net, builder_config)
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine_buffer)
    stream = torch.cuda.current_stream().cuda_stream

    inputs = {
        'x': x,
    }
    outputs = {
        'one_out':
        torch.empty(x.shape, dtype=str_dtype_to_torch(dtype), device="cuda"),
        'zero_out':
        torch.empty(x.shape, dtype=str_dtype_to_torch(dtype), device="cuda"),
        'two_out':
        torch.empty(x.shape, dtype=str_dtype_to_torch(dtype), device="cuda"),
    }

    session.run(inputs, outputs, stream)

    assert torch.allclose(outputs['one_out'], torch.ones_like(x))
    assert torch.allclose(outputs['zero_out'], torch.zeros_like(x))
    assert torch.allclose(outputs['two_out'], torch.ones_like(x) * 2)
