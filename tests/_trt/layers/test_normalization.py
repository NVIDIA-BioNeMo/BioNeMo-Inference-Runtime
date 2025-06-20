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
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch
from test_utils.create_and_load_weights import (create_adaln_weights,
                                                load_adaln_weights_trt)
from test_utils.ref_layers import RefAdaLN

from tensorrt_bionemo._trt.layers.normalization import AdaLN
from tensorrt_bionemo.mapping import Mapping


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dim: int = 768
    dim_single_cond: int = 768
    dtype: str = "float32"
    seq_len: int = 128


@pytest.mark.parametrize("sc", [
    Scenario(dim=768, dim_single_cond=768),
])
def test_adaln(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    torch_dtype = str_dtype_to_torch(sc.dtype)
    device = torch.device('cuda')

    ref_adaln = RefAdaLN.load_weights()
    ref_adaln = ref_adaln.to(device)

    weights_and_biases = create_adaln_weights(from_ref=ref_adaln)

    a = torch.randn(bs, sc.seq_len, sc.dim, dtype=torch.float32).cuda()
    s = torch.randn(bs, sc.seq_len, sc.dim_single_cond,
                    dtype=torch.float32).cuda()

    # construct trt network
    builder = tensorrt_llm.Builder()
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()
    with tensorrt_llm.net_guard(net):
        trt_a = Tensor(name='input_a',
                       shape=a.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(sc.dtype))
        trt_s = Tensor(name='input_s',
                       shape=s.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(sc.dtype))

        adaln = AdaLN(dim=sc.dim,
                      dim_single_cond=sc.dim_single_cond,
                      dtype=sc.dtype)
        load_adaln_weights_trt(adaln, weights_and_biases, mapping=Mapping())
        output = adaln(trt_a, trt_s)
        output.mark_output("output", tensorrt_llm.str_dtype_to_trt(sc.dtype))

    builder_config = builder.create_builder_config(
        name="self_pairwise_attention", precision=sc.dtype)

    # Build engine
    engine_buffer = builder.build_engine(net, builder_config)
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine_buffer)
    stream = torch.cuda.current_stream().cuda_stream

    # Verify results
    inputs = {'input_a': a.to(torch_dtype), 'input_s': s.to(torch_dtype)}
    outputs = {'output': torch.empty(a.shape, dtype=torch_dtype, device="cuda")}
    session.run(inputs=inputs, outputs=outputs, stream=stream)

    ref_adaln(a, s)
    ref_output = ref_adaln(a.to(torch_dtype), s.to(torch_dtype))
    torch.cuda.synchronize()
    if sc.dtype == "float32":
        torch.testing.assert_close(outputs['output'],
                                   ref_output,
                                   atol=1e-3,
                                   rtol=1e-4)
