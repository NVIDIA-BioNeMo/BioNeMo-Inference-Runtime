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
from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt
from tensorrt_llm.functional import Tensor
from test_utils.boltz.create_and_load_weights import (
    create_diffusion_transformer_layer_weights,
    load_diffusion_transformer_layer_weights_trt)
from test_utils.boltz.ref_layers import RefDiffusionTransformerLayer

from tensorrt_bionemo._trt.layers.attention import AttentionParams
from tensorrt_bionemo._trt.layers.transformers import DiffusionTransformerLayer
from tensorrt_bionemo.mapping import Mapping


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dim: int = 768
    dim_single_cond: int = 768
    dtype: str = "float32"
    seq_len: int = 128
    num_heads: int = 16
    dim_pairwise: int = 128


@pytest.mark.parametrize("sc", [Scenario(dim=768, dim_single_cond=768)])
def test_diffusion_transformer_layer(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    torch_dtype = str_dtype_to_torch(sc.dtype)
    trt_dtype = str_dtype_to_trt(sc.dtype)
    device = torch.device('cuda')

    ref_module = RefDiffusionTransformerLayer.load_weights()
    ref_module = ref_module.to(device)

    weights_and_biases = create_diffusion_transformer_layer_weights(
        from_ref=ref_module)
    a = torch.randn(bs, sc.seq_len, sc.dim, dtype=torch.float32).cuda()
    s = torch.randn(bs, sc.seq_len, sc.dim_single_cond,
                    dtype=torch.float32).cuda()
    z = torch.randn(bs,
                    sc.num_heads,
                    sc.seq_len,
                    sc.seq_len,
                    dtype=torch.float32).cuda()
    mask = torch.randn(bs, sc.seq_len, dtype=torch.float32).cuda()

    # construct trt network
    builder = tensorrt_llm.Builder()
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()
    with tensorrt_llm.net_guard(net):
        input_a = Tensor(name='input_a', shape=a.shape, dtype=trt_dtype)
        input_s = Tensor(name='input_s', shape=s.shape, dtype=trt_dtype)
        input_z = Tensor(name='input_z', shape=z.shape, dtype=trt_dtype)
        input_mask = Tensor(name='input_mask',
                            shape=mask.shape,
                            dtype=trt_dtype)

        layer = DiffusionTransformerLayer(
            local_layer_idx=0,
            num_heads=ref_module.pair_bias_attn.num_heads,
            dim=sc.dim,
            dim_single_cond=sc.dim_single_cond,
            dim_pairwise=sc.dim_pairwise,
            dtype=sc.dtype,
            mapping=Mapping())

        load_diffusion_transformer_layer_weights_trt(layer, weights_and_biases)

        output_a = layer(input_a,
                         input_s,
                         input_z,
                         input_mask,
                         attention_params=AttentionParams())
        output_a.mark_output("output_a", trt_dtype)

    builder_config = builder.create_builder_config(
        name="diffusion_transformer_layer", precision=sc.dtype)

    # Build engine
    engine_buffer = builder.build_engine(net, builder_config)
    assert engine_buffer is not None
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine_buffer)

    stream = torch.cuda.current_stream().cuda_stream

    # Verify result
    inputs = {'input_a': a, 'input_s': s, 'input_z': z, 'input_mask': mask}
    outputs = {
        'output_a': torch.empty(a.shape, dtype=torch_dtype, device="cuda"),
    }
    session.run(inputs=inputs, outputs=outputs, stream=stream)
    torch.cuda.synchronize()

    with torch.inference_mode():
        ref_output_a = ref_module(a, s, z, mask, compute_pair_bias=False)
        torch.cuda.synchronize()

    trt_output_a = outputs['output_a']
    torch.testing.assert_close(trt_output_a, ref_output_a, atol=1e-3, rtol=1e-4)
