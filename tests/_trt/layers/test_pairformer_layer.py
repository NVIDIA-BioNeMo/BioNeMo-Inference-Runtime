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
from test_utils.boltz.create_and_load_weights import *

from tensorrt_bionemo._trt.layers.attention import AttentionParams
from tensorrt_bionemo._trt.layers.transformers import PairformerLayerV1


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dtype: str = "float32"
    seq_len: int = 32
    token_s: int = 32
    token_z: int = 128
    num_heads: int = 16
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    triangle_attn_backend: str = "VANILLA"
    pairwise_attn_backend: str = "VANILLA"
    support_batch: bool = True


@pytest.mark.parametrize("sc", [
    Scenario(seq_len=64),
    Scenario(seq_len=256),
    Scenario(seq_len=64, triangle_attn_backend="TRIFAST"),
    Scenario(seq_len=256, triangle_attn_backend="TRIFAST"),
    Scenario(seq_len=64, triangle_attn_backend="CUEQUIV"),
    Scenario(seq_len=256, triangle_attn_backend="CUEQUIV"),
    Scenario(seq_len=64, support_batch=False),
    Scenario(seq_len=256, support_batch=False),
    Scenario(seq_len=64, triangle_attn_backend="TRIFAST", support_batch=False),
    Scenario(seq_len=256, triangle_attn_backend="TRIFAST", support_batch=False),
    Scenario(seq_len=64, triangle_attn_backend="CUEQUIV", support_batch=False),
    Scenario(seq_len=256, triangle_attn_backend="CUEQUIV", support_batch=False),
],
                         ids=[
                             "64_vanilla", "256_vanilla", "64_trifast",
                             "256_trifast", "64_cuequiv", "256_cuequiv",
                             "64_vanilla_no_batch", "256_vanilla_no_batch",
                             "64_trifast_no_batch", "256_trifast_no_batch",
                             "64_cuequiv_no_batch", "256_cuequiv_no_batch"
                         ])
def test_pairformer_layer(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    mean = 0.0
    std_dev = 1 if sc.dtype == "float32" else 0.05
    torch_dtype = str_dtype_to_torch(sc.dtype)

    if sc.support_batch:
        s = torch.empty(size=[bs, sc.seq_len, sc.token_s],
                        dtype=torch_dtype,
                        device="cuda",
                        requires_grad=False)
        s.normal_(mean=mean, std=std_dev)

        z = torch.empty(size=[bs, sc.seq_len, sc.seq_len, sc.token_z],
                        dtype=torch_dtype,
                        device="cuda",
                        requires_grad=False)
        z.normal_(mean=mean, std=std_dev)

        mask = torch.randint(0, 2, (bs, sc.seq_len), dtype=torch_dtype).cuda()
        pairmask = torch.randint(0,
                                 2, (bs, sc.seq_len, sc.seq_len),
                                 dtype=torch_dtype).cuda()
    else:
        s = torch.empty(size=[sc.seq_len, sc.token_s],
                        dtype=torch_dtype,
                        device="cuda",
                        requires_grad=False)
        s.normal_(mean=mean, std=std_dev)
        z = torch.empty(size=[sc.seq_len, sc.seq_len, sc.token_z],
                        dtype=torch_dtype,
                        device="cuda",
                        requires_grad=False)
        z.normal_(mean=mean, std=std_dev)
        mask = torch.randint(0, 2, (sc.seq_len, ), dtype=torch_dtype).cuda()
        pairmask = torch.randint(0,
                                 2, (sc.seq_len, sc.seq_len),
                                 dtype=torch_dtype).cuda()

    weights_and_biases = \
        create_pairformer_layer_weights(sc.token_s, sc.token_z, sc.num_heads, sc.pairwise_head_width, sc.pairwise_num_heads, torch_dtype)

    # construct trt network
    builder = tensorrt_llm.Builder()
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()
    with tensorrt_llm.net_guard(net):
        trt_s = Tensor(name='s',
                       shape=s.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(sc.dtype))
        trt_z = Tensor(name='z',
                       shape=z.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(sc.dtype))
        trt_mask = Tensor(name='mask',
                          shape=mask.shape,
                          dtype=tensorrt_llm.str_dtype_to_trt(sc.dtype))
        trt_pairmask = Tensor(name='pairmask',
                              shape=pairmask.shape,
                              dtype=tensorrt_llm.str_dtype_to_trt(sc.dtype))

        pairformer_layer = PairformerLayerV1(
            local_layer_idx=0,
            token_s=sc.token_s,
            token_z=sc.token_z,
            num_heads=sc.num_heads,
            pairwise_head_width=sc.pairwise_head_width,
            pairwise_num_heads=sc.pairwise_num_heads,
            triangle_attn_backend=sc.triangle_attn_backend,
            support_batch=sc.support_batch)

        load_pairformer_layer_weights_trt(pairformer_layer, weights_and_biases)

        output_s, output_z = pairformer_layer(
            trt_s,
            trt_z,
            trt_mask,
            trt_pairmask,
            attention_params=AttentionParams())

        output_s.mark_output("output_s",
                             tensorrt_llm.str_dtype_to_trt(sc.dtype))
        output_z.mark_output("output_z",
                             tensorrt_llm.str_dtype_to_trt(sc.dtype))

    builder_config = builder.create_builder_config(name="pairformer_layer",
                                                   precision=sc.dtype)

    # Build engine
    engine_buffer = builder.build_engine(net, builder_config)
    assert engine_buffer is not None
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine_buffer)

    stream = torch.cuda.current_stream().cuda_stream

    # Verify result
    inputs = {'s': s, 'z': z, 'mask': mask, 'pairmask': pairmask}
    outputs = {
        'output_s': torch.empty(s.shape, dtype=torch_dtype, device="cuda"),
        'output_z': torch.empty(z.shape, dtype=torch_dtype, device="cuda")
    }
    session.run(inputs=inputs, outputs=outputs, stream=stream)
    torch.cuda.synchronize()

    ref_pairformer_layer = RefPairformerLayer(
        token_s=sc.token_s,
        token_z=sc.token_z,
        num_heads=sc.num_heads,
        pairwise_head_width=sc.pairwise_head_width,
        pairwise_num_heads=sc.pairwise_num_heads)

    load_pairformer_layer_weights_ref_torch(ref_pairformer_layer,
                                            weights_and_biases)
    ref_pairformer_layer.to("cuda", dtype=torch_dtype)

    with torch.inference_mode():
        if sc.support_batch:
            ref_output_s, ref_output_z = ref_pairformer_layer(
                s, z, mask, pairmask)
        else:
            ref_output_s, ref_output_z = ref_pairformer_layer(
                s.unsqueeze(0), z.unsqueeze(0), mask.unsqueeze(0),
                pairmask.unsqueeze(0))
            ref_output_s = ref_output_s.squeeze(0)
            ref_output_z = ref_output_z.squeeze(0)
        torch.cuda.synchronize()
    trt_output_s = outputs['output_s']
    trt_output_z = outputs['output_z']

    torch.testing.assert_close(trt_output_s, ref_output_s, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(trt_output_z, ref_output_z, atol=1e-3, rtol=1e-4)
