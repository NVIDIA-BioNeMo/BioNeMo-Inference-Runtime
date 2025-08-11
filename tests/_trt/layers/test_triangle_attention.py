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
from tensorrt_llm._utils import get_sm_version, str_dtype_to_torch
from test_utils.boltz.create_and_load_weights import *
from test_utils.boltz.ref_attn import RefTriangleAttention
from test_utils.openfold.create_and_load_weights import (
    create_msa_attention_weights, load_msa_attention_weights_trt)
from test_utils.openfold.ref_layers import RefMSAAttention

from tensorrt_bionemo._trt.layers.attention import (AttentionParams,
                                                    MSAAttention,
                                                    TriangleAttention)


@dataclass(kw_only=True, frozen=True)
class Scenario:
    triangle_attn_backend: str = "VANILLA"
    bs: int = 1
    si: int = 32
    sj: int = 32
    dtype: str = "float32"
    support_batch: bool = True


@pytest.mark.parametrize("sc", [
    Scenario(bs=2,
             si=64,
             sj=96,
             dtype="float32",
             triangle_attn_backend="VANILLA",
             support_batch=True),
    Scenario(bs=1,
             si=64,
             sj=96,
             dtype="float32",
             triangle_attn_backend="VANILLA",
             support_batch=False),
    Scenario(bs=2,
             si=64,
             sj=96,
             dtype="float32",
             triangle_attn_backend="TRIFAST",
             support_batch=True),
    Scenario(bs=1,
             si=64,
             sj=96,
             dtype="float32",
             triangle_attn_backend="TRIFAST",
             support_batch=False),
    Scenario(bs=2,
             si=64,
             sj=96,
             dtype="float32",
             triangle_attn_backend="CUEQUIV",
             support_batch=True),
    Scenario(bs=1,
             si=64,
             sj=96,
             dtype="float32",
             triangle_attn_backend="CUEQUIV",
             support_batch=False),
],
                         ids=[
                             "vanilla_batch", "vanilla_no_batch",
                             "trifast_batch", "trifast_no_batch",
                             "cuequiv_batch", "cuequiv_no_batch"
                         ])
def test_triangle_attention(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    if sc.triangle_attn_backend == "TRIFAST":
        sm_version = get_sm_version()
        if sm_version not in [80, 86]:
            pytest.skip(
                "trifast is only supported on sm_80 and sm_86 architectures for now"
            )

    ref_attn = RefTriangleAttention.load_weights()
    weights_and_biases = \
        create_triangle_attention_weights(from_ref=ref_attn)
    c_q = c_k = c_v = ref_attn.c_q

    mean = 0.0
    std_dev = 1 if sc.dtype == "float32" else 0.05
    torch_dtype = str_dtype_to_torch(sc.dtype)
    if sc.support_batch:
        shape = [sc.bs, sc.si, sc.sj, c_q]
    else:
        shape = [sc.si, sc.sj, c_q]
    hidden_states = torch.empty(size=shape,
                                dtype=torch_dtype,
                                device="cuda",
                                requires_grad=False)
    hidden_states.normal_(mean=mean, std=std_dev)

    if sc.support_batch:
        shape = [sc.bs, sc.si, 1, 1, sc.sj]
    else:
        shape = [sc.si, 1, 1, sc.sj]
    mask_bias = torch.randint(0,
                              2,
                              shape,
                              dtype=torch_dtype,
                              device="cuda",
                              requires_grad=False)
    triangle_bias = torch.empty(size=[sc.bs, ref_attn.no_heads, sc.sj, sc.sj],
                                dtype=torch_dtype,
                                device="cuda",
                                requires_grad=False)
    triangle_bias.normal_(mean=mean, std=std_dev)

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
        attn_layer = TriangleAttention(
            hidden_size=ref_attn.c_q,
            num_attention_heads=ref_attn.no_heads,
            num_kv_heads=ref_attn.no_heads,
            local_layer_idx=0,
            gating=True,
            dtype=sc.dtype,
            support_batch=sc.support_batch,
            triangle_attn_backend=sc.triangle_attn_backend)
        load_triangle_attention_weights_trt(attn_layer, weights_and_biases)

        input_tensor = trt_hidden_states
        attention_params = AttentionParams()
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
    ref_attn.to("cuda", dtype=torch_dtype)

    with torch.inference_mode():
        if sc.triangle_attn_backend in ["TRIFAST", "CUEQUIV"]:
            mask_bias = mask_bias.to(torch_dtype) * torch.finfo(torch_dtype).min
        if not sc.support_batch:
            hidden_states = hidden_states.unsqueeze(0)
            mask_bias = mask_bias.unsqueeze(0)
        ref_output = ref_attn(hidden_states, hidden_states,
                              [mask_bias, triangle_bias])

    trt_output = outputs['output']
    if not sc.support_batch:
        trt_output = trt_output.unsqueeze(0)
    torch.testing.assert_close(trt_output, ref_output, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("sc", [
    Scenario(bs=2,
             si=64,
             sj=96,
             dtype="float32",
             triangle_attn_backend="VANILLA",
             support_batch=True),
    Scenario(bs=2,
             si=64,
             sj=96,
             dtype="float32",
             triangle_attn_backend="CUEQUIV",
             support_batch=True)
],
                         ids=["vanilla_batch", "cuequiv_batch"])
def test_msa_attention(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch_dtype = str_dtype_to_torch(sc.dtype)
    ref_attn = RefMSAAttention.load_weights(using_tri_attn=True)
    weights_and_biases = \
        create_msa_attention_weights(from_ref=ref_attn)
    c_q = c_k = c_v = ref_attn.c_in
    """
    m: [B, J, I, c_in]
    z: [B, I, I, c_z]
    mask: [B, J, I]
    """

    m = torch.randn(sc.bs,
                    sc.sj,
                    sc.si,
                    ref_attn.c_in,
                    dtype=torch_dtype,
                    device="cuda")
    z = torch.randn(sc.bs,
                    sc.si,
                    sc.si,
                    ref_attn.c_z,
                    dtype=torch_dtype,
                    device="cuda")
    mask = torch.randint(0,
                         2, (sc.bs, sc.sj, sc.si),
                         dtype=torch_dtype,
                         device="cuda")

    # construct trt network
    builder = tensorrt_llm.Builder()
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()

    with tensorrt_llm.net_guard(net):
        trt_m = Tensor(name='m',
                       shape=m.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(sc.dtype))
        trt_z = Tensor(name="z",
                       shape=z.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(sc.dtype))
        trt_mask = Tensor(name="mask",
                          shape=mask.shape,
                          dtype=tensorrt_llm.str_dtype_to_trt(sc.dtype))
        attn_layer = MSAAttention(
            local_layer_idx=0,
            c_in=ref_attn.c_in,
            num_heads=ref_attn.no_heads,
            c_z=ref_attn.c_z,
            triangle_attn_backend=sc.triangle_attn_backend,
            support_batch=sc.support_batch,
            need_project_z=True,
            dtype=sc.dtype)
        load_msa_attention_weights_trt(attn_layer, weights_and_biases)

        attention_params = AttentionParams()
        output = attn_layer(trt_m,
                            trt_z,
                            trt_mask,
                            attention_params=attention_params)
        output.mark_output("output", tensorrt_llm.str_dtype_to_trt(sc.dtype))

    builder_config = builder.create_builder_config(name="tri_attention",
                                                   precision=sc.dtype)

    # Build engine
    engine_buffer = builder.build_engine(net, builder_config)
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine_buffer)
    stream = torch.cuda.current_stream().cuda_stream

    # Verify result
    inputs = {'m': m, 'z': z, 'mask': mask}
    outputs = {
        'output':
        torch.empty(m.shape,
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(sc.dtype),
                    device="cuda")
    }

    session.run(inputs=inputs, outputs=outputs, stream=stream)
    torch.cuda.synchronize()
    ref_attn.to("cuda", dtype=torch_dtype)

    with torch.inference_mode():
        ref_output = ref_attn(m, z, mask)

    trt_output = outputs['output']
    torch.testing.assert_close(trt_output, ref_output, atol=1e-3, rtol=1e-3)
