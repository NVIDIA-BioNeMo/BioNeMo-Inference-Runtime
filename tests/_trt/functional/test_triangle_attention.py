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
from einops import rearrange
from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt
from tensorrt_llm.functional import Tensor
from test_utils.ref_attn import plain_triangle_mha

from tensorrt_bionemo._trt.functional import triangle_attention


@pytest.mark.parametrize("use_trifast", [False])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("si", [64, 128, 256, 512])
@pytest.mark.parametrize("sj", [64, 128, 256, 768, 1056])
def test_triangle_attention(use_trifast, dtype, si, sj):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    num_heads = 4
    head_dim = 32

    q = torch.randn(bs * num_heads,
                    si,
                    sj,
                    head_dim,
                    dtype=str_dtype_to_torch(dtype),
                    device="cuda",
                    requires_grad=False)
    k = torch.randn(bs * num_heads,
                    si,
                    sj,
                    head_dim,
                    dtype=str_dtype_to_torch(dtype),
                    device="cuda",
                    requires_grad=False)
    v = torch.randn(bs * num_heads,
                    si,
                    sj,
                    head_dim,
                    dtype=str_dtype_to_torch(dtype),
                    device="cuda",
                    requires_grad=False)
    mask = torch.randint(0, 2, (bs, si, sj)).bool().cuda()
    # mask = torch.zeros_like(mask).bool().cuda()
    bias = torch.randn(bs * num_heads,
                       sj,
                       sj,
                       dtype=str_dtype_to_torch(dtype) if use_trifast else torch.float32,
                       device="cuda",
                       requires_grad=False)
    # construct trt network
    builder = tensorrt_llm.Builder()
    net = builder.create_network()

    trt_dtype = str_dtype_to_trt(dtype)
    with tensorrt_llm.net_guard(net):
        input_q = Tensor(name="q", shape=q.shape, dtype=trt_dtype)
        input_k = Tensor(name="k", shape=k.shape, dtype=trt_dtype)
        input_v = Tensor(name="v", shape=v.shape, dtype=trt_dtype)
        input_mask = Tensor(name="mask", shape=mask.shape, dtype=trt.bool)
        input_bias = Tensor(name="bias", shape=bias.shape, dtype=trt_dtype)
        output, lse = triangle_attention(input_q,
                                         input_k,
                                         input_v,
                                         input_bias,
                                         input_mask,
                                         num_heads,
                                         head_dim,
                                         dtype=dtype,
                                         use_trifast=use_trifast)

        output.mark_output("output", trt_dtype)
        lse.mark_output("lse", trt_dtype)
    # Build engine
    builder_config = builder.create_builder_config(name="tri_attn",
                                                   precision=dtype)
    engine_buffer = builder.build_engine(net, builder_config)
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine_buffer)
    stream = torch.cuda.current_stream().cuda_stream

    # Verify result
    inputs = {
        'q': q,
        'k': k,
        'v': v,
        'mask': mask,
        'bias': bias,
    }
    outputs = {
        'output':
        torch.empty([bs * num_heads, si, sj, head_dim],
                    dtype=str_dtype_to_torch(dtype),
                    device="cuda"),
        'lse':
        torch.empty([bs * num_heads, sj, sj],
                    dtype=str_dtype_to_torch(dtype),
                    device="cuda")
    }
    session.run(inputs=inputs, outputs=outputs, stream=stream)
    torch.cuda.synchronize()

    with torch.no_grad():
        q = rearrange(q,
                      "(b h) i j d -> b i j (h d)",
                      b=bs,
                      h=num_heads,
                      d=head_dim).contiguous()
        k = rearrange(k,
                      "(b h) i j d -> b i j (h d)",
                      b=bs,
                      h=num_heads,
                      d=head_dim).contiguous()
        v = rearrange(v,
                      "(b h) i j d -> b i j (h d)",
                      b=bs,
                      h=num_heads,
                      d=head_dim).contiguous()
        bias = rearrange(bias, "(b h) i j -> b h i j", b=bs,
                         h=num_heads).contiguous()
        mask = rearrange(mask, "b i j -> b i () () j",
                         b=bs).contiguous().float() * torch.finfo(q.dtype).min
        ref_o = plain_triangle_mha(q, k, v, num_heads, head_dim, [mask, bias])
        ref_o = rearrange(ref_o,
                          "b i j h d -> (b h) i j d",
                          b=bs,
                          h=num_heads,
                          i=si,
                          j=sj,
                          d=head_dim)
    torch.cuda.synchronize()
    if dtype == "float32":
        torch.testing.assert_close(outputs['output'], ref_o)
    else:
        torch.testing.assert_close(outputs['output'],
                                   ref_o,
                                   atol=5e-2,
                                   rtol=1e-4)
    torch.cuda.empty_cache()
