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
import tensorrt_llm_lite
import torch
from einops import rearrange
from tensorrt_llm_lite._utils import str_dtype_to_torch, str_dtype_to_trt
from tensorrt_llm_lite.functional import Tensor
from test_utils.boltz.ref_attn import plain_mha

from tensorrt_bionemo._trt.functional import (AttentionBackend,
                                              triangle_attention)
from tests._torch import make_left_aligned_mask


@pytest.mark.parametrize("use_mask", [True, False], ids=["mask", "nomask"])
@pytest.mark.parametrize("backend", [AttentionBackend.CUEQUIV],
                         ids=["cuequiv"])
@pytest.mark.parametrize("use_tf32", [False])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("si", [64, 128])
@pytest.mark.parametrize("sj", [64, 128])
@pytest.mark.parametrize("sk", [64, 128])
def test_triangle_attention(use_mask, backend, use_tf32, dtype, si, sj, sk):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    num_heads = 4
    head_dim = 32
    use_trifast = backend == AttentionBackend.TRIFAST
    if use_trifast:
        pytest.skip("trifast is disabled for now")

    q = torch.randn(bs * num_heads,
                    si,
                    sj,
                    head_dim,
                    dtype=str_dtype_to_torch(dtype),
                    device="cuda",
                    requires_grad=False)
    k = torch.randn(bs * num_heads,
                    si,
                    sk,
                    head_dim,
                    dtype=str_dtype_to_torch(dtype),
                    device="cuda",
                    requires_grad=False)
    v = torch.randn(bs * num_heads,
                    si,
                    sk,
                    head_dim,
                    dtype=str_dtype_to_torch(dtype),
                    device="cuda",
                    requires_grad=False)
    if use_mask:
        mask = make_left_aligned_mask(bs,
                                      si,
                                      sk,
                                      dtype=torch.float32,
                                      device="cuda").bool()
    else:
        mask = torch.ones((bs, si, sk)).bool().cuda()
    bias = torch.randn(
        bs * num_heads,
        sj,
        sk,
        dtype=str_dtype_to_torch(dtype) if use_trifast else torch.float32,
        device="cuda",
        requires_grad=False)
    # construct trt network
    if use_trifast:
        inputs = {
            'q': q,
            'k': k,
            'v': v,
            'mask': mask,
            'bias': bias,
        }
        input_q_shape = q.shape
        input_k_shape = k.shape
        input_v_shape = v.shape
        input_mask_shape = mask.shape
        input_bias_shape = bias.shape
    else:
        nq = rearrange(q,
                       "(b h) i j d -> b i h j d",
                       b=bs,
                       h=num_heads,
                       d=head_dim).contiguous()
        nk = rearrange(k,
                       "(b h) i j d -> b i h j d",
                       b=bs,
                       h=num_heads,
                       d=head_dim).contiguous()
        nv = rearrange(v,
                       "(b h) i j d -> b i h j d",
                       b=bs,
                       h=num_heads,
                       d=head_dim).contiguous()
        nbias = rearrange(bias, "(b h) i j -> b () h i j", b=bs,
                          h=num_heads).contiguous()
        nmask = rearrange(mask, "b i j -> b i () () j", b=bs).contiguous()

        inputs = {
            'q': nq,
            'k': nk,
            'v': nv,
            'bias': nbias,
        }
        if use_mask:
            inputs['mask'] = nmask
        input_q_shape = nq.shape
        input_k_shape = nk.shape
        input_v_shape = nv.shape
        input_mask_shape = nmask.shape
        input_bias_shape = nbias.shape

    builder = tensorrt_llm_lite.Builder()
    net = builder.create_network()

    trt_dtype = str_dtype_to_trt(dtype)
    trt_dtype_bias = str_dtype_to_trt(dtype if use_trifast else "float32")
    with tensorrt_llm_lite.net_guard(net):
        input_q = Tensor(name="q", shape=input_q_shape, dtype=trt_dtype)
        input_k = Tensor(name="k", shape=input_k_shape, dtype=trt_dtype)
        input_v = Tensor(name="v", shape=input_v_shape, dtype=trt_dtype)
        input_mask = Tensor(name="mask",
                            shape=input_mask_shape,
                            dtype=trt.bool) if use_mask else None
        input_bias = Tensor(name="bias",
                            shape=input_bias_shape,
                            dtype=trt_dtype_bias)
        output, lse = triangle_attention(input_q,
                                         input_k,
                                         input_v,
                                         input_bias,
                                         input_mask,
                                         num_heads,
                                         head_dim,
                                         dtype=dtype,
                                         backend=backend,
                                         use_tf32=use_tf32)

        output.mark_output("output", trt_dtype)
        lse.mark_output("lse", trt_dtype)
    # Build engine
    builder_config = builder.create_builder_config(name="tri_attn",
                                                   precision=dtype)
    engine_buffer = builder.build_engine(net, builder_config)
    session = tensorrt_llm_lite.runtime.Session.from_serialized_engine(
        engine_buffer)
    stream = torch.cuda.current_stream().cuda_stream

    if use_trifast:
        output_shape = (bs * num_heads, si, sj, head_dim)
        lse_shape = (bs * num_heads, sj, sj)
    else:
        output_shape = (bs, si, num_heads, sj, head_dim)
        lse_shape = (bs, si, num_heads, sj)

    outputs = {
        'output':
        torch.empty(output_shape,
                    dtype=str_dtype_to_torch(dtype),
                    device="cuda"),
        'lse':
        torch.empty(
            lse_shape,
            dtype=str_dtype_to_torch(dtype) if use_trifast else torch.float32,
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
        if use_trifast:
            mask = rearrange(mask, "b i j -> b i () () j",
                             b=bs).contiguous().float() * torch.finfo(
                                 q.dtype).min
        else:  # flip mask for cuequiv ops
            mask = rearrange(~mask, "b i j -> b i () () j",
                             b=bs).contiguous().float() * torch.finfo(
                                 q.dtype).min

        ref_o = plain_mha(q, k, v, num_heads, head_dim,
                          [mask, bias.unsqueeze(1)])

        if use_trifast:
            ref_o = rearrange(ref_o,
                              "b i j h d -> (b h) i j d",
                              b=bs,
                              h=num_heads,
                              i=si,
                              j=sj,
                              d=head_dim)
        else:  # use cuequiv ops
            ref_o = rearrange(ref_o,
                              "b i j h d -> b i h j d",
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
