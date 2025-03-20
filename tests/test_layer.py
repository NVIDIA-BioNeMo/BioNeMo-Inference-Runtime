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

import numpy as np
import pytest
import tensorrt_llm
import torch
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch
from test_utils._plain_attn import (RefPairwiseSelfAttention,
                                    RefTriangleAttention)

import tensorrt_bionemo

TriAttnTestScenario = namedtuple("TriAttnTestScenario", [
    "batch_size", "seq_len", "hidden_size", "num_attention_heads",
    "plain_attn_precision", "dtype"
])

SelfPairwiseTestScenario = namedtuple("SelfPairwiseTestScenario", [
    "batch_size", "seq_len", "c_s", "c_z", "num_attention_heads",
    "plain_attn_precision", "dtype"
])


class TestLayer:

    def setUp(self):
        torch.manual_seed(42)
        os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

    @pytest.mark.parametrize("sc", [
        TriAttnTestScenario(batch_size=1,
                            seq_len=5,
                            hidden_size=32,
                            num_attention_heads=16,
                            plain_attn_precision="float32",
                            dtype="float32"),
        TriAttnTestScenario(batch_size=2,
                            seq_len=12,
                            hidden_size=48,
                            num_attention_heads=8,
                            plain_attn_precision="float32",
                            dtype="float32"),
        TriAttnTestScenario(batch_size=12,
                            seq_len=200,
                            hidden_size=64,
                            num_attention_heads=8,
                            plain_attn_precision="float32",
                            dtype="float32"),
    ])
    def test_triangle_attention(self, sc: TriAttnTestScenario):
        self.setUp()
        c_q = c_k = c_v = sc.hidden_size * sc.num_attention_heads

        mean = 0.0
        std_dev = 1 if sc.dtype == "float32" else 0.05
        torch_dtype = str_dtype_to_torch(sc.dtype)
        hidden_states = torch.empty(size=[sc.batch_size, sc.seq_len, c_q],
                                    dtype=torch_dtype,
                                    device="cuda",
                                    requires_grad=False)
        hidden_states.normal_(mean=mean, std=std_dev)

        mask_bias = torch.empty(size=[sc.batch_size, 1, 1, sc.seq_len],
                                dtype=torch_dtype,
                                device="cuda",
                                requires_grad=False)
        mask_bias.normal_(mean=mean, std=std_dev)
        triangle_bias = torch.empty(
            size=[1, sc.num_attention_heads, sc.seq_len, sc.seq_len],
            dtype=torch_dtype,
            device="cuda",
            requires_grad=False)
        triangle_bias.normal_(mean=mean, std=std_dev)

        q_weight = torch.empty(size=[c_q, c_q], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(q_weight)

        # The reason why chose the identity matrix for K and V,
        # see tensorrt_llm/tests/test_layer.py::TestLayer::test_attention
        eye_weight = torch.eye(c_k, dtype=torch_dtype)
        qkv_weight = torch.cat([q_weight, eye_weight, eye_weight], dim=-1)
        out_weight = eye_weight
        gating_weight = eye_weight

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
                                   dtype=tensorrt_llm.str_dtype_to_trt(
                                       sc.dtype))
            trt_triangle_bias = Tensor(name="triangle_bias",
                                       shape=triangle_bias.shape,
                                       dtype=tensorrt_llm.str_dtype_to_trt(
                                           sc.dtype))
            attn_layer = tensorrt_bionemo.layers.TriangleAttention(
                hidden_size=c_q,
                num_attention_heads=sc.num_attention_heads,
                num_kv_heads=sc.num_attention_heads,
                local_layer_idx=0,
                gating=True)
            attn_layer.qkv_proj.weight.value = np.ascontiguousarray(
                qkv_weight.cpu().numpy().transpose(1, 0))
            attn_layer.o_proj.weight.value = np.ascontiguousarray(
                out_weight.cpu().numpy().transpose(1, 0))
            attn_layer.g_proj.weight.value = np.ascontiguousarray(
                gating_weight.cpu().numpy().transpose(1, 0))

            input_tensor = trt_hidden_states
            output = attn_layer(input_tensor,
                                biases=[trt_mask_bias, trt_triangle_bias],
                                plain_attn_precision=sc.plain_attn_precision)
            output.mark_output("output",
                               tensorrt_llm.str_dtype_to_trt(sc.dtype))

        builder_config = builder.create_builder_config(name="tri_attention",
                                                       precision=sc.dtype)

        # Build engine
        engine_buffer = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(
            engine_buffer)
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

        q_weight.to("cuda")
        eye_weight.to("cuda")
        out_weight.to("cuda")
        gating_weight.to("cuda")

        ref_attn = RefTriangleAttention(c_q,
                                        c_k,
                                        c_v,
                                        sc.hidden_size,
                                        sc.num_attention_heads,
                                        gating=True)
        ref_attn.to("cuda", dtype=torch_dtype)

        ref_attn.linear_q.weight.data.copy_(q_weight.transpose(1, 0))
        ref_attn.linear_k.weight.data.copy_(eye_weight)
        ref_attn.linear_v.weight.data.copy_(eye_weight)
        ref_attn.linear_o.weight.data.copy_(out_weight)
        ref_attn.linear_g.weight.data.copy_(gating_weight)

        with torch.inference_mode():
            ref_output = ref_attn(hidden_states, hidden_states,
                                  [mask_bias, triangle_bias])

        trt_output = outputs['output']
        torch.testing.assert_close(trt_output, ref_output, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize("sc", [
        SelfPairwiseTestScenario(batch_size=1,
                                 seq_len=5,
                                 c_s=384,
                                 c_z=128,
                                 num_attention_heads=16,
                                 plain_attn_precision="float32",
                                 dtype="float32"),
        SelfPairwiseTestScenario(batch_size=2,
                                 seq_len=15,
                                 c_s=96,
                                 c_z=64,
                                 num_attention_heads=8,
                                 plain_attn_precision="float32",
                                 dtype="float32"),
        SelfPairwiseTestScenario(batch_size=3,
                                 seq_len=30,
                                 c_s=384,
                                 c_z=128,
                                 num_attention_heads=32,
                                 plain_attn_precision="float32",
                                 dtype="float32"),
    ])
    def test_self_pairwise_attention(self, sc: SelfPairwiseTestScenario):
        self.setUp()
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

        init_norm_weight = torch.empty(size=[sc.c_s], dtype=torch_dtype)
        torch.nn.init.normal_(init_norm_weight, mean=mean, std=std_dev)
        init_norm_bias = torch.empty(size=[sc.c_s], dtype=torch_dtype)
        torch.nn.init.zeros_(init_norm_bias)

        q_weight = torch.empty(size=[sc.c_s, sc.c_s], dtype=torch_dtype)
        q_bias = torch.empty(size=[sc.c_s], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(q_weight)
        torch.nn.init.zeros_(q_bias)

        eye_weight = torch.eye(sc.c_s, dtype=torch_dtype)
        k_weight = eye_weight
        v_weight = eye_weight
        o_weight = eye_weight
        g_weight = eye_weight
        z_weight = torch.empty([sc.c_z, sc.num_attention_heads],
                               dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(z_weight)
        norm_z_weight = torch.empty(size=[sc.c_z], dtype=torch_dtype)
        torch.nn.init.normal_(norm_z_weight, mean=mean, std=std_dev)
        norm_z_bias = torch.empty(size=[sc.c_z], dtype=torch_dtype)
        torch.nn.init.zeros_(norm_z_bias)

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
            attn_layer.norm_s.weight.value = np.ascontiguousarray(
                init_norm_weight.cpu().numpy())
            attn_layer.norm_s.bias.value = np.ascontiguousarray(
                init_norm_bias.cpu().numpy())
            attn_layer.proj_q.weight.value = np.ascontiguousarray(
                q_weight.cpu().numpy().transpose(1, 0))
            attn_layer.proj_q.bias.value = np.ascontiguousarray(
                q_bias.cpu().numpy())
            attn_layer.proj_k.weight.value = np.ascontiguousarray(
                k_weight.cpu().numpy().transpose(1, 0))
            attn_layer.proj_v.weight.value = np.ascontiguousarray(
                v_weight.cpu().numpy().transpose(1, 0))
            attn_layer.proj_o.weight.value = np.ascontiguousarray(
                o_weight.cpu().numpy().transpose(1, 0))
            attn_layer.proj_g.weight.value = np.ascontiguousarray(
                g_weight.cpu().numpy().transpose(1, 0))
            attn_layer.proj_z.weight.value = np.ascontiguousarray(
                z_weight.cpu().numpy().transpose(1, 0))

            attn_layer.proj_z_norm.weight.value = np.ascontiguousarray(
                norm_z_weight.cpu().numpy())
            attn_layer.proj_z_norm.bias.value = np.ascontiguousarray(
                norm_z_bias.cpu().numpy())

            output = attn_layer(trt_s,
                                trt_z,
                                mask=trt_mask,
                                plain_attn_precision=sc.plain_attn_precision)
            output.mark_output("output",
                               tensorrt_llm.str_dtype_to_trt(sc.dtype))
        builder_config = builder.create_builder_config(
            name="self_pairwise_attention", precision=sc.dtype)

        # Build engine
        engine_buffer = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(
            engine_buffer)
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
        init_norm_weight.to("cuda")
        init_norm_bias.to("cuda")
        q_weight.to("cuda")
        q_bias.to("cuda")
        eye_weight.to("cuda")
        z_weight.to("cuda")
        norm_z_weight.to("cuda")
        norm_z_bias.to("cuda")

        ref_attn = RefPairwiseSelfAttention(c_s=sc.c_s,
                                            c_z=sc.c_z,
                                            num_heads=sc.num_attention_heads,
                                            inf=1e6,
                                            initial_norm=True)
        ref_attn.to("cuda", dtype=torch_dtype)

        ref_attn.norm_s.weight.data.copy_(init_norm_weight)
        ref_attn.norm_s.bias.data.copy_(init_norm_bias)

        ref_attn.proj_q.weight.data.copy_(q_weight.transpose(1, 0))
        ref_attn.proj_q.bias.data.copy_(q_bias)

        ref_attn.proj_k.weight.data.copy_(k_weight)
        ref_attn.proj_v.weight.data.copy_(v_weight)
        ref_attn.proj_o.weight.data.copy_(o_weight)
        ref_attn.proj_g.weight.data.copy_(g_weight)
        ref_attn.proj_z[1].weight.data.copy_(z_weight.transpose(1, 0))

        ref_attn.proj_z[0].weight.data.copy_(norm_z_weight)
        ref_attn.proj_z[0].bias.data.copy_(norm_z_bias)

        with torch.inference_mode():
            ref_output = ref_attn(s, z, mask)

        trt_output = outputs['output']
        torch.testing.assert_close(trt_output, ref_output, atol=1e-3, rtol=1e-4)
