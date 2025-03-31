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

import pytest
import tensorrt_llm
import torch
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch
from test_utils.create_and_load_weights import (
    create_self_pairwise_attention_weights_biases,
    create_triangle_attention_node_weights_and_biases,
    create_triangle_attention_weights,
    load_self_pairwise_attention_weights_torch,
    load_self_pairwise_attention_weights_trt,
    load_triangle_attention_node_weights_torch,
    load_triangle_attention_node_weights_trt,
    load_triangle_attention_weights_torch, load_triangle_attention_weights_trt)
from test_utils.ref_attn import RefPairwiseSelfAttention, RefTriangleAttention
from test_utils.ref_layers import RefTriangleAttentionNode

import tensorrt_bionemo
import tensorrt_bionemo.layers.triangle_nodes

TriAttnTestScenario = namedtuple("TriAttnTestScenario", [
    "batch_size", "seq_len", "hidden_size", "num_attention_heads",
    "plain_attn_precision", "dtype"
])

SelfPairwiseTestScenario = namedtuple("SelfPairwiseTestScenario", [
    "batch_size", "seq_len", "c_s", "c_z", "num_attention_heads",
    "plain_attn_precision", "dtype"
])

TriangleAttentionNodeTestScenario = namedtuple(
    "TriangleAttentionNodeTestScenario", [
        "chunk_size", "seq_len", "c_in", "c_hidden", "num_attention_heads",
        "plain_attn_precision", "dtype", "starting"
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

        weights_and_biases = \
            create_triangle_attention_weights(c_q, c_k, c_v, torch_dtype)

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
            load_triangle_attention_weights_trt(attn_layer, weights_and_biases)

            input_tensor = trt_hidden_states
            attention_params = tensorrt_bionemo.layers.attention.AttentionParams(
                plain_attn_precision=sc.plain_attn_precision)
            output = attn_layer(input_tensor,
                                biases=[trt_mask_bias, trt_triangle_bias],
                                attention_params=attention_params)
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

        ref_attn = RefTriangleAttention(c_q,
                                        c_k,
                                        c_v,
                                        sc.hidden_size,
                                        sc.num_attention_heads,
                                        gating=True)
        ref_attn.to("cuda", dtype=torch_dtype)

        load_triangle_attention_weights_torch(ref_attn, weights_and_biases)

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

        weights_and_biases = \
            create_self_pairwise_attention_weights_biases(sc.c_s, sc.c_z, sc.num_attention_heads, torch_dtype)

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
            load_self_pairwise_attention_weights_trt(attn_layer,
                                                     weights_and_biases)

            attention_params = tensorrt_bionemo.layers.attention.AttentionParams(
                plain_attn_precision=sc.plain_attn_precision)
            output = attn_layer(trt_s,
                                trt_z,
                                mask=trt_mask,
                                attention_params=attention_params)
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
        ref_attn = RefPairwiseSelfAttention(c_s=sc.c_s,
                                            c_z=sc.c_z,
                                            num_heads=sc.num_attention_heads,
                                            inf=1e6,
                                            initial_norm=True)
        ref_attn.to("cuda", dtype=torch_dtype)

        load_self_pairwise_attention_weights_torch(ref_attn, weights_and_biases)

        with torch.inference_mode():
            ref_output = ref_attn(s, z, mask)

        trt_output = outputs['output']
        torch.testing.assert_close(trt_output, ref_output, atol=1e-3, rtol=1e-4)

    @pytest.mark.parametrize("sc", [
        TriangleAttentionNodeTestScenario(seq_len=64,
                                          c_in=128,
                                          c_hidden=32,
                                          num_attention_heads=4,
                                          chunk_size=0,
                                          plain_attn_precision="float32",
                                          dtype="float32",
                                          starting=True),
        TriangleAttentionNodeTestScenario(seq_len=32,
                                          c_in=128,
                                          c_hidden=32,
                                          num_attention_heads=4,
                                          chunk_size=32,
                                          plain_attn_precision="float32",
                                          dtype="float32",
                                          starting=True),
        TriangleAttentionNodeTestScenario(seq_len=32,
                                          c_in=128,
                                          c_hidden=32,
                                          num_attention_heads=4,
                                          chunk_size=0,
                                          plain_attn_precision="float32",
                                          dtype="float32",
                                          starting=False),
        TriangleAttentionNodeTestScenario(seq_len=32,
                                          c_in=128,
                                          c_hidden=32,
                                          num_attention_heads=4,
                                          chunk_size=32,
                                          plain_attn_precision="float32",
                                          dtype="float32",
                                          starting=False),
    ])
    def test_triangle_attention_node(self,
                                     sc: TriangleAttentionNodeTestScenario):
        if sc.chunk_size > 0:
            pytest.skip(
                "Chunk size is not error yet. NVBUGS: NVBug 5190992"
            )
        self.setUp()
        mean = 0.0
        std_dev = 1 if sc.dtype == "float32" else 0.005
        torch_dtype = str_dtype_to_torch(sc.dtype)
        hidden_states = torch.empty(size=[sc.seq_len, sc.seq_len, sc.c_in],
                                    dtype=torch_dtype,
                                    device="cuda",
                                    requires_grad=False)
        hidden_states.normal_(mean=mean, std=std_dev)

        mask = torch.empty(size=[sc.seq_len, sc.seq_len],
                           dtype=torch_dtype,
                           device="cuda",
                           requires_grad=False)
        mask.normal_(mean=mean, std=std_dev)

        weights_and_biases = \
            create_triangle_attention_node_weights_and_biases(sc.c_in, sc.c_hidden, sc.num_attention_heads, torch_dtype)

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        net.plugin_config.to_legacy_setting()

        with tensorrt_llm.net_guard(net):
            trt_hidden_states = Tensor(name='input_s',
                                       shape=hidden_states.shape,
                                       dtype=tensorrt_llm.str_dtype_to_trt(
                                           sc.dtype))
            trt_mask = Tensor(name='mask',
                              shape=mask.shape,
                              dtype=tensorrt_llm.str_dtype_to_trt(sc.dtype))

            node_type=tensorrt_bionemo.layers.triangle_nodes.TriangleAttentionNodeType.STARTING \
                if sc.starting else tensorrt_bionemo.layers.triangle_nodes.TriangleAttentionNodeType.ENDING
            tri_attn_node = tensorrt_bionemo.layers.triangle_nodes.TriangleAttentionNode(
                c_in=sc.c_in,
                c_hidden=sc.c_hidden,
                num_heads=sc.num_attention_heads,
                local_layer_idx=0,
                dtype=sc.dtype,
                chunk_size=sc.chunk_size,
                node_type=node_type,
            )
            load_triangle_attention_node_weights_trt(tri_attn_node,
                                                     weights_and_biases)

            attention_params = tensorrt_bionemo.layers.attention.AttentionParams(
                plain_attn_precision=sc.plain_attn_precision)
            output = tri_attn_node(trt_hidden_states, trt_mask,
                                   attention_params)
            output.mark_output("output",
                               tensorrt_llm.str_dtype_to_trt(sc.dtype))
        builder_config = builder.create_builder_config(
            name="triangle_attention_node", precision=sc.dtype)

        # Build engine
        engine_buffer = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(
            engine_buffer)
        stream = torch.cuda.current_stream().cuda_stream
        inputs = {'input_s': hidden_states, 'mask': mask}
        outputs = {
            'output':
            torch.empty(hidden_states.shape,
                        dtype=tensorrt_llm._utils.str_dtype_to_torch(sc.dtype),
                        device="cuda")
        }
        session.run(inputs=inputs, outputs=outputs, stream=stream)
        torch.cuda.synchronize()

        ref_node = RefTriangleAttentionNode(c_in=sc.c_in,
                                            c_hidden=sc.c_hidden,
                                            num_heads=sc.num_attention_heads,
                                            starting=sc.starting)
        ref_node.to("cuda", dtype=torch_dtype)

        load_triangle_attention_node_weights_torch(ref_node, weights_and_biases)

        with torch.inference_mode():
            ref_output = ref_node(hidden_states, mask)

        trt_output = outputs['output']
        torch.testing.assert_close(trt_output, ref_output, atol=1e-3, rtol=1e-4)
