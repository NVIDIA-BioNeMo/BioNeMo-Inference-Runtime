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
from test_utils.create_and_load_weights import *
from test_utils.ref_layers import (RefTriangleAttentionNode,
                                   RefTriangleMultiplicationNode)

import tensorrt_bionemo
from tensorrt_bionemo.layers.triangle_nodes import (
    TriangleAttentionNode, TriangleAttentionNodeType,
    TriangleMultiplicationNode, TriangleMultiplicationNodeType)

TriangleAttentionNodeTestScenario = namedtuple(
    "TriangleAttentionNodeTestScenario", [
        "chunk_size", "seq_len", "c_in", "c_hidden", "num_attention_heads",
        "plain_attn_precision", "dtype", "starting"
    ])

TriangleMultiplicationNodeTypeTestScenario = namedtuple(
    "TriangleMultiplicationNodeTypeTestScenario",
    ["seq_len", "dim", "dtype", "multiplication_type"])


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
def test_triangle_attention_node(sc: TriangleAttentionNodeTestScenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

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

        node_type = TriangleAttentionNodeType.STARTING \
            if sc.starting else TriangleAttentionNodeType.ENDING
        tri_attn_node = TriangleAttentionNode(
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
        output = tri_attn_node(trt_hidden_states, trt_mask, attention_params)
        output.mark_output("output", tensorrt_llm.str_dtype_to_trt(sc.dtype))
    builder_config = builder.create_builder_config(
        name="triangle_attention_node", precision=sc.dtype)

    # Build engine
    engine_buffer = builder.build_engine(net, builder_config)
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine_buffer)
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


@pytest.mark.parametrize("sc", [
    TriangleMultiplicationNodeTypeTestScenario(
        seq_len=32,
        dim=128,
        dtype="float32",
        multiplication_type=TriangleMultiplicationNodeType.OUTGOING),
    TriangleMultiplicationNodeTypeTestScenario(
        seq_len=32,
        dim=128,
        dtype="float32",
        multiplication_type=TriangleMultiplicationNodeType.INCOMING),
])
def test_triangle_multiplication_node(
        sc: TriangleMultiplicationNodeTypeTestScenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

    mean = 0.0
    std_dev = 1 if sc.dtype == "float32" else 0.005
    torch_dtype = str_dtype_to_torch(sc.dtype)
    hidden_states = torch.empty(size=[sc.seq_len, sc.seq_len, sc.dim],
                                dtype=torch_dtype,
                                device="cuda",
                                requires_grad=False)
    mask = torch.empty(size=[sc.seq_len, sc.seq_len],
                       dtype=torch_dtype,
                       device="cuda",
                       requires_grad=False)
    mask.normal_(mean=mean, std=std_dev)
    weights_and_biases = \
        create_triangle_multiplication_node_weights(sc.dim, torch_dtype)
    # construct trt network
    builder = tensorrt_llm.Builder()
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()

    with tensorrt_llm.net_guard(net):
        trt_hidden_states = Tensor(name='input_x',
                                   shape=hidden_states.shape,
                                   dtype=tensorrt_llm.str_dtype_to_trt(
                                       sc.dtype))
        trt_mask = Tensor(name='mask',
                          shape=mask.shape,
                          dtype=tensorrt_llm.str_dtype_to_trt(sc.dtype))

        node = TriangleMultiplicationNode(
            local_layer_idx=0,
            dim=sc.dim,
            dtype=sc.dtype,
            multiplication_type=sc.multiplication_type,
        )
        load_triangle_multiplication_node_weights_trt(node, weights_and_biases)

        output = node(trt_hidden_states, trt_mask)
        output.mark_output("output", tensorrt_llm.str_dtype_to_trt(sc.dtype))
    builder_config = builder.create_builder_config(name="trimul_node",
                                                   precision=sc.dtype)
    # Build engine
    engine_buffer = builder.build_engine(net, builder_config)
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine_buffer)
    inputs = {'input_x': hidden_states, 'mask': mask}
    outputs = {
        'output':
        torch.empty(hidden_states.shape,
                    dtype=tensorrt_llm._utils.str_dtype_to_torch(sc.dtype),
                    device="cuda")
    }
    stream = torch.cuda.current_stream().cuda_stream
    session.run(inputs=inputs, outputs=outputs, stream=stream)
    torch.cuda.synchronize()

    ref_node = RefTriangleMultiplicationNode(
        dim=sc.dim,
        outgoing=sc.multiplication_type ==
        TriangleMultiplicationNodeType.OUTGOING)
    ref_node.to("cuda", dtype=torch_dtype)

    load_triangle_multiplication_node_weights_ref_torch(ref_node,
                                                        weights_and_biases)

    with torch.inference_mode():
        ref_output = ref_node(hidden_states, mask)

    trt_output = outputs['output']
    torch.testing.assert_close(trt_output, ref_output, atol=1e-3, rtol=1e-4)
