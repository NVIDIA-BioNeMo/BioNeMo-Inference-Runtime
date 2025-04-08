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

import numpy as np
import torch
from tensorrt_llm.models.convert_utils import split


def create_triangle_attention_weights(c_q, c_k, c_v, torch_dtype):
    q_weight = torch.empty(size=[c_q, c_q], dtype=torch_dtype)
    torch.nn.init.xavier_uniform_(q_weight)

    # The reason why chose the identity matrix for K and V,
    # see tensorrt_llm/tests/test_layer.py::TestLayer::test_attention
    eye_weight = torch.eye(c_k, dtype=torch_dtype)
    k_weight = eye_weight
    v_weight = eye_weight
    out_weight = eye_weight
    gating_weight = eye_weight

    return q_weight, k_weight, v_weight, out_weight, gating_weight


def load_triangle_attention_weights_torch(module, weights_and_biases):
    q_weight, k_weight, v_weight, out_weight, gating_weight = weights_and_biases
    q_weight.to("cuda")
    k_weight.to("cuda")
    v_weight.to("cuda")
    out_weight.to("cuda")
    gating_weight.to("cuda")

    module.linear_q.weight.data.copy_(q_weight.transpose(1, 0))
    # k,v,o,g are identity matrices
    module.linear_k.weight.data.copy_(k_weight)
    module.linear_v.weight.data.copy_(v_weight)
    module.linear_o.weight.data.copy_(out_weight)
    module.linear_g.weight.data.copy_(gating_weight)


def load_triangle_attention_weights_trt(module,
                                        weights_and_biases,
                                        tp_size=1,
                                        tp_rank=0):
    q_weight, k_weight, v_weight, out_weight, gating_weight = weights_and_biases
    if tp_size > 1:
        q_weight = split(q_weight, tp_size, tp_rank, 1)
        k_weight = split(k_weight, tp_size, tp_rank, 1)
        v_weight = split(v_weight, tp_size, tp_rank, 1)
        out_weight = split(out_weight, tp_size, tp_rank, 0)
        gating_weight = split(gating_weight, tp_size, tp_rank, 1)
    qkv_weights = torch.cat([q_weight, k_weight, v_weight], dim=-1)
    module.qkv_proj.weight.value = np.ascontiguousarray(
        qkv_weights.cpu().numpy().transpose(1, 0))
    module.o_proj.weight.value = np.ascontiguousarray(
        out_weight.cpu().numpy().transpose(1, 0))
    module.g_proj.weight.value = np.ascontiguousarray(
        gating_weight.cpu().numpy().transpose(1, 0))


def create_self_pairwise_attention_weights_biases(c_s, c_z, num_attention_heads,
                                                  torch_dtype):
    init_norm_weight = torch.empty(size=[c_s], dtype=torch_dtype)
    torch.nn.init.uniform_(init_norm_weight)
    init_norm_bias = torch.empty(size=[c_s], dtype=torch_dtype)
    torch.nn.init.zeros_(init_norm_bias)

    q_weight = torch.empty(size=[c_s, c_s], dtype=torch_dtype)
    q_bias = torch.empty(size=[c_s], dtype=torch_dtype)
    torch.nn.init.xavier_uniform_(q_weight)
    torch.nn.init.zeros_(q_bias)

    eye_weight = torch.eye(c_s, dtype=torch_dtype)
    k_weight = eye_weight
    v_weight = eye_weight
    o_weight = eye_weight
    g_weight = eye_weight
    z_weight = torch.empty([c_z, num_attention_heads], dtype=torch_dtype)
    torch.nn.init.xavier_uniform_(z_weight)
    norm_z_weight = torch.empty(size=[c_z], dtype=torch_dtype)
    torch.nn.init.uniform_(norm_z_weight)
    norm_z_bias = torch.empty(size=[c_z], dtype=torch_dtype)
    torch.nn.init.zeros_(norm_z_bias)

    return init_norm_weight, init_norm_bias, q_weight, q_bias, k_weight, v_weight, o_weight, g_weight, z_weight, norm_z_weight, norm_z_bias


def load_self_pairwise_attention_weights_trt(module,
                                             weights_and_biases,
                                             tp_size=1,
                                             tp_rank=0):
    init_norm_weight, init_norm_bias, q_weight, q_bias, \
        k_weight, v_weight, o_weight, g_weight, z_weight, norm_z_weight, norm_z_bias = weights_and_biases
    if tp_size > 1:
        q_weight = split(q_weight, tp_size, tp_rank, 1)
        k_weight = split(k_weight, tp_size, tp_rank, 1)
        v_weight = split(v_weight, tp_size, tp_rank, 1)
        o_weight = split(o_weight, tp_size, tp_rank, 0)
        g_weight = split(g_weight, tp_size, tp_rank, 1)
        z_weight = split(z_weight, tp_size, tp_rank, 1)
    module.norm_s.weight.value = np.ascontiguousarray(
        init_norm_weight.cpu().numpy())
    module.norm_s.bias.value = np.ascontiguousarray(
        init_norm_bias.cpu().numpy())
    module.proj_q.weight.value = np.ascontiguousarray(
        q_weight.cpu().numpy().transpose(1, 0))
    module.proj_q.bias.value = np.ascontiguousarray(q_bias.cpu().numpy())
    # k,v,o,g are identity matrices
    module.proj_k.weight.value = np.ascontiguousarray(k_weight.cpu().numpy())
    module.proj_v.weight.value = np.ascontiguousarray(v_weight.cpu().numpy())
    module.proj_o.weight.value = np.ascontiguousarray(o_weight.cpu().numpy())
    module.proj_g.weight.value = np.ascontiguousarray(g_weight.cpu().numpy())
    module.proj_z.weight.value = np.ascontiguousarray(
        z_weight.cpu().numpy().transpose(1, 0))

    module.proj_z_norm.weight.value = np.ascontiguousarray(
        norm_z_weight.cpu().numpy())
    module.proj_z_norm.bias.value = np.ascontiguousarray(
        norm_z_bias.cpu().numpy())


def load_self_pairwise_attention_weights_torch(module, weights_and_biases):
    init_norm_weight, init_norm_bias, q_weight, q_bias, \
        k_weight, v_weight, o_weight, g_weight, z_weight, norm_z_weight, norm_z_bias = weights_and_biases
    init_norm_weight.to("cuda")
    init_norm_bias.to("cuda")
    q_weight.to("cuda")
    q_bias.to("cuda")
    k_weight.to("cuda")
    v_weight.to("cuda")
    o_weight.to("cuda")
    g_weight.to("cuda")
    z_weight.to("cuda")
    norm_z_weight.to("cuda")
    norm_z_bias.to("cuda")

    module.norm_s.weight.data.copy_(init_norm_weight)
    module.norm_s.bias.data.copy_(init_norm_bias)

    module.proj_q.weight.data.copy_(q_weight.transpose(1, 0))
    module.proj_q.bias.data.copy_(q_bias)

    # k,v,o,g are identity matrices
    module.proj_k.weight.data.copy_(k_weight)
    module.proj_v.weight.data.copy_(v_weight)
    module.proj_o.weight.data.copy_(o_weight)
    module.proj_g.weight.data.copy_(g_weight)
    module.proj_z[1].weight.data.copy_(z_weight.transpose(1, 0))

    module.proj_z[0].weight.data.copy_(norm_z_weight)
    module.proj_z[0].bias.data.copy_(norm_z_bias)


def create_triangle_attention_node_weights_and_biases(c_in, c_hidden,
                                                      num_attention_heads,
                                                      torch_dtype):
    layer_norm_weight = torch.empty(size=[c_in], dtype=torch_dtype)
    torch.nn.init.uniform_(layer_norm_weight)
    layer_norm_bias = torch.empty(size=[c_in], dtype=torch_dtype)
    torch.nn.init.zeros_(layer_norm_bias)

    linear_weight = torch.empty(size=[c_in, num_attention_heads],
                                dtype=torch_dtype)
    torch.nn.init.xavier_uniform_(linear_weight)

    mha_weights_and_biases = create_triangle_attention_weights(
        c_in, c_in, c_in, torch_dtype)
    return layer_norm_weight, layer_norm_bias, linear_weight, mha_weights_and_biases


def load_triangle_attention_node_weights_trt(module,
                                             weights_and_biases,
                                             tp_size=1,
                                             tp_rank=0):
    layer_norm_weight, layer_norm_bias, linear_weight, mha_weights_and_biases = weights_and_biases
    load_triangle_attention_weights_trt(module.mha, mha_weights_and_biases,
                                        tp_size, tp_rank)
    module.layer_norm.weight.value = np.ascontiguousarray(
        layer_norm_weight.cpu().numpy())
    module.layer_norm.bias.value = np.ascontiguousarray(
        layer_norm_bias.cpu().numpy())
    linear_weight = split(linear_weight, tp_size, tp_rank, dim=1)
    module.linear.weight.value = np.ascontiguousarray(
        linear_weight.cpu().numpy().transpose(1, 0))


def load_triangle_attention_node_weights_torch(module, weights_and_biases):
    layer_norm_weight, layer_norm_bias, linear_weight, mha_weights_and_biases = weights_and_biases
    layer_norm_weight.to("cuda")
    layer_norm_bias.to("cuda")
    linear_weight.to("cuda")
    load_triangle_attention_weights_torch(module.mha, mha_weights_and_biases)
    module.layer_norm.weight.data.copy_(layer_norm_weight)
    module.layer_norm.bias.data.copy_(layer_norm_bias)
    module.linear.weight.data.copy_(linear_weight.transpose(1, 0))


def create_triangle_multiplication_node_weights_and_biases(dim, torch_dtype):
    norm_in_weight = torch.empty(size=[dim], dtype=torch_dtype)
    torch.nn.init.uniform_(norm_in_weight)
    norm_in_bias = torch.empty(size=[dim], dtype=torch_dtype)
    torch.nn.init.zeros_(norm_in_bias)

    p_in_weight = torch.rand(2 * dim, dim, dtype=torch_dtype)
    g_in_weight = torch.rand(2 * dim, dim, dtype=torch_dtype)

    norm_out_weight = torch.empty(size=[dim], dtype=torch_dtype)
    torch.nn.init.uniform_(norm_out_weight)
    norm_out_bias = torch.empty(size=[dim], dtype=torch_dtype)
    torch.nn.init.zeros_(norm_out_bias)

    p_out_weight = torch.rand(dim, dim, dtype=torch_dtype)
    g_out_weight = torch.rand(dim, dim, dtype=torch_dtype)

    return norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight, norm_out_bias, p_out_weight, g_out_weight


def load_triangle_multiplication_node_weights_trt(module,
                                                  weights_and_biases,
                                                  tp_size=1,
                                                  tp_rank=0):
    norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight, norm_out_bias, p_out_weight, g_out_weight = weights_and_biases
    dim = p_in_weight.shape[0] // 2
    if tp_size > 1:
        p0_weight = p_in_weight[:dim, :]
        p1_weight = p_in_weight[dim:, :]
        g0_weight = g_in_weight[:dim, :]
        g1_weight = g_in_weight[dim:, :]
        p0_weight = split(p0_weight, tp_size, tp_rank, 0)
        p1_weight = split(p1_weight, tp_size, tp_rank, 0)
        g0_weight = split(g0_weight, tp_size, tp_rank, 0)
        g1_weight = split(g1_weight, tp_size, tp_rank, 0)

        p_in_weight = torch.cat([p0_weight, p1_weight], dim=0)
        g_in_weight = torch.cat([g0_weight, g1_weight], dim=0)
        p_out_weight = split(p_out_weight, tp_size, tp_rank, 0)
        g_out_weight = split(g_out_weight, tp_size, tp_rank, 0)

    module.norm_in.weight.value = np.ascontiguousarray(
        norm_in_weight.cpu().numpy())
    module.norm_in.bias.value = np.ascontiguousarray(norm_in_bias.cpu().numpy())
    module.p_in.weight.value = np.ascontiguousarray(p_in_weight.cpu().numpy())
    module.g_in.weight.value = np.ascontiguousarray(g_in_weight.cpu().numpy())

    module.norm_out.weight.value = np.ascontiguousarray(
        norm_out_weight.cpu().numpy())
    module.norm_out.bias.value = np.ascontiguousarray(
        norm_out_bias.cpu().numpy())
    module.p_out.weight.value = np.ascontiguousarray(p_out_weight.cpu().numpy())
    module.g_out.weight.value = np.ascontiguousarray(g_out_weight.cpu().numpy())


def load_triangle_multiplication_node_weights_torch(module, weights_and_biases):
    norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight, norm_out_bias, p_out_weight, g_out_weight = weights_and_biases
    norm_in_weight.to("cuda")
    norm_in_bias.to("cuda")
    p_in_weight.to("cuda")
    g_in_weight.to("cuda")

    module.norm_in.weight.data.copy_(norm_in_weight)
    module.norm_in.bias.data.copy_(norm_in_bias)
    module.p_in.weight.data.copy_(p_in_weight)
    module.g_in.weight.data.copy_(g_in_weight)

    module.norm_out.weight.data.copy_(norm_out_weight)
    module.norm_out.bias.data.copy_(norm_out_bias)
    module.p_out.weight.data.copy_(p_out_weight)
    module.g_out.weight.data.copy_(g_out_weight)
