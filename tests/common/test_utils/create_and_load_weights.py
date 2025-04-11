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

from tensorrt_bionemo.mapping import Mapping, create_max_tp_mapping

from .ref_attn import RefPairwiseSelfAttention, RefTriangleAttention
from .ref_layers import (RefPairformerLayer, RefTransition,
                         RefTriangleAttentionNode,
                         RefTriangleMultiplicationNode)


def create_triangle_attention_weights(c_q=None,
                                      c_k=None,
                                      c_v=None,
                                      torch_dtype=None,
                                      from_ref: RefTriangleAttention = None):
    if not from_ref:
        q_weight = torch.empty(size=[c_q, c_q], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(q_weight)

        # The reason why chose the identity matrix for K and V,
        # see tensorrt_llm/tests/test_layer.py::TestLayer::test_attention
        eye_weight = torch.eye(c_k, dtype=torch_dtype)
        k_weight = eye_weight.contiguous()  # clone
        v_weight = eye_weight.contiguous()
        out_weight = eye_weight.contiguous()
        gating_weight = eye_weight.contiguous()
    else:
        q_weight = from_ref.linear_q.weight.data
        k_weight = from_ref.linear_k.weight.data
        v_weight = from_ref.linear_v.weight.data
        out_weight = from_ref.linear_o.weight.data
        gating_weight = from_ref.linear_g.weight.data
    return q_weight, k_weight, v_weight, out_weight, gating_weight


def load_triangle_attention_weights_torch(module,
                                          weights_and_biases,
                                          dtype=torch.float32):
    # Load for _torch module
    q_weight, k_weight, v_weight, out_weight, gating_weight = weights_and_biases
    qkv_weights = [
        {
            "weight": q_weight.to(dtype).to("cuda"),
            "bias": None
        },
        {
            "weight": k_weight.to(dtype).to("cuda"),
            "bias": None
        },
        {
            "weight": v_weight.to(dtype).to("cuda"),
            "bias": None
        },
    ]
    o_proj_weights = [{"weight": out_weight.to(dtype).to("cuda"), "bias": None}]
    g_proj_weights = [{
        "weight": gating_weight.to(dtype).to("cuda"),
        "bias": None
    }]
    module.qkv_proj.load_weights(qkv_weights)
    module.o_proj.load_weights(o_proj_weights)
    module.g_proj.load_weights(g_proj_weights)


def load_triangle_attention_weights_ref_torch(module, weights_and_biases):
    # Load for reference torch
    q_weight, k_weight, v_weight, out_weight, gating_weight = weights_and_biases
    q_weight.to("cuda")
    k_weight.to("cuda")
    v_weight.to("cuda")
    out_weight.to("cuda")
    gating_weight.to("cuda")

    module.linear_q.weight.data.copy_(q_weight)
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
        q_weight = split(q_weight, tp_size, tp_rank, 0)
        k_weight = split(k_weight, tp_size, tp_rank, 0)
        v_weight = split(v_weight, tp_size, tp_rank, 0)
        out_weight = split(out_weight, tp_size, tp_rank, 1)
        gating_weight = split(gating_weight, tp_size, tp_rank, 0)
    qkv_weights = torch.cat([q_weight, k_weight, v_weight], dim=0)
    module.qkv_proj.weight.value = np.ascontiguousarray(
        qkv_weights.cpu().numpy())
    module.o_proj.weight.value = np.ascontiguousarray(out_weight.cpu().numpy())
    module.g_proj.weight.value = np.ascontiguousarray(
        gating_weight.cpu().numpy())


def create_self_pairwise_attention_weights(
        c_s=None,
        c_z=None,
        num_attention_heads=None,
        torch_dtype=None,
        from_ref: RefPairwiseSelfAttention = None):
    if not from_ref:
        init_norm_weight = torch.empty(size=[c_s], dtype=torch_dtype)
        torch.nn.init.uniform_(init_norm_weight)
        init_norm_bias = torch.empty(size=[c_s], dtype=torch_dtype)
        torch.nn.init.zeros_(init_norm_bias)

        q_weight = torch.empty(size=[c_s, c_s], dtype=torch_dtype)
        q_bias = torch.empty(size=[c_s], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(q_weight)
        torch.nn.init.zeros_(q_bias)

        eye_weight = torch.eye(c_s, dtype=torch_dtype)
        k_weight = eye_weight.contiguous()
        v_weight = eye_weight.contiguous()
        o_weight = eye_weight.contiguous()
        g_weight = eye_weight.contiguous()
        z_weight = torch.empty([num_attention_heads, c_z], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(z_weight)
        norm_z_weight = torch.empty(size=[c_z], dtype=torch_dtype)
        torch.nn.init.uniform_(norm_z_weight)
        norm_z_bias = torch.empty(size=[c_z], dtype=torch_dtype)
        torch.nn.init.zeros_(norm_z_bias)
    else:
        init_norm_weight = from_ref.norm_s.weight.data
        init_norm_bias = from_ref.norm_s.bias.data
        q_weight = from_ref.proj_q.weight.data
        q_bias = from_ref.proj_q.bias.data
        k_weight = from_ref.proj_k.weight.data
        v_weight = from_ref.proj_v.weight.data
        o_weight = from_ref.proj_o.weight.data
        g_weight = from_ref.proj_g.weight.data
        z_weight = from_ref.proj_z[1].weight.data
        norm_z_weight = from_ref.proj_z[0].weight.data
        norm_z_bias = from_ref.proj_z[0].bias.data

    return init_norm_weight, init_norm_bias, q_weight, q_bias, k_weight, v_weight, o_weight, g_weight, z_weight, norm_z_weight, norm_z_bias


def load_self_pairwise_attention_weights_trt(module,
                                             weights_and_biases,
                                             tp_size=1,
                                             tp_rank=0):
    init_norm_weight, init_norm_bias, q_weight, q_bias, \
        k_weight, v_weight, o_weight, g_weight, z_weight, norm_z_weight, norm_z_bias = weights_and_biases
    if tp_size > 1:
        q_weight = split(q_weight, tp_size, tp_rank, 0)
        q_bias = split(q_bias, tp_size, tp_rank, 0)
        k_weight = split(k_weight, tp_size, tp_rank, 0)
        v_weight = split(v_weight, tp_size, tp_rank, 0)
        o_weight = split(o_weight, tp_size, tp_rank, 1)
        g_weight = split(g_weight, tp_size, tp_rank, 0)
        z_weight = split(z_weight, tp_size, tp_rank, 0)

    kv_weights = torch.cat([k_weight, v_weight], dim=0)
    module.norm_s.weight.value = np.ascontiguousarray(
        init_norm_weight.cpu().numpy())
    module.norm_s.bias.value = np.ascontiguousarray(
        init_norm_bias.cpu().numpy())
    module.proj_q.weight.value = np.ascontiguousarray(q_weight.cpu().numpy())
    module.proj_q.bias.value = np.ascontiguousarray(q_bias.cpu().numpy())
    # k,v,o,g are identity matrices
    module.proj_kv.weight.value = np.ascontiguousarray(kv_weights.cpu().numpy())
    module.proj_o.weight.value = np.ascontiguousarray(o_weight.cpu().numpy())
    module.proj_g.weight.value = np.ascontiguousarray(g_weight.cpu().numpy())
    module.proj_z.weight.value = np.ascontiguousarray(z_weight.cpu().numpy())

    module.proj_z_norm.weight.value = np.ascontiguousarray(
        norm_z_weight.cpu().numpy())
    module.proj_z_norm.bias.value = np.ascontiguousarray(
        norm_z_bias.cpu().numpy())


def load_self_pairwise_attention_weights_torch(module,
                                               weights_and_biases,
                                               dtype=torch.float32):
    init_norm_weight, init_norm_bias, q_weight, q_bias, \
        k_weight, v_weight, o_weight, g_weight, z_weight, norm_z_weight, norm_z_bias = weights_and_biases
    q_proj_weights = [{
        "weight": q_weight.to(dtype).to("cuda"),
        "bias": q_bias.to(dtype).to("cuda")
    }]
    kv_proj_weights = [{
        "weight": k_weight.to(dtype).to("cuda"),
        "bias": None
    }, {
        "weight": v_weight.to(dtype).to("cuda"),
        "bias": None
    }]
    o_proj_weights = [{"weight": o_weight.to(dtype).to("cuda"), "bias": None}]
    g_proj_weights = [{"weight": g_weight.to(dtype).to("cuda"), "bias": None}]

    z_1_proj_weights = [{"weight": z_weight.to(dtype).to("cuda"), "bias": None}]
    module.norm_s.weight.data.copy_(init_norm_weight.to(dtype).to("cuda"))
    module.norm_s.bias.data.copy_(init_norm_bias.to(dtype).to("cuda"))
    module.proj_q.load_weights(q_proj_weights)
    module.proj_kv.load_weights(kv_proj_weights)
    module.proj_o.load_weights(o_proj_weights)
    module.proj_g.load_weights(g_proj_weights)
    module.proj_z[0].weight.data.copy_(norm_z_weight.to(dtype).to("cuda"))
    module.proj_z[0].bias.data.copy_(norm_z_bias.to(dtype).to("cuda"))
    module.proj_z[1].load_weights(z_1_proj_weights)


def load_self_pairwise_attention_weights_ref_torch(module, weights_and_biases):
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

    module.proj_q.weight.data.copy_(q_weight)
    module.proj_q.bias.data.copy_(q_bias)

    # k,v,o,g are identity matrices
    module.proj_k.weight.data.copy_(k_weight)
    module.proj_v.weight.data.copy_(v_weight)
    module.proj_o.weight.data.copy_(o_weight)
    module.proj_g.weight.data.copy_(g_weight)
    module.proj_z[1].weight.data.copy_(z_weight)

    module.proj_z[0].weight.data.copy_(norm_z_weight)
    module.proj_z[0].bias.data.copy_(norm_z_bias)


def create_triangle_attention_node_weights(
        c_in=None,
        c_hidden=None,
        num_attention_heads=None,
        torch_dtype=None,
        from_ref: RefTriangleAttentionNode = None):
    if not from_ref:
        layer_norm_weight = torch.empty(size=[c_in], dtype=torch_dtype)
        torch.nn.init.uniform_(layer_norm_weight)
        layer_norm_bias = torch.empty(size=[c_in], dtype=torch_dtype)
        torch.nn.init.zeros_(layer_norm_bias)

        linear_weight = torch.empty(size=[num_attention_heads, c_in],
                                    dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(linear_weight)

        mha_weights_and_biases = create_triangle_attention_weights(
            c_in, c_in, c_in, torch_dtype)
        ret = {
            "layer_norm": (layer_norm_weight, layer_norm_bias),
            "linear": linear_weight,
            "mha": mha_weights_and_biases
        }
    else:
        ret = {
            "layer_norm":
            (from_ref.layer_norm.weight.data, from_ref.layer_norm.bias.data),
            "linear":
            from_ref.linear.weight.data,
            "mha":
            create_triangle_attention_weights(from_ref=from_ref.mha)
        }
    return ret


def load_triangle_attention_node_weights_trt(module,
                                             weights_and_biases,
                                             tp_size=1,
                                             tp_rank=0):
    layer_norm_weight, layer_norm_bias = weights_and_biases["layer_norm"]
    linear_weight = weights_and_biases["linear"]
    mha_weights_and_biases = weights_and_biases["mha"]
    load_triangle_attention_weights_trt(module.mha, mha_weights_and_biases,
                                        tp_size, tp_rank)
    module.layer_norm.weight.value = np.ascontiguousarray(
        layer_norm_weight.cpu().numpy())
    module.layer_norm.bias.value = np.ascontiguousarray(
        layer_norm_bias.cpu().numpy())
    linear_weight = split(linear_weight, tp_size, tp_rank, dim=0)
    module.linear.weight.value = np.ascontiguousarray(
        linear_weight.cpu().numpy())


def load_triangle_attention_node_weights_ref_torch(module, weights_and_biases):
    layer_norm_weight, layer_norm_bias = weights_and_biases["layer_norm"]
    linear_weight = weights_and_biases["linear"]
    mha_weights_and_biases = weights_and_biases["mha"]
    layer_norm_weight.to("cuda")
    layer_norm_bias.to("cuda")
    linear_weight.to("cuda")
    load_triangle_attention_weights_ref_torch(module.mha,
                                              mha_weights_and_biases)
    module.layer_norm.weight.data.copy_(layer_norm_weight)
    module.layer_norm.bias.data.copy_(layer_norm_bias)
    module.linear.weight.data.copy_(linear_weight)


def load_triangle_attention_node_weights_torch(module,
                                               weights_and_biases,
                                               dtype=torch.float32):
    layer_norm_weight, layer_norm_bias = weights_and_biases["layer_norm"]
    linear_weight = weights_and_biases["linear"]
    mha_weights_and_biases = weights_and_biases["mha"]
    load_triangle_attention_weights_torch(module.mha, mha_weights_and_biases,
                                          dtype)
    module.layer_norm.weight.data.copy_(layer_norm_weight.to(dtype).to("cuda"))
    module.layer_norm.bias.data.copy_(layer_norm_bias.to(dtype).to("cuda"))
    module.linear.load_weights([{
        "weight": linear_weight.to(dtype).to("cuda"),
    }])


def create_triangle_multiplication_node_weights(
        dim=None,
        torch_dtype=None,
        from_ref: RefTriangleMultiplicationNode = None):
    if not from_ref:
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
    else:
        norm_in_weight = from_ref.norm_in.weight.data
        norm_in_bias = from_ref.norm_in.bias.data
        p_in_weight = from_ref.p_in.weight.data
        g_in_weight = from_ref.g_in.weight.data
        norm_out_weight = from_ref.norm_out.weight.data
        norm_out_bias = from_ref.norm_out.bias.data
        p_out_weight = from_ref.p_out.weight.data
        g_out_weight = from_ref.g_out.weight.data
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


def load_triangle_multiplication_node_weights_ref_torch(module,
                                                        weights_and_biases):
    norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight, norm_out_bias, p_out_weight, g_out_weight = weights_and_biases
    norm_in_weight.to("cuda")
    norm_in_bias.to("cuda")
    p_in_weight.to("cuda")
    g_in_weight.to("cuda")
    norm_out_weight.to("cuda")
    norm_out_bias.to("cuda")
    p_out_weight.to("cuda")
    g_out_weight.to("cuda")

    module.norm_in.weight.data.copy_(norm_in_weight)
    module.norm_in.bias.data.copy_(norm_in_bias)
    module.p_in.weight.data.copy_(p_in_weight)
    module.g_in.weight.data.copy_(g_in_weight)

    module.norm_out.weight.data.copy_(norm_out_weight)
    module.norm_out.bias.data.copy_(norm_out_bias)
    module.p_out.weight.data.copy_(p_out_weight)
    module.g_out.weight.data.copy_(g_out_weight)


def load_triangle_multiplication_node_weights_torch(module,
                                                    weights_and_biases,
                                                    dtype=torch.float32):
    norm_in_weight, norm_in_bias, p_in_weight, g_in_weight, norm_out_weight, norm_out_bias, p_out_weight, g_out_weight = weights_and_biases
    module.norm_in.weight.data.copy_(norm_in_weight.to(dtype).to("cuda"))
    module.norm_in.bias.data.copy_(norm_in_bias.to(dtype).to("cuda"))
    dim = p_in_weight.shape[0] // 2
    p0_weight = p_in_weight[:dim, :]
    p1_weight = p_in_weight[dim:, :]
    p_in_weights = [
        {
            "weight": p0_weight.to(dtype).to("cuda"),
            "bias": None
        },
        {
            "weight": p1_weight.to(dtype).to("cuda"),
            "bias": None
        },
    ]
    module.p_in.load_weights(p_in_weights)
    g0_weight = g_in_weight[:dim, :]
    g1_weight = g_in_weight[dim:, :]
    g_in_weights = [
        {
            "weight": g0_weight.to(dtype).to("cuda"),
            "bias": None
        },
        {
            "weight": g1_weight.to(dtype).to("cuda"),
            "bias": None
        },
    ]
    module.g_in.load_weights(g_in_weights)

    module.p_out.load_weights([{
        "weight":
        p_out_weight.to(torch.float32).to("cuda"),
        "bias":
        None
    }])
    module.g_out.load_weights([{
        "weight":
        g_out_weight.to(torch.float32).to("cuda"),
        "bias":
        None
    }])
    module.norm_out.weight.data.copy_(norm_out_weight.to(dtype).to("cuda"))
    module.norm_out.bias.data.copy_(norm_out_bias.to(dtype).to("cuda"))


def create_transition_weights(dim=None,
                              hidden=None,
                              out_dim=None,
                              torch_dtype=None,
                              from_ref: RefTransition = None):
    if not from_ref:
        if out_dim is None:
            out_dim = dim
        norm_weight = torch.empty(size=[dim], dtype=torch_dtype)
        torch.nn.init.uniform_(norm_weight)
        norm_bias = torch.empty(size=[dim], dtype=torch_dtype)
        torch.nn.init.zeros_(norm_bias)

        fc1_weight = torch.empty(size=[hidden, dim], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(fc1_weight)

        fc2_weight = torch.empty(size=[hidden, dim], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(fc2_weight)

        fc3_weight = torch.empty(size=[out_dim, hidden], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(fc3_weight)
    else:
        norm_weight = from_ref.norm.weight.data
        norm_bias = from_ref.norm.bias.data
        fc1_weight = from_ref.fc1.weight.data
        fc2_weight = from_ref.fc2.weight.data
        fc3_weight = from_ref.fc3.weight.data
    return norm_weight, norm_bias, fc1_weight, fc2_weight, fc3_weight


def load_transition_weights_trt(module,
                                weights_and_biases,
                                tp_size=1,
                                tp_rank=0):
    norm_weight, norm_bias, fc1_weight, fc2_weight, fc3_weight = weights_and_biases
    if tp_size > 1:
        fc1_weight = split(fc1_weight, tp_size, tp_rank, 0)
        fc2_weight = split(fc2_weight, tp_size, tp_rank, 0)
        fc3_weight = split(fc3_weight, tp_size, tp_rank, 1)
    fused_fc2_fc1_weight = torch.cat([fc2_weight, fc1_weight], dim=0)

    module.norm.weight.value = np.ascontiguousarray(norm_weight.cpu().numpy())
    module.norm.bias.value = np.ascontiguousarray(norm_bias.cpu().numpy())
    module.fused_fc2_fc1.weight.value = np.ascontiguousarray(
        fused_fc2_fc1_weight.cpu().numpy())
    module.fc3.weight.value = np.ascontiguousarray(fc3_weight.cpu().numpy())


def load_transition_weights_ref_torch(module, weights_and_biases):
    norm_weight, norm_bias, fc1_weight, fc2_weight, fc3_weight = weights_and_biases
    norm_weight.to("cuda")
    norm_bias.to("cuda")
    fc1_weight.to("cuda")
    fc2_weight.to("cuda")
    fc3_weight.to("cuda")

    module.norm.weight.data.copy_(norm_weight)
    module.norm.bias.data.copy_(norm_bias)
    module.fc1.weight.data.copy_(fc1_weight)
    module.fc2.weight.data.copy_(fc2_weight)
    module.fc3.weight.data.copy_(fc3_weight)


def load_transition_weights_torch(module,
                                  weights_and_biases,
                                  dtype=torch.float32):
    norm_weight, norm_bias, fc1_weight, fc2_weight, fc3_weight = weights_and_biases
    module.norm.weight.data.copy_(norm_weight.to(dtype).to("cuda"))
    module.norm.bias.data.copy_(norm_bias.to(dtype).to("cuda"))
    module.fused_fc2_fc1.load_weights([{
        "weight": fc2_weight.to(dtype).to("cuda"),
        "bias": None
    }, {
        "weight": fc1_weight.to(dtype).to("cuda"),
        "bias": None
    }])
    module.fc3.load_weights([{
        "weight": fc3_weight.to(dtype).to("cuda"),
        "bias": None
    }])


def create_pairformer_layer_weights(token_s=None,
                                    token_z=None,
                                    num_heads=None,
                                    pairwise_head_width=None,
                                    pairwise_num_heads=None,
                                    torch_dtype=None,
                                    from_ref: RefPairformerLayer = None):
    ret = {}
    if not from_ref:
        ret["attention"] = create_self_pairwise_attention_weights(
            c_s=token_s,
            c_z=token_z,
            num_attention_heads=num_heads,
            torch_dtype=torch_dtype)
        ret["tri_mul_out"] = create_triangle_multiplication_node_weights(
            dim=token_z, torch_dtype=torch_dtype)
        ret["tri_mul_in"] = create_triangle_multiplication_node_weights(
            dim=token_z, torch_dtype=torch_dtype)
        ret["tri_attn_start"] = create_triangle_attention_node_weights(
            c_in=token_z,
            c_hidden=pairwise_head_width,
            num_attention_heads=pairwise_num_heads,
            torch_dtype=torch_dtype)
        ret["tri_attn_end"] = create_triangle_attention_node_weights(
            c_in=token_z,
            c_hidden=pairwise_head_width,
            num_attention_heads=pairwise_num_heads,
            torch_dtype=torch_dtype)
        ret["transition_s"] = create_transition_weights(dim=token_s,
                                                        hidden=token_s * 4,
                                                        out_dim=token_s,
                                                        torch_dtype=torch_dtype)
        ret["transition_z"] = create_transition_weights(dim=token_z,
                                                        hidden=token_z * 4,
                                                        out_dim=token_z,
                                                        torch_dtype=torch_dtype)
    else:
        ret["attention"] = create_self_pairwise_attention_weights(
            from_ref=from_ref.attention)
        ret["tri_mul_out"] = create_triangle_multiplication_node_weights(
            from_ref=from_ref.tri_mul_out)
        ret["tri_mul_in"] = create_triangle_multiplication_node_weights(
            from_ref=from_ref.tri_mul_in)
        ret["tri_attn_start"] = create_triangle_attention_node_weights(
            from_ref=from_ref.tri_attn_start)
        ret["tri_attn_end"] = create_triangle_attention_node_weights(
            from_ref=from_ref.tri_attn_end)
        ret["transition_s"] = create_transition_weights(
            from_ref=from_ref.transition_s)
        ret["transition_z"] = create_transition_weights(
            from_ref=from_ref.transition_z)
    return ret


def load_pairformer_layer_weights_trt(
        module,
        weights_and_biases,
        tp_size=1,
        tp_rank=0,
        mapping: Mapping = None,
        num_heads=None,
        token_s=None,
        token_z=None,
        max_attention_pairwise_tp_size: bool = False,
        max_transition_tp_size: bool = False):
    m = mapping
    if max_attention_pairwise_tp_size:
        m = create_max_tp_mapping(mapping, num_heads)
    load_self_pairwise_attention_weights_trt(module.attention,
                                             weights_and_biases["attention"],
                                             m.tp_size, m.tp_rank)
    load_triangle_multiplication_node_weights_trt(
        module.tri_mul_out, weights_and_biases["tri_mul_out"], tp_size, tp_rank)
    load_triangle_multiplication_node_weights_trt(
        module.tri_mul_in, weights_and_biases["tri_mul_in"], tp_size, tp_rank)
    load_triangle_attention_node_weights_trt(
        module.tri_attn_start, weights_and_biases["tri_attn_start"], tp_size,
        tp_rank)
    load_triangle_attention_node_weights_trt(module.tri_attn_end,
                                             weights_and_biases["tri_attn_end"],
                                             tp_size, tp_rank)
    m = mapping
    if max_transition_tp_size:
        m = create_max_tp_mapping(mapping, token_s * 4)
    load_transition_weights_trt(module.transition_s,
                                weights_and_biases["transition_s"], m.tp_size,
                                m.tp_rank)
    m = mapping
    if max_transition_tp_size:
        m = create_max_tp_mapping(mapping, token_z * 4)
    load_transition_weights_trt(module.transition_z,
                                weights_and_biases["transition_z"], m.tp_size,
                                m.tp_rank)


def load_pairformer_layer_weights_ref_torch(module, weights_and_biases):
    load_self_pairwise_attention_weights_ref_torch(
        module.attention, weights_and_biases["attention"])
    load_triangle_multiplication_node_weights_ref_torch(
        module.tri_mul_out, weights_and_biases["tri_mul_out"])
    load_triangle_multiplication_node_weights_ref_torch(
        module.tri_mul_in, weights_and_biases["tri_mul_in"])
    load_triangle_attention_node_weights_ref_torch(
        module.tri_attn_start, weights_and_biases["tri_attn_start"])
    load_triangle_attention_node_weights_ref_torch(
        module.tri_attn_end, weights_and_biases["tri_attn_end"])
    load_transition_weights_ref_torch(module.transition_s,
                                      weights_and_biases["transition_s"])
    load_transition_weights_ref_torch(module.transition_z,
                                      weights_and_biases["transition_z"])


def load_pairformer_layer_weights_torch(module,
                                        weights_and_biases,
                                        dtype=torch.float32):
    load_self_pairwise_attention_weights_torch(module.attention,
                                               weights_and_biases["attention"],
                                               dtype)
    load_triangle_multiplication_node_weights_torch(
        module.tri_mul_out, weights_and_biases["tri_mul_out"], dtype)
    load_triangle_multiplication_node_weights_torch(
        module.tri_mul_in, weights_and_biases["tri_mul_in"], dtype)
    load_triangle_attention_node_weights_torch(
        module.tri_attn_start, weights_and_biases["tri_attn_start"], dtype)
    load_triangle_attention_node_weights_torch(
        module.tri_attn_end, weights_and_biases["tri_attn_end"], dtype)
    load_transition_weights_torch(module.transition_s,
                                  weights_and_biases["transition_s"], dtype)
    load_transition_weights_torch(module.transition_z,
                                  weights_and_biases["transition_z"], dtype)
