# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Upstream OpenFold/Boltz reference implementation, mirrored for parity tests.
# Kept in upstream style (star imports, forward refs), not held to these rules.
# ruff: noqa: B007, F403, F405, F841

import numpy as np
import torch

from .ref_attn import *
from .ref_layers import *


def create_triangle_attention_weights(
    c_q=None,
    c_k=None,
    c_v=None,
    num_attention_heads: int = None,
    bias_flags=None,
    torch_dtype=None,
    from_ref: RefTriangleAttention = None,
):
    if not from_ref:
        if bias_flags is None:
            bias_flags = {}
        q_weight = torch.empty(size=[c_q, c_q], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(q_weight)
        q_bias = None
        if bias_flags.get("q", False):
            q_bias = torch.empty(size=[c_q], dtype=torch_dtype)
            torch.nn.init.zeros_(q_bias)

        k_weight = torch.empty(size=[c_k, c_k], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(k_weight)
        k_bias = None
        if bias_flags.get("k", False):
            k_bias = torch.empty(size=[c_k], dtype=torch_dtype)
            torch.nn.init.zeros_(k_bias)

        v_weight = torch.empty(size=[c_v, c_v], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(v_weight)
        v_bias = None
        if bias_flags.get("v", False):
            v_bias = torch.empty(size=[c_v], dtype=torch_dtype)
            torch.nn.init.zeros_(v_bias)

        out_weight = torch.empty(size=[c_q, c_q], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(out_weight)
        out_bias = None
        if bias_flags.get("o", False):
            out_bias = torch.empty(size=[c_q], dtype=torch_dtype)
            torch.nn.init.zeros_(out_bias)

        gating_weight = torch.empty(size=[c_q, c_q], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(gating_weight)
        gating_bias = None
        if bias_flags.get("g", False):
            gating_bias = torch.empty(size=[c_q], dtype=torch_dtype)
            torch.nn.init.zeros_(gating_bias)
    else:
        bias_flags = from_ref.bias_flags
        q_weight = from_ref.linear_q.weight.data
        q_bias = None
        if bias_flags.get("q", False):
            q_bias = from_ref.linear_q.bias.data
        k_weight = from_ref.linear_k.weight.data
        k_bias = None
        if bias_flags.get("k", False):
            k_bias = from_ref.linear_k.bias.data
        v_weight = from_ref.linear_v.weight.data
        v_bias = None
        if bias_flags.get("v", False):
            v_bias = from_ref.linear_v.bias.data
        out_weight = from_ref.linear_o.weight.data
        out_bias = None
        if bias_flags.get("o", False):
            out_bias = from_ref.linear_o.bias.data
        gating_weight = None
        gating_bias = None
        if from_ref.linear_g is not None:
            gating_weight = from_ref.linear_g.weight.data
            if bias_flags.get("g", False):
                gating_bias = from_ref.linear_g.bias.data
    return q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, out_weight, out_bias, gating_weight, gating_bias


def load_triangle_attention_weights_torch(module, weights_and_biases, dtype=torch.float32):
    # Load for _torch module
    q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, out_weight, out_bias, gating_weight, gating_bias = (
        weights_and_biases
    )
    if getattr(module, "qkv_proj", None) is not None:
        qkv_weights = [
            {
                "weight": q_weight.to(dtype).to("cuda"),
                "bias": q_bias.to(dtype).to("cuda") if q_bias is not None else None,
            },
            {
                "weight": k_weight.to(dtype).to("cuda"),
                "bias": k_bias.to(dtype).to("cuda") if k_bias is not None else None,
            },
            {
                "weight": v_weight.to(dtype).to("cuda"),
                "bias": v_bias.to(dtype).to("cuda") if v_bias is not None else None,
            },
        ]
        module.qkv_proj.load_weights(qkv_weights)
    else:
        q_weights = [
            {
                "weight": q_weight.to(dtype).to("cuda"),
                "bias": q_bias.to(dtype).to("cuda") if q_bias is not None else None,
            }
        ]
        kv_weights = [
            {
                "weight": k_weight.to(dtype).to("cuda"),
                "bias": k_bias.to(dtype).to("cuda") if k_bias is not None else None,
            },
            {
                "weight": v_weight.to(dtype).to("cuda"),
                "bias": v_bias.to(dtype).to("cuda") if v_bias is not None else None,
            },
        ]
        module.q_proj.load_weights(q_weights)
        module.kv_proj.load_weights(kv_weights)

    o_proj_weights = [
        {
            "weight": out_weight.to(dtype).to("cuda"),
            "bias": out_bias.to(dtype).to("cuda") if out_bias is not None else None,
        }
    ]
    g_proj_weights = None
    if gating_weight is not None:
        g_proj_weights = [
            {
                "weight": gating_weight.to(dtype).to("cuda"),
                "bias": gating_bias.to(dtype).to("cuda") if gating_bias is not None else None,
            }
        ]

    module.o_proj.load_weights(o_proj_weights)
    if g_proj_weights is not None:
        module.g_proj.load_weights(g_proj_weights)


def load_triangle_attention_weights_ref_torch(module, weights_and_biases):
    # Load for reference torch
    q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, out_weight, out_bias, gating_weight, gating_bias = (
        weights_and_biases
    )
    q_weight.to("cuda")
    if q_bias is not None:
        q_bias.to("cuda")
    k_weight.to("cuda")
    if k_bias is not None:
        k_bias.to("cuda")
    v_weight.to("cuda")
    if v_bias is not None:
        v_bias.to("cuda")
    out_weight.to("cuda")
    if out_bias is not None:
        out_bias.to("cuda")
    gating_weight.to("cuda")
    if gating_bias is not None:
        gating_bias.to("cuda")

    module.linear_q.weight.data.copy_(q_weight)
    if q_bias is not None:
        module.linear_q.bias.data.copy_(q_bias)
    # k,v,o,g are identity matrices
    module.linear_k.weight.data.copy_(k_weight)
    if k_bias is not None:
        module.linear_k.bias.data.copy_(k_bias)
    module.linear_v.weight.data.copy_(v_weight)
    if v_bias is not None:
        module.linear_v.bias.data.copy_(v_bias)
    module.linear_o.weight.data.copy_(out_weight)
    if out_bias is not None:
        module.linear_o.bias.data.copy_(out_bias)
    module.linear_g.weight.data.copy_(gating_weight)
    if gating_bias is not None:
        module.linear_g.bias.data.copy_(gating_bias)


def load_triangle_attention_weights_trt(module, weights_and_biases):
    q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, out_weight, out_bias, gating_weight, gating_bias = (
        weights_and_biases
    )

    qkv_weights = torch.cat([q_weight, k_weight, v_weight], dim=0)
    module.qkv_proj.weight.value = np.ascontiguousarray(qkv_weights.cpu().numpy())
    if q_bias is not None and k_bias is not None and v_bias is not None:
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
        module.qkv_proj.bias.value = np.ascontiguousarray(qkv_bias.cpu().numpy())
    module.o_proj.weight.value = np.ascontiguousarray(out_weight.cpu().numpy())
    if out_bias is not None:
        module.o_proj.bias.value = np.ascontiguousarray(out_bias.cpu().numpy())
    module.g_proj.weight.value = np.ascontiguousarray(gating_weight.cpu().numpy())
    if gating_bias is not None:
        module.g_proj.bias.value = np.ascontiguousarray(gating_bias.cpu().numpy())


def create_self_pairwise_attention_weights(
    c_s=None,
    c_z=None,
    num_attention_heads=None,
    bias_flags=None,
    compute_pair_bias=True,
    initial_norm=True,
    torch_dtype=None,
    from_ref: RefPairwiseSelfAttention = None,
):
    if not from_ref:
        init_norm_weight = None
        init_norm_bias = None
        if bias_flags is None:
            bias_flags = {}
        if initial_norm:
            init_norm_weight = torch.empty(size=[c_s], dtype=torch_dtype)
            torch.nn.init.uniform_(init_norm_weight)
            init_norm_bias = torch.empty(size=[c_s], dtype=torch_dtype)
            torch.nn.init.zeros_(init_norm_bias)

        q_weight = torch.empty(size=[c_s, c_s], dtype=torch_dtype)
        q_bias = None
        torch.nn.init.xavier_uniform_(q_weight)
        if bias_flags.get("q", True):  # default to True for q with the boltz family
            q_bias = torch.empty(size=[c_s], dtype=torch_dtype)
            torch.nn.init.zeros_(q_bias)

        eye_weight = torch.eye(c_s, dtype=torch_dtype)
        k_weight = eye_weight.contiguous()
        k_bias = None
        if bias_flags.get("k", False):
            k_bias = torch.empty(size=[c_s], dtype=torch_dtype)
            torch.nn.init.zeros_(k_bias)
        v_weight = eye_weight.contiguous()
        v_bias = None
        if bias_flags.get("v", False):
            v_bias = torch.empty(size=[c_s], dtype=torch_dtype)
            torch.nn.init.zeros_(v_bias)
        o_weight = eye_weight.contiguous()
        o_bias = None
        if bias_flags.get("o", False):
            o_bias = torch.empty(size=[c_s], dtype=torch_dtype)
            torch.nn.init.zeros_(o_bias)
        g_weight = eye_weight.contiguous()
        g_bias = None
        if bias_flags.get("g", False):
            g_bias = torch.empty(size=[c_s], dtype=torch_dtype)
            torch.nn.init.zeros_(g_bias)
        if compute_pair_bias:
            z_weight = torch.empty([num_attention_heads, c_z], dtype=torch_dtype)
            torch.nn.init.xavier_uniform_(z_weight)
            z_bias = None
            if bias_flags.get("z", False):
                z_bias = torch.empty(size=[c_z], dtype=torch_dtype)
                torch.nn.init.zeros_(z_bias)
            norm_z_weight = torch.empty(size=[c_z], dtype=torch_dtype)
            torch.nn.init.uniform_(norm_z_weight)
            norm_z_bias = torch.empty(size=[c_z], dtype=torch_dtype)
            torch.nn.init.zeros_(norm_z_bias)
        else:
            z_weight = None
            z_bias = None
            norm_z_weight = None
            norm_z_bias = None
    else:
        bias_flags = from_ref.bias_flags
        init_norm_weight = None
        init_norm_bias = None
        if hasattr(from_ref, "norm_s") and from_ref.norm_s is not None:
            init_norm_weight = from_ref.norm_s.weight.data
            init_norm_bias = from_ref.norm_s.bias.data
        q_weight = from_ref.proj_q.weight.data
        q_bias = None
        if bias_flags.get("q", False):
            q_bias = from_ref.proj_q.bias.data
        k_weight = from_ref.proj_k.weight.data
        k_bias = None
        if bias_flags.get("k", False):
            k_bias = from_ref.proj_k.bias.data
        v_weight = from_ref.proj_v.weight.data
        v_bias = None
        if bias_flags.get("v", False):
            v_bias = from_ref.proj_v.bias.data
        o_weight = from_ref.proj_o.weight.data
        o_bias = None
        if bias_flags.get("o", False):
            o_bias = from_ref.proj_o.bias.data
        g_weight = from_ref.proj_g.weight.data
        g_bias = None
        if bias_flags.get("g", False):
            g_bias = from_ref.proj_g.bias.data
        if from_ref.compute_pair_bias:
            z_weight = from_ref.proj_z[1].weight.data
            z_bias = None
            if bias_flags.get("z", False):
                z_bias = from_ref.proj_z[1].bias.data
            norm_z_weight = from_ref.proj_z[0].weight.data

            if from_ref.proj_z[0].bias is not None:
                norm_z_bias = from_ref.proj_z[0].bias.data
            else:
                norm_z_bias = None
        else:
            z_weight = None
            z_bias = None
            norm_z_weight = None
            norm_z_bias = None

    return (
        init_norm_weight,
        init_norm_bias,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        o_weight,
        o_bias,
        g_weight,
        g_bias,
        z_weight,
        z_bias,
        norm_z_weight,
        norm_z_bias,
    )


def load_self_pairwise_attention_weights_trt(module, weights_and_biases):
    (
        init_norm_weight,
        init_norm_bias,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        o_weight,
        o_bias,
        g_weight,
        g_bias,
        z_weight,
        z_bias,
        norm_z_weight,
        norm_z_bias,
    ) = weights_and_biases

    kv_weights = torch.cat([k_weight, v_weight], dim=0)
    if (
        hasattr(module, "norm_s")
        and module.norm_s is not None
        and init_norm_weight is not None
        and init_norm_bias is not None
    ):
        module.norm_s.weight.value = np.ascontiguousarray(init_norm_weight.cpu().numpy())
        module.norm_s.bias.value = np.ascontiguousarray(init_norm_bias.cpu().numpy())
    module.proj_q.weight.value = np.ascontiguousarray(q_weight.cpu().numpy())
    if q_bias is not None:
        module.proj_q.bias.value = np.ascontiguousarray(q_bias.cpu().numpy())
    # k,v,o,g are identity matrices
    module.proj_kv.weight.value = np.ascontiguousarray(kv_weights.cpu().numpy())
    if k_bias is not None and v_bias is not None:
        module.proj_kv.bias.value = np.ascontiguousarray(torch.cat([k_bias, v_bias], dim=0).cpu().numpy())
    module.proj_o.weight.value = np.ascontiguousarray(o_weight.cpu().numpy())
    if o_bias is not None:
        module.proj_o.bias.value = np.ascontiguousarray(o_bias.cpu().numpy())
    module.proj_g.weight.value = np.ascontiguousarray(g_weight.cpu().numpy())
    if g_bias is not None:
        module.proj_g.bias.value = np.ascontiguousarray(g_bias.cpu().numpy())
    if z_weight is not None:
        module.proj_z.weight.value = np.ascontiguousarray(z_weight.cpu().numpy())
        if z_bias is not None:
            module.proj_z.bias.value = np.ascontiguousarray(z_bias.cpu().numpy())
    if norm_z_weight is not None and norm_z_bias is not None:
        module.proj_z_norm.weight.value = np.ascontiguousarray(norm_z_weight.cpu().numpy())
        module.proj_z_norm.bias.value = np.ascontiguousarray(norm_z_bias.cpu().numpy())


def load_self_pairwise_attention_weights_torch(module, weights_and_biases, dtype=torch.float32):
    (
        init_norm_weight,
        init_norm_bias,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        o_weight,
        o_bias,
        g_weight,
        g_bias,
        z_weight,
        z_bias,
        norm_z_weight,
        norm_z_bias,
    ) = weights_and_biases
    q_proj_weights = [
        {"weight": q_weight.to(dtype).to("cuda"), "bias": q_bias.to(dtype).to("cuda") if q_bias is not None else None}
    ]
    kv_proj_weights = [
        {"weight": k_weight.to(dtype).to("cuda"), "bias": k_bias.to(dtype).to("cuda") if k_bias is not None else None},
        {"weight": v_weight.to(dtype).to("cuda"), "bias": v_bias.to(dtype).to("cuda") if v_bias is not None else None},
    ]
    o_proj_weights = [
        {"weight": o_weight.to(dtype).to("cuda"), "bias": o_bias.to(dtype).to("cuda") if o_bias is not None else None}
    ]
    g_proj_weights = [
        {"weight": g_weight.to(dtype).to("cuda"), "bias": g_bias.to(dtype).to("cuda") if g_bias is not None else None}
    ]

    if (
        hasattr(module, "norm_s")
        and module.norm_s is not None
        and init_norm_weight is not None
        and init_norm_bias is not None
    ):
        module.norm_s.weight.data.copy_(init_norm_weight.to(dtype).to("cuda"))
        module.norm_s.bias.data.copy_(init_norm_bias.to(dtype).to("cuda"))
    module.proj_q.load_weights(q_proj_weights)
    module.proj_kv.load_weights(kv_proj_weights)
    module.proj_o.load_weights(o_proj_weights)
    module.proj_g.load_weights(g_proj_weights)

    if norm_z_weight is not None and z_weight is not None:
        z_1_proj_weights = [
            {
                "weight": z_weight.to(dtype).to("cuda"),
                "bias": z_bias.to(dtype).to("cuda") if z_bias is not None else None,
            }
        ]
        module.proj_z[0].weight.data.copy_(norm_z_weight.to(dtype).to("cuda"))
        # OF3 uses LayerNorm(bias=False) so ref has no LN bias to copy.
        # Mod's LN was constructed with default bias=True, so zero it to
        # mimic the bias-free reference.
        if module.proj_z[0].bias is not None:
            if norm_z_bias is not None:
                module.proj_z[0].bias.data.copy_(norm_z_bias.to(dtype).to("cuda"))
            else:
                module.proj_z[0].bias.data.zero_()
        module.proj_z[1].load_weights(z_1_proj_weights)


def load_self_pairwise_attention_weights_ref_torch(module, weights_and_biases):
    (
        init_norm_weight,
        init_norm_bias,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        o_weight,
        o_bias,
        g_weight,
        g_bias,
        z_weight,
        z_bias,
        norm_z_weight,
        norm_z_bias,
    ) = weights_and_biases
    init_norm_weight.to("cuda")
    init_norm_bias.to("cuda")
    q_weight.to("cuda")
    if q_bias is not None:
        q_bias.to("cuda")
    k_weight.to("cuda")
    if k_bias is not None:
        k_bias.to("cuda")
    v_weight.to("cuda")
    if v_bias is not None:
        v_bias.to("cuda")
    o_weight.to("cuda")
    if o_bias is not None:
        o_bias.to("cuda")
    g_weight.to("cuda")
    if g_bias is not None:
        g_bias.to("cuda")
    z_weight.to("cuda")
    if z_bias is not None:
        z_bias.to("cuda")

    if (
        hasattr(module, "norm_s")
        and module.norm_s is not None
        and init_norm_weight is not None
        and init_norm_bias is not None
    ):
        module.norm_s.weight.data.copy_(init_norm_weight)
        module.norm_s.bias.data.copy_(init_norm_bias)

    module.proj_q.weight.data.copy_(q_weight)
    if q_bias is not None:
        module.proj_q.bias.data.copy_(q_bias)

    # k,v,o,g are identity matrices
    module.proj_k.weight.data.copy_(k_weight)
    if k_bias is not None:
        module.proj_k.bias.data.copy_(k_bias)
    module.proj_v.weight.data.copy_(v_weight)
    if v_bias is not None:
        module.proj_v.bias.data.copy_(v_bias)
    module.proj_o.weight.data.copy_(o_weight)
    if o_bias is not None:
        module.proj_o.bias.data.copy_(o_bias)
    module.proj_g.weight.data.copy_(g_weight)
    if g_bias is not None:
        module.proj_g.bias.data.copy_(g_bias)

    if norm_z_weight is not None and norm_z_bias is not None and z_weight is not None:
        norm_z_weight.to("cuda")
        norm_z_bias.to("cuda")
        module.proj_z[1].weight.data.copy_(z_weight)
        if z_bias is not None:
            module.proj_z[1].bias.data.copy_(z_bias)
        module.proj_z[0].weight.data.copy_(norm_z_weight)
        module.proj_z[0].bias.data.copy_(norm_z_bias)


def create_triangle_attention_node_weights(
    c_in=None, c_hidden=None, num_attention_heads=None, torch_dtype=None, from_ref: RefTriangleAttentionNode = None
):
    if not from_ref:
        layer_norm_weight = torch.empty(size=[c_in], dtype=torch_dtype)
        torch.nn.init.uniform_(layer_norm_weight)
        layer_norm_bias = torch.empty(size=[c_in], dtype=torch_dtype)
        torch.nn.init.zeros_(layer_norm_bias)

        linear_weight = torch.empty(size=[num_attention_heads, c_in], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(linear_weight)

        mha_weights_and_biases = create_triangle_attention_weights(c_in, c_in, c_in, torch_dtype)
        ret = {
            "layer_norm": (layer_norm_weight, layer_norm_bias),
            "linear": linear_weight,
            "mha": mha_weights_and_biases,
        }
    else:
        ret = {
            "layer_norm": (from_ref.layer_norm.weight.data, from_ref.layer_norm.bias.data),
            "linear": from_ref.linear.weight.data,
            "mha": create_triangle_attention_weights(from_ref=from_ref.mha),
        }
    return ret


def load_triangle_attention_node_weights_trt(module, weights_and_biases):
    layer_norm_weight, layer_norm_bias = weights_and_biases["layer_norm"]
    linear_weight = weights_and_biases["linear"]
    mha_weights_and_biases = weights_and_biases["mha"]
    load_triangle_attention_weights_trt(module.mha, mha_weights_and_biases)
    module.layer_norm.weight.value = np.ascontiguousarray(layer_norm_weight.cpu().numpy())
    module.layer_norm.bias.value = np.ascontiguousarray(layer_norm_bias.cpu().numpy())
    module.linear.weight.value = np.ascontiguousarray(linear_weight.cpu().numpy())


def load_triangle_attention_node_weights_ref_torch(module, weights_and_biases):
    layer_norm_weight, layer_norm_bias = weights_and_biases["layer_norm"]
    linear_weight = weights_and_biases["linear"]
    mha_weights_and_biases = weights_and_biases["mha"]
    layer_norm_weight.to("cuda")
    layer_norm_bias.to("cuda")
    linear_weight.to("cuda")
    load_triangle_attention_weights_ref_torch(module.mha, mha_weights_and_biases)
    module.layer_norm.weight.data.copy_(layer_norm_weight)
    module.layer_norm.bias.data.copy_(layer_norm_bias)
    module.linear.weight.data.copy_(linear_weight)


def load_triangle_attention_node_weights_torch(module, weights_and_biases, dtype=torch.float32):
    layer_norm_weight, layer_norm_bias = weights_and_biases["layer_norm"]
    linear_weight = weights_and_biases["linear"]
    mha_weights_and_biases = weights_and_biases["mha"]
    load_triangle_attention_weights_torch(module.mha, mha_weights_and_biases, dtype)
    module.layer_norm.weight.data.copy_(layer_norm_weight.to(dtype).to("cuda"))
    module.layer_norm.bias.data.copy_(layer_norm_bias.to(dtype).to("cuda"))
    module.linear.load_weights(
        [
            {
                "weight": linear_weight.to(dtype).to("cuda"),
            }
        ]
    )


def create_triangle_multiplication_node_weights(
    dim=None, torch_dtype=None, bias_flags=None, from_ref: RefTriangleMultiplicationNode = None
):
    if not from_ref:
        norm_in_weight = torch.empty(size=[dim], dtype=torch_dtype)
        torch.nn.init.uniform_(norm_in_weight)
        norm_in_bias = torch.empty(size=[dim], dtype=torch_dtype)
        torch.nn.init.zeros_(norm_in_bias)
        if bias_flags is None:
            bias_flags = {}
        p_in_weight = torch.rand(2 * dim, dim, dtype=torch_dtype)
        p_in_bias = None
        if bias_flags.get("p_in", False):
            p_in_bias = torch.empty(size=[2 * dim], dtype=torch_dtype)
            torch.nn.init.zeros_(p_in_bias)
        g_in_weight = torch.rand(2 * dim, dim, dtype=torch_dtype)
        g_in_bias = None
        if bias_flags.get("g_in", False):
            g_in_bias = torch.empty(size=[2 * dim], dtype=torch_dtype)
            torch.nn.init.zeros_(g_in_bias)

        norm_out_weight = torch.empty(size=[dim], dtype=torch.float32)
        torch.nn.init.uniform_(norm_out_weight)
        norm_out_bias = torch.empty(size=[dim], dtype=torch.float32)
        torch.nn.init.zeros_(norm_out_bias)

        p_out_weight = torch.rand(dim, dim, dtype=torch.float32)
        p_out_bias = None
        if bias_flags.get("p_out", False):
            p_out_bias = torch.empty(size=[dim], dtype=torch.float32)
            torch.nn.init.zeros_(p_out_bias)
        g_out_weight = torch.rand(dim, dim, dtype=torch.float32)
        g_out_bias = None
        if bias_flags.get("g_out", False):
            g_out_bias = torch.empty(size=[dim], dtype=torch.float32)
            torch.nn.init.zeros_(g_out_bias)
    else:
        bias_flags = from_ref.bias_flags
        norm_in_weight = from_ref.norm_in.weight.data
        if from_ref.norm_in.bias is not None:
            norm_in_bias = from_ref.norm_in.bias.data
        else:
            norm_in_bias = None
        p_in_weight = from_ref.p_in.weight.data
        p_in_bias = None
        if bias_flags.get("p_in", False):
            p_in_bias = from_ref.p_in.bias.data
        g_in_weight = from_ref.g_in.weight.data
        g_in_bias = None
        if bias_flags.get("g_in", False):
            g_in_bias = from_ref.g_in.bias.data
        norm_out_weight = from_ref.norm_out.weight.data
        if from_ref.norm_out.bias is not None:
            norm_out_bias = from_ref.norm_out.bias.data
        else:
            norm_out_bias = None

        p_out_weight = from_ref.p_out.weight.data
        p_out_bias = None
        if bias_flags.get("p_out", False):
            p_out_bias = from_ref.p_out.bias.data
        g_out_weight = from_ref.g_out.weight.data
        g_out_bias = None
        if bias_flags.get("g_out", False):
            g_out_bias = from_ref.g_out.bias.data
    return (
        norm_in_weight,
        norm_in_bias,
        p_in_weight,
        p_in_bias,
        g_in_weight,
        g_in_bias,
        norm_out_weight,
        norm_out_bias,
        p_out_weight,
        p_out_bias,
        g_out_weight,
        g_out_bias,
    )


def load_triangle_multiplication_node_weights_trt(module, weights_and_biases):
    (
        norm_in_weight,
        norm_in_bias,
        p_in_weight,
        p_in_bias,
        g_in_weight,
        g_in_bias,
        norm_out_weight,
        norm_out_bias,
        p_out_weight,
        p_out_bias,
        g_out_weight,
        g_out_bias,
    ) = weights_and_biases
    p_in_weight.shape[0] // 2
    module.norm_in.weight.value = np.ascontiguousarray(norm_in_weight.cpu().numpy())
    module.norm_in.bias.value = np.ascontiguousarray(norm_in_bias.cpu().numpy())
    module.p_in.weight.value = np.ascontiguousarray(p_in_weight.cpu().numpy())
    if p_in_bias is not None:
        module.p_in.bias.value = np.ascontiguousarray(p_in_bias.cpu().numpy())
    module.g_in.weight.value = np.ascontiguousarray(g_in_weight.cpu().numpy())
    if g_in_bias is not None:
        module.g_in.bias.value = np.ascontiguousarray(g_in_bias.cpu().numpy())

    module.norm_out.weight.value = np.ascontiguousarray(norm_out_weight.cpu().numpy())
    module.norm_out.bias.value = np.ascontiguousarray(norm_out_bias.cpu().numpy())
    module.p_out.weight.value = np.ascontiguousarray(p_out_weight.cpu().numpy())
    if p_out_bias is not None:
        module.p_out.bias.value = np.ascontiguousarray(p_out_bias.cpu().numpy())
    module.g_out.weight.value = np.ascontiguousarray(g_out_weight.cpu().numpy())
    if g_out_bias is not None:
        module.g_out.bias.value = np.ascontiguousarray(g_out_bias.cpu().numpy())


def load_triangle_multiplication_node_weights_ref_torch(module, weights_and_biases):
    (
        norm_in_weight,
        norm_in_bias,
        p_in_weight,
        p_in_bias,
        g_in_weight,
        g_in_bias,
        norm_out_weight,
        norm_out_bias,
        p_out_weight,
        p_out_bias,
        g_out_weight,
        g_out_bias,
    ) = weights_and_biases
    norm_in_weight.to("cuda")
    norm_in_bias.to("cuda")
    p_in_weight.to("cuda")
    if p_in_bias is not None:
        p_in_bias.to("cuda")
    g_in_weight.to("cuda")
    if g_in_bias is not None:
        g_in_bias.to("cuda")
    norm_out_weight.to("cuda")
    norm_out_bias.to("cuda")
    p_out_weight.to("cuda")
    if p_out_bias is not None:
        p_out_bias.to("cuda")
    g_out_weight.to("cuda")
    if g_out_bias is not None:
        g_out_bias.to("cuda")

    module.norm_in.weight.data.copy_(norm_in_weight)
    module.norm_in.bias.data.copy_(norm_in_bias)
    module.p_in.weight.data.copy_(p_in_weight)
    if p_in_bias is not None:
        module.p_in.bias.data.copy_(p_in_bias)
    module.g_in.weight.data.copy_(g_in_weight)
    if g_in_bias is not None:
        module.g_in.bias.data.copy_(g_in_bias)

    module.norm_out.weight.data.copy_(norm_out_weight)
    module.norm_out.bias.data.copy_(norm_out_bias)
    module.p_out.weight.data.copy_(p_out_weight)
    if p_out_bias is not None:
        module.p_out.bias.data.copy_(p_out_bias)
    module.g_out.weight.data.copy_(g_out_weight)
    if g_out_bias is not None:
        module.g_out.bias.data.copy_(g_out_bias)


def load_triangle_multiplication_node_weights_torch(module, weights_and_biases, dtype=torch.float32):
    (
        norm_in_weight,
        norm_in_bias,
        p_in_weight,
        p_in_bias,
        g_in_weight,
        g_in_bias,
        norm_out_weight,
        norm_out_bias,
        p_out_weight,
        p_out_bias,
        g_out_weight,
        g_out_bias,
    ) = weights_and_biases
    module.norm_in.weight.data.copy_(norm_in_weight.to(dtype).to("cuda"))
    if norm_in_bias is not None:
        module.norm_in.bias.data.copy_(norm_in_bias.to(dtype).to("cuda"))
    dim = p_in_weight.shape[0] // 2
    p0_weight = p_in_weight[:dim, :]
    p1_weight = p_in_weight[dim:, :]
    p0_bias = None
    p1_bias = None
    if p_in_bias is not None:
        p0_bias = p_in_bias[:dim]
        p1_bias = p_in_bias[dim:]
    g0_bias = None
    g1_bias = None
    if g_in_bias is not None:
        g0_bias = g_in_bias[:dim]
        g1_bias = g_in_bias[dim:]
    p_in_weights = [
        {"weight": p0_weight.to(dtype).to("cuda"), "bias": p0_bias},
        {"weight": p1_weight.to(dtype).to("cuda"), "bias": p1_bias},
    ]
    module.p_in.load_weights(p_in_weights)
    g0_weight = g_in_weight[:dim, :]
    g1_weight = g_in_weight[dim:, :]
    g_in_weights = [
        {"weight": g0_weight.to(dtype).to("cuda"), "bias": g0_bias},
        {"weight": g1_weight.to(dtype).to("cuda"), "bias": g1_bias},
    ]
    module.g_in.load_weights(g_in_weights)

    module.p_out.load_weights([{"weight": p_out_weight.to(torch.float32).to("cuda"), "bias": p_out_bias}])
    module.g_out.load_weights([{"weight": g_out_weight.to(torch.float32).to("cuda"), "bias": g_out_bias}])
    module.norm_out.weight.data.copy_(norm_out_weight.to(torch.float32).to("cuda"))
    if norm_out_bias is not None:
        module.norm_out.bias.data.copy_(norm_out_bias.to(torch.float32).to("cuda"))


def create_transition_weights(dim=None, hidden=None, out_dim=None, torch_dtype=None, from_ref: RefTransition = None):
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


def load_transition_weights_trt(module, weights_and_biases):
    norm_weight, norm_bias, fc1_weight, fc2_weight, fc3_weight = weights_and_biases
    fused_fc2_fc1_weight = torch.cat([fc2_weight, fc1_weight], dim=0)

    module.norm.weight.value = np.ascontiguousarray(norm_weight.cpu().numpy())
    module.norm.bias.value = np.ascontiguousarray(norm_bias.cpu().numpy())
    module.fused_fc2_fc1.weight.value = np.ascontiguousarray(fused_fc2_fc1_weight.cpu().numpy())
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


def load_transition_weights_torch(module, weights_and_biases, dtype=torch.float32):
    norm_weight, norm_bias, fc1_weight, fc2_weight, fc3_weight = weights_and_biases
    module.norm.weight.data.copy_(norm_weight.to(dtype).to("cuda"))
    module.norm.bias.data.copy_(norm_bias.to(dtype).to("cuda"))
    module.fused_fc2_fc1.load_weights(
        [
            {"weight": fc2_weight.to(dtype).to("cuda"), "bias": None},
            {"weight": fc1_weight.to(dtype).to("cuda"), "bias": None},
        ]
    )
    module.fc3.load_weights([{"weight": fc3_weight.to(dtype).to("cuda"), "bias": None}])


def create_pairformer_layer_weights(
    token_s=None,
    token_z=None,
    num_heads=None,
    pairwise_head_width=None,
    pairwise_num_heads=None,
    include_s_path: bool = True,
    torch_dtype=None,
    from_ref: RefPairformerLayer = None,
):
    ret = {}
    if not from_ref:
        if include_s_path:
            ret["attention"] = create_self_pairwise_attention_weights(
                c_s=token_s, c_z=token_z, num_attention_heads=num_heads, torch_dtype=torch_dtype
            )
        ret["tri_mul_out"] = create_triangle_multiplication_node_weights(dim=token_z, torch_dtype=torch_dtype)
        ret["tri_mul_in"] = create_triangle_multiplication_node_weights(dim=token_z, torch_dtype=torch_dtype)
        ret["tri_attn_start"] = create_triangle_attention_node_weights(
            c_in=token_z, c_hidden=pairwise_head_width, num_attention_heads=pairwise_num_heads, torch_dtype=torch_dtype
        )
        ret["tri_attn_end"] = create_triangle_attention_node_weights(
            c_in=token_z, c_hidden=pairwise_head_width, num_attention_heads=pairwise_num_heads, torch_dtype=torch_dtype
        )
        if include_s_path:
            ret["transition_s"] = create_transition_weights(
                dim=token_s, hidden=token_s * 4, out_dim=token_s, torch_dtype=torch_dtype
            )
        ret["transition_z"] = create_transition_weights(
            dim=token_z, hidden=token_z * 4, out_dim=token_z, torch_dtype=torch_dtype
        )
    else:
        if include_s_path:
            ret["attention"] = create_self_pairwise_attention_weights(from_ref=from_ref.attention)
        ret["tri_mul_out"] = create_triangle_multiplication_node_weights(from_ref=from_ref.tri_mul_out)
        ret["tri_mul_in"] = create_triangle_multiplication_node_weights(from_ref=from_ref.tri_mul_in)
        ret["tri_attn_start"] = create_triangle_attention_node_weights(from_ref=from_ref.tri_attn_start)
        ret["tri_attn_end"] = create_triangle_attention_node_weights(from_ref=from_ref.tri_attn_end)
        if include_s_path:
            ret["transition_s"] = create_transition_weights(from_ref=from_ref.transition_s)
        ret["transition_z"] = create_transition_weights(from_ref=from_ref.transition_z)
    return ret


def load_pairformer_layer_weights_trt(
    module, weights_and_biases, num_heads=None, token_s=None, token_z=None, include_s_path: bool = True
):
    if include_s_path:
        load_self_pairwise_attention_weights_trt(module.attention, weights_and_biases["attention"])

    load_triangle_multiplication_node_weights_trt(module.tri_mul_out, weights_and_biases["tri_mul_out"])
    load_triangle_multiplication_node_weights_trt(module.tri_mul_in, weights_and_biases["tri_mul_in"])

    load_triangle_attention_node_weights_trt(module.tri_attn_start, weights_and_biases["tri_attn_start"])
    load_triangle_attention_node_weights_trt(module.tri_attn_end, weights_and_biases["tri_attn_end"])
    if include_s_path:
        load_transition_weights_trt(module.transition_s, weights_and_biases["transition_s"])
    load_transition_weights_trt(module.transition_z, weights_and_biases["transition_z"])


def load_pairformer_layer_weights_ref_torch(module, weights_and_biases):
    if module.attention:
        load_self_pairwise_attention_weights_ref_torch(module.attention, weights_and_biases["attention"])
    load_triangle_multiplication_node_weights_ref_torch(module.tri_mul_out, weights_and_biases["tri_mul_out"])
    load_triangle_multiplication_node_weights_ref_torch(module.tri_mul_in, weights_and_biases["tri_mul_in"])
    load_triangle_attention_node_weights_ref_torch(module.tri_attn_start, weights_and_biases["tri_attn_start"])
    load_triangle_attention_node_weights_ref_torch(module.tri_attn_end, weights_and_biases["tri_attn_end"])
    if module.transition_s:
        load_transition_weights_ref_torch(module.transition_s, weights_and_biases["transition_s"])
    load_transition_weights_ref_torch(module.transition_z, weights_and_biases["transition_z"])


def load_pairformer_layer_weights_torch(module, weights_and_biases, dtype=torch.float32):
    if hasattr(module, "attention"):
        load_self_pairwise_attention_weights_torch(module.attention, weights_and_biases["attention"], dtype)
    load_triangle_multiplication_node_weights_torch(module.tri_mul_out, weights_and_biases["tri_mul_out"], dtype)
    load_triangle_multiplication_node_weights_torch(module.tri_mul_in, weights_and_biases["tri_mul_in"], dtype)
    load_triangle_attention_node_weights_torch(module.tri_attn_start, weights_and_biases["tri_attn_start"], dtype)
    load_triangle_attention_node_weights_torch(module.tri_attn_end, weights_and_biases["tri_attn_end"], dtype)
    if hasattr(module, "transition_s"):
        load_transition_weights_torch(module.transition_s, weights_and_biases["transition_s"], dtype)
    load_transition_weights_torch(module.transition_z, weights_and_biases["transition_z"], dtype)


def create_adaln_weights(dim=None, dim_single_cond=None, torch_dtype=None, from_ref: RefAdaLN = None):
    if not from_ref:
        a_norm_weight = torch.empty(size=[dim], dtype=torch_dtype)
        torch.nn.init.constant_(a_norm_weight, 1.0)
        s_norm_weight = torch.empty(size=[dim_single_cond], dtype=torch_dtype)
        torch.nn.init.uniform_(s_norm_weight)
        s_scale_weight = torch.empty(size=[dim, dim_single_cond], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(s_scale_weight)
        s_scale_bias = torch.empty(size=[dim], dtype=torch_dtype)
        torch.nn.init.uniform_(s_scale_bias)
        s_bias_weight = torch.empty(size=[dim, dim_single_cond], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(s_bias_weight)
    else:
        a_norm_weight = from_ref.a_norm.weight.data
        s_norm_weight = from_ref.s_norm.weight.data
        s_scale_weight = from_ref.s_scale.weight.data
        s_scale_bias = from_ref.s_scale.bias.data
        s_bias_weight = from_ref.s_bias.weight.data
    return a_norm_weight, s_norm_weight, s_scale_weight, s_scale_bias, s_bias_weight


def load_adaln_weights_ref_torch(module, weights_and_biases):
    a_norm_weight, s_norm_weight, s_scale_weight, s_scale_bias, s_bias_weight = weights_and_biases
    s_norm_weight.to("cuda")
    s_scale_weight.to("cuda")
    s_scale_bias.to("cuda")
    s_bias_weight.to("cuda")

    module.s_norm.weight.data.copy_(s_norm_weight)
    module.s_scale.weight.data.copy_(s_scale_weight)
    module.s_scale.bias.data.copy_(s_scale_bias)
    module.s_bias.weight.data.copy_(s_bias_weight)


def load_adaln_weights_torch(module, weights_and_biases, dtype=torch.float32):
    a_norm_weight, s_norm_weight, s_scale_weight, s_scale_bias, s_bias_weight = weights_and_biases

    module.s_norm.weight.data.copy_(s_norm_weight)

    module.fused_s_scale_s_bias.load_weights(
        [
            {"weight": s_scale_weight.to(dtype).to("cuda"), "bias": s_scale_bias.to(dtype).to("cuda")},
            {
                "weight": s_bias_weight.to(dtype).to("cuda"),
                "bias": torch.zeros(s_bias_weight.shape[0], dtype=dtype).to("cuda"),  # s_bias has no bias
            },
        ]
    )


def load_adaln_weights_trt(module, weights_and_biases):
    a_norm_weight, s_norm_weight, s_scale_weight, s_scale_bias, s_bias_weight = weights_and_biases
    s_bias_bias = torch.zeros(s_bias_weight.shape[0], dtype=s_bias_weight.dtype, device=s_bias_weight.device)

    module.s_norm.weight.value = np.ascontiguousarray(s_norm_weight.cpu().numpy())
    fused_s_scale_s_bias_weight = torch.cat([s_scale_weight, s_bias_weight], dim=0)
    fused_s_scale_s_bias_bias = torch.cat([s_scale_bias, s_bias_bias], dim=0)

    module.fused_s_scale_s_bias.weight.value = np.ascontiguousarray(fused_s_scale_s_bias_weight.cpu().numpy())
    module.fused_s_scale_s_bias.bias.value = np.ascontiguousarray(fused_s_scale_s_bias_bias.cpu().numpy())


def create_conditioned_transition_block_weights(
    dim_single=None,
    dim_single_cond=None,
    expansion_factor: int = 2,
    torch_dtype=None,
    from_ref: RefConditionedTransitionBlock = None,
):
    if not from_ref:
        dim_inner = int(dim_single * expansion_factor)
        adaln_weights = create_adaln_weights(dim=dim_single, dim_single_cond=dim_single_cond, torch_dtype=torch_dtype)
        swish_gate_weight = torch.empty(size=[dim_inner * 2, dim_single], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(swish_gate_weight)
        a_to_b_weight = torch.empty(size=[dim_inner, dim_single], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(a_to_b_weight)
        b_to_a_weight = torch.empty(size=[dim_single, dim_inner], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(b_to_a_weight)
        output_projection_weight = torch.empty(size=[dim_single, dim_single_cond], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(output_projection_weight)
        output_projection_bias = torch.empty(size=[dim_single], dtype=torch_dtype)
        torch.nn.init.uniform_(output_projection_bias)
    else:
        adaln_weights = create_adaln_weights(from_ref=from_ref.adaln)
        swish_gate_weight = from_ref.swish_gate[0].weight.data
        if hasattr(from_ref, "a_to_b"):
            a_to_b_weight = from_ref.a_to_b.weight.data
        else:
            a_to_b_weight = None
        b_to_a_weight = from_ref.b_to_a.weight.data
        output_projection_weight = from_ref.output_projection[0].weight.data
        output_projection_bias = from_ref.output_projection[0].bias.data
    return (
        adaln_weights,
        swish_gate_weight,
        a_to_b_weight,
        b_to_a_weight,
        output_projection_weight,
        output_projection_bias,
    )


def load_conditioned_transition_block_weights_ref_torch(module, weights_and_biases):
    adaln_weights, swish_gate_weight, a_to_b_weight, b_to_a_weight, output_projection_weight, output_projection_bias = (
        weights_and_biases
    )
    swish_gate_weight.to("cuda")
    a_to_b_weight.to("cuda")
    b_to_a_weight.to("cuda")
    output_projection_weight.to("cuda")
    output_projection_bias.to("cuda")

    load_adaln_weights_ref_torch(module.adaln, adaln_weights)
    module.swish_gate[0].weight.data.copy_(swish_gate_weight)
    module.a_to_b.weight.data.copy_(a_to_b_weight)
    module.b_to_a.weight.data.copy_(b_to_a_weight)
    module.output_projection[0].weight.data.copy_(output_projection_weight)
    module.output_projection[0].bias.data.copy_(output_projection_bias)


def load_conditioned_transition_block_weights_torch(module, weights_and_biases, dtype=torch.float32):
    adaln_weights, swish_gate_weight, a_to_b_weight, b_to_a_weight, output_projection_weight, output_projection_bias = (
        weights_and_biases
    )
    load_adaln_weights_torch(module.adaln, adaln_weights, dtype)

    swish_gate_weight_0, swish_gate_weight_1 = torch.chunk(swish_gate_weight, 2, dim=0)
    if a_to_b_weight is not None:
        module.fused_swl_a_to_b.load_weights(
            [
                {"weight": swish_gate_weight_0.to(dtype).to("cuda"), "bias": None},
                {"weight": swish_gate_weight_1.to(dtype).to("cuda"), "bias": None},
                {"weight": a_to_b_weight.to(dtype).to("cuda"), "bias": None},
            ]
        )
    else:
        module.fused_swl_a_to_b.load_weights(
            [
                {"weight": swish_gate_weight_0.to(dtype).to("cuda"), "bias": None},
                {"weight": swish_gate_weight_1.to(dtype).to("cuda"), "bias": None},
            ]
        )
    module.b_to_a.load_weights([{"weight": b_to_a_weight.to(dtype).to("cuda"), "bias": None}])
    module.output_projection.load_weights(
        [{"weight": output_projection_weight.to(dtype).to("cuda"), "bias": output_projection_bias.to(dtype).to("cuda")}]
    )


def load_conditioned_transition_block_weights_trt(module, weights_and_biases):
    adaln_weights, swish_gate_weight, a_to_b_weight, b_to_a_weight, output_projection_weight, output_projection_bias = (
        weights_and_biases
    )

    load_adaln_weights_trt(module.adaln, adaln_weights)

    swish_gate_weight_0, swish_gate_weight_1 = torch.chunk(swish_gate_weight, 2, dim=0)

    if a_to_b_weight is not None:
        fused_swl_a_to_b_weight = torch.cat([swish_gate_weight_0, swish_gate_weight_1, a_to_b_weight], dim=0)
    else:
        # In boltz and openfold3 model the position of gate is different.
        fused_swl_a_to_b_weight = torch.cat([swish_gate_weight_1, swish_gate_weight_0], dim=0)

    module.fused_swl_a_to_b.weight.value = np.ascontiguousarray(fused_swl_a_to_b_weight.cpu().numpy())
    module.b_to_a.weight.value = np.ascontiguousarray(b_to_a_weight.cpu().numpy())
    module.output_projection.weight.value = np.ascontiguousarray(output_projection_weight.cpu().numpy())
    module.output_projection.bias.value = np.ascontiguousarray(output_projection_bias.cpu().numpy())


def create_diffusion_transformer_layer_weights(
    num_heads=None,
    dim=None,
    dim_single_cond=None,
    dim_pairwise=None,
    torch_dtype=None,
    compute_pair_bias=True,
    from_ref: RefDiffusionTransformerLayer = None,
):
    ret = {}
    if not from_ref:
        ret["adaln"] = create_adaln_weights(dim=dim, dim_single_cond=dim_single_cond, torch_dtype=torch_dtype)
        ret["pair_bias_attn"] = create_self_pairwise_attention_weights(
            c_s=dim, c_z=dim_pairwise, num_attention_heads=num_heads, torch_dtype=torch_dtype
        )
        ret["transition"] = create_conditioned_transition_block_weights(
            dim_single=dim, dim_single_cond=dim_single_cond, torch_dtype=torch_dtype
        )
        output_projection_weight = torch.empty(size=[dim, dim_single_cond], dtype=torch_dtype)
        torch.nn.init.xavier_uniform_(output_projection_weight)
        output_projection_bias = torch.empty(size=[dim], dtype=torch_dtype)
        torch.nn.init.uniform_(output_projection_bias)
        ret["output_projection"] = (output_projection_weight, output_projection_bias)
    else:
        ret["adaln"] = create_adaln_weights(from_ref=from_ref.adaln)

        ret["pair_bias_attn"] = create_self_pairwise_attention_weights(
            from_ref=from_ref.pair_bias_attn, compute_pair_bias=compute_pair_bias
        )

        ret["transition"] = create_conditioned_transition_block_weights(from_ref=from_ref.transition)

        ret["output_projection"] = (from_ref.output_projection[0].weight.data, from_ref.output_projection[0].bias.data)
    return ret


def load_diffusion_transformer_layer_weights_ref_torch(module, weights_and_biases):
    load_adaln_weights_ref_torch(module.adaln, weights_and_biases["adaln"])
    load_self_pairwise_attention_weights_ref_torch(module.pair_bias_attn, weights_and_biases["pair_bias_attn"])
    load_conditioned_transition_block_weights_ref_torch(module.transition, weights_and_biases["transition"])
    module.output_projection[0].weight.data.copy_(weights_and_biases["output_projection"][0])
    module.output_projection[0].bias.data.copy_(weights_and_biases["output_projection"][1])


def load_diffusion_transformer_layer_weights_torch(module, weights_and_biases, dtype=torch.float32):
    load_adaln_weights_torch(module.adaln, weights_and_biases["adaln"], dtype)
    load_self_pairwise_attention_weights_torch(module.pair_bias_attn, weights_and_biases["pair_bias_attn"], dtype)
    load_conditioned_transition_block_weights_torch(module.transition, weights_and_biases["transition"], dtype)
    module.output_projection.load_weights(
        [
            {
                "weight": weights_and_biases["output_projection"][0].to(dtype).to("cuda"),
                "bias": weights_and_biases["output_projection"][1].to(dtype).to("cuda"),
            }
        ]
    )


def load_diffusion_transformer_layer_weights_trt(module, weights_and_biases):
    load_adaln_weights_trt(module.adaln, weights_and_biases["adaln"])
    load_self_pairwise_attention_weights_trt(module.pair_bias_attn, weights_and_biases["pair_bias_attn"])
    load_conditioned_transition_block_weights_trt(module.transition, weights_and_biases["transition"])

    output_projection_weight = weights_and_biases["output_projection"][0]
    output_projection_bias = weights_and_biases["output_projection"][1]

    module.output_projection.weight.value = np.ascontiguousarray(output_projection_weight.cpu().numpy())
    module.output_projection.bias.value = np.ascontiguousarray(output_projection_bias.cpu().numpy())


# Pairwise conditioning weights
def create_pairwise_conditioning_weights(
    token_z: int = None,
    dim_token_rel_pos_feats: int = None,
    num_transitions: int = 2,
    transition_expansion_factor: int = 2,
    torch_dtype=None,
    from_ref: RefPairwiseConditioning = None,
):
    ret = {}
    if not from_ref:
        ret["init_proj_norm"] = [
            torch.randn(token_z + dim_token_rel_pos_feats, dtype=torch_dtype),
            torch.randn(token_z + dim_token_rel_pos_feats, dtype=torch_dtype),
        ]
        ret["init_proj_linear"] = torch.randn(token_z, token_z + dim_token_rel_pos_feats, dtype=torch_dtype)
        ret["transitions"] = [
            create_transition_weights(
                dim=token_z,
                hidden=token_z * transition_expansion_factor,
                out_dim=token_z,
                torch_dtype=torch_dtype,
            )
            for _ in range(num_transitions)
        ]
    else:
        ret["init_proj_norm"] = [
            from_ref.dim_pairwise_init_proj[0].weight.data,
            from_ref.dim_pairwise_init_proj[0].bias.data,
        ]
        ret["init_proj_linear"] = from_ref.dim_pairwise_init_proj[1].weight.data
        ret["transitions"] = [
            create_transition_weights(from_ref=from_ref.transitions[i]) for i in range(num_transitions)
        ]
    return ret


def load_pairwise_conditioning_weights_ref_torch(module, weights_and_biases):
    init_proj_norm_weight, init_proj_norm_bias = weights_and_biases["init_proj_norm"]
    init_proj_linear_weight = weights_and_biases["init_proj_linear"]
    transitions = weights_and_biases["transitions"]

    module.dim_pairwise_init_proj[0].weight.data.copy_(init_proj_norm_weight)
    module.dim_pairwise_init_proj[0].bias.data.copy_(init_proj_norm_bias)
    module.dim_pairwise_init_proj[1].weight.data.copy_(init_proj_linear_weight)
    for i, transition in enumerate(transitions):
        load_transition_weights_ref_torch(module.transitions[i], transition)


def load_pairwise_conditioning_weights_torch(module, weights_and_biases, dtype=torch.float32):
    init_proj_norm_weight, init_proj_norm_bias = weights_and_biases["init_proj_norm"]
    module.init_proj_norm.weight.data.copy_(init_proj_norm_weight)
    module.init_proj_norm.bias.data.copy_(init_proj_norm_bias)

    init_proj_linear_weight = weights_and_biases["init_proj_linear"]
    module.init_proj_linear.load_weights([{"weight": init_proj_linear_weight.to(dtype).to("cuda"), "bias": None}])

    transitions = weights_and_biases["transitions"]
    for i, transition in enumerate(transitions):
        load_transition_weights_torch(module.transitions[i], transition, dtype)


def load_pairwise_conditioning_weights_trt(module, weights_and_biases):
    init_proj_norm_weight, init_proj_norm_bias = weights_and_biases["init_proj_norm"]
    init_proj_linear_weight = weights_and_biases["init_proj_linear"]
    transitions = weights_and_biases["transitions"]

    module.init_proj_norm.weight.value = np.ascontiguousarray(init_proj_norm_weight.cpu().numpy())
    module.init_proj_norm.bias.value = np.ascontiguousarray(init_proj_norm_bias.cpu().numpy())

    module.init_proj_linear.weight.value = np.ascontiguousarray(init_proj_linear_weight.cpu().numpy())

    for i, transition in enumerate(transitions):
        load_transition_weights_trt(module.transitions[i], transition)


def create_affinity_heads_transformer_weights(
    token_z: int = None, token_s: int = None, torch_dtype=None, from_ref: RefAffinityHeadsTransformer = None
):
    ret = {}
    if not from_ref:
        affinity_out_mlp_linear_0_weight = torch.randn(token_z, token_z, dtype=torch_dtype)
        affinity_out_mlp_linear_0_bias = torch.randn(token_z, dtype=torch_dtype)

        affinity_out_mlp_linear_1 = torch.randn(token_s, token_z, dtype=torch_dtype)
        affinity_out_mlp_linear_1_bias = torch.randn(token_s, dtype=torch_dtype)

        to_affinity_pred_value_0_weight = torch.randn(token_s, token_s, dtype=torch_dtype)
        to_affinity_pred_value_0_bias = torch.randn(token_s, dtype=torch_dtype)

        to_affinity_pred_value_1_weight = torch.randn(token_s, token_s, dtype=torch_dtype)
        to_affinity_pred_value_1_bias = torch.randn(token_s, dtype=torch_dtype)

        to_affinity_pred_value_2_weight = torch.randn(1, token_s, dtype=torch_dtype)
        to_affinity_pred_value_2_bias = torch.randn(1, dtype=torch_dtype)

        to_affinity_pred_score_0_weight = torch.randn(token_s, token_s, dtype=torch_dtype)
        to_affinity_pred_score_0_bias = torch.randn(token_s, dtype=torch_dtype)

        to_affinity_pred_score_1_weight = torch.randn(token_s, token_s, dtype=torch_dtype)
        to_affinity_pred_score_1_bias = torch.randn(token_s, dtype=torch_dtype)

        to_affinity_pred_score_2_weight = torch.randn(1, token_s, dtype=torch_dtype)
        to_affinity_pred_score_2_bias = torch.randn(1, dtype=torch_dtype)

        to_affinity_logits_binary_weight = torch.randn(1, 1, dtype=torch_dtype)
        to_affinity_logits_binary_bias = torch.randn(1, dtype=torch_dtype)

        ret["affinity_out_mlp_linear_0"] = (affinity_out_mlp_linear_0_weight, affinity_out_mlp_linear_0_bias)
        ret["affinity_out_mlp_linear_1"] = (affinity_out_mlp_linear_1, affinity_out_mlp_linear_1_bias)
        ret["to_affinity_pred_value_0"] = (to_affinity_pred_value_0_weight, to_affinity_pred_value_0_bias)
        ret["to_affinity_pred_value_1"] = (to_affinity_pred_value_1_weight, to_affinity_pred_value_1_bias)
        ret["to_affinity_pred_value_2"] = (to_affinity_pred_value_2_weight, to_affinity_pred_value_2_bias)
        ret["to_affinity_pred_score_0"] = (to_affinity_pred_score_0_weight, to_affinity_pred_score_0_bias)
        ret["to_affinity_pred_score_1"] = (to_affinity_pred_score_1_weight, to_affinity_pred_score_1_bias)
        ret["to_affinity_logits_binary"] = (to_affinity_logits_binary_weight, to_affinity_logits_binary_bias)
    else:
        ret["affinity_out_mlp_linear_0"] = (
            from_ref.affinity_out_mlp[0].weight.data,
            from_ref.affinity_out_mlp[0].bias.data,
        )
        ret["affinity_out_mlp_linear_1"] = (
            from_ref.affinity_out_mlp[2].weight.data,
            from_ref.affinity_out_mlp[2].bias.data,
        )
        ret["to_affinity_pred_value_0"] = (
            from_ref.to_affinity_pred_value[0].weight.data,
            from_ref.to_affinity_pred_value[0].bias.data,
        )
        ret["to_affinity_pred_value_1"] = (
            from_ref.to_affinity_pred_value[2].weight.data,
            from_ref.to_affinity_pred_value[2].bias.data,
        )
        ret["to_affinity_pred_value_2"] = (
            from_ref.to_affinity_pred_value[4].weight.data,
            from_ref.to_affinity_pred_value[4].bias.data,
        )
        ret["to_affinity_pred_score_0"] = (
            from_ref.to_affinity_pred_score[0].weight.data,
            from_ref.to_affinity_pred_score[0].bias.data,
        )
        ret["to_affinity_pred_score_1"] = (
            from_ref.to_affinity_pred_score[2].weight.data,
            from_ref.to_affinity_pred_score[2].bias.data,
        )
        ret["to_affinity_pred_score_2"] = (
            from_ref.to_affinity_pred_score[4].weight.data,
            from_ref.to_affinity_pred_score[4].bias.data,
        )
        ret["to_affinity_logits_binary"] = (
            from_ref.to_affinity_logits_binary.weight.data,
            from_ref.to_affinity_logits_binary.bias.data,
        )
    return ret


def load_affinity_heads_transformer_weights_ref_torch(module, weights_and_biases):
    affinity_out_mlp_linear_0_weight, affinity_out_mlp_linear_0_bias = weights_and_biases["affinity_out_mlp_linear_0"]
    module.affinity_out_mlp[0].weight.data.copy_(affinity_out_mlp_linear_0_weight)
    module.affinity_out_mlp[0].bias.data.copy_(affinity_out_mlp_linear_0_bias)

    affinity_out_mlp_linear_1_weight, affinity_out_mlp_linear_1_bias = weights_and_biases["affinity_out_mlp_linear_1"]
    module.affinity_out_mlp[2].weight.data.copy_(affinity_out_mlp_linear_1_weight)
    module.affinity_out_mlp[2].bias.data.copy_(affinity_out_mlp_linear_1_bias)

    to_affinity_pred_value_0_weight, to_affinity_pred_value_0_bias = weights_and_biases["to_affinity_pred_value_0"]
    module.to_affinity_pred_value[0].weight.data.copy_(to_affinity_pred_value_0_weight)
    module.to_affinity_pred_value[0].bias.data.copy_(to_affinity_pred_value_0_bias)

    to_affinity_pred_value_1_weight, to_affinity_pred_value_1_bias = weights_and_biases["to_affinity_pred_value_1"]
    module.to_affinity_pred_value[2].weight.data.copy_(to_affinity_pred_value_1_weight)
    module.to_affinity_pred_value[2].bias.data.copy_(to_affinity_pred_value_1_bias)

    to_affinity_pred_value_2_weight, to_affinity_pred_value_2_bias = weights_and_biases["to_affinity_pred_value_2"]
    module.to_affinity_pred_value[4].weight.data.copy_(to_affinity_pred_value_2_weight)
    module.to_affinity_pred_value[4].bias.data.copy_(to_affinity_pred_value_2_bias)

    to_affinity_pred_score_0_weight, to_affinity_pred_score_0_bias = weights_and_biases["to_affinity_pred_score_0"]
    module.to_affinity_pred_score[0].weight.data.copy_(to_affinity_pred_score_0_weight)
    module.to_affinity_pred_score[0].bias.data.copy_(to_affinity_pred_score_0_bias)

    to_affinity_pred_score_1_weight, to_affinity_pred_score_1_bias = weights_and_biases["to_affinity_pred_score_1"]
    module.to_affinity_pred_score[2].weight.data.copy_(to_affinity_pred_score_1_weight)
    module.to_affinity_pred_score[2].bias.data.copy_(to_affinity_pred_score_1_bias)

    to_affinity_pred_score_2_weight, to_affinity_pred_score_2_bias = weights_and_biases["to_affinity_pred_score_2"]
    module.to_affinity_pred_score[4].weight.data.copy_(to_affinity_pred_score_2_weight)
    module.to_affinity_pred_score[4].bias.data.copy_(to_affinity_pred_score_2_bias)

    to_affinity_logits_binary_weight, to_affinity_logits_binary_bias = weights_and_biases["to_affinity_logits_binary"]
    module.to_affinity_logits_binary.weight.data.copy_(to_affinity_logits_binary_weight)
    module.to_affinity_logits_binary.bias.data.copy_(to_affinity_logits_binary_bias)


def load_affinity_heads_transformer_weights_torch(module, weights_and_biases, dtype=torch.float32):
    affinity_out_mlp_linear_0_weight, affinity_out_mlp_linear_0_bias = weights_and_biases["affinity_out_mlp_linear_0"]
    module.affinity_out_mlp_linear_0.load_weights(
        [
            {
                "weight": affinity_out_mlp_linear_0_weight.to(dtype).to("cuda"),
                "bias": affinity_out_mlp_linear_0_bias.to(dtype).to("cuda"),
            }
        ]
    )
    affinity_out_mlp_linear_1_weight, affinity_out_mlp_linear_1_bias = weights_and_biases["affinity_out_mlp_linear_1"]
    module.affinity_out_mlp_linear_1.load_weights(
        [
            {
                "weight": affinity_out_mlp_linear_1_weight.to(dtype).to("cuda"),
                "bias": affinity_out_mlp_linear_1_bias.to(dtype).to("cuda"),
            }
        ]
    )

    # to_affinity_pred_value
    to_affinity_pred_value_0_weight, to_affinity_pred_value_0_bias = weights_and_biases["to_affinity_pred_value_0"]
    module.to_affinity_pred_value_0.load_weights(
        [
            {
                "weight": to_affinity_pred_value_0_weight.to(dtype).to("cuda"),
                "bias": to_affinity_pred_value_0_bias.to(dtype).to("cuda"),
            }
        ]
    )
    to_affinity_pred_value_1_weight, to_affinity_pred_value_1_bias = weights_and_biases["to_affinity_pred_value_1"]
    module.to_affinity_pred_value_1.load_weights(
        [
            {
                "weight": to_affinity_pred_value_1_weight.to(dtype).to("cuda"),
                "bias": to_affinity_pred_value_1_bias.to(dtype).to("cuda"),
            }
        ]
    )
    to_affinity_pred_value_2_weight, to_affinity_pred_value_2_bias = weights_and_biases["to_affinity_pred_value_2"]
    module.to_affinity_pred_value_2.weight.data.copy_(to_affinity_pred_value_2_weight)
    module.to_affinity_pred_value_2.bias.data.copy_(to_affinity_pred_value_2_bias)

    # to_affinity_pred_score
    to_affinity_pred_score_0_weight, to_affinity_pred_score_0_bias = weights_and_biases["to_affinity_pred_score_0"]
    module.to_affinity_pred_score_0.load_weights(
        [
            {
                "weight": to_affinity_pred_score_0_weight.to(dtype).to("cuda"),
                "bias": to_affinity_pred_score_0_bias.to(dtype).to("cuda"),
            }
        ]
    )
    to_affinity_pred_score_1_weight, to_affinity_pred_score_1_bias = weights_and_biases["to_affinity_pred_score_1"]
    module.to_affinity_pred_score_1.load_weights(
        [
            {
                "weight": to_affinity_pred_score_1_weight.to(dtype).to("cuda"),
                "bias": to_affinity_pred_score_1_bias.to(dtype).to("cuda"),
            }
        ]
    )
    to_affinity_pred_score_2_weight, to_affinity_pred_score_2_bias = weights_and_biases["to_affinity_pred_score_2"]
    module.to_affinity_pred_score_2.weight.data.copy_(to_affinity_pred_score_2_weight)
    module.to_affinity_pred_score_2.bias.data.copy_(to_affinity_pred_score_2_bias)

    # to_affinity_logits_binary
    to_affinity_logits_binary_weight, to_affinity_logits_binary_bias = weights_and_biases["to_affinity_logits_binary"]
    module.to_affinity_logits_binary.weight.data.copy_(to_affinity_logits_binary_weight)
    module.to_affinity_logits_binary.bias.data.copy_(to_affinity_logits_binary_bias)


def load_affinity_heads_transformer_weights_trt(module, weights_and_biases):
    affinity_out_mlp_linear_0_weight, affinity_out_mlp_linear_0_bias = weights_and_biases["affinity_out_mlp_linear_0"]
    affinity_out_mlp_linear_1_weight, affinity_out_mlp_linear_1_bias = weights_and_biases["affinity_out_mlp_linear_1"]
    to_affinity_pred_value_0_weight, to_affinity_pred_value_0_bias = weights_and_biases["to_affinity_pred_value_0"]
    to_affinity_pred_value_1_weight, to_affinity_pred_value_1_bias = weights_and_biases["to_affinity_pred_value_1"]
    to_affinity_pred_value_2_weight, to_affinity_pred_value_2_bias = weights_and_biases["to_affinity_pred_value_2"]
    to_affinity_pred_score_0_weight, to_affinity_pred_score_0_bias = weights_and_biases["to_affinity_pred_score_0"]
    to_affinity_pred_score_1_weight, to_affinity_pred_score_1_bias = weights_and_biases["to_affinity_pred_score_1"]
    to_affinity_pred_score_2_weight, to_affinity_pred_score_2_bias = weights_and_biases["to_affinity_pred_score_2"]
    to_affinity_logits_binary_weight, to_affinity_logits_binary_bias = weights_and_biases["to_affinity_logits_binary"]

    module.affinity_out_mlp_linear_0.weight.value = np.ascontiguousarray(affinity_out_mlp_linear_0_weight.cpu().numpy())
    module.affinity_out_mlp_linear_0.bias.value = np.ascontiguousarray(affinity_out_mlp_linear_0_bias.cpu().numpy())
    module.affinity_out_mlp_linear_1.weight.value = np.ascontiguousarray(affinity_out_mlp_linear_1_weight.cpu().numpy())
    module.affinity_out_mlp_linear_1.bias.value = np.ascontiguousarray(affinity_out_mlp_linear_1_bias.cpu().numpy())

    module.to_affinity_pred_value_0.weight.value = np.ascontiguousarray(to_affinity_pred_value_0_weight.cpu().numpy())
    module.to_affinity_pred_value_0.bias.value = np.ascontiguousarray(to_affinity_pred_value_0_bias.cpu().numpy())
    module.to_affinity_pred_value_1.weight.value = np.ascontiguousarray(to_affinity_pred_value_1_weight.cpu().numpy())
    module.to_affinity_pred_value_1.bias.value = np.ascontiguousarray(to_affinity_pred_value_1_bias.cpu().numpy())
    module.to_affinity_pred_value_2.weight.value = np.ascontiguousarray(to_affinity_pred_value_2_weight.cpu().numpy())
    module.to_affinity_pred_value_2.bias.value = np.ascontiguousarray(to_affinity_pred_value_2_bias.cpu().numpy())

    module.to_affinity_pred_score_0.weight.value = np.ascontiguousarray(to_affinity_pred_score_0_weight.cpu().numpy())
    module.to_affinity_pred_score_0.bias.value = np.ascontiguousarray(to_affinity_pred_score_0_bias.cpu().numpy())
    module.to_affinity_pred_score_1.weight.value = np.ascontiguousarray(to_affinity_pred_score_1_weight.cpu().numpy())
    module.to_affinity_pred_score_1.bias.value = np.ascontiguousarray(to_affinity_pred_score_1_bias.cpu().numpy())
    module.to_affinity_pred_score_2.weight.value = np.ascontiguousarray(to_affinity_pred_score_2_weight.cpu().numpy())
    module.to_affinity_pred_score_2.bias.value = np.ascontiguousarray(to_affinity_pred_score_2_bias.cpu().numpy())

    module.to_affinity_logits_binary.weight.value = np.ascontiguousarray(to_affinity_logits_binary_weight.cpu().numpy())
    module.to_affinity_logits_binary.bias.value = np.ascontiguousarray(to_affinity_logits_binary_bias.cpu().numpy())


def create_affinity_module_weights(
    token_s: int = None,
    token_z: int = None,
    num_dist_bins: int = None,
    pairformer_num_blocks: int = None,
    pairwise_head_width: int = None,
    pairwise_num_heads: int = None,
    torch_dtype=None,
    from_ref: RefAffinityModule = None,
):
    ret = {}
    if not from_ref:
        ret["dist_bin_pairwise_embed"] = torch.randn(num_dist_bins, token_z, dtype=torch_dtype)
        ret["s_to_z_prod_in1"] = torch.randn(token_z, token_s, dtype=torch_dtype)
        ret["s_to_z_prod_in2"] = torch.randn(token_z, token_s, dtype=torch_dtype)
        ret["z_norm"] = [torch.randn(token_z, dtype=torch_dtype), torch.randn(token_z, dtype=torch_dtype)]
        ret["z_linear"] = torch.randn(token_z, token_z, dtype=torch_dtype)
        ret["pairwise_conditioner"] = create_pairwise_conditioning_weights(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=2,
            transition_expansion_factor=2,
            torch_dtype=torch_dtype,
        )
        ret["pairformer_stack"] = [
            create_pairformer_layer_weights(
                token_s=token_s,
                token_z=token_z,
                num_heads=pairwise_num_heads,
                pairwise_head_width=pairwise_head_width,
                pairwise_num_heads=pairwise_num_heads,
                include_s_path=False,
                torch_dtype=torch_dtype,
            )
            for _ in range(pairformer_num_blocks)
        ]
        ret["affinity_heads"] = create_affinity_heads_transformer_weights(
            token_z=token_z,
            token_s=token_s,
            torch_dtype=torch_dtype,
        )
    else:
        ret["dist_bin_pairwise_embed"] = from_ref.dist_bin_pairwise_embed.weight.data
        ret["s_to_z_prod_in1"] = from_ref.s_to_z_prod_in1.weight.data
        ret["s_to_z_prod_in2"] = from_ref.s_to_z_prod_in2.weight.data
        ret["z_norm"] = [from_ref.z_norm.weight.data, from_ref.z_norm.bias.data]
        ret["z_linear"] = from_ref.z_linear.weight.data
        ret["pairwise_conditioner"] = create_pairwise_conditioning_weights(from_ref=from_ref.pairwise_conditioner)
        ret["pairformer_stack"] = [
            create_pairformer_layer_weights(
                from_ref=from_ref.pairformer_stack.layers[i],
                include_s_path=False,
            )
            for i in range(from_ref.pairformer_num_blocks)
        ]
        ret["affinity_heads"] = create_affinity_heads_transformer_weights(from_ref=from_ref.affinity_heads)
    return ret


def load_affinity_module_weights_ref_torch(module, weights_and_biases):
    dist_bin_pairwise_embed_weight = weights_and_biases["dist_bin_pairwise_embed"]
    module.dist_bin_pairwise_embed.weight.data.copy_(dist_bin_pairwise_embed_weight)

    s_to_z_prod_in1_weight = weights_and_biases["s_to_z_prod_in1"]
    module.s_to_z_prod_in1.weight.data.copy_(s_to_z_prod_in1_weight)

    s_to_z_prod_in2_weight = weights_and_biases["s_to_z_prod_in2"]
    module.s_to_z_prod_in2.weight.data.copy_(s_to_z_prod_in2_weight)

    z_norm_weight, z_norm_bias = weights_and_biases["z_norm"]
    module.z_norm.weight.data.copy_(z_norm_weight)
    module.z_norm.bias.data.copy_(z_norm_bias)

    z_linear_weight = weights_and_biases["z_linear"]
    module.z_linear.weight.data.copy_(z_linear_weight)

    pairwise_conditioner_weights = weights_and_biases["pairwise_conditioner"]
    load_pairwise_conditioning_weights_ref_torch(module.pairwise_conditioner, pairwise_conditioner_weights)

    pairformer_stack_weights = weights_and_biases["pairformer_stack"]
    for i in range(len(pairformer_stack_weights)):
        load_pairformer_layer_weights_ref_torch(module.pairformer_stack[i], pairformer_stack_weights[i])

    affinity_heads_weights = weights_and_biases["affinity_heads"]
    load_affinity_heads_transformer_weights_ref_torch(module.affinity_heads, affinity_heads_weights)


def load_affinity_module_weights_torch(module, weights_and_biases, dtype=torch.float32):
    dist_bin_pairwise_embed_weight = weights_and_biases["dist_bin_pairwise_embed"]

    module.dist_bin_pairwise_embed.weight.data.copy_(dist_bin_pairwise_embed_weight.to(dtype).to("cuda"))

    s_to_z_prod_in1_weight = weights_and_biases["s_to_z_prod_in1"]
    s_to_z_prod_in2_weight = weights_and_biases["s_to_z_prod_in2"]
    module.fused_s_to_z.load_weights(
        [
            {"weight": s_to_z_prod_in1_weight.to(dtype).to("cuda"), "bias": None},
            {"weight": s_to_z_prod_in2_weight.to(dtype).to("cuda"), "bias": None},
        ]
    )

    z_norm_weight, z_norm_bias = weights_and_biases["z_norm"]
    module.z_norm.weight.data.copy_(z_norm_weight)
    module.z_norm.bias.data.copy_(z_norm_bias)

    z_linear_weight = weights_and_biases["z_linear"]
    module.z_linear.load_weights([{"weight": z_linear_weight.to(dtype).to("cuda"), "bias": None}])

    pairwise_conditioner_weights = weights_and_biases["pairwise_conditioner"]
    load_pairwise_conditioning_weights_torch(module.pairwise_conditioner, pairwise_conditioner_weights, dtype)

    pairformer_stack_weights = weights_and_biases["pairformer_stack"]
    for i in range(len(pairformer_stack_weights)):
        load_pairformer_layer_weights_torch(module.pairformer_stack.layers[i], pairformer_stack_weights[i], dtype)

    affinity_heads_weights = weights_and_biases["affinity_heads"]
    load_affinity_heads_transformer_weights_torch(module.affinity_heads, affinity_heads_weights, dtype)


def load_affinity_module_weights_trt(module, weights_and_biases):
    dist_bin_pairwise_embed_weight = weights_and_biases["dist_bin_pairwise_embed"]

    module.dist_bin_pairwise_embed.weight.value = np.ascontiguousarray(dist_bin_pairwise_embed_weight.cpu().numpy())

    s_to_z_prod_in1_weight = weights_and_biases["s_to_z_prod_in1"]
    s_to_z_prod_in2_weight = weights_and_biases["s_to_z_prod_in2"]
    fused_s_to_z = torch.cat([s_to_z_prod_in1_weight, s_to_z_prod_in2_weight], dim=0)

    module.fused_s_to_z.weight.value = np.ascontiguousarray(fused_s_to_z.cpu().numpy())

    z_norm_weight, z_norm_bias = weights_and_biases["z_norm"]

    module.z_norm.weight.value = np.ascontiguousarray(z_norm_weight.cpu().numpy())
    module.z_norm.bias.value = np.ascontiguousarray(z_norm_bias.cpu().numpy())

    z_linear_weight = weights_and_biases["z_linear"]

    module.z_linear.weight.value = np.ascontiguousarray(z_linear_weight.cpu().numpy())

    pairwise_conditioner_weights = weights_and_biases["pairwise_conditioner"]
    load_pairwise_conditioning_weights_trt(module.pairwise_conditioner, pairwise_conditioner_weights)

    pairformer_stack_weights = weights_and_biases["pairformer_stack"]
    for i in range(len(pairformer_stack_weights)):
        load_pairformer_layer_weights_trt(
            module.pairformer_stack.layers[i],
            pairformer_stack_weights[i],
            include_s_path=False,
        )

    affinity_heads_weights = weights_and_biases["affinity_heads"]
    load_affinity_heads_transformer_weights_trt(module.affinity_heads, affinity_heads_weights)


def create_pair_weighted_averaging_weights(
    c_m: int = None,
    c_z: int = None,
    c_h: int = None,
    num_heads: int = None,
    torch_dtype=None,
    from_ref: RefPairWeightedAveraging = None,
):
    if not from_ref:
        norm_m_weight = torch.empty(size=[c_m], dtype=torch_dtype)
        torch.nn.init.uniform_(norm_m_weight)
        norm_m_bias = torch.empty(size=[c_m], dtype=torch_dtype)
        torch.nn.init.zeros_(norm_m_bias)
        norm_z_weight = torch.empty(size=[c_z], dtype=torch_dtype)
        torch.nn.init.uniform_(norm_z_weight)
        norm_z_bias = torch.empty(size=[c_z], dtype=torch_dtype)
        torch.nn.init.zeros_(norm_z_bias)
        proj_m_weight = torch.empty(size=[c_m, c_h * num_heads], dtype=torch_dtype)
        torch.nn.init.uniform_(proj_m_weight)
        proj_g_weight = torch.empty(size=[c_m, c_h * num_heads], dtype=torch_dtype)
        torch.nn.init.uniform_(proj_g_weight)
        proj_z_weight = torch.empty(size=[c_z, num_heads], dtype=torch_dtype)
        torch.nn.init.uniform_(proj_z_weight)
        proj_o_weight = torch.empty(size=[c_h * num_heads, c_m], dtype=torch_dtype)
        torch.nn.init.uniform_(proj_o_weight)
    else:
        norm_m_weight = from_ref.norm_m.weight.data
        norm_m_bias = from_ref.norm_m.bias.data
        norm_z_weight = from_ref.norm_z.weight.data
        norm_z_bias = from_ref.norm_z.bias.data
        proj_m_weight = from_ref.proj_m.weight.data
        proj_g_weight = from_ref.proj_g.weight.data
        proj_z_weight = from_ref.proj_z.weight.data
        proj_o_weight = from_ref.proj_o.weight.data
    return (
        norm_m_weight,
        norm_m_bias,
        norm_z_weight,
        norm_z_bias,
        proj_m_weight,
        proj_g_weight,
        proj_z_weight,
        proj_o_weight,
    )


def load_pair_weighted_averaging_weights_ref_torch(module, weights_and_biases):
    (
        norm_m_weight,
        norm_m_bias,
        norm_z_weight,
        norm_z_bias,
        proj_m_weight,
        proj_g_weight,
        proj_z_weight,
        proj_o_weight,
    ) = weights_and_biases

    module.norm_m.weight.data.copy_(norm_m_weight.to("cuda"))
    module.norm_m.bias.data.copy_(norm_m_bias.to("cuda"))
    module.norm_z.weight.data.copy_(norm_z_weight.to("cuda"))
    module.norm_z.bias.data.copy_(norm_z_bias.to("cuda"))
    module.proj_m.weight.data.copy_(proj_m_weight.to("cuda"))
    module.proj_g.weight.data.copy_(proj_g_weight.to("cuda"))
    module.proj_z.weight.data.copy_(proj_z_weight.to("cuda"))
    module.proj_o.weight.data.copy_(proj_o_weight.to("cuda"))


def load_pair_weighted_averaging_weights_torch(module, weights_and_biases, dtype=torch.float32):
    (
        norm_m_weight,
        norm_m_bias,
        norm_z_weight,
        norm_z_bias,
        proj_m_weight,
        proj_g_weight,
        proj_z_weight,
        proj_o_weight,
    ) = weights_and_biases

    module.norm_m.weight.data.copy_(norm_m_weight.to("cuda"))
    module.norm_m.bias.data.copy_(norm_m_bias.to("cuda"))
    module.norm_z.weight.data.copy_(norm_z_weight.to("cuda"))
    module.norm_z.bias.data.copy_(norm_z_bias.to("cuda"))

    module.fused_proj_m_g.load_weights(
        [
            {"weight": proj_m_weight.to(dtype).to("cuda"), "bias": None},
            {"weight": proj_g_weight.to(dtype).to("cuda"), "bias": None},
        ]
    )
    module.proj_z.load_weights([{"weight": proj_z_weight.to(dtype).to("cuda"), "bias": None}])
    module.proj_o.load_weights([{"weight": proj_o_weight.to(dtype).to("cuda"), "bias": None}])


def create_outer_product_mean_weights(
    c_in: int = None,
    c_hidden: int = None,
    c_out: int = None,
    torch_dtype=None,
    bias_flags: dict[str, bool] = None,
    from_ref: RefOuterProductMean = None,
):
    if not from_ref:
        norm_weight = torch.empty(size=[c_in], dtype=torch_dtype)
        torch.nn.init.uniform_(norm_weight)
        norm_bias = torch.empty(size=[c_in], dtype=torch_dtype)
        torch.nn.init.zeros_(norm_bias)
        proj_a_weight = torch.empty(size=[c_hidden, c_in], dtype=torch_dtype)
        proj_a_bias = None
        if bias_flags["proj_a"]:
            proj_a_bias = torch.zeros(size=[c_hidden], dtype=torch_dtype)
        torch.nn.init.uniform_(proj_a_weight)
        proj_b_weight = torch.empty(size=[c_hidden, c_in], dtype=torch_dtype)
        proj_b_bias = None
        if bias_flags["proj_b"]:
            proj_b_bias = torch.zeros(size=[c_hidden], dtype=torch_dtype)
        torch.nn.init.uniform_(proj_b_weight)
        proj_o_weight = torch.empty(size=[c_out, c_hidden * c_hidden], dtype=torch_dtype)
        torch.nn.init.uniform_(proj_o_weight)
        proj_o_bias = None
        if bias_flags["proj_o"]:
            proj_o_bias = torch.zeros(size=[c_out], dtype=torch_dtype)
    else:
        bias_flags = from_ref.bias_flags
        norm_weight = from_ref.norm.weight.data
        norm_bias = from_ref.norm.bias.data
        proj_a_weight = from_ref.proj_a.weight.data
        proj_a_bias = None
        if bias_flags["proj_a"]:
            proj_a_bias = from_ref.proj_a.bias.data
        proj_b_weight = from_ref.proj_b.weight.data
        proj_b_bias = None
        if bias_flags["proj_b"]:
            proj_b_bias = from_ref.proj_b.bias.data
        proj_o_weight = from_ref.proj_o.weight.data
        proj_o_bias = None
        if bias_flags["proj_o"]:
            proj_o_bias = from_ref.proj_o.bias.data
    return norm_weight, norm_bias, proj_a_weight, proj_a_bias, proj_b_weight, proj_b_bias, proj_o_weight, proj_o_bias


def load_outer_product_mean_weights_ref_torch(module, weights_and_biases):
    norm_weight, norm_bias, proj_a_weight, proj_a_bias, proj_b_weight, proj_b_bias, proj_o_weight, proj_o_bias = (
        weights_and_biases
    )

    module.norm.weight.data.copy_(norm_weight.to("cuda"))
    module.norm.bias.data.copy_(norm_bias.to("cuda"))
    module.proj_a.weight.data.copy_(proj_a_weight.to("cuda"))
    if proj_a_bias is not None:
        module.proj_a.bias.data.copy_(proj_a_bias.to("cuda"))

    module.proj_b.weight.data.copy_(proj_b_weight.to("cuda"))
    if proj_b_bias is not None:
        module.proj_b.bias.data.copy_(proj_b_bias.to("cuda"))

    module.proj_o.weight.data.copy_(proj_o_weight.to("cuda"))
    if proj_o_bias is not None:
        module.proj_o.bias.data.copy_(proj_o_bias.to("cuda"))


def load_outer_product_mean_weights_torch(module, weights_and_biases, dtype=torch.float32):
    norm_weight, norm_bias, proj_a_weight, proj_a_bias, proj_b_weight, proj_b_bias, proj_o_weight, proj_o_bias = (
        weights_and_biases
    )

    module.norm.weight.data.copy_(norm_weight.to("cuda"))
    module.norm.bias.data.copy_(norm_bias.to("cuda"))

    module.fused_proj_a_b.load_weights(
        [
            {
                "weight": proj_a_weight.to(dtype).to("cuda"),
                "bias": proj_a_bias.to(dtype).to("cuda") if proj_a_bias is not None else None,
            },
            {
                "weight": proj_b_weight.to(dtype).to("cuda"),
                "bias": proj_b_bias.to(dtype).to("cuda") if proj_b_bias is not None else None,
            },
        ]
    )
    module.proj_o.load_weights(
        [
            {
                "weight": proj_o_weight.to(dtype).to("cuda"),
                "bias": proj_o_bias.to(dtype).to("cuda") if proj_o_bias is not None else None,
            }
        ]
    )


def load_outer_product_mean_weights_trt(module, weights_and_biases):
    norm_weight, norm_bias, proj_a_weight, proj_a_bias, proj_b_weight, proj_b_bias, proj_o_weight, proj_o_bias = (
        weights_and_biases
    )
    module.norm.weight.value = np.ascontiguousarray(norm_weight.cpu().numpy())
    module.norm.bias.value = np.ascontiguousarray(norm_bias.cpu().numpy())

    fused_proj_a_b_weight = torch.cat([proj_a_weight, proj_b_weight], dim=0)

    module.fused_proj_a_b.weight.value = np.ascontiguousarray(fused_proj_a_b_weight.cpu().numpy())
    if proj_a_bias is not None and proj_b_bias is not None:
        fused_proj_a_b_bias = torch.cat([proj_a_bias, proj_b_bias], dim=0)
        module.fused_proj_a_b.bias.value = np.ascontiguousarray(fused_proj_a_b_bias.cpu().numpy())
    module.proj_o.weight.value = np.ascontiguousarray(proj_o_weight.cpu().numpy())
    if proj_o_bias is not None:
        module.proj_o.bias.value = np.ascontiguousarray(proj_o_bias.cpu().numpy())


def create_msa_layer_weights(
    msa_s: int = None,
    token_z: int = None,
    pairwise_head_width: int = 32,
    pairwise_num_heads: int = 4,
    torch_dtype=None,
    from_ref: RefMSALayer = None,
):
    if not from_ref:
        msa_transition_weights = create_transition_weights(dim=msa_s, hidden=msa_s * 4, torch_dtype=torch_dtype)
        pair_weighted_averaging_weights = create_pair_weighted_averaging_weights(
            c_m=msa_s, c_z=token_z, c_h=32, num_heads=8, torch_dtype=torch_dtype
        )
        pairformer_layer_weights = create_pairformer_layer_weights(
            token_s=None,
            token_z=token_z,
            num_heads=pairwise_num_heads,
            pairwise_head_width=pairwise_head_width,
            include_s_path=False,
            torch_dtype=torch_dtype,
        )
        outer_product_mean_weights = create_outer_product_mean_weights(
            c_in=msa_s, c_hidden=32, c_out=token_z, torch_dtype=torch_dtype
        )
    else:
        msa_transition_weights = create_transition_weights(from_ref=from_ref.msa_transition)
        pair_weighted_averaging_weights = create_pair_weighted_averaging_weights(
            from_ref=from_ref.pair_weighted_averaging
        )
        pairformer_layer_weights = create_pairformer_layer_weights(
            from_ref=from_ref.pairformer_layer, include_s_path=False
        )
        outer_product_mean_weights = create_outer_product_mean_weights(from_ref=from_ref.outer_product_mean)
    return msa_transition_weights, pair_weighted_averaging_weights, pairformer_layer_weights, outer_product_mean_weights


def load_msa_layer_weights_ref_torch(module, weights_and_biases):
    msa_transition_weights, pair_weighted_averaging_weights, pairformer_layer_weights, outer_product_mean_weights = (
        weights_and_biases
    )

    load_transition_weights_ref_torch(module.msa_transition, msa_transition_weights)
    load_pair_weighted_averaging_weights_ref_torch(module.pair_weighted_averaging, pair_weighted_averaging_weights)
    load_pairformer_layer_weights_ref_torch(module.pairformer_layer, pairformer_layer_weights)
    load_outer_product_mean_weights_ref_torch(module.outer_product_mean, outer_product_mean_weights)


def load_msa_layer_weights_torch(module, weights_and_biases, dtype=torch.float32):
    msa_transition_weights, pair_weighted_averaging_weights, pairformer_layer_weights, outer_product_mean_weights = (
        weights_and_biases
    )

    load_transition_weights_torch(module.msa_transition, msa_transition_weights, dtype)
    load_pair_weighted_averaging_weights_torch(module.pair_weighted_averaging, pair_weighted_averaging_weights, dtype)
    load_pairformer_layer_weights_torch(module.pairformer_layer, pairformer_layer_weights, dtype)
    load_outer_product_mean_weights_torch(module.outer_product_mean, outer_product_mean_weights, dtype)


def create_msa_module_weights(
    msa_s: int = None,
    token_z: int = None,
    token_s: int = None,
    msa_blocks: int = None,
    num_tokens: int = None,
    pairwise_head_width: int = 32,
    pairwise_num_heads: int = 4,
    use_paired_feature: bool = True,
    torch_dtype: torch.dtype = torch.float32,
    from_ref: RefMSAModule = None,
):
    if not from_ref:
        s_proj_weight = torch.empty(size=[msa_s, token_s], dtype=torch_dtype)
        torch.nn.init.uniform_(s_proj_weight)
        msa_proj_weight = torch.empty(size=[msa_s, num_tokens + 2 + int(use_paired_feature)], dtype=torch_dtype)
        torch.nn.init.uniform_(msa_proj_weight)
        msa_layers_weights = []
        for i in range(msa_blocks):
            msa_layers_weights.append(
                create_msa_layer_weights(
                    msa_s=msa_s,
                    token_z=token_z,
                    pairwise_head_width=pairwise_head_width,
                    pairwise_num_heads=pairwise_num_heads,
                    torch_dtype=torch_dtype,
                )
            )
    else:
        s_proj_weight = from_ref.s_proj.weight.data
        msa_proj_weight = from_ref.msa_proj.weight.data
        msa_layers_weights = []
        for i in range(from_ref.msa_blocks):
            msa_layers_weights.append(create_msa_layer_weights(from_ref=from_ref.layers[i]))
    return s_proj_weight, msa_proj_weight, msa_layers_weights


def load_msa_module_weights_ref_torch(module, weights_and_biases):
    s_proj_weight, msa_proj_weight, msa_layers_weights = weights_and_biases
    module.s_proj.weight.data.copy_(s_proj_weight.to("cuda"))
    module.msa_proj.weight.data.copy_(msa_proj_weight.to("cuda"))
    for i in range(len(msa_layers_weights)):
        load_msa_layer_weights_ref_torch(module.layers[i], msa_layers_weights[i])


def load_msa_module_weights_torch(module, weights_and_biases, dtype=torch.float32):
    s_proj_weight, msa_proj_weight, msa_layers_weights = weights_and_biases
    module.s_proj.load_weights([{"weight": s_proj_weight.to(dtype).to("cuda"), "bias": None}])
    module.msa_proj.load_weights([{"weight": msa_proj_weight.to(dtype).to("cuda"), "bias": None}])
    for i in range(len(msa_layers_weights)):
        load_msa_layer_weights_torch(module.layers[i], msa_layers_weights[i], dtype)


def create_atom_embedding_weights(from_ref: RefAtomEmbedding = None):
    """TODO: Implement for the structure prediction path"""
    embed_atom_features_weight = from_ref.embed_atom_features.weight.data
    embed_atom_features_bias = from_ref.embed_atom_features.bias.data
    embed_atompair_ref_pos_weight = from_ref.embed_atompair_ref_pos.weight.data
    embed_atompair_ref_dist_weight = from_ref.embed_atompair_ref_dist.weight.data
    embed_atompair_mask_weight = from_ref.embed_atompair_mask.weight.data
    c_to_p_trans_k_weight = from_ref.c_to_p_trans_k[1].weight.data
    c_to_p_trans_q_weight = from_ref.c_to_p_trans_q[1].weight.data

    p_mlp_1_weight = from_ref.p_mlp[1].weight.data
    p_mlp_3_weight = from_ref.p_mlp[3].weight.data
    p_mlp_5_weight = from_ref.p_mlp[5].weight.data

    return (
        embed_atom_features_weight,
        embed_atom_features_bias,
        embed_atompair_ref_pos_weight,
        embed_atompair_ref_dist_weight,
        embed_atompair_mask_weight,
        c_to_p_trans_k_weight,
        c_to_p_trans_q_weight,
        p_mlp_1_weight,
        p_mlp_3_weight,
        p_mlp_5_weight,
    )


def load_atom_embedding_weights_ref_torch(module, weights_and_biases):
    """TODO: Implement for the structure prediction path"""
    (
        embed_atom_features_weight,
        embed_atom_features_bias,
        embed_atompair_ref_pos_weight,
        embed_atompair_ref_dist_weight,
        embed_atompair_mask_weight,
        c_to_p_trans_k_weight,
        c_to_p_trans_q_weight,
        p_mlp_1_weight,
        p_mlp_3_weight,
        p_mlp_5_weight,
    ) = weights_and_biases

    module.embed_atom_features.weight.data.copy_(embed_atom_features_weight.to("cuda"))
    module.embed_atom_features.bias.data.copy_(embed_atom_features_bias.to("cuda"))
    module.embed_atompair_ref_pos.weight.data.copy_(embed_atompair_ref_pos_weight.to("cuda"))
    module.embed_atompair_ref_dist.weight.data.copy_(embed_atompair_ref_dist_weight.to("cuda"))
    module.embed_atompair_mask.weight.data.copy_(embed_atompair_mask_weight.to("cuda"))
    module.c_to_p_trans_k[1].weight.data.copy_(c_to_p_trans_k_weight.to("cuda"))
    module.c_to_p_trans_q[1].weight.data.copy_(c_to_p_trans_q_weight.to("cuda"))
    module.p_mlp[1].weight.data.copy_(p_mlp_1_weight.to("cuda"))
    module.p_mlp[3].weight.data.copy_(p_mlp_3_weight.to("cuda"))
    module.p_mlp[5].weight.data.copy_(p_mlp_5_weight.to("cuda"))


def load_atom_embedding_weights_torch(module, weights_and_biases, dtype=torch.float32):
    """TODO: Implement for the structure prediction path"""
    (
        embed_atom_features_weight,
        embed_atom_features_bias,
        embed_atompair_ref_pos_weight,
        embed_atompair_ref_dist_weight,
        embed_atompair_mask_weight,
        c_to_p_trans_k_weight,
        c_to_p_trans_q_weight,
        p_mlp_1_weight,
        p_mlp_3_weight,
        p_mlp_5_weight,
    ) = weights_and_biases
    module.embed_atom_features.load_weights(
        [
            {
                "weight": embed_atom_features_weight.to(dtype).to("cuda"),
                "bias": embed_atom_features_bias.to(dtype).to("cuda") if embed_atom_features_bias is not None else None,
            }
        ]
    )
    module.embed_atompair_ref_pos.load_weights(
        [{"weight": embed_atompair_ref_pos_weight.to(dtype).to("cuda"), "bias": None}]
    )
    module.embed_atompair_ref_dist.load_weights(
        [{"weight": embed_atompair_ref_dist_weight.to(dtype).to("cuda"), "bias": None}]
    )
    module.embed_atompair_mask.load_weights([{"weight": embed_atompair_mask_weight.to(dtype).to("cuda"), "bias": None}])
    module.c_to_p_trans_k[1].load_weights([{"weight": c_to_p_trans_k_weight.to(dtype).to("cuda"), "bias": None}])
    module.c_to_p_trans_q[1].load_weights([{"weight": c_to_p_trans_q_weight.to(dtype).to("cuda"), "bias": None}])
    module.p_mlp[1].load_weights([{"weight": p_mlp_1_weight.to(dtype).to("cuda"), "bias": None}])
    module.p_mlp[3].load_weights([{"weight": p_mlp_3_weight.to(dtype).to("cuda"), "bias": None}])
    module.p_mlp[5].load_weights([{"weight": p_mlp_5_weight.to(dtype).to("cuda"), "bias": None}])


def create_atom_attention_encoder_weights(from_ref: RefAtomAttentionEncoder = None):
    num_heads = from_ref.atom_encoder.diffusion_transformer.layers[0].pair_bias_attn.num_heads
    dim = from_ref.atom_encoder.diffusion_transformer.layers[0].adaln.dim
    dim_single_cond = from_ref.atom_encoder.diffusion_transformer.dim_single_cond
    dim_pairwise = from_ref.atom_encoder.diffusion_transformer.dim_pairwise
    compute_pair_bias = from_ref.atom_encoder.diffusion_transformer.layers[0].pair_bias_attn.compute_pair_bias
    atom_transformer_weights = []
    for layer in range(len(from_ref.atom_encoder.diffusion_transformer.layers)):
        diffusion_layers_weight_dict = create_diffusion_transformer_layer_weights(
            num_heads=num_heads,
            dim=dim,
            dim_single_cond=dim_single_cond,
            dim_pairwise=dim_pairwise,
            torch_dtype=torch.float32,
            compute_pair_bias=compute_pair_bias,
            from_ref=from_ref.atom_encoder.diffusion_transformer.layers[layer],
        )
        atom_transformer_weights.append(diffusion_layers_weight_dict)
    atom_to_token_trans_weight = from_ref.atom_to_token_trans[0].weight.data
    r_to_q_trans_weight = from_ref.r_to_q_trans.weight.data
    return atom_transformer_weights, atom_to_token_trans_weight, r_to_q_trans_weight


def load_atom_attention_encoder_weights_torch(module, weights_and_biases, dtype=torch.float32):
    atom_transformer_weights, atom_to_token_trans_weight, r_to_q_trans_weight = weights_and_biases
    for layer in range(len(atom_transformer_weights)):
        load_diffusion_transformer_layer_weights_torch(
            module.atom_encoder.diffusion_transformer.layers[layer], atom_transformer_weights[layer]
        )

    module.atom_to_token_trans[0].load_weights(
        [{"weight": atom_to_token_trans_weight.to(torch.float32).to("cuda"), "bias": None}]
    )
    module.r_to_q_trans.load_weights([{"weight": r_to_q_trans_weight.to(dtype).to("cuda"), "bias": None}])


def create_atom_attention_decoder_weights(from_ref: RefAtomAttentionDecoder = None):
    num_heads = from_ref.atom_decoder.diffusion_transformer.layers[0].pair_bias_attn.num_heads
    dim = from_ref.atom_decoder.diffusion_transformer.layers[0].adaln.dim
    dim_single_cond = from_ref.atom_decoder.diffusion_transformer.dim_single_cond
    dim_pairwise = from_ref.atom_decoder.diffusion_transformer.dim_pairwise
    compute_pair_bias = from_ref.atom_decoder.diffusion_transformer.layers[0].pair_bias_attn.compute_pair_bias
    atom_transformer_weights = []

    for layer in range(len(from_ref.atom_decoder.diffusion_transformer.layers)):
        diffusion_layers_weight_dict = create_diffusion_transformer_layer_weights(
            num_heads=num_heads,
            dim=dim,
            dim_single_cond=dim_single_cond,
            dim_pairwise=dim_pairwise,
            torch_dtype=torch.float32,
            compute_pair_bias=compute_pair_bias,
            from_ref=from_ref.atom_decoder.diffusion_transformer.layers[layer],
        )
        atom_transformer_weights.append(diffusion_layers_weight_dict)

    a_to_q_trans_weight = from_ref.a_to_q_trans.weight.data
    atom_feat_to_atom_pos_update_norm_weight = from_ref.atom_feat_to_atom_pos_update[0].weight.data
    atom_feat_to_atom_pos_update_norm_bias = from_ref.atom_feat_to_atom_pos_update[0].bias.data
    atom_feat_to_atom_pos_update_linear_weight = from_ref.atom_feat_to_atom_pos_update[1].weight.data

    return (
        atom_transformer_weights,
        a_to_q_trans_weight,
        atom_feat_to_atom_pos_update_norm_weight,
        atom_feat_to_atom_pos_update_norm_bias,
        atom_feat_to_atom_pos_update_linear_weight,
    )


def load_atom_attention_decoder_weights_torch(module, weights_and_biases, dtype=torch.float32):
    (
        atom_transformer_weights,
        a_to_q_trans_weight,
        atom_feat_to_atom_pos_update_norm_weight,
        atom_feat_to_atom_pos_update_norm_bias,
        atom_feat_to_atom_pos_update_linear_weight,
    ) = weights_and_biases

    for layer in range(len(atom_transformer_weights)):
        load_diffusion_transformer_layer_weights_torch(
            module.atom_decoder.diffusion_transformer.layers[layer], atom_transformer_weights[layer]
        )
    module.a_to_q_trans.load_weights([{"weight": a_to_q_trans_weight.to(dtype).to("cuda"), "bias": None}])
    module.atom_feat_to_atom_pos_update[0].weight.data.copy_(atom_feat_to_atom_pos_update_norm_weight.to("cuda"))
    module.atom_feat_to_atom_pos_update[0].bias.data.copy_(atom_feat_to_atom_pos_update_norm_bias.to("cuda"))

    module.atom_feat_to_atom_pos_update[1].load_weights(
        [{"weight": atom_feat_to_atom_pos_update_linear_weight.to(dtype).to("cuda"), "bias": None}]
    )


def create_single_conditioning_weights(from_ref: RefSingleConditioning = None):
    transition_weights = []
    for layer in range(len(from_ref.transitions)):
        transition_weights.append(create_transition_weights(from_ref=from_ref.transitions[layer]))

    norm_single_weight = from_ref.norm_single.weight.data
    norm_single_bias = from_ref.norm_single.bias.data
    single_embed_weight = from_ref.single_embed.weight.data
    single_embed_bias = from_ref.single_embed.bias.data
    norm_fourier_weight = from_ref.norm_fourier.weight.data
    norm_fourier_bias = from_ref.norm_fourier.bias.data
    fourier_embed_weight = from_ref.fourier_embed.proj.weight.data
    fourier_embed_bias = from_ref.fourier_embed.proj.bias.data
    fourier_to_single_weight = from_ref.fourier_to_single.weight.data

    return (
        transition_weights,
        single_embed_weight,
        single_embed_bias,
        norm_single_weight,
        norm_single_bias,
        norm_fourier_weight,
        norm_fourier_bias,
        fourier_embed_weight,
        fourier_embed_bias,
        fourier_to_single_weight,
    )


def load_single_conditioning_weights_torch(module, weights_and_biases, dtype=torch.float32):
    (
        transition_weights,
        single_embed_weight,
        single_embed_bias,
        norm_single_weight,
        norm_single_bias,
        norm_fourier_weight,
        norm_fourier_bias,
        fourier_embed_weight,
        fourier_embed_bias,
        fourier_to_single_weight,
    ) = weights_and_biases

    module.norm_single.weight.data.copy_(norm_single_weight.to("cuda"))
    module.norm_single.bias.data.copy_(norm_single_bias.to("cuda"))
    module.single_embed.load_weights(
        [
            {
                "weight": single_embed_weight.to(dtype).to("cuda"),
                "bias": single_embed_bias.to(dtype).to("cuda") if single_embed_bias is not None else None,
            }
        ]
    )

    module.norm_fourier.weight.data.copy_(norm_fourier_weight.to("cuda"))
    module.norm_fourier.bias.data.copy_(norm_fourier_bias.to("cuda"))

    module.fourier_embed.proj.load_weights(
        [
            {
                "weight": fourier_embed_weight.to(dtype).to("cuda"),
                "bias": fourier_embed_bias.to(dtype).to("cuda") if fourier_embed_bias is not None else None,
            }
        ]
    )
    module.fourier_to_single.load_weights([{"weight": fourier_to_single_weight.to(dtype).to("cuda"), "bias": None}])
    for layer in range(len(transition_weights)):
        load_transition_weights_torch(module.transitions[layer], transition_weights[layer], dtype)


def create_diffusion_module_weights(from_ref: RefDiffusionModule = None):

    token_transformer_weights = []
    for layer in range(len(from_ref.token_transformer.layers)):
        token_transformer_weights.append(
            create_diffusion_transformer_layer_weights(from_ref=from_ref.token_transformer.layers[layer])
        )

    single_conditioning_weights = create_single_conditioning_weights(from_ref=from_ref.single_conditioner)
    atom_attention_encoder_weights = create_atom_attention_encoder_weights(from_ref=from_ref.atom_attention_encoder)
    atom_attention_decoder_weights = create_atom_attention_decoder_weights(from_ref=from_ref.atom_attention_decoder)

    s_to_a_linear_weight = from_ref.s_to_a_linear[0].weight.data
    s_to_a_linear_bias = from_ref.s_to_a_linear[0].bias.data
    s_to_a_linear_linear_weight = from_ref.s_to_a_linear[1].weight.data

    a_norm_weight = from_ref.a_norm.weight.data
    a_norm_bias = from_ref.a_norm.bias.data

    return (
        single_conditioning_weights,
        atom_attention_encoder_weights,
        atom_attention_decoder_weights,
        token_transformer_weights,
        s_to_a_linear_weight,
        s_to_a_linear_bias,
        s_to_a_linear_linear_weight,
        a_norm_weight,
        a_norm_bias,
    )


def load_diffusion_module_weights_torch(module, weights_and_biases, dtype=torch.float32):
    (
        single_conditioning_weights,
        atom_attention_encoder_weights,
        atom_attention_decoder_weights,
        token_transformer_weights,
        s_to_a_linear_weight,
        s_to_a_linear_bias,
        s_to_a_linear_linear_weight,
        a_norm_weight,
        a_norm_bias,
    ) = weights_and_biases

    load_single_conditioning_weights_torch(module.single_conditioner, single_conditioning_weights)
    for layer in range(len(token_transformer_weights)):
        load_diffusion_transformer_layer_weights_torch(
            module.token_transformer.layers[layer], token_transformer_weights[layer]
        )

    load_atom_attention_encoder_weights_torch(module.atom_attention_encoder, atom_attention_encoder_weights)
    load_atom_attention_decoder_weights_torch(module.atom_attention_decoder, atom_attention_decoder_weights)

    module.s_to_a_linear[0].weight.data.copy_(s_to_a_linear_weight.to("cuda"))
    module.s_to_a_linear[0].bias.data.copy_(s_to_a_linear_bias.to("cuda"))

    module.s_to_a_linear[1].load_weights([{"weight": s_to_a_linear_linear_weight.to(dtype).to("cuda"), "bias": None}])

    module.a_norm.weight.data.copy_(a_norm_weight.to("cuda"))
    module.a_norm.bias.data.copy_(a_norm_bias.to("cuda"))


def create_template_module_weights(from_ref: RefTemplateV2Module = None):
    """Collect the weights of a :class:`RefTemplateV2Module` so they can be
    loaded into the TRT-BNM :class:`TemplateV2Module` for tests.
    """
    assert from_ref is not None, "from_ref is required"
    z_norm_weight = from_ref.z_norm.weight.data
    z_norm_bias = from_ref.z_norm.bias.data
    v_norm_weight = from_ref.v_norm.weight.data
    v_norm_bias = from_ref.v_norm.bias.data
    z_proj_weight = from_ref.z_proj.weight.data
    a_proj_weight = from_ref.a_proj.weight.data
    u_proj_weight = from_ref.u_proj.weight.data
    pairformer_layers_weights = [
        create_pairformer_layer_weights(from_ref=from_ref.pairformer.layers[i], include_s_path=False)
        for i in range(from_ref.template_blocks)
    ]
    return (
        z_norm_weight,
        z_norm_bias,
        v_norm_weight,
        v_norm_bias,
        z_proj_weight,
        a_proj_weight,
        u_proj_weight,
        pairformer_layers_weights,
    )


def load_template_module_weights_torch(module, weights_and_biases, dtype=torch.float32):
    """Load template-v2 module weights produced by
    :func:`create_template_module_weights` into a TRT-BNM
    :class:`TemplateV2Module` instance.
    """
    (
        z_norm_weight,
        z_norm_bias,
        v_norm_weight,
        v_norm_bias,
        z_proj_weight,
        a_proj_weight,
        u_proj_weight,
        pairformer_layers_weights,
    ) = weights_and_biases

    module.z_norm.weight.data.copy_(z_norm_weight.to("cuda"))
    module.z_norm.bias.data.copy_(z_norm_bias.to("cuda"))
    module.v_norm.weight.data.copy_(v_norm_weight.to("cuda"))
    module.v_norm.bias.data.copy_(v_norm_bias.to("cuda"))
    module.z_proj.load_weights([{"weight": z_proj_weight.to(dtype).to("cuda"), "bias": None}])
    module.a_proj.load_weights([{"weight": a_proj_weight.to(dtype).to("cuda"), "bias": None}])
    module.u_proj.load_weights([{"weight": u_proj_weight.to(dtype).to("cuda"), "bias": None}])
    for i, layer_weights in enumerate(pairformer_layers_weights):
        load_pairformer_layer_weights_torch(module.pairformer.layers[i], layer_weights, dtype)
