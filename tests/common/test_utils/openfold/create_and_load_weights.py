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

# isort: off
from tensorrt_llm.models.convert_utils import split
from test_utils.boltz.create_and_load_weights import (
    create_outer_product_mean_weights, create_self_pairwise_attention_weights,
    create_triangle_attention_node_weights, create_triangle_attention_weights,
    create_triangle_multiplication_node_weights,
    load_outer_product_mean_weights_ref_torch,
    load_outer_product_mean_weights_torch, load_outer_product_mean_weights_trt,
    load_self_pairwise_attention_weights_ref_torch,
    load_triangle_attention_node_weights_ref_torch,
    load_triangle_attention_node_weights_torch,
    load_triangle_attention_node_weights_trt,
    load_triangle_attention_weights_ref_torch,
    load_triangle_attention_weights_torch, load_triangle_attention_weights_trt,
    load_triangle_multiplication_node_weights_ref_torch,
    load_triangle_multiplication_node_weights_torch,
    load_triangle_multiplication_node_weights_trt)
from test_utils.openfold.ref_layers import (
    RefEvoformerBlock, RefExtraMSABlock, RefInputEmbedder, RefMSAAttention,
    RefMSAColumnGlobalAttention, RefMSATransition, RefPairTransition,
    RefRecyclingEmbedder, RefTemplatePairStackBlock,
    RefTemplatePointwiseAttention)
# isort: on
from tensorrt_bionemo.mapping import Mapping


def create_msa_attention_weights(c_in=None,
                                 c_hidden=None,
                                 no_heads=None,
                                 pair_bias=False,
                                 c_z=None,
                                 using_tri_attn: bool = True,
                                 from_ref: RefMSAAttention = None):

    if not from_ref:
        layer_norm_m_weight = torch.randn(c_in, dtype=torch.float32)
        layer_norm_m_bias = torch.randn(c_in, dtype=torch.float32)
        if pair_bias:
            layer_norm_z_weight = torch.randn(c_z, dtype=torch.float32)
            layer_norm_z_bias = torch.randn(c_z, dtype=torch.float32)
            linear_z_weight = torch.randn(no_heads, c_z, dtype=torch.float32)
        else:
            layer_norm_z_weight = None
            layer_norm_z_bias = None
            linear_z_weight = None
        if not using_tri_attn:
            mha_weights = create_self_pairwise_attention_weights(
                c_s=c_in,
                c_z=c_z,
                num_attention_heads=no_heads,
                compute_pair_bias=False,
                bias_flags={
                    "q": False,
                    "k": False,
                    "v": False,
                    "g": True,
                    "o": True
                },
                initial_norm=False)
        else:
            mha_weights = create_triangle_attention_weights(c_q=c_in,
                                                            c_k=c_in,
                                                            c_v=c_in,
                                                            bias_flags={
                                                                "q": False,
                                                                "k": False,
                                                                "v": False,
                                                                "g": True,
                                                                "o": True
                                                            })
    else:
        layer_norm_m_weight = from_ref.layer_norm_m.weight.data
        layer_norm_m_bias = from_ref.layer_norm_m.bias.data
        if from_ref.pair_bias:
            layer_norm_z_weight = from_ref.layer_norm_z.weight.data
            layer_norm_z_bias = from_ref.layer_norm_z.bias.data
            linear_z_weight = from_ref.linear_z.weight.data
        else:
            layer_norm_z_weight = None
            layer_norm_z_bias = None
            linear_z_weight = None
        if not from_ref.using_tri_attn:
            mha_weights = create_self_pairwise_attention_weights(
                from_ref=from_ref.mha)
        else:
            mha_weights = create_triangle_attention_weights(
                from_ref=from_ref.mha)

    return layer_norm_m_weight, layer_norm_m_bias, layer_norm_z_weight, layer_norm_z_bias, linear_z_weight, mha_weights


def load_msa_attention_weights_ref_torch(module, weights_and_biases):
    layer_norm_m_weight, layer_norm_m_bias, layer_norm_z_weight, layer_norm_z_bias, linear_z_weight, mha_weights = weights_and_biases
    if not module.using_tri_attn:
        load_self_pairwise_attention_weights_ref_torch(module.mha, mha_weights)
    else:
        load_triangle_attention_weights_ref_torch(module.mha, mha_weights)
    module.layer_norm_m.weight.data.copy_(layer_norm_m_weight.to("cuda"))
    module.layer_norm_m.bias.data.copy_(layer_norm_m_bias.to("cuda"))
    if layer_norm_z_weight is not None:
        module.layer_norm_z.weight.data.copy_(layer_norm_z_weight.to("cuda"))
        module.layer_norm_z.bias.data.copy_(layer_norm_z_bias.to("cuda"))
    if linear_z_weight is not None:
        module.linear_z.weight.data.copy_(linear_z_weight.to("cuda"))


def load_msa_attention_weights_torch(module, weights_and_biases):
    layer_norm_m_weight, layer_norm_m_bias, layer_norm_z_weight, layer_norm_z_bias, linear_z_weight, mha_weights = weights_and_biases
    load_triangle_attention_weights_torch(module.mha, mha_weights)
    module.layer_norm_m.weight.data.copy_(layer_norm_m_weight.to("cuda"))
    module.layer_norm_m.bias.data.copy_(layer_norm_m_bias.to("cuda"))
    if layer_norm_z_weight is not None:
        module.proj_z_norm.weight.data.copy_(layer_norm_z_weight.to("cuda"))
        module.proj_z_norm.bias.data.copy_(layer_norm_z_bias.to("cuda"))
    if linear_z_weight is not None:
        module.proj_z.load_weights([{
            "weight": linear_z_weight.to("cuda"),
            "bias": None
        }])


def load_msa_attention_weights_trt(module,
                                   weights_and_biases,
                                   using_tri_attn: bool = True,
                                   mapping: Mapping = None):
    mapping = mapping or Mapping()
    layer_norm_m_weight, layer_norm_m_bias, layer_norm_z_weight, layer_norm_z_bias, linear_z_weight, mha_weights = weights_and_biases

    assert using_tri_attn, "Using tri attn is not supported yet for trt"
    load_triangle_attention_weights_trt(module.mha, mha_weights,
                                        mapping.tp_size, mapping.tp_rank)
    module.layer_norm_m.weight.value = np.ascontiguousarray(
        layer_norm_m_weight.cpu().numpy())
    module.layer_norm_m.bias.value = np.ascontiguousarray(
        layer_norm_m_bias.cpu().numpy())
    if layer_norm_z_weight is not None:
        module.proj_z_norm.weight.value = np.ascontiguousarray(
            layer_norm_z_weight.cpu().numpy())
        module.proj_z_norm.bias.value = np.ascontiguousarray(
            layer_norm_z_bias.cpu().numpy())
    if linear_z_weight is not None:
        linear_z_weight = split(linear_z_weight,
                                mapping.tp_size,
                                mapping.tp_rank,
                                dim=0)
        module.proj_z.weight.value = np.ascontiguousarray(
            linear_z_weight.cpu().numpy())


def create_msa_transition_weights(from_ref: RefMSATransition = None):
    layer_norm_weight = from_ref.layer_norm.weight.data
    layer_norm_bias = from_ref.layer_norm.bias.data
    linear_1_weight = from_ref.linear_1.weight.data
    linear_1_bias = from_ref.linear_1.bias.data
    linear_2_weight = from_ref.linear_2.weight.data
    linear_2_bias = from_ref.linear_2.bias.data
    return layer_norm_weight, layer_norm_bias, linear_1_weight, linear_1_bias, linear_2_weight, linear_2_bias


def load_msa_transition_weights_ref_torch(module, weights_and_biases):
    layer_norm_weight, layer_norm_bias, linear_1_weight, linear_1_bias, linear_2_weight, linear_2_bias = weights_and_biases
    module.layer_norm.weight.data.copy_(layer_norm_weight.to("cuda"))
    module.layer_norm.bias.data.copy_(layer_norm_bias.to("cuda"))
    module.linear_1.weight.data.copy_(linear_1_weight.to("cuda"))
    module.linear_1.bias.data.copy_(linear_1_bias.to("cuda"))
    module.linear_2.weight.data.copy_(linear_2_weight.to("cuda"))
    module.linear_2.bias.data.copy_(linear_2_bias.to("cuda"))


def load_msa_transition_weights_torch(module, weights_and_biases):
    layer_norm_weight, layer_norm_bias, linear_1_weight, linear_1_bias, linear_2_weight, linear_2_bias = weights_and_biases
    module.layer_norm.weight.data.copy_(layer_norm_weight.to("cuda"))
    module.layer_norm.bias.data.copy_(layer_norm_bias.to("cuda"))
    module.linear_1.load_weights([{
        "weight":
        linear_1_weight.to("cuda"),
        "bias":
        linear_1_bias.to("cuda") if linear_1_bias is not None else None
    }])
    module.linear_2.load_weights([{
        "weight":
        linear_2_weight.to("cuda"),
        "bias":
        linear_2_bias.to("cuda") if linear_2_bias is not None else None
    }])


def load_msa_transition_weights_trt(module,
                                    weights_and_biases,
                                    mapping: Mapping = None):
    mapping = mapping or Mapping()
    layer_norm_weight, layer_norm_bias, linear_1_weight, linear_1_bias, linear_2_weight, linear_2_bias = weights_and_biases
    module.layer_norm.weight.value = np.ascontiguousarray(
        layer_norm_weight.cpu().numpy())
    module.layer_norm.bias.value = np.ascontiguousarray(
        layer_norm_bias.cpu().numpy())
    if mapping.tp_size > 1:
        linear_1_weight = split(linear_1_weight, mapping.tp_size,
                                mapping.tp_rank, 0)
        linear_1_bias = split(linear_1_bias, mapping.tp_size, mapping.tp_rank,
                              0)
        linear_2_weight = split(linear_2_weight, mapping.tp_size,
                                mapping.tp_rank, 1)

    module.linear_1.weight.value = np.ascontiguousarray(
        linear_1_weight.cpu().numpy())
    module.linear_1.bias.value = np.ascontiguousarray(
        linear_1_bias.cpu().numpy())
    module.linear_2.weight.value = np.ascontiguousarray(
        linear_2_weight.cpu().numpy())
    module.linear_2.bias.value = np.ascontiguousarray(
        linear_2_bias.cpu().numpy())


def create_pair_transition_weights(from_ref: RefPairTransition = None):
    layer_norm_weight = from_ref.layer_norm.weight.data
    layer_norm_bias = from_ref.layer_norm.bias.data
    linear_1_weight = from_ref.linear_1.weight.data
    linear_1_bias = from_ref.linear_1.bias.data
    linear_2_weight = from_ref.linear_2.weight.data
    linear_2_bias = from_ref.linear_2.bias.data
    return layer_norm_weight, layer_norm_bias, linear_1_weight, linear_1_bias, linear_2_weight, linear_2_bias


def load_pair_transition_weights_ref_torch(module, weights_and_biases):
    layer_norm_weight, layer_norm_bias, linear_1_weight, linear_1_bias, linear_2_weight, linear_2_bias = weights_and_biases
    module.layer_norm.weight.data.copy_(layer_norm_weight.to("cuda"))
    module.layer_norm.bias.data.copy_(layer_norm_bias.to("cuda"))
    module.linear_1.weight.data.copy_(linear_1_weight.to("cuda"))
    module.linear_1.bias.data.copy_(linear_1_bias.to("cuda"))
    module.linear_2.weight.data.copy_(linear_2_weight.to("cuda"))
    module.linear_2.bias.data.copy_(linear_2_bias.to("cuda"))


def load_pair_transition_weights_torch(module, weights_and_biases):
    layer_norm_weight, layer_norm_bias, linear_1_weight, linear_1_bias, linear_2_weight, linear_2_bias = weights_and_biases
    module.layer_norm.weight.data.copy_(layer_norm_weight.to("cuda"))
    module.layer_norm.bias.data.copy_(layer_norm_bias.to("cuda"))
    module.linear_1.load_weights([{
        "weight":
        linear_1_weight.to("cuda"),
        "bias":
        linear_1_bias.to("cuda") if linear_1_bias is not None else None
    }])
    module.linear_2.load_weights([{
        "weight":
        linear_2_weight.to("cuda"),
        "bias":
        linear_2_bias.to("cuda") if linear_2_bias is not None else None
    }])


def load_pair_transition_weights_trt(module,
                                     weights_and_biases,
                                     mapping: Mapping = None):
    mapping = mapping or Mapping()
    layer_norm_weight, layer_norm_bias, linear_1_weight, linear_1_bias, linear_2_weight, linear_2_bias = weights_and_biases
    module.layer_norm.weight.value = np.ascontiguousarray(
        layer_norm_weight.cpu().numpy())
    module.layer_norm.bias.value = np.ascontiguousarray(
        layer_norm_bias.cpu().numpy())
    if mapping.tp_size > 1:
        linear_1_weight = split(linear_1_weight, mapping.tp_size,
                                mapping.tp_rank, 0)
        linear_1_bias = split(linear_1_bias, mapping.tp_size, mapping.tp_rank,
                              0)
        linear_2_weight = split(linear_2_weight, mapping.tp_size,
                                mapping.tp_rank, 1)

    module.linear_1.weight.value = np.ascontiguousarray(
        linear_1_weight.cpu().numpy())
    module.linear_1.bias.value = np.ascontiguousarray(
        linear_1_bias.cpu().numpy())
    module.linear_2.weight.value = np.ascontiguousarray(
        linear_2_weight.cpu().numpy())
    module.linear_2.bias.value = np.ascontiguousarray(
        linear_2_bias.cpu().numpy())


def create_evoformer_block_weights(from_ref: RefEvoformerBlock = None):
    msa_att_row_weights = create_msa_attention_weights(
        from_ref=from_ref.msa_att_row)
    msa_att_col_weights = create_msa_attention_weights(
        from_ref=from_ref.msa_att_col)
    msa_transition_weights = create_msa_transition_weights(
        from_ref=from_ref.msa_transition)
    outer_product_mean_weights = create_outer_product_mean_weights(
        from_ref=from_ref.outer_product_mean)
    tri_mul_out_weights = create_triangle_multiplication_node_weights(
        from_ref=from_ref.tri_mul_out)
    tri_mul_in_weights = create_triangle_multiplication_node_weights(
        from_ref=from_ref.tri_mul_in)
    tri_attn_start_weights = create_triangle_attention_node_weights(
        from_ref=from_ref.tri_attn_start)
    tri_attn_end_weights = create_triangle_attention_node_weights(
        from_ref=from_ref.tri_attn_end)
    pair_transition_weights = create_pair_transition_weights(
        from_ref=from_ref.pair_transition)

    return msa_att_row_weights, msa_att_col_weights, msa_transition_weights, outer_product_mean_weights, \
            tri_mul_out_weights, tri_mul_in_weights, tri_attn_start_weights, tri_attn_end_weights, pair_transition_weights


def load_evoformer_block_weights_ref_torch(module, weights_and_biases):
    msa_att_row_weights, msa_att_col_weights, msa_transition_weights, outer_product_mean_weights, \
            tri_mul_out_weights, tri_mul_in_weights, tri_attn_start_weights, tri_attn_end_weights, pair_transition_weights = weights_and_biases
    load_msa_attention_weights_ref_torch(module.msa_att_row,
                                         msa_att_row_weights)
    load_msa_attention_weights_ref_torch(module.msa_att_col,
                                         msa_att_col_weights)
    load_msa_transition_weights_ref_torch(module.msa_transition,
                                          msa_transition_weights)
    load_outer_product_mean_weights_ref_torch(module.outer_product_mean,
                                              outer_product_mean_weights)
    load_triangle_multiplication_node_weights_ref_torch(module.tri_mul_out,
                                                        tri_mul_out_weights)
    load_triangle_multiplication_node_weights_ref_torch(module.tri_mul_in,
                                                        tri_mul_in_weights)
    load_triangle_attention_node_weights_ref_torch(module.tri_attn_start,
                                                   tri_attn_start_weights)
    load_triangle_attention_node_weights_ref_torch(module.tri_attn_end,
                                                   tri_attn_end_weights)
    load_pair_transition_weights_ref_torch(module.pair_transition,
                                           pair_transition_weights)


def load_evoformer_block_weights_torch(module, weights_and_biases):
    msa_att_row_weights, msa_att_col_weights, msa_transition_weights, outer_product_mean_weights, \
            tri_mul_out_weights, tri_mul_in_weights, tri_attn_start_weights, tri_attn_end_weights, pair_transition_weights = weights_and_biases
    load_msa_attention_weights_torch(module.msa_att_row, msa_att_row_weights)
    load_msa_attention_weights_torch(module.msa_att_col, msa_att_col_weights)
    load_msa_transition_weights_torch(module.msa_transition,
                                      msa_transition_weights)
    load_outer_product_mean_weights_torch(module.outer_product_mean,
                                          outer_product_mean_weights)
    load_triangle_multiplication_node_weights_torch(module.tri_mul_out,
                                                    tri_mul_out_weights)
    load_triangle_multiplication_node_weights_torch(module.tri_mul_in,
                                                    tri_mul_in_weights)
    load_triangle_attention_node_weights_torch(module.tri_attn_start,
                                               tri_attn_start_weights)
    load_triangle_attention_node_weights_torch(module.tri_attn_end,
                                               tri_attn_end_weights)
    load_pair_transition_weights_torch(module.pair_transition,
                                       pair_transition_weights)


def load_evoformer_block_weights_trt(module,
                                     weights_and_biases,
                                     mapping: Mapping = None):
    mapping = mapping or Mapping()
    msa_att_row_weights, msa_att_col_weights, msa_transition_weights, outer_product_mean_weights, \
            tri_mul_out_weights, tri_mul_in_weights, tri_attn_start_weights, tri_attn_end_weights, pair_transition_weights = weights_and_biases

    load_msa_attention_weights_trt(module.msa_att_row, msa_att_row_weights,
                                   mapping)
    load_msa_attention_weights_trt(module.msa_att_col, msa_att_col_weights,
                                   mapping)
    load_msa_transition_weights_trt(module.msa_transition,
                                    msa_transition_weights, mapping)
    load_outer_product_mean_weights_trt(module.outer_product_mean,
                                        outer_product_mean_weights, mapping)
    load_triangle_multiplication_node_weights_trt(module.tri_mul_out,
                                                  tri_mul_out_weights, mapping)
    load_triangle_multiplication_node_weights_trt(module.tri_mul_in,
                                                  tri_mul_in_weights, mapping)
    load_triangle_attention_node_weights_trt(module.tri_attn_start,
                                             tri_attn_start_weights, mapping)
    load_triangle_attention_node_weights_trt(module.tri_attn_end,
                                             tri_attn_end_weights, mapping)
    load_pair_transition_weights_trt(module.pair_transition,
                                     pair_transition_weights, mapping)


def create_msa_global_attention_weights(
        from_ref: RefMSAColumnGlobalAttention = None):
    layer_norm_m_weight = from_ref.layer_norm_m.weight.data
    layer_norm_m_bias = from_ref.layer_norm_m.bias.data
    linear_q_weight = from_ref.global_attention.linear_q.weight.data
    linear_k_weight = from_ref.global_attention.linear_k.weight.data
    linear_v_weight = from_ref.global_attention.linear_v.weight.data
    linear_g_weight = from_ref.global_attention.linear_g.weight.data
    linear_g_bias = from_ref.global_attention.linear_g.bias.data
    linear_o_weight = from_ref.global_attention.linear_o.weight.data
    linear_o_bias = from_ref.global_attention.linear_o.bias.data
    return layer_norm_m_weight, layer_norm_m_bias, \
        linear_q_weight, linear_k_weight, linear_v_weight, \
        linear_g_weight, linear_g_bias, linear_o_weight, linear_o_bias


def load_msa_global_attention_weights_ref_torch(module, weights_and_biases):
    layer_norm_m_weight, layer_norm_m_bias, \
        linear_q_weight, linear_k_weight, linear_v_weight, \
        linear_g_weight, linear_g_bias, linear_o_weight, linear_o_bias = weights_and_biases
    module.layer_norm_m.weight.data.copy_(layer_norm_m_weight.to("cuda"))
    module.layer_norm_m.bias.data.copy_(layer_norm_m_bias.to("cuda"))
    module.global_attention.linear_q.weight.data.copy_(
        linear_q_weight.to("cuda"))
    module.global_attention.linear_k.weight.data.copy_(
        linear_k_weight.to("cuda"))
    module.global_attention.linear_v.weight.data.copy_(
        linear_v_weight.to("cuda"))
    module.global_attention.linear_g.weight.data.copy_(
        linear_g_weight.to("cuda"))
    module.global_attention.linear_g.bias.data.copy_(linear_g_bias.to("cuda"))
    module.global_attention.linear_o.weight.data.copy_(
        linear_o_weight.to("cuda"))
    module.global_attention.linear_o.bias.data.copy_(linear_o_bias.to("cuda"))


def load_msa_global_attention_weights_torch(module, weights_and_biases):
    layer_norm_m_weight, layer_norm_m_bias, \
        linear_q_weight, linear_k_weight, linear_v_weight, \
        linear_g_weight, linear_g_bias, linear_o_weight, linear_o_bias = weights_and_biases
    module.layer_norm_m.weight.data.copy_(layer_norm_m_weight.to("cuda"))
    module.layer_norm_m.bias.data.copy_(layer_norm_m_bias.to("cuda"))
    module.global_attention.proj_q.load_weights([{
        "weight":
        linear_q_weight.to("cuda"),
        "bias":
        None
    }])
    module.global_attention.fused_proj_kv.load_weights([{
        "weight":
        linear_k_weight.to("cuda"),
        "bias":
        None
    }, {
        "weight":
        linear_v_weight.to("cuda"),
        "bias":
        None
    }])
    module.global_attention.proj_g.load_weights([{
        "weight":
        linear_g_weight.to("cuda"),
        "bias":
        linear_g_bias.to("cuda")
    }])
    module.global_attention.proj_o.load_weights([{
        "weight":
        linear_o_weight.to("cuda"),
        "bias":
        linear_o_bias.to("cuda")
    }])


def create_extra_msa_block_weights(from_ref: RefExtraMSABlock = None):
    msa_att_row_weights = create_msa_attention_weights(
        from_ref=from_ref.msa_att_row)
    msa_att_col_weights = create_msa_global_attention_weights(
        from_ref=from_ref.msa_att_col)
    msa_transition_weights = create_msa_transition_weights(
        from_ref=from_ref.msa_transition)
    outer_product_mean_weights = create_outer_product_mean_weights(
        from_ref=from_ref.outer_product_mean)
    tri_mul_out_weights = create_triangle_multiplication_node_weights(
        from_ref=from_ref.tri_mul_out)
    tri_mul_in_weights = create_triangle_multiplication_node_weights(
        from_ref=from_ref.tri_mul_in)
    tri_attn_start_weights = create_triangle_attention_node_weights(
        from_ref=from_ref.tri_attn_start)
    tri_attn_end_weights = create_triangle_attention_node_weights(
        from_ref=from_ref.tri_attn_end)
    pair_transition_weights = create_pair_transition_weights(
        from_ref=from_ref.pair_transition)

    return msa_att_row_weights, msa_att_col_weights, msa_transition_weights, outer_product_mean_weights, \
            tri_mul_out_weights, tri_mul_in_weights, tri_attn_start_weights, tri_attn_end_weights, pair_transition_weights


def load_extra_msa_block_weights_ref_torch(module, weights_and_biases):
    msa_att_row_weights, msa_att_col_weights, msa_transition_weights, outer_product_mean_weights, \
            tri_mul_out_weights, tri_mul_in_weights, tri_attn_start_weights, tri_attn_end_weights, pair_transition_weights = weights_and_biases
    load_msa_attention_weights_ref_torch(module.msa_att_row,
                                         msa_att_row_weights)
    load_msa_global_attention_weights_ref_torch(module.msa_att_col,
                                                msa_att_col_weights)
    load_msa_transition_weights_ref_torch(module.msa_transition,
                                          msa_transition_weights)
    load_outer_product_mean_weights_ref_torch(module.outer_product_mean,
                                              outer_product_mean_weights)
    load_triangle_multiplication_node_weights_ref_torch(module.tri_mul_out,
                                                        tri_mul_out_weights)
    load_triangle_multiplication_node_weights_ref_torch(module.tri_mul_in,
                                                        tri_mul_in_weights)
    load_triangle_attention_node_weights_ref_torch(module.tri_attn_start,
                                                   tri_attn_start_weights)
    load_triangle_attention_node_weights_ref_torch(module.tri_attn_end,
                                                   tri_attn_end_weights)
    load_pair_transition_weights_ref_torch(module.pair_transition,
                                           pair_transition_weights)


def load_extra_msa_block_weights_torch(module, weights_and_biases):
    msa_att_row_weights, msa_att_col_weights, msa_transition_weights, outer_product_mean_weights, \
            tri_mul_out_weights, tri_mul_in_weights, tri_attn_start_weights, tri_attn_end_weights, pair_transition_weights = weights_and_biases
    load_msa_attention_weights_torch(module.msa_att_row, msa_att_row_weights)
    load_msa_global_attention_weights_torch(module.msa_att_col,
                                            msa_att_col_weights)
    load_msa_transition_weights_torch(module.msa_transition,
                                      msa_transition_weights)
    load_outer_product_mean_weights_torch(module.outer_product_mean,
                                          outer_product_mean_weights)
    load_triangle_multiplication_node_weights_torch(module.tri_mul_out,
                                                    tri_mul_out_weights)
    load_triangle_multiplication_node_weights_torch(module.tri_mul_in,
                                                    tri_mul_in_weights)
    load_triangle_attention_node_weights_torch(module.tri_attn_start,
                                               tri_attn_start_weights)
    load_triangle_attention_node_weights_torch(module.tri_attn_end,
                                               tri_attn_end_weights)
    load_pair_transition_weights_torch(module.pair_transition,
                                       pair_transition_weights)


def create_input_embedder_weights(from_ref: RefInputEmbedder = None):
    linear_tf_z_i_weight = from_ref.linear_tf_z_i.weight.data
    linear_tf_z_i_bias = from_ref.linear_tf_z_i.bias.data
    linear_tf_z_j_weight = from_ref.linear_tf_z_j.weight.data
    linear_tf_z_j_bias = from_ref.linear_tf_z_j.bias.data
    linear_tf_m_weight = from_ref.linear_tf_m.weight.data
    linear_tf_m_bias = from_ref.linear_tf_m.bias.data
    linear_msa_m_weight = from_ref.linear_msa_m.weight.data
    linear_msa_m_bias = from_ref.linear_msa_m.bias.data
    linear_relpos_weight = from_ref.linear_relpos.weight.data
    linear_relpos_bias = from_ref.linear_relpos.bias.data

    return linear_tf_z_i_weight, linear_tf_z_i_bias, \
        linear_tf_z_j_weight, linear_tf_z_j_bias, \
        linear_tf_m_weight, linear_tf_m_bias, \
        linear_msa_m_weight, linear_msa_m_bias, \
        linear_relpos_weight, linear_relpos_bias


def load_input_embedder_weights_torch(module, weights_and_biases):
    linear_tf_z_i_weight, linear_tf_z_i_bias, \
    linear_tf_z_j_weight, linear_tf_z_j_bias, \
    linear_tf_m_weight, linear_tf_m_bias, \
    linear_msa_m_weight, linear_msa_m_bias, \
    linear_relpos_weight, linear_relpos_bias = weights_and_biases

    module.fused_linear_tf_z.load_weights([{
        "weight":
        linear_tf_z_i_weight.to("cuda"),
        "bias":
        linear_tf_z_i_bias.to("cuda")
    }, {
        "weight":
        linear_tf_z_j_weight.to("cuda"),
        "bias":
        linear_tf_z_j_bias.to("cuda")
    }])
    module.linear_tf_m.load_weights([{
        "weight": linear_tf_m_weight.to("cuda"),
        "bias": linear_tf_m_bias.to("cuda")
    }])
    module.linear_msa_m.load_weights([{
        "weight": linear_msa_m_weight.to("cuda"),
        "bias": linear_msa_m_bias.to("cuda")
    }])
    module.linear_relpos.load_weights([{
        "weight": linear_relpos_weight.to("cuda"),
        "bias": linear_relpos_bias.to("cuda")
    }])


def create_recycling_embedder_weights(from_ref: RefRecyclingEmbedder = None):
    linear_weight = from_ref.linear.weight.data
    linear_bias = from_ref.linear.bias.data
    layer_norm_m_weight = from_ref.layer_norm_m.weight.data
    layer_norm_m_bias = from_ref.layer_norm_m.bias.data
    layer_norm_z_weight = from_ref.layer_norm_z.weight.data
    layer_norm_z_bias = from_ref.layer_norm_z.bias.data

    return linear_weight, linear_bias, layer_norm_m_weight, layer_norm_m_bias, layer_norm_z_weight, layer_norm_z_bias


def load_recycling_embedder_weights_torch(module, weights_and_biases):
    linear_weight, linear_bias, layer_norm_m_weight, layer_norm_m_bias, layer_norm_z_weight, layer_norm_z_bias = weights_and_biases

    module.linear.load_weights([{
        "weight": linear_weight.to("cuda"),
        "bias": linear_bias.to("cuda")
    }])
    module.layer_norm_m.weight.data.copy_(layer_norm_m_weight.to("cuda"))
    module.layer_norm_m.bias.data.copy_(layer_norm_m_bias.to("cuda"))
    module.layer_norm_z.weight.data.copy_(layer_norm_z_weight.to("cuda"))
    module.layer_norm_z.bias.data.copy_(layer_norm_z_bias.to("cuda"))


def create_template_pair_stack_block_weights(
        from_ref: RefTemplatePairStackBlock = None):
    tri_mul_out_weights = create_triangle_multiplication_node_weights(
        from_ref=from_ref.tri_mul_out)
    tri_mul_in_weights = create_triangle_multiplication_node_weights(
        from_ref=from_ref.tri_mul_in)
    tri_attn_start_weights = create_triangle_attention_node_weights(
        from_ref=from_ref.tri_attn_start)
    tri_attn_end_weights = create_triangle_attention_node_weights(
        from_ref=from_ref.tri_attn_end)
    pair_transition_weights = create_pair_transition_weights(
        from_ref=from_ref.pair_transition)
    return tri_mul_out_weights, tri_mul_in_weights, tri_attn_start_weights, tri_attn_end_weights, pair_transition_weights


def load_template_pair_stack_block_weights_torch(module, weights_and_biases):
    tri_mul_out_weights, tri_mul_in_weights, tri_attn_start_weights, tri_attn_end_weights, pair_transition_weights = weights_and_biases
    load_triangle_multiplication_node_weights_torch(module.tri_mul_out,
                                                    tri_mul_out_weights)
    load_triangle_multiplication_node_weights_torch(module.tri_mul_in,
                                                    tri_mul_in_weights)
    load_triangle_attention_node_weights_torch(module.tri_attn_start,
                                               tri_attn_start_weights)
    load_triangle_attention_node_weights_torch(module.tri_attn_end,
                                               tri_attn_end_weights)
    load_pair_transition_weights_torch(module.pair_transition,
                                       pair_transition_weights)


def create_template_pointwise_attention_weights(
        from_ref: RefTemplatePointwiseAttention = None):
    mha_weights = create_triangle_attention_weights(from_ref=from_ref.mha)
    return mha_weights


def load_template_pointwise_attention_weights_torch(module, weights_and_biases):
    mha_weights = weights_and_biases
    load_triangle_attention_weights_torch(module.mha, mha_weights)
