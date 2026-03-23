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

import torch
from torch import nn

from tensorrt_bionemo._torch.modules.openfold2.template import TemplatePairBlock

from tests.common.test_utils.boltz.create_and_load_weights import (
    load_triangle_multiplication_node_weights_torch,
    load_triangle_attention_node_weights_torch,
    load_outer_product_mean_weights_torch)

from tests.common.test_utils.openfold3.create_and_load_weights import (
    create_msa_pair_weighted_averaging_weights,
    create_pair_transition_weights,
    load_msa_pair_weighted_averaging_weights_torch,
    load_pair_transition_weights_torch,
    )
from tests.common.test_utils.openfold3.ref_layers_from_oss import (
    RefTriangleAttentionFromOF3OSS,
    RefTriangleMultiplicationFromOF3OSS,
    RefOuterProductMeanFromOF3OSS,
)

def create_template_pair_block_weights_from_of3oss_torch(
        from_ref: TemplatePairBlock = None):
    tri_mul_out_weights = create_triangle_multiplication_node_weights_from_of3oss(
        from_ref=from_ref.tri_mul_out)
    tri_mul_in_weights = create_triangle_multiplication_node_weights_from_of3oss(
        from_ref=from_ref.tri_mul_in)
    tri_attn_start_weights = create_triangle_attention_node_weights_from_of3oss(
        from_ref=from_ref.tri_att_start)
    tri_attn_end_weights = create_triangle_attention_node_weights_from_of3oss(
        from_ref=from_ref.tri_att_end)
    pair_transition_weights = create_pair_transition_weights(
        from_ref=from_ref.pair_transition)
    return tri_mul_out_weights, tri_mul_in_weights, tri_attn_start_weights, tri_attn_end_weights, pair_transition_weights


def load_template_pair_block_weights_from_of3oss_torch(module,
                                                    weights_and_biases,
                                                    dtype=torch.float32):
    
    tri_mul_out_weights, tri_mul_in_weights, tri_att_start_weights, tri_att_end_weights, pair_transition_weights = weights_and_biases

    load_triangle_multiplication_node_weights_torch(module.tri_mul_out,
                                                    tri_mul_out_weights, dtype)
    load_triangle_multiplication_node_weights_torch(module.tri_mul_in,
                                                    tri_mul_in_weights, dtype)
    load_triangle_attention_node_weights_torch(module.tri_attn_start,
                                               tri_att_start_weights, dtype)
    load_triangle_attention_node_weights_torch(module.tri_attn_end,
                                               tri_att_end_weights, dtype)
    load_pair_transition_weights_torch(module.pair_transition,
                                       pair_transition_weights, dtype)


def create_msa_module_block_weights_from_of3oss_torch(from_ref):
    
    msa_att_row_weights = create_msa_pair_weighted_averaging_weights(
        from_ref=from_ref.msa_att_row)
    msa_transition_weights = create_pair_transition_weights(
        from_ref=from_ref.msa_transition)
    outer_product_mean_weights = create_outer_product_mean_weights_from_of3oss(
        from_ref=from_ref.outer_product_mean)
    tri_mul_out_weights = create_triangle_multiplication_node_weights_from_of3oss(
        from_ref=from_ref.tri_mul_out if not hasattr(from_ref, "pair_stack") else from_ref.pair_stack.tri_mul_out)
    tri_mul_in_weights = create_triangle_multiplication_node_weights_from_of3oss(
        from_ref=from_ref.tri_mul_in if not hasattr(from_ref, "pair_stack") else from_ref.pair_stack.tri_mul_in)
    tri_attn_start_weights = create_triangle_attention_node_weights_from_of3oss(
        from_ref=from_ref.tri_attn_start if not hasattr(from_ref, "pair_stack") else from_ref.pair_stack.tri_att_start)
    tri_attn_end_weights = create_triangle_attention_node_weights_from_of3oss(
        from_ref=from_ref.tri_attn_end if not hasattr(from_ref, "pair_stack") else from_ref.pair_stack.tri_att_end)
    pair_transition_weights = create_pair_transition_weights(
        from_ref=from_ref.pair_transition if not hasattr(from_ref, "pair_stack") else from_ref.pair_stack.pair_transition)

    return msa_att_row_weights, msa_transition_weights, outer_product_mean_weights, \
            tri_mul_out_weights, tri_mul_in_weights, tri_attn_start_weights, tri_attn_end_weights, pair_transition_weights


def load_msa_module_block_weights_from_of3oss_torch(module,
                                                    weights_and_biases,
                                                    dtype=torch.float32):
    
    msa_att_row_weights, msa_transition_weights, outer_product_mean_weights, \
            tri_mul_out_weights, tri_mul_in_weights, tri_attn_start_weights, tri_attn_end_weights, pair_transition_weights = weights_and_biases
    load_msa_pair_weighted_averaging_weights_torch(module.msa_att_row,
                                                   msa_att_row_weights, dtype)

    load_pair_transition_weights_torch(module.msa_transition,
                                       msa_transition_weights, dtype)

    load_outer_product_mean_weights_torch(module.outer_product_mean,
                                          outer_product_mean_weights)
    load_triangle_multiplication_node_weights_torch(module.tri_mul_out,
                                                    tri_mul_out_weights, dtype)
    load_triangle_multiplication_node_weights_torch(module.tri_mul_in,
                                                    tri_mul_in_weights, dtype)
    load_triangle_attention_node_weights_torch(module.tri_attn_start,
                                               tri_attn_start_weights, dtype)
    load_triangle_attention_node_weights_torch(module.tri_attn_end,
                                               tri_attn_end_weights, dtype)
    load_pair_transition_weights_torch(module.pair_transition,
                                       pair_transition_weights, dtype)


def create_outer_product_mean_weights_from_of3oss(
    from_ref: RefOuterProductMeanFromOF3OSS = None):
    names = ["layer_norm", "linear_1", "linear_2", "linear_out"]
    out_as_list = []
    for name in names:
        if hasattr(from_ref, name):
            layer = getattr(from_ref, name)
            out_as_list.append(layer.weight)
            out_as_list.append(getattr(layer.bias, "data", None))
    return tuple(out_as_list)


def create_triangle_attention_node_weights_from_of3oss(
    from_ref: RefTriangleAttentionFromOF3OSS = None):

    ret = {
        "layer_norm":
        (from_ref.layer_norm.weight.data, from_ref.layer_norm.bias.data),
        "linear":from_ref.linear_z.weight.data,
        "mha": create_triangle_attention_weights_from_of3_oss(from_ref=from_ref.mha)
    }
    return ret


def create_triangle_attention_weights_from_of3_oss(
    from_ref: RefTriangleAttentionFromOF3OSS):
    
    q_weight = from_ref.linear_q.weight.data
    q_bias = getattr(from_ref.linear_q.bias, "data", None)
    k_weight = from_ref.linear_k.weight.data
    k_bias = getattr(from_ref.linear_k.bias, "data", None)

    v_weight = from_ref.linear_v.weight.data
    v_bias = getattr(from_ref.linear_v.bias, "data", None)

    out_weight = from_ref.linear_o.weight.data
    out_bias = getattr(from_ref.linear_o.bias, "data", None)

    gating_weight = None
    gating_bias = None
    if from_ref.linear_g is not None:
        gating_weight = from_ref.linear_g.weight.data
        gating_bias = getattr(from_ref.linear_g.bias, "data", None)

    return q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, out_weight, out_bias, gating_weight, gating_bias


def create_triangle_multiplication_node_weights_from_of3oss(
    from_ref: RefTriangleMultiplicationFromOF3OSS = None):
    """Reformat the weights from an OF3 OSS checkpoint to TRTBNM _torch format.
    """
    
    norm_in_weight = getattr(from_ref.layer_norm_in.weight, "data", None)
    norm_in_bias = getattr(from_ref.layer_norm_in.bias, "data", None)

    a_g_weight = from_ref.linear_a_g.weight.data
    a_g_bias = getattr(from_ref.linear_a_g.bias, "data", None)

    a_p_weight = from_ref.linear_a_p.weight.data
    a_p_bias = getattr(from_ref.linear_a_p.bias, "data", None)
    
    b_g_weight = from_ref.linear_b_g.weight.data
    b_g_bias = getattr(from_ref.linear_b_g.bias, "data", None)
    
    b_p_weight = from_ref.linear_b_p.weight.data
    b_p_bias = getattr(from_ref.linear_b_p.bias, "data", None)

    p_in_weight = torch.cat((a_p_weight, b_p_weight), dim=0)
    if a_p_bias is not None and b_p_bias is not None:
        p_in_bias = torch.cat((a_p_bias, b_p_bias), dim=0)
    else:
        p_in_bias = None
    
    g_in_weight = torch.cat((a_g_weight, b_g_weight), dim=0)
    if a_g_bias is not None and b_g_bias is not None:
        g_in_bias = torch.cat((a_g_bias, b_g_bias), dim=0)
    else:
        g_in_bias = None

    norm_out_weight = from_ref.layer_norm_out.weight.data
    norm_out_bias = getattr(from_ref.layer_norm_out.bias, "data", None)

    p_out_weight = from_ref.linear_z.weight.data
    p_out_bias = getattr(from_ref.linear_z.bias, "data", None)

    g_out_weight = from_ref.linear_g.weight.data
    g_out_bias = getattr(from_ref.linear_g.bias, "data", None)

    return norm_in_weight, norm_in_bias, p_in_weight, p_in_bias, g_in_weight, g_in_bias, \
            norm_out_weight, norm_out_bias, p_out_weight, p_out_bias, g_out_weight, g_out_bias