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
from tensorrt_llm.models.convert_utils import split

from tensorrt_bionemo.confs.modules.transformers import PairformerConfig
from tensorrt_bionemo.hf.checkpoints import load_hf_weights
from tensorrt_bionemo.mapping import Mapping, create_max_tp_mapping


def get_pairwise_attn_weights(mapping: Mapping,
                              state_dict: dict,
                              prefix: str,
                              tbm_prefix: str,
                              max_attention_pairwise_tp_size: bool = True,
                              num_heads: int = 16):
    init_norm_weight = state_dict[f"{prefix}.norm_s.weight"]
    init_norm_bias = state_dict[f"{prefix}.norm_s.bias"]
    q_weight = state_dict[f"{prefix}.proj_q.weight"]
    q_bias = state_dict[f"{prefix}.proj_q.bias"]
    k_weight = state_dict[f"{prefix}.proj_k.weight"]
    v_weight = state_dict[f"{prefix}.proj_v.weight"]
    o_weight = state_dict[f"{prefix}.proj_o.weight"]
    g_weight = state_dict[f"{prefix}.proj_g.weight"]

    norm_z_weight = state_dict[f"{prefix}.proj_z.0.weight"]
    norm_z_bias = state_dict[f"{prefix}.proj_z.0.bias"]
    z_weight = state_dict[f"{prefix}.proj_z.1.weight"]

    if max_attention_pairwise_tp_size:
        mapping = create_max_tp_mapping(mapping=mapping, dim=num_heads)
    tp_size = mapping.tp_size
    tp_rank = mapping.tp_rank
    if tp_size > 1:
        q_weight = split(q_weight, tp_size, tp_rank, 0)
        q_bias = split(q_bias, tp_size, tp_rank, 0)
        k_weight = split(k_weight, tp_size, tp_rank, 0)
        v_weight = split(v_weight, tp_size, tp_rank, 0)
        o_weight = split(o_weight, tp_size, tp_rank, 1)
        g_weight = split(g_weight, tp_size, tp_rank, 0)
        z_weight = split(z_weight, tp_size, tp_rank, 0)

    kv_weights = torch.cat([k_weight, v_weight], dim=0)

    ret = {
        f"{tbm_prefix}.norm_s.weight": init_norm_weight,
        f"{tbm_prefix}.norm_s.bias": init_norm_bias,
        f"{tbm_prefix}.proj_q.weight": q_weight,
        f"{tbm_prefix}.proj_q.bias": q_bias,
        f"{tbm_prefix}.proj_kv.weight": kv_weights,
        f"{tbm_prefix}.proj_g.weight": g_weight,
        f"{tbm_prefix}.proj_z_norm.weight": norm_z_weight,
        f"{tbm_prefix}.proj_z_norm.bias": norm_z_bias,
        f"{tbm_prefix}.proj_z.weight": z_weight,
        f"{tbm_prefix}.proj_o.weight": o_weight,
    }
    return ret


def get_tri_attn_node_weights(mapping: Mapping, state_dict: dict, prefix: str,
                              tbm_prefix: str):
    layer_norm_weight = state_dict[f"{prefix}.layer_norm.weight"]
    layer_norm_bias = state_dict[f"{prefix}.layer_norm.bias"]
    linear_weight = state_dict[f"{prefix}.linear.weight"]

    mha_q_weight = state_dict[f"{prefix}.mha.linear_q.weight"]
    mha_k_weight = state_dict[f"{prefix}.mha.linear_k.weight"]
    mha_v_weight = state_dict[f"{prefix}.mha.linear_v.weight"]
    mha_o_weight = state_dict[f"{prefix}.mha.linear_o.weight"]
    mha_g_weight = state_dict[f"{prefix}.mha.linear_g.weight"]

    tp_size = mapping.tp_size
    tp_rank = mapping.tp_rank
    if tp_size > 1:
        linear_weight = split(linear_weight, tp_size, tp_rank, 0)
        mha_q_weight = split(mha_q_weight, tp_size, tp_rank, 0)
        mha_k_weight = split(mha_k_weight, tp_size, tp_rank, 0)
        mha_v_weight = split(mha_v_weight, tp_size, tp_rank, 0)
        mha_o_weight = split(mha_o_weight, tp_size, tp_rank, 1)
        mha_g_weight = split(mha_g_weight, tp_size, tp_rank, 0)
    mha_qkv_weights = torch.cat([mha_q_weight, mha_k_weight, mha_v_weight],
                                dim=0)

    ret = {
        f"{tbm_prefix}.layer_norm.weight": layer_norm_weight,
        f"{tbm_prefix}.layer_norm.bias": layer_norm_bias,
        f"{tbm_prefix}.linear.weight": linear_weight,
        f"{tbm_prefix}.mha.qkv_proj.weight": mha_qkv_weights,
        f"{tbm_prefix}.mha.o_proj.weight": mha_o_weight,
        f"{tbm_prefix}.mha.g_proj.weight": mha_g_weight,
    }
    return ret


def get_tri_mul_node_weights(mapping: Mapping, state_dict: dict, prefix: str,
                             tbm_prefix: str):
    norm_in_weight = state_dict[f"{prefix}.norm_in.weight"]
    norm_in_bias = state_dict[f"{prefix}.norm_in.bias"]
    p_in_weight = state_dict[f"{prefix}.p_in.weight"]
    g_in_weight = state_dict[f"{prefix}.g_in.weight"]
    norm_out_weight = state_dict[f"{prefix}.norm_out.weight"]
    norm_out_bias = state_dict[f"{prefix}.norm_out.bias"]
    p_out_weight = state_dict[f"{prefix}.p_out.weight"]
    g_out_weight = state_dict[f"{prefix}.g_out.weight"]

    tp_size = mapping.tp_size
    tp_rank = mapping.tp_rank
    if tp_size > 1:
        dim = p_in_weight.shape[0] // 2
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
    ret = {
        f"{tbm_prefix}.norm_in.weight": norm_in_weight,
        f"{tbm_prefix}.norm_in.bias": norm_in_bias,
        f"{tbm_prefix}.p_in.weight": p_in_weight,
        f"{tbm_prefix}.g_in.weight": g_in_weight,
        f"{tbm_prefix}.norm_out.weight": norm_out_weight,
        f"{tbm_prefix}.norm_out.bias": norm_out_bias,
        f"{tbm_prefix}.p_out.weight": p_out_weight,
        f"{tbm_prefix}.g_out.weight": g_out_weight,
    }
    return ret


def get_transition_weights(mapping: Mapping,
                           state_dict: dict,
                           prefix: str,
                           tbm_prefix: str,
                           max_transition_tp_size: bool = True,
                           dim: int = 128):
    norm_weight = state_dict[f"{prefix}.norm.weight"]
    norm_bias = state_dict[f"{prefix}.norm.bias"]
    fc1_weight = state_dict[f"{prefix}.fc1.weight"]
    fc2_weight = state_dict[f"{prefix}.fc2.weight"]
    fc3_weight = state_dict[f"{prefix}.fc3.weight"]

    if max_transition_tp_size:
        mapping = create_max_tp_mapping(mapping=mapping, dim=dim)
    tp_size = mapping.tp_size
    tp_rank = mapping.tp_rank
    if tp_size > 1:
        fc1_weight = split(fc1_weight, tp_size, tp_rank, 0)
        fc2_weight = split(fc2_weight, tp_size, tp_rank, 0)
        fc3_weight = split(fc3_weight, tp_size, tp_rank, 1)
    fused_fc2_fc1_weight = torch.cat([fc2_weight, fc1_weight], dim=0)

    ret = {
        f"{tbm_prefix}.norm.weight": norm_weight,
        f"{tbm_prefix}.norm.bias": norm_bias,
        f"{tbm_prefix}.fused_fc2_fc1.weight": fused_fc2_fc1_weight,
        f"{tbm_prefix}.fc3.weight": fc3_weight,
    }
    return ret


def convert_hf_pairformer(config: PairformerConfig,
                          mapping: Mapping,
                          pairformer_type: str = "prediction"):
    """
    Convert a pairformer model from a Hugging Face checkpoint to a TensorRT model weights.

    Args:
        mapping: A mapping object that defines the mapping of the model.
        pairformer_type: The type of pairformer to convert. 'structure' or 'confidence'
    """
    prefix = "pairformer_module.layers"
    if pairformer_type == "confidence":
        prefix = f"confidence_module.{prefix}"
    tbm_prefix = "layers"
    weights = {}
    state_dict = load_hf_weights(name="boltz-1")

    for i in range(config.num_blocks):
        layer_prefix = f"{prefix}.{i}"
        layer_tbm_prefix = f"{tbm_prefix}.{i}"
        weights.update(
            get_pairwise_attn_weights(mapping, state_dict,
                                      f"{layer_prefix}.attention",
                                      f"{layer_tbm_prefix}.attention",
                                      config.max_attention_pairwise_tp_size,
                                      config.num_heads))
        weights.update(
            get_tri_attn_node_weights(mapping, state_dict,
                                      f"{layer_prefix}.tri_att_start",
                                      f"{layer_tbm_prefix}.tri_attn_start"))
        weights.update(
            get_tri_attn_node_weights(mapping, state_dict,
                                      f"{layer_prefix}.tri_att_end",
                                      f"{layer_tbm_prefix}.tri_attn_end"))
        weights.update(
            get_tri_mul_node_weights(mapping, state_dict,
                                     f"{layer_prefix}.tri_mul_out",
                                     f"{layer_tbm_prefix}.tri_mul_out"))
        weights.update(
            get_tri_mul_node_weights(mapping, state_dict,
                                     f"{layer_prefix}.tri_mul_in",
                                     f"{layer_tbm_prefix}.tri_mul_in"))
        weights.update(
            get_transition_weights(mapping, state_dict,
                                   f"{layer_prefix}.transition_s",
                                   f"{layer_tbm_prefix}.transition_s",
                                   config.max_transition_tp_size,
                                   config.token_s * 4))
        weights.update(
            get_transition_weights(mapping, state_dict,
                                   f"{layer_prefix}.transition_z",
                                   f"{layer_tbm_prefix}.transition_z",
                                   config.max_transition_tp_size,
                                   config.token_z * 4))
    return weights
