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
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.models.convert_utils import split

from tensorrt_bionemo.hubs import load_weights
from tensorrt_bionemo.mapping import Mapping, create_max_tp_mapping

from .configs import MSAModuleConfig, PairformerConfig, TokenTransformerConfig


def get_pairwise_attn_weights(mapping: Mapping,
                              state_dict: dict,
                              prefix: str,
                              tbm_prefix: str,
                              max_attention_pairwise_tp_size: bool = True,
                              num_heads: int = 16,
                              attention_initial_norm: bool = True,
                              compute_pair_bias: bool = True,
                              dtype: str = "float32"):
    torch_dtype = str_dtype_to_torch(dtype)

    q_weight = state_dict[f"{prefix}.proj_q.weight"]
    q_bias = state_dict[f"{prefix}.proj_q.bias"]
    k_weight = state_dict[f"{prefix}.proj_k.weight"]
    v_weight = state_dict[f"{prefix}.proj_v.weight"]
    o_weight = state_dict[f"{prefix}.proj_o.weight"]
    g_weight = state_dict[f"{prefix}.proj_g.weight"]

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
        f"{tbm_prefix}.proj_q.weight": q_weight.to(torch_dtype),
        f"{tbm_prefix}.proj_q.bias": q_bias.to(torch_dtype),
        f"{tbm_prefix}.proj_kv.weight": kv_weights.to(torch_dtype),
        f"{tbm_prefix}.proj_g.weight": g_weight.to(torch_dtype),
        f"{tbm_prefix}.proj_o.weight": o_weight.to(torch_dtype),
    }
    if attention_initial_norm:
        init_norm_weight = state_dict[f"{prefix}.norm_s.weight"]
        init_norm_bias = state_dict[f"{prefix}.norm_s.bias"]
        ret[f"{tbm_prefix}.norm_s.weight"] = init_norm_weight.to(torch_dtype)
        ret[f"{tbm_prefix}.norm_s.bias"] = init_norm_bias.to(torch_dtype)

    if compute_pair_bias:
        # This flag is used to in TokenTransformer version 2, where we don't compute pair bias in the attention layer
        norm_z_weight = state_dict[f"{prefix}.proj_z.0.weight"]
        norm_z_bias = state_dict.get(f"{prefix}.proj_z.0.bias", None)
        z_weight = state_dict[f"{prefix}.proj_z.1.weight"]
        if tp_size > 1:
            z_weight = split(z_weight, tp_size, tp_rank, 0)
        ret.update({
            f"{tbm_prefix}.proj_z_norm.weight":
            norm_z_weight.to(torch_dtype),
            f"{tbm_prefix}.proj_z.weight":
            z_weight.to(torch_dtype),
        })
        if norm_z_bias is not None:
            ret.update({
                f"{tbm_prefix}.proj_z_norm.bias":
                norm_z_bias.to(torch_dtype),
            })
    return ret


def get_tri_attn_node_weights(mapping: Mapping,
                              state_dict: dict,
                              prefix: str,
                              tbm_prefix: str,
                              dtype: str = "float32"):
    torch_dtype = str_dtype_to_torch(dtype)
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
        f"{tbm_prefix}.layer_norm.weight": layer_norm_weight.to(torch_dtype),
        f"{tbm_prefix}.layer_norm.bias": layer_norm_bias.to(torch_dtype),
        f"{tbm_prefix}.linear.weight": linear_weight.to(torch_dtype),
        f"{tbm_prefix}.mha.qkv_proj.weight": mha_qkv_weights.to(torch_dtype),
        f"{tbm_prefix}.mha.o_proj.weight": mha_o_weight.to(torch_dtype),
        f"{tbm_prefix}.mha.g_proj.weight": mha_g_weight.to(torch_dtype),
    }
    return ret


def get_tri_mul_node_weights(mapping: Mapping,
                             state_dict: dict,
                             prefix: str,
                             tbm_prefix: str,
                             max_tri_mul_tp_size: bool = True,
                             dtype: str = "float32"):
    torch_dtype = str_dtype_to_torch(dtype)
    norm_in_weight = state_dict[f"{prefix}.norm_in.weight"]
    norm_in_bias = state_dict[f"{prefix}.norm_in.bias"]
    p_in_weight = state_dict[f"{prefix}.p_in.weight"]
    g_in_weight = state_dict[f"{prefix}.g_in.weight"]
    norm_out_weight = state_dict[f"{prefix}.norm_out.weight"]
    norm_out_bias = state_dict[f"{prefix}.norm_out.bias"]
    p_out_weight = state_dict[f"{prefix}.p_out.weight"]
    g_out_weight = state_dict[f"{prefix}.g_out.weight"]

    if max_tri_mul_tp_size:
        mapping = create_max_tp_mapping(mapping=mapping,
                                        dim=p_in_weight.shape[0])
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
        f"{tbm_prefix}.norm_in.weight": norm_in_weight.to(torch_dtype),
        f"{tbm_prefix}.norm_in.bias": norm_in_bias.to(torch_dtype),
        f"{tbm_prefix}.p_in.weight": p_in_weight.to(torch_dtype),
        f"{tbm_prefix}.g_in.weight": g_in_weight.to(torch_dtype),
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
                           dim: int = 128,
                           dtype: str = "float32"):
    torch_dtype = str_dtype_to_torch(dtype)
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
        f"{tbm_prefix}.norm.weight": norm_weight.to(torch_dtype),
        f"{tbm_prefix}.norm.bias": norm_bias.to(torch_dtype),
        f"{tbm_prefix}.fused_fc2_fc1.weight":
        fused_fc2_fc1_weight.to(torch_dtype),
        f"{tbm_prefix}.fc3.weight": fc3_weight.to(torch_dtype),
    }
    return ret


def convert_hf_pairformer(config: PairformerConfig,
                          mapping: Mapping,
                          pairformer_type: str = "structure",
                          local_checkpoint: str = None,
                          model_name: str = "boltz-1"):
    """
    Convert a pairformer model from a Hugging Face checkpoint to a TensorRT model weights.

    Args:
        mapping: A mapping object that defines the mapping of the model.
        pairformer_type: The type of pairformer to convert. 'structure' or 'confidence'
    """
    mapping = mapping if mapping is not None else Mapping()
    prefix = "pairformer_module.layers"
    if pairformer_type == "confidence":
        prefix = f"confidence_module.{prefix}"
    tbm_prefix = "layers"
    weights = {}

    state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    logger.info(
        f"Loading weights for {pairformer_type} pairformer, dtype: {config.dtype}"
    )
    for i in range(config.num_blocks):
        layer_prefix = f"{prefix}.{i}"
        layer_tbm_prefix = f"{tbm_prefix}.{i}"
        if not config.no_update_s:
            weights.update(
                get_pairwise_attn_weights(mapping,
                                          state_dict,
                                          f"{layer_prefix}.attention",
                                          f"{layer_tbm_prefix}.attention",
                                          config.max_attention_pairwise_tp_size,
                                          config.num_heads,
                                          dtype=config.dtype))
        weights.update(
            get_tri_attn_node_weights(mapping,
                                      state_dict,
                                      f"{layer_prefix}.tri_att_start",
                                      f"{layer_tbm_prefix}.tri_attn_start",
                                      dtype=config.dtype))
        weights.update(
            get_tri_attn_node_weights(mapping,
                                      state_dict,
                                      f"{layer_prefix}.tri_att_end",
                                      f"{layer_tbm_prefix}.tri_attn_end",
                                      dtype=config.dtype))
        weights.update(
            get_tri_mul_node_weights(mapping,
                                     state_dict,
                                     f"{layer_prefix}.tri_mul_out",
                                     f"{layer_tbm_prefix}.tri_mul_out",
                                     config.max_tri_mul_tp_size,
                                     dtype=config.dtype))
        weights.update(
            get_tri_mul_node_weights(mapping,
                                     state_dict,
                                     f"{layer_prefix}.tri_mul_in",
                                     f"{layer_tbm_prefix}.tri_mul_in",
                                     config.max_tri_mul_tp_size,
                                     dtype=config.dtype))
        if not config.no_update_s:
            weights.update(
                get_transition_weights(mapping,
                                       state_dict,
                                       f"{layer_prefix}.transition_s",
                                       f"{layer_tbm_prefix}.transition_s",
                                       config.max_transition_tp_size,
                                       config.token_s * 4,
                                       dtype=config.dtype))
        weights.update(
            get_transition_weights(mapping,
                                   state_dict,
                                   f"{layer_prefix}.transition_z",
                                   f"{layer_tbm_prefix}.transition_z",
                                   config.max_transition_tp_size,
                                   config.token_z * 4,
                                   dtype=config.dtype))
    return weights


def convert_hf_pairformer_torch(config: PairformerConfig = None,
                                mapping: Mapping = None,
                                local_checkpoint: str = None,
                                model_name: str = "boltz-1",
                                weights: dict = None,
                                pairformer_type: str = "structure",
                                prefix: str = None,
                                **kwargs):
    """
    This function is used to convert PyTorch weights to dict for pairformer v1 torch backend.
    Args:
        local_checkpoint: The directory to load the checkpoint from. If local_checkpoint is None, the function will load from HuggingFace
        world_size: The number of processes to use.
        rank: The rank of the process.
        weights: The model weights to load into the module. If weights is None, the function will load the weights from the local_checkpoint.
        pairformer_type: The type of pairformer v1 to convert. 'structure' or 'confidence'
        num_layers: The number of layers to convert. If num_layers is None, the function will automatically calculate the number of layers.
    Returns:
        dict: The weights is loaded from the Pairformer Torch backend.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights

    module_state_dict = {}
    if prefix is None:
        prefix = "pairformer_module."
        if pairformer_type == "confidence":
            prefix = f"confidence_module.{prefix}"

    for k, v in state_dict.items():
        if k.startswith(prefix):
            module_state_dict[k.replace(prefix, "")] = v
    tbnm_state_dict = {}
    for i in range(config.num_blocks):
        # weight for pairwise attention
        if f"layers.{i}.attention.norm_s.weight" in module_state_dict:
            tbnm_state_dict[f"layers.{i}.attention.norm_s"] = [{
                'weight':
                module_state_dict[f"layers.{i}.attention.norm_s.weight"],
                'bias':
                module_state_dict[f"layers.{i}.attention.norm_s.bias"]
            }]
        tbnm_state_dict[f"layers.{i}.attention.proj_q"] = [{
            'weight':
            module_state_dict[f"layers.{i}.attention.proj_q.weight"],
            'bias':
            module_state_dict[f"layers.{i}.attention.proj_q.bias"]
        }]
        tbnm_state_dict[f"layers.{i}.attention.proj_kv"] = [{
            'weight':
            module_state_dict[f"layers.{i}.attention.proj_k.weight"],
        }, {
            'weight':
            module_state_dict[f"layers.{i}.attention.proj_v.weight"],
        }]
        tbnm_state_dict[f"layers.{i}.attention.proj_g"] = [{
            'weight':
            module_state_dict[f"layers.{i}.attention.proj_g.weight"],
        }]
        if f"layers.{i}.attention.proj_z.0.weight" in module_state_dict:  # for attention pair bias v2
            tbnm_state_dict[f"layers.{i}.attention.proj_z.0"] = [{
                'weight':
                module_state_dict[f"layers.{i}.attention.proj_z.0.weight"],
                'bias':
                module_state_dict[f"layers.{i}.attention.proj_z.0.bias"]
            }]
            tbnm_state_dict[f"layers.{i}.attention.proj_z.1"] = [{
                'weight':
                module_state_dict[f"layers.{i}.attention.proj_z.1.weight"],
            }]
        tbnm_state_dict[f"layers.{i}.attention.proj_o"] = [{
            'weight':
            module_state_dict[f"layers.{i}.attention.proj_o.weight"]
        }]

        # weight for tri_mul_out and tri_mul_in
        for name in ["tri_mul_out", "tri_mul_in"]:
            tbnm_state_dict[f"layers.{i}.{name}.norm_in"] = [{
                'weight':
                module_state_dict[f"layers.{i}.{name}.norm_in.weight"],
                'bias':
                module_state_dict[f"layers.{i}.{name}.norm_in.bias"]
            }]

            w = module_state_dict[f"layers.{i}.{name}.p_in.weight"]
            p_in_0_weight, p_in_1_weight = w.chunk(2, dim=0)
            tbnm_state_dict[f"layers.{i}.{name}.p_in"] = [{
                'weight':
                p_in_0_weight,
            }, {
                'weight':
                p_in_1_weight,
            }]

            w = module_state_dict[f"layers.{i}.{name}.g_in.weight"]
            g_in_0_weight, g_in_1_weight = w.chunk(2, dim=0)
            tbnm_state_dict[f"layers.{i}.{name}.g_in"] = [{
                'weight':
                g_in_0_weight,
            }, {
                'weight':
                g_in_1_weight,
            }]
            tbnm_state_dict[f"layers.{i}.{name}.norm_out"] = [{
                'weight':
                module_state_dict[f"layers.{i}.{name}.norm_out.weight"],
                'bias':
                module_state_dict[f"layers.{i}.{name}.norm_out.bias"]
            }]
            tbnm_state_dict[f"layers.{i}.{name}.p_out"] = [{
                'weight':
                module_state_dict[f"layers.{i}.{name}.p_out.weight"],
            }]
            tbnm_state_dict[f"layers.{i}.{name}.g_out"] = [{
                'weight':
                module_state_dict[f"layers.{i}.{name}.g_out.weight"],
            }]

        # weight for tri_attn_start and tri_attn_end
        for name in ["start", "end"]:
            tbnm_state_dict[f"layers.{i}.tri_attn_{name}.layer_norm"] = [{
                'weight':
                module_state_dict[
                    f"layers.{i}.tri_att_{name}.layer_norm.weight"],
                'bias':
                module_state_dict[f"layers.{i}.tri_att_{name}.layer_norm.bias"]
            }]
            tbnm_state_dict[f"layers.{i}.tri_attn_{name}.linear"] = [{
                'weight':
                module_state_dict[f"layers.{i}.tri_att_{name}.linear.weight"],
            }]
            tbnm_state_dict[f"layers.{i}.tri_attn_{name}.mha.qkv_proj"] = [{
                'weight':
                module_state_dict[
                    f"layers.{i}.tri_att_{name}.mha.linear_q.weight"],
            }, {
                'weight':
                module_state_dict[
                    f"layers.{i}.tri_att_{name}.mha.linear_k.weight"],
            }, {
                'weight':
                module_state_dict[
                    f"layers.{i}.tri_att_{name}.mha.linear_v.weight"],
            }]
            tbnm_state_dict[f"layers.{i}.tri_attn_{name}.mha.o_proj"] = [{
                'weight':
                module_state_dict[
                    f"layers.{i}.tri_att_{name}.mha.linear_o.weight"],
            }]
            tbnm_state_dict[f"layers.{i}.tri_attn_{name}.mha.g_proj"] = [{
                'weight':
                module_state_dict[
                    f"layers.{i}.tri_att_{name}.mha.linear_g.weight"],
            }]
        # weight for transition_s and transition_z
        for name in ["s", "z"]:
            tbnm_state_dict[f"layers.{i}.transition_{name}.norm"] = [{
                'weight':
                module_state_dict[f"layers.{i}.transition_{name}.norm.weight"],
                'bias':
                module_state_dict[f"layers.{i}.transition_{name}.norm.bias"]
            }]
            tbnm_state_dict[f"layers.{i}.transition_{name}.fused_fc2_fc1"] = [{
                'weight':
                module_state_dict[f"layers.{i}.transition_{name}.fc2.weight"],
            }, {
                'weight':
                module_state_dict[f"layers.{i}.transition_{name}.fc1.weight"],
            }]
            tbnm_state_dict[f"layers.{i}.transition_{name}.fc3"] = [{
                'weight':
                module_state_dict[f"layers.{i}.transition_{name}.fc3.weight"],
            }]
    return tbnm_state_dict


def get_adaln_weights(mapping: Mapping,
                      state_dict: dict,
                      prefix: str,
                      tbm_prefix: str,
                      dim: int,
                      dim_single_cond: int,
                      dtype: str = "float32"):
    torch_dtype = str_dtype_to_torch(dtype)

    s_norm_weight = state_dict[f"{prefix}.s_norm.weight"]
    a_norm_weight = torch.ones([dim],
                               dtype=torch_dtype,
                               device=s_norm_weight.device)
    s_scale_weight = state_dict[f"{prefix}.s_scale.weight"]
    s_scale_bias = state_dict[f"{prefix}.s_scale.bias"]
    s_bias_weight = state_dict[f"{prefix}.s_bias.weight"]
    s_bias_bias = torch.zeros([dim],
                              dtype=torch_dtype,
                              device=s_scale_bias.device)

    tp_size = mapping.tp_size
    tp_rank = mapping.tp_rank
    if tp_size > 1:
        s_scale_weight = split(s_scale_weight, tp_size, tp_rank, 0)
        s_scale_bias = split(s_scale_bias, tp_size, tp_rank, 0)
        s_bias_weight = split(s_bias_weight, tp_size, tp_rank, 0)
        s_bias_bias = split(s_bias_bias, tp_size, tp_rank, 0)

    fused_s_scale_s_bias_weights = torch.cat([s_scale_weight, s_bias_weight],
                                             dim=0)
    fused_s_scale_s_bias_bias = torch.cat([s_scale_bias, s_bias_bias], dim=0)

    ret = {
        f"{tbm_prefix}.a_norm.weight":
        a_norm_weight.to(torch_dtype),
        f"{tbm_prefix}.s_norm.weight":
        s_norm_weight.to(torch_dtype),
        f"{tbm_prefix}.fused_s_scale_s_bias.weight":
        fused_s_scale_s_bias_weights.to(torch_dtype),
        f"{tbm_prefix}.fused_s_scale_s_bias.bias":
        fused_s_scale_s_bias_bias.to(torch_dtype),
    }
    return ret


def get_conditioned_transition_block_weights(mapping: Mapping,
                                             state_dict: dict,
                                             prefix: str,
                                             tbm_prefix: str,
                                             dim: int,
                                             dim_single_cond: int,
                                             expansion_factor: int = 2,
                                             dtype: str = "float32"):
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}
    ret.update(
        get_adaln_weights(mapping,
                          state_dict,
                          f"{prefix}.adaln",
                          f"{tbm_prefix}.adaln",
                          dim,
                          dim_single_cond,
                          dtype=dtype))
    swish_gate_weight = state_dict[f"{prefix}.swish_gate.0.weight"]
    a_to_b_weight = state_dict[f"{prefix}.a_to_b.weight"]
    b_to_a_weight = state_dict[f"{prefix}.b_to_a.weight"]
    output_projection_weight = state_dict[
        f"{prefix}.output_projection.0.weight"]
    output_projection_bias = state_dict[f"{prefix}.output_projection.0.bias"]

    dim_inner = int(dim * expansion_factor)
    tp_size = mapping.tp_size
    tp_rank = mapping.tp_rank
    if tp_size > 1:
        swish_gate_weight_0, swish_gate_weight_1 = swish_gate_weight.split(
            [dim_inner, dim_inner], dim=0)

        swish_gate_weight_0 = split(swish_gate_weight_0, tp_size, tp_rank, 0)
        swish_gate_weight_1 = split(swish_gate_weight_1, tp_size, tp_rank, 0)
        swish_gate_weight = torch.cat(
            [swish_gate_weight_0, swish_gate_weight_1], dim=0)
        a_to_b_weight = split(a_to_b_weight, tp_size, tp_rank, 0)
        b_to_a_weight = split(b_to_a_weight, tp_size, tp_rank, 1)
        output_projection_weight = split(output_projection_weight, tp_size,
                                         tp_rank, 0)
        output_projection_bias = split(output_projection_bias, tp_size, tp_rank,
                                       0)

    fused_swl_a_to_b_weight = torch.cat([swish_gate_weight, a_to_b_weight],
                                        dim=0)
    ret.update({
        f"{tbm_prefix}.fused_swl_a_to_b.weight":
        fused_swl_a_to_b_weight.to(torch_dtype),
        f"{tbm_prefix}.b_to_a.weight":
        b_to_a_weight.to(torch_dtype),
        f"{tbm_prefix}.output_projection.weight":
        output_projection_weight.to(torch_dtype),
        f"{tbm_prefix}.output_projection.bias":
        output_projection_bias.to(torch_dtype),
    })
    return ret


def get_output_projection_weights(mapping: Mapping,
                                  state_dict: dict,
                                  prefix: str,
                                  tbm_prefix: str,
                                  dtype: str = "float32"):
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}
    output_projection_weight = state_dict[f"{prefix}.0.weight"]
    output_projection_bias = state_dict[f"{prefix}.0.bias"]
    tp_size = mapping.tp_size
    tp_rank = mapping.tp_rank
    if tp_size > 1:
        output_projection_weight = split(output_projection_weight, tp_size,
                                         tp_rank, 0)
        output_projection_bias = split(output_projection_bias, tp_size, tp_rank,
                                       0)

    ret.update({
        f"{tbm_prefix}.weight": output_projection_weight.to(torch_dtype),
        f"{tbm_prefix}.bias": output_projection_bias.to(torch_dtype),
    })
    return ret


def get_post_norm_weights(mapping: Mapping,
                          state_dict: dict,
                          prefix: str,
                          tbm_prefix: str,
                          dtype: str = "float32"):
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}
    weight_key = f"{prefix}.weight"
    bias_key = f"{prefix}.bias"
    if weight_key in state_dict and bias_key in state_dict:
        return {
            f"{tbm_prefix}.weight": state_dict[weight_key].to(torch_dtype),
            f"{tbm_prefix}.bias": state_dict[bias_key].to(torch_dtype),
        }
    return ret


def convert_hf_token_transformer(config: TokenTransformerConfig = None,
                                 mapping: Mapping = None,
                                 local_checkpoint: str = None,
                                 model_name: str = "boltz-1"):
    """
    Convert a token transformer model from a Hugging Face checkpoint to a TensorRT model weights.
    """
    mapping = mapping if mapping is not None else Mapping()
    prefix = "structure_module.score_model.token_transformer.layers"
    tbm_prefix = "layers"
    weights = {}
    state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    logger.info(f"Loading weights for token transformer, dtype: {config.dtype}")
    for i in range(config.num_blocks):
        layer_prefix = f"{prefix}.{i}"
        layer_tbm_prefix = f"{tbm_prefix}.{i}"
        weights.update(
            get_adaln_weights(mapping,
                              state_dict,
                              f"{layer_prefix}.adaln",
                              f"{layer_tbm_prefix}.adaln",
                              config.dim,
                              config.dim_single_cond,
                              dtype=config.dtype))
        weights.update(
            get_pairwise_attn_weights(
                mapping,
                state_dict,
                f"{layer_prefix}.pair_bias_attn",
                f"{layer_tbm_prefix}.pair_bias_attn",
                max_attention_pairwise_tp_size=True,
                num_heads=config.num_heads,
                attention_initial_norm=config.attention_initial_norm,
                compute_pair_bias=config.version == "v1",
                dtype=config.dtype))
        weights.update(
            get_conditioned_transition_block_weights(
                mapping,
                state_dict,
                f"{layer_prefix}.transition",
                f"{layer_tbm_prefix}.transition",
                config.dim,
                config.dim_single_cond,
                expansion_factor=config.expansion_factor,
                dtype=config.dtype))
        weights.update(
            get_output_projection_weights(
                mapping,
                state_dict,
                f"{layer_prefix}.output_projection",
                f"{layer_tbm_prefix}.output_projection",
                dtype=config.dtype))
        weights.update(
            get_post_norm_weights(mapping,
                                  state_dict,
                                  f"{layer_prefix}.post_lnorm",
                                  f"{layer_tbm_prefix}.post_lnorm",
                                  dtype=config.dtype))

    return weights


def convert_hf_token_transformer_torch(config: TokenTransformerConfig,
                                       mapping: Mapping = None,
                                       local_checkpoint: str = None,
                                       model_name: str = "boltz-1",
                                       weights: dict = None,
                                       **kwargs):
    """
    Convert a token transformer model from a Hugging Face checkpoint to a PyTorch model weights.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    prefix = "structure_module.score_model.token_transformer."
    module_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith(prefix):
            module_state_dict[k.replace(prefix, "")] = v
    tbnm_state_dict = {}
    dim = module_state_dict[f"layers.0.adaln.s_bias.weight"].shape[0]
    dtype = module_state_dict[f"layers.0.adaln.s_bias.weight"].dtype

    for i in range(config.num_blocks):
        # weight for adaln
        tbnm_state_dict[f"layers.{i}.adaln.a_norm"] = [{
            "weight":
            torch.ones([dim], dtype=dtype)
        }]
        tbnm_state_dict[f"layers.{i}.adaln.s_norm"] = [{
            "weight":
            module_state_dict[f"layers.{i}.adaln.s_norm.weight"]
        }]
        tbnm_state_dict[f"layers.{i}.adaln.fused_s_scale_s_bias"] = [{
            "weight":
            module_state_dict[f"layers.{i}.adaln.s_scale.weight"],
            "bias":
            module_state_dict[f"layers.{i}.adaln.s_scale.bias"]
        }, {
            "weight":
            module_state_dict[f"layers.{i}.adaln.s_bias.weight"],
            "bias":
            torch.zeros([dim], dtype=dtype)
        }]

        # weight for pairwise attention
        tbnm_state_dict[f"layers.{i}.pair_bias_attn.proj_q"] = [{
            'weight':
            module_state_dict[f"layers.{i}.pair_bias_attn.proj_q.weight"],
            'bias':
            module_state_dict[f"layers.{i}.pair_bias_attn.proj_q.bias"]
        }]
        tbnm_state_dict[f"layers.{i}.pair_bias_attn.proj_kv"] = [{
            'weight':
            module_state_dict[f"layers.{i}.pair_bias_attn.proj_k.weight"],
        }, {
            'weight':
            module_state_dict[f"layers.{i}.pair_bias_attn.proj_v.weight"],
        }]
        tbnm_state_dict[f"layers.{i}.pair_bias_attn.proj_g"] = [{
            'weight':
            module_state_dict[f"layers.{i}.pair_bias_attn.proj_g.weight"],
        }]

        if f"layers.{i}.pair_bias_attn.proj_z.0.weight" in module_state_dict:  # v2
            tbnm_state_dict[f"layers.{i}.pair_bias_attn.proj_z.0"] = [{
                'weight':
                module_state_dict[f"layers.{i}.pair_bias_attn.proj_z.0.weight"],
                'bias':
                module_state_dict[f"layers.{i}.pair_bias_attn.proj_z.0.bias"]
            }]
            tbnm_state_dict[f"layers.{i}.pair_bias_attn.proj_z.1"] = [{
                'weight':
                module_state_dict[f"layers.{i}.pair_bias_attn.proj_z.1.weight"],
            }]
        tbnm_state_dict[f"layers.{i}.pair_bias_attn.proj_o"] = [{
            'weight':
            module_state_dict[f"layers.{i}.pair_bias_attn.proj_o.weight"]
        }]

        # weight for output_projection
        tbnm_state_dict[f"layers.{i}.output_projection"] = [{
            'weight':
            module_state_dict[f"layers.{i}.output_projection.0.weight"],
            'bias':
            module_state_dict[f"layers.{i}.output_projection.0.bias"]
        }]

        # weight for conditioned transition block
        tbnm_state_dict[f"layers.{i}.transition.adaln.a_norm"] = [{
            "weight":
            torch.ones([dim], dtype=dtype)
        }]
        tbnm_state_dict[f"layers.{i}.transition.adaln.s_norm"] = [{
            "weight":
            module_state_dict[f"layers.{i}.transition.adaln.s_norm.weight"]
        }]
        tbnm_state_dict[f"layers.{i}.transition.adaln.fused_s_scale_s_bias"] = [
            {
                "weight":
                module_state_dict[
                    f"layers.{i}.transition.adaln.s_scale.weight"],
                "bias":
                module_state_dict[f"layers.{i}.transition.adaln.s_scale.bias"]
            }, {
                "weight":
                module_state_dict[f"layers.{i}.transition.adaln.s_bias.weight"],
                "bias":
                torch.zeros([dim], dtype=dtype)
            }
        ]
        swish_gate_weight = module_state_dict[
            f"layers.{i}.transition.swish_gate.0.weight"]
        swish_gate_weight_0, swish_gate_weight_1 = torch.chunk(
            swish_gate_weight, 2, dim=0)
        tbnm_state_dict[f"layers.{i}.transition.fused_swl_a_to_b"] = [{
            "weight":
            swish_gate_weight_0,
        }, {
            "weight":
            swish_gate_weight_1,
        }, {
            "weight":
            module_state_dict[f"layers.{i}.transition.a_to_b.weight"],
        }]
        tbnm_state_dict[f"layers.{i}.transition.b_to_a"] = [{
            "weight":
            module_state_dict[f"layers.{i}.transition.b_to_a.weight"],
        }]
        tbnm_state_dict[f"layers.{i}.transition.output_projection"] = [{
            "weight":
            module_state_dict[
                f"layers.{i}.transition.output_projection.0.weight"],
            "bias":
            module_state_dict[f"layers.{i}.transition.output_projection.0.bias"]
        }]

    return tbnm_state_dict


def convert_hf_msa_module_torch(config: MSAModuleConfig,
                                mapping: Mapping = None,
                                local_checkpoint: str = None,
                                model_name: str = "boltz-1",
                                weights: dict = None,
                                **kwargs):
    """
    Convert a msa module model from a Hugging Face checkpoint to a PyTorch model weights.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights

    prefix = "msa_module."
    module_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith(prefix):
            module_state_dict[k.replace(prefix, "")] = v

    tbnm_state_dict = {}
    tbnm_state_dict[f"s_proj"] = [{
        "weight": module_state_dict[f"s_proj.weight"],
    }]
    tbnm_state_dict[f"msa_proj"] = [{
        "weight":
        module_state_dict[f"msa_proj.weight"],
    }]

    for i in range(config.msa_blocks):
        # Weight for msa transition
        tbnm_state_dict[f"layers.{i}.msa_transition.norm"] = [{
            'weight':
            module_state_dict[f"layers.{i}.msa_transition.norm.weight"],
            'bias':
            module_state_dict[f"layers.{i}.msa_transition.norm.bias"]
        }]
        tbnm_state_dict[f"layers.{i}.msa_transition.fused_fc2_fc1"] = [{
            'weight':
            module_state_dict[f"layers.{i}.msa_transition.fc2.weight"],
        }, {
            'weight':
            module_state_dict[f"layers.{i}.msa_transition.fc1.weight"],
        }]
        tbnm_state_dict[f"layers.{i}.msa_transition.fc3"] = [{
            'weight':
            module_state_dict[f"layers.{i}.msa_transition.fc3.weight"],
        }]

        # Weight for pair_weighted_averaging
        tbnm_state_dict[f"layers.{i}.pair_weighted_averaging.norm_m"] = [{
            'weight':
            module_state_dict[
                f"layers.{i}.pair_weighted_averaging.norm_m.weight"],
            'bias':
            module_state_dict[f"layers.{i}.pair_weighted_averaging.norm_m.bias"]
        }]
        tbnm_state_dict[f"layers.{i}.pair_weighted_averaging.norm_z"] = [{
            'weight':
            module_state_dict[
                f"layers.{i}.pair_weighted_averaging.norm_z.weight"],
            'bias':
            module_state_dict[f"layers.{i}.pair_weighted_averaging.norm_z.bias"]
        }]
        tbnm_state_dict[
            f"layers.{i}.pair_weighted_averaging.fused_proj_m_g"] = [{
                'weight':
                module_state_dict[
                    f"layers.{i}.pair_weighted_averaging.proj_m.weight"],
            }, {
                'weight':
                module_state_dict[
                    f"layers.{i}.pair_weighted_averaging.proj_g.weight"],
            }]
        tbnm_state_dict[f"layers.{i}.pair_weighted_averaging.proj_z"] = [{
            'weight':
            module_state_dict[
                f"layers.{i}.pair_weighted_averaging.proj_z.weight"]
        }]
        tbnm_state_dict[f"layers.{i}.pair_weighted_averaging.proj_o"] = [{
            'weight':
            module_state_dict[
                f"layers.{i}.pair_weighted_averaging.proj_o.weight"]
        }]

        # Weight for pairformer_layer
        for name in ["tri_mul_out", "tri_mul_in"]:
            tbnm_state_dict[f"layers.{i}.pairformer_layer.{name}.norm_in"] = [{
                'weight':
                module_state_dict[f"layers.{i}.{name}.norm_in.weight"],
                'bias':
                module_state_dict[f"layers.{i}.{name}.norm_in.bias"]
            }]

            w = module_state_dict[f"layers.{i}.{name}.p_in.weight"]
            p_in_0_weight, p_in_1_weight = w.chunk(2, dim=0)
            tbnm_state_dict[f"layers.{i}.pairformer_layer.{name}.p_in"] = [{
                'weight':
                p_in_0_weight,
            }, {
                'weight':
                p_in_1_weight,
            }]

            w = module_state_dict[f"layers.{i}.{name}.g_in.weight"]
            g_in_0_weight, g_in_1_weight = w.chunk(2, dim=0)
            tbnm_state_dict[f"layers.{i}.pairformer_layer.{name}.g_in"] = [{
                'weight':
                g_in_0_weight,
            }, {
                'weight':
                g_in_1_weight,
            }]
            tbnm_state_dict[f"layers.{i}.pairformer_layer.{name}.norm_out"] = [{
                'weight':
                module_state_dict[f"layers.{i}.{name}.norm_out.weight"],
                'bias':
                module_state_dict[f"layers.{i}.{name}.norm_out.bias"]
            }]
            tbnm_state_dict[f"layers.{i}.pairformer_layer.{name}.p_out"] = [{
                'weight':
                module_state_dict[f"layers.{i}.{name}.p_out.weight"],
            }]
            tbnm_state_dict[f"layers.{i}.pairformer_layer.{name}.g_out"] = [{
                'weight':
                module_state_dict[f"layers.{i}.{name}.g_out.weight"],
            }]

        # weight for tri_attn_start and tri_attn_end
        for name in ["start", "end"]:
            tbnm_state_dict[
                f"layers.{i}.pairformer_layer.tri_attn_{name}.layer_norm"] = [{
                    'weight':
                    module_state_dict[
                        f"layers.{i}.tri_att_{name}.layer_norm.weight"],
                    'bias':
                    module_state_dict[
                        f"layers.{i}.tri_att_{name}.layer_norm.bias"]
                }]
            tbnm_state_dict[
                f"layers.{i}.pairformer_layer.tri_attn_{name}.linear"] = [{
                    'weight':
                    module_state_dict[
                        f"layers.{i}.tri_att_{name}.linear.weight"],
                }]
            tbnm_state_dict[
                f"layers.{i}.pairformer_layer.tri_attn_{name}.mha.qkv_proj"] = [
                    {
                        'weight':
                        module_state_dict[
                            f"layers.{i}.tri_att_{name}.mha.linear_q.weight"],
                    }, {
                        'weight':
                        module_state_dict[
                            f"layers.{i}.tri_att_{name}.mha.linear_k.weight"],
                    }, {
                        'weight':
                        module_state_dict[
                            f"layers.{i}.tri_att_{name}.mha.linear_v.weight"],
                    }
                ]
            tbnm_state_dict[
                f"layers.{i}.pairformer_layer.tri_attn_{name}.mha.o_proj"] = [{
                    'weight':
                    module_state_dict[
                        f"layers.{i}.tri_att_{name}.mha.linear_o.weight"],
                }]
            tbnm_state_dict[
                f"layers.{i}.pairformer_layer.tri_attn_{name}.mha.g_proj"] = [{
                    'weight':
                    module_state_dict[
                        f"layers.{i}.tri_att_{name}.mha.linear_g.weight"],
                }]
        # weight for transition_z
        tbnm_state_dict[f"layers.{i}.pairformer_layer.transition_z.norm"] = [{
            'weight':
            module_state_dict[f"layers.{i}.z_transition.norm.weight"],
            'bias':
            module_state_dict[f"layers.{i}.z_transition.norm.bias"]
        }]
        tbnm_state_dict[
            f"layers.{i}.pairformer_layer.transition_z.fused_fc2_fc1"] = [{
                'weight':
                module_state_dict[f"layers.{i}.z_transition.fc2.weight"],
            }, {
                'weight':
                module_state_dict[f"layers.{i}.z_transition.fc1.weight"],
            }]
        tbnm_state_dict[f"layers.{i}.pairformer_layer.transition_z.fc3"] = [{
            'weight':
            module_state_dict[f"layers.{i}.z_transition.fc3.weight"],
        }]

        # weight for outer_product_mean
        tbnm_state_dict[f"layers.{i}.outer_product_mean.norm"] = [{
            'weight':
            module_state_dict[f"layers.{i}.outer_product_mean.norm.weight"],
            'bias':
            module_state_dict[f"layers.{i}.outer_product_mean.norm.bias"]
        }]
        tbnm_state_dict[f"layers.{i}.outer_product_mean.fused_proj_a_b"] = [{
            'weight':
            module_state_dict[f"layers.{i}.outer_product_mean.proj_a.weight"],
        }, {
            'weight':
            module_state_dict[f"layers.{i}.outer_product_mean.proj_b.weight"],
        }]
        tbnm_state_dict[f"layers.{i}.outer_product_mean.proj_o"] = [{
            'weight':
            module_state_dict[f"layers.{i}.outer_product_mean.proj_o.weight"],
            'bias':
            module_state_dict[f"layers.{i}.outer_product_mean.proj_o.bias"]
        }]
    return tbnm_state_dict
