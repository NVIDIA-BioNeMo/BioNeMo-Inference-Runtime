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
from tensorrt_llm_lite._utils import str_dtype_to_torch
from tensorrt_llm_lite.logger import logger

from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.hubs import load_weights
from tensorrt_bionemo.mapping import Mapping


def get_linear_weights(state_dict: dict,
                       prefix: str,
                       tbm_prefix: str,
                       dtype: str = "float32"):
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}
    weight = state_dict[f"{prefix}.weight"]
    bias = state_dict[f"{prefix}.bias"]
    ret[f"{tbm_prefix}.weight"] = weight.to(torch_dtype)
    ret[f"{tbm_prefix}.bias"] = bias.to(torch_dtype)
    return ret


def get_outer_product_mean_weights(state_dict: dict,
                                   prefix: str,
                                   tbm_prefix: str,
                                   dtype: str = "float32",
                                   mapping: Mapping = None):
    mapping if mapping else Mapping()
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}

    norm_weight = state_dict[f"{prefix}.layer_norm.weight"]
    norm_bias = state_dict[f"{prefix}.layer_norm.bias"]

    proj_a_weight = state_dict[f"{prefix}.linear_1.weight"]
    proj_a_bias = state_dict.get(f"{prefix}.linear_1.bias", None)

    proj_b_weight = state_dict[f"{prefix}.linear_2.weight"]
    proj_b_bias = state_dict.get(f"{prefix}.linear_2.bias", None)

    proj_o_weight = state_dict[f"{prefix}.linear_out.weight"]
    proj_o_bias = state_dict.get(f"{prefix}.linear_out.bias", None)

    fused_proj_a_b_weight = torch.cat([proj_a_weight, proj_b_weight], dim=0)
    fused_proj_a_b_bias = torch.cat([proj_a_bias, proj_b_bias], dim=0)

    ret[f"{tbm_prefix}.norm.weight"] = norm_weight.to(torch_dtype)
    ret[f"{tbm_prefix}.norm.bias"] = norm_bias.to(torch_dtype)

    ret[f"{tbm_prefix}.fused_proj_a_b.weight"] = fused_proj_a_b_weight.to(
        torch_dtype)
    if proj_a_bias is not None and proj_b_bias is not None:
        fused_proj_a_b_bias = torch.cat([proj_a_bias, proj_b_bias], dim=0)
        ret[f"{tbm_prefix}.fused_proj_a_b.bias"] = fused_proj_a_b_bias.to(
            torch_dtype)

    ret[f"{tbm_prefix}.proj_o.weight"] = proj_o_weight.to(torch_dtype)
    if proj_o_bias is not None:
        ret[f"{tbm_prefix}.proj_o.bias"] = proj_o_bias.to(torch_dtype)

    return ret


def get_msa_attention_weights(state_dict: dict,
                              prefix: str,
                              tbm_prefix: str,
                              dtype: str = "float32",
                              pair_bias: bool = True,
                              mapping: Mapping = None):
    mapping if mapping else Mapping()
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}

    layer_norm_m_weight = state_dict[f"{prefix}.layer_norm_m.weight"]
    layer_norm_m_bias = state_dict[f"{prefix}.layer_norm_m.bias"]

    if pair_bias:
        layer_norm_z_weight = state_dict[f"{prefix}.layer_norm_z.weight"]
        layer_norm_z_bias = state_dict[f"{prefix}.layer_norm_z.bias"]
        linear_z_weight = state_dict[f"{prefix}.linear_z.weight"]
    else:
        layer_norm_z_weight = None
        layer_norm_z_bias = None
        linear_z_weight = None

    mha_q_weight = state_dict[f"{prefix}.mha.linear_q.weight"]
    mha_k_weight = state_dict[f"{prefix}.mha.linear_k.weight"]
    mha_v_weight = state_dict[f"{prefix}.mha.linear_v.weight"]
    mha_o_weight = state_dict[f"{prefix}.mha.linear_o.weight"]
    mha_o_bias = state_dict.get(f"{prefix}.mha.linear_o.bias", None)
    mha_g_weight = state_dict[f"{prefix}.mha.linear_g.weight"]
    mha_g_bias = state_dict.get(f"{prefix}.mha.linear_g.bias", None)

    mha_qkv_weights = torch.cat([mha_q_weight, mha_k_weight, mha_v_weight],
                                dim=0)

    ret[f"{tbm_prefix}.layer_norm_m.weight"] = layer_norm_m_weight.to(
        torch_dtype)
    ret[f"{tbm_prefix}.layer_norm_m.bias"] = layer_norm_m_bias.to(torch_dtype)

    if pair_bias:
        ret[f"{tbm_prefix}.proj_z_norm.weight"] = layer_norm_z_weight.to(
            torch_dtype)
        ret[f"{tbm_prefix}.proj_z_norm.bias"] = layer_norm_z_bias.to(
            torch_dtype)
        ret[f"{tbm_prefix}.proj_z.weight"] = linear_z_weight.to(torch_dtype)

    ret[f"{tbm_prefix}.mha.qkv_proj.weight"] = mha_qkv_weights.to(torch_dtype)
    ret[f"{tbm_prefix}.mha.o_proj.weight"] = mha_o_weight.to(torch_dtype)
    if mha_o_bias is not None:
        ret[f"{tbm_prefix}.mha.o_proj.bias"] = mha_o_bias.to(torch_dtype)
    ret[f"{tbm_prefix}.mha.g_proj.weight"] = mha_g_weight.to(torch_dtype)
    if mha_g_bias is not None:
        ret[f"{tbm_prefix}.mha.g_proj.bias"] = mha_g_bias.to(torch_dtype)

    return ret


def get_msa_transition_weights(state_dict: dict,
                               prefix: str,
                               tbm_prefix: str,
                               dtype: str = "float32",
                               mapping: Mapping = None):
    mapping if mapping else Mapping()
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}

    layer_norm_weight = state_dict[f"{prefix}.layer_norm.weight"]
    layer_norm_bias = state_dict[f"{prefix}.layer_norm.bias"]

    linear_1_weight = state_dict[f"{prefix}.linear_1.weight"]
    linear_1_bias = state_dict.get(f"{prefix}.linear_1.bias", None)

    linear_2_weight = state_dict[f"{prefix}.linear_2.weight"]
    linear_2_bias = state_dict.get(f"{prefix}.linear_2.bias", None)

    ret[f"{tbm_prefix}.layer_norm.weight"] = layer_norm_weight.to(torch_dtype)
    ret[f"{tbm_prefix}.layer_norm.bias"] = layer_norm_bias.to(torch_dtype)

    ret[f"{tbm_prefix}.linear_1.weight"] = linear_1_weight.to(torch_dtype)
    if linear_1_bias is not None:
        ret[f"{tbm_prefix}.linear_1.bias"] = linear_1_bias.to(torch_dtype)

    ret[f"{tbm_prefix}.linear_2.weight"] = linear_2_weight.to(torch_dtype)
    if linear_2_bias is not None:
        ret[f"{tbm_prefix}.linear_2.bias"] = linear_2_bias.to(torch_dtype)

    return ret


def get_tri_mul_node_weights(state_dict: dict,
                             prefix: str,
                             tbm_prefix: str,
                             dtype: str = "float32",
                             mapping: Mapping = None):
    # TODO: add support for max_tri_mul_tp_size
    mapping if mapping else Mapping()
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}

    norm_in_weight = state_dict[f"{prefix}.layer_norm_in.weight"]
    norm_in_bias = state_dict[f"{prefix}.layer_norm_in.bias"]
    norm_out_weight = state_dict[f"{prefix}.layer_norm_out.weight"]
    norm_out_bias = state_dict[f"{prefix}.layer_norm_out.bias"]

    is_fused = f"{prefix}.linear_ab_p.weight" in state_dict
    if not is_fused:
        p_in_0_weight = state_dict[f"{prefix}.linear_a_p.weight"]
        p_in_0_bias = state_dict[f"{prefix}.linear_a_p.bias"]
        p_in_1_weight = state_dict[f"{prefix}.linear_b_p.weight"]
        p_in_1_bias = state_dict[f"{prefix}.linear_b_p.bias"]
        g_in_0_weight = state_dict[f"{prefix}.linear_a_g.weight"]
        g_in_0_bias = state_dict[f"{prefix}.linear_a_g.bias"]
        g_in_1_weight = state_dict[f"{prefix}.linear_b_g.weight"]
        g_in_1_bias = state_dict[f"{prefix}.linear_b_g.bias"]
    else:
        linear_ab_p_weight = state_dict[f"{prefix}.linear_ab_p.weight"]
        linear_ab_p_bias = state_dict[f"{prefix}.linear_ab_p.bias"]
        p_in_0_weight, p_in_1_weight = linear_ab_p_weight.chunk(2, dim=0)
        p_in_0_bias, p_in_1_bias = linear_ab_p_bias.chunk(2, dim=0)
        linear_ab_g_weight = state_dict[f"{prefix}.linear_ab_g.weight"]
        linear_ab_g_bias = state_dict[f"{prefix}.linear_ab_g.bias"]
        g_in_0_weight, g_in_1_weight = linear_ab_g_weight.chunk(2, dim=0)
        g_in_0_bias, g_in_1_bias = linear_ab_g_bias.chunk(2, dim=0)

    p_out_weight = state_dict[f"{prefix}.linear_z.weight"]
    p_out_bias = state_dict[f"{prefix}.linear_z.bias"]
    g_out_weight = state_dict[f"{prefix}.linear_g.weight"]
    g_out_bias = state_dict[f"{prefix}.linear_g.bias"]

    p_in_weight = torch.cat([p_in_0_weight, p_in_1_weight], dim=0).contiguous()
    p_in_bias = torch.cat([p_in_0_bias, p_in_1_bias], dim=0).contiguous()
    g_in_weight = torch.cat([g_in_0_weight, g_in_1_weight], dim=0).contiguous()
    g_in_bias = torch.cat([g_in_0_bias, g_in_1_bias], dim=0).contiguous()

    ret = {
        f"{tbm_prefix}.norm_in.weight": norm_in_weight.to(torch_dtype),
        f"{tbm_prefix}.norm_in.bias": norm_in_bias.to(torch_dtype),
        f"{tbm_prefix}.p_in.weight": p_in_weight.to(torch_dtype),
        f"{tbm_prefix}.p_in.bias": p_in_bias.to(torch_dtype),
        f"{tbm_prefix}.g_in.weight": g_in_weight.to(torch_dtype),
        f"{tbm_prefix}.g_in.bias": g_in_bias.to(torch_dtype),
        f"{tbm_prefix}.norm_out.weight": norm_out_weight,
        f"{tbm_prefix}.norm_out.bias": norm_out_bias,
        f"{tbm_prefix}.p_out.weight": p_out_weight,
        f"{tbm_prefix}.p_out.bias": p_out_bias.to(torch_dtype),
        f"{tbm_prefix}.g_out.weight": g_out_weight,
        f"{tbm_prefix}.g_out.bias": g_out_bias.to(torch_dtype),
    }
    return ret


def get_tri_attn_node_weights(state_dict: dict,
                              prefix: str,
                              tbm_prefix: str,
                              mapping: Mapping = None,
                              dtype: str = "float32"):
    mapping if mapping else Mapping()
    torch_dtype = str_dtype_to_torch(dtype)
    layer_norm_weight = state_dict[f"{prefix}.layer_norm.weight"]
    layer_norm_bias = state_dict[f"{prefix}.layer_norm.bias"]
    linear_weight = state_dict[f"{prefix}.linear.weight"]

    mha_q_weight = state_dict[f"{prefix}.mha.linear_q.weight"]
    mha_k_weight = state_dict[f"{prefix}.mha.linear_k.weight"]
    mha_v_weight = state_dict[f"{prefix}.mha.linear_v.weight"]
    mha_o_weight = state_dict[f"{prefix}.mha.linear_o.weight"]
    mha_o_bias = state_dict[f"{prefix}.mha.linear_o.bias"]
    mha_g_weight = state_dict[f"{prefix}.mha.linear_g.weight"]
    mha_g_bias = state_dict[f"{prefix}.mha.linear_g.bias"]

    mha_qkv_weights = torch.cat([mha_q_weight, mha_k_weight, mha_v_weight],
                                dim=0)

    ret = {
        f"{tbm_prefix}.layer_norm.weight": layer_norm_weight.to(torch_dtype),
        f"{tbm_prefix}.layer_norm.bias": layer_norm_bias.to(torch_dtype),
        f"{tbm_prefix}.linear.weight": linear_weight.to(torch_dtype),
        f"{tbm_prefix}.mha.qkv_proj.weight": mha_qkv_weights.to(torch_dtype),
        f"{tbm_prefix}.mha.o_proj.weight": mha_o_weight.to(torch_dtype),
        f"{tbm_prefix}.mha.o_proj.bias": mha_o_bias.to(torch_dtype),
        f"{tbm_prefix}.mha.g_proj.weight": mha_g_weight.to(torch_dtype),
        f"{tbm_prefix}.mha.g_proj.bias": mha_g_bias.to(torch_dtype),
    }
    return ret


def get_pair_transition_weights(state_dict: dict,
                                prefix: str,
                                tbm_prefix: str,
                                dtype: str = "float32",
                                mapping: Mapping = None):
    mapping if mapping else Mapping()
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}

    layer_norm_weight = state_dict[f"{prefix}.layer_norm.weight"]
    layer_norm_bias = state_dict[f"{prefix}.layer_norm.bias"]

    linear_1_weight = state_dict[f"{prefix}.linear_1.weight"]
    linear_1_bias = state_dict.get(f"{prefix}.linear_1.bias", None)

    linear_2_weight = state_dict[f"{prefix}.linear_2.weight"]
    linear_2_bias = state_dict.get(f"{prefix}.linear_2.bias", None)

    ret[f"{tbm_prefix}.layer_norm.weight"] = layer_norm_weight.to(torch_dtype)
    ret[f"{tbm_prefix}.layer_norm.bias"] = layer_norm_bias.to(torch_dtype)

    ret[f"{tbm_prefix}.linear_1.weight"] = linear_1_weight.to(torch_dtype)
    if linear_1_bias is not None:
        ret[f"{tbm_prefix}.linear_1.bias"] = linear_1_bias.to(torch_dtype)

    ret[f"{tbm_prefix}.linear_2.weight"] = linear_2_weight.to(torch_dtype)
    if linear_2_bias is not None:
        ret[f"{tbm_prefix}.linear_2.bias"] = linear_2_bias.to(torch_dtype)

    return ret


def convert_hf_evoformer(config: BaseConfig,
                         mapping: Mapping = None,
                         local_checkpoint: str = None,
                         model_name: str = "openfold2_ptm_1"):
    """
    Convert a pairformer model from a Hugging Face checkpoint to a TensorRT model weights.
    """
    mapping = mapping if mapping is not None else Mapping()
    prefix = "evoformer"

    state_dict = load_weights(name=model_name, cache_path=local_checkpoint)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k.replace("core.", "").replace("pair_stack.",
                                                          "")] = v
    state_dict = new_state_dict
    weights = {}
    weights.update(
        get_linear_weights(state_dict, f"{prefix}.linear", f"linear"))

    logger.info(
        f"Loading weights for evoformer, model_name: {model_name}, dtype: {config.dtype}, num_blocks: {config.no_blocks}"
    )

    for i in range(config.no_blocks):
        weights.update(
            get_outer_product_mean_weights(
                state_dict,
                f"{prefix}.blocks.{i}.outer_product_mean",
                f"blocks.{i}.outer_product_mean",
                dtype=config.dtype,
                mapping=mapping))
        weights.update(
            get_msa_attention_weights(state_dict,
                                      f"{prefix}.blocks.{i}.msa_att_row",
                                      f"blocks.{i}.msa_att_row",
                                      dtype=config.dtype,
                                      mapping=mapping))
        weights.update(
            get_msa_attention_weights(
                state_dict,
                f"{prefix}.blocks.{i}.msa_att_col._msa_att",
                f"blocks.{i}.msa_att_col",
                dtype=config.dtype,
                pair_bias=False,
                mapping=mapping))
        weights.update(
            get_msa_transition_weights(state_dict,
                                       f"{prefix}.blocks.{i}.msa_transition",
                                       f"blocks.{i}.msa_transition",
                                       dtype=config.dtype,
                                       mapping=mapping))
        weights.update(
            get_pair_transition_weights(state_dict,
                                        f"{prefix}.blocks.{i}.pair_transition",
                                        f"blocks.{i}.pair_transition",
                                        dtype=config.dtype,
                                        mapping=mapping))
        weights.update(
            get_tri_mul_node_weights(state_dict,
                                     f"{prefix}.blocks.{i}.tri_mul_in",
                                     f"blocks.{i}.tri_mul_in",
                                     dtype=config.dtype,
                                     mapping=mapping))
        weights.update(
            get_tri_mul_node_weights(state_dict,
                                     f"{prefix}.blocks.{i}.tri_mul_out",
                                     f"blocks.{i}.tri_mul_out",
                                     dtype=config.dtype,
                                     mapping=mapping))
        weights.update(
            get_tri_attn_node_weights(state_dict,
                                      f"{prefix}.blocks.{i}.tri_att_start",
                                      f"blocks.{i}.tri_attn_start",
                                      dtype=config.dtype,
                                      mapping=mapping))
        weights.update(
            get_tri_attn_node_weights(state_dict,
                                      f"{prefix}.blocks.{i}.tri_att_end",
                                      f"blocks.{i}.tri_attn_end",
                                      dtype=config.dtype,
                                      mapping=mapping))

    return weights


def get_trimul_torch_weights(state_dict: dict,
                             prefix: str,
                             tbm_prefix: str,
                             dtype: str = "float32",
                             mapping: Mapping = None):
    mapping if mapping else Mapping()
    str_dtype_to_torch(dtype)

    is_fused = f"{prefix}.linear_ab_p.weight" in state_dict
    if not is_fused:
        p_in_0_weight = state_dict[f"{prefix}.linear_a_p.weight"]
        p_in_0_bias = state_dict[f"{prefix}.linear_a_p.bias"]
        p_in_1_weight = state_dict[f"{prefix}.linear_b_p.weight"]
        p_in_1_bias = state_dict[f"{prefix}.linear_b_p.bias"]
        g_in_0_weight = state_dict[f"{prefix}.linear_a_g.weight"]
        g_in_0_bias = state_dict[f"{prefix}.linear_a_g.bias"]
        g_in_1_weight = state_dict[f"{prefix}.linear_b_g.weight"]
        g_in_1_bias = state_dict[f"{prefix}.linear_b_g.bias"]
    else:
        linear_ab_p_weight = state_dict[f"{prefix}.linear_ab_p.weight"]
        linear_ab_p_bias = state_dict[f"{prefix}.linear_ab_p.bias"]
        p_in_0_weight, p_in_1_weight = linear_ab_p_weight.chunk(2, dim=0)
        p_in_0_bias, p_in_1_bias = linear_ab_p_bias.chunk(2, dim=0)
        linear_ab_g_weight = state_dict[f"{prefix}.linear_ab_g.weight"]
        linear_ab_g_bias = state_dict[f"{prefix}.linear_ab_g.bias"]
        g_in_0_weight, g_in_1_weight = linear_ab_g_weight.chunk(2, dim=0)
        g_in_0_bias, g_in_1_bias = linear_ab_g_bias.chunk(2, dim=0)

    ret = {}
    ret[f"{tbm_prefix}.norm_in"] = [{
        "weight":
        state_dict[f"{prefix}.layer_norm_in.weight"],
        "bias":
        state_dict[f"{prefix}.layer_norm_in.bias"],
    }]

    ret[f"{tbm_prefix}.p_in"] = [{
        "weight": p_in_0_weight,
        "bias": p_in_0_bias,
    }, {
        "weight": p_in_1_weight,
        "bias": p_in_1_bias,
    }]

    ret[f"{tbm_prefix}.g_in"] = [{
        "weight": g_in_0_weight,
        "bias": g_in_0_bias,
    }, {
        "weight": g_in_1_weight,
        "bias": g_in_1_bias,
    }]

    ret[f"{tbm_prefix}.p_out"] = [{
        "weight":
        state_dict[f"{prefix}.linear_z.weight"],
        "bias":
        state_dict[f"{prefix}.linear_z.bias"],
    }]
    ret[f"{tbm_prefix}.g_out"] = [{
        "weight":
        state_dict[f"{prefix}.linear_g.weight"],
        "bias":
        state_dict[f"{prefix}.linear_g.bias"],
    }]

    ret[f"{tbm_prefix}.norm_out"] = [{
        "weight":
        state_dict[f"{prefix}.layer_norm_out.weight"],
        "bias":
        state_dict[f"{prefix}.layer_norm_out.bias"],
    }]

    return ret


def get_triattn_torch_weights(state_dict: dict,
                              prefix: str,
                              tbm_prefix: str,
                              dtype: str = "float32",
                              mapping: Mapping = None):
    mapping if mapping else Mapping()
    str_dtype_to_torch(dtype)
    ret = {}
    ret[f"{tbm_prefix}.layer_norm"] = [{
        "weight":
        state_dict[f"{prefix}.layer_norm.weight"],
        "bias":
        state_dict[f"{prefix}.layer_norm.bias"],
    }]
    ret[f"{tbm_prefix}.linear"] = [{
        "weight":
        state_dict[f"{prefix}.linear.weight"],
        "bias":
        None
    }]
    ret[f"{tbm_prefix}.mha.qkv_proj"] = [{
        "weight":
        state_dict[f"{prefix}.mha.linear_q.weight"],
        "bias":
        None
    }, {
        "weight":
        state_dict[f"{prefix}.mha.linear_k.weight"],
        "bias":
        None
    }, {
        "weight":
        state_dict[f"{prefix}.mha.linear_v.weight"],
        "bias":
        None
    }]
    ret[f"{tbm_prefix}.mha.o_proj"] = [{
        "weight":
        state_dict[f"{prefix}.mha.linear_o.weight"],
        "bias":
        state_dict[f"{prefix}.mha.linear_o.bias"],
    }]
    ret[f"{tbm_prefix}.mha.g_proj"] = [{
        "weight":
        state_dict[f"{prefix}.mha.linear_g.weight"],
        "bias":
        state_dict[f"{prefix}.mha.linear_g.bias"],
    }]
    return ret


def convert_hf_evoformer_torch(config: BaseConfig,
                               mapping: Mapping = None,
                               local_checkpoint: str = None,
                               model_name: str = "openfold2_ptm_1",
                               weights: dict = None):
    """
    Convert a pairformer model from a Hugging Face checkpoint to a PyTorch model weights.
    Args:
        module: The module to load the weights into.
        checkpoint_dir: The directory to load the checkpoint from.
        world_size: The number of processes to use.
        rank: The rank of the process.
        weights: The weights to load into the module.
        model_name: The name of the model to load the weights from.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights

    prefix = "evoformer."

    module_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith(prefix):
            module_state_dict[k.replace(prefix,
                                        "").replace("core.", "").replace(
                                            "pair_stack.", "")] = v
    tbnm_state_dict = {}
    tbnm_state_dict[f"linear"] = [{
        "weight": module_state_dict[f"linear.weight"],
        "bias": module_state_dict[f"linear.bias"],
    }]

    for i in range(config.no_blocks):
        # Update weights for msa_att_row
        tbnm_state_dict[f"blocks.{i}.msa_att_row.layer_norm_m"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_att_row.layer_norm_m.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.msa_att_row.layer_norm_m.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.msa_att_row.proj_z_norm"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_att_row.layer_norm_z.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.msa_att_row.layer_norm_z.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.msa_att_row.proj_z"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_att_row.linear_z.weight"],
            "bias":
            None
        }]
        tbnm_state_dict[f"blocks.{i}.msa_att_row.mha.qkv_proj"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_att_row.mha.linear_q.weight"],
            "bias":
            None
        }, {
            "weight":
            module_state_dict[f"blocks.{i}.msa_att_row.mha.linear_k.weight"],
            "bias":
            None
        }, {
            "weight":
            module_state_dict[f"blocks.{i}.msa_att_row.mha.linear_v.weight"],
            "bias":
            None
        }]
        tbnm_state_dict[f"blocks.{i}.msa_att_row.mha.o_proj"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_att_row.mha.linear_o.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.msa_att_row.mha.linear_o.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.msa_att_row.mha.g_proj"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_att_row.mha.linear_g.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.msa_att_row.mha.linear_g.bias"],
        }]

        # Update weights for msa_att_col
        tbnm_state_dict[f"blocks.{i}.msa_att_col.layer_norm_m"] = [{
            "weight":
            module_state_dict[
                f"blocks.{i}.msa_att_col._msa_att.layer_norm_m.weight"],
            "bias":
            module_state_dict[
                f"blocks.{i}.msa_att_col._msa_att.layer_norm_m.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.msa_att_col.mha.qkv_proj"] = [{
            "weight":
            module_state_dict[
                f"blocks.{i}.msa_att_col._msa_att.mha.linear_q.weight"],
            "bias":
            None
        }, {
            "weight":
            module_state_dict[
                f"blocks.{i}.msa_att_col._msa_att.mha.linear_k.weight"],
            "bias":
            None
        }, {
            "weight":
            module_state_dict[
                f"blocks.{i}.msa_att_col._msa_att.mha.linear_v.weight"],
            "bias":
            None
        }]
        tbnm_state_dict[f"blocks.{i}.msa_att_col.mha.o_proj"] = [{
            "weight":
            module_state_dict[
                f"blocks.{i}.msa_att_col._msa_att.mha.linear_o.weight"],
            "bias":
            module_state_dict[
                f"blocks.{i}.msa_att_col._msa_att.mha.linear_o.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.msa_att_col.mha.g_proj"] = [{
            "weight":
            module_state_dict[
                f"blocks.{i}.msa_att_col._msa_att.mha.linear_g.weight"],
            "bias":
            module_state_dict[
                f"blocks.{i}.msa_att_col._msa_att.mha.linear_g.bias"],
        }]

        # Update weights for msa_transition
        tbnm_state_dict[f"blocks.{i}.msa_transition.layer_norm"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_transition.layer_norm.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.msa_transition.layer_norm.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.msa_transition.linear_1"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_transition.linear_1.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.msa_transition.linear_1.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.msa_transition.linear_2"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_transition.linear_2.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.msa_transition.linear_2.bias"],
        }]

        # Update weights for outer_product_mean
        tbnm_state_dict[f"blocks.{i}.outer_product_mean.norm"] = [{
            "weight":
            module_state_dict[
                f"blocks.{i}.outer_product_mean.layer_norm.weight"],
            "bias":
            module_state_dict[
                f"blocks.{i}.outer_product_mean.layer_norm.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.outer_product_mean.fused_proj_a_b"] = [{
            "weight":
            module_state_dict[
                f"blocks.{i}.outer_product_mean.linear_1.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.outer_product_mean.linear_1.bias"],
        }, {
            "weight":
            module_state_dict[
                f"blocks.{i}.outer_product_mean.linear_2.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.outer_product_mean.linear_2.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.outer_product_mean.proj_o"] = [{
            "weight":
            module_state_dict[
                f"blocks.{i}.outer_product_mean.linear_out.weight"],
            "bias":
            module_state_dict[
                f"blocks.{i}.outer_product_mean.linear_out.bias"],
        }]

        # Update weights for tri_mul_out, tri_mul_in
        for name in ["tri_mul_out", "tri_mul_in"]:
            tbm_prefix = f"blocks.{i}.{name}"
            original_prefix = f"blocks.{i}.{name}"
            tbnm_state_dict.update(
                get_trimul_torch_weights(module_state_dict,
                                         original_prefix,
                                         tbm_prefix,
                                         dtype=config.dtype,
                                         mapping=mapping))

        # weight for tri_attn_start and tri_attn_end
        for name in ["start", "end"]:
            tbm_prefix = f"blocks.{i}.tri_attn_{name}"
            original_prefix = f"blocks.{i}.tri_att_{name}"
            tbnm_state_dict.update(
                get_triattn_torch_weights(module_state_dict,
                                          original_prefix,
                                          tbm_prefix,
                                          dtype=config.dtype,
                                          mapping=mapping))

        # weight for pair_transition
        tbnm_state_dict[f"blocks.{i}.pair_transition.layer_norm"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.pair_transition.layer_norm.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.pair_transition.layer_norm.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.pair_transition.linear_1"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.pair_transition.linear_1.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.pair_transition.linear_1.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.pair_transition.linear_2"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.pair_transition.linear_2.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.pair_transition.linear_2.bias"],
        }]
    return tbnm_state_dict


def convert_hf_extra_msa_stack_torch(config: BaseConfig,
                                     mapping: Mapping = None,
                                     local_checkpoint: str = None,
                                     model_name: str = "openfold2_ptm_1",
                                     weights: dict = None):
    """
    Convert a extra msa stack model from a Hugging Face checkpoint to a PyTorch model weights.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights

    prefix = "extra_msa_stack."

    module_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith(prefix):
            module_state_dict[k.replace(prefix,
                                        "").replace("core.", "").replace(
                                            "pair_stack.", "")] = v
    tbnm_state_dict = {}
    for i in range(config.no_blocks):
        # Update weights for msa_att_row
        tbnm_state_dict[f"blocks.{i}.msa_att_row.layer_norm_m"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_att_row.layer_norm_m.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.msa_att_row.layer_norm_m.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.msa_att_row.proj_z_norm"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_att_row.layer_norm_z.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.msa_att_row.layer_norm_z.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.msa_att_row.proj_z"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_att_row.linear_z.weight"],
            "bias":
            None
        }]
        tbnm_state_dict[f"blocks.{i}.msa_att_row.mha.qkv_proj"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_att_row.mha.linear_q.weight"],
            "bias":
            None
        }, {
            "weight":
            module_state_dict[f"blocks.{i}.msa_att_row.mha.linear_k.weight"],
            "bias":
            None
        }, {
            "weight":
            module_state_dict[f"blocks.{i}.msa_att_row.mha.linear_v.weight"],
            "bias":
            None
        }]
        tbnm_state_dict[f"blocks.{i}.msa_att_row.mha.o_proj"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_att_row.mha.linear_o.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.msa_att_row.mha.linear_o.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.msa_att_row.mha.g_proj"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_att_row.mha.linear_g.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.msa_att_row.mha.linear_g.bias"],
        }]

        # Update weights for msa_att_col
        tbnm_state_dict[f"blocks.{i}.msa_att_col.layer_norm_m"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_att_col.layer_norm_m.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.msa_att_col.layer_norm_m.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.msa_att_col.global_attention.proj_q"] = [{
            "weight":
            module_state_dict[
                f"blocks.{i}.msa_att_col.global_attention.linear_q.weight"],
            "bias":
            None
        }]
        tbnm_state_dict[
            f"blocks.{i}.msa_att_col.global_attention.fused_proj_kv"] = [{
                "weight":
                module_state_dict[
                    f"blocks.{i}.msa_att_col.global_attention.linear_k.weight"],
                "bias":
                None
            }, {
                "weight":
                module_state_dict[
                    f"blocks.{i}.msa_att_col.global_attention.linear_v.weight"],
                "bias":
                None
            }]
        tbnm_state_dict[f"blocks.{i}.msa_att_col.global_attention.proj_o"] = [{
            "weight":
            module_state_dict[
                f"blocks.{i}.msa_att_col.global_attention.linear_o.weight"],
            "bias":
            module_state_dict[
                f"blocks.{i}.msa_att_col.global_attention.linear_o.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.msa_att_col.global_attention.proj_g"] = [{
            "weight":
            module_state_dict[
                f"blocks.{i}.msa_att_col.global_attention.linear_g.weight"],
            "bias":
            module_state_dict[
                f"blocks.{i}.msa_att_col.global_attention.linear_g.bias"],
        }]

        # Update weights for msa_transition
        tbnm_state_dict[f"blocks.{i}.msa_transition.layer_norm"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_transition.layer_norm.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.msa_transition.layer_norm.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.msa_transition.linear_1"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_transition.linear_1.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.msa_transition.linear_1.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.msa_transition.linear_2"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.msa_transition.linear_2.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.msa_transition.linear_2.bias"],
        }]

        # Update weights for outer_product_mean
        tbnm_state_dict[f"blocks.{i}.outer_product_mean.norm"] = [{
            "weight":
            module_state_dict[
                f"blocks.{i}.outer_product_mean.layer_norm.weight"],
            "bias":
            module_state_dict[
                f"blocks.{i}.outer_product_mean.layer_norm.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.outer_product_mean.fused_proj_a_b"] = [{
            "weight":
            module_state_dict[
                f"blocks.{i}.outer_product_mean.linear_1.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.outer_product_mean.linear_1.bias"],
        }, {
            "weight":
            module_state_dict[
                f"blocks.{i}.outer_product_mean.linear_2.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.outer_product_mean.linear_2.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.outer_product_mean.proj_o"] = [{
            "weight":
            module_state_dict[
                f"blocks.{i}.outer_product_mean.linear_out.weight"],
            "bias":
            module_state_dict[
                f"blocks.{i}.outer_product_mean.linear_out.bias"],
        }]

        # Update weights for tri_mul_out, tri_mul_in
        for name in ["tri_mul_out", "tri_mul_in"]:
            tbm_prefix = f"blocks.{i}.{name}"
            original_prefix = f"blocks.{i}.{name}"
            tbnm_state_dict.update(
                get_trimul_torch_weights(module_state_dict,
                                         original_prefix,
                                         tbm_prefix,
                                         dtype=config.dtype,
                                         mapping=mapping))

        # weight for tri_attn_start and tri_attn_end
        for name in ["start", "end"]:
            tbm_prefix = f"blocks.{i}.tri_attn_{name}"
            original_prefix = f"blocks.{i}.tri_att_{name}"
            tbnm_state_dict.update(
                get_triattn_torch_weights(module_state_dict,
                                          original_prefix,
                                          tbm_prefix,
                                          dtype=config.dtype,
                                          mapping=mapping))

        # weight for pair_transition
        tbnm_state_dict[f"blocks.{i}.pair_transition.layer_norm"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.pair_transition.layer_norm.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.pair_transition.layer_norm.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.pair_transition.linear_1"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.pair_transition.linear_1.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.pair_transition.linear_1.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.pair_transition.linear_2"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.pair_transition.linear_2.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.pair_transition.linear_2.bias"],
        }]
    return tbnm_state_dict


def convert_hf_input_embedder_torch(config: BaseConfig,
                                    mapping: Mapping = None,
                                    local_checkpoint: str = None,
                                    model_name: str = "openfold2_ptm_1",
                                    weights: dict = None):
    """
    Convert a input embedder model from a Hugging Face checkpoint to a PyTorch model weights.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights

    prefix = "input_embedder."

    module_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith(prefix):
            module_state_dict[k.replace(prefix, "")] = v

    tbnm_state_dict = {}
    tbnm_state_dict["fused_linear_tf_z"] = [{
        "weight":
        module_state_dict["linear_tf_z_i.weight"],
        "bias":
        module_state_dict["linear_tf_z_i.bias"],
    }, {
        "weight":
        module_state_dict["linear_tf_z_j.weight"],
        "bias":
        module_state_dict["linear_tf_z_j.bias"],
    }]
    tbnm_state_dict["linear_tf_m"] = [{
        "weight":
        module_state_dict["linear_tf_m.weight"],
        "bias":
        module_state_dict["linear_tf_m.bias"],
    }]
    tbnm_state_dict["linear_msa_m"] = [{
        "weight":
        module_state_dict["linear_msa_m.weight"],
        "bias":
        module_state_dict["linear_msa_m.bias"],
    }]
    tbnm_state_dict["linear_relpos"] = [{
        "weight":
        module_state_dict["linear_relpos.weight"],
        "bias":
        module_state_dict["linear_relpos.bias"],
    }]
    return tbnm_state_dict


def convert_hf_recycling_embedder_torch(config: BaseConfig,
                                        mapping: Mapping = None,
                                        local_checkpoint: str = None,
                                        model_name: str = "openfold2_ptm_1",
                                        weights: dict = None):
    """
    Convert a recycling embedder model from a Hugging Face checkpoint to a PyTorch model weights.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights

    prefix = "recycling_embedder."
    module_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            module_state_dict[k.replace(prefix, "")] = v

    tbnm_state_dict = {}
    tbnm_state_dict["linear"] = [{
        "weight": module_state_dict["linear.weight"],
        "bias": module_state_dict["linear.bias"],
    }]
    tbnm_state_dict["layer_norm_m"] = [{
        "weight":
        module_state_dict["layer_norm_m.weight"],
        "bias":
        module_state_dict["layer_norm_m.bias"],
    }]
    tbnm_state_dict["layer_norm_z"] = [{
        "weight":
        module_state_dict["layer_norm_z.weight"],
        "bias":
        module_state_dict["layer_norm_z.bias"],
    }]
    return tbnm_state_dict


def convert_hf_extra_msa_embedder_torch(config: BaseConfig,
                                        mapping: Mapping = None,
                                        local_checkpoint: str = None,
                                        model_name: str = "openfold2_ptm_1",
                                        weights: dict = None):
    """
    Convert a extra msa embedder model from a Hugging Face checkpoint to a PyTorch model weights.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights

    prefix = "extra_msa_embedder."
    module_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            module_state_dict[k.replace(prefix, "")] = v

    tbnm_state_dict = {}
    tbnm_state_dict["linear"] = [{
        "weight": module_state_dict["linear.weight"],
        "bias": module_state_dict["linear.bias"],
    }]
    return tbnm_state_dict


def convert_hf_template_embedder_torch(config: BaseConfig,
                                       mapping: Mapping = None,
                                       local_checkpoint: str = None,
                                       model_name: str = "openfold2_ptm_1",
                                       weights: dict = None):
    """
    Convert a template embedder model from a Hugging Face checkpoint to a PyTorch model weights.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    module_state_dict = {}
    for k, v in state_dict.items():
        if "template" in k:
            module_state_dict[k.replace("template_embedder.", "")] = v

    tbnm_state_dict = {}
    # Weight for template single embedder
    prefix = "template_angle_embedder"
    if f"{prefix}.linear_1.weight" not in module_state_dict:
        prefix = "template_single_embedder"
    tbnm_state_dict[f"template_single_embedder.linear_1"] = [{
        "weight":
        module_state_dict[f"{prefix}.linear_1.weight"],
        "bias":
        module_state_dict[f"{prefix}.linear_1.bias"],
    }]
    tbnm_state_dict[f"template_single_embedder.linear_2"] = [{
        "weight":
        module_state_dict[f"{prefix}.linear_2.weight"],
        "bias":
        module_state_dict[f"{prefix}.linear_2.bias"],
    }]

    # Weight for template pair embedder
    prefix = "template_pair_embedder"
    tbnm_state_dict[f"{prefix}.linear"] = [{
        "weight":
        module_state_dict[f"{prefix}.linear.weight"],
        "bias":
        module_state_dict[f"{prefix}.linear.bias"],
    }]

    # Weight for template pair stack
    prefix = "template_pair_stack"
    no_blocks = config.template_pair_stack.no_blocks
    for i in range(no_blocks):
        # Update weights for tri_mul_out, tri_mul_in
        # TODO: Refactor this to use a single function for loading trimul and triattn weights
        for name in ["tri_mul_out", "tri_mul_in"]:
            tbm_prefix = f"{prefix}.blocks.{i}.{name}"
            original_prefix = f"{prefix}.blocks.{i}.{name}"
            tbnm_state_dict.update(
                get_trimul_torch_weights(module_state_dict,
                                         original_prefix,
                                         tbm_prefix,
                                         dtype=config.dtype,
                                         mapping=mapping))

        # weight for tri_attn_start and tri_attn_end
        for name in ["start", "end"]:
            tbm_prefix = f"{prefix}.blocks.{i}.tri_attn_{name}"
            original_prefix = f"{prefix}.blocks.{i}.tri_att_{name}"
            tbnm_state_dict.update(
                get_triattn_torch_weights(module_state_dict,
                                          original_prefix,
                                          tbm_prefix,
                                          dtype=config.dtype,
                                          mapping=mapping))

        # weight for pair_transition
        tbnm_state_dict[f"{prefix}.blocks.{i}.pair_transition.layer_norm"] = [{
            "weight":
            module_state_dict[
                f"{prefix}.blocks.{i}.pair_transition.layer_norm.weight"],
            "bias":
            module_state_dict[
                f"{prefix}.blocks.{i}.pair_transition.layer_norm.bias"],
        }]
        tbnm_state_dict[f"{prefix}.blocks.{i}.pair_transition.linear_1"] = [{
            "weight":
            module_state_dict[
                f"{prefix}.blocks.{i}.pair_transition.linear_1.weight"],
            "bias":
            module_state_dict[
                f"{prefix}.blocks.{i}.pair_transition.linear_1.bias"],
        }]
        tbnm_state_dict[f"{prefix}.blocks.{i}.pair_transition.linear_2"] = [{
            "weight":
            module_state_dict[
                f"{prefix}.blocks.{i}.pair_transition.linear_2.weight"],
            "bias":
            module_state_dict[
                f"{prefix}.blocks.{i}.pair_transition.linear_2.bias"],
        }]
    tbnm_state_dict[f"{prefix}.layer_norm"] = [{
        "weight":
        module_state_dict[f"{prefix}.layer_norm.weight"],
        "bias":
        module_state_dict[f"{prefix}.layer_norm.bias"],
    }]

    # Weight for template pointwise attention
    prefix = "template_pointwise_att"
    tbnm_state_dict[f"{prefix}.mha.q_proj"] = [{
        "weight":
        module_state_dict[f"{prefix}.mha.linear_q.weight"],
        "bias":
        None
    }]
    tbnm_state_dict[f"{prefix}.mha.kv_proj"] = [{
        "weight":
        module_state_dict[f"{prefix}.mha.linear_k.weight"],
        "bias":
        None
    }, {
        "weight":
        module_state_dict[f"{prefix}.mha.linear_v.weight"],
        "bias":
        None
    }]
    tbnm_state_dict[f"{prefix}.mha.o_proj"] = [{
        "weight":
        module_state_dict[f"{prefix}.mha.linear_o.weight"],
        "bias":
        module_state_dict[f"{prefix}.mha.linear_o.bias"],
    }]

    return tbnm_state_dict


def convert_hf_template_embedder_multimer_torch(
        config: BaseConfig,
        mapping: Mapping = None,
        local_checkpoint: str = None,
        model_name: str = "alphafold2_multimer_1",
        weights: dict = None):

    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights

    module_state_dict = {}
    for k, v in state_dict.items():
        if "template" in k:
            module_state_dict[k.replace("template_embedder.", "")] = v

    tbnm_state_dict = {}
    # Weight for template single embedder
    prefix = "template_single_embedder"
    tbnm_state_dict[f"template_single_embedder.template_single_embedder"] = [{
        "weight":
        module_state_dict[f"{prefix}.template_single_embedder.weight"],
        "bias":
        module_state_dict[f"{prefix}.template_single_embedder.bias"],
    }]
    tbnm_state_dict[f"template_single_embedder.template_projector"] = [{
        "weight":
        module_state_dict[f"{prefix}.template_projector.weight"],
        "bias":
        module_state_dict[f"{prefix}.template_projector.bias"],
    }]
    # Weight for template pair embedder
    prefix = "template_pair_embedder"
    sub_names = [
        "dgram_linear",
        "aatype_linear_1",
        "aatype_linear_2",
        "query_embedding_layer_norm",
        "query_embedding_linear",
        "pseudo_beta_mask_linear",
        "x_linear",
        "y_linear",
        "z_linear",
        "backbone_mask_linear",
    ]
    for name in sub_names:
        tbnm_state_dict[f"{prefix}.{name}"] = [{
            "weight":
            module_state_dict[f"{prefix}.{name}.weight"],
            "bias":
            module_state_dict[f"{prefix}.{name}.bias"],
        }]
    # Weight for template pair stack
    prefix = "template_pair_stack"
    no_blocks = config.template_pair_stack.no_blocks
    for i in range(no_blocks):
        # Update weights for tri_mul_out, tri_mul_in
        # TODO: Refactor this to use a single function for loading trimul and triattn weights
        for name in ["tri_mul_out", "tri_mul_in"]:
            tbm_prefix = f"{prefix}.blocks.{i}.{name}"
            original_prefix = f"{prefix}.blocks.{i}.{name}"
            tbnm_state_dict.update(
                get_trimul_torch_weights(module_state_dict,
                                         original_prefix,
                                         tbm_prefix,
                                         dtype=config.dtype,
                                         mapping=mapping))

        # weight for tri_attn_start and tri_attn_end
        for name in ["start", "end"]:
            tbm_prefix = f"{prefix}.blocks.{i}.tri_attn_{name}"
            original_prefix = f"{prefix}.blocks.{i}.tri_att_{name}"
            tbnm_state_dict.update(
                get_triattn_torch_weights(module_state_dict,
                                          original_prefix,
                                          tbm_prefix,
                                          dtype=config.dtype,
                                          mapping=mapping))
        # weight for pair_transition
        tbnm_state_dict[f"{prefix}.blocks.{i}.pair_transition.layer_norm"] = [{
            "weight":
            module_state_dict[
                f"{prefix}.blocks.{i}.pair_transition.layer_norm.weight"],
            "bias":
            module_state_dict[
                f"{prefix}.blocks.{i}.pair_transition.layer_norm.bias"],
        }]
        tbnm_state_dict[f"{prefix}.blocks.{i}.pair_transition.linear_1"] = [{
            "weight":
            module_state_dict[
                f"{prefix}.blocks.{i}.pair_transition.linear_1.weight"],
            "bias":
            module_state_dict[
                f"{prefix}.blocks.{i}.pair_transition.linear_1.bias"],
        }]
        tbnm_state_dict[f"{prefix}.blocks.{i}.pair_transition.linear_2"] = [{
            "weight":
            module_state_dict[
                f"{prefix}.blocks.{i}.pair_transition.linear_2.weight"],
            "bias":
            module_state_dict[
                f"{prefix}.blocks.{i}.pair_transition.linear_2.bias"],
        }]
    tbnm_state_dict[f"{prefix}.layer_norm"] = [{
        "weight":
        module_state_dict[f"{prefix}.layer_norm.weight"],
        "bias":
        module_state_dict[f"{prefix}.layer_norm.bias"],
    }]

    tbnm_state_dict[f"linear_t"] = [{
        "weight":
        module_state_dict[f"linear_t.weight"],
        "bias":
        module_state_dict[f"linear_t.bias"],
    }]

    return tbnm_state_dict


def convert_hf_confidence_module_torch(config: BaseConfig,
                                       mapping: Mapping = None,
                                       local_checkpoint: str = None,
                                       model_name: str = "openfold2_ptm_1",
                                       weights: dict = None):
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    prefix = "aux_heads."
    tbnm_state_dict = {}
    tbnm_state_dict["plddt.linear_1"] = [{
        "weight":
        state_dict[f"{prefix}plddt.linear_1.weight"],
        "bias":
        state_dict[f"{prefix}plddt.linear_1.bias"],
    }]
    tbnm_state_dict["plddt.linear_2"] = [{
        "weight":
        state_dict[f"{prefix}plddt.linear_2.weight"],
        "bias":
        state_dict[f"{prefix}plddt.linear_2.bias"],
    }]
    tbnm_state_dict["plddt.linear_3"] = [{
        "weight":
        state_dict[f"{prefix}plddt.linear_3.weight"],
        "bias":
        state_dict[f"{prefix}plddt.linear_3.bias"],
    }]
    tbnm_state_dict["plddt.layer_norm"] = [{
        "weight":
        state_dict[f"{prefix}plddt.layer_norm.weight"],
        "bias":
        state_dict[f"{prefix}plddt.layer_norm.bias"],
    }]

    tbnm_state_dict["distogram.linear"] = [{
        "weight":
        state_dict[f"{prefix}distogram.linear.weight"],
        "bias":
        state_dict[f"{prefix}distogram.linear.bias"],
    }]

    tbnm_state_dict["masked_msa.linear"] = [{
        "weight":
        state_dict[f"{prefix}masked_msa.linear.weight"],
        "bias":
        state_dict[f"{prefix}masked_msa.linear.bias"],
    }]

    tbnm_state_dict["experimentally_resolved.linear"] = [{
        "weight":
        state_dict[f"{prefix}experimentally_resolved.linear.weight"],
        "bias":
        state_dict[f"{prefix}experimentally_resolved.linear.bias"],
    }]

    if config.tm.enabled:
        tbnm_state_dict["tm.linear"] = [{
            "weight":
            state_dict[f"{prefix}tm.linear.weight"],
            "bias":
            state_dict[f"{prefix}tm.linear.bias"],
        }]

    return tbnm_state_dict


def convert_hf_structure_module_torch(config: BaseConfig,
                                      mapping: Mapping = None,
                                      local_checkpoint: str = None,
                                      model_name: str = "openfold2_ptm_1",
                                      weights: dict = None):
    """
    Convert a structure module model from a Hugging Face checkpoint to a PyTorch model weights.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights

    tbnm_state_dict = {}
    layer_path = "structure_module."
    tbnm_state_dict["layer_norm_s"] = [{
        "weight":
        state_dict[f"{layer_path}layer_norm_s.weight"],
        "bias":
        state_dict[f"{layer_path}layer_norm_s.bias"],
    }]
    tbnm_state_dict["layer_norm_z"] = [{
        "weight":
        state_dict[f"{layer_path}layer_norm_z.weight"],
        "bias":
        state_dict[f"{layer_path}layer_norm_z.bias"],
    }]
    tbnm_state_dict["linear_in"] = [{
        "weight":
        state_dict[f"{layer_path}linear_in.weight"],
        "bias":
        state_dict[f"{layer_path}linear_in.bias"],
    }]

    tbnm_state_dict["ipa.linear_q"] = [{
        "weight":
        state_dict[f"{layer_path}ipa.linear_q.weight"],
        "bias":
        state_dict.get(f"{layer_path}ipa.linear_q.bias", None),
    }]

    if not config.is_multimer:
        tbnm_state_dict["ipa.linear_q_points.linear"] = [{
            "weight":
            state_dict[f"{layer_path}ipa.linear_q_points.linear.weight"],
            "bias":
            state_dict[f"{layer_path}ipa.linear_q_points.linear.bias"],
        }]

        tbnm_state_dict["ipa.linear_kv"] = [{
            "weight":
            state_dict[f"{layer_path}ipa.linear_kv.weight"],
            "bias":
            state_dict[f"{layer_path}ipa.linear_kv.bias"],
        }]
        tbnm_state_dict["ipa.linear_kv_points.linear"] = [{
            "weight":
            state_dict[f"{layer_path}ipa.linear_kv_points.linear.weight"],
            "bias":
            state_dict[f"{layer_path}ipa.linear_kv_points.linear.bias"],
        }]
    else:
        if f"{layer_path}ipa.linear_q_points.linear.weight" in state_dict.keys(
        ):
            tbnm_state_dict["ipa.linear_q_points.linear"] = [{
                "weight":
                state_dict[f"{layer_path}ipa.linear_q_points.linear.weight"],
                "bias":
                state_dict[f"{layer_path}ipa.linear_q_points.linear.bias"],
            }]
        else:
            tbnm_state_dict["ipa.linear_q_points.linear"] = [{
                "weight":
                state_dict[f"{layer_path}ipa.linear_q_points.weight"],
                "bias":
                state_dict[f"{layer_path}ipa.linear_q_points.bias"],
            }]

        tbnm_state_dict["ipa.linear_k"] = [{
            "weight":
            state_dict[f"{layer_path}ipa.linear_k.weight"],
            "bias":
            state_dict.get(f"{layer_path}ipa.linear_k.bias", None),
        }]
        tbnm_state_dict["ipa.linear_v"] = [{
            "weight":
            state_dict[f"{layer_path}ipa.linear_v.weight"],
            "bias":
            state_dict.get(f"{layer_path}ipa.linear_v.bias", None),
        }]

        if f"{layer_path}ipa.linear_k_points.linear.weight" in state_dict.keys(
        ):
            tbnm_state_dict["ipa.linear_k_points.linear"] = [{
                "weight":
                state_dict[f"{layer_path}ipa.linear_k_points.linear.weight"],
                "bias":
                state_dict[f"{layer_path}ipa.linear_k_points.linear.bias"],
            }]
        else:
            tbnm_state_dict["ipa.linear_k_points.linear"] = [{
                "weight":
                state_dict[f"{layer_path}ipa.linear_k_points.weight"],
                "bias":
                state_dict[f"{layer_path}ipa.linear_k_points.bias"],
            }]

        if f"{layer_path}ipa.linear_v_points.linear.weight" in state_dict.keys(
        ):
            tbnm_state_dict["ipa.linear_v_points.linear"] = [{
                "weight":
                state_dict[f"{layer_path}ipa.linear_v_points.linear.weight"],
                "bias":
                state_dict[f"{layer_path}ipa.linear_v_points.linear.bias"],
            }]
        else:
            tbnm_state_dict["ipa.linear_v_points.linear"] = [{
                "weight":
                state_dict[f"{layer_path}ipa.linear_v_points.weight"],
                "bias":
                state_dict[f"{layer_path}ipa.linear_v_points.bias"],
            }]

    tbnm_state_dict["ipa.linear_b"] = [{
        "weight":
        state_dict[f"{layer_path}ipa.linear_b.weight"],
        "bias":
        state_dict[f"{layer_path}ipa.linear_b.bias"],
    }]
    tbnm_state_dict["ipa.linear_out"] = [{
        "weight":
        state_dict[f"{layer_path}ipa.linear_out.weight"],
        "bias":
        state_dict[f"{layer_path}ipa.linear_out.bias"],
    }]

    tbnm_state_dict["layer_norm_ipa"] = [{
        "weight":
        state_dict[f"{layer_path}layer_norm_ipa.weight"],
        "bias":
        state_dict[f"{layer_path}layer_norm_ipa.bias"],
    }]

    for transition_layer in range(config.no_transition_layers):
        tbnm_state_dict[f"transition.layers.{transition_layer}.linear_1"] = [{
            "weight":
            state_dict[
                f"{layer_path}transition.layers.{transition_layer}.linear_1.weight"],
            "bias":
            state_dict[
                f"{layer_path}transition.layers.{transition_layer}.linear_1.bias"],
        }]
        tbnm_state_dict[f"transition.layers.{transition_layer}.linear_2"] = [{
            "weight":
            state_dict[
                f"{layer_path}transition.layers.{transition_layer}.linear_2.weight"],
            "bias":
            state_dict[
                f"{layer_path}transition.layers.{transition_layer}.linear_2.bias"],
        }]
        tbnm_state_dict[f"transition.layers.{transition_layer}.linear_3"] = [{
            "weight":
            state_dict[
                f"{layer_path}transition.layers.{transition_layer}.linear_3.weight"],
            "bias":
            state_dict[
                f"{layer_path}transition.layers.{transition_layer}.linear_3.bias"],
        }]
    tbnm_state_dict[f"transition.layer_norm"] = [{
        "weight":
        state_dict[f"{layer_path}transition.layer_norm.weight"],
        "bias":
        state_dict[f"{layer_path}transition.layer_norm.bias"],
    }]

    tbnm_state_dict["bb_update.linear"] = [{
        "weight":
        state_dict[f"{layer_path}bb_update.linear.weight"],
        "bias":
        state_dict[f"{layer_path}bb_update.linear.bias"],
    }]

    tbnm_state_dict["angle_resnet.linear_in"] = [{
        "weight":
        state_dict[f"{layer_path}angle_resnet.linear_in.weight"],
        "bias":
        state_dict[f"{layer_path}angle_resnet.linear_in.bias"],
    }]
    tbnm_state_dict["angle_resnet.linear_initial"] = [{
        "weight":
        state_dict[f"{layer_path}angle_resnet.linear_initial.weight"],
        "bias":
        state_dict[f"{layer_path}angle_resnet.linear_initial.bias"],
    }]
    for block_layer in range(config.no_resnet_blocks):
        tbnm_state_dict[f"angle_resnet.layers.{block_layer}.linear_1"] = [{
            "weight":
            state_dict[
                f"{layer_path}angle_resnet.layers.{block_layer}.linear_1.weight"],
            "bias":
            state_dict[
                f"{layer_path}angle_resnet.layers.{block_layer}.linear_1.bias"],
        }]
        tbnm_state_dict[f"angle_resnet.layers.{block_layer}.linear_2"] = [{
            "weight":
            state_dict[
                f"{layer_path}angle_resnet.layers.{block_layer}.linear_2.weight"],
            "bias":
            state_dict[
                f"{layer_path}angle_resnet.layers.{block_layer}.linear_2.bias"],
        }]
    tbnm_state_dict["angle_resnet.linear_out"] = [{
        "weight":
        state_dict[f"{layer_path}angle_resnet.linear_out.weight"],
        "bias":
        state_dict[f"{layer_path}angle_resnet.linear_out.bias"],
    }]

    tbnm_state_dict["ipa.head_weights"] = state_dict[
        f"{layer_path}ipa.head_weights"]

    return tbnm_state_dict
