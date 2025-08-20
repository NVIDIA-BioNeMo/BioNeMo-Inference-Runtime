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

from tensorrt_bionemo.hubs.checkpoint import load_hf_weights
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.openfold2.configs import (EvoformerStackConfig,
                                                       ExtraMSAStackConfig)


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
    m = mapping if mapping else Mapping()
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

    if m.tp_size > 1:
        proj_a_weight = split(proj_a_weight, m.tp_size, m.tp_rank, 0)
        proj_a_bias = split(proj_a_bias, m.tp_size, m.tp_rank,
                            0) if proj_a_bias is not None else None
        proj_b_weight = split(proj_b_weight, m.tp_size, m.tp_rank, 0)
        proj_b_bias = split(proj_b_bias, m.tp_size, m.tp_rank,
                            0) if proj_b_bias is not None else None
        proj_o_weight = split(proj_o_weight, m.tp_size, m.tp_rank,
                              1)  # ignore slip bias for row tp

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

    tp_size = mapping.tp_size
    tp_rank = mapping.tp_rank
    if tp_size > 1:
        linear_z_weight = split(linear_z_weight, tp_size, tp_rank, 0)
        mha_q_weight = split(mha_q_weight, tp_size, tp_rank, 0)
        mha_k_weight = split(mha_k_weight, tp_size, tp_rank, 0)
        mha_v_weight = split(mha_v_weight, tp_size, tp_rank, 0)
        mha_o_weight = split(mha_o_weight, tp_size, tp_rank, 1)
        mha_g_weight = split(mha_g_weight, tp_size, tp_rank, 0)
        if mha_g_bias is not None:
            mha_g_bias = split(mha_g_bias, tp_size, tp_rank, 0)

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
    m = mapping if mapping else Mapping()
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}

    layer_norm_weight = state_dict[f"{prefix}.layer_norm.weight"]
    layer_norm_bias = state_dict[f"{prefix}.layer_norm.bias"]

    linear_1_weight = state_dict[f"{prefix}.linear_1.weight"]
    linear_1_bias = state_dict.get(f"{prefix}.linear_1.bias", None)

    linear_2_weight = state_dict[f"{prefix}.linear_2.weight"]
    linear_2_bias = state_dict.get(f"{prefix}.linear_2.bias", None)

    if m.tp_size > 1:
        linear_1_weight = split(linear_1_weight, m.tp_size, m.tp_rank, 0)
        linear_1_bias = split(linear_1_bias, m.tp_size, m.tp_rank,
                              0) if linear_1_bias is not None else None
        linear_2_weight = split(linear_2_weight, m.tp_size, m.tp_rank, 1)

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

    p_in_0_weight = state_dict[f"{prefix}.linear_a_p.weight"]
    p_in_0_bias = state_dict[f"{prefix}.linear_a_p.bias"]
    p_in_1_weight = state_dict[f"{prefix}.linear_b_p.weight"]
    p_in_1_bias = state_dict[f"{prefix}.linear_b_p.bias"]
    g_in_0_weight = state_dict[f"{prefix}.linear_a_g.weight"]
    g_in_0_bias = state_dict[f"{prefix}.linear_a_g.bias"]
    g_in_1_weight = state_dict[f"{prefix}.linear_b_g.weight"]
    g_in_1_bias = state_dict[f"{prefix}.linear_b_g.bias"]

    p_out_weight = state_dict[f"{prefix}.linear_z.weight"]
    p_out_bias = state_dict[f"{prefix}.linear_z.bias"]
    g_out_weight = state_dict[f"{prefix}.linear_g.weight"]
    g_out_bias = state_dict[f"{prefix}.linear_g.bias"]

    tp_size = mapping.tp_size
    tp_rank = mapping.tp_rank
    if tp_size > 1:
        p_in_0_weight = split(p_in_0_weight, tp_size, tp_rank, 0)
        p_in_0_bias = split(p_in_0_bias, tp_size, tp_rank, 0)
        p_in_1_weight = split(p_in_1_weight, tp_size, tp_rank, 0)
        p_in_1_bias = split(p_in_1_bias, tp_size, tp_rank, 0)
        g_in_0_weight = split(g_in_0_weight, tp_size, tp_rank, 0)
        g_in_0_bias = split(g_in_0_bias, tp_size, tp_rank, 0)
        g_in_1_weight = split(g_in_1_weight, tp_size, tp_rank, 0)
        g_in_1_bias = split(g_in_1_bias, tp_size, tp_rank, 0)

        p_out_weight = split(p_out_weight, tp_size, tp_rank, 0)
        p_out_bias = split(p_out_bias, tp_size, tp_rank, 0)
        g_out_weight = split(g_out_weight, tp_size, tp_rank, 0)
        g_out_bias = split(g_out_bias, tp_size, tp_rank, 0)

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

    tp_size = mapping.tp_size
    tp_rank = mapping.tp_rank
    if tp_size > 1:
        linear_weight = split(linear_weight, tp_size, tp_rank, 0)
        mha_q_weight = split(mha_q_weight, tp_size, tp_rank, 0)
        mha_k_weight = split(mha_k_weight, tp_size, tp_rank, 0)
        mha_v_weight = split(mha_v_weight, tp_size, tp_rank, 0)
        mha_o_weight = split(mha_o_weight, tp_size, tp_rank, 1)
        mha_g_weight = split(mha_g_weight, tp_size, tp_rank, 0)
        mha_g_bias = split(mha_g_bias, tp_size, tp_rank, 0)
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
    m = mapping if mapping else Mapping()
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}

    layer_norm_weight = state_dict[f"{prefix}.layer_norm.weight"]
    layer_norm_bias = state_dict[f"{prefix}.layer_norm.bias"]

    linear_1_weight = state_dict[f"{prefix}.linear_1.weight"]
    linear_1_bias = state_dict.get(f"{prefix}.linear_1.bias", None)

    linear_2_weight = state_dict[f"{prefix}.linear_2.weight"]
    linear_2_bias = state_dict.get(f"{prefix}.linear_2.bias", None)

    if m.tp_size > 1:
        linear_1_weight = split(linear_1_weight, m.tp_size, m.tp_rank, 0)
        linear_1_bias = split(linear_1_bias, m.tp_size, m.tp_rank,
                              0) if linear_1_bias is not None else None
        linear_2_weight = split(linear_2_weight, m.tp_size, m.tp_rank, 1)

    ret[f"{tbm_prefix}.layer_norm.weight"] = layer_norm_weight.to(torch_dtype)
    ret[f"{tbm_prefix}.layer_norm.bias"] = layer_norm_bias.to(torch_dtype)

    ret[f"{tbm_prefix}.linear_1.weight"] = linear_1_weight.to(torch_dtype)
    if linear_1_bias is not None:
        ret[f"{tbm_prefix}.linear_1.bias"] = linear_1_bias.to(torch_dtype)

    ret[f"{tbm_prefix}.linear_2.weight"] = linear_2_weight.to(torch_dtype)
    if linear_2_bias is not None:
        ret[f"{tbm_prefix}.linear_2.bias"] = linear_2_bias.to(torch_dtype)

    return ret


def convert_hf_evoformer(config: EvoformerStackConfig,
                         mapping: Mapping = None,
                         local_checkpoint: str = None,
                         model_name: str = "openfold2_ptm_1"):
    """
    Convert a pairformer model from a Hugging Face checkpoint to a TensorRT model weights.
    """
    mapping = mapping if mapping is not None else Mapping()
    prefix = "evoformer"

    if local_checkpoint is not None:
        state_dict = torch.load(local_checkpoint,
                                map_location="cpu",
                                weights_only=False)
    else:
        state_dict = load_hf_weights(name=model_name)

    weights = {}
    weights.update(get_linear_weights(state_dict, f"{prefix}.linear",
                                      f"linear"))

    logger.info(
        f"Loading weights for evoformer, model_name: {model_name}, dtype: {config.dtype}, num_blocks: {config.no_blocks}"
    )

    for i in range(config.no_blocks):
        weights.update(
            get_outer_product_mean_weights(
                state_dict,
                f"{prefix}.blocks.{i}.core.outer_product_mean",
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
            get_msa_transition_weights(
                state_dict,
                f"{prefix}.blocks.{i}.core.msa_transition",
                f"blocks.{i}.msa_transition",
                dtype=config.dtype,
                mapping=mapping))
        weights.update(
            get_pair_transition_weights(
                state_dict,
                f"{prefix}.blocks.{i}.core.pair_transition",
                f"blocks.{i}.pair_transition",
                dtype=config.dtype,
                mapping=mapping))
        weights.update(
            get_tri_mul_node_weights(state_dict,
                                     f"{prefix}.blocks.{i}.core.tri_mul_in",
                                     f"blocks.{i}.tri_mul_in",
                                     dtype=config.dtype,
                                     mapping=mapping))
        weights.update(
            get_tri_mul_node_weights(state_dict,
                                     f"{prefix}.blocks.{i}.core.tri_mul_out",
                                     f"blocks.{i}.tri_mul_out",
                                     dtype=config.dtype,
                                     mapping=mapping))
        weights.update(
            get_tri_attn_node_weights(state_dict,
                                      f"{prefix}.blocks.{i}.core.tri_att_start",
                                      f"blocks.{i}.tri_attn_start",
                                      dtype=config.dtype,
                                      mapping=mapping))
        weights.update(
            get_tri_attn_node_weights(state_dict,
                                      f"{prefix}.blocks.{i}.core.tri_att_end",
                                      f"blocks.{i}.tri_attn_end",
                                      dtype=config.dtype,
                                      mapping=mapping))

    return weights


def _load_openfold2_weights(local_checkpoint: str = None,
                            weights: dict = None,
                            model_name: str = "openfold2_ptm_1"):
    state_dict = None

    if weights is not None:
        state_dict = weights

    if state_dict is None and local_checkpoint is not None:
        state_dict = torch.load(local_checkpoint,
                                map_location="cpu",
                                weights_only=False)
    elif state_dict is None:
        logger.info(
            f"`weights` and `local_checkpoint` aren't both provided, loading from HuggingFace"
        )
        ckpt = load_hf_weights(name=model_name, return_raw=True)
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
    return state_dict


def convert_hf_evoformer_torch(config: EvoformerStackConfig,
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
    state_dict = _load_openfold2_weights(local_checkpoint, weights, model_name)
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
            module_state_dict[f"blocks.{i}.outer_product_mean.layer_norm.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.outer_product_mean.fused_proj_a_b"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.outer_product_mean.linear_1.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.outer_product_mean.linear_1.bias"],
        }, {
            "weight":
            module_state_dict[f"blocks.{i}.outer_product_mean.linear_2.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.outer_product_mean.linear_2.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.outer_product_mean.proj_o"] = [{
            "weight":
            module_state_dict[
                f"blocks.{i}.outer_product_mean.linear_out.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.outer_product_mean.linear_out.bias"],
        }]

        # Update weights for tri_mul_out, tri_mul_in
        for name in ["tri_mul_out", "tri_mul_in"]:
            tbnm_state_dict[f"blocks.{i}.{name}.norm_in"] = [{
                "weight":
                module_state_dict[f"blocks.{i}.{name}.layer_norm_in.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.{name}.layer_norm_in.bias"],
            }]
            tbnm_state_dict[f"blocks.{i}.{name}.p_in"] = [{
                "weight":
                module_state_dict[f"blocks.{i}.{name}.linear_a_p.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.{name}.linear_a_p.bias"],
            }, {
                "weight":
                module_state_dict[f"blocks.{i}.{name}.linear_b_p.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.{name}.linear_b_p.bias"],
            }]
            tbnm_state_dict[f"blocks.{i}.{name}.g_in"] = [{
                "weight":
                module_state_dict[f"blocks.{i}.{name}.linear_a_g.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.{name}.linear_a_g.bias"],
            }, {
                "weight":
                module_state_dict[f"blocks.{i}.{name}.linear_b_g.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.{name}.linear_b_g.bias"],
            }]
            tbnm_state_dict[f"blocks.{i}.{name}.p_out"] = [{
                "weight":
                module_state_dict[f"blocks.{i}.{name}.linear_z.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.{name}.linear_z.bias"],
            }]
            tbnm_state_dict[f"blocks.{i}.{name}.g_out"] = [{
                "weight":
                module_state_dict[f"blocks.{i}.{name}.linear_g.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.{name}.linear_g.bias"],
            }]
            tbnm_state_dict[f"blocks.{i}.{name}.norm_out"] = [{
                "weight":
                module_state_dict[f"blocks.{i}.{name}.layer_norm_out.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.{name}.layer_norm_out.bias"],
            }]

        # weight for tri_attn_start and tri_attn_end
        for name in ["start", "end"]:
            tbnm_state_dict[f"blocks.{i}.tri_attn_{name}.layer_norm"] = [{
                "weight":
                module_state_dict[
                    f"blocks.{i}.tri_att_{name}.layer_norm.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.tri_att_{name}.layer_norm.bias"],
            }]
            tbnm_state_dict[f"blocks.{i}.tri_attn_{name}.linear"] = [{
                "weight":
                module_state_dict[f"blocks.{i}.tri_att_{name}.linear.weight"],
                "bias":
                None
            }]
            tbnm_state_dict[f"blocks.{i}.tri_attn_{name}.mha.qkv_proj"] = [{
                "weight":
                module_state_dict[
                    f"blocks.{i}.tri_att_{name}.mha.linear_q.weight"],
                "bias":
                None
            }, {
                "weight":
                module_state_dict[
                    f"blocks.{i}.tri_att_{name}.mha.linear_k.weight"],
                "bias":
                None
            }, {
                "weight":
                module_state_dict[
                    f"blocks.{i}.tri_att_{name}.mha.linear_v.weight"],
                "bias":
                None
            }]
            tbnm_state_dict[f"blocks.{i}.tri_attn_{name}.mha.o_proj"] = [{
                "weight":
                module_state_dict[
                    f"blocks.{i}.tri_att_{name}.mha.linear_o.weight"],
                "bias":
                module_state_dict[
                    f"blocks.{i}.tri_att_{name}.mha.linear_o.bias"],
            }]
            tbnm_state_dict[f"blocks.{i}.tri_attn_{name}.mha.g_proj"] = [{
                "weight":
                module_state_dict[
                    f"blocks.{i}.tri_att_{name}.mha.linear_g.weight"],
                "bias":
                module_state_dict[
                    f"blocks.{i}.tri_att_{name}.mha.linear_g.bias"],
            }]
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


def convert_hf_extra_msa_stack_torch(config: ExtraMSAStackConfig,
                                     mapping: Mapping = None,
                                     local_checkpoint: str = None,
                                     model_name: str = "openfold2_ptm_1",
                                     weights: dict = None):
    """
    Convert a extra msa stack model from a Hugging Face checkpoint to a PyTorch model weights.
    """
    state_dict = _load_openfold2_weights(local_checkpoint, weights, model_name)
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
            module_state_dict[f"blocks.{i}.outer_product_mean.layer_norm.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.outer_product_mean.fused_proj_a_b"] = [{
            "weight":
            module_state_dict[f"blocks.{i}.outer_product_mean.linear_1.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.outer_product_mean.linear_1.bias"],
        }, {
            "weight":
            module_state_dict[f"blocks.{i}.outer_product_mean.linear_2.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.outer_product_mean.linear_2.bias"],
        }]
        tbnm_state_dict[f"blocks.{i}.outer_product_mean.proj_o"] = [{
            "weight":
            module_state_dict[
                f"blocks.{i}.outer_product_mean.linear_out.weight"],
            "bias":
            module_state_dict[f"blocks.{i}.outer_product_mean.linear_out.bias"],
        }]

        # Update weights for tri_mul_out, tri_mul_in
        for name in ["tri_mul_out", "tri_mul_in"]:
            tbnm_state_dict[f"blocks.{i}.{name}.norm_in"] = [{
                "weight":
                module_state_dict[f"blocks.{i}.{name}.layer_norm_in.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.{name}.layer_norm_in.bias"],
            }]
            tbnm_state_dict[f"blocks.{i}.{name}.p_in"] = [{
                "weight":
                module_state_dict[f"blocks.{i}.{name}.linear_a_p.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.{name}.linear_a_p.bias"],
            }, {
                "weight":
                module_state_dict[f"blocks.{i}.{name}.linear_b_p.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.{name}.linear_b_p.bias"],
            }]
            tbnm_state_dict[f"blocks.{i}.{name}.g_in"] = [{
                "weight":
                module_state_dict[f"blocks.{i}.{name}.linear_a_g.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.{name}.linear_a_g.bias"],
            }, {
                "weight":
                module_state_dict[f"blocks.{i}.{name}.linear_b_g.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.{name}.linear_b_g.bias"],
            }]
            tbnm_state_dict[f"blocks.{i}.{name}.p_out"] = [{
                "weight":
                module_state_dict[f"blocks.{i}.{name}.linear_z.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.{name}.linear_z.bias"],
            }]
            tbnm_state_dict[f"blocks.{i}.{name}.g_out"] = [{
                "weight":
                module_state_dict[f"blocks.{i}.{name}.linear_g.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.{name}.linear_g.bias"],
            }]
            tbnm_state_dict[f"blocks.{i}.{name}.norm_out"] = [{
                "weight":
                module_state_dict[f"blocks.{i}.{name}.layer_norm_out.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.{name}.layer_norm_out.bias"],
            }]

        # weight for tri_attn_start and tri_attn_end
        for name in ["start", "end"]:
            tbnm_state_dict[f"blocks.{i}.tri_attn_{name}.layer_norm"] = [{
                "weight":
                module_state_dict[
                    f"blocks.{i}.tri_att_{name}.layer_norm.weight"],
                "bias":
                module_state_dict[f"blocks.{i}.tri_att_{name}.layer_norm.bias"],
            }]
            tbnm_state_dict[f"blocks.{i}.tri_attn_{name}.linear"] = [{
                "weight":
                module_state_dict[f"blocks.{i}.tri_att_{name}.linear.weight"],
                "bias":
                None
            }]
            tbnm_state_dict[f"blocks.{i}.tri_attn_{name}.mha.qkv_proj"] = [{
                "weight":
                module_state_dict[
                    f"blocks.{i}.tri_att_{name}.mha.linear_q.weight"],
                "bias":
                None
            }, {
                "weight":
                module_state_dict[
                    f"blocks.{i}.tri_att_{name}.mha.linear_k.weight"],
                "bias":
                None
            }, {
                "weight":
                module_state_dict[
                    f"blocks.{i}.tri_att_{name}.mha.linear_v.weight"],
                "bias":
                None
            }]
            tbnm_state_dict[f"blocks.{i}.tri_attn_{name}.mha.o_proj"] = [{
                "weight":
                module_state_dict[
                    f"blocks.{i}.tri_att_{name}.mha.linear_o.weight"],
                "bias":
                module_state_dict[
                    f"blocks.{i}.tri_att_{name}.mha.linear_o.bias"],
            }]
            tbnm_state_dict[f"blocks.{i}.tri_attn_{name}.mha.g_proj"] = [{
                "weight":
                module_state_dict[
                    f"blocks.{i}.tri_att_{name}.mha.linear_g.weight"],
                "bias":
                module_state_dict[
                    f"blocks.{i}.tri_att_{name}.mha.linear_g.bias"],
            }]
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
