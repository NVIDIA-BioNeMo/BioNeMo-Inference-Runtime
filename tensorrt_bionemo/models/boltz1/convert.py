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
import re

import torch
import torch.nn as nn
import tqdm
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.models.convert_utils import split

from tensorrt_bionemo.configs import PairformerConfig, TokenTransformerConfig
from tensorrt_bionemo.hubs.checkpoint import load_hf_weights
from tensorrt_bionemo.mapping import Mapping, create_max_tp_mapping


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
        norm_z_bias = state_dict[f"{prefix}.proj_z.0.bias"]
        z_weight = state_dict[f"{prefix}.proj_z.1.weight"]
        if tp_size > 1:
            z_weight = split(z_weight, tp_size, tp_rank, 0)
        ret.update({
            f"{tbm_prefix}.proj_z_norm.weight":
            norm_z_weight.to(torch_dtype),
            f"{tbm_prefix}.proj_z_norm.bias":
            norm_z_bias.to(torch_dtype),
            f"{tbm_prefix}.proj_z.weight":
            z_weight.to(torch_dtype),
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
    if local_checkpoint is not None:
        state_dict = torch.load(local_checkpoint,
                                map_location="cpu",
                                weights_only=False)["state_dict"]
    else:
        state_dict = load_hf_weights(name=model_name)
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


def torch_load_qkv_weights(module: nn.Module, weights: dict, name: str):
    weight = weights[f"{name}.weight"]
    bias = weights.get(f"{name}.bias", None)
    q_weight, k_weight, v_weight = weight.chunk(3, dim=0)
    q_bias, k_bias, v_bias = None, None, None
    if bias is not None:
        q_bias, k_bias, v_bias = bias.chunk(3, dim=0)
    module_dtype = module.dtype
    module.load_weights([
        {
            "weight": q_weight.to(module_dtype),
            "bias": q_bias.to(module_dtype) if q_bias is not None else None
        },
        {
            "weight": k_weight.to(module_dtype),
            "bias": k_bias.to(module_dtype) if k_bias is not None else None
        },
        {
            "weight": v_weight.to(module_dtype),
            "bias": v_bias.to(module_dtype) if v_bias is not None else None
        },
    ])


def torch_load_kv_weights(module: nn.Module, weights: dict, name: str):
    weight = weights[f"{name}.weight"]
    bias = weights.get(f"{name}.bias", None)
    k_weight, v_weight = weight.chunk(2, dim=0)
    k_bias, v_bias = None, None
    if bias is not None:
        k_bias, v_bias = bias.chunk(2, dim=0)
    module_dtype = module.dtype
    module.load_weights([
        {
            "weight": k_weight.contiguous().to(module_dtype),
            "bias": k_bias.to(module_dtype) if k_bias is not None else None
        },
        {
            "weight": v_weight.contiguous().to(module_dtype),
            "bias": v_bias.to(module_dtype) if v_bias is not None else None
        },
    ])


def torch_load_vanilla_weights(module: nn.Module, weights: dict, name: str):
    weight = weights[f"{name}.weight"]
    bias = weights.get(f"{name}.bias", None)
    module_dtype = module.dtype
    module.load_weights([{
        "weight":
        weight.to(module_dtype),
        "bias":
        bias.to(module_dtype) if bias is not None else None
    }])


def torch_pairformer_load_fn(pretrained_module: nn.Module,
                             checkpoint_dir: str = None,
                             world_size: int = 1,
                             rank: int = 0,
                             weights: dict = None,
                             pairformer_type: str = "structure",
                             **kwargs):
    """
    Load a pairformer model from a PyTorch checkpoint.
    This function is used to in the tensorrt_bionemo.runtime.backend_builder.BackendBuilder.load_weights method.

    Args:
        module: The module to load the weights into.
        checkpoint_dir: The directory to load the checkpoint from.
        world_size: The number of processes to use.
        rank: The rank of the process.
        weights: The weights to load into the module.
        pairformer_type: The type of pairformer to convert. 'structure' or 'confidence'
    """
    if weights is None:
        weights = convert_hf_pairformer(module.config,
                                        Mapping(),
                                        pairformer_type,
                                        local_checkpoint=checkpoint_dir)

    for name, module in tqdm.tqdm(list(module.named_modules()),
                                  desc="Loading weights"):
        if len(module._parameters) > 0:
            if name.endswith(".attention.proj_z.0"):
                prefix = ".".join(name.split(".")[:-1])
                weight = weights[f"{prefix}_norm.weight"]
                bias = weights[f"{prefix}_norm.bias"]
                module.bias.data.copy_(bias.to(module.weight.dtype))
                module.weight.data.copy_(weight.to(module.weight.dtype))
            elif name.endswith(".attention.proj_z.1"):
                prefix = ".".join(name.split(".")[:-1])
                weight = weights[f"{prefix}.weight"]
                bias = weights.get(f"{prefix}.bias", None)
                if bias is not None:
                    bias = bias.to(module.weight.dtype)
                module.load_weights([{
                    "weight": weight.to(module.weight.dtype),
                    "bias": bias
                }])
            elif hasattr(module, "load_weights"):
                weight = weights[f"{name}.weight"]
                bias = weights.get(f"{name}.bias", None)
                module_dtype = module.dtype
                if "qkv_proj" in name:
                    torch_load_qkv_weights(module, weights, name)
                elif "kv_proj" in name or "proj_kv" in name or "p_in" in name or "g_in" in name:
                    torch_load_kv_weights(module, weights, name)
                elif "fused_fc2_fc1" in name:
                    torch_load_kv_weights(module, weights, name)
                else:
                    module.load_weights([{
                        "weight":
                        weight.to(module_dtype),
                        "bias":
                        bias.to(module_dtype) if bias is not None else None
                    }])
            else:
                for n, p in module._parameters.items():
                    if f"{name}.{n}" in weights:
                        p.data.copy_(weights[f"{name}.{n}"].to(p.dtype))
                    else:
                        logger.warning(f"Missing weights for {name}.{n}")


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


def convert_hf_token_transformer(config: TokenTransformerConfig,
                                 mapping: Mapping,
                                 local_checkpoint: str = None,
                                 model_name: str = "boltz-1"):
    """
    Convert a token transformer model from a Hugging Face checkpoint to a TensorRT model weights.
    """
    mapping = mapping if mapping is not None else Mapping()
    prefix = "structure_module.score_model.token_transformer.layers"
    tbm_prefix = "layers"
    weights = {}
    if local_checkpoint is not None:
        state_dict = torch.load(local_checkpoint,
                                map_location="cpu",
                                weights_only=False)["state_dict"]
    else:
        state_dict = load_hf_weights(name=model_name)
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


def torch_load_adaln_weights(module: nn.Module, weights: dict, name: str):
    if "s_norm" in name:  # skip a_norm because it's initialized by torch
        module.weight.data.copy_(weights[f"{name}.weight"].to(
            module.weight.dtype))
        if module.bias is not None:
            module.bias.data.copy_(weights[f"{name}.bias"].to(
                module.bias.dtype))
    elif "fused_s_scale_s_bias" in name:
        weight = weights[f"{name}.weight"]
        bias = weights.get(f"{name}.bias", None)
        s_scale_weight, s_bias_weight = weight.chunk(2, dim=0)
        s_scale_bias, s_bias_bias = None, None
        if bias is not None:
            s_scale_bias, s_bias_bias = bias.chunk(2, dim=0)
        module.load_weights([
            {
                "weight":
                s_scale_weight.to(module.weight.dtype),
                "bias":
                s_scale_bias.to(module.bias.dtype)
                if s_scale_bias is not None else None
            },
            {
                "weight":
                s_bias_weight.to(module.weight.dtype),
                "bias":
                s_bias_bias.to(module.bias.dtype)
                if s_bias_bias is not None else None
            },
        ])


def torch_token_transformer_load_fn(pretrained_module: nn.Module,
                                    checkpoint_dir: str = None,
                                    world_size: int = 1,
                                    rank: int = 0,
                                    weights: dict = None,
                                    **kwargs):
    """
    Load a token transformer model from a PyTorch checkpoint.
    """
    if weights is None:
        weights = convert_hf_token_transformer(pretrained_module.config,
                                               Mapping(),
                                               local_checkpoint=checkpoint_dir)

    version = pretrained_module.config.version
    for name, module in tqdm.tqdm(list(pretrained_module.named_modules()),
                                  desc="Loading weights"):
        if len(module._parameters) > 0:
            if re.search(r"layers\.\d+\.adaln\..+", name):
                torch_load_adaln_weights(module, weights, name)
            elif "pair_bias_attn" in name:
                # TODO: move to a function
                if name.endswith(".proj_z.0"):
                    if version != "v1":  # skip for v2
                        continue
                    prefix = ".".join(name.split(".")[:-1])
                    weight = weights[f"{prefix}_norm.weight"]
                    bias = weights[f"{prefix}_norm.bias"]
                    module.bias.data.copy_(bias.to(module.weight.dtype))
                    module.weight.data.copy_(weight.to(module.weight.dtype))
                elif name.endswith(".proj_z.1"):
                    if version != "v1":  # skip for v2
                        continue
                    prefix = ".".join(name.split(".")[:-1])
                    weight = weights[f"{prefix}.weight"]
                    bias = weights.get(f"{prefix}.bias", None)
                    if bias is not None:
                        bias = bias.to(module.weight.dtype)
                    module.load_weights([{
                        "weight":
                        weight.to(module.weight.dtype),
                        "bias":
                        bias
                    }])
                elif "proj_kv" in name:
                    torch_load_kv_weights(module, weights, name)
                elif hasattr(module, "load_weights"):
                    torch_load_vanilla_weights(module, weights, name)
            elif "transition" in name:
                if "adaln" in name:
                    torch_load_adaln_weights(module, weights, name)
                elif "fused_swl_a_to_b" in name:
                    torch_load_qkv_weights(module, weights, name)
                elif "output_projection" in name:
                    torch_load_vanilla_weights(module, weights, name)
                elif "b_to_a" in name:
                    torch_load_vanilla_weights(module, weights, name)
            elif "output_projection" in name:
                torch_load_vanilla_weights(module, weights, name)
            elif "post_lnorm" in name:
                weight = weights[f"{name}.weight"]
                bias = weights[f"{name}.bias"]
                module.bias.data.copy_(bias.to(module.weight.dtype))
                module.weight.data.copy_(weight.to(module.weight.dtype))
            else:
                logger.warning(f"Missing weights for {name}")
