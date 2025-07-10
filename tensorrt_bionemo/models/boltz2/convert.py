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
import torch.nn as nn
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.logger import logger

from tensorrt_bionemo.configs import AffinityModuleConfig, PairformerConfig
from tensorrt_bionemo.hubs.checkpoint import load_hf_weights
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.boltz1.convert import \
    convert_hf_token_transformer as boltz1_convert_hf_token_transformer
from tensorrt_bionemo.models.boltz1.convert import (get_pairwise_attn_weights,
                                                    get_transition_weights,
                                                    get_tri_attn_node_weights,
                                                    get_tri_mul_node_weights,
                                                    torch_load_kv_weights)
from tensorrt_bionemo.models.boltz1.convert import \
    torch_pairformer_load_fn as boltz1_torch_pairformer_load_fn
from tensorrt_bionemo.models.boltz1.convert import \
    torch_token_transformer_load_fn as boltz1_torch_token_transformer_load_fn


def get_post_pre_norm_weights(state_dict: dict,
                              prefix: str,
                              tbm_prefix: str,
                              post_layer_norm: bool = False,
                              dtype: str = "float32"):
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}
    pre_norm_weight = state_dict[f"{prefix}.pre_norm_s.weight"]
    pre_norm_bias = state_dict[f"{prefix}.pre_norm_s.bias"]

    ret[f"{tbm_prefix}.pre_norm_s.weight"] = pre_norm_weight.to(torch_dtype)
    ret[f"{tbm_prefix}.pre_norm_s.bias"] = pre_norm_bias.to(torch_dtype)

    if post_layer_norm:
        post_norm_weight = state_dict[f"{prefix}.post_norm_s.weight"]
        post_norm_bias = state_dict[f"{prefix}.post_norm_s.bias"]
        ret[f"{tbm_prefix}.post_norm_s.weight"] = post_norm_weight.to(
            torch_dtype)
        ret[f"{tbm_prefix}.post_norm_s.bias"] = post_norm_bias.to(torch_dtype)
    return ret


def convert_hf_pairformer(config: PairformerConfig,
                          mapping: Mapping,
                          pairformer_type: str = "structure",
                          local_checkpoint: str = None,
                          model_name: str = "boltz-2"):
    """
    Convert a pairformer model from a Hugging Face checkpoint to a TensorRT model weights.
    """
    mapping = mapping if mapping is not None else Mapping()
    prefix = "pairformer_module.layers"
    if pairformer_type == "confidence":
        prefix = "pairformer_stack.layers"
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
        f"Loading weights for {pairformer_type} pairformer, dtype: {config.dtype}, num_blocks: {config.num_blocks}"
    )

    for i in range(config.num_blocks):
        layer_prefix = f"{prefix}.{i}"
        layer_tbm_prefix = f"{tbm_prefix}.{i}"
        weights.update(
            get_post_pre_norm_weights(state_dict,
                                      f"{layer_prefix}",
                                      f"{layer_tbm_prefix}",
                                      config.post_layer_norm,
                                      dtype="float32"))
        weights.update(
            get_pairwise_attn_weights(
                mapping,
                state_dict,
                f"{layer_prefix}.attention",
                f"{layer_tbm_prefix}.attention",
                config.max_attention_pairwise_tp_size,
                config.num_heads,
                attention_initial_norm=config.attention_initial_norm,
                dtype="float32"))
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
        weights.update(
            get_transition_weights(mapping,
                                   state_dict,
                                   f"{layer_prefix}.transition_s",
                                   f"{layer_tbm_prefix}.transition_s",
                                   config.max_transition_tp_size,
                                   config.token_s * 4,
                                   dtype="float32"))
        weights.update(
            get_transition_weights(mapping,
                                   state_dict,
                                   f"{layer_prefix}.transition_z",
                                   f"{layer_tbm_prefix}.transition_z",
                                   config.max_transition_tp_size,
                                   config.token_z * 4,
                                   dtype=config.dtype))
    return weights


def torch_pairformer_load_fn(module: nn.Module,
                             checkpoint_dir: str = None,
                             world_size: int = 1,
                             rank: int = 0,
                             weights: dict = None,
                             pairformer_type: str = "structure",
                             **kwargs):
    """
    Load a pairformer model from a PyTorch checkpoint.

    Args:
        module: The module to load the weights into.
        checkpoint_dir: The directory to load the checkpoint from.
        world_size: The number of processes to use.
        rank: The rank of the process.
        weights: The weights to load into the module.
        pairformer_type: The type of pairformer to convert. 'structure' or 'confidence'
    """
    weights = convert_hf_pairformer(module.config,
                                    Mapping(),
                                    pairformer_type,
                                    local_checkpoint=checkpoint_dir)

    boltz1_torch_pairformer_load_fn(module, checkpoint_dir, world_size, rank,
                                    weights, pairformer_type, **kwargs)

    for name, module in list(module.named_modules()):
        if name.endswith(".pre_norm_s") or name.endswith(".post_norm_s"):
            weight = weights[f"{name}.weight"]
            bias = weights[f"{name}.bias"]
            module.bias.data.copy_(bias.to(module.weight.dtype))
            module.weight.data.copy_(weight.to(module.weight.dtype))


def torch_token_transformer_load_fn(*args, **kwargs):
    boltz1_torch_token_transformer_load_fn(*args, **kwargs)


def convert_hf_token_transformer(*args, **kwargs):
    if kwargs.get("model_name") is None:
        kwargs["model_name"] = "boltz-2"
    return boltz1_convert_hf_token_transformer(*args, **kwargs)


def get_pairwise_conditioner_weights(mapping: Mapping,
                                     state_dict: dict,
                                     prefix: str,
                                     tbm_prefix: str,
                                     num_transitions: int,
                                     dtype: str = "float32"):
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}
    init_proj_norm_weight = state_dict[
        f"{prefix}.dim_pairwise_init_proj.0.weight"]
    init_proj_norm_bias = state_dict[f"{prefix}.dim_pairwise_init_proj.0.bias"]
    init_proj_linear_weight = state_dict[
        f"{prefix}.dim_pairwise_init_proj.1.weight"]

    if mapping.tp_size > 1:
        init_proj_linear_weight = split(init_proj_linear_weight,
                                        mapping.tp_size, mapping.tp_rank, 0)

    ret.update({
        f"{tbm_prefix}.init_proj_norm.weight":
        init_proj_norm_weight.to(torch_dtype),
        f"{tbm_prefix}.init_proj_norm.bias":
        init_proj_norm_bias.to(torch_dtype),
        f"{tbm_prefix}.init_proj_linear.weight":
        init_proj_linear_weight.to(torch_dtype),
    })
    for i in range(num_transitions):
        transition_layer_prefix = f"{prefix}.transitions.{i}"
        transition_layer_tbm_prefix = f"{tbm_prefix}.transitions.{i}"
        ret.update(
            get_transition_weights(mapping,
                                   state_dict,
                                   transition_layer_prefix,
                                   transition_layer_tbm_prefix,
                                   True,
                                   init_proj_linear_weight.shape[0] * 4,
                                   dtype=dtype))
    return ret


def get_pairformer_no_seq_weights(mapping: Mapping,
                                  state_dict: dict,
                                  prefix: str,
                                  tbm_prefix: str,
                                  max_tri_mul_tp_size: bool = True,
                                  max_transition_tp_size: bool = True,
                                  token_z: int = 128,
                                  dtype: str = "float32"):
    str_dtype_to_torch(dtype)
    weights = {}
    weights.update(
        get_tri_attn_node_weights(mapping,
                                  state_dict,
                                  f"{prefix}.tri_att_start",
                                  f"{tbm_prefix}.tri_attn_start",
                                  dtype=dtype))
    weights.update(
        get_tri_attn_node_weights(mapping,
                                  state_dict,
                                  f"{prefix}.tri_att_end",
                                  f"{tbm_prefix}.tri_attn_end",
                                  dtype=dtype))
    weights.update(
        get_tri_mul_node_weights(mapping,
                                 state_dict,
                                 f"{prefix}.tri_mul_out",
                                 f"{tbm_prefix}.tri_mul_out",
                                 max_tri_mul_tp_size,
                                 dtype=dtype))
    weights.update(
        get_tri_mul_node_weights(mapping,
                                 state_dict,
                                 f"{prefix}.tri_mul_in",
                                 f"{tbm_prefix}.tri_mul_in",
                                 max_tri_mul_tp_size,
                                 dtype=dtype))

    weights.update(
        get_transition_weights(mapping,
                               state_dict,
                               f"{prefix}.transition_z",
                               f"{tbm_prefix}.transition_z",
                               max_transition_tp_size,
                               token_z * 4,
                               dtype=dtype))
    return weights


def get_affinity_heads_weights(mapping: Mapping,
                               state_dict: dict,
                               prefix: str,
                               tbm_prefix: str,
                               dtype: str = "float32"):
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}
    affinity_out_mlp_linear_0_weight = state_dict[
        f"{prefix}.affinity_out_mlp.0.weight"]
    affinity_out_mlp_linear_0_bias = state_dict[
        f"{prefix}.affinity_out_mlp.0.bias"]
    affinity_out_mlp_linear_1_weight = state_dict[
        f"{prefix}.affinity_out_mlp.2.weight"]
    affinity_out_mlp_linear_1_bias = state_dict[
        f"{prefix}.affinity_out_mlp.2.bias"]

    to_affinity_pred_value_0_weight = state_dict[
        f"{prefix}.to_affinity_pred_value.0.weight"]
    to_affinity_pred_value_0_bias = state_dict[
        f"{prefix}.to_affinity_pred_value.0.bias"]
    to_affinity_pred_value_1_weight = state_dict[
        f"{prefix}.to_affinity_pred_value.2.weight"]
    to_affinity_pred_value_1_bias = state_dict[
        f"{prefix}.to_affinity_pred_value.2.bias"]
    to_affinity_pred_value_2_weight = state_dict[
        f"{prefix}.to_affinity_pred_value.4.weight"]
    to_affinity_pred_value_2_bias = state_dict[
        f"{prefix}.to_affinity_pred_value.4.bias"]

    to_affinity_pred_score_0_weight = state_dict[
        f"{prefix}.to_affinity_pred_score.0.weight"]
    to_affinity_pred_score_0_bias = state_dict[
        f"{prefix}.to_affinity_pred_score.0.bias"]
    to_affinity_pred_score_1_weight = state_dict[
        f"{prefix}.to_affinity_pred_score.2.weight"]
    to_affinity_pred_score_1_bias = state_dict[
        f"{prefix}.to_affinity_pred_score.2.bias"]
    to_affinity_pred_score_2_weight = state_dict[
        f"{prefix}.to_affinity_pred_score.4.weight"]
    to_affinity_pred_score_2_bias = state_dict[
        f"{prefix}.to_affinity_pred_score.4.bias"]

    to_affinity_logits_binary_weight = state_dict[
        f"{prefix}.to_affinity_logits_binary.weight"]
    to_affinity_logits_binary_bias = state_dict[
        f"{prefix}.to_affinity_logits_binary.bias"]

    if mapping.tp_size > 1:
        affinity_out_mlp_linear_0_weight = split(
            affinity_out_mlp_linear_0_weight, mapping.tp_size, mapping.tp_rank,
            0)
        affinity_out_mlp_linear_0_bias = split(affinity_out_mlp_linear_0_bias,
                                               mapping.tp_size, mapping.tp_rank,
                                               0)
        affinity_out_mlp_linear_1_weight = split(
            affinity_out_mlp_linear_1_weight, mapping.tp_size, mapping.tp_rank,
            1)
        affinity_out_mlp_linear_1_bias = split(affinity_out_mlp_linear_1_bias,
                                               mapping.tp_size, mapping.tp_rank,
                                               1)

        to_affinity_pred_value_0_weight = split(to_affinity_pred_value_0_weight,
                                                mapping.tp_size,
                                                mapping.tp_rank, 0)
        to_affinity_pred_value_0_bias = split(to_affinity_pred_value_0_bias,
                                              mapping.tp_size, mapping.tp_rank,
                                              0)
        to_affinity_pred_value_1_weight = split(to_affinity_pred_value_1_weight,
                                                mapping.tp_size,
                                                mapping.tp_rank, 1)
        to_affinity_pred_value_1_bias = split(to_affinity_pred_value_1_bias,
                                              mapping.tp_size, mapping.tp_rank,
                                              1)

        to_affinity_pred_score_0_weight = split(to_affinity_pred_score_0_weight,
                                                mapping.tp_size,
                                                mapping.tp_rank, 0)
        to_affinity_pred_score_0_bias = split(to_affinity_pred_score_0_bias,
                                              mapping.tp_size, mapping.tp_rank,
                                              0)
        to_affinity_pred_score_1_weight = split(to_affinity_pred_score_1_weight,
                                                mapping.tp_size,
                                                mapping.tp_rank, 1)
        to_affinity_pred_score_1_bias = split(to_affinity_pred_score_1_bias,
                                              mapping.tp_size, mapping.tp_rank,
                                              1)

    ret.update({
        f"{tbm_prefix}.affinity_out_mlp_linear_0.weight":
        affinity_out_mlp_linear_0_weight.to(torch_dtype),
        f"{tbm_prefix}.affinity_out_mlp_linear_0.bias":
        affinity_out_mlp_linear_0_bias.to(torch_dtype),
        f"{tbm_prefix}.affinity_out_mlp_linear_1.weight":
        affinity_out_mlp_linear_1_weight.to(torch_dtype),
        f"{tbm_prefix}.affinity_out_mlp_linear_1.bias":
        affinity_out_mlp_linear_1_bias.to(torch_dtype),
        f"{tbm_prefix}.to_affinity_pred_value_0.weight":
        to_affinity_pred_value_0_weight.to(torch_dtype),
        f"{tbm_prefix}.to_affinity_pred_value_0.bias":
        to_affinity_pred_value_0_bias.to(torch_dtype),
        f"{tbm_prefix}.to_affinity_pred_value_1.weight":
        to_affinity_pred_value_1_weight.to(torch_dtype),
        f"{tbm_prefix}.to_affinity_pred_value_1.bias":
        to_affinity_pred_value_1_bias.to(torch_dtype),
        f"{tbm_prefix}.to_affinity_pred_value_2.weight":
        to_affinity_pred_value_2_weight.to(torch_dtype),
        f"{tbm_prefix}.to_affinity_pred_value_2.bias":
        to_affinity_pred_value_2_bias.to(torch_dtype),
        f"{tbm_prefix}.to_affinity_pred_score_0.weight":
        to_affinity_pred_score_0_weight.to(torch_dtype),
        f"{tbm_prefix}.to_affinity_pred_score_0.bias":
        to_affinity_pred_score_0_bias.to(torch_dtype),
        f"{tbm_prefix}.to_affinity_pred_score_1.weight":
        to_affinity_pred_score_1_weight.to(torch_dtype),
        f"{tbm_prefix}.to_affinity_pred_score_1.bias":
        to_affinity_pred_score_1_bias.to(torch_dtype),
        f"{tbm_prefix}.to_affinity_pred_score_2.weight":
        to_affinity_pred_score_2_weight.to(torch_dtype),
        f"{tbm_prefix}.to_affinity_pred_score_2.bias":
        to_affinity_pred_score_2_bias.to(torch_dtype),
        f"{tbm_prefix}.to_affinity_logits_binary.weight":
        to_affinity_logits_binary_weight.to(torch_dtype),
        f"{tbm_prefix}.to_affinity_logits_binary.bias":
        to_affinity_logits_binary_bias.to(torch_dtype),
    })
    return ret


def torch_affinity_module_load_fn(
        module: nn.Module,
        checkpoint_dir: str = None,
        world_size: int = 1,
        rank: int = 0,
        weights: dict = None,
        affinity_module_name: str = "affinity_module1",
        **kwargs):
    """
    Load a affinity module model from a PyTorch checkpoint.
    """
    if weights is None:
        # For torch backend, we only need for weights for world_size = 1
        weights = convert_hf_affinity_module(module.config,
                                             Mapping(),
                                             affinity_module_name,
                                             local_checkpoint=checkpoint_dir)

    for name, module in list(module.named_modules()):
        if len(module._parameters) > 0:
            if hasattr(module, "load_weights"):
                weight = weights[f"{name}.weight"]
                bias = weights.get(f"{name}.bias", None)
                module_dtype = module.dtype
                if "kv_proj" in name or "proj_kv" in name or "p_in" in name or "g_in" in name:
                    torch_load_kv_weights(module, weights, name)
                elif "fused_fc2_fc1" in name or "fused_s_to_z" in name:
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


def convert_hf_affinity_module(config: AffinityModuleConfig,
                               mapping: Mapping,
                               affinity_module_name: str = "affinity_module1",
                               local_checkpoint: str = None,
                               model_name: str = "boltz-2-affinity"):
    """
    Convert a affinity module model from a Hugging Face checkpoint to a TensorRT model weights.
    """
    mapping = mapping if mapping is not None else Mapping()
    prefix = affinity_module_name
    weights = {}
    if local_checkpoint is not None:
        state_dict = torch.load(local_checkpoint,
                                map_location="cpu",
                                weights_only=False)["state_dict"]
    else:
        state_dict = load_hf_weights(name=model_name)

    logger.info(
        f"Loading weights for {affinity_module_name} affinity module, dtype: {config.dtype}, num_dist_bins: {config.num_dist_bins}"
    )

    torch_dtype = str_dtype_to_torch(config.dtype)
    dist_bin_pairwise_embed_weight = state_dict[
        f"{prefix}.dist_bin_pairwise_embed.weight"]
    if mapping.tp_size > 1:
        dist_bin_pairwise_embed_weight = split(dist_bin_pairwise_embed_weight,
                                               mapping.tp_size, mapping.tp_rank,
                                               0)
    weights.update({
        f"dist_bin_pairwise_embed.weight":
        dist_bin_pairwise_embed_weight.to(torch_dtype),
    })
    s_to_z_prod_in1_weight = state_dict[f"{prefix}.s_to_z_prod_in1.weight"]
    s_to_z_prod_in2_weight = state_dict[f"{prefix}.s_to_z_prod_in2.weight"]
    z_norm_weight, z_norm_bias = state_dict[
        f"{prefix}.z_norm.weight"], state_dict[f"{prefix}.z_norm.bias"]
    z_linear_weight = state_dict[f"{prefix}.z_linear.weight"]

    if mapping.tp_size > 1:
        s_to_z_prod_in1_weight = split(s_to_z_prod_in1_weight, mapping.tp_size,
                                       mapping.tp_rank, 0)
        s_to_z_prod_in2_weight = split(s_to_z_prod_in2_weight, mapping.tp_size,
                                       mapping.tp_rank, 0)
        z_linear_weight = split(z_linear_weight, mapping.tp_size,
                                mapping.tp_rank, 0)
    fused_s_to_z_weight = torch.cat(
        [s_to_z_prod_in1_weight, s_to_z_prod_in2_weight], dim=0)

    weights.update({
        f"z_norm.weight": z_norm_weight.to(torch_dtype),
        f"z_norm.bias": z_norm_bias.to(torch_dtype),
        f"z_linear.weight": z_linear_weight.to(torch_dtype),
        f"fused_s_to_z.weight": fused_s_to_z_weight.to(torch_dtype),
    })

    weights.update(
        get_pairwise_conditioner_weights(mapping,
                                         state_dict,
                                         f"{prefix}.pairwise_conditioner",
                                         f"pairwise_conditioner",
                                         2,
                                         dtype=config.dtype))
    for i in range(config.pairformer_num_blocks):
        layer_prefix = f"{prefix}.pairformer_stack.layers.{i}"
        layer_tbm_prefix = f"pairformer_stack.layers.{i}"
        weights.update(
            get_pairformer_no_seq_weights(mapping,
                                          state_dict,
                                          layer_prefix,
                                          layer_tbm_prefix,
                                          dtype=config.dtype,
                                          max_tri_mul_tp_size=True,
                                          max_transition_tp_size=True,
                                          token_z=config.token_z))
    weights.update(
        get_affinity_heads_weights(mapping,
                                   state_dict,
                                   f"{prefix}.affinity_heads",
                                   f"affinity_heads",
                                   dtype=config.dtype))
    return weights
