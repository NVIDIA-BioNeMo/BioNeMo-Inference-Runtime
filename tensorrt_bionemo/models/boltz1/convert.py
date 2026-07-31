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

from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.hubs import load_weights
from tensorrt_bionemo.logger import logger
from tensorrt_bionemo.utils import str_dtype_to_torch


def get_pairwise_attn_weights(state_dict: dict,
                              prefix: str,
                              tbm_prefix: str,
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


def get_tri_attn_node_weights(state_dict: dict,
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


def get_tri_mul_node_weights(state_dict: dict,
                             prefix: str,
                             tbm_prefix: str,
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


def get_transition_weights(state_dict: dict,
                           prefix: str,
                           tbm_prefix: str,
                           dim: int = 128,
                           dtype: str = "float32"):
    torch_dtype = str_dtype_to_torch(dtype)
    norm_weight = state_dict[f"{prefix}.norm.weight"]
    norm_bias = state_dict[f"{prefix}.norm.bias"]
    fc1_weight = state_dict[f"{prefix}.fc1.weight"]
    fc2_weight = state_dict[f"{prefix}.fc2.weight"]
    fc3_weight = state_dict[f"{prefix}.fc3.weight"]

    fused_fc2_fc1_weight = torch.cat([fc2_weight, fc1_weight], dim=0)

    ret = {
        f"{tbm_prefix}.norm.weight": norm_weight.to(torch_dtype),
        f"{tbm_prefix}.norm.bias": norm_bias.to(torch_dtype),
        f"{tbm_prefix}.fused_fc2_fc1.weight":
        fused_fc2_fc1_weight.to(torch_dtype),
        f"{tbm_prefix}.fc3.weight": fc3_weight.to(torch_dtype),
    }
    return ret


def convert_hf_pairformer(config: BaseConfig,
                          pairformer_type: str = "structure",
                          local_checkpoint: str = None,
                          model_name: str = "boltz-1"):
    """
    Convert a pairformer model from a Hugging Face checkpoint to a TensorRT model weights.

    Args:
        pairformer_type: The type of pairformer to convert. 'structure' or 'confidence'
    """
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
                get_pairwise_attn_weights(state_dict,
                                          f"{layer_prefix}.attention",
                                          f"{layer_tbm_prefix}.attention",
                                          config.num_heads,
                                          dtype=config.dtype))
        weights.update(
            get_tri_attn_node_weights(state_dict,
                                      f"{layer_prefix}.tri_att_start",
                                      f"{layer_tbm_prefix}.tri_attn_start",
                                      dtype=config.dtype))
        weights.update(
            get_tri_attn_node_weights(state_dict,
                                      f"{layer_prefix}.tri_att_end",
                                      f"{layer_tbm_prefix}.tri_attn_end",
                                      dtype=config.dtype))
        weights.update(
            get_tri_mul_node_weights(state_dict,
                                     f"{layer_prefix}.tri_mul_out",
                                     f"{layer_tbm_prefix}.tri_mul_out",
                                     dtype=config.dtype))
        weights.update(
            get_tri_mul_node_weights(state_dict,
                                     f"{layer_prefix}.tri_mul_in",
                                     f"{layer_tbm_prefix}.tri_mul_in",
                                     dtype=config.dtype))
        if not config.no_update_s:
            weights.update(
                get_transition_weights(state_dict,
                                       f"{layer_prefix}.transition_s",
                                       f"{layer_tbm_prefix}.transition_s",
                                       config.token_s * 4,
                                       dtype=config.dtype))
        weights.update(
            get_transition_weights(state_dict,
                                   f"{layer_prefix}.transition_z",
                                   f"{layer_tbm_prefix}.transition_z",
                                   config.token_z * 4,
                                   dtype=config.dtype))
    return weights


def convert_hf_pairformer_torch(config: BaseConfig = None,
                                local_checkpoint: str = None,
                                model_name: str = "boltz-1",
                                weights: dict = None,
                                pairformer_type: str = "structure",
                                prefix: str = None,
                                **kwargs):
    """
    This function is used to convert PyTorch weights to dict for pairformer v1 torch backend.
    Args:
        config: The configuration for the pairformer module.
        local_checkpoint: The directory to load the checkpoint from. If local_checkpoint is None, the function will load from HuggingFace
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


def get_adaln_weights(state_dict: dict,
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


def get_conditioned_transition_block_weights(state_dict: dict,
                                             prefix: str,
                                             tbm_prefix: str,
                                             dim: int,
                                             dim_single_cond: int,
                                             expansion_factor: int = 2,
                                             dtype: str = "float32"):
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}
    ret.update(
        get_adaln_weights(state_dict,
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

    int(dim * expansion_factor)
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


def get_output_projection_weights(state_dict: dict,
                                  prefix: str,
                                  tbm_prefix: str,
                                  dtype: str = "float32"):
    torch_dtype = str_dtype_to_torch(dtype)
    ret = {}
    output_projection_weight = state_dict[f"{prefix}.0.weight"]
    output_projection_bias = state_dict[f"{prefix}.0.bias"]
    ret.update({
        f"{tbm_prefix}.weight": output_projection_weight.to(torch_dtype),
        f"{tbm_prefix}.bias": output_projection_bias.to(torch_dtype),
    })
    return ret


def get_post_norm_weights(state_dict: dict,
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


def convert_hf_diffusion_transformer(config: BaseConfig = None,
                                     local_checkpoint: str = None,
                                     model_name: str = "boltz-1"):
    """
    Convert a token transformer model from a Hugging Face checkpoint to a TensorRT model weights.
    Args:
        config: The configuration for the diffusion transformer module.
        local_checkpoint: The directory to load the checkpoint from. If local_checkpoint is None, the function will load from HuggingFace
        model_name: The name of the model to load.
    """
    prefix = "structure_module.score_model.token_transformer.layers"
    tbm_prefix = "layers"
    weights = {}
    state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    logger.info(
        f"Loading weights for token transformer, dtype: {config.dtype}")
    for i in range(config.num_blocks):
        layer_prefix = f"{prefix}.{i}"
        layer_tbm_prefix = f"{tbm_prefix}.{i}"
        weights.update(
            get_adaln_weights(state_dict,
                              f"{layer_prefix}.adaln",
                              f"{layer_tbm_prefix}.adaln",
                              config.dim,
                              config.dim_single_cond,
                              dtype=config.dtype))
        weights.update(
            get_pairwise_attn_weights(
                state_dict,
                f"{layer_prefix}.pair_bias_attn",
                f"{layer_tbm_prefix}.pair_bias_attn",
                num_heads=config.num_heads,
                attention_initial_norm=config.attention_initial_norm,
                # compute_pair_bias=config.version == "v1",
                compute_pair_bias=False,
                dtype=config.dtype))
        weights.update(
            get_conditioned_transition_block_weights(
                state_dict,
                f"{layer_prefix}.transition",
                f"{layer_tbm_prefix}.transition",
                config.dim,
                config.dim_single_cond,
                expansion_factor=config.expansion_factor,
                dtype=config.dtype))
        weights.update(
            get_output_projection_weights(
                state_dict,
                f"{layer_prefix}.output_projection",
                f"{layer_tbm_prefix}.output_projection",
                dtype=config.dtype))
        weights.update(
            get_post_norm_weights(state_dict,
                                  f"{layer_prefix}.post_lnorm",
                                  f"{layer_tbm_prefix}.post_lnorm",
                                  dtype=config.dtype))

    return weights


def convert_hf_diffusion_transformer_torch(config: BaseConfig,
                                           local_checkpoint: str = None,
                                           model_name: str = "boltz-1",
                                           weights: dict = None,
                                           **kwargs):
    """
    Convert a token transformer model from a Hugging Face checkpoint to a PyTorch model weights.
    Args:
        config: The configuration for the diffusion transformer module.
        local_checkpoint: The directory to load the checkpoint from. If local_checkpoint is None, the function will load from HuggingFace
        model_name: The name of the model to load.
        weights: The weights to load. If weights is None, the function will load from HuggingFace
        kwargs: Additional arguments for the conversion.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    if "prefix" in kwargs:
        prefix = kwargs["prefix"]
    else:
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

        if f"layers.{i}.pair_bias_attn.proj_z.0.weight" in module_state_dict and config.version == "v1":
            tbnm_state_dict[f"layers.{i}.pair_bias_attn.proj_z.0"] = [{
                'weight':
                module_state_dict[
                    f"layers.{i}.pair_bias_attn.proj_z.0.weight"],
                'bias':
                module_state_dict[f"layers.{i}.pair_bias_attn.proj_z.0.bias"]
            }]
            tbnm_state_dict[f"layers.{i}.pair_bias_attn.proj_z.1"] = [{
                'weight':
                module_state_dict[
                    f"layers.{i}.pair_bias_attn.proj_z.1.weight"],
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
        tbnm_state_dict[
            f"layers.{i}.transition.adaln.fused_s_scale_s_bias"] = [{
                "weight":
                module_state_dict[
                    f"layers.{i}.transition.adaln.s_scale.weight"],
                "bias":
                module_state_dict[f"layers.{i}.transition.adaln.s_scale.bias"]
            }, {
                "weight":
                module_state_dict[
                    f"layers.{i}.transition.adaln.s_bias.weight"],
                "bias":
                torch.zeros([dim], dtype=dtype)
            }]
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
            module_state_dict[
                f"layers.{i}.transition.output_projection.0.bias"]
        }]

    return tbnm_state_dict


def convert_hf_msa_module_torch(config: BaseConfig,
                                local_checkpoint: str = None,
                                model_name: str = "boltz-1",
                                weights: dict = None,
                                **kwargs):
    """
    Convert a msa module model from a Hugging Face checkpoint to a PyTorch model weights.
    Args:
        config: The configuration for the msa module.
        local_checkpoint: The directory to load the checkpoint from. If local_checkpoint is None, the function will load from HuggingFace
        model_name: The name of the model to load.
        weights: The weights to load. If weights is None, the function will load from HuggingFace
        kwargs: Additional arguments for the conversion.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights

    if "prefix" in kwargs:
        prefix = kwargs["prefix"]
    else:
        prefix = "msa_module."
    module_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith(prefix):
            module_state_dict[k.replace(prefix, "")] = v

    tbnm_state_dict = {}
    tbnm_state_dict[f"s_proj"] = [{
        "weight":
        module_state_dict[f"s_proj.weight"],
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
            module_state_dict[
                f"layers.{i}.pair_weighted_averaging.norm_m.bias"]
        }]
        tbnm_state_dict[f"layers.{i}.pair_weighted_averaging.norm_z"] = [{
            'weight':
            module_state_dict[
                f"layers.{i}.pair_weighted_averaging.norm_z.weight"],
            'bias':
            module_state_dict[
                f"layers.{i}.pair_weighted_averaging.norm_z.bias"]
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
            tbnm_state_dict[f"layers.{i}.pairformer_layer.{name}.norm_out"] = [
                {
                    'weight':
                    module_state_dict[f"layers.{i}.{name}.norm_out.weight"],
                    'bias':
                    module_state_dict[f"layers.{i}.{name}.norm_out.bias"]
                }
            ]
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


def convert_hf_input_embedder_torch(config: BaseConfig,
                                    local_checkpoint: str = None,
                                    model_name: str = "boltz-1",
                                    weights: dict = None,
                                    **kwargs):
    """
    Convert a Boltz1x input embedder model from a Hugging Face checkpoint to a PyTorch model weights.
    Args:
        config: The configuration for the input embedder module. Boltz1Config.input_embedder
        local_checkpoint: The directory to load the checkpoint from. If local_checkpoint is None, the function will load from HuggingFace
        model_name: The name of the model to load.
        weights: The weights to load. If weights is None, the function will load from HuggingFace
        kwargs: Additional arguments for the conversion.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    if "prefix" in kwargs:
        prefix = kwargs["prefix"]
    else:
        prefix = "input_embedder."

    weights = {}
    layer_path = f"{prefix}atom_attention_encoder"
    atom_transformer_weights = convert_hf_diffusion_transformer_torch(
        config.diffusion_transformer,
        local_checkpoint=local_checkpoint,
        model_name=model_name,
        weights=state_dict,
        prefix=f"{layer_path}.atom_encoder.diffusion_transformer.")

    for k, v in atom_transformer_weights.items():
        weights[
            f"atom_attention_encoder.atom_encoder.diffusion_transformer.{k}"] = v
    weights["atom_attention_encoder.atom_to_token_trans.0"] = [{
        "weight":
        state_dict[f"{layer_path}.atom_to_token_trans.0.weight"],
        "bias":
        None
    }]
    weights_biases_path = {
        "embed_atom_features":
        (f"{layer_path}.embed_atom_features.weight", None),
        "embed_atompair_ref_pos":
        (f"{layer_path}.embed_atompair_ref_pos.weight", None),
        "embed_atompair_ref_dist":
        (f"{layer_path}.embed_atompair_ref_dist.weight", None),
        "embed_atompair_mask":
        (f"{layer_path}.embed_atompair_mask.weight", None),
        "c_to_p_trans_k.1": (f"{layer_path}.c_to_p_trans_k.1.weight", None),
        "c_to_p_trans_q.1": (f"{layer_path}.c_to_p_trans_q.1.weight", None),
        "p_mlp.1": (f"{layer_path}.p_mlp.1.weight", None),
        "p_mlp.3": (f"{layer_path}.p_mlp.3.weight", None),
        "p_mlp.5": (f"{layer_path}.p_mlp.5.weight", None),
    }
    for name, (weights_path, bias_path) in weights_biases_path.items():
        weights[f"atom_embedding.{name}"] = [{
            "weight":
            state_dict[weights_path],
            "bias":
            state_dict[bias_path] if bias_path is not None else None
        }]

    assert config.diffusion_transformer.version == "v2", "Need version 2 here"
    for i in range(config.diffusion_transformer.num_blocks):
        weights[f"atom_enc_proj_z.{i}.0"] = [{
            "weight":
            state_dict[
                f"{layer_path}.atom_encoder.diffusion_transformer.layers.{i}.pair_bias_attn.proj_z.0.weight"],
            "bias":
            state_dict[
                f"{layer_path}.atom_encoder.diffusion_transformer.layers.{i}.pair_bias_attn.proj_z.0.bias"],
        }]
        weights[f"atom_enc_proj_z.{i}.1"] = [{
            "weight":
            state_dict[
                f"{layer_path}.atom_encoder.diffusion_transformer.layers.{i}.pair_bias_attn.proj_z.1.weight"],
            "bias":
            None,
        }]
    return weights


def convert_hf_structure_module_torch(config: BaseConfig,
                                      local_checkpoint: str = None,
                                      model_name: str = "boltz-1",
                                      weights: dict = None,
                                      **kwargs):
    """
    Convert a atom diffusion model from a Hugging Face checkpoint to a PyTorch model weights.
    Args:
        config: The configuration for the atom diffusion module. Boltz1Config.structure_module.atom_diffusion
        local_checkpoint: The directory to load the checkpoint from. If local_checkpoint is None, the function will load from HuggingFace
        model_name: The name of the model to load.
        weights: The weights to load. If weights is None, the function will load from HuggingFace
        kwargs: Additional arguments for the conversion.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    weights = {}
    layer_path = "structure_module.out_token_feat_update"
    atom_diffusion_config = config.atom_diffusion

    # Weights for out_token_feat_update
    weights["out_token_feat_update.norm_next"] = [{
        "weight":
        state_dict[f"{layer_path}.norm_next.weight"],
        "bias":
        state_dict[f"{layer_path}.norm_next.bias"]
    }]
    weights["out_token_feat_update.norm_fourier"] = [{
        "weight":
        state_dict[f"{layer_path}.norm_fourier.weight"],
        "bias":
        state_dict[f"{layer_path}.norm_fourier.bias"]
    }]
    weights["out_token_feat_update.fourier_embed.proj"] = [{
        "weight":
        state_dict[f"{layer_path}.fourier_embed.proj.weight"],
        "bias":
        state_dict[f"{layer_path}.fourier_embed.proj.bias"]
    }]
    # weight for conditioned transition block
    weights[f"out_token_feat_update.transition_block.adaln.a_norm"] = [{
        "weight":
        torch.ones([2 * atom_diffusion_config.token_s],
                   dtype=str_dtype_to_torch(atom_diffusion_config.dtype))
    }]
    weights[f"out_token_feat_update.transition_block.adaln.s_norm"] = [{
        "weight":
        state_dict[f"{layer_path}.transition_block.adaln.s_norm.weight"]
    }]
    weights[
        f"out_token_feat_update.transition_block.adaln.fused_s_scale_s_bias"] = [
            {
                "weight":
                state_dict[
                    f"{layer_path}.transition_block.adaln.s_scale.weight"],
                "bias":
                state_dict[f"{layer_path}.transition_block.adaln.s_scale.bias"]
            }, {
                "weight":
                state_dict[
                    f"{layer_path}.transition_block.adaln.s_bias.weight"],
                "bias":
                torch.zeros([2 * atom_diffusion_config.token_s],
                            dtype=str_dtype_to_torch(
                                atom_diffusion_config.dtype))
            }
        ]
    swish_gate_weight = state_dict[
        f"{layer_path}.transition_block.swish_gate.0.weight"]
    swish_gate_weight_0, swish_gate_weight_1 = torch.chunk(swish_gate_weight,
                                                           2,
                                                           dim=0)
    weights[f"out_token_feat_update.transition_block.fused_swl_a_to_b"] = [{
        "weight":
        swish_gate_weight_0,
    }, {
        "weight":
        swish_gate_weight_1,
    }, {
        "weight":
        state_dict[f"{layer_path}.transition_block.a_to_b.weight"],
    }]
    weights[f"out_token_feat_update.transition_block.b_to_a"] = [{
        "weight":
        state_dict[f"{layer_path}.transition_block.b_to_a.weight"],
    }]
    weights[f"out_token_feat_update.transition_block.output_projection"] = [{
        "weight":
        state_dict[
            f"{layer_path}.transition_block.output_projection.0.weight"],
        "bias":
        state_dict[f"{layer_path}.transition_block.output_projection.0.bias"]
    }]

    # Weight for score model
    score_model_config = config.score_model
    layer_path = "structure_module.score_model"
    # Load for single conditioner, s_to_a_linear and a_norm
    weights["score_model.single_conditioner.norm_single"] = [{
        "weight":
        state_dict[f"{layer_path}.single_conditioner.norm_single.weight"],
        "bias":
        state_dict[f"{layer_path}.single_conditioner.norm_single.bias"]
    }]

    weights["score_model.single_conditioner.single_embed"] = [{
        "weight":
        state_dict[f"{layer_path}.single_conditioner.single_embed.weight"],
        "bias":
        state_dict.get(f"{layer_path}.single_conditioner.single_embed.bias",
                       None)
    }]
    weights["score_model.single_conditioner.fourier_embed.proj"] = [{
        "weight":
        state_dict[
            f"{layer_path}.single_conditioner.fourier_embed.proj.weight"],
        "bias":
        state_dict.get(
            f"{layer_path}.single_conditioner.fourier_embed.proj.bias", None)
    }]
    weights["score_model.single_conditioner.norm_fourier"] = [{
        "weight":
        state_dict[f"{layer_path}.single_conditioner.norm_fourier.weight"],
        "bias":
        state_dict[f"{layer_path}.single_conditioner.norm_fourier.bias"]
    }]
    weights["score_model.single_conditioner.fourier_to_single"] = [{
        "weight":
        state_dict[
            f"{layer_path}.single_conditioner.fourier_to_single.weight"],
        "bias":
        state_dict.get(
            f"{layer_path}.single_conditioner.fourier_to_single.bias", None)
    }]
    for i in range(score_model_config.conditioning_transition_layers):
        weights[f"score_model.single_conditioner.transitions.{i}.norm"] = [{
            "weight":
            state_dict[
                f"{layer_path}.single_conditioner.transitions.{i}.norm.weight"],
            "bias":
            state_dict[
                f"{layer_path}.single_conditioner.transitions.{i}.norm.bias"]
        }]
        weights[f"score_model.single_conditioner.transitions.{i}.fused_fc2_fc1"] = [{
            "weight":
            state_dict[
                f"{layer_path}.single_conditioner.transitions.{i}.fc2.weight"],
            "bias":
            state_dict.get(
                f"{layer_path}.single_conditioner.transitions.{i}.fc2.bias",
                None)
        }, {
            "weight":
            state_dict[
                f"{layer_path}.single_conditioner.transitions.{i}.fc1.weight"],
            "bias":
            state_dict.get(
                f"{layer_path}.single_conditioner.transitions.{i}.fc1.bias",
                None)
        }]
        weights[f"score_model.single_conditioner.transitions.{i}.fc3"] = [{
            'weight':
            state_dict[
                f"{layer_path}.single_conditioner.transitions.{i}.fc3.weight"],
        }]
    weights["score_model.s_to_a_linear.0"] = [{
        "weight":
        state_dict[f"{layer_path}.s_to_a_linear.0.weight"],
        "bias":
        state_dict[f"{layer_path}.s_to_a_linear.0.bias"]
    }]
    weights["score_model.s_to_a_linear.1"] = [{
        "weight":
        state_dict[f"{layer_path}.s_to_a_linear.1.weight"],
        "bias":
        None
    }]
    weights["score_model.a_norm"] = [{
        "weight":
        state_dict[f"{layer_path}.a_norm.weight"],
        "bias":
        state_dict[f"{layer_path}.a_norm.bias"]
    }]
    # Load for atom attention encoder
    DiT_weights = convert_hf_diffusion_transformer_torch(
        score_model_config.atom_encoder,
        local_checkpoint=local_checkpoint,
        model_name=model_name,
        weights=state_dict,
        prefix=
        f"{layer_path}.atom_attention_encoder.atom_encoder.diffusion_transformer."
    )
    for k, v in DiT_weights.items():
        weights[
            f"score_model.atom_attention_encoder.atom_encoder.diffusion_transformer.{k}"] = v
    weights["score_model.atom_attention_encoder.atom_to_token_trans.0"] = [{
        "weight":
        state_dict[
            f"{layer_path}.atom_attention_encoder.atom_to_token_trans.0.weight"],
        "bias":
        None
    }]
    weights["score_model.atom_attention_encoder.r_to_q_trans"] = [{
        "weight":
        state_dict[f"{layer_path}.atom_attention_encoder.r_to_q_trans.weight"],
        "bias":
        None
    }]
    # Load for atom attention decoder
    DiT_weights = convert_hf_diffusion_transformer_torch(
        score_model_config.atom_decoder,
        local_checkpoint=local_checkpoint,
        model_name=model_name,
        weights=state_dict,
        prefix=
        f"{layer_path}.atom_attention_decoder.atom_decoder.diffusion_transformer."
    )
    for k, v in DiT_weights.items():
        weights[
            f"score_model.atom_attention_decoder.atom_decoder.diffusion_transformer.{k}"] = v

    weights["score_model.atom_attention_decoder.a_to_q_trans"] = [{
        "weight":
        state_dict[f"{layer_path}.atom_attention_decoder.a_to_q_trans.weight"],
        "bias":
        None
    }]
    weights["score_model.atom_attention_decoder.atom_feat_to_atom_pos_update.0"] = [{
        "weight":
        state_dict[
            f"{layer_path}.atom_attention_decoder.atom_feat_to_atom_pos_update.0.weight"],
        "bias":
        state_dict[
            f"{layer_path}.atom_attention_decoder.atom_feat_to_atom_pos_update.0.bias"]
    }]
    weights["score_model.atom_attention_decoder.atom_feat_to_atom_pos_update.1"] = [{
        "weight":
        state_dict[
            f"{layer_path}.atom_attention_decoder.atom_feat_to_atom_pos_update.1.weight"],
        "bias":
        None
    }]

    # Load for token transformer
    DiT_weights = convert_hf_diffusion_transformer_torch(
        score_model_config.token_transformer,
        local_checkpoint=local_checkpoint,
        model_name=model_name,
        weights=state_dict,
        prefix=f"{layer_path}.token_transformer.")
    for k, v in DiT_weights.items():
        weights[f"score_model.token_transformer.{k}"] = v

    return weights


def convert_hf_diffusion_conditioning_torch(config: BaseConfig,
                                            local_checkpoint: str = None,
                                            model_name: str = "boltz-1",
                                            weights: dict = None,
                                            **kwargs) -> dict:
    """
    Convert a diffusion conditioning model from a Hugging Face checkpoint to a PyTorch model weights.
    Args:
        config: The configuration for the diffusion conditioning module. Boltz1Config.structure_module.score_model
        local_checkpoint: The directory to load the checkpoint from. If local_checkpoint is None, the function will load from HuggingFace
        model_name: The name of the model to load.
        weights: The weights to load. If weights is None, the function will load from HuggingFace
        kwargs: Additional arguments for the conversion.
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    tbnm_state_dict = {}
    prefix = "structure_module.score_model."
    module_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            module_state_dict[k.replace(prefix, "")] = v
    all_keys = len([
        k for k in state_dict.keys()
        if k.startswith(f"{prefix}pairwise_conditioner.transitions")
    ])
    keys_layer_0 = len([
        k for k in state_dict.keys()
        if k.startswith(f"{prefix}pairwise_conditioner.transitions.0.")
    ])

    num_transitions = all_keys // keys_layer_0
    tbnm_state_dict["pairwise_conditioner.init_proj_norm"] = [{
        "weight":
        module_state_dict[
            "pairwise_conditioner.dim_pairwise_init_proj.0.weight"],
        "bias":
        module_state_dict["pairwise_conditioner.dim_pairwise_init_proj.0.bias"]
    }]
    tbnm_state_dict["pairwise_conditioner.init_proj_linear"] = [{
        "weight":
        module_state_dict[
            "pairwise_conditioner.dim_pairwise_init_proj.1.weight"]
    }]
    for i in range(num_transitions):
        tbnm_state_dict[f"pairwise_conditioner.transitions.{i}.norm"] = [{
            'weight':
            module_state_dict[
                f"pairwise_conditioner.transitions.{i}.norm.weight"],
            'bias':
            module_state_dict[
                f"pairwise_conditioner.transitions.{i}.norm.bias"]
        }]
        tbnm_state_dict[
            f"pairwise_conditioner.transitions.{i}.fused_fc2_fc1"] = [{
                'weight':
                module_state_dict[
                    f"pairwise_conditioner.transitions.{i}.fc2.weight"],
            }, {
                'weight':
                module_state_dict[
                    f"pairwise_conditioner.transitions.{i}.fc1.weight"],
            }]
        tbnm_state_dict[f"pairwise_conditioner.transitions.{i}.fc3"] = [{
            'weight':
            module_state_dict[
                f"pairwise_conditioner.transitions.{i}.fc3.weight"],
        }]

    layer_path = f"atom_attention_encoder"
    weights_biases_path = {
        "embed_atom_features":
        (f"{layer_path}.embed_atom_features.weight", None),
        "embed_atompair_ref_pos":
        (f"{layer_path}.embed_atompair_ref_pos.weight", None),
        "embed_atompair_ref_dist":
        (f"{layer_path}.embed_atompair_ref_dist.weight", None),
        "embed_atompair_mask":
        (f"{layer_path}.embed_atompair_mask.weight", None),
        "c_to_p_trans_k.1": (f"{layer_path}.c_to_p_trans_k.1.weight", None),
        "c_to_p_trans_q.1": (f"{layer_path}.c_to_p_trans_q.1.weight", None),
        "p_mlp.1": (f"{layer_path}.p_mlp.1.weight", None),
        "p_mlp.3": (f"{layer_path}.p_mlp.3.weight", None),
        "p_mlp.5": (f"{layer_path}.p_mlp.5.weight", None),
        "s_to_c_trans.0": (f"{layer_path}.s_to_c_trans.0.weight",
                           f"{layer_path}.s_to_c_trans.0.bias"),
        "z_to_p_trans.0": (f"{layer_path}.z_to_p_trans.0.weight",
                           f"{layer_path}.z_to_p_trans.0.bias"),
        "s_to_c_trans.1": (f"{layer_path}.s_to_c_trans.1.weight", None),
        "z_to_p_trans.1": (f"{layer_path}.z_to_p_trans.1.weight", None),
    }

    for name, (weights_path, bias_path) in weights_biases_path.items():
        tbnm_state_dict[f"atom_embedding.{name}"] = [{
            "weight":
            module_state_dict[weights_path],
            "bias":
            module_state_dict[bias_path] if bias_path is not None else None
        }]

    # Add weights for computing biases
    layer_path = "atom_attention_encoder.atom_encoder"
    for i in range(config.atom_encoder.num_blocks):
        tbnm_state_dict[f"atom_enc_proj_z.{i}.0"] = [{
            "weight":
            module_state_dict[
                f"{layer_path}.diffusion_transformer.layers.{i}.pair_bias_attn.proj_z.0.weight"],
            "bias":
            module_state_dict[
                f"{layer_path}.diffusion_transformer.layers.{i}.pair_bias_attn.proj_z.0.bias"]
        }]
        tbnm_state_dict[f"atom_enc_proj_z.{i}.1"] = [{
            "weight":
            module_state_dict[
                f"{layer_path}.diffusion_transformer.layers.{i}.pair_bias_attn.proj_z.1.weight"],
            "bias":
            None,
        }]
    layer_path = "token_transformer"
    for i in range(config.token_transformer.num_blocks):
        tbnm_state_dict[f"token_trans_proj_z.{i}.0"] = [{
            "weight":
            module_state_dict[
                f"{layer_path}.layers.{i}.pair_bias_attn.proj_z.0.weight"],
            "bias":
            module_state_dict[
                f"{layer_path}.layers.{i}.pair_bias_attn.proj_z.0.bias"]
        }]
        tbnm_state_dict[f"token_trans_proj_z.{i}.1"] = [{
            "weight":
            module_state_dict[
                f"{layer_path}.layers.{i}.pair_bias_attn.proj_z.1.weight"],
            "bias":
            None,
        }]
    layer_path = "atom_attention_decoder.atom_decoder"
    for i in range(config.atom_decoder.num_blocks):
        tbnm_state_dict[f"atom_dec_proj_z.{i}.0"] = [{
            "weight":
            module_state_dict[
                f"{layer_path}.diffusion_transformer.layers.{i}.pair_bias_attn.proj_z.0.weight"],
            "bias":
            module_state_dict[
                f"{layer_path}.diffusion_transformer.layers.{i}.pair_bias_attn.proj_z.0.bias"]
        }]
        tbnm_state_dict[f"atom_dec_proj_z.{i}.1"] = [{
            "weight":
            module_state_dict[
                f"{layer_path}.diffusion_transformer.layers.{i}.pair_bias_attn.proj_z.1.weight"],
            "bias":
            None,
        }]
    return tbnm_state_dict


def convert_hf_confidence_torch(config: BaseConfig,
                                local_checkpoint: str = None,
                                model_name: str = "boltz-1",
                                weights: dict = None,
                                **kwargs) -> dict:
    """
    Convert a confidence model from a Hugging Face checkpoint to a PyTorch model weights.
    Args:
        config: The configuration for the confidence module.
        local_checkpoint: The directory to load the checkpoint from.
        model_name: The name of the model to load.
        weights: The weights to load. If weights is None, the function will load from HuggingFace
        kwargs: Additional arguments for the conversion.
    Returns:
        dict: [
           "pairformer": dict,
           "msa_module": dict,
           "heads": dict,
           ...
        ]
    """
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    tbnm_state_dict = {}
    prefix = "confidence_module."
    tbnm_state_dict["dist_bin_pairwise_embed"] = [{
        "weight":
        state_dict[f"{prefix}dist_bin_pairwise_embed.weight"],
        "bias":
        None
    }]
    tbnm_state_dict["s_diffusion_norm"] = [{
        "weight":
        state_dict[f"{prefix}s_diffusion_norm.weight"],
        "bias":
        state_dict.get(f"{prefix}s_diffusion_norm.bias", None)
    }]
    tbnm_state_dict["s_diffusion_to_s"] = [{
        "weight":
        state_dict[f"{prefix}s_diffusion_to_s.weight"],
        "bias":
        state_dict.get(f"{prefix}s_diffusion_to_s.bias", None)
    }]
    tbnm_state_dict["s_to_z"] = [{
        "weight":
        state_dict[f"{prefix}s_to_z.weight"],
        "bias":
        state_dict.get(f"{prefix}s_to_z.bias", None)
    }]
    tbnm_state_dict["s_to_z_transpose"] = [{
        "weight":
        state_dict[f"{prefix}s_to_z_transpose.weight"],
        "bias":
        state_dict.get(f"{prefix}s_to_z_transpose.bias", None)
    }]
    if config.add_s_to_z_prod:
        tbnm_state_dict["s_to_z_prod_in1"] = [{
            "weight":
            state_dict[f"{prefix}s_to_z_prod_in1.weight"],
            "bias":
            state_dict.get(f"{prefix}s_to_z_prod_in1.bias", None)
        }]
        tbnm_state_dict["s_to_z_prod_in2"] = [{
            "weight":
            state_dict[f"{prefix}s_to_z_prod_in2.weight"],
            "bias":
            state_dict.get(f"{prefix}s_to_z_prod_in2.bias", None)
        }]
        tbnm_state_dict["s_to_z_prod_out"] = [{
            "weight":
            state_dict[f"{prefix}s_to_z_prod_out.weight"],
            "bias":
            state_dict.get(f"{prefix}s_to_z_prod_out.bias", None)
        }]
    tbnm_state_dict["s_init"] = [{
        "weight":
        state_dict[f"{prefix}s_init.weight"],
        "bias":
        state_dict.get(f"{prefix}s_init.bias", None)
    }]
    tbnm_state_dict["z_init_1"] = [{
        "weight":
        state_dict[f"{prefix}z_init_1.weight"],
        "bias":
        state_dict.get(f"{prefix}z_init_1.bias", None)
    }]
    tbnm_state_dict["z_init_2"] = [{
        "weight":
        state_dict[f"{prefix}z_init_2.weight"],
        "bias":
        state_dict.get(f"{prefix}z_init_2.bias", None)
    }]
    input_embedder_weights = convert_hf_input_embedder_torch(
        config=config.input_embedder,
        local_checkpoint=local_checkpoint,
        model_name=model_name,
        weights=state_dict,
        prefix=f"{prefix}input_embedder.")
    # tbnm_state_dict["input_embedder"] = input_embedder_weights
    for k, v in input_embedder_weights.items():
        tbnm_state_dict[f"input_embedder.{k}"] = v

    tbnm_state_dict["rel_pos.linear"] = [{
        "weight":
        state_dict[f"{prefix}rel_pos.linear_layer.weight"],
        "bias":
        state_dict.get(f"{prefix}rel_pos.linear_layer.bias", None)
    }]
    tbnm_state_dict["token_bonds"] = [{
        "weight":
        state_dict[f"{prefix}token_bonds.weight"],
        "bias":
        state_dict.get(f"{prefix}token_bonds.bias", None)
    }]
    tbnm_state_dict["s_norm"] = [{
        "weight":
        state_dict[f"{prefix}s_norm.weight"],
        "bias":
        state_dict.get(f"{prefix}s_norm.bias", None)
    }]
    tbnm_state_dict["z_norm"] = [{
        "weight":
        state_dict[f"{prefix}z_norm.weight"],
        "bias":
        state_dict.get(f"{prefix}z_norm.bias", None)
    }]
    tbnm_state_dict["s_recycle"] = [{
        "weight":
        state_dict[f"{prefix}s_recycle.weight"],
        "bias":
        state_dict.get(f"{prefix}s_recycle.bias", None)
    }]
    tbnm_state_dict["z_recycle"] = [{
        "weight":
        state_dict[f"{prefix}z_recycle.weight"],
        "bias":
        state_dict.get(f"{prefix}z_recycle.bias", None)
    }]

    pairformer_weights = convert_hf_pairformer_torch(
        config=config.pairformer,
        local_checkpoint=local_checkpoint,
        model_name=model_name,
        weights=state_dict,
        pairformer_type="confidence",
    )
    # tbnm_state_dict["pairformer"] = pairformer_weights
    for k, v in pairformer_weights.items():
        tbnm_state_dict[f"pairformer_module.{k}"] = v
    msa_module_weights = convert_hf_msa_module_torch(
        config=config.msa_module,
        local_checkpoint=local_checkpoint,
        model_name=model_name,
        weights=state_dict,
        prefix=f"{prefix}msa_module.")
    # tbnm_state_dict["msa_module"] = msa_module_weights
    for k, v in msa_module_weights.items():
        tbnm_state_dict[f"msa_module.{k}"] = v
    tbnm_state_dict["final_s_norm"] = [{
        "weight":
        state_dict[f"{prefix}final_s_norm.weight"],
        "bias":
        state_dict.get(f"{prefix}final_s_norm.bias", None)
    }]
    tbnm_state_dict["final_z_norm"] = [{
        "weight":
        state_dict[f"{prefix}final_z_norm.weight"],
        "bias":
        state_dict.get(f"{prefix}final_z_norm.bias", None)
    }]

    # Convert for heads
    head_weights = {}
    head_prefix = f"{prefix}confidence_heads."

    head_weights["to_pde_logits"] = [{
        "weight":
        state_dict[f"{head_prefix}to_pde_logits.weight"],
        "bias":
        state_dict.get(f"{head_prefix}to_pde_logits.bias", None)
    }]
    head_weights["to_plddt_logits"] = [{
        "weight":
        state_dict[f"{head_prefix}to_plddt_logits.weight"],
        "bias":
        state_dict.get(f"{head_prefix}to_plddt_logits.bias", None)
    }]
    head_weights["to_resolved_logits"] = [{
        "weight":
        state_dict[f"{head_prefix}to_resolved_logits.weight"],
        "bias":
        state_dict.get(f"{head_prefix}to_resolved_logits.bias", None)
    }]
    if config.heads.compute_pae:
        head_weights["to_pae_logits"] = [{
            "weight":
            state_dict[f"{head_prefix}to_pae_logits.weight"],
            "bias":
            state_dict.get(f"{head_prefix}to_pae_logits.bias", None)
        }]
    # tbnm_state_dict["heads"] = head_weights
    for k, v in head_weights.items():
        tbnm_state_dict[f"confidence_heads.{k}"] = v
    return tbnm_state_dict
