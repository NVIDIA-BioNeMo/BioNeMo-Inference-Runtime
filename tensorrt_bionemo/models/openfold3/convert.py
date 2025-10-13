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

from collections import OrderedDict

import torch
from tensorrt_llm import str_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.models.convert_utils import split

from tensorrt_bionemo.hubs import load_weights
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.boltz1.convert import (
    get_adaln_weights, get_output_projection_weights, get_pairwise_attn_weights,
    get_post_norm_weights, get_transition_weights, get_tri_attn_node_weights,
    get_tri_mul_node_weights)
from tensorrt_bionemo.models.openfold3.configs import (PairformerConfig,
                                                       TokenTransformerConfig)


def convert_hf_pairformer(config: PairformerConfig,
                          mapping: Mapping,
                          local_checkpoint: str = None,
                          model_name: str = "openfold3"):
    state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    assert state_dict is not None
    # Get all weights relate to pairformer stack
    # This procedure maps the weights from the OF3 to Boltz1
    # So we can reuse the Boltz1 convert function
    pairformer_state_dict = {}
    prefix = "pairformer_stack"
    for name, param in state_dict.items():
        if name.startswith(prefix):
            oringal_name = name
            block_number = int(name.split(".")[2])
            name = name.replace(prefix, "pairformer_module")
            name = name.replace("blocks.", "layers.")
            if "attn_pair_bias" in name:
                name = name.replace("attn_pair_bias", "attention")
                name = name.replace("layer_norm_a", "norm_s")
                name = name.replace("layer_norm_z", "proj_z.0")
                name = name.replace("linear_z", "proj_z.1")
                name = name.replace("mha.linear_q", "proj_q")
                name = name.replace("mha.linear_k", "proj_k")
                name = name.replace("mha.linear_v", "proj_v")
                name = name.replace("mha.linear_o", "proj_o")
                name = name.replace("mha.linear_g", "proj_g")
                pairformer_state_dict[name] = param
                continue

            if "single_transition" in name:
                name = name.replace("single_transition", "transition_s")
                name = name.replace("layer_norm", "norm")
                name = name.replace("swiglu.linear_a", "fc1")
                name = name.replace("swiglu.linear_b", "fc2")
                name = name.replace("linear_out", "fc3")
                pairformer_state_dict[name] = param
                continue

            if "pair_stack" in name:
                name = name.replace("pair_stack.", "")
                if "tri_att_start" in name or "tri_att_end" in name:
                    name = name.replace("linear_z", "linear")
                    pairformer_state_dict[name] = param
                    continue
                if "tri_mul_out" in name or "tri_mul_in" in name:
                    tri_mul_type = "tri_mul_out" if "tri_mul_out" in name else "tri_mul_in"
                    if "linear_a_p" in name or "linear_b_p" in name:
                        name = name.replace(".linear_a_p.weight", "")
                        name = name.replace(".linear_b_p.weight", "")
                        p_in_name = name + ".p_in.weight"
                        if p_in_name not in pairformer_state_dict:
                            p0 = state_dict[
                                f"{prefix}.blocks.{block_number}.pair_stack.{tri_mul_type}.linear_a_p.weight"]
                            p1 = state_dict[
                                f"{prefix}.blocks.{block_number}.pair_stack.{tri_mul_type}.linear_b_p.weight"]
                            pairformer_state_dict[p_in_name] = torch.cat(
                                [p0, p1], dim=0)
                        continue
                    if "linear_a_g" in name or "linear_b_g" in name:
                        name = name.replace(".linear_a_g.weight", "")
                        name = name.replace(".linear_b_g.weight", "")
                        g_in_name = name + ".g_in.weight"
                        if g_in_name not in pairformer_state_dict:
                            g0 = state_dict[
                                f"{prefix}.blocks.{block_number}.pair_stack.{tri_mul_type}.linear_a_g.weight"]
                            g1 = state_dict[
                                f"{prefix}.blocks.{block_number}.pair_stack.{tri_mul_type}.linear_b_g.weight"]
                            pairformer_state_dict[g_in_name] = torch.cat(
                                [g0, g1], dim=0)
                        continue
                    name = name.replace("linear_z", "p_out")
                    name = name.replace("linear_g", "g_out")
                    name = name.replace("layer_norm_in", "norm_in")
                    name = name.replace("layer_norm_out", "norm_out")
                    pairformer_state_dict[name] = param
                    continue
                if "pair_transition" in name:
                    name = name.replace("pair_transition", "transition_z")
                    name = name.replace("layer_norm", "norm")
                    name = name.replace("swiglu.linear_a", "fc1")
                    name = name.replace("swiglu.linear_b", "fc2")
                    name = name.replace("linear_out", "fc3")
                    pairformer_state_dict[name] = param
                    continue
                logger.warning(f"Miss converting the weight: {oringal_name}")

    mapping = mapping if mapping is not None else Mapping()
    prefix = "pairformer_module.layers"
    tbm_prefix = "layers"
    weights = {}
    for i in range(config.num_blocks):
        layer_prefix = f"{prefix}.{i}"
        layer_tbm_prefix = f"{tbm_prefix}.{i}"
        if not config.no_update_s:
            weights.update(
                get_pairwise_attn_weights(mapping,
                                          pairformer_state_dict,
                                          f"{layer_prefix}.attention",
                                          f"{layer_tbm_prefix}.attention",
                                          config.max_attention_pairwise_tp_size,
                                          config.num_heads,
                                          dtype=config.dtype))
        weights.update(
            get_tri_attn_node_weights(mapping,
                                      pairformer_state_dict,
                                      f"{layer_prefix}.tri_att_start",
                                      f"{layer_tbm_prefix}.tri_attn_start",
                                      dtype=config.dtype))
        weights.update(
            get_tri_attn_node_weights(mapping,
                                      pairformer_state_dict,
                                      f"{layer_prefix}.tri_att_end",
                                      f"{layer_tbm_prefix}.tri_attn_end",
                                      dtype=config.dtype))
        weights.update(
            get_tri_mul_node_weights(mapping,
                                     pairformer_state_dict,
                                     f"{layer_prefix}.tri_mul_out",
                                     f"{layer_tbm_prefix}.tri_mul_out",
                                     config.max_tri_mul_tp_size,
                                     dtype=config.dtype))
        weights.update(
            get_tri_mul_node_weights(mapping,
                                     pairformer_state_dict,
                                     f"{layer_prefix}.tri_mul_in",
                                     f"{layer_tbm_prefix}.tri_mul_in",
                                     config.max_tri_mul_tp_size,
                                     dtype=config.dtype))
        if not config.no_update_s:
            weights.update(
                get_transition_weights(mapping,
                                       pairformer_state_dict,
                                       f"{layer_prefix}.transition_s",
                                       f"{layer_tbm_prefix}.transition_s",
                                       config.max_transition_tp_size,
                                       config.token_s * 4,
                                       dtype=config.dtype))
        weights.update(
            get_transition_weights(mapping,
                                   pairformer_state_dict,
                                   f"{layer_prefix}.transition_z",
                                   f"{layer_tbm_prefix}.transition_z",
                                   config.max_transition_tp_size,
                                   config.token_z * 4,
                                   dtype=config.dtype))
    return weights


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

    # dim_inner = int(dim * expansion_factor)
    tp_size = mapping.tp_size
    tp_rank = mapping.tp_rank
    if tp_size > 1:
        swish_gate_weight = split(swish_gate_weight, tp_size, tp_rank, 0)
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


def convert_hf_token_transformer(config: TokenTransformerConfig,
                                 mapping: Mapping,
                                 local_checkpoint: str = None,
                                 model_name: str = "openfold3"):
    state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    assert state_dict is not None

    tktn_state_dict = OrderedDict()
    prefix = "sample_diffusion.diffusion_module.diffusion_transformer.blocks"
    replace_prefix = "token_transformer.layers"

    # Mapping the weights from OF3 to Boltz1, so we can reuse the Boltz1 convert functions
    for name, param in state_dict.items():
        if name.startswith(prefix):
            name = name.replace(prefix, replace_prefix)
            int(name.split(".")[2])
            if "attention_pair_bias" in name:
                # change name for AdaLN
                if "layer_norm_a" in name:
                    name = name.replace("attention_pair_bias.layer_norm_a",
                                        "adaln")
                    name = name.replace("layer_norm_s", "s_norm")
                    name = name.replace("linear_g", "s_scale")
                    name = name.replace("linear_s", "s_bias")
                    tktn_state_dict[name] = param
                    continue
                if "linear_ada_out" in name:
                    name = name.replace("attention_pair_bias.linear_ada_out",
                                        "output_projection.0")
                    tktn_state_dict[name] = param
                    continue
                if "mha" in name:
                    name = name.replace("attention_pair_bias.mha",
                                        "pair_bias_attn")
                    name = name.replace("linear_q", "proj_q")
                    name = name.replace("linear_k", "proj_k")
                    name = name.replace("linear_v", "proj_v")
                    name = name.replace("linear_o", "proj_o")
                    name = name.replace("linear_g", "proj_g")
                    tktn_state_dict[name] = param
                    continue
                if "layer_norm_z" in name or "linear_z" in name:
                    name = name.replace("attention_pair_bias.layer_norm_z",
                                        "pair_bias_attn.proj_z.0")
                    name = name.replace("attention_pair_bias.linear_z",
                                        "pair_bias_attn.proj_z.1")
                    tktn_state_dict[name] = param
                    continue
                logger.warning(f"Miss converting the weight: {name}")
            if "conditioned_transition" in name:
                if "layer_norm." in name:
                    name = name.replace("conditioned_transition.layer_norm",
                                        "transition.adaln")
                    name = name.replace("layer_norm_s", "s_norm")
                    name = name.replace("linear_g", "s_scale")
                    name = name.replace("linear_s", "s_bias")
                    tktn_state_dict[name] = param
                    continue
                if "swiglu" in name:
                    name = name.replace(
                        "conditioned_transition.swiglu.linear_a.weight",
                        "transition.swish_gate.0.weight")
                    name = name.replace(
                        "conditioned_transition.swiglu.linear_b.weight",
                        "transition.a_to_b.weight")
                    tktn_state_dict[name] = param
                    continue
                if "linear_out" in name or "linear_g" in name:
                    name = name.replace("conditioned_transition.linear_out",
                                        "transition.b_to_a")
                    name = name.replace("conditioned_transition.linear_g",
                                        "transition.output_projection.0")
                    tktn_state_dict[name] = param
                    continue
                logger.warning(f"Miss converting the weight: {name}")

    mapping = mapping if mapping is not None else Mapping()
    prefix = "token_transformer.layers"
    tbm_prefix = "layers"
    weights = {}

    logger.info(f"Loading weights for token transformer, dtype: {config.dtype}")
    for i in range(config.num_blocks):
        layer_prefix = f"{prefix}.{i}"
        layer_tbm_prefix = f"{tbm_prefix}.{i}"
        weights.update(
            get_adaln_weights(mapping,
                              tktn_state_dict,
                              f"{layer_prefix}.adaln",
                              f"{layer_tbm_prefix}.adaln",
                              config.dim,
                              config.dim_single_cond,
                              dtype=config.dtype))
        weights.update(
            get_pairwise_attn_weights(
                mapping,
                tktn_state_dict,
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
                tktn_state_dict,
                f"{layer_prefix}.transition",
                f"{layer_tbm_prefix}.transition",
                config.dim,
                config.dim_single_cond,
                expansion_factor=config.expansion_factor,
                dtype=config.dtype))
        weights.update(
            get_output_projection_weights(
                mapping,
                tktn_state_dict,
                f"{layer_prefix}.output_projection",
                f"{layer_tbm_prefix}.output_projection",
                dtype=config.dtype))
        weights.update(
            get_post_norm_weights(mapping,
                                  tktn_state_dict,
                                  f"{layer_prefix}.post_lnorm",
                                  f"{layer_tbm_prefix}.post_lnorm",
                                  dtype=config.dtype))

    return weights
