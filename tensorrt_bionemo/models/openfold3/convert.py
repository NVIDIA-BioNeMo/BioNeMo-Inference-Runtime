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
from tensorrt_llm_lite import str_dtype_to_torch
from tensorrt_llm_lite.logger import logger

from tensorrt_bionemo.configs import (BaseConfig, DiffusionTransformerConfig,
                                      PairformerConfig)
from tensorrt_bionemo.hubs import load_weights
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.boltz1.convert import (
    get_adaln_weights, get_output_projection_weights,
    get_pairwise_attn_weights, get_post_norm_weights, get_transition_weights,
    get_tri_attn_node_weights, get_tri_mul_node_weights)


def split(*args, **kwargs):
    # Do nothing: TRT for multiple gpus is deprecated
    pass


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
                get_pairwise_attn_weights(
                    mapping,
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


def convert_hf_pairformer_torch(config: PairformerConfig,
                                mapping: Mapping = None,
                                local_checkpoint: str = None,
                                model_name: str = "openfold3",
                                weights: dict = None):
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
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
    module_state_dict = {
        k.replace("pairformer_module.", ""): v
        for k, v in pairformer_state_dict.items()
    }
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
        output_projection_bias = split(output_projection_bias, tp_size,
                                       tp_rank, 0)

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


def convert_hf_diffusion_transformer(config: DiffusionTransformerConfig,
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
            # int(name.split(".")[2])
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

    logger.info(
        f"Loading weights for token transformer, dtype: {config.dtype}")
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


def _template_module_weight(tbnm_state_dict, module_state_dict, name):
    tbnm_state_dict[name] = [{
        "weight":
        module_state_dict[f"{name}.weight"],
        "bias":
        module_state_dict.get(f"{name}.bias", None)
    }]


def convert_hf_diffusion_transformer_torch(config: DiffusionTransformerConfig,
                                           mapping: Mapping = None,
                                           local_checkpoint: str = None,
                                           model_name: str = "openfold3",
                                           weights: dict = None,
                                           **kwargs):
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    assert state_dict is not None
    if "prefix" in kwargs:
        prefix = kwargs["prefix"]
    else:
        prefix = "sample_diffusion.diffusion_module.diffusion_transformer.blocks"
    replace_prefix = "layers"
    module_state_dict = {}

    # Mapping the weights from OF3 to Boltz1, so we can reuse the Boltz1 convert functions
    for name, param in state_dict.items():
        if name.startswith(prefix):
            name = name.replace(prefix, replace_prefix)
            if "attention_pair_bias" in name:
                # change name for AdaLN
                if config.use_separate_layer_norm:
                    if "layer_norm_a_q" in name:

                        name = name.replace(
                            "attention_pair_bias.layer_norm_a_q",
                            "pair_bias_attn.layer_norm_a_q")
                        name = name.replace("layer_norm_s.", "s_norm.")
                        name = name.replace("linear_g.", "s_scale.")
                        name = name.replace("linear_s.", "s_bias.")
                        module_state_dict[name] = param
                        continue
                    if "layer_norm_a_k" in name:

                        name = name.replace(
                            "attention_pair_bias.layer_norm_a_k",
                            "pair_bias_attn.layer_norm_a_k")
                        name = name.replace("layer_norm_a.", "a_norm.")
                        name = name.replace("layer_norm_s.", "s_norm.")
                        name = name.replace("linear_g.", "s_scale.")
                        name = name.replace("linear_s.", "s_bias.")
                        module_state_dict[name] = param
                        continue
                else:
                    if "layer_norm_a" in name:
                        name = name.replace("attention_pair_bias.layer_norm_a",
                                            "adaln")
                        name = name.replace("layer_norm_s", "s_norm")
                        name = name.replace("linear_g", "s_scale")
                        name = name.replace("linear_s", "s_bias")
                        module_state_dict[name] = param
                        continue

                if "linear_ada_out" in name:
                    name = name.replace("attention_pair_bias.linear_ada_out",
                                        "output_projection.0")
                    module_state_dict[name] = param
                    continue

                if "mha" in name:
                    name = name.replace("attention_pair_bias.mha",
                                        "pair_bias_attn")
                    name = name.replace("linear_q", "proj_q")
                    name = name.replace("linear_k", "proj_k")
                    name = name.replace("linear_v", "proj_v")
                    name = name.replace("linear_o", "proj_o")
                    name = name.replace("linear_g", "proj_g")
                    module_state_dict[name] = param
                    continue
                if "layer_norm_z" in name or "linear_z" in name:
                    if getattr(config, 'shared_pair_norm', False):
                        # shared_pair_norm: no per-block LayerNorm; linear_z
                        # maps directly to proj_z.0 (the only element).
                        if "linear_z" in name:
                            name = name.replace("attention_pair_bias.linear_z",
                                                "pair_bias_attn.proj_z.0")
                            module_state_dict[name] = param
                        # layer_norm_z is at transformer level, not per-block
                    else:
                        name = name.replace("attention_pair_bias.layer_norm_z",
                                            "pair_bias_attn.proj_z.0")
                        name = name.replace("attention_pair_bias.linear_z",
                                            "pair_bias_attn.proj_z.1")
                        module_state_dict[name] = param
                    continue
                logger.warning(f"Miss converting the weight: {name}")
            if "conditioned_transition" in name:
                if "layer_norm." in name:
                    name = name.replace("conditioned_transition.layer_norm",
                                        "transition.adaln")
                    name = name.replace("layer_norm_s", "s_norm")
                    name = name.replace("linear_g", "s_scale")
                    name = name.replace("linear_s", "s_bias")
                    module_state_dict[name] = param
                    continue
                if "swiglu" in name:
                    name = name.replace(
                        "conditioned_transition.swiglu.linear_a.weight",
                        "transition.swish_gate.0.weight")
                    name = name.replace(
                        "conditioned_transition.swiglu.linear_b.weight",
                        "transition.a_to_b.weight")
                    module_state_dict[name] = param
                    continue
                if "linear_out" in name or "linear_g" in name:
                    name = name.replace("conditioned_transition.linear_out",
                                        "transition.b_to_a")
                    name = name.replace("conditioned_transition.linear_g",
                                        "transition.output_projection.0")
                    module_state_dict[name] = param
                    continue

    tbnm_state_dict = {}
    dim = module_state_dict[f"layers.0.pair_bias_attn.proj_q.weight"].shape[0]
    dtype = module_state_dict[f"layers.0.pair_bias_attn.proj_q.weight"].dtype

    for i in range(config.num_blocks):
        if not config.use_separate_layer_norm:
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
        else:
            tbnm_state_dict[
                f"layers.{i}.pair_bias_attn.layer_norm_a_q.a_norm"] = [{
                    "weight":
                    torch.ones([dim], dtype=dtype)
                }]
            _template_module_weight(
                tbnm_state_dict, module_state_dict,
                f"layers.{i}.pair_bias_attn.layer_norm_a_q.s_norm")

            tbnm_state_dict[
                f"layers.{i}.pair_bias_attn.layer_norm_a_q.fused_s_scale_s_bias"] = [{
                    "weight":
                    module_state_dict[
                        f"layers.{i}.pair_bias_attn.layer_norm_a_q.s_scale.weight"],
                    "bias":
                    module_state_dict[
                        f"layers.{i}.pair_bias_attn.layer_norm_a_q.s_scale.bias"]
                }, {
                    "weight":
                    module_state_dict[
                        f"layers.{i}.pair_bias_attn.layer_norm_a_q.s_bias.weight"],
                    "bias":
                    torch.zeros([dim], dtype=dtype)
                }]

            tbnm_state_dict[
                f"layers.{i}.pair_bias_attn.layer_norm_a_k.a_norm"] = [{
                    "weight":
                    torch.ones([dim], dtype=dtype)
                }]
            _template_module_weight(
                tbnm_state_dict, module_state_dict,
                f"layers.{i}.pair_bias_attn.layer_norm_a_k.s_norm")

            tbnm_state_dict[
                f"layers.{i}.pair_bias_attn.layer_norm_a_k.fused_s_scale_s_bias"] = [{
                    "weight":
                    module_state_dict[
                        f"layers.{i}.pair_bias_attn.layer_norm_a_k.s_scale.weight"],
                    "bias":
                    module_state_dict[
                        f"layers.{i}.pair_bias_attn.layer_norm_a_k.s_scale.bias"]
                }, {
                    "weight":
                    module_state_dict[
                        f"layers.{i}.pair_bias_attn.layer_norm_a_k.s_bias.weight"],
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

        if getattr(config, 'shared_pair_norm', False):
            # proj_z has only the Linear (index 0); layer_norm_z is shared
            # and lives at the transformer level, not per-block.
            if f"layers.{i}.pair_bias_attn.proj_z.0.weight" in module_state_dict:
                tbnm_state_dict[f"layers.{i}.pair_bias_attn.proj_z.0"] = [{
                    'weight':
                    module_state_dict[
                        f"layers.{i}.pair_bias_attn.proj_z.0.weight"],
                }]
        elif f"layers.{i}.pair_bias_attn.proj_z.0.weight" in module_state_dict:  # v2
            tbnm_state_dict[f"layers.{i}.pair_bias_attn.proj_z.0"] = [{
                'weight':
                module_state_dict[f"layers.{i}.pair_bias_attn.proj_z.0.weight"]
                # 'bias':
                # module_state_dict[f"layers.{i}.pair_bias_attn.proj_z.0.bias"]
            }]
            tbnm_state_dict[f"layers.{i}.pair_bias_attn.proj_z.1"] = [{
                'weight':
                module_state_dict[
                    f"layers.{i}.pair_bias_attn.proj_z.1.weight"],
            }]
        elif f"layers.{i}.pair_bias_attn.proj_z.1.weight" in module_state_dict:  # v1
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
        swish_gate_weight_0 = swish_gate_weight
        tbnm_state_dict[f"layers.{i}.transition.fused_swl_a_to_b"] = [{
            "weight":
            module_state_dict[f"layers.{i}.transition.a_to_b.weight"],
        }, {
            "weight":
            swish_gate_weight_0,
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


def convert_hf_input_embedder_torch(config: BaseConfig,
                                    mapping: Mapping = None,
                                    local_checkpoint: str = None,
                                    model_name: str = "openfold3",
                                    weights: dict = None,
                                    **kwargs):
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    assert state_dict is not None

    module_state_dict = {}
    ref_atom_attn_enc_weight_list = [
        "linear_l", "linear_m",
        "pair_mlp.1", "pair_mlp.3", "pair_mlp.5", "linear_q.0"
    ]

    for layer_name in ref_atom_attn_enc_weight_list:
        module_state_dict[f"atom_attn_enc.{layer_name}"] = [{
            "weight":
            state_dict[f"input_embedder.atom_attn_enc.{layer_name}.weight"],
            "bias":
            state_dict.get(f"input_embedder.atom_attn_enc.{layer_name}.bias",
                           None),
        }]

    module_state_dict["atom_attn_enc.ref_atom_feature_embedder.linear_ref_pair_features"] = [
        {"weight": state_dict["input_embedder.atom_attn_enc.ref_atom_feature_embedder.linear_ref_offset.weight"], "bias": None},
        {"weight": state_dict["input_embedder.atom_attn_enc.ref_atom_feature_embedder.linear_inv_sq_dists.weight"], "bias": None},
        {"weight": state_dict["input_embedder.atom_attn_enc.ref_atom_feature_embedder.linear_valid_mask.weight"], "bias": None},
    ]

    ref_atom_attn_enc_weight_list = [
        "ref_atom_feature_embedder.linear_ref_pos",
        "ref_atom_feature_embedder.linear_ref_charge",
        "ref_atom_feature_embedder.linear_ref_mask",
        "ref_atom_feature_embedder.linear_ref_element",
        "ref_atom_feature_embedder.linear_ref_atom_chars"
    ]
    merge_weight = []
    for layer_name in ref_atom_attn_enc_weight_list:
        merge_weight.append({
            "weight":
            state_dict[f"input_embedder.atom_attn_enc.{layer_name}.weight"],
            "bias":
            state_dict.get(f"input_embedder.atom_attn_enc.{layer_name}.bias",
                           None),
        })
    module_state_dict[
        f"atom_attn_enc.ref_atom_feature_embedder.linear_merge_ref_features"] = merge_weight

    atom_attn_enc_coder = convert_hf_diffusion_transformer_torch(
        config=config.atom_transformer_config,
        mapping=mapping,
        model_name=model_name,
        weights=weights,
        prefix="input_embedder.atom_attn_enc.atom_transformer.blocks")
    for key, value in atom_attn_enc_coder.items():
        module_state_dict["atom_attn_enc.atom_transformer." + key] = value

    if getattr(config.atom_transformer_config, 'shared_pair_norm', False):
        # A single LayerNorm lives at the transformer level. Emit it as a
        # top-level key so recursive_calling_load_weights can load it directly
        # into OpenFold3DiffusionTransformer.layer_norm_z.
        layer_norm_z_w = state_dict[
            "input_embedder.atom_attn_enc.atom_transformer.layer_norm_z.weight"]
        module_state_dict["atom_attn_enc.atom_transformer.layer_norm_z"] = \
            [{"weight": layer_norm_z_w}]
        
    input_embedder_weight_list = [
        "linear_s", "linear_relpos", "linear_token_bonds"
    ]
    for layer_name in input_embedder_weight_list:
        module_state_dict[f"{layer_name}"] = [{
            "weight":
            state_dict[f"input_embedder.{layer_name}.weight"],
            "bias":
            state_dict.get(f"input_embedder.{layer_name}.bias", None),
        }]

    module_state_dict["linear_z_ij"] = [
        {
            "weight": state_dict["input_embedder.linear_z_i.weight"],
            "bias": state_dict.get("input_embedder.linear_z_i.bias", None),
        },
        {
            "weight": state_dict["input_embedder.linear_z_j.weight"],
            "bias": state_dict.get("input_embedder.linear_z_j.bias", None),
        },
    ]
    return module_state_dict


def convert_hf_template_embedder_torch(config: BaseConfig,
                                       mapping: Mapping = None,
                                       local_checkpoint: str = None,
                                       model_name: str = "openfold3",
                                       weights: dict = None,
                                       **kwargs):
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    assert state_dict is not None
    prefix = "template_embedder."
    module_state_dict = {}

    replace_name_dict = {
        "tri_mul_out.layer_norm_in": "tri_mul_out.norm_in",
        "tri_mul_out.layer_norm_out": "tri_mul_out.norm_out",
        "tri_mul_out.linear_g": "tri_mul_out.g_out",
        "tri_mul_out.linear_z": "tri_mul_out.p_out",
        "tri_mul_in.layer_norm_in": "tri_mul_in.norm_in",
        "tri_mul_in.layer_norm_out": "tri_mul_in.norm_out",
        "tri_mul_in.linear_g": "tri_mul_in.g_out",
        "tri_mul_in.linear_z": "tri_mul_in.p_out",
        "tri_att_start.layer_norm": "tri_attn_start.layer_norm",
        "tri_att_start.linear_z": "tri_attn_start.linear",
        "tri_att_start.mha.linear_o": "tri_attn_start.mha.o_proj",
        "tri_att_start.mha.linear_g": "tri_attn_start.mha.g_proj",
        "tri_att_end.layer_norm": "tri_attn_end.layer_norm",
        "tri_att_end.linear_z": "tri_attn_end.linear",
        "tri_att_end.mha.linear_o": "tri_attn_end.mha.o_proj",
        "tri_att_end.mha.linear_g": "tri_attn_end.mha.g_proj",
        "pair_transition.layer_norm": "pair_transition.norm",
        "pair_transition.linear_out": "pair_transition.fc3",
    }

    for name in state_dict.keys():
        if name.startswith(prefix) and ".weight" in name:
            layer_name = name.replace(prefix, "")
            for replace_name, replace_name_ in replace_name_dict.items():
                if replace_name in layer_name:
                    layer_name = layer_name.replace(replace_name,
                                                    replace_name_)

            module_state_dict[layer_name.replace(".weight", "")] = [{
                "weight":
                state_dict[name],
                "bias":
                state_dict.get(name.replace(".weight", ".bias"), None),
            }]

    merge_weights_list = {
        "tri_mul_out.p_in":
        ["tri_mul_out.linear_a_p", "tri_mul_out.linear_b_p"],
        "tri_mul_out.g_in":
        ["tri_mul_out.linear_a_g", "tri_mul_out.linear_b_g"],
        "tri_mul_in.p_in": ["tri_mul_in.linear_a_p", "tri_mul_in.linear_b_p"],
        "tri_mul_in.g_in": ["tri_mul_in.linear_a_g", "tri_mul_in.linear_b_g"],
        "tri_attn_start.mha.qkv_proj": [
            "tri_att_start.mha.linear_q", "tri_att_start.mha.linear_k",
            "tri_att_start.mha.linear_v"
        ],
        "tri_attn_end.mha.qkv_proj": [
            "tri_att_end.mha.linear_q", "tri_att_end.mha.linear_k",
            "tri_att_end.mha.linear_v"
        ],
        "pair_transition.fused_fc2_fc1":
        ["pair_transition.swiglu.linear_b", "pair_transition.swiglu.linear_a"]
    }

    clean_layer_list = []
    merge_layer_dict = {}

    for key, value in merge_weights_list.items():
        for name_ in module_state_dict.keys():
            weight_list = []
            start_name = value[0]
            if start_name in name_:
                weight_list.append(module_state_dict[name_][0])
                clean_layer_list.append(name_)

                for same_group_name in value[1:]:
                    merge_layer_name = name_.replace(start_name,
                                                     same_group_name)
                    weight_list.append(module_state_dict[merge_layer_name][0])
                    clean_layer_list.append(merge_layer_name)

                merge_layer_key = name_.replace(start_name, key)
                merge_layer_dict[merge_layer_key] = weight_list
    module_state_dict.update(merge_layer_dict)
    for name in clean_layer_list:
        module_state_dict.pop(name)

    template_pair_embedder_merge_feats_weight_list = [
        "template_pair_embedder.dgram_linear",
        "template_pair_embedder.pseudo_beta_mask_linear",
        "template_pair_embedder.aatype_linear_1",
        "template_pair_embedder.aatype_linear_2",
        "template_pair_embedder.x_linear", "template_pair_embedder.y_linear",
        "template_pair_embedder.z_linear",
        "template_pair_embedder.backbone_mask_linear"
    ]
    merge_weight = []
    for layer_name in template_pair_embedder_merge_feats_weight_list:
        merge_weight.append({
            "weight":
            state_dict[f"template_embedder.{layer_name}.weight"],
            "bias":
            state_dict.get(f"template_embedder.{layer_name}.bias", None),
        })
        module_state_dict.pop(layer_name)
    module_state_dict[
        f"template_pair_embedder.template_pair_embedder_merge_feats"] = merge_weight

    return module_state_dict


def convert_hf_msa_stack_torch(config: BaseConfig,
                               mapping: Mapping = None,
                               local_checkpoint: str = None,
                               model_name: str = "openfold3",
                               weights: dict = None,
                               **kwargs):
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    assert state_dict is not None
    prefix = "msa_module."
    #This part is to replace the names of the weights to the names of the tensorrt-bionemo model
    module_state_dict = {}
    replace_name_dict = {
        "msa_att_row.layer_norm_m": "msa_att_row.norm_m",
        "msa_att_row.layer_norm_z": "msa_att_row.norm_z",
        "msa_att_row.linear_z": "msa_att_row.proj_z",
        "msa_att_row.linear_o": "msa_att_row.proj_o",
        "msa_transition.layer_norm": "msa_transition.norm",
        "msa_transition.linear_out": "msa_transition.fc3",
        "outer_product_mean.layer_norm": "outer_product_mean.norm",
        "outer_product_mean.linear_out": "outer_product_mean.proj_o",
        "pair_stack.tri_mul_out.layer_norm_in": "tri_mul_out.norm_in",
        "pair_stack.tri_mul_out.layer_norm_out": "tri_mul_out.norm_out",
        "pair_stack.tri_mul_out.linear_g": "tri_mul_out.g_out",
        "pair_stack.tri_mul_out.linear_z": "tri_mul_out.p_out",
        "pair_stack.tri_mul_in.layer_norm_in": "tri_mul_in.norm_in",
        "pair_stack.tri_mul_in.layer_norm_out": "tri_mul_in.norm_out",
        "pair_stack.tri_mul_in.linear_g": "tri_mul_in.g_out",
        "pair_stack.tri_mul_in.linear_z": "tri_mul_in.p_out",
        "pair_stack.tri_att_start.layer_norm": "tri_attn_start.layer_norm",
        "pair_stack.tri_att_start.linear_z": "tri_attn_start.linear",
        "pair_stack.tri_att_start.mha.linear_o": "tri_attn_start.mha.o_proj",
        "pair_stack.tri_att_start.mha.linear_g": "tri_attn_start.mha.g_proj",
        "pair_stack.tri_att_end.layer_norm": "tri_attn_end.layer_norm",
        "pair_stack.tri_att_end.linear_z": "tri_attn_end.linear",
        "pair_stack.tri_att_end.mha.linear_o": "tri_attn_end.mha.o_proj",
        "pair_stack.tri_att_end.mha.linear_g": "tri_attn_end.mha.g_proj",
        "pair_stack.pair_transition.layer_norm": "pair_transition.norm",
        "pair_stack.pair_transition.linear_out": "pair_transition.fc3",
    }
    for name in state_dict.keys():

        if name.startswith(prefix) and ".weight" in name:
            layer_name = name.replace(prefix, "")

            for replace_name, replace_name_ in replace_name_dict.items():
                if replace_name in layer_name:
                    layer_name = layer_name.replace(replace_name,
                                                    replace_name_)

            module_state_dict[layer_name.replace(".weight", "")] = [{
                "weight":
                state_dict[name],
                "bias":
                state_dict.get(name.replace(".weight", ".bias"), None),
            }]

    #This part is to merge the weights of the same group (weight fusion)

    merge_weights_list = {
        "msa_att_row.fused_proj_m_g":
        ["msa_att_row.linear_v", "msa_att_row.linear_g"],
        "msa_transition.fused_fc2_fc1":
        ["msa_transition.swiglu.linear_b", "msa_transition.swiglu.linear_a"],
        "outer_product_mean.fused_proj_a_b":
        ["outer_product_mean.linear_1", "outer_product_mean.linear_2"],
        "tri_mul_out.p_in": [
            "pair_stack.tri_mul_out.linear_a_p",
            "pair_stack.tri_mul_out.linear_b_p"
        ],
        "tri_mul_out.g_in": [
            "pair_stack.tri_mul_out.linear_a_g",
            "pair_stack.tri_mul_out.linear_b_g"
        ],
        "tri_mul_in.p_in": [
            "pair_stack.tri_mul_in.linear_a_p",
            "pair_stack.tri_mul_in.linear_b_p"
        ],
        "tri_mul_in.g_in": [
            "pair_stack.tri_mul_in.linear_a_g",
            "pair_stack.tri_mul_in.linear_b_g"
        ],
        "tri_attn_start.mha.qkv_proj": [
            "pair_stack.tri_att_start.mha.linear_q",
            "pair_stack.tri_att_start.mha.linear_k",
            "pair_stack.tri_att_start.mha.linear_v"
        ],
        "tri_attn_end.mha.qkv_proj": [
            "pair_stack.tri_att_end.mha.linear_q",
            "pair_stack.tri_att_end.mha.linear_k",
            "pair_stack.tri_att_end.mha.linear_v"
        ],
        "pair_transition.fused_fc2_fc1": [
            "pair_stack.pair_transition.swiglu.linear_b",
            "pair_stack.pair_transition.swiglu.linear_a"
        ]
    }
    clean_layer_list = []
    merge_layer_dict = {}

    for key, value in merge_weights_list.items():
        for name_ in module_state_dict.keys():
            weight_list = []
            start_name = value[0]
            if start_name in name_:
                weight_list.append(module_state_dict[name_][0])
                clean_layer_list.append(name_)

                for same_group_name in value[1:]:
                    merge_layer_name = name_.replace(start_name,
                                                     same_group_name)
                    weight_list.append(module_state_dict[merge_layer_name][0])
                    clean_layer_list.append(merge_layer_name)

                merge_layer_key = name_.replace(start_name, key)
                merge_layer_dict[merge_layer_key] = weight_list
    module_state_dict.update(merge_layer_dict)
    for name in clean_layer_list:
        module_state_dict.pop(name)

    return module_state_dict

def convert_hf_msa_module_embedder_torch(config: BaseConfig,
                                         mapping: Mapping = None,
                                         local_checkpoint: str = None,
                                         model_name: str = "openfold3",
                                         weights: dict = None,
                                         **kwargs):
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    assert state_dict is not None
    prefix = "msa_module_embedder."
    module_state_dict = {}
    for name in state_dict.keys():
        if name.startswith(prefix) and ".weight" in name:
            layer_name = name.replace(prefix, "")
            module_state_dict[layer_name.replace(".weight", "")] = [{
                "weight":
                state_dict[name],
                "bias":
                state_dict.get(name.replace(".weight", ".bias"), None),
            }]
    return module_state_dict

def convert_hf_auxiliary_heads_torch(config: BaseConfig,
                                     mapping: Mapping = None,
                                     local_checkpoint: str = None,
                                     model_name: str = "openfold3",
                                     weights: dict = None,
                                     **kwargs):
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    assert state_dict is not None
    prefix = "aux_heads."
    module_state_dict = {}
    for name in state_dict.keys():
        if "pairformer_stack" in name:
            continue

        if name.startswith(prefix) and ".weight" in name:
            layer_name = name.replace(prefix, "")
            module_state_dict[layer_name.replace(".weight", "")] = [{
                "weight":
                state_dict[name],
                "bias":
                state_dict.get(name.replace(".weight", ".bias"), None),
            }]

    pair_embed_filtered_state_dict = {}
    for key, value in state_dict.items():
        if "pairformer_stack" in key and "aux_heads" in key:
            pair_embed_filtered_state_dict[key.replace("aux_heads.pairformer_embedding.", "")] = value
    pair_former_weights = convert_hf_pairformer_torch(config=config.pairformer,
                                                      mapping=mapping,
                                                      local_checkpoint=local_checkpoint,
                                                      model_name=model_name,
                                                      weights=pair_embed_filtered_state_dict)
    pair_embed_state_dict = {}
    for key, value in pair_former_weights.items():
        pair_embed_state_dict[f"pairformer_embedding.pairformer_stack.{key}"] = value
    module_state_dict.update(pair_embed_state_dict)
    return module_state_dict

def convert_hf_diffusion_module_torch(config: BaseConfig,
                                      mapping: Mapping = None,
                                      local_checkpoint: str = None,
                                      model_name: str = "openfold3",
                                      weights: dict = None,
                                      **kwargs):
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    assert state_dict is not None
    prefix = "diffusion_module."
    module_state_dict = {}

    replace_name_dict = {
        "transition_z.0.layer_norm":
        "transition_z.0.norm",
        "transition_z.0.linear_out":
        "transition_z.0.fc3",
        "transition_z.1.layer_norm":
        "transition_z.1.norm",
        "transition_z.1.linear_out":
        "transition_z.1.fc3",
        "transition_s.0.layer_norm":
        "transition_s.0.norm",
        "transition_s.0.linear_out":
        "transition_s.0.fc3",
        "transition_s.1.layer_norm":
        "transition_s.1.norm",
        "transition_s.1.linear_out":
        "transition_s.1.fc3",
        "diffusion_conditioning.fourier_emb":
        "diffusion_conditioning.fourier_emb.proj"
    }
    ignore_layer_list = [
        "sample_diffusion", "diffusion_transformer", "attention_pair_bias",
        "atom_transformer"
    ]

    for name in state_dict.keys():

        if any(ignore_layer in name for ignore_layer in ignore_layer_list):
            continue

        if name.startswith(prefix) and (".weight" in name or ".w" in name):
            layer_name = name.replace(prefix, "")

            for replace_name, replace_name_ in replace_name_dict.items():
                if replace_name in layer_name:
                    layer_name = layer_name.replace(replace_name,
                                                    replace_name_)

            if "fourier" not in layer_name:
                module_state_dict[layer_name.replace(".weight", "")] = [{
                    "weight":
                    state_dict[name],
                    "bias":
                    state_dict.get(name.replace(".weight", ".bias"), None),
                }]
            else:
                fourier_emb_weight = state_dict[name]
                if len(fourier_emb_weight.shape) == 1:
                    fourier_emb_weight = fourier_emb_weight.unsqueeze(-1)
                module_state_dict[layer_name.replace(".w", "")] = [{
                    "weight":
                    fourier_emb_weight,
                    "bias":
                    state_dict.get(name.replace(".w", ".b"), None),
                }]

    merge_weights_list = {
        "transition_z.0.fused_fc2_fc1":
        ["transition_z.0.swiglu.linear_b", "transition_z.0.swiglu.linear_a"],
        "transition_z.1.fused_fc2_fc1":
        ["transition_z.1.swiglu.linear_b", "transition_z.1.swiglu.linear_a"],
        "transition_s.0.fused_fc2_fc1":
        ["transition_s.0.swiglu.linear_b", "transition_s.0.swiglu.linear_a"],
        "transition_s.1.fused_fc2_fc1":
        ["transition_s.1.swiglu.linear_b", "transition_s.1.swiglu.linear_a"],
    }
    clean_layer_list = []
    merge_layer_dict = {}

    for key, value in merge_weights_list.items():
        for name_ in module_state_dict.keys():
            weight_list = []
            start_name = value[0]
            if start_name in name_:
                weight_list.append(module_state_dict[name_][0])
                clean_layer_list.append(name_)

                for same_group_name in value[1:]:
                    merge_layer_name = name_.replace(start_name,
                                                     same_group_name)
                    weight_list.append(module_state_dict[merge_layer_name][0])
                    clean_layer_list.append(merge_layer_name)

                merge_layer_key = name_.replace(start_name, key)
                merge_layer_dict[merge_layer_key] = weight_list
    module_state_dict.update(merge_layer_dict)
    for name in clean_layer_list:
        module_state_dict.pop(name)
    
    ref_atom_attn_enc_weight_list = [
        "ref_atom_feature_embedder.linear_ref_pos",
        "ref_atom_feature_embedder.linear_ref_charge",
        "ref_atom_feature_embedder.linear_ref_mask",
        "ref_atom_feature_embedder.linear_ref_element",
        "ref_atom_feature_embedder.linear_ref_atom_chars"
    ]
    merge_weight = []
    for layer_name in ref_atom_attn_enc_weight_list:
        merge_weight.append(module_state_dict[f"atom_attn_enc.{layer_name}"][0])
        module_state_dict.pop(f"atom_attn_enc.{layer_name}")

    module_state_dict[f"atom_attn_enc.ref_atom_feature_embedder.linear_merge_ref_features"] = merge_weight

    ref_pair_keys = [
        "atom_attn_enc.ref_atom_feature_embedder.linear_ref_offset",
        "atom_attn_enc.ref_atom_feature_embedder.linear_inv_sq_dists",
        "atom_attn_enc.ref_atom_feature_embedder.linear_valid_mask",
    ]
    ref_pair_weights = []
    for key in ref_pair_keys:
        ref_pair_weights.append(module_state_dict.pop(key)[0])
    module_state_dict["atom_attn_enc.ref_atom_feature_embedder.linear_ref_pair_features"] = ref_pair_weights

    atom_attn_enc_coder = convert_hf_diffusion_transformer_torch(
        config=config.atom_transformer_encoder_config,
        model_name=model_name,
        weights=weights,
        prefix="diffusion_module.atom_attn_enc.atom_transformer.blocks")
    atom_attn_enc_coder_weights = {}
    for key, value in atom_attn_enc_coder.items():
        atom_attn_enc_coder_weights[
            f"atom_attn_enc.atom_transformer.{key}"] = value
    if getattr(config.atom_transformer_encoder_config, 'shared_pair_norm', False):
        lnz_w = state_dict["diffusion_module.atom_attn_enc.atom_transformer.layer_norm_z.weight"]
        atom_attn_enc_coder_weights["atom_attn_enc.atom_transformer.layer_norm_z"] = [{"weight": lnz_w}]
    module_state_dict.update(atom_attn_enc_coder_weights)

    diffusion_transformer_coder = convert_hf_diffusion_transformer_torch(
        config=config.diffusion_transformer_config.token_transformer,
        model_name=model_name,
        weights=weights,
        prefix="diffusion_module.diffusion_transformer.blocks")
    diffusion_transformer_coder_weights = {}
    for key, value in diffusion_transformer_coder.items():
        diffusion_transformer_coder_weights[
            f"diffusion_transformer.{key}"] = value
    module_state_dict.update(diffusion_transformer_coder_weights)

    atom_attn_dec_coder = convert_hf_diffusion_transformer_torch(
        config=config.atom_transformer_decoder_config,
        model_name=model_name,
        weights=weights,
        prefix="diffusion_module.atom_attn_dec.atom_transformer.blocks")
    atom_attn_dec_coder_weights = {}
    for key, value in atom_attn_dec_coder.items():
        atom_attn_dec_coder_weights[
            f"atom_attn_dec.atom_transformer.{key}"] = value
    if getattr(config.atom_transformer_decoder_config, 'shared_pair_norm', False):
        lnz_w = state_dict["diffusion_module.atom_attn_dec.atom_transformer.layer_norm_z.weight"]
        atom_attn_dec_coder_weights["atom_attn_dec.atom_transformer.layer_norm_z"] = [{"weight": lnz_w}]
    module_state_dict.update(atom_attn_dec_coder_weights)

    return module_state_dict

def convert_hf_openfold3_torch(config: BaseConfig,
                               mapping: Mapping = None,
                               local_checkpoint: str = None,
                               model_name: str = "openfold3",
                               weights: dict = None,
                               **kwargs):
    if weights is None:
        state_dict = load_weights(name=model_name, cache_path=local_checkpoint)
    else:
        state_dict = weights
    assert state_dict is not None
    
    input_embedder_weights = convert_hf_input_embedder_torch(config=config.input_embedder_config,
                                                              mapping=mapping,
                                                              model_name=model_name,
                                                              weights=state_dict)
    msa_stack_weights = convert_hf_msa_stack_torch(config=config.msa_stack_module_config,
                                                  mapping=mapping,
                                                  model_name=model_name,
                                                  weights=state_dict)
    msa_module_embedder_weights = convert_hf_msa_module_embedder_torch(config=config.msa_module_embedder_config,
                                                                       mapping=mapping,
                                                                       model_name=model_name,
                                                                       weights=state_dict)
    diffusion_module_weights = convert_hf_diffusion_module_torch(config=config.diffusion_module_config,
                                                                 mapping=mapping,
                                                                 model_name=model_name,
                                                                 weights=state_dict)
    template_embedder_weights = convert_hf_template_embedder_torch(config=config.template_embedder_config,
                                                                   mapping=mapping,
                                                                   model_name=model_name,
                                                                   weights=state_dict)
    pairformer_stack_weights = convert_hf_pairformer_torch(config=config.trunk.pairformer,
                                                          mapping=mapping,
                                                          model_name=model_name,
                                                          weights=state_dict)
    auxiliary_heads_weights = convert_hf_auxiliary_heads_torch(config=config.auxiliary_heads_config,
                                                                 mapping=mapping,
                                                                 model_name=model_name,
                                                                 weights=state_dict)
    layer_norm_z_weights = {
        "weight": state_dict["layer_norm_z.weight"],
        "bias": state_dict["layer_norm_z.bias"]
    }
    layer_norm_s_weights = {
        "weight": state_dict["layer_norm_s.weight"],
        "bias": state_dict["layer_norm_s.bias"]
    }
    linear_z_weights = [{
        "weight": state_dict["linear_z.weight"],
        "bias": None
    }]
    linear_s_weights = [{
        "weight": state_dict["linear_s.weight"],
        "bias": None
    }]

    return {
        "input_embedder": input_embedder_weights,
        "msa_stack": msa_stack_weights,
        "msa_module_embedder": msa_module_embedder_weights,
        "diffusion_module": diffusion_module_weights,
        "template_embedder": template_embedder_weights,
        "pairformer_stack": pairformer_stack_weights,
        "auxiliary_heads": auxiliary_heads_weights,
        "layer_norm_z": layer_norm_z_weights,
        "linear_z": linear_z_weights,
        "linear_s": linear_s_weights,
        "layer_norm_s": layer_norm_s_weights,
    }
