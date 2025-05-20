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

from tensorrt_bionemo.confs.modules.transformers import PairformerConfig
from tensorrt_bionemo.hf.checkpoints import load_hf_weights
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.boltz1.convert import (get_pairwise_attn_weights,
                                                    get_transition_weights,
                                                    get_tri_attn_node_weights,
                                                    get_tri_mul_node_weights)


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
                          local_checkpoint: str = None):
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
        state_dict = load_hf_weights(name="boltz-2")

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
