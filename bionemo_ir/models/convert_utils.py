# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Checkpoint key mappings for Boltz-style triangle and transition layers.

Protenix checkpoints name these sublayers as Boltz does, so its conversion
uses these helpers without importing a Boltz model package.
"""

import torch

from bionemo_ir.utils import str_dtype_to_torch

__all__ = ["get_transition_weights", "get_tri_attn_node_weights", "get_tri_mul_node_weights"]


def get_tri_attn_node_weights(state_dict: dict, prefix: str, bioir_prefix: str, dtype: str = "float32"):
    torch_dtype = str_dtype_to_torch(dtype)
    layer_norm_weight = state_dict[f"{prefix}.layer_norm.weight"]
    layer_norm_bias = state_dict[f"{prefix}.layer_norm.bias"]
    linear_weight = state_dict[f"{prefix}.linear.weight"]

    mha_q_weight = state_dict[f"{prefix}.mha.linear_q.weight"]
    mha_k_weight = state_dict[f"{prefix}.mha.linear_k.weight"]
    mha_v_weight = state_dict[f"{prefix}.mha.linear_v.weight"]
    mha_o_weight = state_dict[f"{prefix}.mha.linear_o.weight"]
    mha_g_weight = state_dict[f"{prefix}.mha.linear_g.weight"]

    mha_in_proj_weight = torch.cat([mha_q_weight, mha_k_weight, mha_v_weight, mha_g_weight], dim=0)

    ret = {
        f"{bioir_prefix}.layer_norm.weight": layer_norm_weight.to(torch_dtype),
        f"{bioir_prefix}.layer_norm.bias": layer_norm_bias.to(torch_dtype),
        f"{bioir_prefix}.pair_bias_proj.weight": linear_weight.to(torch_dtype),
        f"{bioir_prefix}.mha.in_proj.weight": mha_in_proj_weight.to(torch_dtype),
        f"{bioir_prefix}.mha.o_proj.weight": mha_o_weight.to(torch_dtype),
    }
    return ret


def get_tri_mul_node_weights(state_dict: dict, prefix: str, bioir_prefix: str, dtype: str = "float32"):
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
        f"{bioir_prefix}.norm_in.weight": norm_in_weight.to(torch_dtype),
        f"{bioir_prefix}.norm_in.bias": norm_in_bias.to(torch_dtype),
        f"{bioir_prefix}.p_in.weight": p_in_weight.to(torch_dtype),
        f"{bioir_prefix}.g_in.weight": g_in_weight.to(torch_dtype),
        f"{bioir_prefix}.norm_out.weight": norm_out_weight,
        f"{bioir_prefix}.norm_out.bias": norm_out_bias,
        f"{bioir_prefix}.p_out.weight": p_out_weight,
        f"{bioir_prefix}.g_out.weight": g_out_weight,
    }
    return ret


def get_transition_weights(state_dict: dict, prefix: str, bioir_prefix: str, dim: int = 128, dtype: str = "float32"):
    torch_dtype = str_dtype_to_torch(dtype)
    norm_weight = state_dict[f"{prefix}.norm.weight"]
    norm_bias = state_dict[f"{prefix}.norm.bias"]
    fc1_weight = state_dict[f"{prefix}.fc1.weight"]
    fc2_weight = state_dict[f"{prefix}.fc2.weight"]
    fc3_weight = state_dict[f"{prefix}.fc3.weight"]

    fused_fc2_fc1_weight = torch.cat([fc2_weight, fc1_weight], dim=0)

    ret = {
        f"{bioir_prefix}.norm.weight": norm_weight.to(torch_dtype),
        f"{bioir_prefix}.norm.bias": norm_bias.to(torch_dtype),
        f"{bioir_prefix}.fused_fc2_fc1.weight": fused_fc2_fc1_weight.to(torch_dtype),
        f"{bioir_prefix}.fc3.weight": fc3_weight.to(torch_dtype),
    }
    return ret
