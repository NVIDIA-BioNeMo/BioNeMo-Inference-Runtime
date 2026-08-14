# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Layout reference: RoseTTAFold3 (RosettaCommons/foundry), BSD-3-Clause.
# https://github.com/RosettaCommons/foundry/tree/production/models/rf3
# Only upstream parameter and module names are reproduced here.

"""
Weight conversion: RF3 PairformerBlock -> BioIR PairformerLayerV1.

Handles:
- Name renames (tri_mul_outgoing -> tri_mul_out, etc.)
- QKV fusion for triangle attention (separate q,k,v -> fused qkv_proj)
- KV fusion for attention pair bias (separate k,v -> fused proj_kv)
- Gate+Input fusion for transition (linear_1,linear_2 -> fused_fc2_fc1)
- Bias handling (the source to_g has bias, BioIR g_proj has no bias for tri_attn)

No checkpoint needed — works with any state_dict matching the RF3 PairformerBlock layout.
"""

import torch


def convert_tri_mul_weights(state_dict, prefix, bioir_prefix):
    """Convert TriangleMultiplication weights. Layout is identical, only name differs."""
    return {
        f"{bioir_prefix}.norm_in.weight": state_dict[f"{prefix}.norm_in.weight"],
        f"{bioir_prefix}.norm_in.bias": state_dict[f"{prefix}.norm_in.bias"],
        f"{bioir_prefix}.p_in.weight": state_dict[f"{prefix}.p_in.weight"],
        f"{bioir_prefix}.g_in.weight": state_dict[f"{prefix}.g_in.weight"],
        f"{bioir_prefix}.norm_out.weight": state_dict[f"{prefix}.norm_out.weight"],
        f"{bioir_prefix}.norm_out.bias": state_dict[f"{prefix}.norm_out.bias"],
        f"{bioir_prefix}.p_out.weight": state_dict[f"{prefix}.p_out.weight"],
        f"{bioir_prefix}.g_out.weight": state_dict[f"{prefix}.g_out.weight"],
    }


def convert_tri_attn_weights(state_dict, prefix, bioir_prefix):
    """Convert TriangleAttention weights.

    Fusions:
    - to_q + to_k + to_v -> mha.qkv_proj (cat dim=0)
    - to_g -> mha.g_proj (bias dropped — BioIR g_proj has no bias)
    - to_out -> mha.o_proj (bias dropped — BioIR o_proj has no bias)
    - norm -> layer_norm
    - to_b -> linear
    """
    q_weight = state_dict[f"{prefix}.to_q.weight"]
    k_weight = state_dict[f"{prefix}.to_k.weight"]
    v_weight = state_dict[f"{prefix}.to_v.weight"]
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

    return {
        f"{bioir_prefix}.layer_norm.weight": state_dict[f"{prefix}.norm.weight"],
        f"{bioir_prefix}.layer_norm.bias": state_dict[f"{prefix}.norm.bias"],
        f"{bioir_prefix}.linear.weight": state_dict[f"{prefix}.to_b.weight"],
        f"{bioir_prefix}.mha.qkv_proj.weight": qkv_weight,
        f"{bioir_prefix}.mha.o_proj.weight": state_dict[f"{prefix}.to_out.weight"],
        f"{bioir_prefix}.mha.g_proj.weight": state_dict[f"{prefix}.to_g.weight"],
    }


def convert_transition_weights(state_dict, prefix, bioir_prefix):
    """Convert Transition weights.

    Fusions:
    - linear_2 + linear_1 -> fused_fc2_fc1 (gate first! cat dim=0)
    - linear_3 -> fc3
    - layer_norm_1 -> norm
    """
    gate_weight = state_dict[f"{prefix}.linear_2.weight"]
    input_weight = state_dict[f"{prefix}.linear_1.weight"]
    fused_weight = torch.cat([gate_weight, input_weight], dim=0)

    return {
        f"{bioir_prefix}.norm.weight": state_dict[f"{prefix}.layer_norm_1.weight"],
        f"{bioir_prefix}.norm.bias": state_dict[f"{prefix}.layer_norm_1.bias"],
        f"{bioir_prefix}.fused_fc2_fc1.weight": fused_weight,
        f"{bioir_prefix}.fc3.weight": state_dict[f"{prefix}.linear_3.weight"],
    }


def convert_attention_pair_bias_weights(state_dict, prefix, bioir_prefix):
    """Convert AttentionPairBiasPairformer weights.

    Fusions:
    - to_k + to_v -> proj_kv (cat dim=0)

    Renames:
    - to_q -> proj_q (BioIR adds a bias; initialize to zero)
    - to_g -> proj_g
    - to_a -> proj_o
    - to_b -> proj_z.1 (pair bias linear)
    - ln_0 -> proj_z.0 (pair bias norm)
    - ln_1 -> norm_s (input norm on single rep)
    """
    k_weight = state_dict[f"{prefix}.to_k.weight"]
    v_weight = state_dict[f"{prefix}.to_v.weight"]
    kv_weight = torch.cat([k_weight, v_weight], dim=0)

    q_weight = state_dict[f"{prefix}.to_q.weight"]
    # BioIR proj_q has a bias; the source module does not. Initialize to zero.
    q_bias = torch.zeros(q_weight.shape[0], dtype=q_weight.dtype)

    return {
        f"{bioir_prefix}.norm_s.weight": state_dict[f"{prefix}.ln_1.weight"],
        f"{bioir_prefix}.norm_s.bias": state_dict[f"{prefix}.ln_1.bias"],
        f"{bioir_prefix}.proj_q.weight": q_weight,
        f"{bioir_prefix}.proj_q.bias": q_bias,
        f"{bioir_prefix}.proj_kv.weight": kv_weight,
        f"{bioir_prefix}.proj_g.weight": state_dict[f"{prefix}.to_g.weight"],
        f"{bioir_prefix}.proj_o.weight": state_dict[f"{prefix}.to_a.weight"],
        f"{bioir_prefix}.proj_z.0.weight": state_dict[f"{prefix}.ln_0.weight"],
        f"{bioir_prefix}.proj_z.0.bias": state_dict[f"{prefix}.ln_0.bias"],
        f"{bioir_prefix}.proj_z.1.weight": state_dict[f"{prefix}.to_b.weight"],
    }


def convert_pairformer_block_weights(state_dict, prefix="", bioir_prefix=""):
    """Convert a single RF3 PairformerBlock to BioIR PairformerLayerV1 weights.

    Args:
        state_dict: Source checkpoint state_dict (or subset for one block).
        prefix: Key prefix for source weights (e.g., "pairformer_stack.0").
        bioir_prefix: Key prefix for BioIR weights (e.g., "layers.0").

    Returns:
        Dict of converted weights usable by both Torch and TRT backends.
    """
    dot = "." if prefix else ""
    tdot = "." if bioir_prefix else ""

    weights = {}
    weights.update(
        convert_tri_mul_weights(state_dict, f"{prefix}{dot}tri_mul_outgoing", f"{bioir_prefix}{tdot}tri_mul_out")
    )
    weights.update(
        convert_tri_mul_weights(state_dict, f"{prefix}{dot}tri_mul_incoming", f"{bioir_prefix}{tdot}tri_mul_in")
    )
    weights.update(
        convert_tri_attn_weights(state_dict, f"{prefix}{dot}tri_attn_start", f"{bioir_prefix}{tdot}tri_attn_start")
    )
    weights.update(
        convert_tri_attn_weights(state_dict, f"{prefix}{dot}tri_attn_end", f"{bioir_prefix}{tdot}tri_attn_end")
    )
    weights.update(
        convert_transition_weights(state_dict, f"{prefix}{dot}z_transition", f"{bioir_prefix}{tdot}transition_z")
    )
    weights.update(
        convert_transition_weights(state_dict, f"{prefix}{dot}s_transition", f"{bioir_prefix}{tdot}transition_s")
    )
    weights.update(
        convert_attention_pair_bias_weights(
            state_dict, f"{prefix}{dot}attention_pair_bias", f"{bioir_prefix}{tdot}attention"
        )
    )
    return weights


def convert_pairformer_stack_weights(state_dict, num_blocks, prefix="pairformer_stack", bioir_prefix="layers"):
    """Convert a full stack of PairformerBlocks.

    Args:
        state_dict: Full model state_dict.
        num_blocks: Number of pairformer blocks.
        prefix: Source prefix for the stack (e.g., "pairformer_stack").
        bioir_prefix: BioIR prefix (e.g., "layers").

    Returns:
        Dict of converted weights for the full PairformerModule.
    """
    weights = {}
    for i in range(num_blocks):
        weights.update(convert_pairformer_block_weights(state_dict, f"{prefix}.{i}", f"{bioir_prefix}.{i}"))
    return weights
