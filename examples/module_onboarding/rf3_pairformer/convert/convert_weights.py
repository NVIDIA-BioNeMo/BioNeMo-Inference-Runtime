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
- Q/K/V/gate/pair-bias fusion for triangle attention (-> fused mha.in_proj)
- Q/gate/K/V fusion for attention pair bias (-> fused in_proj)
- Transition fusion (linear_2, then the SiLU-activated linear_1 -> fused_fc2_fc1)
- Triangle attention gate and output biases (to_g.bias, to_out.bias). They load
  only into the nodes that ``integration.swap.build_pairformer_module`` builds.

No checkpoint needed — works with any state_dict matching the RF3 PairformerBlock layout.
"""

import torch
import torch.nn.functional as F

from bionemo_ir._torch.layers.attention import pair_bias_rows


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
    - to_q + to_k + to_v + to_g + to_b -> mha.in_proj (cat dim=0), with to_b
      zero-padded to pair_bias_rows(H) rows. Its bias is to_g's over the gate
      rows and zero over the rest, which have none in RF3.
    - to_out -> mha.o_proj, with its bias
    - norm -> layer_norm
    """
    q_weight = state_dict[f"{prefix}.to_q.weight"]
    k_weight = state_dict[f"{prefix}.to_k.weight"]
    v_weight = state_dict[f"{prefix}.to_v.weight"]
    pair_bias_weight = state_dict[f"{prefix}.to_b.weight"]
    num_heads = pair_bias_weight.shape[0]
    bias_rows = pair_bias_rows(num_heads)
    in_proj_weight = torch.cat(
        [
            q_weight,
            k_weight,
            v_weight,
            state_dict[f"{prefix}.to_g.weight"],
            F.pad(pair_bias_weight, (0, 0, 0, bias_rows - num_heads)),
        ],
        dim=0,
    )
    g_bias = state_dict[f"{prefix}.to_g.bias"]
    qkv_rows = q_weight.shape[0] + k_weight.shape[0] + v_weight.shape[0]
    in_proj_bias = torch.cat([g_bias.new_zeros(qkv_rows), g_bias, g_bias.new_zeros(bias_rows)])

    return {
        f"{bioir_prefix}.layer_norm.weight": state_dict[f"{prefix}.norm.weight"],
        f"{bioir_prefix}.layer_norm.bias": state_dict[f"{prefix}.norm.bias"],
        f"{bioir_prefix}.mha.in_proj.weight": in_proj_weight,
        f"{bioir_prefix}.mha.in_proj.bias": in_proj_bias,
        f"{bioir_prefix}.mha.o_proj.weight": state_dict[f"{prefix}.to_out.weight"],
        f"{bioir_prefix}.mha.o_proj.bias": state_dict[f"{prefix}.to_out.bias"],
    }


def convert_transition_weights(state_dict, prefix, bioir_prefix):
    """Convert Transition weights.

    RF3 computes linear_3(silu(linear_1(x)) * linear_2(x)). BioIR applies SiLU
    to the second half of fused_fc2_fc1.

    Fusions:
    - linear_2, then linear_1 -> fused_fc2_fc1 (cat dim=0)
    - linear_3 -> fc3
    - layer_norm_1 -> norm
    """
    gate_weight = state_dict[f"{prefix}.linear_1.weight"]  # SiLU-activated
    value_weight = state_dict[f"{prefix}.linear_2.weight"]
    fused_weight = torch.cat([value_weight, gate_weight], dim=0)  # gate second

    return {
        f"{bioir_prefix}.norm.weight": state_dict[f"{prefix}.layer_norm_1.weight"],
        f"{bioir_prefix}.norm.bias": state_dict[f"{prefix}.layer_norm_1.bias"],
        f"{bioir_prefix}.fused_fc2_fc1.weight": fused_weight,
        f"{bioir_prefix}.fc3.weight": state_dict[f"{prefix}.linear_3.weight"],
    }


def convert_attention_pair_bias_weights(state_dict, prefix, bioir_prefix):
    """Convert AttentionPairBiasPairformer weights.

    Fusions:
    - to_q + to_g + to_k + to_v -> in_proj (cat dim=0, in that order; BioIR
      in_proj has a bias, which the source module does not: zero it). RF3
      stores to_g as Sequential(Linear, Sigmoid): to_g.0

    Renames:
    - to_a -> proj_o
    - to_b -> proj_z.1 (pair bias linear)
    - ln_0 -> proj_z.0 (pair bias norm)
    - ln_1 -> norm_s (input norm on single rep)
    """
    g_key = f"{prefix}.to_g.0.weight" if f"{prefix}.to_g.0.weight" in state_dict else f"{prefix}.to_g.weight"
    in_proj_weight = torch.cat(
        [
            state_dict[f"{prefix}.to_q.weight"],
            state_dict[g_key],
            state_dict[f"{prefix}.to_k.weight"],
            state_dict[f"{prefix}.to_v.weight"],
        ],
        dim=0,
    )

    return {
        f"{bioir_prefix}.norm_s.weight": state_dict[f"{prefix}.ln_1.weight"],
        f"{bioir_prefix}.norm_s.bias": state_dict[f"{prefix}.ln_1.bias"],
        f"{bioir_prefix}.in_proj.weight": in_proj_weight,
        f"{bioir_prefix}.in_proj.bias": in_proj_weight.new_zeros(in_proj_weight.shape[0]),
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
        Flat dict of converted weights for ``PairformerModule.load_state_dict``.
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
        Flat dict of converted weights for the full PairformerModule.
    """
    weights = {}
    for i in range(num_blocks):
        weights.update(convert_pairformer_block_weights(state_dict, f"{prefix}.{i}", f"{bioir_prefix}.{i}"))
    return weights
