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
Weight conversion: RF3 DiffusionTransformerBlock -> TRT-BNM DiffusionTransformerLayer.

Checkpoint layout (variant A; variant B renames ada_ln_1 -> ln_1 — both handled below):
- c_token=768, c_s=384, c_z=128, n_head=16, 24 blocks
- AdaLN named ada_ln_1 (attention) and ada_ln (transition)
- to_g is Sequential (to_g.0.weight), not plain Linear
- QK normalization present (query_layer_norm, key_layer_norm)
- Attention output gate present (linear_output_project on attention block)
"""

import torch


def convert_adaln_weights(state_dict, prefix, tbm_prefix):
    """Convert AdaLN: separate gain/bias -> fused_s_scale_s_bias."""
    gain_w = state_dict[f"{prefix}.to_gain.0.weight"]
    bias_w = state_dict[f"{prefix}.to_bias.weight"]
    fused_weight = torch.cat([gain_w, bias_w], dim=0)

    gain_b = state_dict[f"{prefix}.to_gain.0.bias"]
    zero_b = torch.zeros_like(gain_b)
    fused_bias = torch.cat([gain_b, zero_b], dim=0)

    return {
        f"{tbm_prefix}.s_norm.weight": state_dict[f"{prefix}.ln_s.weight"],
        f"{tbm_prefix}.fused_s_scale_s_bias.weight": fused_weight,
        f"{tbm_prefix}.fused_s_scale_s_bias.bias": fused_bias,
    }


def convert_attention_weights(state_dict, prefix, tbm_prefix, dim):
    """Convert AttentionPairBias weights from an RF3 checkpoint.

    Key differences from simple test variant:
    - to_g is Sequential: to_g.0.weight (not to_g.weight)
    - QK norm present: query_layer_norm, key_layer_norm
    - Output gate on attention: linear_output_project
    """
    k_weight = state_dict[f"{prefix}.to_k.weight"]
    v_weight = state_dict[f"{prefix}.to_v.weight"]
    kv_weight = torch.cat([k_weight, v_weight], dim=0)

    q_weight = state_dict[f"{prefix}.to_q.weight"]
    q_bias = torch.zeros(q_weight.shape[0], dtype=q_weight.dtype)

    # to_g may be Sequential (to_g.0.weight) or plain Linear (to_g.weight)
    g_key = f"{prefix}.to_g.0.weight" if f"{prefix}.to_g.0.weight" in state_dict else f"{prefix}.to_g.weight"

    return {
        f"{tbm_prefix}.proj_q.weight": q_weight,
        f"{tbm_prefix}.proj_q.bias": q_bias,
        f"{tbm_prefix}.proj_kv.weight": kv_weight,
        f"{tbm_prefix}.proj_g.weight": state_dict[g_key],
        f"{tbm_prefix}.proj_o.weight": state_dict[f"{prefix}.to_a.weight"],
    }


def convert_pair_bias_norm_weights(state_dict, prefix, tbm_prefix, for_trt=False):
    """Convert pair bias norm (ln_0 + to_b).

    Torch backend uses Sequential: proj_z.0 (LayerNorm) + proj_z.1 (Linear)
    TRT backend uses flat: proj_z_norm (LayerNorm) + proj_z (Linear)
    """
    if for_trt:
        # TRT naming: proj_z_norm + proj_z
        base = tbm_prefix.rsplit(".proj_z", 1)[0]
        return {
            f"{base}.proj_z_norm.weight": state_dict[f"{prefix}.ln_0.weight"],
            f"{base}.proj_z_norm.bias": state_dict[f"{prefix}.ln_0.bias"],
            f"{base}.proj_z.weight": state_dict[f"{prefix}.to_b.weight"],
        }
    else:
        # Torch naming: proj_z.0 + proj_z.1
        return {
            f"{tbm_prefix}.0.weight": state_dict[f"{prefix}.ln_0.weight"],
            f"{tbm_prefix}.0.bias": state_dict[f"{prefix}.ln_0.bias"],
            f"{tbm_prefix}.1.weight": state_dict[f"{prefix}.to_b.weight"],
        }


def convert_output_gate_weights(state_dict, prefix, tbm_prefix):
    """Convert attention output gate (linear_output_project -> output_projection)."""
    return {
        f"{tbm_prefix}.weight": state_dict[f"{prefix}.linear_output_project.0.weight"],
        f"{tbm_prefix}.bias": state_dict[f"{prefix}.linear_output_project.0.bias"],
    }


def convert_conditioned_transition_weights(state_dict, prefix, tbm_prefix):
    """Convert ConditionedTransitionBlock with using_silu=True (2-way fusion)."""
    linear_1_w = state_dict[f"{prefix}.linear_1.weight"]
    linear_2_w = state_dict[f"{prefix}.linear_2.weight"]
    fused_weight = torch.cat([linear_1_w, linear_2_w], dim=0)

    return {
        f"{tbm_prefix}.fused_swl_a_to_b.weight": fused_weight,
        f"{tbm_prefix}.b_to_a.weight": state_dict[f"{prefix}.linear_3.weight"],
        f"{tbm_prefix}.output_projection.weight": state_dict[f"{prefix}.linear_output_project.0.weight"],
        f"{tbm_prefix}.output_projection.bias": state_dict[f"{prefix}.linear_output_project.0.bias"],
    }


def convert_dit_block_weights(state_dict, prefix="", tbm_prefix="", dim=768, n_head=16, for_trt=False):
    """Convert a single RF3 DiffusionTransformerBlock from a released checkpoint.

    Args:
        for_trt: If True, use TRT weight naming (proj_z_norm/proj_z instead of proj_z.0/proj_z.1).
    """
    dot = "." if prefix else ""
    tdot = "." if tbm_prefix else ""

    weights = {}

    # AdaLN (attention input norm) — named ada_ln_1 in the released checkpoint, ln_1 in the source variant
    adaln_prefix = f"{prefix}{dot}attention_pair_bias.ada_ln_1"
    if f"{adaln_prefix}.ln_s.weight" not in state_dict:
        adaln_prefix = f"{prefix}{dot}attention_pair_bias.ln_1"
    weights.update(convert_adaln_weights(state_dict, adaln_prefix, f"{tbm_prefix}{tdot}adaln"))

    # Attention projections
    weights.update(
        convert_attention_weights(
            state_dict, f"{prefix}{dot}attention_pair_bias", f"{tbm_prefix}{tdot}pair_bias_attn", dim
        )
    )

    # Pair bias norm
    weights.update(
        convert_pair_bias_norm_weights(
            state_dict, f"{prefix}{dot}attention_pair_bias", f"{tbm_prefix}{tdot}pair_bias_attn.proj_z", for_trt=for_trt
        )
    )

    # Attention output gate — only present in checkpoint variant A
    output_gate_key = f"{prefix}{dot}attention_pair_bias.linear_output_project.0.weight"
    if output_gate_key in state_dict:
        weights.update(
            convert_output_gate_weights(
                state_dict, f"{prefix}{dot}attention_pair_bias", f"{tbm_prefix}{tdot}output_projection"
            )
        )

    # Conditioned transition block — AdaLN
    weights.update(
        convert_adaln_weights(
            state_dict, f"{prefix}{dot}conditioned_transition_block.ada_ln", f"{tbm_prefix}{tdot}transition.adaln"
        )
    )

    # Conditioned transition block — linear layers
    weights.update(
        convert_conditioned_transition_weights(
            state_dict, f"{prefix}{dot}conditioned_transition_block", f"{tbm_prefix}{tdot}transition"
        )
    )

    return weights


def convert_dit_stack_weights(
    state_dict, num_blocks, prefix="blocks", tbm_prefix="layers", dim=768, n_head=16, for_trt=False
):
    """Convert a full stack of DiffusionTransformerBlocks from a released checkpoint."""
    weights = {}
    for i in range(num_blocks):
        weights.update(
            convert_dit_block_weights(state_dict, f"{prefix}.{i}", f"{tbm_prefix}.{i}", dim, n_head, for_trt=for_trt)
        )
    return weights
