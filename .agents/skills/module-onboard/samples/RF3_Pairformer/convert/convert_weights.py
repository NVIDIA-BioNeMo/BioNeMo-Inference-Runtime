"""
Weight conversion: BakerLab RF3 PairformerBlock -> TRT-BNM PairformerLayerV1.

Handles:
- Name renames (tri_mul_outgoing -> tri_mul_out, etc.)
- QKV fusion for triangle attention (separate q,k,v -> fused qkv_proj)
- KV fusion for attention pair bias (separate k,v -> fused proj_kv)
- Gate+Input fusion for transition (linear_1,linear_2 -> fused_fc2_fc1)
- Bias handling (customer to_g has bias, TRT-BNM g_proj has no bias for tri_attn)

No checkpoint needed — works with any state_dict matching the BakerLab PairformerBlock layout.
"""

import torch


def convert_tri_mul_weights(state_dict, prefix, tbm_prefix):
    """Convert TriangleMultiplication weights. Layout is identical, only name differs."""
    return {
        f"{tbm_prefix}.norm_in.weight": state_dict[f"{prefix}.norm_in.weight"],
        f"{tbm_prefix}.norm_in.bias": state_dict[f"{prefix}.norm_in.bias"],
        f"{tbm_prefix}.p_in.weight": state_dict[f"{prefix}.p_in.weight"],
        f"{tbm_prefix}.g_in.weight": state_dict[f"{prefix}.g_in.weight"],
        f"{tbm_prefix}.norm_out.weight": state_dict[f"{prefix}.norm_out.weight"],
        f"{tbm_prefix}.norm_out.bias": state_dict[f"{prefix}.norm_out.bias"],
        f"{tbm_prefix}.p_out.weight": state_dict[f"{prefix}.p_out.weight"],
        f"{tbm_prefix}.g_out.weight": state_dict[f"{prefix}.g_out.weight"],
    }


def convert_tri_attn_weights(state_dict, prefix, tbm_prefix):
    """Convert TriangleAttention weights.

    Fusions:
    - to_q + to_k + to_v -> mha.qkv_proj (cat dim=0)
    - to_g -> mha.g_proj (bias dropped — TRT-BNM g_proj has no bias)
    - to_out -> mha.o_proj (bias dropped — TRT-BNM o_proj has no bias)
    - norm -> layer_norm
    - to_b -> linear
    """
    q_weight = state_dict[f"{prefix}.to_q.weight"]
    k_weight = state_dict[f"{prefix}.to_k.weight"]
    v_weight = state_dict[f"{prefix}.to_v.weight"]
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

    return {
        f"{tbm_prefix}.layer_norm.weight": state_dict[f"{prefix}.norm.weight"],
        f"{tbm_prefix}.layer_norm.bias": state_dict[f"{prefix}.norm.bias"],
        f"{tbm_prefix}.linear.weight": state_dict[f"{prefix}.to_b.weight"],
        f"{tbm_prefix}.mha.qkv_proj.weight": qkv_weight,
        f"{tbm_prefix}.mha.o_proj.weight": state_dict[f"{prefix}.to_out.weight"],
        f"{tbm_prefix}.mha.g_proj.weight": state_dict[f"{prefix}.to_g.weight"],
    }


def convert_transition_weights(state_dict, prefix, tbm_prefix):
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
        f"{tbm_prefix}.norm.weight": state_dict[f"{prefix}.layer_norm_1.weight"],
        f"{tbm_prefix}.norm.bias": state_dict[f"{prefix}.layer_norm_1.bias"],
        f"{tbm_prefix}.fused_fc2_fc1.weight": fused_weight,
        f"{tbm_prefix}.fc3.weight": state_dict[f"{prefix}.linear_3.weight"],
    }


def convert_attention_pair_bias_weights(state_dict, prefix, tbm_prefix):
    """Convert AttentionPairBiasPairformer weights.

    Fusions:
    - to_k + to_v -> proj_kv (cat dim=0)

    Renames:
    - to_q -> proj_q (TRT-BNM adds a bias; initialize to zero)
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
    # TRT-BNM proj_q has a bias; customer does not. Initialize to zero.
    q_bias = torch.zeros(q_weight.shape[0], dtype=q_weight.dtype)

    return {
        f"{tbm_prefix}.norm_s.weight": state_dict[f"{prefix}.ln_1.weight"],
        f"{tbm_prefix}.norm_s.bias": state_dict[f"{prefix}.ln_1.bias"],
        f"{tbm_prefix}.proj_q.weight": q_weight,
        f"{tbm_prefix}.proj_q.bias": q_bias,
        f"{tbm_prefix}.proj_kv.weight": kv_weight,
        f"{tbm_prefix}.proj_g.weight": state_dict[f"{prefix}.to_g.weight"],
        f"{tbm_prefix}.proj_o.weight": state_dict[f"{prefix}.to_a.weight"],
        f"{tbm_prefix}.proj_z.0.weight": state_dict[f"{prefix}.ln_0.weight"],
        f"{tbm_prefix}.proj_z.0.bias": state_dict[f"{prefix}.ln_0.bias"],
        f"{tbm_prefix}.proj_z.1.weight": state_dict[f"{prefix}.to_b.weight"],
    }


def convert_pairformer_block_weights(state_dict, prefix="", tbm_prefix=""):
    """Convert a single BakerLab PairformerBlock to TRT-BNM PairformerLayerV1 weights.

    Args:
        state_dict: Customer checkpoint state_dict (or subset for one block).
        prefix: Key prefix for customer weights (e.g., "pairformer_stack.0").
        tbm_prefix: Key prefix for TRT-BNM weights (e.g., "layers.0").

    Returns:
        Dict of converted weights usable by both Torch and TRT backends.
    """
    dot = "." if prefix else ""
    tdot = "." if tbm_prefix else ""

    weights = {}
    weights.update(convert_tri_mul_weights(
        state_dict, f"{prefix}{dot}tri_mul_outgoing", f"{tbm_prefix}{tdot}tri_mul_out"))
    weights.update(convert_tri_mul_weights(
        state_dict, f"{prefix}{dot}tri_mul_incoming", f"{tbm_prefix}{tdot}tri_mul_in"))
    weights.update(convert_tri_attn_weights(
        state_dict, f"{prefix}{dot}tri_attn_start", f"{tbm_prefix}{tdot}tri_attn_start"))
    weights.update(convert_tri_attn_weights(
        state_dict, f"{prefix}{dot}tri_attn_end", f"{tbm_prefix}{tdot}tri_attn_end"))
    weights.update(convert_transition_weights(
        state_dict, f"{prefix}{dot}z_transition", f"{tbm_prefix}{tdot}transition_z"))
    weights.update(convert_transition_weights(
        state_dict, f"{prefix}{dot}s_transition", f"{tbm_prefix}{tdot}transition_s"))
    weights.update(convert_attention_pair_bias_weights(
        state_dict, f"{prefix}{dot}attention_pair_bias", f"{tbm_prefix}{tdot}attention"))
    return weights


def convert_pairformer_stack_weights(state_dict, num_blocks, prefix="pairformer_stack", tbm_prefix="layers"):
    """Convert a full stack of PairformerBlocks.

    Args:
        state_dict: Full model state_dict.
        num_blocks: Number of pairformer blocks.
        prefix: Customer prefix for the stack (e.g., "pairformer_stack").
        tbm_prefix: TRT-BNM prefix (e.g., "layers").

    Returns:
        Dict of converted weights for the full PairformerModule.
    """
    weights = {}
    for i in range(num_blocks):
        weights.update(convert_pairformer_block_weights(
            state_dict, f"{prefix}.{i}", f"{tbm_prefix}.{i}"))
    return weights
