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
"""Protenix checkpoint -> TRT-BioNeMo weight conversion.

The protenix-v2 checkpoint stores the OSS module names (``{"model": ...}`` /
``module.`` already stripped by the hub loader). The TRT-BioNeMo input embedder
reuses the shared fused DiT primitives, so the atom-transformer parameters have
a different layout and must be remapped (fused KV / SwiGLU / AdaLN
concatenations); atom feature projections are fused along their input channels.
"""

from __future__ import annotations

import torch

from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.models.boltz1.convert import (
    get_transition_weights,
    get_tri_attn_node_weights,
    get_tri_mul_node_weights,
)
from tensorrt_bionemo.utils import str_dtype_to_torch

_FUSED_ATOM_LINEARS = {
    "linear_no_bias_ref": ("ref_pos", "ref_charge", "f"),
    "linear_no_bias_pair": ("d", "invd", "v"),
}
_ATOM_LINEARS = ("cl", "cm", "q")


def _join(prefix: str, rest: str) -> str:
    return f"{prefix}.{rest}" if prefix else rest


def _convert_atom_encoder_linears(
    out: dict, weights: dict, src_prefix: str, dst_prefix: str, dtype: torch.dtype
) -> None:
    """Convert fused and passthrough atom-encoder projections."""
    for target, sources in _FUSED_ATOM_LINEARS.items():
        out[_join(dst_prefix, f"{target}.weight")] = torch.cat(
            [weights[_join(src_prefix, f"linear_no_bias_{source}.weight")] for source in sources],
            dim=-1,
        ).to(dtype)
    for name in _ATOM_LINEARS:
        key = f"linear_no_bias_{name}.weight"
        out[_join(dst_prefix, key)] = weights[_join(src_prefix, key)].to(dtype)


def _copy_to(out: dict, weights: dict, src: str, tgt: str, dtype: torch.dtype, *, unsqueeze: int | None = None) -> None:
    """Copy ``weights[src]`` into ``out[tgt]`` with an optional dtype cast."""
    value = weights[src].to(dtype)
    if unsqueeze is not None:
        value = value.unsqueeze(unsqueeze)
    out[tgt] = value


def _merge_prefix(out: dict, prefix: str, converted: dict) -> None:
    """Merge ``converted`` into ``out`` under ``prefix.`` (or flat if empty)."""
    if not prefix:
        out.update(converted)
        return
    for key, value in converted.items():
        out[f"{prefix}.{key}"] = value


def _convert_adaln(out: dict, weights: dict, src: str, tgt: str, dtype: torch.dtype) -> None:
    """OSS ``AdaptiveLayerNorm`` -> TRT-BioNeMo ``AdaLN``.

    Fuses ``[s_scale; s_bias]``; the s_bias half has a zeroed bias slot
    (OSS ``linear_nobias_s`` has no bias).
    """
    out[f"{tgt}.s_norm.weight"] = weights[f"{src}.layernorm_s.weight"].to(dtype)
    out[f"{tgt}.fused_s_scale_s_bias.weight"] = torch.cat(
        [weights[f"{src}.linear_s.weight"], weights[f"{src}.linear_nobias_s.weight"]], dim=0
    ).to(dtype)
    s_bias = weights[f"{src}.linear_s.bias"]
    out[f"{tgt}.fused_s_scale_s_bias.bias"] = torch.cat([s_bias, torch.zeros_like(s_bias)], dim=0).to(dtype)


def _convert_dit_attention(
    out: dict, weights: dict, src_apb: str, tgt_layer: str, tgt_pba: str, dtype: torch.dtype
) -> None:
    """Shared DiT attention remap: fused KV ``[k; v]``, proj_z LN+Linear, gate."""
    attn = f"{src_apb}.attention"
    out[f"{tgt_pba}.proj_q.weight"] = weights[f"{attn}.linear_q.weight"].to(dtype)
    out[f"{tgt_pba}.proj_q.bias"] = weights[f"{attn}.linear_q.bias"].to(dtype)
    # Fused KV pack order is part of the load contract: [k; v].
    out[f"{tgt_pba}.proj_kv.weight"] = torch.cat(
        [weights[f"{attn}.linear_k.weight"], weights[f"{attn}.linear_v.weight"]], dim=0
    ).to(dtype)
    out[f"{tgt_pba}.proj_g.weight"] = weights[f"{attn}.linear_g.weight"].to(dtype)
    out[f"{tgt_pba}.proj_o.weight"] = weights[f"{attn}.linear_o.weight"].to(dtype)
    out[f"{tgt_pba}.proj_z.0.weight"] = weights[f"{src_apb}.layernorm_z.weight"].to(dtype)
    out[f"{tgt_pba}.proj_z.1.weight"] = weights[f"{src_apb}.linear_nobias_z.weight"].to(dtype)
    out[f"{tgt_layer}.output_projection.weight"] = weights[f"{src_apb}.linear_a_last.weight"].to(dtype)
    out[f"{tgt_layer}.output_projection.bias"] = weights[f"{src_apb}.linear_a_last.bias"].to(dtype)


def _convert_dit_transition(out: dict, weights: dict, src_blk: str, tgt_layer: str, dtype: torch.dtype) -> None:
    """Shared conditioned transition: AdaLN + fused SwiGLU pack ``[a2; a1]``."""
    src_ct = f"{src_blk}.conditioned_transition_block"
    tgt_tr = f"{tgt_layer}.transition"
    _convert_adaln(out, weights, f"{src_ct}.adaln", f"{tgt_tr}.adaln", dtype)
    # FusedSwiGLU packs z[:d]=value, z[d:2d]=gate; OSS silu(a1)*a2 ->
    # gate=a1, value=a2 -> pack [a2, a1].
    out[f"{tgt_tr}.fused_swl_a_to_b.weight"] = torch.cat(
        [weights[f"{src_ct}.linear_nobias_a2.weight"], weights[f"{src_ct}.linear_nobias_a1.weight"]], dim=0
    ).to(dtype)
    out[f"{tgt_tr}.b_to_a.weight"] = weights[f"{src_ct}.linear_nobias_b.weight"].to(dtype)
    out[f"{tgt_tr}.output_projection.weight"] = weights[f"{src_ct}.linear_s.weight"].to(dtype)
    out[f"{tgt_tr}.output_projection.bias"] = weights[f"{src_ct}.linear_s.bias"].to(dtype)


def _convert_protenix_atom_dit_block(
    out: dict, weights: dict, src_blk: str, tgt_layer: str, dtype: torch.dtype
) -> None:
    """Atom/local DiT block: chained-KV AdaLN + shared attention/transition."""
    src_apb = f"{src_blk}.attention_pair_bias"
    tgt_pba = f"{tgt_layer}.pair_bias_attn"
    _convert_adaln(out, weights, f"{src_apb}.layernorm_a", f"{tgt_pba}.layer_norm_a_q", dtype)
    _convert_adaln(out, weights, f"{src_apb}.layernorm_kv", f"{tgt_pba}.layer_norm_a_k", dtype)
    _convert_dit_attention(out, weights, src_apb, tgt_layer, tgt_pba, dtype)
    _convert_dit_transition(out, weights, src_blk, tgt_layer, dtype)


def _convert_protenix_token_dit_block(
    out: dict, weights: dict, src_blk: str, tgt_layer: str, dtype: torch.dtype
) -> None:
    """Token/global DiT block: single AdaLN + shared attention/transition."""
    src_apb = f"{src_blk}.attention_pair_bias"
    tgt_pba = f"{tgt_layer}.pair_bias_attn"
    _convert_adaln(out, weights, f"{src_apb}.layernorm_a", f"{tgt_layer}.adaln", dtype)
    _convert_dit_attention(out, weights, src_apb, tgt_layer, tgt_pba, dtype)
    _convert_dit_transition(out, weights, src_blk, tgt_layer, dtype)


def convert_hf_input_embedder_torch(config: BaseConfig, weights: dict, prefix: str = "input_embedder") -> dict:
    """Convert protenix-v2 input-embedder weights to a TRT-BioNeMo state_dict.

    Args:
        config: ``InputFeatureEmbedderConfig`` for the target module (drives the
            block count via ``atom_transformer_config.num_blocks``, the dtype,
            and whether ESM is enabled).
        weights: checkpoint state dict (OSS module names). Input-embedder keys
            are looked up under ``prefix`` (e.g. ``input_embedder.*``).
        prefix: top-level key prefix of the input embedder in ``weights``. Pass
            ``""`` when ``weights`` is already input-embedder-relative.

    Returns:
        Flat state dict keyed by ``ProtenixInputFeatureEmbedder`` parameter
        names — load with ``module.load_state_dict(result)``.
    """
    dtype = config.torch_dtype
    num_blocks = config.atom_transformer_config.num_blocks
    ae = "atom_attention_encoder"

    def src_ae(rest: str) -> str:
        return _join(prefix, f"{ae}.{rest}")

    out: dict[str, torch.Tensor] = {}

    _convert_atom_encoder_linears(out, weights, _join(prefix, ae), ae, dtype)
    for i in (1, 3, 5):
        out[f"{ae}.small_mlp.{i}.weight"] = weights[src_ae(f"small_mlp.{i}.weight")].to(dtype)

    # Atom transformer: OSS DiffusionTransformerBlock -> shared
    # DiffusionTransformerLayer.
    for i in range(num_blocks):
        _convert_protenix_atom_dit_block(
            out,
            weights,
            src_ae(f"atom_transformer.diffusion_transformer.blocks.{i}"),
            f"{ae}.atom_transformer.layers.{i}",
            dtype,
        )

    if config.esm_enabled:
        out["linear_esm.weight"] = weights[_join(prefix, "linear_esm.weight")].to(dtype)

    return out


def convert_relative_position_encoding_torch(
    config: BaseConfig, weights: dict, prefix: str = "relative_position_encoding"
) -> dict:
    """Convert protenix-v2 relative-position-encoding weights to a state_dict.

    Protenix reuses the shared ``RelativePositionEncoder`` primitive, whose
    projection is named ``linear`` (vs OSS ``linear_no_bias``); the module is a
    single bias-free ``relp -> c_z`` projection, so this is a name remap + a
    dtype-cast copy.

    Args:
        config: ``RelativePositionEncodingConfig`` (drives the target dtype).
        weights: checkpoint state dict (OSS module names). The projection is
            looked up under ``prefix`` (e.g. ``relative_position_encoding.*``).
        prefix: top-level key prefix in ``weights``. Pass ``""`` when ``weights``
            is already relative to this module.

    Returns:
        Flat state dict keyed by shared ``RelativePositionEncoder`` parameter
        names — load with ``module.load_state_dict(result)``.
    """
    dtype = config.torch_dtype
    return {"linear.weight": weights[_join(prefix, "linear_no_bias.weight")].to(dtype)}


def convert_constraint_embedder_torch(config: BaseConfig, weights: dict, prefix: str = "constraint_embedder") -> dict:
    """Convert protenix-v2 constraint-embedder weights to a
    TRT-BioNeMo state_dict.

    Each enabled sub-embedder is a single bias-free ``c_z_input -> c_z``
    projection with matching OSS / TRT-BioNeMo names, so this is a per-embedder
    dtype-cast copy. Disabled sub-embedders contribute nothing (protenix-v2
    disables all of them -> empty dict).

    Args:
        config: ``ConstraintEmbedderConfig`` (its ``*_enable`` flags select
            which projections to convert; drives the target dtype).
        weights: checkpoint state dict (OSS module names), looked up under
            ``prefix``. Pass ``prefix=""`` when already constraint-relative.

    Returns:
        Flat state dict keyed by ``ProtenixConstraintEmbedder`` parameter names.
    """
    dtype = config.torch_dtype
    out: dict[str, torch.Tensor] = {}
    enabled = (
        (config.pocket_enable, "pocket_z_embedder"),
        (config.contact_enable, "contact_z_embedder"),
        (config.contact_atom_enable, "contact_atom_z_embedder"),
    )
    for is_enabled, name in enabled:
        if is_enabled:
            out[f"{name}.weight"] = weights[_join(prefix, f"{name}.weight")].to(dtype)
    return out


def _openfold_pairformer_block_intermediate(weights: dict, src_blk: str, dst_blk: str, out: dict) -> None:
    """Rename one OSS Protenix (OpenFold-style) pairformer block into the
    "Boltz-ish" intermediate names the shared ``get_*_node_weights`` helpers
    expect, keyed under ``dst_blk``. Fuses the tri-mul a/b projections
    (``p_in = [a_p, b_p]``, ``g_in = [a_g, b_g]``) and renames the pair
    transition SwiGLU (``linear_no_bias_a -> fc1``, ``linear_no_bias_b -> fc2``,
    ``linear_no_bias -> fc3``); OSS tri-attn already uses the OpenFold
    ``mha.linear_{q,k,v,o,g}`` / ``linear`` names the helpers read directly.
    """
    for mul in ("tri_mul_out", "tri_mul_in"):
        s, d = f"{src_blk}.{mul}", f"{dst_blk}.{mul}"
        out[f"{d}.norm_in.weight"] = weights[f"{s}.layer_norm_in.weight"]
        out[f"{d}.norm_in.bias"] = weights[f"{s}.layer_norm_in.bias"]
        out[f"{d}.norm_out.weight"] = weights[f"{s}.layer_norm_out.weight"]
        out[f"{d}.norm_out.bias"] = weights[f"{s}.layer_norm_out.bias"]
        out[f"{d}.p_in.weight"] = torch.cat(
            [weights[f"{s}.linear_a_p.weight"], weights[f"{s}.linear_b_p.weight"]], dim=0
        )
        out[f"{d}.g_in.weight"] = torch.cat(
            [weights[f"{s}.linear_a_g.weight"], weights[f"{s}.linear_b_g.weight"]], dim=0
        )
        out[f"{d}.p_out.weight"] = weights[f"{s}.linear_z.weight"]
        out[f"{d}.g_out.weight"] = weights[f"{s}.linear_g.weight"]

    for att in ("tri_att_start", "tri_att_end"):
        s, d = f"{src_blk}.{att}", f"{dst_blk}.{att}"
        out[f"{d}.layer_norm.weight"] = weights[f"{s}.layer_norm.weight"]
        out[f"{d}.layer_norm.bias"] = weights[f"{s}.layer_norm.bias"]
        out[f"{d}.linear.weight"] = weights[f"{s}.linear.weight"]
        for p in ("q", "k", "v", "o", "g"):
            out[f"{d}.mha.linear_{p}.weight"] = weights[f"{s}.mha.linear_{p}.weight"]

    s, d = f"{src_blk}.pair_transition", f"{dst_blk}.transition_z"
    out[f"{d}.norm.weight"] = weights[f"{s}.layernorm1.weight"]
    out[f"{d}.norm.bias"] = weights[f"{s}.layernorm1.bias"]
    out[f"{d}.fc1.weight"] = weights[f"{s}.linear_no_bias_a.weight"]
    out[f"{d}.fc2.weight"] = weights[f"{s}.linear_no_bias_b.weight"]
    out[f"{d}.fc3.weight"] = weights[f"{s}.linear_no_bias.weight"]


def _convert_pair_path(
    weights: dict,
    src_pair: str,
    tgt_layer: str,
    out: dict,
    dtype_str: str,
    transition_dim: int,
    transition_name: str = "transition_z",
) -> None:
    """OpenFold-style pair block -> TRT-BioNeMo keys via shared Boltz1 helpers.

    ``transition_name`` is the target pair-transition attribute
    (``transition_z`` or ``pair_transition``).
    """
    pf_sd: dict[str, torch.Tensor] = {}
    _openfold_pairformer_block_intermediate(weights, src_pair, "b", pf_sd)
    out.update(get_tri_mul_node_weights(pf_sd, "b.tri_mul_out", f"{tgt_layer}.tri_mul_out", dtype=dtype_str))
    out.update(get_tri_mul_node_weights(pf_sd, "b.tri_mul_in", f"{tgt_layer}.tri_mul_in", dtype=dtype_str))
    out.update(get_tri_attn_node_weights(pf_sd, "b.tri_att_start", f"{tgt_layer}.tri_attn_start", dtype=dtype_str))
    out.update(get_tri_attn_node_weights(pf_sd, "b.tri_att_end", f"{tgt_layer}.tri_attn_end", dtype=dtype_str))
    out.update(
        get_transition_weights(
            pf_sd, "b.transition_z", f"{tgt_layer}.{transition_name}", dim=transition_dim, dtype=dtype_str
        )
    )


def convert_template_embedder_torch(config: BaseConfig, weights: dict, prefix: str = "template_embedder") -> dict:
    """Convert protenix-v2 template-embedder weights to a TRT-BioNeMo
    state_dict.

    Outer projections keep identical names; the inner pair stack routes through
    :func:`_convert_pair_path` (same path as MSA/pairformer converters).
    """
    dtype = config.torch_dtype
    # Inner pair stack weights match pairformer_dtype (bf16) for plain load.
    pf_dtype_str = config.pairformer_dtype
    out: dict[str, torch.Tensor] = {}

    for ln in ("layernorm_z", "layernorm_v"):
        _copy_to(out, weights, _join(prefix, f"{ln}.weight"), f"{ln}.weight", dtype)
        _copy_to(out, weights, _join(prefix, f"{ln}.bias"), f"{ln}.bias", dtype)
    for lin in ("linear_no_bias_z", "linear_no_bias_a", "linear_no_bias_u"):
        _copy_to(out, weights, _join(prefix, f"{lin}.weight"), f"{lin}.weight", dtype)

    for i in range(config.n_blocks):
        _convert_pair_path(
            weights,
            _join(prefix, f"pairformer_stack.blocks.{i}"),
            f"pairformer_stack.layers.{i}",
            out,
            pf_dtype_str,
            transition_dim=config.c * config.num_intermediate_factor,
        )
    return out


def _convert_pairformer_single_path(weights: dict, src_blk: str, tgt_layer: str, out: dict, dtype: torch.dtype) -> None:
    """Convert the OSS ``attention_pair_bias`` + ``single_transition`` (single
    ``s`` path) of one pairformer block into flat ``AttentionPairBias`` /
    ``Transition`` keys under ``tgt_layer``."""
    apb = f"{src_blk}.attention_pair_bias"
    attn = f"{apb}.attention"
    a = f"{tgt_layer}.attention"
    out[f"{a}.norm_s.weight"] = weights[f"{apb}.layernorm_a.weight"].to(dtype)
    out[f"{a}.norm_s.bias"] = weights[f"{apb}.layernorm_a.bias"].to(dtype)
    out[f"{a}.proj_q.weight"] = weights[f"{attn}.linear_q.weight"].to(dtype)
    out[f"{a}.proj_q.bias"] = weights[f"{attn}.linear_q.bias"].to(dtype)
    out[f"{a}.proj_kv.weight"] = torch.cat(
        [weights[f"{attn}.linear_k.weight"], weights[f"{attn}.linear_v.weight"]], dim=0
    ).to(dtype)
    out[f"{a}.proj_g.weight"] = weights[f"{attn}.linear_g.weight"].to(dtype)
    out[f"{a}.proj_o.weight"] = weights[f"{attn}.linear_o.weight"].to(dtype)
    out[f"{a}.proj_z.0.weight"] = weights[f"{apb}.layernorm_z.weight"].to(dtype)
    out[f"{a}.proj_z.0.bias"] = weights[f"{apb}.layernorm_z.bias"].to(dtype)
    out[f"{a}.proj_z.1.weight"] = weights[f"{apb}.linear_nobias_z.weight"].to(dtype)

    st = f"{src_blk}.single_transition"
    t = f"{tgt_layer}.transition_s"
    out[f"{t}.norm.weight"] = weights[f"{st}.layernorm1.weight"].to(dtype)
    out[f"{t}.norm.bias"] = weights[f"{st}.layernorm1.bias"].to(dtype)
    out[f"{t}.fused_fc2_fc1.weight"] = torch.cat(
        [weights[f"{st}.linear_no_bias_b.weight"], weights[f"{st}.linear_no_bias_a.weight"]], dim=0
    ).to(dtype)
    out[f"{t}.fc3.weight"] = weights[f"{st}.linear_no_bias.weight"].to(dtype)


def convert_pairformer_stack_torch(config: BaseConfig, weights: dict, prefix: str = "pairformer_stack") -> dict:
    """Convert protenix-v2 pairformer-stack weights to a ``PairformerModule``
    state_dict.

    Reuses the shared Boltz1 helpers for the OpenFold-style pair path and maps
    the OSS ``attention_pair_bias`` / ``single_transition`` single path to
    ``AttentionPairBias`` / ``Transition``. Flat state_dict -> ``load_state_dict``.

    Args:
        config: ``PairformerConfig`` (block count, dims, dtype).
        weights: checkpoint state dict (OSS names) under ``prefix``.
        prefix: top-level key prefix (``""`` when already stack-relative).

    Returns:
        Flat state dict keyed by ``PairformerModule`` parameter names.
    """
    dtype = config.torch_dtype
    dtype_str = config.dtype
    out: dict[str, torch.Tensor] = {}
    for i in range(config.num_blocks):
        src_blk = _join(prefix, f"blocks.{i}")
        layer = f"layers.{i}"
        _convert_pair_path(weights, src_blk, layer, out, dtype_str, config.token_z * 4)
        _convert_pairformer_single_path(weights, src_blk, layer, out, dtype)
    return out


def _convert_opm(weights: dict, src: str, tgt: str, out: dict, dtype: torch.dtype) -> None:
    """OSS ``OuterProductMean`` -> TRT-BioNeMo ``OuterProductMean``
    (fused a/b)."""
    out[f"{tgt}.norm.weight"] = weights[f"{src}.layer_norm.weight"].to(dtype)
    out[f"{tgt}.norm.bias"] = weights[f"{src}.layer_norm.bias"].to(dtype)
    out[f"{tgt}.fused_proj_a_b.weight"] = torch.cat(
        [weights[f"{src}.linear_1.weight"], weights[f"{src}.linear_2.weight"]], dim=0
    ).to(dtype)
    out[f"{tgt}.proj_o.weight"] = weights[f"{src}.linear_out.weight"].to(dtype)
    out[f"{tgt}.proj_o.bias"] = weights[f"{src}.linear_out.bias"].to(dtype)


def _convert_msa_pwa(weights: dict, src: str, tgt: str, out: dict, dtype: torch.dtype) -> None:
    """OSS ``MSAPairWeightedAveraging`` -> TRT-BioNeMo
    ``PairWeightedAveraging`` (fused v/g)."""
    out[f"{tgt}.norm_m.weight"] = weights[f"{src}.layernorm_m.weight"].to(dtype)
    out[f"{tgt}.norm_m.bias"] = weights[f"{src}.layernorm_m.bias"].to(dtype)
    out[f"{tgt}.norm_z.weight"] = weights[f"{src}.layernorm_z.weight"].to(dtype)
    out[f"{tgt}.norm_z.bias"] = weights[f"{src}.layernorm_z.bias"].to(dtype)
    out[f"{tgt}.fused_proj_m_g.weight"] = torch.cat(
        [weights[f"{src}.linear_no_bias_mv.weight"], weights[f"{src}.linear_no_bias_mg.weight"]], dim=0
    ).to(dtype)
    out[f"{tgt}.proj_z.weight"] = weights[f"{src}.linear_no_bias_z.weight"].to(dtype)
    out[f"{tgt}.proj_o.weight"] = weights[f"{src}.linear_no_bias_out.weight"].to(dtype)


def _convert_swiglu_transition(weights: dict, src: str, tgt: str, out: dict, dtype: torch.dtype) -> None:
    """OSS ``Transition`` (SwiGLU) -> TRT-BioNeMo ``Transition``
    (fused fc2/fc1)."""
    out[f"{tgt}.norm.weight"] = weights[f"{src}.layernorm1.weight"].to(dtype)
    out[f"{tgt}.norm.bias"] = weights[f"{src}.layernorm1.bias"].to(dtype)
    out[f"{tgt}.fused_fc2_fc1.weight"] = torch.cat(
        [weights[f"{src}.linear_no_bias_b.weight"], weights[f"{src}.linear_no_bias_a.weight"]], dim=0
    ).to(dtype)
    out[f"{tgt}.fc3.weight"] = weights[f"{src}.linear_no_bias.weight"].to(dtype)


def convert_msa_module_torch(config: BaseConfig, weights: dict, prefix: str = "msa_module") -> dict:
    """Convert protenix-v2 MSA-module weights to a ``ProtenixMSAModule``
    state_dict.

    Maps the Protenix feature-embedding linears + per-block OPM /
    MSA-pair-weighted-averaging / transition / pair stack onto the reused
    ``MSAModuleStack``. The last block drops the MSA sublayers. Flat state_dict
    -> ``load_state_dict``.

    Args:
        config: ``ProtenixMSAModuleConfig``.
        weights: checkpoint state dict (OSS names) under ``prefix``.
        prefix: top-level key prefix (``""`` when already MSA-relative).

    Returns:
        Flat state dict keyed by ``ProtenixMSAModule`` parameter names.
    """
    dtype = config.torch_dtype
    dtype_str = config.dtype
    out: dict[str, torch.Tensor] = {}

    # Feature-embedding wrapper (identical names).
    for lin in ("linear_no_bias_m", "linear_no_bias_s"):
        out[f"{lin}.weight"] = weights[_join(prefix, f"{lin}.weight")].to(dtype)

    for i in range(config.no_blocks):
        src_blk = _join(prefix, f"blocks.{i}")
        tgt = f"msa_stack.blocks.{i}"
        _convert_opm(weights, f"{src_blk}.outer_product_mean_msa", f"{tgt}.outer_product_mean", out, dtype)
        if i != config.no_blocks - 1:  # non-last block keeps the MSA sublayers
            _convert_msa_pwa(
                weights, f"{src_blk}.msa_stack.msa_pair_weighted_averaging", f"{tgt}.msa_att_row", out, dtype
            )
            _convert_swiglu_transition(
                weights, f"{src_blk}.msa_stack.transition_m", f"{tgt}.msa_transition", out, dtype
            )
        _convert_pair_path(
            weights,
            f"{src_blk}.pair_stack",
            tgt,
            out,
            dtype_str,
            config.c_z * config.transition_n,
            transition_name="pair_transition",
        )
    return out


def convert_diffusion_conditioning_torch(
    config: BaseConfig, weights: dict, prefix: str = "diffusion_module.diffusion_conditioning"
) -> dict:
    """Convert protenix-v2 diffusion-conditioning weights to a state_dict.

    The Protenix ``DiffusionConditioning`` reuses shared primitives: the
    ``relpe`` (``RelativePositionEncoder``, ``linear_no_bias -> linear``), two
    pair + two single SwiGLU ``Transition`` s (fused fc2/fc1), the
    ``FourierEmbedding`` (OSS frozen ``w`` / ``b`` -> ``proj`` weight ``[c, 1]``
    / bias), and bias-free ``Linear`` / scale-only ``LayerNorm`` projections.

    Args:
        config: ``DiffusionConditioningConfig`` (drives the fp32 projection /
            single-path dtype, cached ``z_pair_dtype``, and transition hidden
            dims).
        weights: checkpoint state dict (OSS names), looked up under ``prefix``
            (e.g. ``diffusion_module.diffusion_conditioning.*``). Pass
            ``prefix=""`` when already conditioning-relative.

    Returns:
        Flat state dict keyed by ``ProtenixDiffusionConditioning`` parameter
        names — load with ``module.load_state_dict(result)``.
    """
    dtype = config.torch_dtype
    z_pair_dtype = str_dtype_to_torch(config.z_pair_dtype)
    out: dict[str, torch.Tensor] = {}

    # relpe (shared RPE): OSS linear_no_bias -> linear.
    out["relpe.linear.weight"] = weights[_join(prefix, "relpe.linear_no_bias.weight")].to(dtype)

    # Scale-only LayerNorms + bias-free projections (identical names).
    for ln in ("layernorm_z", "layernorm_s", "layernorm_n"):
        out[f"{ln}.weight"] = weights[_join(prefix, f"{ln}.weight")].to(dtype)
    for lin in ("linear_no_bias_z", "linear_no_bias_s", "linear_no_bias_n"):
        out[f"{lin}.weight"] = weights[_join(prefix, f"{lin}.weight")].to(dtype)

    # FourierEmbedding: OSS frozen vectors w / b -> Linear(1, c) proj.
    out["fourier_embedding.proj.weight"] = weights[_join(prefix, "fourier_embedding.w")].to(dtype).unsqueeze(-1)
    out["fourier_embedding.proj.bias"] = weights[_join(prefix, "fourier_embedding.b")].to(dtype)

    # Pair transitions run in the cached ``z_pair_dtype`` (bf16); the
    # noise-dependent single transitions stay at the module dtype (fp32).
    for src, tgt in (("transition_z1", "transition_z.0"), ("transition_z2", "transition_z.1")):
        _convert_swiglu_transition(weights, _join(prefix, src), tgt, out, z_pair_dtype)
    for src, tgt in (("transition_s1", "transition_s.0"), ("transition_s2", "transition_s.1")):
        _convert_swiglu_transition(weights, _join(prefix, src), tgt, out, dtype)
    return out


def convert_diffusion_atom_encoder_torch(
    config: BaseConfig, weights: dict, prefix: str = "diffusion_module.atom_attention_encoder"
) -> dict:
    """Convert protenix-v2 diffusion atom-encoder (``has_coords=True``) weights.

    Same layout as the input-embedder atom encoder (reference-conformer
    projections + ``small_mlp`` + the shared atom
    ``ProtenixDiffusionTransformer``, converted per-block by
    :func:`_convert_protenix_atom_dit_block`) plus the coordinate-conditioning
    projections: scale-only ``layernorm_s`` / ``layernorm_z`` and bias-free
    ``linear_no_bias_s`` / ``linear_no_bias_z`` / ``linear_no_bias_r`` (all
    identical OSS / TRT-BioNeMo names).

    Args:
        config: ``DiffusionAtomAttentionEncoderConfig`` (block count via
            ``atom_transformer_config.num_blocks``; drives the dtype).
        weights: checkpoint state dict (OSS names), looked up under ``prefix``
            (e.g. ``diffusion_module.atom_attention_encoder.*``). Pass
            ``prefix=""`` when already encoder-relative.

    Returns:
        Flat state dict keyed by ``ProtenixAtomAttentionEncoder`` (has_coords)
        parameter names.
    """
    dtype = config.torch_dtype
    out: dict[str, torch.Tensor] = {}

    _convert_atom_encoder_linears(out, weights, prefix, "", dtype)
    for i in (1, 3, 5):
        out[f"small_mlp.{i}.weight"] = weights[_join(prefix, f"small_mlp.{i}.weight")].to(dtype)

    # has_coords conditioning: scale-only LayerNorms + bias-free projections.
    for ln in ("layernorm_s", "layernorm_z"):
        out[f"{ln}.weight"] = weights[_join(prefix, f"{ln}.weight")].to(dtype)
    for lin in ("linear_no_bias_s", "linear_no_bias_z", "linear_no_bias_r"):
        out[f"{lin}.weight"] = weights[_join(prefix, f"{lin}.weight")].to(dtype)

    for i in range(config.atom_transformer_config.num_blocks):
        _convert_protenix_atom_dit_block(
            out,
            weights,
            _join(prefix, f"atom_transformer.diffusion_transformer.blocks.{i}"),
            f"atom_transformer.layers.{i}",
            dtype,
        )
    return out


def convert_atom_attention_decoder_torch(
    config: BaseConfig, weights: dict, prefix: str = "diffusion_module.atom_attention_decoder"
) -> dict:
    """Convert protenix-v2 atom-attention-decoder weights to a state_dict.

    The reference-conformer-free decoder is a token->atom projection
    (``linear_no_bias_a``), the shared local-windowed atom
    ``ProtenixDiffusionTransformer`` (converted per-block by
    :func:`_convert_protenix_atom_dit_block`), a scale-only ``layernorm_q``, and
    the coordinate-update projection ``linear_no_bias_out``.

    Args:
        config: ``AtomAttentionDecoderConfig`` (block count via
            ``atom_transformer_config.num_blocks``; drives the dtype).
        weights: checkpoint state dict (OSS names), looked up under ``prefix``
            (e.g. ``diffusion_module.atom_attention_decoder.*``). Pass
            ``prefix=""`` when already decoder-relative.

    Returns:
        Flat state dict keyed by ``ProtenixAtomAttentionDecoder`` parameter
        names — load with ``module.load_state_dict(result)``.
    """
    dtype = config.torch_dtype
    out: dict[str, torch.Tensor] = {}

    for name in ("linear_no_bias_a", "linear_no_bias_out"):
        out[f"{name}.weight"] = weights[_join(prefix, f"{name}.weight")].to(dtype)
    out["layernorm_q.weight"] = weights[_join(prefix, "layernorm_q.weight")].to(dtype)

    for i in range(config.atom_transformer_config.num_blocks):
        _convert_protenix_atom_dit_block(
            out,
            weights,
            _join(prefix, f"atom_transformer.diffusion_transformer.blocks.{i}"),
            f"atom_transformer.layers.{i}",
            dtype,
        )
    return out


def convert_diffusion_module_torch(config: BaseConfig, weights: dict, prefix: str = "diffusion_module") -> dict:
    """Combine conditioning / atom enc-dec / token DiT converters + Alg. 20 LN/Linear."""
    dtype = config.torch_dtype
    out: dict[str, torch.Tensor] = {}

    # Alg. 20 line 4 projections (OSS layernorm_s is scale-only).
    _copy_to(out, weights, _join(prefix, "layernorm_s.weight"), "layernorm_s.weight", dtype)
    _copy_to(out, weights, _join(prefix, "layernorm_a.weight"), "layernorm_a.weight", dtype)
    _copy_to(out, weights, _join(prefix, "linear_no_bias_s.weight"), "linear_no_bias_s.weight", dtype)

    _merge_prefix(
        out,
        "diffusion_conditioning",
        convert_diffusion_conditioning_torch(
            config.diffusion_conditioning_config, weights, prefix=_join(prefix, "diffusion_conditioning")
        ),
    )
    _merge_prefix(
        out,
        "atom_attention_encoder",
        convert_diffusion_atom_encoder_torch(
            config.atom_encoder_config, weights, prefix=_join(prefix, "atom_attention_encoder")
        ),
    )
    _merge_prefix(
        out,
        "atom_attention_decoder",
        convert_atom_attention_decoder_torch(
            config.atom_decoder_config, weights, prefix=_join(prefix, "atom_attention_decoder")
        ),
    )

    tt_prefix = _join(prefix, "diffusion_transformer")
    for i in range(config.token_transformer_config.num_blocks):
        _convert_protenix_token_dit_block(
            out, weights, f"{tt_prefix}.blocks.{i}", f"diffusion_transformer.layers.{i}", dtype
        )
    return out


def convert_input_projections_torch(config: BaseConfig, weights: dict, prefix: str = "") -> dict:
    """Model-level s/z init projections (AF3 Alg. 1); identical OSS names."""
    dtype = config.torch_dtype
    names = ("linear_no_bias_sinit", "linear_no_bias_zinit1", "linear_no_bias_zinit2", "linear_no_bias_token_bond")
    return {name: weights[_join(prefix, f"{name}.weight")].to(dtype) for name in names}


def convert_trunk_torch(config: BaseConfig, weights: dict, prefix: str = "") -> dict:
    """Recycling projections + template / MSA / pairformer sub-converters."""
    dtype = config.torch_dtype
    out: dict[str, torch.Tensor] = {}

    for ln in ("layernorm_z_cycle", "layernorm_s"):
        _copy_to(out, weights, _join(prefix, f"{ln}.weight"), f"{ln}.weight", dtype)
        _copy_to(out, weights, _join(prefix, f"{ln}.bias"), f"{ln}.bias", dtype)
    for lin in ("linear_no_bias_z_cycle", "linear_no_bias_s"):
        _copy_to(out, weights, _join(prefix, f"{lin}.weight"), f"{lin}.weight", dtype)

    _merge_prefix(
        out,
        "template_embedder",
        convert_template_embedder_torch(
            config.template_embedder_config, weights, prefix=_join(prefix, "template_embedder")
        ),
    )
    _merge_prefix(
        out,
        "msa_module",
        convert_msa_module_torch(config.msa_module_config, weights, prefix=_join(prefix, "msa_module")),
    )
    _merge_prefix(
        out,
        "pairformer_stack",
        convert_pairformer_stack_torch(config.pairformer_config, weights, prefix=_join(prefix, "pairformer_stack")),
    )
    return out


def convert_distogram_head_torch(config: BaseConfig, weights: dict, prefix: str = "distogram_head") -> dict:
    """Distogram head: identical ``linear.weight`` / ``linear.bias`` names."""
    dtype = config.torch_dtype
    return {
        "linear.weight": weights[_join(prefix, "linear.weight")].to(dtype),
        "linear.bias": weights[_join(prefix, "linear.bias")].to(dtype),
    }


def convert_confidence_head_torch(config: BaseConfig, weights: dict, prefix: str = "confidence_head") -> dict:
    """Confidence head projections/norms + pairformer sub-conversion."""
    dtype = config.torch_dtype
    out: dict[str, torch.Tensor] = {}
    for lin in (
        "linear_no_bias_s1",
        "linear_no_bias_s2",
        "linear_no_bias_d",
        "linear_no_bias_d_wo_onehot",
        "linear_no_bias_pae",
        "linear_no_bias_pde",
    ):
        _copy_to(out, weights, _join(prefix, f"{lin}.weight"), f"{lin}.weight", dtype)
    for param in ("plddt_weight", "resolved_weight"):
        _copy_to(out, weights, _join(prefix, param), param, dtype)
    for ln in ("input_strunk_ln", "pae_ln", "pde_ln", "plddt_ln", "resolved_ln"):
        _copy_to(out, weights, _join(prefix, f"{ln}.weight"), f"{ln}.weight", dtype)
        _copy_to(out, weights, _join(prefix, f"{ln}.bias"), f"{ln}.bias", dtype)
    _merge_prefix(
        out,
        "pairformer_stack",
        convert_pairformer_stack_torch(config.pairformer_config, weights, prefix=_join(prefix, "pairformer_stack")),
    )
    return out
