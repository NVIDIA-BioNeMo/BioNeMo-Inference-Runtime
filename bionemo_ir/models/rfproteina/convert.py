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
"""RFProteína checkpoint -> TRT-BNM weight conversion.

The released Lightning checkpoint stores the OSS module names under an ``nn.``
prefix (the score network). Conversion is *forward*: read the known OSS source
keys and write the TRT-BNM ``LocalLatentsTransformer`` parameter names, driven
by the module ``config`` (block counts / toggles). Most names carry over; the
remaps that fuse or restructure OSS projections are:

  * velocity feature factories -> native representation-builder embeddings,
    projections, norms, and pair-conditioning AdaLN.
  * ``AdaptiveLayerNorm`` (``norm_cond`` / ``to_gamma`` / ``to_beta``) -> shared
    ``AdaLN`` (``s_norm`` + fused ``fused_s_scale_s_bias`` = ``[gamma; beta]``,
    beta's bias half zeroed).
  * ``PairBiasAttention`` -> shared ``AttentionPairBias``: fused ``to_qkv``
    (order q, k, v) and the gate ``to_g`` packed into ``in_proj``
    (= ``[q; g; k; v]``); ``node_norm`` / ``q_layer_norm`` / ``k_layer_norm`` /
    ``to_out_node`` / ``pair_norm`` + ``to_bias`` renamed.
  * ``Transition`` (``swish_linear.0`` / ``linear_out``) -> shared ``Transition``
    (``fused_fc2_fc1`` / ``fc3``).
  * ``TransitionADALN`` -> shared ``ConditionedTransitionBlock``
    (``fused_swl_a_to_b`` / ``b_to_a`` + AdaLN-zero ``output_projection``).
  * triangle-multiplication outgoing branch projections concatenated
    (``p_in = [a_p; b_p]``, ``g_in = [a_g; b_g]``).
  * the unified encoder's element and charge embeddings multiplied into their
    atom37 slots' columns of the single projection, which keeps only the dense
    columns (see :func:`fold_atom_chemistry`).
  * the OSS ``mhba`` attention wrapper is folded directly into the trunk layer.
"""

from __future__ import annotations

import importlib
import io
import pickle
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from bionemo_ir._torch.layers.transformers.local_latents_decoder import ATOM37_SIZE
from bionemo_ir._torch.modules.rfproteina.contracts import RFProteinaConversionReport
from bionemo_ir.configs import BaseConfig
from bionemo_ir.models.rfproteina.config import (
    RFProteinaDecoderConfig,
    RFProteinaEncoderConfig,
    RFProteinaRepresentationConfig,
    RFProteinaTrunkConfig,
)
from bionemo_ir.utils import str_dtype_to_torch

_SOURCE_PREFIX = "nn"
_ENCODER_SOURCE_PREFIX = "encoder"
_DECODER_SOURCE_PREFIX = "decoder"

#: Trunk config fields that are pinned by the checkpoint's tensor shapes, as
#: opposed to runtime knobs (backend, dtype, masking) chosen by the caller.
_ARCHITECTURE_FIELDS = (
    "token_dim",
    "pair_dim",
    "condition_dim",
    "latent_dim",
    "num_layers",
    "num_heads",
    "condition_transition_expansion",
    "transition_expansion",
    "pair_update_every",
    "trimul_hidden_dim",
    "trimul_bias",
    "pair_transition_expansion",
    "pair_write",
    "recycle_enabled",
    "use_qk_norm",
    "use_insertion_head",
    "use_sequence_readout",
)
_ENCODER_ARCHITECTURE_FIELDS = (
    "token_dim",
    "pair_dim",
    "condition_dim",
    "latent_dim",
    "num_layers",
    "num_heads",
    "condition_transition_expansion",
    "transition_expansion",
    "num_residue_types",
    "input_schema",
    "token_type_embedding_dim",
    "use_pair_bias",
    "sequence_separation_bins",
    "distance_bins",
    "orientation_bins",
    "use_qk_norm",
)


def _join(prefix: str, rest: str) -> str:
    """Join a key prefix and remainder (``rest`` alone when prefix is empty)."""
    return f"{prefix}.{rest}" if prefix else rest


def _copy(out: dict, weights: Mapping[str, torch.Tensor], src: str, tgt: str, dtype: torch.dtype) -> None:
    """Copy ``weights[src]`` into ``out[tgt]`` with a dtype cast."""
    out[tgt] = weights[src].to(dtype)


def _cat(weights: Mapping[str, torch.Tensor], srcs: tuple[str, ...], dtype: torch.dtype) -> torch.Tensor:
    """Concatenate several source tensors along dim 0 (fused projections)."""
    return torch.cat([weights[s] for s in srcs], dim=0).to(dtype)


def _convert_adaln(
    out: dict, weights: Mapping[str, torch.Tensor], src: str, tgt: str, dtype: torch.dtype, norm_bias: bool = False
) -> None:
    """OSS ``AdaptiveLayerNorm`` -> shared ``AdaLN``.

    Fuses ``[to_gamma; to_beta]`` into ``fused_s_scale_s_bias``; ``to_beta`` is
    bias-free, so the bias half is zero-padded. ``norm_bias`` carries the
    condition-norm bias across for the LayerNorm-trained autoencoder.
    """
    _copy(out, weights, f"{src}.norm_cond.weight", f"{tgt}.s_norm.weight", dtype)
    if norm_bias:
        _copy(out, weights, f"{src}.norm_cond.bias", f"{tgt}.s_norm.bias", dtype)
    out[f"{tgt}.fused_s_scale_s_bias.weight"] = _cat(
        weights, (f"{src}.to_gamma.0.weight", f"{src}.to_beta.weight"), dtype
    )
    gamma_b = weights[f"{src}.to_gamma.0.bias"]
    out[f"{tgt}.fused_s_scale_s_bias.bias"] = torch.cat([gamma_b, torch.zeros_like(gamma_b)], dim=0).to(dtype)


def _convert_adaln_zero_gate(
    out: dict, weights: Mapping[str, torch.Tensor], src_scale_output: str, tgt_linear: str, dtype: torch.dtype
) -> None:
    """OSS ``AdaptiveOutputScale`` gate linear -> a plain target linear."""
    gate = f"{src_scale_output}.to_adaln_zero_gamma.0"
    _copy(out, weights, f"{gate}.weight", f"{tgt_linear}.weight", dtype)
    _copy(out, weights, f"{gate}.bias", f"{tgt_linear}.bias", dtype)


def _convert_attention(
    out: dict,
    weights: Mapping[str, torch.Tensor],
    src: str,
    tgt: str,
    dtype: torch.dtype,
    use_qk_norm: bool,
    norm_bias: bool = False,
    use_pair_bias: bool = True,
) -> None:
    """OSS ``PairBiasAttention`` -> shared ``AttentionPairBias``.

    The fused ``to_qkv`` (order q, k, v) and the gate ``to_g`` pack into
    ``in_proj`` as ``[q; g; k; v]`` — the segment order is part of the load
    contract. ``norm_bias`` carries the LayerNorm biases across for the
    autoencoder, whose norms are not RMS.
    """

    def copy_norm(src_name: str, tgt_name: str) -> None:
        _copy(out, weights, f"{src}.{src_name}.weight", f"{tgt}.{tgt_name}.weight", dtype)
        if norm_bias:
            _copy(out, weights, f"{src}.{src_name}.bias", f"{tgt}.{tgt_name}.bias", dtype)

    copy_norm("node_norm", "norm_s")
    if use_qk_norm:
        copy_norm("q_layer_norm", "q_norm")
        copy_norm("k_layer_norm", "k_norm")

    for suffix in ("weight", "bias"):
        q, k, v = weights[f"{src}.to_qkv.{suffix}"].chunk(3)
        out[f"{tgt}.in_proj.{suffix}"] = torch.cat((q, weights[f"{src}.to_g.{suffix}"], k, v)).to(dtype)

    _copy(out, weights, f"{src}.to_out_node.weight", f"{tgt}.proj_o.weight", dtype)
    _copy(out, weights, f"{src}.to_out_node.bias", f"{tgt}.proj_o.bias", dtype)
    if use_pair_bias:
        copy_norm("pair_norm", "proj_z.0")
        _copy(out, weights, f"{src}.to_bias.weight", f"{tgt}.proj_z.1.weight", dtype)


def _convert_transition(out: dict, weights: Mapping[str, torch.Tensor], src: str, tgt: str, dtype: torch.dtype) -> None:
    """OSS ``Transition`` (bias-free SwiGLU) -> shared ``Transition``."""
    _copy(out, weights, f"{src}.swish_linear.0.weight", f"{tgt}.fused_fc2_fc1.weight", dtype)
    _copy(out, weights, f"{src}.linear_out.weight", f"{tgt}.fc3.weight", dtype)


def _convert_conditioned_transition(
    out: dict, weights: Mapping[str, torch.Tensor], src: str, tgt: str, dtype: torch.dtype, norm_bias: bool = False
) -> None:
    """OSS ``TransitionADALN`` -> shared ``ConditionedTransitionBlock``."""
    _convert_adaln(out, weights, f"{src}.adaln", f"{tgt}.adaln", dtype, norm_bias)
    _copy(out, weights, f"{src}.transition.swish_linear.0.weight", f"{tgt}.fused_swl_a_to_b.weight", dtype)
    _copy(out, weights, f"{src}.transition.linear_out.weight", f"{tgt}.b_to_a.weight", dtype)
    _convert_adaln_zero_gate(out, weights, f"{src}.scale_output", f"{tgt}.output_projection", dtype)


def _convert_trunk_layer(
    out: dict,
    weights: Mapping[str, torch.Tensor],
    src: str,
    tgt: str,
    dtype: torch.dtype,
    use_qk_norm: bool,
    norm_bias: bool = False,
    use_pair_bias: bool = True,
) -> None:
    """OSS ``MultiheadAttnAndTransition`` -> ``RFProteinaTrunkLayer``.

    The OSS ``mhba`` attention wrapper is folded into the trunk layer: its
    ``adaln`` / ``mha`` / ``scale_output`` map to the layer's ``adaln`` /
    ``mha`` / ``mha.output_projection``, the attention's output gate, which
    the fused epilogue applies with the residual add. ``norm_bias`` selects
    the autoencoder's LayerNorm variant over the velocity trunk's RMSNorm.
    """
    _convert_adaln(out, weights, f"{src}.mhba.adaln", f"{tgt}.adaln", dtype, norm_bias)
    _convert_attention(
        out,
        weights,
        f"{src}.mhba.mha",
        f"{tgt}.mha",
        dtype,
        use_qk_norm,
        norm_bias,
        use_pair_bias,
    )
    _convert_adaln_zero_gate(out, weights, f"{src}.mhba.scale_output", f"{tgt}.mha.output_projection", dtype)
    _convert_conditioned_transition(out, weights, f"{src}.transition", f"{tgt}.transition", dtype, norm_bias)


def _convert_trimul(out: dict, weights: Mapping[str, torch.Tensor], src: str, tgt: str, dtype: torch.dtype) -> None:
    """OSS triangle-multiplication node -> shared ``TriangleMultiplicationNode``.

    The outgoing a/b branch projections are concatenated (``p_in = [a_p; b_p]``,
    ``g_in = [a_g; b_g]``); the rest are renames.
    """
    for tgt_name, src_name in (
        ("norm_in", "layer_norm_in"),
        ("norm_out", "layer_norm_out"),
        ("p_out", "linear_z"),
        ("g_out", "linear_g"),
    ):
        _copy(out, weights, f"{src}.{src_name}.weight", f"{tgt}.{tgt_name}.weight", dtype)
        bias_key = f"{src}.{src_name}.bias"
        if bias_key in weights:
            _copy(out, weights, bias_key, f"{tgt}.{tgt_name}.bias", dtype)
    for tgt_name, a, b in (("p_in", "linear_a_p", "linear_b_p"), ("g_in", "linear_a_g", "linear_b_g")):
        out[f"{tgt}.{tgt_name}.weight"] = _cat(weights, (f"{src}.{a}.weight", f"{src}.{b}.weight"), dtype)
        bias_keys = (f"{src}.{a}.bias", f"{src}.{b}.bias")
        if all(key in weights for key in bias_keys):
            out[f"{tgt}.{tgt_name}.bias"] = _cat(weights, bias_keys, dtype)
        elif any(key in weights for key in bias_keys):
            raise KeyError(f"triangle projection {src}.{tgt_name} has only one of its a/b biases")


def _convert_pair_update(
    out: dict, weights: Mapping[str, torch.Tensor], src: str, tgt: str, dtype: torch.dtype
) -> None:
    """OSS ``PairReprUpdate`` -> ``RFProteinaPairUpdate``."""
    _copy(out, weights, f"{src}.layer_norm_in.weight", f"{tgt}.layer_norm_in.weight", dtype)
    _copy(out, weights, f"{src}.layer_norm_in.bias", f"{tgt}.layer_norm_in.bias", dtype)
    _copy(out, weights, f"{src}.linear_x.weight", f"{tgt}.linear_x.weight", dtype)
    _convert_trimul(out, weights, f"{src}.tri_mult_out", f"{tgt}.tri_mult_out", dtype)
    _convert_trimul(out, weights, f"{src}.tri_mult_in", f"{tgt}.tri_mult_in", dtype)
    for name in ("layer_norm", "linear_1", "linear_2"):
        _copy(out, weights, f"{src}.transition_out.{name}.weight", f"{tgt}.transition_out.{name}.weight", dtype)
        _copy(out, weights, f"{src}.transition_out.{name}.bias", f"{tgt}.transition_out.{name}.bias", dtype)


def convert_rfproteina_trunk_torch(
    config: BaseConfig, weights: Mapping[str, torch.Tensor], prefix: str = _SOURCE_PREFIX
) -> dict:
    """Convert released RFProteína trunk weights to a TRT-BNM state_dict.

    Args:
        config: the trunk :class:`~bionemo_ir.configs.BaseConfig` (see
            ``RFProteinaTrunkConfig``): drives ``num_layers``,
            ``pair_update_every``, ``use_qk_norm`` / ``use_insertion_head``, and
            the target ``dtype``.
        weights: released checkpoint state dict (OSS module names under the
            ``nn.`` prefix).
        prefix: top-level key prefix of the score network in ``weights``.

    Returns:
        Flat state dict keyed by ``LocalLatentsTransformer`` parameter names —
        load with ``module.load_state_dict(result)``.
    """
    _reject_unmerged_lora(weights)
    dtype = config.torch_dtype
    out: dict[str, torch.Tensor] = {}

    # Conditioning transitions (bias-free SwiGLU, no pre-norm).
    _convert_transition(out, weights, _join(prefix, "transition_c_1"), "transition_c_1", dtype)
    _convert_transition(out, weights, _join(prefix, "transition_c_2"), "transition_c_2", dtype)

    # Trunk layers (attention + conditioned transition).
    for i in range(config.num_layers):
        _convert_trunk_layer(
            out, weights, _join(prefix, f"transformer_layers.{i}"), f"transformer_layers.{i}", dtype, config.use_qk_norm
        )

    # Pair-representation updates (every ``pair_update_every`` layers).
    for i in range(config.num_layers - 1):
        if i % config.pair_update_every == 0:
            _convert_pair_update(
                out, weights, _join(prefix, f"pair_update_layers.{i}"), f"pair_update_layers.{i}", dtype
            )

    if config.recycle_enabled:
        for name in ("s_cycle_norm", "z_cycle_norm", "s_cycle_proj", "z_cycle_proj"):
            _copy(out, weights, _join(prefix, f"{name}.weight"), f"{name}.weight", dtype)

    # Output heads (RMSNorm + bias-free projection, identical names).
    heads = ["local_latents_linear", "ca_linear"]
    if config.use_sequence_readout:
        heads.append("sequence_readout")
    for head in heads:
        _copy(out, weights, _join(prefix, f"{head}.0.weight"), f"{head}.0.weight", dtype)
        _copy(out, weights, _join(prefix, f"{head}.1.weight"), f"{head}.1.weight", dtype)
    if config.use_insertion_head:
        for proj in ("insertion_q_proj", "insertion_k_proj"):
            _copy(out, weights, _join(prefix, f"{proj}.weight"), f"{proj}.weight", dtype)

    return out


def convert_rfproteina_representation_builder_torch(
    config: BaseConfig, weights: Mapping[str, torch.Tensor], prefix: str = _SOURCE_PREFIX
) -> dict:
    """Convert the score network's learned feature factories.

    Tensor feature construction itself has no parameters; this maps the
    embeddings, projections, pair norms, and pair-conditioning AdaLN into the
    native :class:`RFProteinaRepresentationBuilder`.
    """
    _reject_unmerged_lora(weights)
    dtype = config.torch_dtype
    pair_adaln_dtype = str_dtype_to_torch(config.pair_adaln_dtype)
    out: dict[str, torch.Tensor] = {}

    single = _join(prefix, "init_repr_factory")
    if config.input_schema == "sequence_probability":
        single_embeddings = (
            (2, "embedding", "token_type_embedding"),
            (7, "ss_embedding", "secondary_structure_embedding"),
            (16, "embedding", "charge_embedding"),
            (17, "embedding", "element_embedding"),
        )
    elif config.input_schema == "latest":
        single_embeddings = (
            (2, "embedding", "token_type_embedding"),
            (3, "embedding", "conditional_residue_embedding"),
            (7, "ss_embedding", "secondary_structure_embedding"),
            (14, "embedding", "charge_embedding"),
            (15, "embedding", "element_embedding"),
        )
    else:
        single_embeddings = (
            (4, "embedding", "token_type_embedding"),
            (5, "embedding", "conditional_residue_embedding"),
            (9, "ss_embedding", "secondary_structure_embedding"),
            (13, "embedding", "charge_embedding"),
            (14, "embedding", "element_embedding"),
        )
    for source_index, source_name, target in single_embeddings:
        _copy(out, weights, f"{single}.feat_creators.{source_index}.{source_name}.weight", f"{target}.weight", dtype)
    _copy(out, weights, f"{single}.linear_out.weight", "single_projection.weight", dtype)

    condition = _join(prefix, "cond_factory")
    for source_name, target in (
        ("embedding_C", "cath_class_embedding"),
        ("embedding_A", "cath_architecture_embedding"),
        ("embedding_T", "cath_topology_embedding"),
    ):
        _copy(out, weights, f"{condition}.feat_creators.2.{source_name}.weight", f"{target}.weight", dtype)
    _copy(out, weights, f"{condition}.linear_out.weight", "condition_projection.weight", dtype)

    pair = _join(prefix, "pair_repr_builder")
    pair_representation = f"{pair}.init_repr_factory"
    _copy(out, weights, f"{pair_representation}.linear_out.weight", "pair_projection.weight", dtype)
    norm_parameters = ("weight", "bias") if config.factory_norm == "layer_norm" else ("weight",)
    for parameter in norm_parameters:
        _copy(out, weights, f"{pair_representation}.ln_out.{parameter}", f"pair_norm.{parameter}", dtype)

    pair_condition = f"{pair}.cond_factory"
    _copy(out, weights, f"{pair_condition}.linear_out.weight", "pair_condition_projection.weight", dtype)
    for parameter in norm_parameters:
        _copy(out, weights, f"{pair_condition}.ln_out.{parameter}", f"pair_condition_norm.{parameter}", dtype)
    for source_name, target_name in (
        ("norm_cond.weight", "norm_cond.weight"),
        ("to_gamma.0.weight", "to_gamma.0.weight"),
        ("to_gamma.0.bias", "to_gamma.0.bias"),
        ("to_beta.weight", "to_beta.weight"),
    ):
        _copy(out, weights, f"{pair}.adaln.{source_name}", f"pair_adaln.{target_name}", pair_adaln_dtype)

    expected_inputs = {
        f"{single}.linear_out.weight": config.single_input_dim,
        f"{condition}.linear_out.weight": config.condition_input_dim,
        f"{pair_representation}.linear_out.weight": config.pair_input_dim,
        f"{pair_condition}.linear_out.weight": config.pair_condition_input_dim,
    }
    mismatched = [
        (name, weights[name].shape[1], expected)
        for name, expected in expected_inputs.items()
        if weights[name].shape[1] != expected
    ]
    if mismatched:
        detail = ", ".join(f"{name} (checkpoint {actual}, config {expected})" for name, actual, expected in mismatched)
        raise ValueError(f"RFProteína representation config does not match checkpoint: {detail}")
    return out


#: Single-feature creators the unified encoder conversion implements, by the
#: index they occupy in ``init_repr_factory.feat_creators``. A checkpoint that
#: carries a creator outside this set trained with an input feature this build
#: does not build, which no amount of shape inference can supply.
_KNOWN_ENCODER_FEAT_CREATORS = frozenset({0, 5})


def _unimplemented_feat_creators(weights: dict, prefix: str, known: frozenset[int]) -> list[str]:
    """Names of ``feat_creators`` entries the conversion has no builder for."""
    seen: dict[int, list[str]] = {}
    marker = _join(prefix, "init_repr_factory.feat_creators.")
    for key in weights:
        if not key.startswith(marker):
            continue
        tail = key[len(marker) :]
        index, _, rest = tail.partition(".")
        if index.isdigit() and int(index) not in known:
            seen.setdefault(int(index), []).append(rest)
    return [f"feat_creators.{i} ({', '.join(sorted(parts))})" for i, parts in sorted(seen.items())]


def fold_atom_chemistry(
    projection: torch.Tensor, element_embedding: torch.Tensor, charge_embedding: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multiply the released element and charge embeddings into their atom37 slots' projection columns.

    The released encoder projects ``concat(element_embedding[e], charge_embedding[c])`` read at every
    atom37 slot and flattened slot by slot. The projection is linear and has no bias, so slot ``s``
    contributes ``W_s[:, :E] @ element_embedding[e] + W_s[:, E:] @ charge_embedding[c]``, where ``W_s``
    holds that slot's columns. Each product depends on one small integer, so it is a row of a table.

    Args:
        projection: ``[token_dim, 37 * (E + C)]`` chemistry columns of the released single projection.
        element_embedding: ``[num_elements, E]`` released element table.
        charge_embedding: ``[num_charges, C]`` released charge table.

    Returns:
        ``[37 * num_elements, token_dim]`` and ``[37 * num_charges, token_dim]`` float32 tables, ordered
        slot first: row ``s * num_elements + k`` is element ``k`` seen through slot ``s``'s columns.

    Raises:
        ValueError: If the projection is not ``37 * (E + C)`` wide.
    """
    token_dim = projection.shape[0]
    element_width, charge_width = element_embedding.shape[1], charge_embedding.shape[1]
    if projection.shape[1] != ATOM37_SIZE * (element_width + charge_width):
        raise ValueError(
            f"chemistry projection must be {ATOM37_SIZE} * ({element_width} + {charge_width}) columns wide, "
            f"got {projection.shape[1]}"
        )
    columns = projection.reshape(token_dim, ATOM37_SIZE, element_width + charge_width).float()
    element = torch.einsum("ke,dse->skd", element_embedding.float(), columns[..., :element_width])
    charge = torch.einsum("kc,dsc->skd", charge_embedding.float(), columns[..., element_width:])
    return element.reshape(-1, token_dim), charge.reshape(-1, token_dim)


def _convert_encoder_inputs(
    out: dict, weights: Mapping[str, torch.Tensor], prefix: str, config: BaseConfig, dtype: torch.dtype
) -> None:
    """Single-projection inputs: the projection, the token-kind table, and the chemistry lookups.

    An encoder that reads per-atom chemistry gets the released element and charge embeddings multiplied
    into the projection (:func:`fold_atom_chemistry`), so ``element_embedding.weight`` and
    ``atom_charge_embedding.weight`` are lookup tables in ``token_dim`` and the projection keeps only
    the dense columns.
    """
    projection = weights[_join(prefix, "init_repr_factory.linear_out.weight")]
    if getattr(config, "atom_chemistry", False) and config.input_schema == "unified":
        dense = config.dense_input_dim
        out["init_repr_factory.linear_out.weight"] = projection[:, :dense].to(dtype)
        element, charge = fold_atom_chemistry(
            projection[:, dense:],
            weights[_join(prefix, "init_repr_factory.feat_creators.5.element_embedding.weight")],
            weights[_join(prefix, "init_repr_factory.feat_creators.5.charge_embedding.weight")],
        )
        out["element_embedding.weight"] = element.to(dtype)
        out["atom_charge_embedding.weight"] = charge.to(dtype)
    else:
        out["init_repr_factory.linear_out.weight"] = projection.to(dtype)
    if config.input_schema == "unified":
        _copy(
            out,
            weights,
            _join(prefix, "init_repr_factory.feat_creators.0.embedding.weight"),
            "token_type_embedding.weight",
            dtype,
        )


def convert_rfproteina_encoder_torch(
    config: BaseConfig, weights: Mapping[str, torch.Tensor], prefix: str = _ENCODER_SOURCE_PREFIX
) -> dict:
    """Convert released autoencoder encoder weights to a TRT-BNM state dict.

    See :func:`_convert_encoder_inputs` for how per-atom chemistry is folded into the single projection.
    """
    _reject_unmerged_lora(weights)
    dtype = config.torch_dtype
    transformer_dtype = str_dtype_to_torch(config.transformer_dtype)
    out: dict[str, torch.Tensor] = {}

    _convert_encoder_inputs(out, weights, prefix, config, dtype)
    if config.use_pair_bias:
        _copy(
            out,
            weights,
            _join(prefix, "pair_rep_factory.linear_out.weight"),
            "pair_rep_factory.linear_out.weight",
            dtype,
        )
    _convert_transition(out, weights, _join(prefix, "transition_c_1"), "transition_c_1", dtype)
    _convert_transition(out, weights, _join(prefix, "transition_c_2"), "transition_c_2", dtype)

    for i in range(config.num_layers):
        _convert_trunk_layer(
            out,
            weights,
            _join(prefix, f"transformer_layers.{i}"),
            f"transformer_layers.{i}",
            transformer_dtype,
            config.use_qk_norm,
            norm_bias=True,
            use_pair_bias=config.use_pair_bias,
        )

    head = "latent_decoder_mean_n_log_scale"
    for param in ("weight", "bias"):
        _copy(out, weights, _join(prefix, f"{head}.0.{param}"), f"{head}.0.{param}", dtype)
    _copy(out, weights, _join(prefix, f"{head}.1.weight"), f"{head}.1.weight", dtype)
    return out


def convert_rfproteina_decoder_torch(
    config: BaseConfig, weights: Mapping[str, torch.Tensor], prefix: str = _DECODER_SOURCE_PREFIX
) -> dict:
    """Convert released autoencoder decoder weights to a TRT-BNM state_dict.

    Same layer grammar as the velocity trunk, with three differences: the norms
    are LayerNorm rather than RMSNorm (so biases come across), there are no
    pair-update blocks, and the input factories and output heads differ.

    Args:
        config: the decoder :class:`~bionemo_ir.configs.BaseConfig` (see
            ``RFProteinaDecoderConfig``).
        weights: released autoencoder checkpoint state dict.
        prefix: top-level key prefix of the decoder in ``weights``.

    Returns:
        Flat state dict keyed by ``RFProteinaLatentDecoder`` parameter names.
    """
    _reject_unmerged_lora(weights)
    dtype = config.torch_dtype
    transformer_dtype = str_dtype_to_torch(config.transformer_dtype)
    out: dict[str, torch.Tensor] = {}

    _copy(
        out, weights, _join(prefix, "init_repr_factory.linear_out.weight"), "init_repr_factory.linear_out.weight", dtype
    )
    if config.input_schema == "unified":
        for source_index, target_name in ((2, "token_type_embedding"), (3, "conditional_residue_embedding")):
            _copy(
                out,
                weights,
                _join(prefix, f"init_repr_factory.feat_creators.{source_index}.embedding.weight"),
                f"{target_name}.weight",
                dtype,
            )
    if config.use_pair_bias:
        _copy(
            out,
            weights,
            _join(prefix, "pair_rep_factory.linear_out.weight"),
            "pair_rep_factory.linear_out.weight",
            dtype,
        )

    _convert_transition(out, weights, _join(prefix, "transition_c_1"), "transition_c_1", dtype)
    _convert_transition(out, weights, _join(prefix, "transition_c_2"), "transition_c_2", dtype)

    for i in range(config.num_layers):
        _convert_trunk_layer(
            out,
            weights,
            _join(prefix, f"transformer_layers.{i}"),
            f"transformer_layers.{i}",
            transformer_dtype,
            config.use_qk_norm,
            norm_bias=True,
            use_pair_bias=config.use_pair_bias,
        )

    # Output heads: LayerNorm + bias-free projection, identical names.
    heads = ("logit_linear", "struct_linear")
    if getattr(config, "use_atom_presence", False):
        heads += ("atom_presence_linear",)
    for head in heads:
        for param in ("weight", "bias"):
            _copy(out, weights, _join(prefix, f"{head}.0.{param}"), f"{head}.0.{param}", dtype)
        _copy(out, weights, _join(prefix, f"{head}.1.weight"), f"{head}.1.weight", dtype)

    return out


#: ``lora_alpha`` of the released vocab-147 flow checkpoint's adapter (rank 32, so scaling 2.0). The tensors do not record it.
RELEASED_LORA_ALPHA = 64.0

_LORA_KEY = re.compile(r"^(?P<module>.+)\.lora_A\.(?P<adapter>[^.]+)\.weight$")


def _lora_modules(weights: Mapping[str, torch.Tensor]) -> list[str]:
    return sorted({m["module"] for key in weights if (m := _LORA_KEY.match(key))})


def _reject_unmerged_lora(weights: Mapping[str, torch.Tensor]) -> None:
    """Refuse PEFT LoRA tensors instead of failing later on a missing plain ``weight``, or dropping them."""
    modules = _lora_modules(weights)
    if modules:
        raise ValueError(
            f"RFProteína weights hold {len(modules)} unmerged PEFT LoRA adapters (e.g. '{modules[0]}.lora_A'); BioIR "
            "converts plain weight/bias entries. Fold them first with "
            "merge_rfproteina_lora(weights, lora_alpha=...), or pass the state dict of a model whose adapters are "
            "already merged. The released flow checkpoint uses rank 32 and lora_alpha=64.0 (scaling 2.0), recorded "
            "in the Hub bundle's adapter/adapter_config.json."
        )


def merge_rfproteina_lora(
    weights: Mapping[str, torch.Tensor], *, lora_alpha: float, use_rslora: bool = False
) -> dict[str, torch.Tensor]:
    """Fold PEFT LoRA adapters into their base layers.

    An adapted layer ``m`` stores ``m.base_layer.{weight,bias}`` and ``m.lora_{A,B}.<adapter>.weight``; it computes
    ``W + scaling * B @ A``, with ``scaling = lora_alpha / r`` (``lora_alpha / sqrt(r)`` for rsLoRA). The result holds
    that sum as plain ``m.weight`` and ``m.bias``, with no adapter or ``base_layer`` key left.

    Args:
        weights: released checkpoint state dict that contains adapters.
        lora_alpha: the adapter's ``lora_alpha``. The tensors do not record it.
        use_rslora: whether the adapter used rank-stabilised scaling.

    Returns:
        A new dict; tensors that are not part of an adapted layer pass through unchanged.

    Raises:
        ValueError: on a layer with several adapters, or adapter tensors without their base weight.
    """
    out = dict(weights)
    for module in _lora_modules(weights):
        names = {m["adapter"] for key in weights if (m := _LORA_KEY.match(key)) and m["module"] == module}
        if len(names) != 1:
            raise ValueError(f"BioIR merges one LoRA adapter per layer; {module!r} has {sorted(names)}")
        (adapter,) = names
        a, b = weights[f"{module}.lora_A.{adapter}.weight"], weights[f"{module}.lora_B.{adapter}.weight"]
        base = weights.get(f"{module}.base_layer.weight")
        if base is None:
            raise ValueError(f"LoRA tensors under {module!r} have no '{module}.base_layer.weight' to merge into")
        rank = a.shape[0]
        scaling = lora_alpha / (rank**0.5 if use_rslora else rank)
        delta = (b.to(torch.float32) @ a.to(torch.float32)) * scaling
        out[f"{module}.weight"] = (base.to(torch.float32) + delta).to(base.dtype)
        if f"{module}.base_layer.bias" in weights:
            out[f"{module}.bias"] = weights[f"{module}.base_layer.bias"]
        for key in (
            f"{module}.base_layer.weight",
            f"{module}.base_layer.bias",
            f"{module}.lora_A.{adapter}.weight",
            f"{module}.lora_B.{adapter}.weight",
        ):
            out.pop(key, None)
    leftover = [key for key in out if ".lora_" in key]
    if leftover:
        raise ValueError(f"LoRA tensors outside a mergeable layer: {leftover[:3]}")
    return out


def _block_indices(weights: Mapping[str, torch.Tensor], prefix: str, group: str) -> list[int]:
    """Sorted block indices present under ``<prefix>.<group>.<i>.``."""
    pattern = re.compile(rf"^{re.escape(_join(prefix, group))}\.(\d+)\.")
    return sorted({int(m.group(1)) for key in weights if (m := pattern.match(key))})


def rfproteina_trunk_config_from_weights(
    weights: Mapping[str, torch.Tensor], prefix: str = _SOURCE_PREFIX, **overrides
) -> RFProteinaTrunkConfig:
    """Derive the trunk architecture from released checkpoint tensor shapes.

    Released checkpoints differ in width and depth, so tensor headers rather
    than a hard-coded default decide how the trunk is built. The September
    2026 release also carries learned recycle projections; their presence
    identifies its recycling/hadamard-write trunk contract.

    Args:
        weights: released checkpoint state dict (OSS module names).
        prefix: top-level key prefix of the score network in ``weights``.
        overrides: non-architecture fields to set, e.g. ``dtype`` or
            ``pairwise_attention_backend``.

    Returns:
        A :class:`~...config.RFProteinaTrunkConfig` matching ``weights``.
    """
    _reject_unmerged_lora(weights)

    def shape(rest: str) -> tuple[int, ...]:
        return tuple(weights[_join(prefix, rest)].shape)

    layers = _block_indices(weights, prefix, "transformer_layers")
    pair_updates = _block_indices(weights, prefix, "pair_update_layers")
    if not layers or not pair_updates:
        raise KeyError(f"No RFProteína trunk blocks found under the '{prefix}.' prefix")

    num_heads, pair_dim = shape("transformer_layers.0.mhba.mha.to_bias.weight")
    token_dim = shape("transformer_layers.0.mhba.mha.to_qkv.weight")[1]
    condition_dim = shape("transformer_layers.0.mhba.adaln.to_gamma.0.weight")[1]
    recycle_enabled = _join(prefix, "s_cycle_proj.weight") in weights
    derived = {
        "token_dim": token_dim,
        "pair_dim": pair_dim,
        "condition_dim": condition_dim,
        "latent_dim": shape("local_latents_linear.1.weight")[0],
        "num_layers": max(layers) + 1,
        "num_heads": num_heads,
        "condition_transition_expansion": shape("transition_c_1.linear_out.weight")[1] // condition_dim,
        "transition_expansion": shape("transformer_layers.0.transition.transition.linear_out.weight")[1] // token_dim,
        # Pair updates sit on every ``pair_update_every``-th trunk layer.
        "pair_update_every": pair_updates[1] - pair_updates[0] if len(pair_updates) > 1 else 1,
        "trimul_hidden_dim": shape("pair_update_layers.0.tri_mult_out.linear_a_p.weight")[0],
        "trimul_bias": _join(prefix, "pair_update_layers.0.tri_mult_out.linear_a_p.bias") in weights,
        "pair_transition_expansion": shape("pair_update_layers.0.transition_out.linear_1.weight")[0] // pair_dim,
        "pair_write": "hadamard" if recycle_enabled else "sum",
        "recycle_enabled": recycle_enabled,
        "use_qk_norm": _join(prefix, "transformer_layers.0.mhba.mha.q_layer_norm.weight") in weights,
        "use_insertion_head": _join(prefix, "insertion_q_proj.weight") in weights,
        "use_sequence_readout": _join(prefix, "sequence_readout.1.weight") in weights,
    }
    if derived["use_sequence_readout"]:
        derived["sequence_readout_classes"] = shape("sequence_readout.1.weight")[0]
    return RFProteinaTrunkConfig(**{**derived, **overrides})


def rfproteina_trunk_config_from_checkpoint(
    checkpoint: str | Path | Mapping[str, torch.Tensor], prefix: str = _SOURCE_PREFIX, **overrides
) -> RFProteinaTrunkConfig:
    """Derive the trunk architecture from a released checkpoint file.

    See :func:`rfproteina_trunk_config_from_weights`; the checkpoint is memory
    mapped, so only the tensor headers are read.
    """
    return rfproteina_trunk_config_from_weights(load_rfproteina_checkpoint_state(checkpoint), prefix, **overrides)


def _representation_schema(weights: Mapping[str, torch.Tensor], prefix: str) -> str:
    """Recognize a velocity-network feature schema from its factory parameters."""

    def has(rest: str) -> bool:
        return _join(prefix, rest) in weights

    charge_at_sequence_probability = has("init_repr_factory.feat_creators.16.embedding.weight")
    residue_at_latest = has("init_repr_factory.feat_creators.3.embedding.weight")
    residue_at_legacy = has("init_repr_factory.feat_creators.5.embedding.weight")
    if charge_at_sequence_probability and not residue_at_latest:
        return "sequence_probability"
    if residue_at_latest:
        return "latest"
    if residue_at_legacy:
        return "legacy_self_conditioned"
    raise KeyError(f"No RFProteína representation schema found under the '{prefix}.' prefix")


# A saturating count of remaining recycle passes; anything wider is a feature this
# build does not recognise rather than a very deep recycle schedule.
_MAX_RECYCLE_REMAINING_DIM = 16


def _recycle_conditioning(config: RFProteinaRepresentationConfig, condition_width: int) -> dict[str, int | str]:
    """Name the recycle feature a condition projection of this width was trained with.

    Both times and the CATH code account for a fixed part of the width. What is
    left is the recycle track: one time embedding for a pass index, a short
    one-hot for a count of remaining passes, or nothing at all.
    """
    leftover = condition_width - config.condition_base_dim
    if leftover == 0:
        return {"recycle_conditioning": "none"}
    if leftover == config.time_embedding_dim:
        return {"recycle_conditioning": "recycle_index"}
    if 0 < leftover <= _MAX_RECYCLE_REMAINING_DIM:
        return {"recycle_conditioning": "recycle_remaining", "recycle_remaining_dim": leftover}
    raise ValueError(
        f"RFProteína condition projection takes {condition_width} features, which leaves {leftover} "
        "for the recycle track; this build reads a pass-index embedding or a remaining-pass one-hot"
    )


def rfproteina_representation_config_from_weights(
    weights: Mapping[str, torch.Tensor], prefix: str = _SOURCE_PREFIX, **overrides
) -> RFProteinaRepresentationConfig:
    """Derive the velocity feature schema from a released score-network checkpoint.

    ``latent_dim`` is taken from the trunk head unless the caller passes it.
    The returned config's input widths have to match the checkpoint projections.
    """
    _reject_unmerged_lora(weights)
    latent_dim = overrides.pop("latent_dim", None)
    if latent_dim is None:
        latent_dim = int(weights[_join(prefix, "local_latents_linear.1.weight")].shape[0])
    schema = overrides.pop("input_schema", None) or _representation_schema(weights, prefix)
    pair_norm_bias = _join(prefix, "pair_repr_builder.init_repr_factory.ln_out.bias") in weights
    condition_norm_bias = _join(prefix, "pair_repr_builder.cond_factory.ln_out.bias") in weights
    if pair_norm_bias != condition_norm_bias:
        raise ValueError("RFProteína pair factory norms disagree on whether they have a bias")
    overrides.setdefault("factory_norm", "layer_norm" if pair_norm_bias else "rms_norm")
    if "num_token_kinds" not in overrides:
        # The token-kind embedding sits at a schema-dependent index in the feature factory.
        creator = 4 if schema == "legacy_self_conditioned" else 2
        kinds = _join(prefix, f"init_repr_factory.feat_creators.{creator}.embedding.weight")
        if kinds in weights:
            overrides["num_token_kinds"] = int(weights[kinds].shape[0])
    config = RFProteinaRepresentationConfig(input_schema=schema, latent_dim=int(latent_dim), **overrides)
    if "recycle_conditioning" not in overrides:
        condition_width = int(weights[_join(prefix, "cond_factory.linear_out.weight")].shape[1])
        config = config.model_copy(update=_recycle_conditioning(config, condition_width))
    expected = {
        "single": config.single_input_dim,
        "condition": config.condition_input_dim,
        "pair": config.pair_input_dim,
        "pair_condition": config.pair_condition_input_dim,
    }
    actual = {
        "single": int(weights[_join(prefix, "init_repr_factory.linear_out.weight")].shape[1]),
        "condition": int(weights[_join(prefix, "cond_factory.linear_out.weight")].shape[1]),
        "pair": int(weights[_join(prefix, "pair_repr_builder.init_repr_factory.linear_out.weight")].shape[1]),
        "pair_condition": int(weights[_join(prefix, "pair_repr_builder.cond_factory.linear_out.weight")].shape[1]),
    }
    mismatched = [
        f"{name} (checkpoint {actual[name]}, config {expected[name]})"
        for name in expected
        if actual[name] != expected[name]
    ]
    if mismatched:
        raise ValueError("RFProteína representation config does not match checkpoint: " + ", ".join(mismatched))
    return config


def rfproteina_encoder_config_from_weights(
    weights: Mapping[str, torch.Tensor], prefix: str = _ENCODER_SOURCE_PREFIX, **overrides
) -> RFProteinaEncoderConfig:
    """Derive the released autoencoder encoder architecture from its tensors."""
    _reject_unmerged_lora(weights)

    def shape(rest: str) -> tuple[int, ...]:
        return tuple(weights[_join(prefix, rest)].shape)

    layers = _block_indices(weights, prefix, "transformer_layers")
    if not layers:
        raise KeyError(f"No RFProteína encoder blocks found under the '{prefix}.' prefix")

    pair_bias_key = _join(prefix, "transformer_layers.0.mhba.mha.to_bias.weight")
    use_pair_bias = pair_bias_key in weights
    if use_pair_bias:
        num_heads, pair_dim = tuple(weights[pair_bias_key].shape)
    else:
        # Head count and the dormant pair width leave no tensor-shape trace in
        # a no-pair-bias checkpoint. Both released unified AEs use 12 / 256;
        # callers can override either for a future release.
        num_heads = int(overrides.get("num_heads", 12))
        pair_dim = int(overrides.get("pair_dim", 256))
    token_dim = shape("transformer_layers.0.mhba.mha.to_qkv.weight")[1]
    condition_dim = shape("transformer_layers.0.mhba.adaln.to_gamma.0.weight")[1]
    single_input_dim = shape("init_repr_factory.linear_out.weight")[1]
    token_embedding_key = _join(prefix, "init_repr_factory.feat_creators.0.embedding.weight")
    input_schema = "unified" if token_embedding_key in weights else "legacy"
    chemistry_key = _join(prefix, "init_repr_factory.feat_creators.5.element_embedding.weight")
    atom_chemistry = chemistry_key in weights
    if input_schema == "unified":
        token_type_embedding_dim = shape("init_repr_factory.feat_creators.0.embedding.weight")[1]
        num_residue_types = 33
    else:
        token_type_embedding_dim = 32
        # All non-residue legacy single features have a fixed width.
        num_residue_types = single_input_dim - (1 + 2 * (37 * 4) + 3 * 21 + (4 * 21 + 4))
    derived = {
        "input_schema": input_schema,
        "token_dim": token_dim,
        "pair_dim": pair_dim,
        "condition_dim": condition_dim,
        "latent_dim": shape("latent_decoder_mean_n_log_scale.1.weight")[0] // 2,
        "num_layers": max(layers) + 1,
        "num_heads": num_heads,
        "condition_transition_expansion": shape("transition_c_1.linear_out.weight")[1] // condition_dim,
        "transition_expansion": shape("transformer_layers.0.transition.transition.linear_out.weight")[1] // token_dim,
        "num_residue_types": num_residue_types,
        "token_type_embedding_dim": token_type_embedding_dim,
        "num_token_kinds": (
            shape("init_repr_factory.feat_creators.0.embedding.weight")[0] if input_schema == "unified" else 5
        ),
        "atom_chemistry": atom_chemistry,
        "sequence_separation_bins": 64 if input_schema == "unified" else 127,
        "use_pair_bias": use_pair_bias,
        "precompute_bias": use_pair_bias,
        "use_qk_norm": _join(prefix, "transformer_layers.0.mhba.mha.q_layer_norm.weight") in weights,
    }
    if atom_chemistry:
        elements, element_dim = shape("init_repr_factory.feat_creators.5.element_embedding.weight")
        charges, charge_dim = shape("init_repr_factory.feat_creators.5.charge_embedding.weight")
        derived |= {
            "num_elements": elements,
            "element_embedding_dim": element_dim,
            "num_atom_charges": charges,
            "atom_charge_embedding_dim": charge_dim,
        }
    config = RFProteinaEncoderConfig(**{**derived, **overrides})
    expected_single = shape("init_repr_factory.linear_out.weight")[1]
    if config.single_input_dim != expected_single:
        unimplemented = _unimplemented_feat_creators(weights, prefix, _KNOWN_ENCODER_FEAT_CREATORS)
        detail = (
            f"; the checkpoint carries input features this build does not implement: {'; '.join(unimplemented)}"
            if unimplemented
            else ""
        )
        raise ValueError(
            "RFProteína encoder single features are "
            f"{expected_single}-dimensional in the checkpoint but the config "
            f"asks for {config.single_input_dim}{detail}"
        )
    if use_pair_bias:
        expected_pair = shape("pair_rep_factory.linear_out.weight")[1]
    else:
        expected_pair = 0
    if config.pair_input_dim != expected_pair:
        raise ValueError(
            "RFProteína encoder pair features are "
            f"{expected_pair}-dimensional in the checkpoint but the config "
            f"asks for {config.pair_input_dim}"
        )
    return config


def rfproteina_encoder_config_from_checkpoint(
    checkpoint: str | Path | Mapping[str, torch.Tensor], prefix: str = _ENCODER_SOURCE_PREFIX, **overrides
) -> RFProteinaEncoderConfig:
    """Derive the encoder architecture from an autoencoder checkpoint."""
    return rfproteina_encoder_config_from_weights(load_rfproteina_checkpoint_state(checkpoint), prefix, **overrides)


def _decoder_atom37_occupancy(num_residue_types: int) -> tuple[tuple[int, ...], ...] | None:
    """Occupancy rows for a vocabulary the decoder does not carry a table for.

    The decoder ships the table the released checkpoints use. A wider
    vocabulary is a schema, so it is resolved here, where the checkpoint that
    fixes it is already open, rather than inside the layer.
    """
    vocabulary = _decoder_vocabulary(num_residue_types)
    return None if vocabulary is None else vocabulary.atom37_occupancy()


def _decoder_vocabulary(num_residue_types: int):
    """The vocabulary a decoder of this width needs, or ``None`` for the built-in one."""
    from bionemo_ir._torch.layers.transformers.local_latents_decoder import built_in_atom37_classes
    from bionemo_ir.data.schemas import residue_vocabulary, vendored_residue_vocabularies

    if num_residue_types <= built_in_atom37_classes():
        return None
    vocabulary = residue_vocabulary(num_residue_types)
    if vocabulary is None:
        available = sorted({built_in_atom37_classes(), *vendored_residue_vocabularies()})
        raise ValueError(
            f"no residue vocabulary for a {num_residue_types}-class decoder; this build carries {available}. "
            "Add the checkpoint's vocabulary under bionemo_ir/data/schemas/vocabularies/."
        )
    return vocabulary


def _decoder_allowed_classes(num_residue_types: int, num_token_kinds: int) -> tuple[tuple[int, ...], ...] | None:
    """Classes each token kind may decode as, for a wider vocabulary."""
    vocabulary = _decoder_vocabulary(num_residue_types)
    return None if vocabulary is None else vocabulary.allowed_residue_types(num_token_kinds)


def rfproteina_decoder_config_from_weights(
    weights: Mapping[str, torch.Tensor], prefix: str = _DECODER_SOURCE_PREFIX, **overrides
) -> RFProteinaDecoderConfig:
    """Derive the decoder architecture from autoencoder checkpoint shapes.

    The pair feature split is the one thing the weights cannot reveal: the pair
    projection only sees the concatenated width, so the bin counts come from
    ``overrides`` or the config defaults and are checked against that width.

    Args:
        weights: released autoencoder checkpoint state dict.
        prefix: top-level key prefix of the decoder in ``weights``.
        overrides: non-derivable fields, e.g. ``dtype`` or the bin counts.

    Returns:
        A :class:`~...config.RFProteinaDecoderConfig` matching ``weights``.
    """
    _reject_unmerged_lora(weights)

    def shape(rest: str) -> tuple[int, ...]:
        return tuple(weights[_join(prefix, rest)].shape)

    layers = _block_indices(weights, prefix, "transformer_layers")
    if not layers:
        raise KeyError(f"No RFProteína decoder blocks found under the '{prefix}.' prefix")

    pair_bias_key = _join(prefix, "transformer_layers.0.mhba.mha.to_bias.weight")
    use_pair_bias = pair_bias_key in weights
    if use_pair_bias:
        num_heads, pair_dim = tuple(weights[pair_bias_key].shape)
    else:
        num_heads = int(overrides.get("num_heads", 12))
        pair_dim = int(overrides.get("pair_dim", 256))
    token_dim = shape("transformer_layers.0.mhba.mha.to_qkv.weight")[1]
    condition_dim = shape("transformer_layers.0.mhba.adaln.to_gamma.0.weight")[1]
    token_embedding_key = _join(prefix, "init_repr_factory.feat_creators.2.embedding.weight")
    conditional_embedding_key = _join(prefix, "init_repr_factory.feat_creators.3.embedding.weight")
    input_schema = "unified" if token_embedding_key in weights and conditional_embedding_key in weights else "legacy"
    token_type_embedding_dim = (
        shape("init_repr_factory.feat_creators.2.embedding.weight")[1] if input_schema == "unified" else 32
    )
    num_conditional_residue_types = (
        shape("init_repr_factory.feat_creators.3.embedding.weight")[0] if input_schema == "unified" else 33
    )
    single_input_dim = shape("init_repr_factory.linear_out.weight")[1]
    latent_dim = single_input_dim - 3
    if input_schema == "unified":
        latent_dim -= token_type_embedding_dim + num_conditional_residue_types
    derived = {
        "input_schema": input_schema,
        "token_dim": token_dim,
        "pair_dim": pair_dim,
        "condition_dim": condition_dim,
        "latent_dim": latent_dim,
        "num_layers": max(layers) + 1,
        "num_heads": num_heads,
        "condition_transition_expansion": shape("transition_c_1.linear_out.weight")[1] // condition_dim,
        "transition_expansion": shape("transformer_layers.0.transition.transition.linear_out.weight")[1] // token_dim,
        "num_residue_types": shape("logit_linear.1.weight")[0],
        "token_type_embedding_dim": token_type_embedding_dim,
        "num_token_kinds": (
            shape("init_repr_factory.feat_creators.2.embedding.weight")[0] if input_schema == "unified" else 5
        ),
        "use_atom_presence": _join(prefix, "atom_presence_linear.1.weight") in weights,
        "atom37_occupancy": _decoder_atom37_occupancy(int(shape("logit_linear.1.weight")[0])),
        "allowed_residue_classes": _decoder_allowed_classes(
            int(shape("logit_linear.1.weight")[0]),
            shape("init_repr_factory.feat_creators.2.embedding.weight")[0] if input_schema == "unified" else 5,
        ),
        "num_conditional_residue_types": num_conditional_residue_types,
        "sequence_separation_bins": 64 if input_schema == "unified" else 127,
        "use_pair_bias": use_pair_bias,
        "precompute_bias": use_pair_bias,
        "use_qk_norm": _join(prefix, "transformer_layers.0.mhba.mha.q_layer_norm.weight") in weights,
    }
    config = RFProteinaDecoderConfig(**{**derived, **overrides})

    expected = shape("pair_rep_factory.linear_out.weight")[1] if use_pair_bias else 0
    actual = config.pair_input_dim
    if actual != expected:
        raise ValueError(
            f"RFProteína decoder pair features are {expected}-dimensional in "
            f"the checkpoint but the config asks for {actual} "
            f"(sequence_separation_bins={config.sequence_separation_bins} + "
            f"pair_distance_bins={config.pair_distance_bins})"
        )
    return config


def rfproteina_decoder_config_from_checkpoint(
    checkpoint: str | Path | Mapping[str, torch.Tensor], prefix: str = _DECODER_SOURCE_PREFIX, **overrides
) -> RFProteinaDecoderConfig:
    """Derive the decoder architecture from an autoencoder checkpoint file.

    See :func:`rfproteina_decoder_config_from_weights`; the checkpoint is
    memory mapped, so only the tensor headers are read.
    """
    return rfproteina_decoder_config_from_weights(load_rfproteina_checkpoint_state(checkpoint), prefix, **overrides)


def _check_architecture(config: BaseConfig, weights: Mapping[str, torch.Tensor], prefix: str) -> None:
    """Raise if ``config`` describes a different trunk than ``weights``.

    ``load_state_dict`` would also reject the mismatch, but only after a wall
    of per-tensor size errors that does not say which knob is wrong.
    """
    derived = rfproteina_trunk_config_from_weights(weights, prefix)
    mismatched = [
        (field, getattr(config, field), getattr(derived, field))
        for field in _ARCHITECTURE_FIELDS
        if getattr(config, field) != getattr(derived, field)
    ]
    if mismatched:
        detail = ", ".join(f"{field} (config {have}, checkpoint {want})" for field, have, want in mismatched)
        raise ValueError(
            f"RFProteína trunk config does not match the checkpoint: {detail}."
            " Build the trunk with rfproteina_trunk_config_from_checkpoint()."
        )


def _check_encoder_architecture(config: BaseConfig, weights: Mapping[str, torch.Tensor], prefix: str) -> None:
    derived = rfproteina_encoder_config_from_weights(weights, prefix)
    mismatched = [
        (field, getattr(config, field), getattr(derived, field))
        for field in _ENCODER_ARCHITECTURE_FIELDS
        if getattr(config, field) != getattr(derived, field)
    ]
    if mismatched:
        detail = ", ".join(f"{field} (config {have}, checkpoint {want})" for field, have, want in mismatched)
        raise ValueError(
            "RFProteína encoder config does not match the checkpoint: "
            f"{detail}. Build it with "
            "rfproteina_encoder_config_from_checkpoint()."
        )


class _AccessTrackingDict(dict):
    """``dict`` that records which keys were read via ``__getitem__``.

    Lets the forward converter stay a plain ``weights[src]`` reader while the
    component loader still reports source keys owned by other checkpoint
    components.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.accessed: set[str] = set()

    def __getitem__(self, key: str) -> torch.Tensor:
        self.accessed.add(key)
        return super().__getitem__(key)


#: What the component loaders read weights from: a released checkpoint file, or
#: a state dictionary a caller already holds (weights an embedding application
#: selected, say, including any EMA the host applied).
RFProteinaWeightSource = "str | Path | Mapping[str, torch.Tensor]"


#: Globals a released checkpoint legitimately needs to rebuild tensors and their
#: containers. Everything else in the file is training metadata we never read.
_WEIGHT_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("torch._utils", "_rebuild_tensor"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch._utils", "_rebuild_parameter"),
        ("torch._utils", "_rebuild_sparse_tensor"),
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
    }
)


class _MetadataStub(dict):
    """Inert stand-in for a global outside :data:`_WEIGHT_GLOBALS`.

    Unpickling the training metadata must never import or call what the file
    names. Every such global resolves to this class instead, so the opcodes that
    would have built a config object build an empty dict that nothing reads.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def __setstate__(self, state: Any) -> None:
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> _MetadataStub:
        return _MetadataStub()


class _WeightsOnlyUnpickler(pickle.Unpickler):
    """Resolve tensor-rebuilding globals only; stub the rest."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _WEIGHT_GLOBALS or (module == "torch" and name.endswith("Storage")):
            return getattr(importlib.import_module(module), name)
        return _MetadataStub


class _RestrictedPickleModule:
    """``pickle_module`` for :func:`torch.load`; it only needs these two names."""

    Unpickler = _WeightsOnlyUnpickler

    @staticmethod
    def load(file: io.IOBase, **kwargs: Any) -> Any:
        return _WeightsOnlyUnpickler(file, **kwargs).load()


def load_rfproteina_checkpoint_state(checkpoint: str | Path | Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    """Return a released RFProteína state dictionary.

    An in-memory mapping passes through, so a host that has already loaded and
    selected its weights never reopens the file.

    The released files are Lightning checkpoints: beside ``state_dict`` they
    carry ``hyper_parameters``, ``callbacks``, ``loops`` and ``optimizer_states``,
    and ``hyper_parameters`` holds OmegaConf ``DictConfig`` objects. Weights-only
    unpickling parses the whole stream, so that metadata -- which this function
    never reads -- rejects the file with ``Unsupported global: DictConfig``.
    Allowlisting it does not help: after ``DictConfig``, ``ContainerMetadata``,
    ``typing.Any``, ``dict`` and ``defaultdict``, torch's weights-only unpickler
    still refuses ``SETITEM`` on a ``defaultdict``.

    So resolve the tensor-rebuilding globals ourselves and stub every other one.
    No global the file names is imported or called, which is the property
    ``weights_only=True`` was there for.
    """
    if isinstance(checkpoint, Mapping):
        return checkpoint.get("state_dict", checkpoint) if "state_dict" in checkpoint else checkpoint
    loaded = torch.load(
        str(checkpoint),
        map_location="cpu",
        mmap=True,
        weights_only=False,  # Guarded by _RestrictedPickleModule, not by torch.
        pickle_module=_RestrictedPickleModule,
    )
    state = loaded.get("state_dict", loaded) if isinstance(loaded, Mapping) else loaded
    if not isinstance(state, Mapping):
        raise TypeError("RFProteína checkpoint does not contain a state dictionary")
    tensors = {key: value for key, value in state.items() if torch.is_tensor(value)}
    if not tensors:
        raise TypeError("RFProteína checkpoint state dictionary holds no tensors")
    return tensors


def load_rfproteina_trunk_checkpoint(
    module: nn.Module,
    checkpoint: str | Path | Mapping[str, torch.Tensor],
    strict: bool = True,
) -> RFProteinaConversionReport:
    """Load a released checkpoint into the RFProteína Torch trunk.

    Builds the target state dict via :func:`convert_rfproteina_trunk_torch`
    (driven by ``module.config``), loads it, and returns a coverage report.

    Raises:
        ValueError: if ``module.config`` describes a different architecture
            than the checkpoint.
    """
    state = load_rfproteina_checkpoint_state(checkpoint)
    _check_architecture(module.config, state, _SOURCE_PREFIX)

    weights = _AccessTrackingDict(state)
    converted = convert_rfproteina_trunk_torch(module.config, weights)
    return _load_converted(module, weights, converted, _SOURCE_PREFIX, "trunk", strict)


def load_rfproteina_representation_builder_checkpoint(
    module: nn.Module,
    checkpoint: str | Path | Mapping[str, torch.Tensor],
    strict: bool = True,
) -> RFProteinaConversionReport:
    """Load learned score-network feature factories into the native builder."""
    weights = _AccessTrackingDict(load_rfproteina_checkpoint_state(checkpoint))
    converted = convert_rfproteina_representation_builder_torch(module.config, weights)
    return _load_converted(module, weights, converted, _SOURCE_PREFIX, "representation builder", strict)


def load_rfproteina_encoder_checkpoint(
    module: nn.Module,
    checkpoint: str | Path | Mapping[str, torch.Tensor],
    strict: bool = True,
) -> RFProteinaConversionReport:
    """Strictly load the encoder half of a released autoencoder checkpoint."""
    state = load_rfproteina_checkpoint_state(checkpoint)
    _check_encoder_architecture(module.config, state, _ENCODER_SOURCE_PREFIX)
    weights = _AccessTrackingDict(state)
    converted = convert_rfproteina_encoder_torch(module.config, weights)
    return _load_converted(module, weights, converted, _ENCODER_SOURCE_PREFIX, "encoder", strict)


def load_rfproteina_decoder_checkpoint(
    module: nn.Module,
    checkpoint: str | Path | Mapping[str, torch.Tensor],
    strict: bool = True,
) -> RFProteinaConversionReport:
    """Load a released autoencoder checkpoint into the Torch decoder.

    Builds the target state dict via :func:`convert_rfproteina_decoder_torch`
    (driven by ``module.config``), loads it, and returns a coverage report. The
    encoder half of the checkpoint is reported as ignored.
    """
    weights = _AccessTrackingDict(load_rfproteina_checkpoint_state(checkpoint))
    converted = convert_rfproteina_decoder_torch(module.config, weights)
    return _load_converted(module, weights, converted, _DECODER_SOURCE_PREFIX, "decoder", strict)


def _load_converted(
    module: nn.Module, weights: _AccessTrackingDict, converted: dict, source_prefix: str, what: str, strict: bool
) -> RFProteinaConversionReport:
    """Load a converted state dict and report source/target coverage."""
    module_keys = set(module.state_dict())
    missing = sorted(module_keys - set(converted))
    if strict and missing:
        raise KeyError(f"Missing RFProteína {what} weights: {missing}")

    prefix = f"{source_prefix}."
    ignored = sorted(key[len(prefix) :] for key in weights if key.startswith(prefix) and key not in weights.accessed)
    report = RFProteinaConversionReport(
        loaded_keys=tuple(sorted(k for k in converted if k in module_keys)),
        ignored_source_keys=tuple(ignored),
        missing_target_keys=tuple(missing),
    )
    module.load_state_dict(converted, strict=strict)
    return report
