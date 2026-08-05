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
"""Protenix model configuration.

Defaults track the ``protenix-v2`` checkpoint (ByteDance OSS
``configs_base.py`` + ``configs_model_type.py`` overrides).
"""

from tensorrt_bionemo.configs import BaseConfig, DiffusionTransformerConfig, PairformerConfig
from tensorrt_bionemo.hubs import FoldingSupportMatrix as SupMat
from tensorrt_bionemo.models.openfold3.config import MSAModuleStackConfig


class _Default:
    """protenix-v2 dimension defaults.

    Only ``c_z`` is raised (128 -> 256) vs the base config; input-embedder
    atom/token dims stay at base values.
    """

    c_s: int = 384
    c_z: int = 256  # protenix-v2: 128 -> 256
    c_token: int = 384
    c_atom: int = 128
    c_atompair: int = 16
    # InputFeatureEmbedder: c_token + restype(32) + profile(32) + deletion_mean(1)
    c_s_inputs: int = 449
    n_queries: int = 32
    n_keys: int = 128
    atom_n_blocks: int = 3
    atom_n_heads: int = 4
    r_max: int = 32
    s_max: int = 2
    # Diffusion token channel is wider than the embedder (768 vs 384); EDM data std
    diffusion_c_token: int = 768
    c_noise_embedding: int = 256
    sigma_data: float = 16.0
    # Template (hidden_scale_up): stack at c=64, heads = c // 32
    template_c: int = 64
    template_n_blocks: int = 2
    template_pairwise_head_width: int = 32
    template_num_intermediate_factor: int = 2
    max_atoms_per_token: int = 24  # DNA G = 23
    n_cycle: int = 10
    c_m: int = 128  # protenix-v2: 64 -> 128
    msa_n_blocks: int = 4
    pairformer_n_blocks: int = 48
    pairformer_n_heads: int = 16
    pairwise_head_width: int = 32


def _atom_transformer_config() -> DiffusionTransformerConfig:
    """Local-windowed atom DiT: chained KV AdaLN, AdaLN-zero gate, bias_proj.

    ``chain_kv_norm`` / ``pair_norm`` / ``initial_norm`` are extras on
    ``BaseConfig``. ``precompute_bias`` mega-GEMMs all layers' pair bias.
    """
    return DiffusionTransformerConfig(
        num_blocks=_Default.atom_n_blocks,
        num_heads=_Default.atom_n_heads,
        dim=_Default.c_atom,
        dim_single_cond=_Default.c_atom,
        dim_pairwise=_Default.c_atompair,
        bias_proj=True,
        pair_norm=True,
        post_layer_norm=False,
        initial_norm=False,
        attention_initial_norm=False,
        use_ada_layer_norm=True,
        use_separate_layer_norm=True,
        chain_kv_norm=True,
        attn_output_gate=True,
        conditioned_transition_using_silu=True,
        transition_expansion_factor=2,
        # Mega-GEMM: one GEMM for all layers' pair bias (folds per-layer LN gamma).
        precompute_bias=True,
        pairwise_attention_backend="SDPA",
        version="v2",
    )


class InputFeatureEmbedderConfig(BaseConfig):
    """``protenix-v2`` input embedder (no coords, ESM off). Output width ``c_s_inputs``."""

    c_token: int = _Default.c_token
    c_atom: int = _Default.c_atom
    c_atompair: int = _Default.c_atompair
    c_s_inputs: int = _Default.c_s_inputs
    n_queries: int = _Default.n_queries
    n_keys: int = _Default.n_keys
    has_coords: bool = False
    # protenix-v2 disables ESM; embedding_dim is the class fallback (esm2-3b).
    esm_enabled: bool = False
    esm_embedding_dim: int = 2560
    atom_transformer_config: DiffusionTransformerConfig = _atom_transformer_config()


class RelativePositionEncodingConfig(BaseConfig):
    """RPE: ``fix_sym_check=True``, ``cyclic_pos_enc=False``; relp width ``4*r_max+2*s_max+7``."""

    r_max: int = _Default.r_max
    s_max: int = _Default.s_max
    c_z: int = _Default.c_z
    fix_sym_check: bool = True
    cyclic_pos_enc: bool = False


class TemplateEmbedderConfig(BaseConfig):
    """Template embedder with ``hidden_scale_up`` pair stack at ``c=64``."""

    c: int = _Default.template_c
    c_z: int = _Default.c_z
    n_blocks: int = _Default.template_n_blocks
    pairwise_head_width: int = _Default.template_pairwise_head_width
    pairwise_num_heads: int = _Default.template_c // _Default.template_pairwise_head_width
    num_intermediate_factor: int = _Default.template_num_intermediate_factor
    # Inner pair stack bf16; outer projections/LNs follow ``dtype``.
    pairformer_dtype: str = "bfloat16"
    # bf16 stack precision (not fp32 tri-mul accum) — matches trunk/MSA; fp32
    # would double tri-mul activation and block fused kernels.
    trimul_high_precision: bool = False


class ConstraintEmbedderConfig(BaseConfig):
    """Optional constraint pair embedders; all off in protenix-v2 (returns ``None``).

    Substructure embedder is not ported (raises if enabled).
    """

    c_constraint_z: int = _Default.c_z
    pocket_enable: bool = False
    pocket_c_z_input: int = 1
    contact_enable: bool = False
    contact_c_z_input: int = 2
    contact_atom_enable: bool = False
    contact_atom_c_z_input: int = 2
    substructure_enable: bool = False


def _token_transformer_config() -> DiffusionTransformerConfig:
    """Global token DiT on ``c_token=768``, conditioned on trunk ``c_s``/``c_z``."""
    return DiffusionTransformerConfig(
        num_blocks=24,
        num_heads=16,
        dim=_Default.diffusion_c_token,
        dim_single_cond=_Default.c_s,
        dim_pairwise=_Default.c_z,
        bias_proj=True,
        pair_norm=True,
        initial_norm=True,
        attention_initial_norm=False,
        use_ada_layer_norm=True,
        use_separate_layer_norm=False,
        chain_kv_norm=False,
        attn_output_gate=True,
        conditioned_transition_using_silu=True,
        transition_expansion_factor=2,
        precompute_bias=True,
        pairwise_attention_backend="SDPA",
        version="v2",
    )


class DiffusionConditioningConfig(BaseConfig):
    """Diffusion conditioning. Pair path has its own ``relpe`` (separate from trunk RPE).

    OSS pair-conditioning / noise single path stay fp32; cached ``pair_z`` is
    ``z_pair_dtype``.
    """

    c_s: int = _Default.c_s
    c_z: int = _Default.c_z
    c_s_inputs: int = _Default.c_s_inputs
    c_noise_embedding: int = _Default.c_noise_embedding
    sigma_data: float = _Default.sigma_data
    relpe_config: BaseConfig = RelativePositionEncodingConfig()
    # Pair-conditioning / noise single path stay fp32; store rollout-cached
    # ``pair_z`` in bf16 (consumed by bf16 transformers; fp32 would double resident mem).
    z_pair_dtype: str = "bfloat16"
    dtype: str = "float32"


class DiffusionAtomAttentionEncoderConfig(BaseConfig):
    """Coord-conditioned atom encoder (``has_coords=True``); fp32 (diffusion upcasts)."""

    c_token: int = _Default.diffusion_c_token
    c_atom: int = _Default.c_atom
    c_atompair: int = _Default.c_atompair
    c_s: int = _Default.c_s
    c_z: int = _Default.c_z
    n_queries: int = _Default.n_queries
    n_keys: int = _Default.n_keys
    has_coords: bool = True
    atom_transformer_config: DiffusionTransformerConfig = _atom_transformer_config()
    dtype: str = "float32"


class AtomAttentionDecoderConfig(BaseConfig):
    """Atom decoder projecting token ``a`` to per-atom coord updates; fp32."""

    c_token: int = _Default.diffusion_c_token
    c_atom: int = _Default.c_atom
    c_atompair: int = _Default.c_atompair
    n_queries: int = _Default.n_queries
    n_keys: int = _Default.n_keys
    atom_transformer_config: DiffusionTransformerConfig = _atom_transformer_config()
    dtype: str = "float32"


class SampleDiffusionConfig(BaseConfig):
    """Diffusion sampling + noise schedule (OSS ``inference_noise_scheduler`` defaults).

    ``enable_diffusion_shared_vars_cache`` precomputes step-invariant shared vars
    once per rollout (numerically identical to the uncached path).
    """

    n_step: int = 200
    s_max: float = 160.0
    s_min: float = 4e-4
    rho: float = 7.0
    gamma0: float = 0.8
    gamma_min: float = 1.0
    noise_scale_lambda: float = 1.003
    step_scale_eta: float = 1.5
    enable_diffusion_shared_vars_cache: bool = True


class DiffusionModuleConfig(BaseConfig):
    """Diffusion module; runs in fp32 (OSS upcasts token path / EDM math)."""

    c_s: int = _Default.c_s
    c_z: int = _Default.c_z
    c_token: int = _Default.diffusion_c_token
    n_queries: int = _Default.n_queries
    n_keys: int = _Default.n_keys
    sigma_data: float = _Default.sigma_data
    diffusion_conditioning_config: BaseConfig = DiffusionConditioningConfig()
    atom_encoder_config: BaseConfig = DiffusionAtomAttentionEncoderConfig()
    token_transformer_config: DiffusionTransformerConfig = _token_transformer_config()
    atom_decoder_config: BaseConfig = AtomAttentionDecoderConfig()
    dtype: str = "float32"


class DistogramHeadConfig(BaseConfig):
    """Distogram head; fp32 (OSS disables autocast)."""

    c_z: int = _Default.c_z
    no_bins: int = 64
    dtype: str = "float32"


def _pairformer_config(num_blocks: int = _Default.pairformer_n_blocks) -> PairformerConfig:
    """Pairformer: ``version=v1`` + ``attention_initial_norm``; trunk + confidence head."""
    return PairformerConfig(
        token_s=_Default.c_s,
        token_z=_Default.c_z,
        num_blocks=num_blocks,
        num_heads=_Default.pairformer_n_heads,
        pairwise_head_width=_Default.pairwise_head_width,
        pairwise_num_heads=_Default.c_z // _Default.pairwise_head_width,
        no_update_s=False,
        attention_initial_norm=True,
        version="v1",
        dtype="bfloat16",
    )


class ConfidenceHeadConfig(BaseConfig):
    """Confidence head: projections/readouts fp32, inner pairformer bf16."""

    c_s: int = _Default.c_s
    c_z: int = _Default.c_z
    c_s_inputs: int = _Default.c_s_inputs
    b_pae: int = 64
    b_pde: int = 64
    b_plddt: int = 50
    b_resolved: int = 2
    max_atoms_per_token: int = _Default.max_atoms_per_token
    distance_bin_start: float = 3.25
    distance_bin_end: float = 52.0
    distance_bin_step: float = 1.25
    pairformer_config: BaseConfig = _pairformer_config(num_blocks=4)
    dtype: str = "float32"


class ProtenixMSAModuleConfig(MSAModuleStackConfig):
    """MSA module wrapping OF3 ``MSAModuleStack``; embed uses ``c_s_inputs``/``msa_input_dim``."""

    c_m: int = _Default.c_m
    c_z: int = _Default.c_z
    # MSAStack default per-head width c=8 (OSS MSABlock does not pass c).
    c_hidden_msa_att: int = 8
    c_hidden_opm: int = 32
    c_hidden_mul: int = _Default.c_z
    c_hidden_pair_att: int = _Default.pairwise_head_width
    no_heads_msa: int = 8
    no_heads_pair: int = _Default.c_z // _Default.pairwise_head_width
    transition_n: int = 4
    no_blocks: int = _Default.msa_n_blocks
    opm_first: bool = True
    # OPM / PWA fall back to CHUNK_REGISTRY defaults. MSA Transition is row-chunked
    # via ``MSA_TRANSITION`` (OSS msa_chunk_size=2048) — needed for deep MSAs.
    opm_chunk_size: int = None
    opm_mask_chunk_size: int = None
    msa_att_row_chunk_size: int = None
    dtype: str = "bfloat16"
    c_s_inputs: int = _Default.c_s_inputs
    msa_input_dim: int = 34  # one-hot msa (32) + has_deletion (1) + deletion (1)


class TrunkConfig(BaseConfig):
    """Recycling trunk (template + MSA + pairformer).

    ``use_template`` gates the embedder in the loop (OSS always runs it when
    ``n_blocks > 0``; the OSS flag only controls the featurizer).
    """

    c_s: int = _Default.c_s
    c_z: int = _Default.c_z
    n_cycle: int = _Default.n_cycle
    # bf16 halves resident pair state but is opt-in: it drifts from fp32 over a
    # ten-cycle recycling run.
    pair_state_dtype: str = "float32"
    use_template: bool = True
    template_embedder_config: BaseConfig = TemplateEmbedderConfig()
    msa_module_config: BaseConfig = ProtenixMSAModuleConfig()
    pairformer_config: BaseConfig = _pairformer_config()


class ConfidenceSummaryConfig(BaseConfig):
    """Postprocess bin ranges and ``ranking_score`` weights."""

    plddt_bins: tuple = (0.0, 1.0, 50)
    pde_bins: tuple = (0.0, 32.0, 64)
    pae_bins: tuple = (0.0, 32.0, 64)
    distogram_bins: tuple = (2.3125, 21.6875, 64)
    contact_threshold: float = 8.0
    af3_clash_threshold: float = 1.1
    iptm_weight: float = 0.8
    ptm_weight: float = 0.2
    disorder_weight: float = 0.5
    clash_penalty: float = 100.0


class ProtenixConfig(BaseConfig):
    """Top-level protenix-v2 model config."""

    c_s: int = _Default.c_s
    c_z: int = _Default.c_z
    c_s_inputs: int = _Default.c_s_inputs
    n_queries: int = _Default.n_queries
    n_keys: int = _Default.n_keys
    input_embedder_config: BaseConfig = InputFeatureEmbedderConfig()
    relative_position_encoding_config: BaseConfig = RelativePositionEncodingConfig()
    constraint_embedder_config: BaseConfig = ConstraintEmbedderConfig()
    trunk_config: BaseConfig = TrunkConfig()
    diffusion_module_config: BaseConfig = DiffusionModuleConfig()
    sample_diffusion_config: BaseConfig = SampleDiffusionConfig()
    distogram_head_config: BaseConfig = DistogramHeadConfig()
    confidence_head_config: BaseConfig = ConfidenceHeadConfig()
    confidence_summary_config: BaseConfig = ConfidenceSummaryConfig()


PRETRAINED_CONFIG_REGISTRY = {SupMat.ProtenixV2: ProtenixConfig}
