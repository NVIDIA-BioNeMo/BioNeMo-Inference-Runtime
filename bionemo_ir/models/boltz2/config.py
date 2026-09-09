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


from bionemo_ir.configs import BaseConfig, DiffusionTransformerConfig, PairformerConfig
from bionemo_ir.hubs import FoldingSupportMatrix as SupMat
from bionemo_ir.pipeline.models.boltz2.const import num_tokens


class _Default:
    token_s: int = 384
    token_z: int = 128
    atom_s: int = 128
    atom_z: int = 16
    num_bins: int = 64
    atoms_per_window_queries: int = 32
    atoms_per_window_keys: int = 128
    use_no_atom_char: bool = False
    use_atom_backbone_feat: bool = False
    use_residue_feats_atoms: bool = False
    fix_sym_check: bool = True
    cyclic_pos_enc: bool = True
    bond_type_feature: bool = True
    template_dim: int = 64
    template_blocks: int = 2
    template_num_bins: int = 38


class InputEmbedderConfig(BaseConfig):
    atom_s: int = _Default.atom_s
    atom_z: int = _Default.atom_z
    token_s: int = _Default.token_s
    token_z: int = _Default.token_z
    atoms_per_window_queries: int = _Default.atoms_per_window_queries
    atoms_per_window_keys: int = _Default.atoms_per_window_keys
    atom_feature_dim: int = 388
    add_method_conditioning: bool = True
    add_modified_flag: bool = True
    add_cyclic_flag: bool = True
    add_mol_type_feat: bool = True
    use_no_atom_char: bool = _Default.use_no_atom_char
    use_atom_backbone_feat: bool = _Default.use_atom_backbone_feat
    use_residue_feats_atoms: bool = _Default.use_residue_feats_atoms
    diffusion_transformer: DiffusionTransformerConfig = DiffusionTransformerConfig(
        num_blocks=3,
        num_heads=4,
        dim=_Default.atom_s,
        dim_single_cond=_Default.atom_s,
        dim_pairwise=_Default.atom_z,
        bias_proj=False,
        conditioned_transition_using_silu=False,
        expansion_factor=2,
        version="v2",
    )


class TemplateV2ModuleConfig(BaseConfig):
    """Boltz-2 TemplateV2 config. Inner pairformer is bf16 trimul by default."""

    token_z: int = _Default.token_z
    template_dim: int = _Default.template_dim
    template_blocks: int = _Default.template_blocks
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    min_dist: float = 3.25
    max_dist: float = 50.75
    num_bins: int = _Default.template_num_bins
    num_tokens: int = num_tokens
    pairformer: PairformerConfig = PairformerConfig(
        token_s=_Default.token_s,
        token_z=_Default.template_dim,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        num_blocks=_Default.template_blocks,
        num_heads=16,
        no_update_s=True,
        dtype="bfloat16",
        trimul_high_precision=False,
        attention_initial_norm=False,
        version="v2",
    )


# Shared by Boltz-1 and Boltz-2; `version` selects the layer variant.
class MSAModuleConfig(BaseConfig):
    msa_s: int = None
    token_z: int = None
    token_s: int = None
    msa_blocks: int = None
    num_tokens: int = None
    pairwise_head_width: int = None
    pairwise_num_heads: int = None
    use_paired_feature: bool = True
    version: str = "v1"


class TrunkConfig(BaseConfig):
    use_templates_v2: bool = False
    msa_module: MSAModuleConfig = MSAModuleConfig(
        msa_s=64,
        token_z=_Default.token_z,
        token_s=_Default.token_s,
        msa_blocks=4,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        num_tokens=num_tokens,
        use_paired_feature=True,
        trimul_high_precision=False,
        version="v2",
    )
    pairformer: PairformerConfig = PairformerConfig(
        token_s=_Default.token_s,
        token_z=_Default.token_z,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        num_blocks=64,
        num_heads=16,
        trimul_high_precision=False,
        attention_initial_norm=False,
        version="v2",
    )
    template_module: TemplateV2ModuleConfig = TemplateV2ModuleConfig()


class AtomDiffusionConfig(BaseConfig):
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    sigma_data: int = 16
    rho: int = 7
    P_mean: float = -1.2
    P_std: float = 1.5
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    accumulate_token_repr: bool = False
    num_sampling_steps: int = 50
    token_s: int = _Default.token_s
    dim_fourier: int = 256
    version: str = "v2"


class ScoreModelConfig(BaseConfig):
    atom_s: int = _Default.atom_s
    atom_z: int = _Default.atom_z
    token_s: int = _Default.token_s
    token_z: int = _Default.token_z
    dim_fourier: int = 256
    atoms_per_window_queries: int = _Default.atoms_per_window_queries
    atoms_per_window_keys: int = _Default.atoms_per_window_keys
    conditioning_transition_layers: int = 2

    # Precision for the PairwiseConditioning FFN (its [N, N, 2*hidden] intermediate dominates
    # memory). bf16 halves it vs fp32; the output is cast back to the structure-module dtype so
    # downstream conditioning is unchanged.
    pairwise_conditioning_dtype: str = "bfloat16"
    # Precision for the token-transformer bias [N, N, depth*heads]. bf16 halves it vs fp32; the
    # token transformer consumes it at bf16 anyway, so this is essentially free.
    token_trans_bias_dtype: str = "bfloat16"

    atom_encoder: DiffusionTransformerConfig = DiffusionTransformerConfig(
        num_blocks=3,
        num_heads=4,
        dim=_Default.atom_s,
        dim_single_cond=_Default.atom_s,
        bias_proj=False,
        conditioned_transition_using_silu=False,
        expansion_factor=2,
        version="v2",
    )
    token_transformer: DiffusionTransformerConfig = DiffusionTransformerConfig(
        num_blocks=24,
        num_heads=16,
        dim=2 * _Default.token_s,
        dim_single_cond=2 * _Default.token_s,
        dim_pairwise=_Default.token_z,
        bias_proj=False,
        conditioned_transition_using_silu=False,
        expansion_factor=2,
        version="v2",
    )
    atom_decoder: DiffusionTransformerConfig = DiffusionTransformerConfig(
        num_blocks=3,
        num_heads=4,
        dim=_Default.atom_s,
        dim_single_cond=_Default.atom_s,
        bias_proj=False,
        conditioned_transition_using_silu=False,
        expansion_factor=2,
        version="v2",
    )
    version: str = "v2"


class StructureModuleConfig(BaseConfig):
    atom_diffusion: AtomDiffusionConfig = AtomDiffusionConfig()
    score_model: ScoreModelConfig = ScoreModelConfig()
    version: str = "v2"


class ConfidenceHeadsConfig(BaseConfig):
    token_s: int = _Default.token_s
    token_z: int = _Default.token_z
    num_plddt_bins: int = 50
    num_pde_bins: int = 64
    num_pae_bins: int = 64
    token_level_confidence: bool = True
    use_separate_heads: bool = True


class ConfidenceModuleConfig(BaseConfig):
    token_s: int = _Default.token_s
    token_z: int = _Default.token_z
    num_dist_bins: int = 64
    token_level_confidence: bool = True
    max_dist: int = 22
    no_update_s: bool = False
    add_s_to_z_prod: bool = True
    add_s_input_to_s: bool = True
    add_z_input_to_z: bool = True
    fix_sym_check: bool = _Default.fix_sym_check
    cyclic_pos_enc: bool = _Default.cyclic_pos_enc
    maximum_bond_distance: int = 0
    bond_type_feature: bool = _Default.bond_type_feature
    conditioning_cutoff_min: float = 4.0
    conditioning_cutoff_max: float = 20.0
    return_latent_feats: bool = False
    relative_position_encoder: BaseConfig = BaseConfig(period_broadcast=False)
    pairformer: PairformerConfig = PairformerConfig(
        token_s=token_s,
        token_z=token_z,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        num_blocks=8,
        num_heads=16,
        trimul_high_precision=False,
        attention_initial_norm=False,
        version="v2",
    )
    confidence_heads: ConfidenceHeadsConfig = ConfidenceHeadsConfig()


class Boltz2Config(BaseConfig):
    # Optional cap on stacked templates (T dim). None (default) = no cap =
    # OSS-faithful. Set an int to bound the template module's T*N^2 pair memory
    # (memory-management deviation from OSS, which stacks all templates).
    max_templates: int | None = None
    token_s: int = _Default.token_s
    token_z: int = _Default.token_z
    atom_s: int = _Default.atom_s
    atom_z: int = _Default.atom_z
    num_bins: int = _Default.num_bins
    atoms_per_window_queries: int = _Default.atoms_per_window_queries
    atoms_per_window_keys: int = _Default.atoms_per_window_keys
    fix_sym_check: bool = _Default.fix_sym_check
    cyclic_pos_enc: bool = _Default.cyclic_pos_enc
    bond_type_feature: bool = _Default.bond_type_feature
    min_dist: float = (2.0,)
    max_dist: float = (22.0,)
    conditioning_cutoff_min: float = 4.0
    conditioning_cutoff_max: float = 20.0
    num_distograms: int = 1
    use_no_atom_char: bool = _Default.use_no_atom_char
    use_atom_backbone_feat: bool = _Default.use_atom_backbone_feat
    use_residue_feats_atoms: bool = _Default.use_residue_feats_atoms
    confidence_prediction: bool = True
    skip_run_structure: bool = False
    recompute_rel_pos: bool = True

    input_embedder: InputEmbedderConfig = InputEmbedderConfig()

    # v2 template module OFF by default (most folding runs supply no template);
    # enable via TrunkConfig(use_templates_v2=True) when templates are provided.
    # The pipeline emits dummy template_* features plus ``has_templates=False``
    # when none are supplied; the model then skips TemplateV2Module.
    trunk: TrunkConfig = TrunkConfig(use_templates_v2=False)

    structure_module: StructureModuleConfig = StructureModuleConfig()

    confidence_module: ConfidenceModuleConfig = ConfidenceModuleConfig()


class AffinityModuleConfig(BaseConfig):
    token_s: int = _Default.token_s
    token_z: int = _Default.token_z
    num_dist_bins: int = 64
    max_dist: int = 22
    pairformer_num_blocks: int = 8
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4


class AffinityEnsembleConfig(AffinityModuleConfig):
    module1: AffinityModuleConfig = AffinityModuleConfig(pairformer_num_blocks=8)
    module2: AffinityModuleConfig = AffinityModuleConfig(pairformer_num_blocks=4)


class Boltz2AffinityConfig(Boltz2Config):
    # The affinity checkpoint (boltz2_aff.ckpt) has no template_module weights,
    # so keep the template module off for affinity (avoid random-init weights).
    trunk: TrunkConfig = TrunkConfig(use_templates_v2=False)
    affinity: AffinityEnsembleConfig = AffinityEnsembleConfig()


PRETRAINED_CONFIG_REGISTRY = {
    SupMat.Boltz2: Boltz2Config,
    SupMat.Boltz2Affinity: Boltz2AffinityConfig,
}
