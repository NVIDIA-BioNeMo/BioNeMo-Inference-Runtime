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

from tensorrt_bionemo.configs import (BaseConfig, DiffusionTransformerConfig,
                                      MSAModuleConfig, PairformerConfig)
from tensorrt_bionemo.pipeline.boltz.const import NUM_TOKENS


class Boltz2Config(BaseConfig):
    token_s: int = 384
    token_z: int = 128
    atom_s: int = 128
    atom_z: int = 16
    num_bins: int = 64
    atoms_per_window_queries: int = 32
    atoms_per_window_keys: int = 128
    fix_sym_check: bool = True
    cyclic_pos_enc: bool = True
    bond_type_feature: bool = True
    conditioning_cutoff_min: float = 4.0
    conditioning_cutoff_max: float = 20.0
    num_distograms: int = 1
    use_no_atom_char: bool = False
    use_atom_backbone_feat: bool = False
    use_residue_feats_atoms: bool = False

    input_embedder: BaseConfig = BaseConfig(
        atom_s=atom_s,
        atom_z=atom_z,
        token_s=token_s,
        token_z=token_z,
        atoms_per_window_queries=atoms_per_window_queries,
        atoms_per_window_keys=atoms_per_window_keys,
        atom_feature_dim=388,
        add_method_conditioning=True,
        add_modified_flag=True,
        add_cyclic_flag=True,
        add_mol_type_feat=True,
        use_no_atom_char=use_no_atom_char,
        use_atom_backbone_feat=use_atom_backbone_feat,
        use_residue_feats_atoms=use_residue_feats_atoms,
        diffusion_transformer=DiffusionTransformerConfig(
            num_blocks=3,
            num_heads=4,
            dim=atom_s,
            dim_single_cond=atom_s,
            dim_pairwise=atom_z,
            bias_proj=False,
            conditioned_transition_using_silu=False,
            expansion_factor=2,
            version="v2",
        ),
    )

    trunk: BaseConfig = BaseConfig(
        msa_module=MSAModuleConfig(
            msa_s=64,
            token_z=token_z,
            token_s=token_s,
            msa_blocks=4,
            pairwise_head_width=32,
            pairwise_num_heads=4,
            num_tokens=NUM_TOKENS,
            use_paired_feature=True,
            opm_chunk_size=16,
            opm_mask_chunk_size=256,
            trimul_high_precision=False,
            version="v2",
        ),
        pairformer=PairformerConfig(
            token_s=token_s,
            token_z=token_z,
            pairwise_head_width=32,
            pairwise_num_heads=4,
            num_blocks=64,
            num_heads=16,
            trimul_high_precision=False,
            attention_initial_norm=False,
            version="v2",
        ),
    )

    structure_module: BaseConfig = BaseConfig(
        atom_diffusion=BaseConfig(
            sigma_min=0.0004,
            sigma_max=10.0,
            sigma_data=16,
            rho=7,
            P_mean=-1.2,
            P_std=1.5,
            gamma_0=0.8,
            gamma_min=1.0,
            noise_scale=1.0,
            step_scale=1.638,
            coordinate_augmentation=True,
            alignment_reverse_diff=True,
            synchronize_sigmas=False,
            accumulate_token_repr=False,
            num_sampling_steps=50,
            token_s=token_s,
            dim_fourier=256,
            version="v2",
        ),
        score_model=BaseConfig(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            dim_fourier=256,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            conditioning_transition_layers=2,
            atom_encoder=DiffusionTransformerConfig(
                num_blocks=3,
                num_heads=4,
                dim=atom_s,
                dim_single_cond=atom_s,
                bias_proj=False,
                conditioned_transition_using_silu=False,
                expansion_factor=2,
                version="v2",
            ),
            token_transformer=DiffusionTransformerConfig(
                num_blocks=24,
                num_heads=16,
                dim=2 * token_s,
                dim_single_cond=2 * token_s,
                dim_pairwise=token_z,
                bias_proj=False,
                conditioned_transition_using_silu=False,
                expansion_factor=2,
                version="v2",
            ),
            atom_decoder=DiffusionTransformerConfig(
                num_blocks=3,
                num_heads=4,
                dim=atom_s,
                dim_single_cond=atom_s,
                bias_proj=False,
                conditioned_transition_using_silu=False,
                expansion_factor=2,
                version="v2",
            ),
            version="v2",
        ))

    confidence: BaseConfig = BaseConfig(pairformer=PairformerConfig(
        token_s=token_s,
        token_z=token_z,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        num_blocks=8,
        num_heads=16,
        trimul_high_precision=False,
        attention_initial_norm=False,
        version="v2",
    ))


class Boltz2AffinityConfig(Boltz2Config):
    affinity: BaseConfig = BaseConfig(
        module1=BaseConfig(
            token_s=Boltz2Config().token_s,
            token_z=Boltz2Config().token_z,
            num_dist_bins=64,
            max_dist=22,
            pairformer_num_blocks=8,
            pairwise_head_width=32,
            pairwise_num_heads=4,
        ),
        module2=BaseConfig(
            token_s=Boltz2Config().token_s,
            token_z=Boltz2Config().token_z,
            num_dist_bins=64,
            max_dist=22,
            pairformer_num_blocks=4,
            pairwise_head_width=32,
            pairwise_num_heads=4,
        ),
    )
