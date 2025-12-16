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
from pydantic import model_validator

from tensorrt_bionemo.configs import (BaseConfig, DiffusionTransformerConfig,
                                      MSAModuleConfig, PairformerConfig)
from tensorrt_bionemo.hubs import FoldingSupportMatrix as SupMat
from tensorrt_bionemo.pipeline.boltz.const import NUM_TOKENS


class _Default:
    token_s: int = 384
    token_z: int = 128
    atom_s: int = 128
    atom_z: int = 16
    num_bins: int = 64
    atoms_per_window_queries: int = 32
    atoms_per_window_keys: int = 128


class InputEmbedderConfig(BaseConfig):
    atom_s: int = _Default.atom_s
    atom_z: int = _Default.atom_z
    token_s: int = _Default.token_s
    token_z: int = _Default.token_z
    atoms_per_window_queries: int = _Default.atoms_per_window_queries
    atoms_per_window_keys: int = _Default.atoms_per_window_keys
    atom_feature_dim: int = 389
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


class TrunkConfig(BaseConfig):
    msa_module: MSAModuleConfig = MSAModuleConfig(
        msa_s=64,
        token_z=_Default.token_z,
        token_s=_Default.token_s,
        msa_blocks=4,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        num_tokens=NUM_TOKENS,
        use_paired_feature=False,
        opm_chunk_size=16,
        opm_mask_chunk_size=256,
        trimul_high_precision=False,
        version="v1",
    )
    pairformer: PairformerConfig = PairformerConfig(
        token_s=_Default.token_s,
        token_z=_Default.token_z,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        num_blocks=48,
        num_heads=16,
        trimul_high_precision=False,
        attention_initial_norm=True,
        version="v1",
    )


class AtomDiffusionConfig(BaseConfig):
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: int = 16
    rho: int = 7
    P_mean: float = -1.2
    P_std: float = 1.5
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.0
    step_scale: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    accumulate_token_repr: bool = True
    num_sampling_steps: int = 200
    token_s: int = _Default.token_s
    dim_fourier: int = 256
    version: str = "v1"


class ScoreModelConfig(BaseConfig):
    atom_s: int = _Default.atom_s
    atom_z: int = _Default.atom_z
    token_s: int = _Default.token_s
    token_z: int = _Default.token_z
    dim_fourier: int = 256
    atoms_per_window_queries: int = _Default.atoms_per_window_queries
    atoms_per_window_keys: int = _Default.atoms_per_window_keys
    conditioning_transition_layers: int = 2
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
        version="v2")
    version: str = "v1"


class StructureModuleConfig(BaseConfig):
    atom_diffusion: AtomDiffusionConfig = AtomDiffusionConfig()
    score_model: ScoreModelConfig = ScoreModelConfig()
    version: str = "v1"


class ConfidenceHeadsConfig(BaseConfig):
    token_s: int = _Default.token_s
    token_z: int = _Default.token_z
    num_plddt_bins: int = 50
    num_pde_bins: int = 64
    num_pae_bins: int = 64
    compute_pae: bool = True


class ConfidenceModuleConfig(BaseConfig):
    token_s: int = _Default.token_s
    token_z: int = _Default.token_z
    num_dist_bins: int = 64
    max_dist: int = 22
    add_s_to_z_prod: bool = True
    add_s_input_to_s: bool = True
    use_s_diffusion: bool = True
    add_z_input_to_z: bool = True
    heads: ConfidenceHeadsConfig = ConfidenceHeadsConfig()
    msa_module: MSAModuleConfig = MSAModuleConfig()
    pairformer: PairformerConfig = PairformerConfig()
    input_embedder: InputEmbedderConfig = InputEmbedderConfig()


class Boltz1Config(BaseConfig):
    token_s: int = _Default.token_s
    token_z: int = _Default.token_z
    atom_s: int = _Default.atom_s
    atom_z: int = _Default.atom_z
    num_bins: int = _Default.num_bins
    atoms_per_window_queries: int = _Default.atoms_per_window_queries
    atoms_per_window_keys: int = _Default.atoms_per_window_keys

    input_embedder: InputEmbedderConfig = InputEmbedderConfig()

    trunk: TrunkConfig = TrunkConfig()

    structure_module: StructureModuleConfig = StructureModuleConfig()

    confidence_module: ConfidenceModuleConfig = ConfidenceModuleConfig()

    @model_validator(mode="after")
    def fill_confidence_config(self) -> "Boltz1Config":
        self.confidence_module.input_embedder = self.input_embedder
        self.confidence_module.msa_module = self.trunk.msa_module
        self.confidence_module.pairformer = self.trunk.pairformer
        return self


PRETRAINED_CONFIG_REGISTRY = {
    SupMat.Boltz1: Boltz1Config,
}
