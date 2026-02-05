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

from tensorrt_bionemo.configs import (BaseConfig, EvoformerStackConfig,
                                      ExtraMSAStackConfig)
from tensorrt_bionemo.hubs import FoldingSupportMatrix as SupMat


class _Default:
    c_z: int = 128
    c_s: int = 384
    c_m: int = 256
    c_e: int = 64
    c_t: int = 64
    is_multimer: bool = False
    enable_extra_msa: bool = True
    enable_template: bool = True
    max_extra_msa: int = 1024
    skip_template_pair_stack: bool = False

    # Config for pipeline
    msa_cluster_features: bool = True
    max_recycling_iters: int = 3
    reduce_msa_clusters_by_max_templates: bool = False
    use_template_torsion_angles: bool = True
    max_msa_clusters: int = 512
    max_templates: int = 4
    resample_msa_in_recycling: bool = True


class InputEmbedderConfig(BaseConfig):
    c_z: int = _Default.c_z
    c_m: int = _Default.c_m
    tf_dim: int = 22
    msa_dim: int = 49
    relpos_k: int = 32


class InputEmbedderMultimerConfig(BaseConfig):
    c_z: int = _Default.c_z
    c_m: int = _Default.c_m
    tf_dim: int = 21
    msa_dim: int = 49
    max_relative_chain: int = 2
    max_relative_idx: int = 32
    use_chain_relative: bool = True


class RecyclingEmbedderConfig(BaseConfig):
    c_z: int = _Default.c_z
    c_m: int = _Default.c_m
    min_bin: float = 3.25
    max_bin: float = 20.75
    no_bins: int = 15


class ExtraMsaEmbedderConfig(BaseConfig):
    c_in: int = 25
    c_out: int = _Default.c_e


# Define configs for template embedding
# Monomer
class TemplateDistogramConfig(BaseConfig):
    min_bin: float = 3.25
    max_bin: float = 50.75
    no_bins: int = 39


class TemplateSingleEmbedderConfig(BaseConfig):
    c_in: int = 57
    c_out: int = _Default.c_m


class TemplatePairEmbedderConfig(BaseConfig):
    c_in: int = 88
    c_out: int = _Default.c_t


class TemplatePairStackConfig(BaseConfig):
    c_t: int = _Default.c_t
    c_hidden_tri_att: int = 16
    c_hidden_tri_mul: int = 64
    no_blocks: int = 2
    no_heads: int = 4
    pair_transition_n: int = 2
    tri_mul_first: bool = False
    trimul_high_precision: bool = False
    triangle_attn_node_chunk_size: int = 512


class TemplatePointwiseAttentionConfig(BaseConfig):
    c_t: int = _Default.c_t
    c_z: int = _Default.c_z
    c_hidden: int = 16
    no_heads: int = 4
    chunk_size: int = 256


class TemplateEmbedderConfig(BaseConfig):
    embed_angles: bool = True
    use_unit_vector: bool = False
    distogram: TemplateDistogramConfig = TemplateDistogramConfig()
    template_single_embedder: TemplateSingleEmbedderConfig = TemplateSingleEmbedderConfig(
    )
    template_pair_embedder: TemplatePairEmbedderConfig = TemplatePairEmbedderConfig(
    )
    template_pair_stack: TemplatePairStackConfig = TemplatePairStackConfig()
    template_pointwise_attention: TemplatePointwiseAttentionConfig = TemplatePointwiseAttentionConfig(
    )


# multimer
class TemplateSingleEmbedderMultimerConfig(BaseConfig):
    c_in: int = 34
    c_out: int = _Default.c_m


class TemplatePairEmbedderMultimerConfig(BaseConfig):
    c_in: int = _Default.c_z
    c_out: int = _Default.c_t
    c_dgram: int = 39
    c_aatype: int = 22


class TemplateEmbedderMultimerConfig(BaseConfig):
    c_t: int = _Default.c_t
    c_z: int = _Default.c_z
    embed_angles: bool = True
    use_unit_vector: bool = True
    distogram: TemplateDistogramConfig = TemplateDistogramConfig()
    template_single_embedder: TemplateSingleEmbedderMultimerConfig = TemplateSingleEmbedderMultimerConfig(
    )
    template_pair_embedder: TemplatePairEmbedderMultimerConfig = TemplatePairEmbedderMultimerConfig(
    )
    template_pair_stack: TemplatePairStackConfig = TemplatePairStackConfig()


class TrunkConfig(BaseConfig):
    evoformer_stack: EvoformerStackConfig = EvoformerStackConfig(
        c_m=_Default.c_m,
        c_z=_Default.c_z,
        c_s=_Default.c_s,
        c_hidden_msa_att=32,
        c_hidden_opm=32,
        c_hidden_mul=128,
        c_hidden_pair_att=32,
        no_heads_msa=8,
        no_heads_pair=4,
        transition_n=4,
        no_blocks=48,
        no_column_attention=False,
        opm_first=False,
        n_seq=516,
        trimul_high_precision=False)
    extra_msa_stack: ExtraMSAStackConfig = ExtraMSAStackConfig(
        c_m=_Default.c_e,
        c_z=_Default.c_z,
        c_hidden_msa_att=8,
        c_hidden_opm=32,
        c_hidden_mul=128,
        c_hidden_pair_att=32,
        no_heads_msa=8,
        no_heads_pair=4,
        no_blocks=4,
        opm_first=False,
        transition_n=4,
        trimul_high_precision=False,
        opm_chunk_size=16,
        opm_mask_chunk_size=256,
    )


class StructureModuleConfig(BaseConfig):
    c_s: int = _Default.c_s
    c_z: int = _Default.c_z
    c_ipa: int = 16
    c_resnet: int = 128
    no_heads_ipa: int = 12
    no_qk_points: int = 4
    no_v_points: int = 8
    no_blocks: int = 8
    no_transition_layers: int = 1
    no_resnet_blocks: int = 2
    no_angles: int = 7
    trans_scale_factor: float = 10.0
    epsilon: float = 1e-05
    inf: float = 100000.0
    is_multimer: bool = _Default.is_multimer


class PerResidueLddtConfig(BaseConfig):
    no_bins: int = 50
    c_in: int = 384
    c_hidden: int = 128


class ConfidenceDistogramConfig(BaseConfig):
    c_z: int = 128
    no_bins: int = 64


class MaskedMsaConfig(BaseConfig):
    c_m: int = 256
    c_out: int = 23


class ExperimentallyResolvedConfig(BaseConfig):
    c_s: int = 384
    c_out: int = 37


class TmConfig(BaseConfig):
    enabled: bool = True
    c_z: int = 128
    no_bins: int = 64
    iptm_weight: float = 0.8
    ptm_weight: float = 0.2


class ConfidenceModuleConfig(BaseConfig):
    per_residue_lddt: PerResidueLddtConfig = PerResidueLddtConfig()
    distogram: ConfidenceDistogramConfig = ConfidenceDistogramConfig()
    masked_msa: MaskedMsaConfig = MaskedMsaConfig()
    experimentally_resolved: ExperimentallyResolvedConfig = ExperimentallyResolvedConfig(
    )
    tm: TmConfig = TmConfig()
    epsilon: float = 1e-5


class OpenFold2Config(BaseConfig):
    c_z: int = _Default.c_z
    c_s: int = _Default.c_s
    c_m: int = _Default.c_m
    c_e: int = _Default.c_e
    max_extra_msa: int = _Default.max_extra_msa
    is_multimer: bool = _Default.is_multimer
    enable_extra_msa: bool = _Default.enable_extra_msa
    enable_template: bool = _Default.enable_template
    skip_template_pair_stack: bool = _Default.skip_template_pair_stack

    # Config for pipeline
    max_recycling_iters: int = _Default.max_recycling_iters
    reduce_msa_clusters_by_max_templates: bool = _Default.reduce_msa_clusters_by_max_templates
    use_template_torsion_angles: bool = _Default.use_template_torsion_angles
    max_msa_clusters: int = _Default.max_msa_clusters
    msa_cluster_features: bool = _Default.msa_cluster_features
    max_templates: int = _Default.max_templates
    resample_msa_in_recycling: bool = _Default.resample_msa_in_recycling

    input_embedder: InputEmbedderConfig = InputEmbedderConfig()
    recycling_embedder: RecyclingEmbedderConfig = RecyclingEmbedderConfig()
    extra_msa_embedder: ExtraMsaEmbedderConfig = ExtraMsaEmbedderConfig()
    template_embedder: TemplateEmbedderConfig = TemplateEmbedderConfig()
    trunk: TrunkConfig = TrunkConfig()
    structure_module: StructureModuleConfig = StructureModuleConfig()
    confidence_module: ConfidenceModuleConfig = ConfidenceModuleConfig()


class OpenFold2MultimerConfig(OpenFold2Config):

    input_embedder: InputEmbedderMultimerConfig = InputEmbedderMultimerConfig()
    template_embedder: TemplateEmbedderMultimerConfig = TemplateEmbedderMultimerConfig(
    )

    @model_validator(mode="after")
    def fill_config(self) -> "OpenFold2MultimerConfig":
        self.trunk.evoformer_stack.opm_first = True
        self.trunk.extra_msa_stack.opm_first = True
        self.is_multimer = True
        self.structure_module.is_multimer = True
        self.max_extra_msa = 1152
        self.enable_template = True
        self.confidence_module.masked_msa.c_out = 22
        self.structure_module.trans_scale_factor = 20.0
        self.recycle_early_stop_tolerance = 0.5
        self.max_msa_clusters = 252
        self.max_recycling_iters = 20
        return self


"""
Setting                         config_preset          AlphaFold params                    OpenFold params
---------------------------------------------------------------------------------------------------------------
With template, no ptm           model_1                params_model_1.npz                  finetuning_[2-5].pt
                                model_2                params_model_2.npz

With template, with ptm         model_1_ptm            params_model_1_ptm.npz              finetuning_ptm_[1-2].pt
                                model_2_ptm            params_model_2_ptm.npz

Without template, no ptm        model_3                params_model_3.npz                  finetuning_no_templ_[1-2].pt
                                model_4                params_model_4.npz
                                model_5                params_model_5.npz

Without template, with ptm      model_3_ptm            params_model_3_ptm.npz              finetuning_no_templ_ptm_1.pt
                                model_4_ptm            params_model_4_ptm.npz
                                model_5_ptm            params_model_5_ptm.npz
"""


class OpenFold2_FT2_Config(OpenFold2Config):

    @model_validator(mode="after")
    def fill_config(self) -> "OpenFold2_FT2_Config":
        self.enable_template = True
        self.confidence_module.tm.enabled = False
        return self


class OpenFold2_FT3_Config(OpenFold2_FT2_Config):

    @model_validator(mode="after")
    def fill_config(self) -> "OpenFold2_FT3_Config":
        self.confidence_module.tm.enabled = False
        return self


class OpenFold2_FT4_Config(OpenFold2_FT2_Config):

    @model_validator(mode="after")
    def fill_config(self) -> "OpenFold2_FT4_Config":
        self.confidence_module.tm.enabled = False
        return self


class OpenFold2_FT5_Config(OpenFold2_FT2_Config):

    @model_validator(mode="after")
    def fill_config(self) -> "OpenFold2_FT5_Config":
        self.confidence_module.tm.enabled = False
        return self


class OpenFold2_PTM1_Config(OpenFold2Config):

    @model_validator(mode="after")
    def fill_config(self) -> "OpenFold2_PTM1_Config":
        self.max_extra_msa = 5120
        self.enable_template = True
        return self


class OpenFold2_PTM2_Config(OpenFold2_PTM1_Config):
    pass


class OpenFold2_NoTempl1_Config(OpenFold2Config):

    @model_validator(mode="after")
    def fill_config(self) -> "OpenFold2_NoTempl1_Config":
        self.enable_template = False
        self.confidence_module.tm.enabled = False
        return self


class OpenFold2_NoTempl2_Config(OpenFold2_NoTempl1_Config):
    pass


class OpenFold2_NoTempl_PTM1_Config(OpenFold2Config):

    @model_validator(mode="after")
    def fill_config(self) -> "OpenFold2_NoTempl_PTM1_Config":
        self.enable_template = False
        return self


class OpenFold2_NoTempl_PTM2_Config(OpenFold2_NoTempl_PTM1_Config):
    pass


class AlphaFold2_1_Config(OpenFold2Config):

    @model_validator(mode="after")
    def fill_config(self) -> "AlphaFold2_1_Config":
        self.enable_template = True
        self.max_extra_msa = 5120
        self.confidence_module.tm.enabled = False
        self.reduce_msa_clusters_by_max_templates = True
        self.use_template_torsion_angles = True
        return self


class AlphaFold2_2_Config(OpenFold2Config):

    @model_validator(mode="after")
    def fill_config(self) -> "AlphaFold2_2_Config":
        self.enable_template = True
        self.confidence_module.tm.enabled = False
        self.reduce_msa_clusters_by_max_templates = True
        self.use_template_torsion_angles = True
        return self


class AlphaFold2_3_Config(OpenFold2Config):

    @model_validator(mode="after")
    def fill_config(self) -> "AlphaFold2_3_Config":
        self.enable_template = False
        self.max_extra_msa = 5120
        self.confidence_module.tm.enabled = False
        return self


class AlphaFold2_4_Config(OpenFold2Config):

    @model_validator(mode="after")
    def fill_config(self) -> "AlphaFold2_4_Config":
        self.enable_template = False
        self.max_extra_msa = 5120
        self.confidence_module.tm.enabled = False
        return self


class AlphaFold2_5_Config(OpenFold2Config):

    @model_validator(mode="after")
    def fill_config(self) -> "AlphaFold2_5_Config":
        self.enable_template = False
        self.confidence_module.tm.enabled = False
        return self


class AlphaFold2_Multimer_1_Config(OpenFold2MultimerConfig):
    # Parent validator call first -> child validator
    @model_validator(mode="after")
    def fill_msa_config(self) -> "OpenFold2MultimerConfig":
        self.max_extra_msa = 2048
        return self


class AlphaFold2_Multimer_2_Config(AlphaFold2_Multimer_1_Config):
    pass


class AlphaFold2_Multimer_3_Config(AlphaFold2_Multimer_1_Config):
    pass


class AlphaFold2_Multimer_4_Config(OpenFold2MultimerConfig):
    pass


class AlphaFold2_Multimer_5_Config(OpenFold2MultimerConfig):
    pass


PRETRAINED_CONFIG_REGISTRY = {
    SupMat.OpenFold2_FT2: OpenFold2_FT2_Config,
    SupMat.OpenFold2_FT3: OpenFold2_FT3_Config,
    SupMat.OpenFold2_FT4: OpenFold2_FT4_Config,
    SupMat.OpenFold2_FT5: OpenFold2_FT5_Config,
    SupMat.OpenFold2_PTM1: OpenFold2_PTM1_Config,
    SupMat.OpenFold2_PTM2: OpenFold2_PTM2_Config,
    SupMat.OpenFold2_NoTempl1: OpenFold2_NoTempl1_Config,
    SupMat.OpenFold2_NoTempl2: OpenFold2_NoTempl2_Config,
    SupMat.OpenFold2_NoTempl_PTM1: OpenFold2_NoTempl_PTM1_Config,
    SupMat.OpenFold2_NoTempl_PTM2: OpenFold2_NoTempl_PTM2_Config,
    SupMat.AlphaFold2_1: AlphaFold2_1_Config,
    SupMat.AlphaFold2_2: AlphaFold2_2_Config,
    SupMat.AlphaFold2_3: AlphaFold2_3_Config,
    SupMat.AlphaFold2_4: AlphaFold2_4_Config,
    SupMat.AlphaFold2_5: AlphaFold2_5_Config,
    SupMat.AlphaFold2_Multimer_1: AlphaFold2_Multimer_1_Config,
    SupMat.AlphaFold2_Multimer_2: AlphaFold2_Multimer_2_Config,
    SupMat.AlphaFold2_Multimer_3: AlphaFold2_Multimer_3_Config,
    SupMat.AlphaFold2_Multimer_4: AlphaFold2_Multimer_4_Config,
    SupMat.AlphaFold2_Multimer_5: AlphaFold2_Multimer_5_Config,
}
