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
    triangle_attn_node_chunk_size: int = 0


class TemplatePointwiseAttentionConfig(BaseConfig):
    c_t: int = _Default.c_t
    c_z: int = _Default.c_z
    c_hidden: int = 16
    no_heads: int = 4


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
    use_unit_vector: bool = False
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
    )


class OpenFold2Config(BaseConfig):
    c_z: int = _Default.c_z
    c_s: int = _Default.c_s
    c_m: int = _Default.c_m
    c_e: int = _Default.c_e
    max_extra_msa: int = _Default.max_extra_msa
    is_multimer: bool = _Default.is_multimer
    enable_extra_msa: bool = _Default.enable_extra_msa
    enable_template: bool = _Default.enable_template

    input_embedder: InputEmbedderConfig = InputEmbedderConfig()
    recycling_embedder: RecyclingEmbedderConfig = RecyclingEmbedderConfig()
    extra_msa_embedder: ExtraMsaEmbedderConfig = ExtraMsaEmbedderConfig()
    template_embedder: TemplateEmbedderConfig = TemplateEmbedderConfig()
    trunk: TrunkConfig = TrunkConfig()


class OpenFold2MultimerConfig(OpenFold2Config):

    input_embedder: InputEmbedderMultimerConfig = InputEmbedderMultimerConfig()
    template_embedder: TemplateEmbedderMultimerConfig = TemplateEmbedderMultimerConfig(
    )

    @model_validator(mode="after")
    def fill_config(self) -> "OpenFold2MultimerConfig":
        self.trunk.evoformer_stack.opm_first = True
        self.trunk.extra_msa_stack.opm_first = True
        self.is_multimer = True
        self.max_extra_msa = 1152
        self.enable_template = True
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
        return self


class OpenFold2_FT3_Config(OpenFold2_FT2_Config):
    pass


class OpenFold2_FT4_Config(OpenFold2_FT2_Config):
    pass


class OpenFold2_FT5_Config(OpenFold2_FT2_Config):
    pass


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
        return self


class AlphaFold2_2_Config(OpenFold2Config):

    @model_validator(mode="after")
    def fill_config(self) -> "AlphaFold2_2_Config":
        self.enable_template = True
        return self


class AlphaFold2_3_Config(OpenFold2Config):

    @model_validator(mode="after")
    def fill_config(self) -> "AlphaFold2_3_Config":
        self.enable_template = False
        self.max_extra_msa = 5120
        return self


class AlphaFold2_4_Config(OpenFold2Config):

    @model_validator(mode="after")
    def fill_config(self) -> "AlphaFold2_4_Config":
        self.enable_template = False
        self.max_extra_msa = 5120
        return self


class AlphaFold2_5_Config(OpenFold2Config):

    @model_validator(mode="after")
    def fill_config(self) -> "AlphaFold2_5_Config":
        self.enable_template = False
        return self


class AlphaFold2_Multimer_1_Config(OpenFold2MultimerConfig):
    pass


class AlphaFold2_Multimer_2_Config(OpenFold2MultimerConfig):
    pass


class AlphaFold2_Multimer_3_Config(OpenFold2MultimerConfig):
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
