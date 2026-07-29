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

from tensorrt_bionemo.configs import (BaseConfig, DiffusionTransformerConfig,
                                      EvoformerStackConfig, PairformerConfig)
from tensorrt_bionemo.registry import SupMat
from tensorrt_bionemo._torch.graph_optimization.config import \
    GraphOptimizationConfig, CUDAGraphOptimizationConfig, InputKeyMethod, GraphOptimizationMode
    
from tensorrt_bionemo._torch.graph_optimization.config import \
    InputRoutingConfigFactory

class _Default:
    c_z: int = 128
    c_s: int = 384
    n_query: int = 32
    n_key: int = 128
    c_s_input: int = 449


class InputEmbedderAllAtomConfig(BaseConfig):
    c_s_input: int = 449
    c_atom_ref_element: int = 119
    c_atom_ref_name_chars: int = 256
    c_atom: int = 128
    c_atom_pair: int = 16
    c_token: int = 384
    c_hidden: int = 32
    n_transition: int = 2
    n_query: int = _Default.n_query
    n_key: int = _Default.n_key
    c_s: int = _Default.c_s
    c_z: int = _Default.c_z
    max_relative_idx: int = 32
    max_relative_chain: int = 2
    add_noisy_pos: bool = False
    atom_transformer_config: DiffusionTransformerConfig = DiffusionTransformerConfig(
        num_blocks=3,
        num_heads=4,
        dim=128,
        dim_single_cond=128,
        dim_pairwise=16,
        post_layer_norm=False,
        bias_proj=True,
        dtype="float32",
        initial_norm=False,
        attention_initial_norm=False,
        use_ada_layer_norm=True,
        use_separate_layer_norm=True,
        conditioned_transition_using_silu=True,
        version="v1",
        shared_pair_norm=True)


class TemplateEmbedderConfig(BaseConfig):
    c_z: int = _Default.c_z

    template_pair_embedder: BaseConfig = BaseConfig(
        c_in=128,
        c_dgram=39,
        c_aatype=32,
        c_out=64,
    )

    template_pair_stack: BaseConfig = BaseConfig(c_t=64,
                                                 c_hidden_tri_att=16,
                                                 c_hidden_tri_mul=64,
                                                 no_blocks=2,
                                                 no_heads=4,
                                                 tri_mul_first=True,
                                                 trimul_high_precision=False,
                                                 pair_transition_n=2,
                                                 transition_type="swiglu")


class MSAModuleStackConfig(EvoformerStackConfig):
    c_m: int = 64
    c_z: int = 128
    c_hidden_msa_att: int = 8
    c_hidden_opm: int = 32
    c_hidden_mul: int = 128
    c_hidden_pair_att: int = 32
    transition_type: str = 'swiglu'
    transition_n: int = 4
    no_blocks: int = 4
    no_heads_msa: int = 8
    no_heads_pair: int = 4
    opm_first: bool = True
    last_block: bool = False
    inf: float = 1e9
    support_batch: bool = True
    msa_att_row_chunk_size: int = 4
    trimul_high_precision: bool = False


class MSAModuleEmbedderConfig(BaseConfig):
    c_m_feats: int = 34
    c_m: int = 64
    c_s_input: int = 449
    subsample_main_msa: bool = False
    subsample_all_msa: bool = True
    min_subsampled_all_msa: int = 1024
    max_subsampled_all_msa: int = 1024

def of3_dit_input_routing_config():
    input_routing = InputRoutingConfigFactory()
    input_routing.set_input_acceptance_dim(
        dim_name="num_tokens",
        dim_len_max=1024,
    )
    # Inputs (kwargs by name); z carries the token axis on dims -2 and -3.
    input_routing.input_dim_is_acceptance("a", -2, "num_tokens")
    input_routing.input_dim_is_acceptance("s", -2, "num_tokens")
    input_routing.input_dim_is_acceptance("z", -2, "num_tokens")
    input_routing.input_dim_is_acceptance("z", -3, "num_tokens")
    input_routing.input_dim_is_acceptance("mask", -1, "num_tokens")

    return input_routing.export_config()

def of3_diff_mod_input_routing_config():
    
    input_routing = InputRoutingConfigFactory()
    input_routing.set_input_acceptance_dim(
        dim_name="num_tokens",
        dim_len_max=1024,
    )
    # DiffusionModule.forward args (kwargs by name): the token axis rides
    # token_mask (-1), si_input/si_trunk (-2), and zij_trunk (-2 and -3).
    # xl_noisy/atom_mask carry the atom axis, so they are left untied.
    input_routing.input_dim_is_acceptance("si_input", -2, "num_tokens")
    input_routing.input_dim_is_acceptance("si_trunk", -2, "num_tokens")
    input_routing.input_dim_is_acceptance("zij_trunk", -2, "num_tokens")
    input_routing.input_dim_is_acceptance("zij_trunk", -3, "num_tokens")
    input_routing.input_dim_is_acceptance("token_mask", -1, "num_tokens")

    return input_routing.export_config()

class DiffusionModuleConfig(BaseConfig):
    c_s_input: int = _Default.c_s_input
    c_atom_ref_element: int = 119
    c_atom_ref_name_chars: int = 256
    c_atom: int = 128
    c_atom_pair: int = 16
    c_hidden: int = 32
    n_transition: int = 2
    n_query: int = 32
    n_key: int = 128
    c_s: int = _Default.c_s
    c_z: int = _Default.c_z
    c_token: int = 768
    sigma_data: float = 16
    eps: float = 1e-5
    inf: float = 1e9
    add_noisy_pos: bool = True
    diffusion_conditioning_config: BaseConfig = BaseConfig(
        c_fourier_emb=256, max_relative_idx=32, max_relative_chain=2)
    atom_transformer_encoder_config: BaseConfig = BaseConfig(
        num_blocks=3,
        num_heads=4,
        dim=128,
        dim_single_cond=128,
        dim_pairwise=16,
        post_layer_norm=False,
        bias_proj=True,
        dtype="float32",
        eps=1e-5,
        inf=1e9,
        initial_norm=False,
        attention_initial_norm=False,
        use_ada_layer_norm=True,
        use_separate_layer_norm=True,
        conditioned_transition_using_silu=True,
        version="v1",
        shared_pair_norm=True,
    )
    diffusion_transformer_config: BaseConfig = BaseConfig(
        token_transformer=DiffusionTransformerConfig(
            num_blocks=24,
            num_heads=16,
            dim=768,
            dim_single_cond=c_s,
            dim_pairwise=c_z,
            expansion_factor=2,
            bias_proj=True,
            use_separate_layer_norm=False,
            conditioned_transition_using_silu=True,
            version="v1",
            dtype="float32",
            graph_optimization_config=CUDAGraphOptimizationConfig(
                graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
                input_key_method=InputKeyMethod.EXACT,
                input_routing_config=of3_dit_input_routing_config()
            )
        ) 
    )
    atom_transformer_decoder_config: BaseConfig = BaseConfig(
        num_blocks=3,
        num_heads=4,
        dim=128,
        dim_single_cond=128,
        dim_pairwise=16,
        post_layer_norm=False,
        bias_proj=True,
        dtype="float32",
        eps=1e-5,
        inf=1e9,
        initial_norm=False,
        attention_initial_norm=False,
        use_ada_layer_norm=True,
        use_separate_layer_norm=True,
        conditioned_transition_using_silu=True,
        version="v1",
        shared_pair_norm=True,
    )
    graph_optimization_config: GraphOptimizationConfig =  CUDAGraphOptimizationConfig(
        graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
        input_key_method=InputKeyMethod.EXACT,
        input_routing_config=of3_diff_mod_input_routing_config())


class SampleDiffusionConfig(BaseConfig):
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    use_conditioning: bool = True


class AuxiliaryHeadsConfig(BaseConfig):
    c_s_input: int = 449
    c_z: int = 128
    min_bin: float = 3.25
    max_bin: float = 50.75
    no_bin: int = 39
    max_atoms_per_token: int = 23
    inf: float = 1e9
    memory_efficient_mode: bool = True
    pairformer: PairformerConfig = PairformerConfig(
        token_s=384,
        token_z=128,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        num_blocks=4,
        num_heads=16,
        trimul_high_precision=False,
        version="v1",
        dtype="float32")
    pde: BaseConfig = BaseConfig(
        c_z=128,
        c_out=64,
    )
    lddt: BaseConfig = BaseConfig(
        c_s=384,
        max_atoms_per_token=23,
        c_out=50,
    )

    distogram: BaseConfig = BaseConfig(
        c_z=128,
        c_out=64,
    )
    experimentally_resolved: BaseConfig = BaseConfig(
        c_s=384,
        max_atoms_per_token=23,
        c_out=2,
    )
    pae: BaseConfig = BaseConfig(
        enabled=True,
        c_z=128,
        c_out=64,
    )


class NoiseScheduleConfig(BaseConfig):
    sigma_data: int = 16
    s_max: float = 160.0
    s_min: float = 0.0004
    p: int = 7


def of3_pairformer_input_routing_config():
    input_routing = InputRoutingConfigFactory()
    
    # input acceptance
    input_routing.set_input_acceptance_dim(
        dim_name="num_tokens",
        dim_len_max=1024,
    )
    # Inputs (kwargs by name); z carries the token axis on dims -2 and -3.
    input_routing.input_dim_is_acceptance("s", -2, "num_tokens")
    input_routing.input_dim_is_acceptance("z", -2, "num_tokens")
    input_routing.input_dim_is_acceptance("z", -3, "num_tokens")
    input_routing.input_dim_is_acceptance("mask", -1, "num_tokens")
    input_routing.input_dim_is_acceptance("pair_mask", -1, "num_tokens")
    input_routing.input_dim_is_acceptance("pair_mask", -2, "num_tokens")
    
    # bucketing
    input_routing.set_padded_dim(
        dim_name="num_tokens",
        dim_len_min=4,
        dim_len_max=2048,
        num_intervals=8,
        spacing_method="linear",
    )
    # Inputs (kwargs by name); z carries the token axis on dims -2 and -3.
    input_routing.input_dim_is_padded("s", -2, "num_tokens")
    input_routing.input_dim_is_padded("z", -2, "num_tokens")
    input_routing.input_dim_is_padded("z", -3, "num_tokens")
    input_routing.input_dim_is_padded("mask", -1, "num_tokens")
    input_routing.input_dim_is_padded("pair_mask", -1, "num_tokens")
    input_routing.input_dim_is_padded("pair_mask", -2, "num_tokens")
    # Output tensor 0 (the updated token representation) token axis.
    input_routing.output_dim_is_padded(0, -2, "num_tokens")
    input_routing.output_dim_is_padded(1, -2, "num_tokens")
    input_routing.output_dim_is_padded(1, -3, "num_tokens")

    return input_routing.export_config()

class OpenFold3Config(BaseConfig):
    c_z: int = _Default.c_z
    c_s: int = _Default.c_s
    num_recycles: int = 3
    n_query: int = _Default.n_query
    n_key: int = _Default.n_key
    no_rollout_steps: int = 200
    no_rollout_samples: int = 1

    input_embedder_config: BaseConfig = InputEmbedderAllAtomConfig()
    template_embedder_config: BaseConfig = TemplateEmbedderConfig()
    msa_module_embedder_config: BaseConfig = MSAModuleEmbedderConfig()
    msa_stack_module_config: BaseConfig = MSAModuleStackConfig()
    diffusion_module_config: BaseConfig = DiffusionModuleConfig()
    auxiliary_heads_config: BaseConfig = AuxiliaryHeadsConfig()
    sample_diffusion_config: BaseConfig = SampleDiffusionConfig()
    noise_schedule_config: BaseConfig = NoiseScheduleConfig()

    trunk: BaseConfig = BaseConfig(
        pairformer=PairformerConfig(
        token_s=c_s,
        token_z=c_z,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        num_blocks=48,
        num_heads=16,
        trimul_high_precision=False,
        version="v1",
        dtype="float32",
        graph_optimization_config=CUDAGraphOptimizationConfig(
            graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
            input_key_method=InputKeyMethod.BUCKETED_SHAPES,
            input_routing_config=of3_pairformer_input_routing_config()
            )
        )
    )

    structure_module: BaseConfig = BaseConfig(
        score_model=BaseConfig(token_transformer=DiffusionTransformerConfig(
            num_blocks=24,
            num_heads=16,
            dim=768,
            dim_single_cond=c_s,
            dim_pairwise=c_z,
            expansion_factor=2,
            bias_proj=True,
            use_separate_layer_norm=False,
            conditioned_transition_using_silu=True,
            attention_initial_norm=False,
            post_layer_norm=False,
            version="v1",
            dtype="float32"), ))


PRETRAINED_CONFIG_REGISTRY = {SupMat.OpenFold3: OpenFold3Config}
