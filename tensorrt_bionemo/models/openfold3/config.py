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
                                      PairformerConfig)


class _Default:
    c_z: int = 128
    c_s: int = 384
    n_query: int = 32
    n_key: int = 128


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
        use_seperate_layer_norm=True,
        conditioned_transition_using_silu=True,
        version="v1")


class OpenFold3Config(BaseConfig):
    c_z: int = 128
    c_s: int = 384

    trunk: BaseConfig = BaseConfig(pairformer=PairformerConfig(
        token_s=c_s,
        token_z=c_z,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        num_blocks=48,
        num_heads=16,
        trimul_high_precision=False,
        version="v1",
        dtype="float32"), )

    structure_module: BaseConfig = BaseConfig(
        score_model=BaseConfig(token_transformer=DiffusionTransformerConfig(
            num_blocks=24,
            num_heads=16,
            dim=768,
            dim_single_cond=c_s,
            dim_pairwise=c_z,
            expansion_factor=2,
            bias_proj=True,
            conditioned_transition_using_silu=True,
            attention_initial_norm=False,
            post_layer_norm=False,
            version="v1",
            dtype="float32"), ))
