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
            version="v1",
            dtype="float32"), ))
