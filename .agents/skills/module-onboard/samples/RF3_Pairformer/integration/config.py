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
#
# Layout reference: RoseTTAFold3 (RosettaCommons/foundry), BSD-3-Clause.
# https://github.com/RosettaCommons/foundry/tree/production/models/rf3
# Only upstream parameter and module names are reproduced here.

"""BioIR PairformerConfig for RF3 Pairformer hyperparameters."""

from bionemo_ir.configs.modules import PairformerConfig


def make_rf3_pairformer_config(
    num_blocks: int = 48,
    c_s: int = 384,
    c_z: int = 128,
    num_heads: int = 16,
    pairwise_head_width: int = 32,
    pairwise_num_heads: int = 4,
    dtype: str = "bfloat16",
    # Torch backend defaults
    triangle_attention_backend: str = "CuTeDSL",
    pairwise_attention_backend: str = "CuTeDSL",
) -> PairformerConfig:
    return PairformerConfig(
        num_blocks=num_blocks,
        token_s=c_s,
        token_z=c_z,
        num_heads=num_heads,
        pairwise_head_width=pairwise_head_width,
        pairwise_num_heads=pairwise_num_heads,
        dtype=dtype,
        triangle_attention_backend=triangle_attention_backend,
        pairwise_attention_backend=pairwise_attention_backend,
        attention_initial_norm=True,
        version="v1",
    )
