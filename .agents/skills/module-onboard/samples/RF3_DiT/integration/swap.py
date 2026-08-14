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

"""
Module swap: replace RF3 DiffusionModule's diffusion_transformer
with BioIR DiffusionTransformerLayer stack.
"""

import os
import sys

import torch.nn as nn

from bionemo_ir._torch.layers.transformers.diffusion_transformer import (
    DiffusionTransformerLayer,
)

from .adapter import RF3DiTStackAdapter
from .config import make_rf3_dit_config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from convert.convert_weights import convert_dit_block_weights


def swap_diffusion_transformer(
    diffusion_module,
    source_state_dict=None,
    num_blocks: int = 24,
    c_token: int = 384,
    c_s: int = 384,
    c_tokenpair: int = 128,
    num_heads: int = 16,
    dtype: str = "bfloat16",
    device: str = "cuda",
    pairwise_attention_backend: str = "SDPA",
):
    """Replace diffusion_module.diffusion_transformer with BioIR equivalent.

    Args:
        diffusion_module: RF3 DiffusionModule instance.
        source_state_dict: Optional state_dict from the source model's DiffusionTransformer.
            If None, BioIR modules use default initialization.
    """
    config = make_rf3_dit_config(
        num_blocks=num_blocks,
        c_token=c_token,
        c_s=c_s,
        c_tokenpair=c_tokenpair,
        num_heads=num_heads,
        dtype=dtype,
        pairwise_attention_backend=pairwise_attention_backend,
    )
    torch_dtype = config.torch_dtype

    layers = nn.ModuleList()
    for i in range(num_blocks):
        layer = DiffusionTransformerLayer(
            layer_idx=i,
            num_heads=num_heads,
            dim=c_token,
            dim_single_cond=c_s,
            dim_pairwise=c_tokenpair,
            dtype=torch_dtype,
            attn_backend=pairwise_attention_backend,
            initial_norm=True,
            bias_proj=True,
            pair_norm=True,
            attn_output_gate=False,
            conditioned_transition_using_silu=True,
        )
        if source_state_dict is not None:
            converted = convert_dit_block_weights(source_state_dict, f"blocks.{i}", "", c_token, num_heads)
            layer.load_state_dict({k: v.to(device) for k, v in converted.items()})
        layers.append(layer)

    layers = layers.to(device).eval()
    adapter = RF3DiTStackAdapter(layers)
    diffusion_module.diffusion_transformer = adapter
    return diffusion_module
