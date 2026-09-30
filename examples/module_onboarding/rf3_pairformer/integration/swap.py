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
Module swap: replace the RF3 Recycler's pairformer_stack with a BioIR PairformerModule.
"""

import torch.nn as nn
from convert.convert_weights import convert_pairformer_stack_weights

from bionemo_ir._torch.layers.transformers.pairformer import PairformerModule
from bionemo_ir._torch.layers.triangle_nodes import TriangleAttentionEndingNode, TriangleAttentionStartingNode
from bionemo_ir.configs.modules import PairformerConfig

from .adapter import RF3PairformerAdapter
from .config import make_rf3_pairformer_config

# RF3's triangle attention has biases on its gate and output projections.
TRIANGLE_ATTENTION_BIASES = {"q": False, "k": False, "v": False, "g": True, "o": True}


def build_pairformer_module(config: PairformerConfig) -> PairformerModule:
    """Build a PairformerModule whose triangle attention carries RF3's biases.

    PairformerConfig has no switch for these biases, so rebuild each layer's two
    triangle attention nodes with the layer's arguments plus ``mha_bias_flags``.
    """
    module = PairformerModule(config)
    shape = (config.token_z, config.pairwise_head_width, config.pairwise_num_heads)
    for i, layer in enumerate(module.layers):
        options = {
            "inf": config.mask_inf,
            "layer_idx": i,
            "dtype": config.torch_dtype,
            "skip_create_weights": config.skip_create_weights,
            "attn_backend": config.triangle_attention_backend,
            "mha_bias_flags": TRIANGLE_ATTENTION_BIASES,
        }
        layer.tri_attn_start = TriangleAttentionStartingNode(*shape, **options)
        layer.tri_attn_end = TriangleAttentionEndingNode(
            *shape, transposed_bias=config.tri_attn_transposed_bias, **options
        )
    return module


def swap_pairformer_stack(recycler, source_state_dict, num_blocks=48, device="cuda", **config_kwargs):
    """Replace ``recycler.pairformer_stack`` with a BioIR PairformerModule.

    Args:
        recycler: RF3 Recycler nn.Module instance.
        source_state_dict: State dict of the recycler, with keys under
            ``pairformer_stack.{i}``. Take it before calling this function; the
            swap discards the original stack.
        num_blocks: Number of pairformer blocks.
        device: Target device.
        **config_kwargs: Forwarded to ``make_rf3_pairformer_config``, such as
            ``dtype`` and the attention backends.
    """
    config = make_rf3_pairformer_config(num_blocks=num_blocks, **config_kwargs)
    module = build_pairformer_module(config)
    converted = convert_pairformer_stack_weights(source_state_dict, num_blocks=num_blocks)
    # The converter returns flat parameter names, so load it as a state dict.
    # PairformerModule.load_weights takes per-submodule groups.
    module.load_state_dict(converted, strict=True)

    # The Recycler runs `for block in self.pairformer_stack: S_I, Z_II = block(S_I, Z_II)`,
    # so a one-element ModuleList holds the adapter, which runs every layer.
    recycler.pairformer_stack = nn.ModuleList([RF3PairformerAdapter(module.to(device).eval())])
    return recycler
