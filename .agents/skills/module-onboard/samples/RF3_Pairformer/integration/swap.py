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
Module swap: replace RF3 Recycler's pairformer_stack with TRT-BNM PairformerModule.
"""

from tensorrt_bionemo._torch.layers.transformers.pairformer import PairformerModule

from .adapter import RF3PairformerAdapter
from .config import make_rf3_pairformer_config


def swap_pairformer_stack(
    recycler,
    source_state_dict=None,
    num_blocks: int = 48,
    c_s: int = 384,
    c_z: int = 128,
    dtype: str = "bfloat16",
    device: str = "cuda",
    triangle_attention_backend: str = "CUEQUIV",
    pairwise_attention_backend: str = "SDPA",
):
    """Replace recycler.pairformer_stack with a TRT-BNM PairformerModule.

    Args:
        recycler: RF3 Recycler nn.Module instance.
        source_state_dict: Optional state_dict from the source model's pairformer_stack.
            If None, TRT-BNM module uses default (random) initialization.
        num_blocks: Number of pairformer blocks.
        c_s: Single representation dimension.
        c_z: Pair representation dimension.
        dtype: Weight dtype string.
        device: Target device.
        triangle_attention_backend: Triangle attention backend for Torch path.
        pairwise_attention_backend: Pairwise attention backend for Torch path.
    """
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from convert.convert_weights import convert_pairformer_stack_weights

    config = make_rf3_pairformer_config(
        num_blocks=num_blocks,
        c_s=c_s,
        c_z=c_z,
        dtype=dtype,
        triangle_attention_backend=triangle_attention_backend,
        pairwise_attention_backend=pairwise_attention_backend,
    )
    trtbnm_module = PairformerModule(config)

    if source_state_dict is not None:
        converted = convert_pairformer_stack_weights(source_state_dict, num_blocks=num_blocks)
        trtbnm_module.load_weights(converted)

    trtbnm_module = trtbnm_module.to(device)
    adapter = RF3PairformerAdapter(trtbnm_module)
    recycler.pairformer_stack = adapter
    return recycler
