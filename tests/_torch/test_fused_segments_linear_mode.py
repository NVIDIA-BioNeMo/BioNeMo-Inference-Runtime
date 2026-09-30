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

import pytest
import torch

from bionemo_ir._torch.layers.attention import AttentionPairBias
from bionemo_ir._torch.layers.linear import Linear, WeightMode, WeightsLoadingConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Linear allocates its weights on CUDA")
def test_fused_segments_linear_mode_pads_rows_and_zero_fills_missing_biases():
    linear = Linear(
        in_features=8,
        out_features=4 + 4 + 32,
        bias=True,
        weights_loading_config=WeightsLoadingConfig(
            weight_mode=WeightMode.FUSED_SEGMENTS_LINEAR,
            segment_sizes=(4, 4, 32),
        ),
    )
    linear.weight.data.fill_(float("nan"))
    linear.bias.data.fill_(float("nan"))
    q_weight, g_weight, g_bias, b_weight = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4), torch.randn(3, 8)

    linear.load_weights(
        [{"weight": q_weight, "bias": None}, {"weight": g_weight, "bias": g_bias}, {"weight": b_weight}]
    )

    weight, bias = linear.weight.cpu(), linear.bias.cpu()
    torch.testing.assert_close(weight, torch.cat([q_weight, g_weight, b_weight, torch.zeros(29, 8)]))
    torch.testing.assert_close(bias, torch.cat([torch.zeros(4), g_bias, torch.zeros(32)]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Linear allocates its weights on CUDA")
def test_attention_pair_bias_split_projection_after_deferred_weight_creation():
    attn = AttentionPairBias(layer_idx=0, c_s=16, c_z=8, num_heads=2, dtype=torch.float32, skip_create_weights=True)
    for module in attn.modules():
        if isinstance(module, Linear):
            module.create_weights()
    q_size, kv_size = attn.q_size, attn.kv_size
    attn.in_proj.load_weights(
        [
            {"weight": torch.randn(q_size, 16), "bias": torch.randn(q_size)},
            {"weight": torch.randn(q_size, 16)},
            {"weight": torch.randn(kv_size, 16)},
            {"weight": torch.randn(kv_size, 16)},
        ]
    )
    s = torch.randn(2, 3, 16, device="cuda")

    fused = attn._prep_qkvg(s, s)
    # A distinct kv_in tensor takes the separate [q; g] and [k; v] GEMMs.
    split = attn._prep_qkvg(s, s.clone())

    for actual, expected in zip(split, fused, strict=True):
        torch.testing.assert_close(actual, expected)
