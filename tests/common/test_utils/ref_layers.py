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
from typing import Optional

import torch
import torch.nn as nn
from test_utils.ref_attn import RefTriangleAttention

from tensorrt_bionemo._torch.hf.checkpoints import load_hf_weights


class RefTriangleAttentionNode(nn.Module):
    """ Reference: https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/layers/triangular_attention/attention.py """

    def __init__(self,
                 c_in: int,
                 c_hidden: int,
                 num_heads: int,
                 starting: bool = True,
                 inf: float = 1e9):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = nn.LayerNorm(self.c_in)

        self.linear = nn.Linear(c_in, self.num_heads, bias=False)

        self.mha = RefTriangleAttention(self.c_in, self.c_in, self.c_in,
                                        self.c_hidden, self.num_heads)

    @classmethod
    def load_weights(
            cls,
            model: str = "boltz-1",
            triattn_layer_path: str = "pairformer_module.layers.0.tri_att_start",
            no_heads: int = 4,
            starting: bool = True,
            state_dict: Optional[dict] = None) -> 'RefTriangleAttentionNode':
        if state_dict is None:
            state_dict = load_hf_weights(model, local_files_only=False)
        mha = RefTriangleAttention.load_weights(
            state_dict=state_dict,
            triattn_layer_path=triattn_layer_path + ".mha",
            no_heads=no_heads)
        c_in = mha.c_q
        c_hidden = mha.c_hidden
        num_heads = mha.no_heads

        node = cls(c_in, c_hidden, num_heads, starting)
        node.mha = mha

        weights_path = [
            f"{triattn_layer_path}.linear.weight",
            f"{triattn_layer_path}.layer_norm.weight",
        ]
        biases_path = [
            None,
            f"{triattn_layer_path}.layer_norm.bias",
        ]
        layers = [
            node.linear,
            node.layer_norm,
        ]
        for weights_path, bias_path, layer in zip(weights_path, biases_path,
                                                  layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return node

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        if not self.starting:
            x = x.transpose(-2, -3)
        # [*, I, J, C_in]
        x = self.layer_norm(x)
        # [*, H, I, J]
        lx = self.linear(x)
        if lx.dim() == 4:
            triangle_bias = torch.permute(
                lx, (0, 3, 1, 2))  # TA.permute_final_dims(lx, (2, 0, 1))
        elif lx.dim() == 3:
            triangle_bias = torch.permute(
                lx, (2, 0, 1))  # TA.permute_final_dims(lx, (2, 0, 1))
        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        mask_bias: Optional[torch.Tensor] = None
        if mask is not None:
            if not self.starting:
                mask = mask.transpose(-1, -2)
            # [*, I, 1, 1, J]
            mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
            biases = [mask_bias, triangle_bias]
        else:
            biases = [triangle_bias]

        x = self.mha(q_x=x, kv_x=x, biases=biases)
        if not self.starting:
            x = x.transpose(-2, -3)
        return x
