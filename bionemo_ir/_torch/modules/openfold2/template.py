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

# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# Modified by NVIDIA Corporation and affiliates.

import torch
import torch.nn as nn
from einops import rearrange

from bionemo_ir._torch.layers.attention import CrossTriangleAttention
from bionemo_ir._torch.layers.transformers.template_pair import TemplatePairBlock, TemplatePairStack

__all__ = ["TemplatePairBlock", "TemplatePairStack", "TemplatePointwiseAttention"]


class TemplatePointwiseAttention(nn.Module):
    def __init__(
        self,
        c_t: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        inf: float = 1e9,
        eps: float = 1e-5,
        triangle_attn_backend: str = "VANILLA",
        chunk_size: int = 256,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
    ):
        super().__init__()

        self.c_t = c_t
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.chunk_size = chunk_size
        self.mha = CrossTriangleAttention(
            layer_idx=0,
            q_hidden_size=self.c_z,
            kv_hidden_size=self.c_t,
            head_dim=self.c_hidden,
            num_attention_heads=self.no_heads,
            num_key_value_heads=self.no_heads,
            gating=False,
            bias_flags={"q": False, "k": False, "v": False, "g": False, "z": False, "o": True},
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            attn_backend=triangle_attn_backend,
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor, template_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            z:
                [*, N_res, N_res, C_t] pair embedding
            template_mask:
                [*, N_templ] template mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        TODO: Support the chunking or flash-attention here.
        """
        if template_mask is None:
            template_mask = t.new_ones(t.shape[:-3])

        bias = self.inf * (template_mask[..., None, None, None, None, :] - 1)

        # [*, N_res, N_res, 1, C_z]
        z = z.unsqueeze(-2)

        # [*, N_res, N_res, N_temp, C_t]
        batch_dims = " ".join([f"b_{i}" for i in range(t.ndim - 4)])
        t = rearrange(t, f"{batch_dims} t i j c -> {batch_dims} i j t c")

        # [*, 1, 1, 1, N_temp]
        biases = [bias]
        if self.chunk_size > 1:
            seq_len = z.shape[1]
            niters = (seq_len + self.chunk_size - 1) // self.chunk_size
            outputs = []
            for i in range(niters):
                start = i * self.chunk_size
                end = start + self.chunk_size
                z_chunk = z[:, start:end:, :, :]
                t_chunk = t[:, start:end:, :, :]
                z_chunk = self.mha(q_x=z_chunk, kv_x=t_chunk, biases=biases)
                outputs.append(z_chunk)
            z = torch.cat(outputs, dim=1)
        else:
            z = self.mha(q_x=z, kv_x=t, biases=biases)

        # [*, N_res, N_res, C_z]
        z = z.squeeze(-2)

        return z
