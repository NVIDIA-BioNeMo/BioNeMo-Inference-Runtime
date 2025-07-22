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
import triton
from einops import rearrange
from torch.library import wrap_triton

from tensorrt_bionemo.triton_kernels.trifast import create_autotuner

from .interface import AttentionBackend, AttentionMetadata


class TrifastAttentionMetadata(AttentionMetadata):
    closest_n: Optional[int] = 16


class TrifastAttention(AttentionBackend[TrifastAttentionMetadata]):

    Metadata = TrifastAttentionMetadata

    def __init__(self,
                 layer_idx: int,
                 num_heads: int,
                 head_dim: int,
                 num_kv_heads: Optional[int] = None):
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads)
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads == num_kv_heads, "num_heads must be equal to num_kv_heads"

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: Optional[list[torch.Tensor]] = None,
        metadata: Optional[AttentionMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Implementation of trifast attention."""
        mask = biases[0]
        bias = biases[1]
        if metadata is None:
            metadata = TrifastAttentionMetadata()
        assert q.ndim == k.ndim == v.ndim, "q, k, v must have the same number of dimensions"
        q.ndim
        if q.ndim == 3:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        if mask.ndim == 4:
            mask = mask.unsqueeze(0)
        if bias.ndim == 3:
            bias = bias.unsqueeze(0)

        bs, i, j, hd = q.shape
        q = rearrange(q, "b i j (h d) -> (b h) i j d",
                      h=self.num_heads).contiguous()
        k = rearrange(k, "b i j (h d) -> (b h) i j d",
                      h=self.num_heads).contiguous()
        v = rearrange(v, "b i j (h d) -> (b h) i j d",
                      h=self.num_heads).contiguous()
        bias = rearrange(bias, "b h i j -> (b h) i j").contiguous()
        mask = rearrange(mask, "b i () () j -> b i j").bool().contiguous()

        sm_scale = self.head_dim**-0.5
        bh = bs * self.num_heads

        o = torch.zeros_like(q)
        l = torch.zeros((bh, j, j), device=q.device, dtype=torch.float32)

        def grid(x):
            return (triton.cdiv(j, x["BLOCK_J"]), i, bh)

        trifast_attention_kernel_fwd = create_autotuner()

        CLOSEST_N = metadata.closest_n
        wrap_triton(trifast_attention_kernel_fwd)[grid](o,
                                                        o.stride(0),
                                                        o.stride(1),
                                                        o.stride(2),
                                                        o.stride(3),
                                                        l,
                                                        l.stride(0),
                                                        l.stride(1),
                                                        l.stride(2),
                                                        q,
                                                        q.stride(0),
                                                        q.stride(1),
                                                        q.stride(2),
                                                        q.stride(3),
                                                        k,
                                                        k.stride(0),
                                                        k.stride(1),
                                                        k.stride(2),
                                                        k.stride(3),
                                                        v,
                                                        v.stride(0),
                                                        v.stride(1),
                                                        v.stride(2),
                                                        v.stride(3),
                                                        bias,
                                                        bias.stride(0),
                                                        bias.stride(1),
                                                        bias.stride(2),
                                                        mask,
                                                        mask.stride(0),
                                                        mask.stride(1),
                                                        mask.stride(2),
                                                        neg_inf=torch.finfo(
                                                            q.dtype).min,
                                                        sm_scale=sm_scale,
                                                        batch_size=bs,
                                                        si=i,
                                                        seq_len=j,
                                                        heads=self.num_heads,
                                                        DIM=self.head_dim,
                                                        CLOSEST_N=CLOSEST_N)

        # l = rearrange(l, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
        o = rearrange(o, "(b h) i j d -> b i j h d", h=self.num_heads,
                      b=bs).contiguous()
        return o
