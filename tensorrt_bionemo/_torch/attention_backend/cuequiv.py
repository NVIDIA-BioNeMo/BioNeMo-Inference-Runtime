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
from cuequivariance_torch.primitives.triangle import triangle_attention

from ..tensor_utils import permute_final_dims
from .interface import AttentionBackend, AttentionMetadata


class CuEquivAttentionMetadata(AttentionMetadata):
    flip_mask: bool = True


@torch.compiler.disable
def _invoke_triangle_attention_kernel(q: torch.Tensor, k: torch.Tensor,
                                      v: torch.Tensor, bias: torch.Tensor,
                                      mask: torch.Tensor,
                                      sm_scale: float) -> torch.Tensor:
    o = triangle_attention(q, k, v, bias, mask=mask, scale=sm_scale)
    return o


class CuEquivAttention(AttentionBackend[CuEquivAttentionMetadata]):

    Metadata = CuEquivAttentionMetadata

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
        """Implementation of cuequiv attention."""
        mask = biases[0]
        bias = biases[1]
        if metadata is None:
            metadata = CuEquivAttentionMetadata()
        assert q.ndim == k.ndim == v.ndim, "q, k, v must have the same number of dimensions"
        # Steps:
        # 1. Flatten the batch dimensions
        # 2. Permute q, k, v to the correct shape
        # 3. Flatten the bias and mask - expand the bias to the broadcasting-able shape
        # 4. Cast mask to boolean, invert if needed
        # 5. Unflatten batch dimensions for output
        n_batch_dims = q.ndim - 3
        if n_batch_dims == 0:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            mask = mask.unsqueeze(0)
            bias = bias.unsqueeze(0)
        batch_dims = q.shape[:-3]
        n_batch_dims = len(batch_dims)
        i, j, hd = q.shape[-3:]
        d = hd // self.num_heads

        # b i j (h d) -> b i h j d
        bijhd = list(batch_dims) + [i, j, self.num_heads, d]
        q = q.view(*bijhd).flatten(0, n_batch_dims - 1)
        q = permute_final_dims(q, (0, 2, 1, 3)).contiguous()
        k = k.view(*bijhd).flatten(0, n_batch_dims - 1)
        k = permute_final_dims(k, (0, 2, 1, 3)).contiguous()
        v = v.view(*bijhd).flatten(0, n_batch_dims - 1)
        v = permute_final_dims(v, (0, 2, 1, 3)).contiguous()

        bias = bias.flatten(0, n_batch_dims - 1)
        # b h i j -> b () h i j
        bias = bias.unsqueeze(-4).contiguous()

        mask = mask.flatten(0, n_batch_dims - 1)
        mask = mask.bool().contiguous()
        if metadata.flip_mask:
            mask = ~mask

        sm_scale = self.head_dim**-0.5
        o = _invoke_triangle_attention_kernel(q, k, v, bias, mask, sm_scale)
        #  b i h j d -> b i j h d
        o = permute_final_dims(o, (0, 2, 1, 3)).contiguous()
        if n_batch_dims > 1:
            o = o.view(*batch_dims, *o.shape[-4:])
        return o
