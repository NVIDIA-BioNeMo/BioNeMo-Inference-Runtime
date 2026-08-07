# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import cuequivariance_ops_torch as _cueq_ops  # noqa: F401 – registers torch.ops.cuequivariance
import torch

from ...tensor_utils import permute_final_dims
from ..interface import AttentionBackend, AttentionMetadata


class CuEquivAttentionMetadata(AttentionMetadata):
    flip_mask: bool = True


def _invoke_triangle_attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None,
    sm_scale: float,
    kv_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Call the cuequivariance triangle attention custom op directly.

    Using torch.ops.cuequivariance.triangle_attention (which has Meta + CUDA
    dispatch keys registered) instead of the Python wrapper avoids graph breaks
    under torch.compile.
    """
    actual_s_kv = None
    if kv_lengths is not None:
        # The public API uses [B, N, 1, 1, 1]; the custom op consumes [B, N].
        actual_s_kv = kv_lengths.reshape(q.shape[0], q.shape[1]).contiguous()
    o, _lse, _max = torch.ops.cuequivariance.triangle_attention(
        q,
        k,
        v,
        bias,
        mask,
        actual_s_kv,
        scale=sm_scale,
    )
    return o


class CuEquivAttention(AttentionBackend[CuEquivAttentionMetadata]):
    Metadata = CuEquivAttentionMetadata

    def __init__(self, layer_idx: int, num_heads: int, head_dim: int, num_kv_heads: int | None = None):
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads)
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads == num_kv_heads, "num_heads must be equal to num_kv_heads"
        capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        self._sm_version = capability[0] * 10 + capability[1]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        metadata: AttentionMetadata | None = None,
        use_kv_lengths: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Run cuEquivariance triangle attention.

        ``use_kv_lengths`` requires a prefix-shaped mask and enables the SM100f
        length fast path. Dense masks preserve arbitrary mask patterns.
        """
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
        q = permute_final_dims(q, (0, 2, 1, 3))
        k = k.view(*bijhd).flatten(0, n_batch_dims - 1)
        k = permute_final_dims(k, (0, 2, 1, 3))
        v = v.view(*bijhd).flatten(0, n_batch_dims - 1)
        v = permute_final_dims(v, (0, 2, 1, 3))

        use_kv_lengths = (
            use_kv_lengths
            and self._sm_version in (100, 103)
            and q.dtype in (torch.float16, torch.bfloat16)
            and d % 8 == 0
            and d <= 256
            and k.shape[3] > 0
            and k.shape[3] % 8 == 0
        )
        if not use_kv_lengths:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        bias = bias.flatten(0, n_batch_dims - 1)
        # b h i j -> b () h i j
        bias = bias.unsqueeze(-4)

        mask = mask.flatten(0, n_batch_dims - 1)
        mask = mask.bool().contiguous()
        if metadata.flip_mask:
            mask = ~mask

        kv_lengths = None
        if use_kv_lengths:
            kv_lengths = mask.sum(dim=-1, keepdim=True, dtype=torch.int32).detach().contiguous()
            mask = None

        sm_scale = self.head_dim**-0.5
        o = _invoke_triangle_attention_kernel(q, k, v, bias, mask, sm_scale, kv_lengths=kv_lengths)
        if kv_lengths is not None:
            o = o * (kv_lengths > 0).to(dtype=o.dtype)
        #  b i h j d -> b i j h d
        o = permute_final_dims(o, (0, 2, 1, 3)).contiguous()
        if n_batch_dims > 1:
            o = o.view(*batch_dims, *o.shape[-4:])
        return o
