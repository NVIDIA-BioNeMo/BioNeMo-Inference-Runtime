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

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_bionemo._torch.auto_chunk import CHUNK_REGISTRY, PAIR_WEIGHTED_AVERAGING, ChunkPolicy, chunk_apply
from tensorrt_bionemo._torch.custom_ops.pair_weighted_averaging import (
    PairWeightedAveragingCuTe,
    get_pair_weighted_averaging_op,
)

from .linear import Linear, WeightMode, WeightsLoadingConfig


class PairWeightedAveraging(nn.Module):
    """Pair weighted averaging layer."""

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_h: int,
        num_heads: int,
        inf: float = 1e9,
        eps: float = 1e-5,
        dtype: torch.dtype = None,
        skip_create_weights: bool = False,
        chunk_policy: ChunkPolicy | None = None,
    ) -> None:
        """
        Args:
            c_m(int): The dimension of the input sequence.
            c_z(int): The dimension of the input pairwise tensor.
            c_h(int): The dimension of the hidden.
            num_heads(int): The number of heads.
            inf(float): The infinity value.
            eps(float): The epsilon value.
            dtype(torch.dtype): The data type of the input tensor.
            skip_create_weights(bool): Whether to skip creating weights.
            chunk_policy(Optional[ChunkPolicy]): Sequence-row chunking policy; ``None`` uses the shared
                ``pair_weighted_averaging`` policy from ``CHUNK_REGISTRY``.
        """
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_h = c_h
        self.inf = inf
        # Sequence-row chunking policy for the eager fallback (registry default unless explicitly
        # overridden). The fused kernel is already memory-bounded and takes precedence when
        # available; otherwise chunking bounds the [B, H, S, N, D] eager intermediate.
        self.chunk_policy = chunk_policy if chunk_policy is not None else CHUNK_REGISTRY.get(PAIR_WEIGHTED_AVERAGING)

        self.num_heads = num_heads
        self.norm_m = nn.LayerNorm(self.c_m, dtype=dtype, eps=eps)
        self.norm_z = nn.LayerNorm(self.c_z, dtype=dtype, eps=eps)

        self.fused_proj_m_g = Linear(
            self.c_m,
            2 * self.c_h * self.num_heads,
            bias=False,
            dtype=dtype,
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_KV_LINEAR),
            skip_create_weights=skip_create_weights,
        )
        self.proj_z = Linear(
            self.c_z,
            self.num_heads,
            bias=False,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        self.proj_o = Linear(
            self.c_h * self.num_heads,
            c_m,
            bias=False,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

        # Eligibility for the fused PWA CuTe op (the einsum -> sigmoid(gate) -> proj_o
        # chain): the kernel has fixed dims (H=8, D=c_h=32, c_m=64). The op itself further
        # gates on SM/dtype/j-pad and falls back.
        self._pwa_op_eligible = self.num_heads == 8 and self.c_h == 32 and c_m == 64

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            m(torch.Tensor): The input sequence tensor (B, S, N, D)
            z(torch.Tensor): The input pairwise tensor (B, N, N, D)
            mask(torch.Tensor): The pairwise mask tensor (B, N, N)
        Returns:
            torch.Tensor: The output tensor (B, S, N, D)
        """
        m = self.norm_m(m)
        z = self.norm_z(z)

        # Fused CuTe path: collapses einsum -> gate -> proj_o so the
        # [B,H,S,N,D] intermediate is never materialized. The op further gates
        # on SM/dtype/j-padding support.
        if self._pwa_op_eligible:
            op = get_pair_weighted_averaging_op(m.dtype)
            if isinstance(op, PairWeightedAveragingCuTe):
                return self._forward_fused(m, z, mask, op)

        # Inference-only.
        if self.chunk_policy is not None:
            return chunk_apply(self._forward_impl, m, policy=self.chunk_policy, cat_dim=1, z=z, mask=mask)
        return self._forward_impl(m, z, mask)

    def _forward_impl(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        vg = self.fused_proj_m_g(m)
        v, g = vg.split([self.c_h * self.num_heads, self.c_h * self.num_heads], dim=-1)
        v = v.reshape(*v.shape[:3], self.num_heads, self.c_h)
        v = v.permute(0, 3, 1, 2, 4)
        g = g.sigmoid()

        b = self.proj_z(z)
        b = b.permute(0, 3, 1, 2)
        b = b + (1 - mask[:, None]) * -self.inf
        w = torch.softmax(b, dim=-1)

        o = torch.einsum("bhij,bhsjd->bhsid", w, v)
        o = o.permute(0, 2, 3, 1, 4)  # [B, S, N, H, D]
        o = o.reshape(*o.shape[:3], self.num_heads * self.c_h)
        o = self.proj_o(g * o)
        return o

    def _forward_fused(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        op: PairWeightedAveragingCuTe,
    ) -> torch.Tensor:
        """Fused PWA via the CuTe op. Prepares the kernel inputs -- softmax'd pair weights ``w``
        (zero-padded on j to a multiple of 8), values ``v``, the RAW (pre-sigmoid) gate ``g``, and
        ``proj_o.weight`` -- and returns the [B,S,N,c_m] projection (the kernel applies the sigmoid,
        fuses the value-GEMM + gate + proj_o, and never materializes o[B,H,S,N,D])."""
        vg = self.fused_proj_m_g(m)
        v, g = vg.split([self.c_h * self.num_heads, self.c_h * self.num_heads], dim=-1)
        # v, g are [B, S, N, H*D] views of vg (last dim H*D contiguous, N strided). The kernel reads
        # per-head D-blocks straight from this layout (head h = the h-th D-block), so we pass them
        # AS-IS -- no permute, no .contiguous(): that avoids two full
        # [B,S,N,H*D]-sized copies, which would dominate the memory and
        # latency cost of this path.

        b = self.proj_z(z)
        b = b.permute(0, 3, 1, 2)  # [B, H, N, N]
        b = b + (1 - mask[:, None]) * -self.inf
        w = torch.softmax(b, dim=-1)  # [B, H, N, N]

        N = w.shape[-1]
        Jp = (N + 7) // 8 * 8
        if Jp != N:
            w = F.pad(w, (0, Jp - N))  # [B, H, N, Jp]   (pad value 0 -- required; w only)

        # g is the RAW gate [B, S, N, H*D] (kernel applies sigmoid); proj_o.weight is [c_m, H*D].
        return op(w, v, g, self.proj_o.weight)
