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

from bionemo_ir._torch.custom_ops.fused_ln_proj_moveaxis_pad import LNProjMoveaxisPad
from bionemo_ir._torch.custom_ops.pair_weighted_averaging import (
    PairWeightedAveragingCuTe,
    get_pair_weighted_averaging_op,
    is_profitable_pwa_shape,
    is_supported_pwa_dims,
    transposed_value_buffer,
)
from bionemo_ir._torch.utils import CHUNK_REGISTRY, PAIR_WEIGHTED_AVERAGING, ChunkPolicy, chunk_apply
from bionemo_ir.dsl_kernels.triton.pwa_prologue import pwa_prologue

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
        # Chunk the eager [B, H, S, N, D] intermediate along S.
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

        # Each fused kernel is specialized for (H=num_heads, D=c_h, c_m).
        self._pwa_op_eligible = is_supported_pwa_dims(self.num_heads, self.c_h, c_m)
        self._pwa_op = get_pair_weighted_averaging_op(
            dtype or torch.get_default_dtype(),
            D=self.c_h,
            c_m=self.c_m,
            H=self.num_heads,
        )
        # The fused path normalizes and projects z in one read and never materializes norm_z(z).
        self._pair_bias = (
            LNProjMoveaxisPad(D=self.c_z, H=self.num_heads, dtype=dtype or torch.get_default_dtype(), eps=eps)
            if self._pwa_op_eligible and isinstance(self._pwa_op, PairWeightedAveragingCuTe)
            else None
        )
        # A kernel that loads v^T gets it from one batched GEMM instead of a relayout copy per call.
        self._transposed_v = self._pair_bias is not None and self._pwa_op.loads_transposed_value(self.c_h, self.c_m)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            m(torch.Tensor): Sequence tensor [B, S, N, c_m].
            z(torch.Tensor): Pair tensor [B, N, N, c_z].
            mask(torch.Tensor): Pair mask [B, N, N].
        Returns:
            torch.Tensor: Output tensor [B, S, N, c_m].
        """
        m = self.norm_m(m)

        # Fuse einsum -> gate -> proj_o without materializing the whole [B, H, S, N, D] intermediate.
        # A large S still chunks that projection: the fused kernel is row-wise along S, and the unchunked
        # [B, S, N, 2*H*D] activation is the memory blow-up the eager path already avoids.
        op = self._pwa_op
        dtype_str = {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(m.dtype)
        if (
            dtype_str is not None
            and self._pwa_op_eligible
            and isinstance(op, PairWeightedAveragingCuTe)
            and is_profitable_pwa_shape(
                self.num_heads, self.c_h, self.c_m, m.shape[2], sm_version=op.sm_version, dtype_str=dtype_str
            )
            and op.launch_available(m.shape[2], m.shape[2], m.shape[1], m.dtype, self.num_heads, self.c_h, self.c_m)
        ):
            return self._forward_fused(m, z, mask, op)

        # Eager fallback, optionally chunked along S.
        z = self.norm_z(z)
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

    def _pair_weights(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Normalize, project and softmax the raw pair once. The result does not depend on the MSA row."""
        norm = self.norm_z
        if (
            self._pair_bias is not None
            and isinstance(norm, nn.LayerNorm)
            and norm.weight is not None
            and norm.bias is not None
            and norm.weight.dtype == norm.bias.dtype == z.dtype
        ):
            b = self._pair_bias(z, norm.weight, norm.bias, self.proj_z.weight, proj_z=self.proj_z)  # [B, H, N, N]
        else:
            b = self.proj_z(norm(z)).permute(0, 3, 1, 2)  # [B, H, N, N]
        # The prologue applies the mask, the softmax and the J padding in one pass. PyTorch's expression below
        # takes a bias pass, a long-row softmax and a pad: 3.9 ms against 0.7 ms at 8192 tokens on B200.
        w = pwa_prologue(b, mask, self.inf)  # [B, H, N, Jp], or None outside its envelope
        if w is not None:
            return w
        b = b + (1 - mask[:, None]) * -self.inf
        w = torch.softmax(b, dim=-1)  # [B, H, N, N]
        n_tokens = w.shape[-1]
        padded = (n_tokens + 7) // 8 * 8
        if padded != n_tokens:
            w = F.pad(w, (0, padded - n_tokens))  # [B, H, N, Jp], zero-padded along J
        return w

    def _fused_rows(
        self,
        m: torch.Tensor,
        w: torch.Tensor,
        op: PairWeightedAveragingCuTe,
    ) -> torch.Tensor:
        """Project one slice of MSA rows and run the fused kernel. Output is [B, S, N, c_m]."""
        if self._transposed_v:
            # One batched GEMM writes v^T = W_v @ m^T [B, S, H*D, N], and the op takes its transposed view as v.
            # g stays [B, S, N, H*D].
            B, S, N, _ = m.shape
            hidden = self.c_h * self.num_heads
            weight = self.fused_proj_m_g.weight
            m = m.contiguous()
            v_t = transposed_value_buffer(B, S, hidden, N, m.dtype, m.device)
            torch.bmm(
                weight[:hidden].expand(B * S, hidden, self.c_m),
                m.view(B * S, N, self.c_m).transpose(1, 2),
                out=v_t.view(B * S, hidden, N),
            )
            g = self.fused_proj_m_g.apply_linear(m, weight[hidden:], None)
            return op(w, v_t.transpose(-1, -2), g, self.proj_o.weight)
        vg = self.fused_proj_m_g(m)
        v, g = vg.split([self.c_h * self.num_heads, self.c_h * self.num_heads], dim=-1)
        # Keep v/g as [B, S, N, H*D] views; each head occupies one contiguous D block.
        # g is pre-sigmoid [B, S, N, H*D]; proj_o.weight is [c_m, H*D].
        return op(w, v, g, self.proj_o.weight)

    def _forward_fused(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        op: PairWeightedAveragingCuTe,
    ) -> torch.Tensor:
        """Run fused PWA on the raw pair ``z`` and return [B, S, N, c_m], chunked along S when the policy says so."""
        w = self._pair_weights(z, mask)
        if self.chunk_policy is not None:
            return chunk_apply(self._fused_rows, m, policy=self.chunk_policy, cat_dim=1, w=w, op=op)
        return self._fused_rows(m, w, op)
