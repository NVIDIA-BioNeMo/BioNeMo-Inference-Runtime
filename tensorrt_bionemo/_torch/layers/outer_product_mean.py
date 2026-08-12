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

from tensorrt_bionemo._torch.auto_chunk import CHUNK_REGISTRY, OUTER_PRODUCT_MEAN, ChunkPolicy, chunk_apply
from tensorrt_bionemo._torch.custom_ops.outer_product_mean import OuterProductMeanCuTe, get_outer_product_mean_op

from .linear import Linear, WeightMode, WeightsLoadingConfig


class OuterProductMean(nn.Module):
    """Outer product mean layer."""

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        c_out: int,
        eps: float = 1e-5,
        mask_eps: float = 1e-3,
        norm_mask_by_eps: bool = False,
        norm_before_output: bool = True,
        cast_to_float_before_einsum: bool = False,
        bias_flags: dict[str, bool] | None = None,
        dtype: torch.dtype | None = None,
        skip_create_weights: bool = False,
        chunk_policy: ChunkPolicy | None = None,
    ) -> None:
        """Initialize the outer product mean layer.

        Args:
            c_in: Input channel dimension.
            c_hidden: Hidden channel dimension.
            c_out: Output channel dimension.
            norm_before_output: Whether to normalize the output before projection (this for OpenFold family models).
            norm_mask_by_eps: Add mask by mask_eps to avoid zero division (this for OpenFold family models).
        """
        super().__init__()
        if bias_flags is None:
            bias_flags = {"proj_a": False, "proj_b": False, "proj_o": True}
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.c_out = c_out
        self.eps = eps
        self.mask_eps = mask_eps
        self.norm_mask_by_eps = norm_mask_by_eps
        self.cast_to_float_before_einsum = cast_to_float_before_einsum
        self.dtype = dtype
        self.norm_before_output = norm_before_output
        # Output token-row chunking policy: chunk when the token dim N exceeds the (memory-scaled)
        # threshold. ``None`` uses the registry default.
        self.chunk_policy = chunk_policy if chunk_policy is not None else CHUNK_REGISTRY.get(OUTER_PRODUCT_MEAN)
        # The fused custom op handles the full OPM without materializing the
        # [B, N, N, c_hidden**2] intermediate. If it is unavailable, forward
        # falls through to the registry-driven eager row-chunking path.
        self._opm_eligible = self.c_hidden == 32 and self.c_out == 128
        self._opm_op = get_outer_product_mean_op(
            dtype or torch.get_default_dtype(),
            C=self.c_hidden,
            D=self.c_hidden,
            C_z=self.c_out,
        )
        self.norm = nn.LayerNorm(c_in, eps=eps, dtype=dtype)
        self.fused_proj_a_b = Linear(
            c_in,
            2 * self.c_hidden,
            bias=bias_flags["proj_a"] or bias_flags["proj_b"],
            dtype=dtype,
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_KV_LINEAR),
            skip_create_weights=skip_create_weights,
        )
        self.proj_o = Linear(
            self.c_hidden * self.c_hidden,
            c_out,
            bias=bias_flags["proj_o"],
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

    def _compute_num_mask(self, mask: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Pair-occupancy normalizer ``num_mask[b, i, j] = sum_s mask[b, s, i] * mask[b, s, j]``.

        This is exactly ``mask.T @ mask`` over the sequence dim, so ``torch.bmm`` contracts ``S``
        inside the GEMM and never materializes the ``[B, S, N, N]`` outer
        product, so no chunking over ``S`` is needed. ``mask`` is
        ``[B, S, N, 1]``; returns ``[B, N, N, 1]``.
        """
        m = mask.squeeze(-1)  # [B, S, N]
        # The fused kernel requires fp32 normalization. The eager path preserves
        # the floating mask dtype; bool/int masks use fp32 because CUDA bmm has
        # no integer implementation and occupancy counts are represented exactly.
        if dtype is not None and m.dtype != dtype:
            m = m.to(dtype)
        elif not m.is_floating_point():
            m = m.float()
        num_mask = torch.bmm(m.transpose(1, 2), m).unsqueeze(-1)  # [B, N, N, 1]
        if self.norm_mask_by_eps:
            return num_mask + self.mask_eps  # OpenFold family
        return num_mask.clamp(min=1)  # Boltz family

    def _forward_impl(
        self,
        a_rows: torch.Tensor,
        num_mask: torch.Tensor,
        b: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Outer-product-mean for a slice of output token-rows ``i`` (position-wise over ``i``).

        Args:
            a_rows: ``a`` transposed to ``[B, i, S, c_hidden]`` (output-row dim moved to dim=1 so it
                slices in lockstep with ``num_mask``).
            num_mask: ``[B, i, N(j), 1]`` normalizer for these rows.
            b: full ``[B, S, N(j), c_hidden]`` -- the contracted key side, not chunked.
            out_dtype: dtype for the ``proj_o`` input / output.
        """
        a = a_rows.transpose(1, 2)  # [B, S, i, c_hidden]
        # [B, i, N, c_hidden, c_hidden] -- the row-chunked dominant intermediate.
        z = torch.einsum("bsic,bsjd->bijcd", a, b)
        if self.norm_before_output:
            z.div_(num_mask.unsqueeze(-1))
        z = z.reshape(*z.shape[:3], -1)  # [B, i, N, c_hidden**2]
        z = self.proj_o(z.to(out_dtype))
        if not self.norm_before_output:
            z.div_(num_mask)
        return z

    def forward(self, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            m(torch.Tensor): Input tensor of shape (B, S, N, c_in).
            mask(torch.Tensor): Mask tensor of shape (B, S, N).
        Returns:
            torch.Tensor: Output tensor of shape (B, N, N, c_out).
        """
        if m.dtype != self.dtype:
            m = m.to(self.dtype)
            mask = mask.to(self.dtype)
        mask = mask.unsqueeze(-1)
        m = self.norm(m)
        ab = self.fused_proj_a_b(m)
        a, b = ab.split([self.c_hidden, self.c_hidden], dim=-1)

        # Masked projections. Kept in the model dtype so the fused SM80 kernel
        # can consume them directly (it accumulates in fp32 internally); the
        # eager path casts to fp32 below when configured.
        a = a * mask
        b = b * mask

        use_fused_opm = self._opm_eligible and isinstance(self._opm_op, OuterProductMeanCuTe)

        # The fused CuTe OPM kernel expects an fp32 num_mask; the eager
        # PyTorch fallback computes it in the mask's own dtype.
        num_mask = self._compute_num_mask(mask, dtype=torch.float32 if use_fused_opm else mask.dtype)

        if use_fused_opm:
            return self._opm_op(
                a, b, num_mask.squeeze(-1), self.proj_o.weight, self.proj_o.bias, norm_before=self.norm_before_output
            )

        # The fused kernel is unavailable (unsupported dtype/hardware/dims),
        # so use the memory-bounded eager fallback below.
        if self.cast_to_float_before_einsum:
            a = a.float()
            b = b.float()
        policy = self.chunk_policy
        if policy is not None:
            return chunk_apply(
                self._forward_impl, a.transpose(1, 2), num_mask, policy=policy, cat_dim=1, b=b, out_dtype=m.dtype
            )
        # Policy explicitly disabled -> single dense full-row call.
        return self._forward_impl(a.transpose(1, 2), num_mask, b=b, out_dtype=m.dtype)
