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

from tensorrt_bionemo._torch.auto_chunk import (CHUNK_REGISTRY,
                                                OUTER_PRODUCT_MEAN,
                                                ChunkPolicy, chunk_apply)
from tensorrt_bionemo._torch.distributed import (
    AllReduceParams, get_default_tp_group_coordinator)
from tensorrt_bionemo.mapping import Mapping

from .linear import (Linear, TensorParallelMode, WeightMode,
                     WeightsLoadingConfig)


class OuterProductMean(nn.Module):
    """Outer product mean layer."""

    def __init__(self,
                 c_in: int,
                 c_hidden: int,
                 c_out: int,
                 eps: float = 1e-5,
                 mask_eps: float = 1e-3,
                 norm_mask_by_eps: bool = False,
                 norm_before_output: bool = True,
                 cast_to_float_before_einsum: bool = True,
                 bias_flags: dict[str, bool] = {
                     "proj_a": False,
                     "proj_b": False,
                     "proj_o": True
                 },
                 dtype: Optional[torch.dtype] = None,
                 skip_create_weights: bool = False,
                 mapping: Optional[Mapping] = None,
                 chunk_policy: Optional[ChunkPolicy] = None) -> None:
        """Initialize the outer product mean layer.

        Args:
            c_in: Input channel dimension.
            c_hidden: Hidden channel dimension.
            c_out: Output channel dimension.
            norm_before_output: Whether to normalize the output before projection (this for OpenFold family models).
            norm_mask_by_eps: Add mask by mask_eps to avoid zero division (this for OpenFold family models).
        """
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.c_out = c_out
        self.eps = eps
        self.mask_eps = mask_eps
        self.norm_mask_by_eps = norm_mask_by_eps
        self.cast_to_float_before_einsum = cast_to_float_before_einsum
        self.dtype = dtype
        self.mapping = mapping or Mapping()
        self.norm_before_output = norm_before_output
        # Output token-row chunking policy: chunk when the token dim N exceeds the (memory-scaled)
        # threshold. ``None`` uses the registry default.
        self.chunk_policy = (chunk_policy if chunk_policy is not None else
                             CHUNK_REGISTRY.get(OUTER_PRODUCT_MEAN))
        assert self.c_hidden % self.mapping.tp_size == 0, \
            "c_hidden must be divisible by tp_size"
        self.c_hidden = self.c_hidden // self.mapping.tp_size
        self.norm = nn.LayerNorm(c_in, eps=eps, dtype=dtype)
        self.fused_proj_a_b = Linear(
            c_in,
            2 * self.c_hidden * self.mapping.tp_size,
            bias=bias_flags["proj_a"] or bias_flags["proj_b"],
            dtype=dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_KV_LINEAR),
            skip_create_weights=skip_create_weights)
        self.proj_o = Linear(self.c_hidden * self.c_hidden *
                             self.mapping.tp_size * self.mapping.tp_size,
                             c_out,
                             bias=bias_flags["proj_o"],
                             dtype=dtype,
                             mapping=self.mapping,
                             tensor_parallel_mode=TensorParallelMode.ROW,
                             reduce_output=True,
                             skip_create_weights=skip_create_weights)
        self.group_comm = None
        if self.mapping.tp_size > 1:
            self.group_comm = get_default_tp_group_coordinator()
            assert self.group_comm is not None, "TP group coordinator is not initialized, please call register_tp_group_coordinator first"

    def _compute_num_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Pair-occupancy normalizer ``num_mask[b, i, j] = sum_s mask[b, s, i] * mask[b, s, j]``.

        This is exactly ``mask.T @ mask`` over the sequence dim, so ``torch.bmm`` contracts ``S``
        inside the GEMM and never materializes the ``[B, S, N, N]`` outer product (which is why the
        old path chunked ``S``). ``mask`` is ``[B, S, N, 1]``; returns ``[B, N, N, 1]``.
        """
        m = mask.squeeze(-1)  # [B, S, N]
        # ``bmm`` has no integer CUDA kernel; cast bool/int masks to fp32 (exact counts). Float
        # masks (e.g. the cast bf16/fp32 mask) are used as-is, matching the old mul+sum dtype.
        if not m.is_floating_point():
            m = m.float()
        num_mask = torch.bmm(m.transpose(1, 2),
                             m).unsqueeze(-1)  # [B, N, N, 1]
        if self.norm_mask_by_eps:
            return num_mask + self.mask_eps  # OpenFold family
        return num_mask.clamp(min=1)  # Boltz family

    def _forward_impl(
        self,
        a_rows: torch.Tensor,
        num_mask: torch.Tensor,
        b: torch.Tensor,
        out_dtype: torch.dtype,
        all_reduce_params: Optional[AllReduceParams] = None,
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
        z = self.proj_o(z.to(out_dtype), all_reduce_params=all_reduce_params)
        if not self.norm_before_output:
            z.div_(num_mask)
        return z

    def forward(
            self,
            m: torch.Tensor,
            mask: torch.Tensor,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
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
        if self.cast_to_float_before_einsum:
            a = (a * mask).float()
            b = (b * mask).float()
        else:
            a = a * mask
            b = b * mask
        if self.mapping.tp_size > 1:
            b = self.group_comm.all_gather(b, dim=-1)

        # num_mask via a single batched matmul (mask.T @ mask over S) -- no [B, S, N, N] intermediate
        # is materialized.
        num_mask = self._compute_num_mask(mask)

        # Row-chunk the output token dim (concat) to bound the dominant [i, N, c_hidden**2] einsum.
        # ``a``'s token dim is dim=2, so move it to dim=1 to slice in lockstep with ``num_mask``
        # (i at dim=1); ``b`` (the contracted key side) passes through whole. ``chunk_apply`` falls
        # back to a single dense call below the policy threshold.
        policy = self.chunk_policy
        if policy is not None:
            return chunk_apply(self._forward_impl,
                               a.transpose(1, 2),
                               num_mask,
                               policy=policy,
                               cat_dim=1,
                               b=b,
                               out_dtype=m.dtype,
                               all_reduce_params=all_reduce_params)
        # Policy explicitly disabled -> single dense full-row call.
        return self._forward_impl(a.transpose(1, 2),
                                  num_mask,
                                  b=b,
                                  out_dtype=m.dtype,
                                  all_reduce_params=all_reduce_params)
