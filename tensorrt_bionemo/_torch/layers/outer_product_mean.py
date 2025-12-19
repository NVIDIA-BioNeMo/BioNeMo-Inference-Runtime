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
                 chunk_size: Optional[int] = None,
                 mask_chunk_size: Optional[int] = None,
                 dtype: Optional[torch.dtype] = None,
                 skip_create_weights: bool = False,
                 mapping: Optional[Mapping] = None) -> None:
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
        self.chunk_size = chunk_size
        self.mask_chunk_size = mask_chunk_size
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

    @torch.compiler.disable
    def _compute_mask_with_chunking(self, mask: torch.Tensor) -> torch.Tensor:
        for i in range(0, mask.shape[1], self.mask_chunk_size):
            if i == 0:
                num_mask = (
                    mask[:, i:i + self.mask_chunk_size, None, :] *
                    mask[:, i:i + self.mask_chunk_size, :, None]).sum(1)
            else:
                num_mask += (
                    mask[:, i:i + self.mask_chunk_size, None, :] *
                    mask[:, i:i + self.mask_chunk_size, :, None]).sum(1)
        if self.norm_mask_by_eps:
            num_mask = num_mask + self.mask_eps
        else:
            num_mask = num_mask.clamp(min=1)
        return num_mask

    @torch.compiler.disable
    def _compute_output_with_chunking(self, m: torch.Tensor, a: torch.Tensor,
                                      b: torch.Tensor,
                                      num_mask: torch.Tensor) -> torch.Tensor:
        """ This is similar to split on TP but for single device
        See: https://github.com/jwohlwend/boltz/blob/v2.2.0/src/boltz/model/layers/outer_product_mean.py
        """

        for i in range(0, self.c_hidden, self.chunk_size):
            a_chunk = a[:, :, :, i:i + self.chunk_size]
            proj_o_sliced_weight = self.proj_o.weight[:, i * self.c_hidden:
                                                      (i + self.chunk_size) *
                                                      self.c_hidden]
            z = torch.einsum("bsic,bsjd->bijcd", a_chunk, b)
            z = z.reshape(*z.shape[:3], -1)
            if self.norm_before_output:
                z = z / num_mask
            # Project to output
            if i == 0:
                z_out = z.to(m) @ proj_o_sliced_weight.T
            else:
                z_out = z_out + z.to(m) @ proj_o_sliced_weight.T
        if self.proj_o.bias is not None:
            z_out = z_out + self.proj_o.bias  # add bias
        if not self.norm_before_output:
            z_out = z_out / num_mask
        return z_out

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

        if self.mask_chunk_size is not None:
            num_mask = self._compute_mask_with_chunking(mask)
        else:
            mask = mask[:, :, None, :] * mask[:, :, :, None]
            if self.norm_mask_by_eps:
                # This for OF family models
                num_mask = mask.sum(1) + self.mask_eps
            else:
                # This for Boltz family models
                num_mask = mask.sum(1).clamp(min=1)

        if self.chunk_size is None:
            z = torch.einsum("bsic,bsjd->bijcd", a, b)
        else:
            return self._compute_output_with_chunking(m, a, b, num_mask)

        z = z.reshape(*z.shape[:3], -1)
        if self.norm_before_output:
            z = z / num_mask

        z = self.proj_o(z.to(m.dtype), all_reduce_params=all_reduce_params)
        if not self.norm_before_output:
            z = z / num_mask
        return z
