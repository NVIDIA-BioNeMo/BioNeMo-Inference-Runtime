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
from tensorrt_llm.functional import AllReduceParams

from tensorrt_bionemo._torch.distributed import DPCommManager
from tensorrt_bionemo.mapping import Mapping

from .linear import Linear, TensorParallelMode, WeightMode, WeightsLoadingConfig


class OuterProductMean(nn.Module):
    """Outer product mean layer."""

    def __init__(self,
                 c_in: int,
                 c_hidden: int,
                 c_out: int,
                 eps: float = 1e-5,
                 dtype: torch.dtype = None,
                 skip_create_weights: bool = False,
                 mapping: Optional[Mapping] = None) -> None:
        """Initialize the outer product mean layer.

        Args:
            c_in: Input channel dimension.
            c_hidden: Hidden channel dimension.
            c_out: Output channel dimension.
        """
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.c_out = c_out
        self.eps = eps
        self.dtype = dtype
        self.mapping = mapping or Mapping()
        assert self.c_hidden % self.mapping.tp_size == 0, \
            "c_hidden must be divisible by tp_size"
        self.c_hidden = self.c_hidden // self.mapping.tp_size
        self.norm = nn.LayerNorm(c_in, eps=eps, dtype=dtype)
        self.fused_proj_a_b = Linear(
            c_in,
            2 * self.c_hidden * self.mapping.tp_size,
            bias=False,
            dtype=dtype,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_KV_LINEAR),
            skip_create_weights=skip_create_weights)
        self.proj_o = Linear(self.c_hidden * self.c_hidden *
                             self.mapping.tp_size,
                             c_out,
                             bias=True,
                             dtype=dtype,
                             mapping=self.mapping,
                             tensor_parallel_mode=TensorParallelMode.ROW,
                             reduce_output=True,
                             skip_create_weights=skip_create_weights)

        self.dp_comm = None
        if self.mapping.tp_size > 1:
            DPCommManager.init_dp_comm(self.mapping)
            self.dp_comm = DPCommManager()

    def forward(
            self,
            m: torch.Tensor,
            mask: torch.Tensor,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        """Forward pass.
        TODO: Support chunking mechanism.
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
        a = (a * mask).float()
        b = (b * mask).float()

        mask = mask[:, :, None, :] * mask[:, :, :, None]
        num_mask = mask.sum(1).clamp(min=1)

        if self.mapping.tp_size == 1:
            z = torch.einsum("bsic,bsjd->bijcd", a, b)
        else:
            # ring communication to compute z
            buffers = [a, a_recv]  # double buffers
            send_idx = 0
            recv_idx = 1
            for i in range(1, self.mapping.tp_size):
                self.dp_comm.batch_isend_irecv(buffers[send_idx],
                                               buffers[recv_idx])
                z = torch.einsum("bsic,bsjd->bijcd", buffers[send_idx], b)
                recv_idx = send_idx
                send_idx ^= 1  # flip the buffer
            z = torch.cat(z, dim=3)
        z = z.reshape(*z.shape[:3], -1)
        z = z / num_mask

        z = self.proj_o(z.to(m.dtype), all_reduce_params=all_reduce_params)
        return z
