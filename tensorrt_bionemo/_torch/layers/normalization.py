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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_bionemo._torch.distributed import \
    get_default_tp_group_coordinator
from tensorrt_bionemo._torch.layers.linear import (Linear, TensorParallelMode,
                                                   WeightMode,
                                                   WeightsLoadingConfig)

from tensorrt_bionemo._torch.custom_ops import get_adaln_layernorm_sigmoid_op
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.runtime.buffers import (PreallocatedBuffers,
                                              ensure_buffer)


class AdaLN(nn.Module):

    def __init__(self,
                 dim: int,
                 dim_single_cond: int,
                 eps: float = 1e-5,
                 dtype: torch.dtype = None,
                 skip_create_weights: bool = False,
                 mapping: Optional[Mapping] = None):
        """Adaptive LayerNorm with a sigmoid-gated affine.

        Uses the fused CuTe DSL kernel when ``tp_size == 1``; under TP
        falls back to the inline torch path (kernel doesn't handle the
        TP slice between LN and the gate). Kernel init / forward
        failures propagate to the caller — no silent fallback.
        """
        super().__init__()
        if mapping is None:
            mapping = Mapping()
        self.dim = dim // mapping.tp_size
        self.dim_single_cond = dim_single_cond
        self.mapping = mapping
        self.tp_group = mapping.tp_group
        self.eps = eps

        self.a_norm = nn.LayerNorm(self.dim * mapping.tp_size,
                                   dtype=dtype,
                                   eps=eps,
                                   elementwise_affine=False,
                                   bias=False)
        self.s_norm = nn.LayerNorm(self.dim_single_cond,
                                   dtype=dtype,
                                   eps=eps,
                                   bias=False)
        # Fused s_scale and s_bias, but s_bias has no bias
        # remember to set it to zero correctly
        self.fused_s_scale_s_bias = Linear(
            self.dim_single_cond,
            2 * mapping.tp_size * self.dim,
            bias=True,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
            skip_create_weights=skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_KV_LINEAR))

        self.group_comm = None
        if mapping.tp_size > 1:
            self.group_comm = get_default_tp_group_coordinator()
            assert self.group_comm(
            ) is not None, "TP group coordinator is not initialized, please call register_tp_group_coordinator first"

        # Fused kernel only handles the tp_size == 1 case because under TP the
        # LayerNorm operates on the full dim while the sigmoid gate operates on
        # the sliced dim — those two steps run on different tensors.
        self._fused_op = None
        if mapping.tp_size == 1:
            self._fused_op = get_adaln_layernorm_sigmoid_op(
                dtype if dtype is not None else torch.float32)

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        buffers: Optional[PreallocatedBuffers] = None,
        buffer_key: str = "adaln_out",
    ) -> torch.Tensor:
        """
        Args:
            a: [B, I, d]
            s: [B, I, d_cond]
            buffers: optional preallocated buffer dict for the kernel output.
            buffer_key: key into ``buffers`` for the AdaLN output tensor.

        Returns:
            a: [B, I, d]
        """
        # Pre-fused-step torch ops are reliable, so compute s_scale / s_bias
        # once up front. Both the fused kernel and the torch fallback consume
        # them — keeping these outside the try block avoids re-doing the
        # ``s_norm`` + linear + split if the kernel fails mid-forward.
        s_normed = self.s_norm(s)
        ss = self.fused_s_scale_s_bias(s_normed)
        s_scale, s_bias = ss.split([self.dim, self.dim], dim=-1)

        if self._fused_op is not None:
            a = a.contiguous()
            # Write to a separate buffer so callers that use ``a`` as a
            # residual after this op see the original values.
            out = ensure_buffer(buffers, buffer_key,
                                a.shape, a.dtype, a.device)
            if out is None:
                out = torch.empty_like(a)
            return self._fused_op(a, s_scale, s_bias,
                                  out=out, eps=self.eps)

        a = self.a_norm(a)
        if self.mapping.tp_size > 1:
            start = self.mapping.tp_rank * self.dim
            end = (self.mapping.tp_rank + 1) * self.dim
            a = a[:, :, start:end]

        a = F.sigmoid(s_scale) * a + s_bias

        if self.mapping.tp_size > 1:
            a = self.group_comm().all_gather(a, dim=-1)
        return a
