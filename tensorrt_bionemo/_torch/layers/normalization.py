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

from tensorrt_bionemo._torch.custom_ops import get_adaln_layernorm_sigmoid_op
from tensorrt_bionemo._torch.layers.linear import (Linear, WeightMode,
                                                   WeightsLoadingConfig)
from tensorrt_bionemo.runtime.buffers import PreallocatedBuffers, ensure_buffer


class AdaLN(nn.Module):

    def __init__(self,
                 dim: int,
                 dim_single_cond: int,
                 eps: float = 1e-5,
                 dtype: torch.dtype = None,
                 skip_create_weights: bool = False):
        """Adaptive LayerNorm with a sigmoid-gated affine.

        Uses the fused CuTe DSL kernel, falling back to the inline torch
        path when the kernel is unavailable. Kernel init / forward
        failures propagate to the caller — no silent fallback.
        """
        super().__init__()
        self.dim = dim
        self.dim_single_cond = dim_single_cond
        self.eps = eps

        self.a_norm = nn.LayerNorm(self.dim,
                                   dtype=dtype,
                                   eps=eps,
                                   elementwise_affine=False,
                                   bias=False)
        self.s_norm = nn.LayerNorm(self.dim_single_cond,
                                   dtype=dtype,
                                   eps=eps,
                                   bias=False)
        # Fused s_scale and s_bias projection. The upstream s_bias
        # projection has no bias term, so the s_bias half of the fused
        # bias vector is kept at zero.
        self.fused_s_scale_s_bias = Linear(
            self.dim_single_cond,
            2 * self.dim,
            bias=True,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_KV_LINEAR))

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
            out = ensure_buffer(buffers, buffer_key, a.shape, a.dtype,
                                a.device)
            if out is None:
                out = torch.empty_like(a)
            return self._fused_op(a, s_scale, s_bias, out=out, eps=self.eps)

        a = self.a_norm(a)
        return F.sigmoid(s_scale) * a + s_bias
