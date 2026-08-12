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


import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_bionemo._torch.custom_ops import get_adaln_layernorm_sigmoid_op
from tensorrt_bionemo._torch.layers.linear import Linear, WeightMode, WeightsLoadingConfig
from tensorrt_bionemo.runtime.buffers import PreallocatedBuffers, ensure_buffer


class AdaLN(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_single_cond: int,
        eps: float = 1e-5,
        dtype: torch.dtype = None,
        skip_create_weights: bool = False,
    ):
        """Adaptive LayerNorm with a sigmoid-gated affine.

        ``get_adaln_layernorm_sigmoid_op`` returns the fused CuTe DSL kernel
        where it is supported and a signature-compatible torch fallback
        otherwise, so this layer has one code path either way.
        """
        super().__init__()
        self.dim = dim
        self.dim_single_cond = dim_single_cond
        self.eps = eps

        self.a_norm = nn.LayerNorm(self.dim, dtype=dtype, eps=eps, elementwise_affine=False, bias=False)
        self.s_norm = nn.LayerNorm(self.dim_single_cond, dtype=dtype, eps=eps, bias=False)
        # Fused s_scale and s_bias projection. The upstream s_bias
        # projection has no bias term, so the s_bias half of the fused
        # bias vector is kept at zero.
        self.fused_s_scale_s_bias = Linear(
            self.dim_single_cond,
            2 * self.dim,
            bias=True,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_KV_LINEAR),
        )

        self._fused_op = get_adaln_layernorm_sigmoid_op(
            dtype if dtype is not None else torch.float32,
            N=self.dim,
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        buffers: PreallocatedBuffers | None = None,
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
        s_normed = self.s_norm(s)
        ss = self.fused_s_scale_s_bias(s_normed)
        s_scale, s_bias = ss.split([self.dim, self.dim], dim=-1)

        # ``_fused_op`` is never None in production -- the dispatcher returns a
        # torch fallback rather than nothing. Tests clear it to force this
        # inline path as an independent reference, so the guard stays.
        if self._fused_op is not None:
            a = a.contiguous()
            # Write to a separate buffer so callers that use ``a`` as a
            # residual after this op see the original values.
            out = ensure_buffer(buffers, buffer_key, a.shape, a.dtype, a.device)
            if out is None:
                out = torch.empty_like(a)
            return self._fused_op(a, s_scale, s_bias, out=out, eps=self.eps)

        a = self.a_norm(a)
        return F.sigmoid(s_scale) * a + s_bias


class HighPrecisionLayerNorm(nn.Module):
    """LayerNorm in fp32, result cast to ``out_dtype``.

    Params keep ``nn.LayerNorm`` names so ``load_state_dict`` still works.
    """

    def __init__(
        self,
        normalized_shape: int | list[int] | tuple[int, ...] | torch.Size,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        out_dtype: torch.dtype = torch.bfloat16,
    ):
        del dtype  # params are always fp32; ``out_dtype`` controls the cast
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.out_dtype = out_dtype
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=torch.float32))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=device, dtype=torch.float32))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    @classmethod
    def from_layernorm(cls, ln: nn.LayerNorm, out_dtype: torch.dtype) -> "HighPrecisionLayerNorm":
        """Clone an ``nn.LayerNorm`` with weights stored in fp32."""
        has_affine = ln.weight is not None
        has_bias = ln.bias is not None
        module = cls(
            ln.normalized_shape,
            eps=ln.eps,
            elementwise_affine=has_affine,
            bias=has_bias,
            device=ln.weight.device if has_affine else None,
            out_dtype=out_dtype,
        )
        if has_affine:
            module.weight.data.copy_(ln.weight.data.float())
            if has_bias:
                module.bias.data.copy_(ln.bias.data.float())
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.layer_norm(x.float(), self.normalized_shape, self.weight, self.bias, self.eps)
        return out.to(dtype=self.out_dtype)


def replace_with_high_precision_layernorm(
    module: nn.Module,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    skip_types: tuple[type, ...] = (),
) -> int:
    """Replace ``nn.LayerNorm`` under ``module`` with :class:`HighPrecisionLayerNorm`.

    Subtrees in ``skip_types`` are left alone. Returns replacement count;
    no-op when ``out_dtype`` is fp32.
    """
    if out_dtype == torch.float32:
        return 0
    n = 0
    for name, child in list(module.named_children()):
        if skip_types and isinstance(child, skip_types):
            continue
        if type(child) is nn.LayerNorm:
            setattr(module, name, HighPrecisionLayerNorm.from_layernorm(child, out_dtype=out_dtype))
            n += 1
        else:
            n += replace_with_high_precision_layernorm(child, out_dtype=out_dtype, skip_types=skip_types)
    return n
