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

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from bionemo_ir._torch.custom_ops import get_adaln_layernorm_sigmoid_op
from bionemo_ir._torch.graph_optimization import rewrite_modules
from bionemo_ir._torch.layers.linear import Linear, WeightMode, WeightsLoadingConfig
from bionemo_ir.dsl_kernels.triton.fused_layer_norm_transpose import layer_norm_transpose
from bionemo_ir.runtime.buffers import PreallocatedBuffers, ensure_buffer

AdaLNNormType = Literal["layer_norm", "rms_norm"]


class AdaLN(nn.Module):
    """Adaptive normalization with a sigmoid-gated affine from a condition.

    LayerNorm and RMSNorm share the same fused scale/bias projection::

        s' = NormCond(s)
        [γ, β] = Linear(s')
        out = sigmoid(γ) * Norm(a) + β

    For either norm type, ``get_adaln_layernorm_sigmoid_op`` returns the
    matching fused CuTe DSL kernel where it is supported and a
    signature-compatible torch fallback otherwise.

    ``NormCond`` is scale-only by default, following AF3. Set
    ``cond_norm_bias=True`` for checkpoints trained with a full LayerNorm
    in the conditioning path.
    """

    def __init__(
        self,
        dim: int,
        dim_single_cond: int,
        eps: float = 1e-5,
        dtype: torch.dtype = None,
        skip_create_weights: bool = False,
        norm_type: AdaLNNormType = "layer_norm",
        cond_norm_bias: bool = False,
    ):
        super().__init__()
        if norm_type not in ("layer_norm", "rms_norm"):
            raise ValueError(f"Unsupported AdaLN norm_type={norm_type!r}; expected 'layer_norm' or 'rms_norm'")
        if cond_norm_bias and norm_type != "layer_norm":
            raise ValueError(
                f"AdaLN(cond_norm_bias=True) requires norm_type='layer_norm', got {norm_type!r} (RMSNorm has no bias)"
            )

        self.dim = dim
        self.dim_single_cond = dim_single_cond
        self.eps = eps
        self.norm_type = norm_type

        if norm_type == "layer_norm":
            self.a_norm = nn.LayerNorm(self.dim, dtype=dtype, eps=eps, elementwise_affine=False, bias=False)
            self.s_norm = nn.LayerNorm(self.dim_single_cond, dtype=dtype, eps=eps, bias=cond_norm_bias)
        else:
            self.a_norm = nn.RMSNorm(dim, eps=eps, elementwise_affine=False, dtype=dtype)
            self.s_norm = nn.RMSNorm(dim_single_cond, eps=eps, elementwise_affine=True, dtype=dtype)

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
            rms_norm=norm_type == "rms_norm",
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        buffers: PreallocatedBuffers | None = None,
        buffer_key: str = "adaln_out",
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            a: [B, I, d]
            s: [B, I, d_cond]
            buffers: optional preallocated buffer dict for the kernel output.
            buffer_key: key into ``buffers`` for the AdaLN output tensor.
            mask: optional binary mask ``[B, I]`` applied after the affine.
                Broadcasts across a sample axis when ``a`` is ``[B, S, I, d]``.

        Returns:
            a: [B, I, d]
        """
        s_normed = self.s_norm(s)
        ss = self.fused_s_scale_s_bias(s_normed)
        s_scale, s_bias = ss.split([self.dim, self.dim], dim=-1)

        # The dispatcher returns a torch fallback when no payload is available.
        # Tests also clear this to force the inline path as an independent
        # reference.
        if self._fused_op is not None:
            a = a.contiguous()
            # Write to a separate buffer so callers that use ``a`` as a
            # residual after this op see the original values.
            out = ensure_buffer(buffers, buffer_key, a.shape, a.dtype, a.device)
            if out is None:
                out = torch.empty_like(a)
            out = self._fused_op(a, s_scale, s_bias, out=out, eps=self.eps)
            return self._maybe_mask(out, mask)

        a = self.a_norm(a)
        a = F.sigmoid(s_scale) * a + s_bias
        return self._maybe_mask(a, mask)

    @staticmethod
    def _maybe_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return x
        # ``[B, I]`` must expand over a sample axis of ``[B, S, I, D]``.
        while mask.ndim < x.ndim - 1:
            mask = mask.unsqueeze(-2)
        return x * mask.unsqueeze(-1).to(x.dtype)


class FusedLayerNorm(nn.LayerNorm):
    """``nn.LayerNorm`` backed by BioIR's single-pass Triton kernel.

    The subclass preserves parameter names and LayerNorm attributes, so state
    dictionaries and parents that directly read ``weight`` or ``bias`` remain
    compatible. Every forward goes through the kernel; there is no ATen
    fallback.
    """

    @classmethod
    def from_layernorm(cls, layer_norm: nn.LayerNorm) -> nn.LayerNorm:
        """Create a fused replacement that shares ``layer_norm`` parameters.

        Multi-axis norms stay as eager ``nn.LayerNorm``: the kernel only
        normalizes the last dimension. Missing weight or bias is replaced by a
        non-persistent ones or zeros buffer so ``forward`` does not allocate.
        """
        if len(layer_norm.normalized_shape) != 1:
            return layer_norm
        reference = layer_norm.weight if layer_norm.weight is not None else layer_norm.bias
        device = None if reference is None else reference.device
        dtype = None if reference is None else reference.dtype
        module = cls(
            layer_norm.normalized_shape,
            eps=layer_norm.eps,
            elementwise_affine=layer_norm.weight is not None,
            bias=layer_norm.bias is not None,
            device=device,
            dtype=dtype,
        )
        if layer_norm.weight is not None:
            module.weight = layer_norm.weight
        if layer_norm.bias is not None:
            module.bias = layer_norm.bias
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = layer_norm_transpose(
            x.reshape(-1, self.normalized_shape[0]),
            self.weight,
            self.bias,
            eps=self.eps,
            elementwise_affine=self.weight is not None or self.bias is not None,
            layout="nd->nd",  # codespell:ignore nd
        )
        return out.view(x.shape)


def replace_with_fused_layernorm(
    module: nn.Module,
    *,
    skip_types: tuple[type[nn.Module], ...] = (),
) -> int:
    """Replace plain ``nn.LayerNorm`` descendants with :class:`FusedLayerNorm`."""
    return rewrite_modules(
        module,
        nn.LayerNorm,
        FusedLayerNorm.from_layernorm,
        skip_types=skip_types,
    )


class FusedRMSNorm(nn.RMSNorm):
    """``nn.RMSNorm`` backed by BioIR's inference-only Triton kernel.

    The subclass preserves parameter names and RMSNorm attributes, so state
    dictionaries and parents that directly read ``weight`` remain compatible.
    Every forward goes through the kernel; there is no ATen fallback.
    """

    @classmethod
    def from_rmsnorm(cls, rms_norm: nn.RMSNorm) -> nn.RMSNorm:
        """Create a fused replacement that shares ``rms_norm`` parameters.

        Multi-axis norms stay as eager ``nn.RMSNorm``: the kernel only
        normalizes the last dimension.
        """
        if len(rms_norm.normalized_shape) != 1:
            return rms_norm
        reference = rms_norm.weight
        module = cls(
            rms_norm.normalized_shape,
            eps=rms_norm.eps,
            elementwise_affine=reference is not None,
            device=None if reference is None else reference.device,
            dtype=None if reference is None else reference.dtype,
        )
        if reference is not None:
            module.weight = reference
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(x.dtype).eps if self.eps is None else self.eps
        out = layer_norm_transpose(
            x.reshape(-1, self.normalized_shape[0]),
            self.weight,
            None,
            eps=eps,
            elementwise_affine=self.weight is not None,
            rms_norm=True,
            layout="nd->nd",  # codespell:ignore nd
        )
        return out.view(x.shape)


def replace_with_fused_rmsnorm(
    module: nn.Module,
    *,
    skip_types: tuple[type[nn.Module], ...] = (),
) -> int:
    """Replace plain ``nn.RMSNorm`` descendants with :class:`FusedRMSNorm`."""
    return rewrite_modules(
        module,
        nn.RMSNorm,
        FusedRMSNorm.from_rmsnorm,
        skip_types=skip_types,
    )


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
        trailing_shape = tuple(x.shape[-len(self.normalized_shape) :])
        if trailing_shape != self.normalized_shape:
            raise ValueError(f"expected trailing shape {self.normalized_shape}, got {trailing_shape}")
        normalized_size = math.prod(self.normalized_shape)
        out = layer_norm_transpose(
            x.reshape(-1, normalized_size),
            None if self.weight is None else self.weight.reshape(-1),
            None if self.bias is None else self.bias.reshape(-1),
            eps=self.eps,
            elementwise_affine=self.elementwise_affine,
            layout="nd->nd",  # codespell:ignore nd
            out_dtype=self.out_dtype,
        )
        return out.view(x.shape)


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
    return rewrite_modules(
        module,
        nn.LayerNorm,
        lambda layer_norm: HighPrecisionLayerNorm.from_layernorm(layer_norm, out_dtype=out_dtype),
        skip_types=skip_types,
    )
