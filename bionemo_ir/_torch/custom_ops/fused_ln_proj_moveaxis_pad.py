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
"""Fused LayerNorm + Linear projection + moveaxis(-1,-3) + pad.

Replaces the common pair-bias preparation sequence::

    z_normed = layer_norm(z)         # [*, I, J, D]
    proj     = linear(z_normed)      # [*, I, J, H]
    out      = moveaxis_pad(proj)    # [*, H, I, J_padded]

with a single Triton kernel that reads ``z`` once and writes the transposed
output. When the fused kernel is not available (e.g. padding disabled or a
non-contiguous input), a split normalization, projection, and layout path is
used automatically.

Usage sites:
    - ``AttentionPairBias`` (pair bias path in DiffusionTransformer layers)
    - ``TriangleAttentionNode`` (pair bias path in Pairformer / Evoformer)
"""

import torch
import torch.nn as nn

from bionemo_ir.dsl_kernels.triton.fused_layer_norm_transpose import layer_norm_transpose
from bionemo_ir.dsl_kernels.triton.fused_ln_proj_moveaxis_pad import FusedLNProjMoveaxisPad as _TritonFusedLNProj
from bionemo_ir.dsl_kernels.triton.moveaxis_pad import MoveaxisPad


class LNProjMoveaxisPad(nn.Module):
    """LayerNorm/RMSNorm + Linear projection + moveaxis(-1,-3) + optional pad.

    Transparently dispatches between:
      - **Fused Triton kernel**: single-pass over ``[*, I, J, D]`` input,
        writing directly to ``[*, H, I, J_padded]``.  Used when
        ``pad_multiple >= 0`` and the input is contiguous.
      - **Split fallback**: single-pass Triton LayerNorm/RMSNorm, cuBLAS
        projection, then fused moveaxis + pad. No ATen normalization fallback
        is used.

    Args:
        D: pair feature dimension (normalised dim).
        H: number of attention heads (Linear output dim).
        dtype: compute dtype (default ``torch.bfloat16``).
        rms_norm: use RMSNorm (no mean subtraction, no bias) instead of
            LayerNorm.
        eps: epsilon added to the normalization variance.
    """

    def __init__(
        self,
        D: int,
        H: int,
        dtype: torch.dtype = torch.bfloat16,
        rms_norm: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self._rms_norm = rms_norm
        self._eps = eps
        self._moveaxis_pad = MoveaxisPad(H=H, dtype=dtype)
        self._fused_kernel: _TritonFusedLNProj | None = None
        try:
            self._fused_kernel = _TritonFusedLNProj(D=D, H=H, dtype=dtype, rms_norm=rms_norm, eps=eps)
        except Exception:
            pass

    def forward(
        self,
        z: torch.Tensor,
        ln_weight: torch.Tensor | None,
        ln_bias: torch.Tensor | None,
        proj_weight: torch.Tensor,
        pad_multiple: int = -1,
        proj_z: nn.Module | None = None,
    ) -> torch.Tensor:
        """Compute LN/RMS + Linear + moveaxis + pad.

        Args:
            z: input pair tensor ``[*, I, J, D]``.
            ln_weight: Optional LayerNorm/RMSNorm weight ``[D]``.
            ln_bias: Optional LayerNorm bias ``[D]``. ``None`` for RMSNorm
                or affine-free LayerNorm.
            proj_weight: Linear projection weight ``[H, D]``.
            pad_multiple: pad J to next multiple (``< 0`` = no padding).
            proj_z: ``nn.Sequential(Norm, Linear)`` or similar module
                used to identify optional normalization and projection bias,
                or to run a projection-only path.

        Returns:
            ``[*, H, I, J_padded]`` contiguous tensor.
        """
        projection = proj_z[-1] if isinstance(proj_z, nn.Sequential) else proj_z
        projection_bias = None if projection is None else getattr(projection, "bias", None)
        # LayerNorm fused path needs the bias; RMSNorm fused path does not.
        has_norm_inputs = ln_weight is not None and (self._rms_norm or ln_bias is not None)
        use_fused = (
            self._fused_kernel is not None
            and z.is_contiguous()
            and has_norm_inputs
            and projection_bias is None
            and pad_multiple >= 0
        )

        if use_fused:
            return self._fused_kernel(z, ln_weight, ln_bias, proj_weight, multiple=pad_multiple)

        # Split fallback: use the existing single-pass normalization kernels,
        # then let cuBLAS handle the projection before the fused layout move.
        norm_module = proj_z[0] if isinstance(proj_z, nn.Sequential) and len(proj_z) > 1 else None
        has_norm = ln_weight is not None or ln_bias is not None or isinstance(norm_module, (nn.LayerNorm, nn.RMSNorm))
        if has_norm or proj_z is None:
            normalized = layer_norm_transpose(
                z.reshape(-1, z.shape[-1]),
                ln_weight,
                ln_bias,
                eps=self._eps,
                elementwise_affine=ln_weight is not None or ln_bias is not None,
                rms_norm=self._rms_norm,
                layout="nd->nd",  # codespell:ignore nd
            ).view_as(z)
            pair_bias = torch.nn.functional.linear(normalized, proj_weight, projection_bias)
        else:
            pair_bias = proj_z(z)
        return self._moveaxis_pad(pair_bias, multiple=pad_multiple)
