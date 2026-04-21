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
output.  When the fused kernel is not available (e.g. padding disabled,
non-contiguous input), the vanilla PyTorch path is used automatically.

Usage sites:
    - ``AttentionPairBias`` (pair bias path in DiffusionTransformer layers)
    - ``TriangleAttentionNode`` (pair bias path in Pairformer / Evoformer)
"""

from typing import Optional

import torch
import torch.nn as nn

from tensorrt_bionemo.dsl_kernels.triton.fused_ln_proj_moveaxis_pad import \
    FusedLNProjMoveaxisPad as _TritonFusedLNProj
from tensorrt_bionemo.dsl_kernels.triton.moveaxis_pad import MoveaxisPad


class LNProjMoveaxisPad(nn.Module):
    """LayerNorm + Linear projection + moveaxis(-1,-3) + optional pad.

    Transparently dispatches between:
      - **Fused Triton kernel**: single-pass over ``[*, I, J, D]`` input,
        writing directly to ``[*, H, I, J_padded]``.  Used when
        ``pad_multiple >= 0`` and the input is contiguous.
      - **Vanilla PyTorch**: ``LayerNorm → Linear → moveaxis_pad``.
        Used as fallback.

    Args:
        D: pair feature dimension (LayerNorm normalised dim).
        H: number of attention heads (Linear output dim).
        dtype: compute dtype (default ``torch.bfloat16``).
    """

    def __init__(self, D: int, H: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self._moveaxis_pad = MoveaxisPad(H=H, dtype=dtype)
        self._fused_kernel: Optional[_TritonFusedLNProj] = None
        try:
            self._fused_kernel = _TritonFusedLNProj(D=D, H=H, dtype=dtype)
        except Exception:
            pass

    def forward(
        self,
        z: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: Optional[torch.Tensor],
        proj_weight: torch.Tensor,
        pad_multiple: int = -1,
        proj_z: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Compute LN + Linear + moveaxis + pad.

        Args:
            z: input pair tensor ``[*, I, J, D]``.
            ln_weight: LayerNorm weight ``[D]``.
            ln_bias: LayerNorm bias ``[D]`` (required for fused path).
            proj_weight: Linear projection weight ``[H, D]``.
            pad_multiple: pad J to next multiple (``< 0`` = no padding).
            proj_z: ``nn.Sequential(LayerNorm, Linear)`` or similar module
                used by the vanilla path.  When ``None``, the vanilla path
                applies ``F.layer_norm`` + ``F.linear`` using the weight
                tensors directly.

        Returns:
            ``[*, H, I, J_padded]`` contiguous tensor.
        """
        use_fused = (self._fused_kernel is not None and z.is_contiguous()
                     and ln_bias is not None and pad_multiple >= 0)

        if use_fused:
            return self._fused_kernel(z,
                                      ln_weight,
                                      ln_bias,
                                      proj_weight,
                                      multiple=pad_multiple)

        # Vanilla path
        if proj_z is not None:
            pair_bias = proj_z(z)
        else:
            pair_bias = torch.nn.functional.layer_norm(z, [z.shape[-1]],
                                                       ln_weight, ln_bias)
            pair_bias = torch.nn.functional.linear(pair_bias, proj_weight)
        return self._moveaxis_pad(pair_bias, multiple=pad_multiple)
