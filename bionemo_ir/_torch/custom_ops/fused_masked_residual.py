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
"""Fused masked residual.

Computes a single- or pair-representation update with one Triton kernel::

    value = value.add_(update).mul_(mask)

It is bit-identical for valid finite values, writes exact zero at invalid
positions, and accepts strided token axes without repacking.
"""

from __future__ import annotations

import torch

from bionemo_ir.dsl_kernels.triton.fused_masked_residual import (
    fused_masked_residual as _triton_masked_residual,
)
from bionemo_ir.dsl_kernels.triton.fused_masked_residual import (
    supports_masked_residual as _triton_supports,
)


def fused_masked_residual(pair: torch.Tensor, update: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    """Add ``update`` into ``pair`` and re-apply ``mask``, in place.

    Args:
        pair: Contiguous ``[B, N, C]`` or ``[B, N, N, C]`` destination.
        update: Same shape as ``pair``. Token axes may be strided.
        mask: Contiguous ``[B, N, 1]`` or ``[B, N, N, 1]`` mask.

    Returns:
        ``pair``, updated, or ``None`` when the kernel does not cover the
        operands so the caller can use its eager fallback.
    """
    if not _triton_supports(pair, update, mask):
        return None
    return _triton_masked_residual(pair, update, mask)
