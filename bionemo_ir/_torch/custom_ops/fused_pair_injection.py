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
"""Fused sequence-to-pair injection with masking.

Replaces the pair-write sequence at the head of every RFProteína pair update::

    sum:       pair = pair + row.unsqueeze(1)
               pair.add_(column.unsqueeze(2))
               pair.mul_(mask)

    hadamard:  pair = pair + row.unsqueeze(1) * column.unsqueeze(2)
               pair.mul_(mask)

with a single Triton kernel. Valid finite values are bit-identical to the chain,
and invalid pairs become exact zero. Unsupported operands use the eager chain.

Usage sites:
    - ``PairUpdateBlock`` (single-to-pair injection)
"""

import torch
import torch.nn as nn

from bionemo_ir.dsl_kernels.triton.fused_pair_injection import FusedPairInjection as _TritonPairInjection

#: Dtypes the Triton kernel pre-compiles for.
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


class PairInjection(nn.Module):
    """Write two per-token projections into a pair representation, then mask.

    Args:
        pair_dim: Pair feature width, ``C``.
        pair_write: ``"sum"`` adds ``row`` and ``column`` into the pair
            separately; ``"hadamard"`` adds their product.
        dtype: Compute dtype (default ``torch.bfloat16``).
    """

    def __init__(
        self,
        pair_dim: int,
        pair_write: str = "sum",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if pair_write not in ("sum", "hadamard"):
            raise ValueError(f"pair_write must be 'sum' or 'hadamard', got {pair_write!r}")
        self.pair_write = pair_write
        self._fused_kernel: _TritonPairInjection | None = None
        try:
            self._fused_kernel = _TritonPairInjection(
                pair_dim=pair_dim,
                hadamard=pair_write == "hadamard",
                dtype=dtype,
            )
        except Exception:
            pass

    def forward(
        self,
        pair: torch.Tensor,
        row: torch.Tensor,
        column: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Inject ``row`` and ``column`` into ``pair`` and re-apply ``mask``.

        The result is always storage this call owns, on both the fused and eager
        paths, so a caller may accumulate later pair-sized residuals into it in
        place without mutating its own input.

        Args:
            pair: ``[B, N, N, C]`` pair representation.
            row: ``[B, N, C]`` projection indexed by the second token axis.
            column: ``[B, N, C]`` projection indexed by the first token axis.
            mask: ``[B, N, N, 1]`` pair mask in the operand dtype, holding 0 or 1.

        Returns:
            ``[B, N, N, C]`` injected pair representation.
        """
        use_fused = self._fused_kernel is not None and pair.device.type == "cuda" and pair.dtype in _SUPPORTED_DTYPES
        if use_fused:
            return self._fused_kernel(pair.contiguous(), row, column, mask.contiguous())

        # The first add allocates, which is what gives the caller owned storage.
        if self.pair_write == "sum":
            injected = pair + row.unsqueeze(1)
            injected.add_(column.unsqueeze(2))
        else:
            injected = pair + row.unsqueeze(1) * column.unsqueeze(2)
        return injected.masked_fill_(~mask.bool(), 0)
