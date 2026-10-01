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
"""Fused attention epilogue.

For the flattened pair row ``m`` and ``k = h*D + d``, one kernel computes

    out[m, n] = residual[m, n] + sum_k bf16(o[m, k] * bf16(sigmoid(g[m, k]))) * Wo[n, k]

in place of the gate, the output projection and the residual add, keeping
their bf16 rounding points. ``out`` may be ``residual`` itself. An optional
output gate ``y`` — the diffusion transformer's adaLN-Zero gate logits —
scales the rounded update by ``bf16(sigmoid(y))`` inside the residual add,
which then rounds once.

Operand views are permutations, never copies: ``o`` stays in the attention
core's heads-inner ``[..., J, H, D]`` layout, and the pair tensors may be the
ending node's transposed views.

A source-free build carries no private kernel adapter; everything re-exported
here must keep working with it absent.
"""

from .ops import SLAB, STREAMED_CHANNELS, AttnEpilogue, get_attn_epilogue_op

__all__ = ["SLAB", "STREAMED_CHANNELS", "AttnEpilogue", "get_attn_epilogue_op"]
