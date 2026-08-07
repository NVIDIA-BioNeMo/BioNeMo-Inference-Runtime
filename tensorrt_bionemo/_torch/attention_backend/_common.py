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

import torch
from einops import rearrange

from .interface import AttentionMetadata


class VanillaAttentionMetadata(AttentionMetadata):
    pass


class SDPAAttentionMetadata(AttentionMetadata):
    pass


def _reshape_heads(tensor: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Reshape ``[..., seq, H*D]`` to ``[..., H, seq, D]``."""
    batch_dims = " ".join([f"b_{i}" for i in range(tensor.ndim - 2)])
    return rearrange(tensor, f"{batch_dims} j (h d) -> {batch_dims} h j d", h=num_heads, d=head_dim)


def prep_qkv_for_vanilla(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare Q/K/V for explicit QK matmul attention."""
    q = _reshape_heads(q, num_heads, head_dim)
    k = _reshape_heads(k, num_heads, head_dim).transpose(-2, -1)
    v = _reshape_heads(v, num_heads, head_dim)
    return q, k, v


def prep_qkv_for_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare Q/K/V for PyTorch scaled-dot-product attention."""
    q = _reshape_heads(q, num_heads, head_dim)
    k = _reshape_heads(k, num_heads, head_dim)
    v = _reshape_heads(v, num_heads, head_dim)
    return q, k, v
