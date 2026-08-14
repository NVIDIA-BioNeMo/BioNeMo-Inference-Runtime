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

import enum
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(kw_only=True)
class AttentionMetadata:
    """
    Metadata for multi-head attention layer.
    """

    # Bias caching for diffusion transformer layers
    bias_cache: dict[str, torch.Tensor] | None = None

    # Function to convert query to keys for sequence-local atom attention.
    # OSS-equivalent zero-pad path: a gather with a sentinel zero row at OOB
    # columns. Pre-bound to ``gather_indices``, ``W=n_query``, ``H=n_key`` so
    # callers only pass the input tensor. Used by both atom attention and
    # ``convert_pair_atom_to_blocks``.
    query_to_keys: Callable | None = None


class AttentionType(str, enum.Enum):
    """
    Predefined attention mask types

    Attributes:
        TRIANGLE: Use bias for triangular attention.
        PAIRWISE:  Use bias for pairwise attention.
    """

    TRIANGLE = "triangle"
    PAIRWISE = "pairwise"


class AttentionBackend[TMetadata: AttentionMetadata]:
    """
    Base class for attention backends.
    """

    Metadata: type[TMetadata] = AttentionMetadata

    def __init__(self, layer_idx: int, num_heads: int, head_dim: int, num_kv_heads: int | None = None):
        """
        Initialize the attention backend.
        Args:
            layer_idx (int): The index of the attention layer.
            num_heads (int): The number of attention heads.
            head_dim (int): The dimension of each attention head.
            num_kv_heads (int | None): The number of key-value heads.
        """
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        metadata: TMetadata | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Perform the attention operation.
        Args:
            q (torch.Tensor): The query tensor. Shape [I, s_q, h_q*d]
            k (torch.Tensor): The key tensor. Shape [I, s_kv, h_kv*d]
            v (torch.Tensor): The value tensor. Shape [I, s_kv, h_kv*d]
            biases (list[torch.Tensor] | None): The biases for the attention layer.
            metadata (TMetadata | None): The metadata for the attention layer.
            **kwargs: Additional keyword arguments.
        Returns:
            torch.Tensor: The output tensor.
        """
        raise NotImplementedError("Subclasses must implement this method.")
