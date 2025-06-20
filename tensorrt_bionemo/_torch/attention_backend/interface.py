# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dataclasses import dataclass
from typing import Generic, Optional, Type, TypeVar, Union

import torch

from tensorrt_bionemo.mapping import Mapping


@dataclass(kw_only=True)
class AttentionMetadata:
    """
    Metadata for multi-head attention layer.
    """
    mapping: Optional[Mapping] = None
    bias_cache: Optional[dict[str, torch.Tensor]] = None


TMetadata = TypeVar("TMetadata", bound=AttentionMetadata)


class PredefinedAttentionBiases(str, enum.Enum):
    """
    Predefined attention mask types

    Attributes:
        TRIANGLE: Use bias for triangular attention.
        PAIRWISE:  Use bias for pairwise attention.
    """

    TRIANGLE = "triangle"
    PAIRWISE = "pairwise"


# May extend to custom attention mask type
AttentionBiases = Union[PredefinedAttentionBiases]


class AttentionBackend(Generic[TMetadata]):
    """
    Base class for attention backends.
    """
    Metadata: Type[TMetadata] = AttentionMetadata

    def __init__(self,
                 layer_idx: int,
                 num_heads: int,
                 head_dim: int,
                 num_kv_heads: Optional[int] = None):
        """
        Initialize the attention backend.
        Args:
            layer_idx (int): The index of the attention layer.
            num_heads (int): The number of attention heads.
            head_dim (int): The dimension of each attention head.
            num_kv_heads (Optional[int]): The number of key-value heads.
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
        biases: Optional[list[torch.Tensor]] = None,
        metadata: TMetadata = None,
        attention_biases: Optional[AttentionBiases] = PredefinedAttentionBiases.
        TRIANGLE,
        **kwargs,
    ) -> torch.Tensor:
        """
        Perform the attention operation.
        Args:
            q (torch.Tensor): The query tensor. Shape [I, s_q, h_q*d]
            k (torch.Tensor): The key tensor. Shape [I, s_kv, h_kv*d]
            v (torch.Tensor): The value tensor. Shape [I, s_kv, h_kv*d]
            biases (Optional[list[torch.Tensor]]): The biases for the attention layer.
            metadata (AttentionMetadata): The metadata for the attention layer.
            attention_biases (Optional[AttentionBiases]): The type of attention biases to use.
            **kwargs: Additional keyword arguments.
        Returns:
            torch.Tensor: The output tensor.
        """
        raise NotImplementedError("Subclasses must implement this method.")
