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

from typing import Optional, Type

from .cuequiv import CuEquivAttention
from .interface import AttentionBackend, AttentionType
from .sdpa import SDPAPairwiseAttention
from .trifast import TrifastAttention
from .vanilla import VanillaPairwiseAttention, VanillaTriangleAttention


def get_attention_backend(
    backend_name: str,
    attention_type: AttentionType = AttentionType.TRIANGLE
) -> Type[AttentionBackend]:
    """Get the attention backend class based on the backend name and attention type."""
    if attention_type == AttentionType.TRIANGLE:
        if backend_name == "VANILLA":
            return VanillaTriangleAttention
        elif backend_name == "TRIFAST":
            return TrifastAttention
        elif backend_name == "CUEQUIV":
            return CuEquivAttention
    elif attention_type == AttentionType.PAIRWISE:
        if backend_name == "VANILLA":
            return VanillaPairwiseAttention
        elif backend_name == "SDPA":
            return SDPAPairwiseAttention
        else:
            raise ValueError(f"Invalid backend name: {backend_name}")
    else:
        raise ValueError(f"Invalid backend name: {backend_name}")


def create_attention(
    backend_name: str,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: Optional[int] = None,
    attention_type: AttentionType = AttentionType.TRIANGLE
) -> AttentionBackend:
    """Create an attention backend based on the backend name and attention type."""
    attn_cls = get_attention_backend(backend_name, attention_type)
    return attn_cls(layer_idx, num_heads, head_dim, num_kv_heads)
