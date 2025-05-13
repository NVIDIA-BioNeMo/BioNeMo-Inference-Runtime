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

from .interface import AttentionBackend
from .trifast import TrifastAttention
from .vanilla import VanillaAttention


def get_attention_backend(backend_name: str) -> Type[AttentionBackend]:
    """Get the attention backend class based on the backend name."""
    if backend_name == "VANILLA":
        return VanillaAttention
    elif backend_name == "TRIFAST":
        return TrifastAttention
    else:
        raise ValueError(f"Invalid backend name: {backend_name}")


def create_attention(backend_name: str,
                     layer_idx: int,
                     num_heads: int,
                     head_dim: int,
                     num_kv_heads: Optional[int] = None) -> AttentionBackend:
    """Create an attention backend based on the backend name."""
    attn_cls = get_attention_backend(backend_name)
    return attn_cls(layer_idx, num_heads, head_dim, num_kv_heads)
