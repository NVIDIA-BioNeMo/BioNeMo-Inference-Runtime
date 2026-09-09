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

from ._common import SDPAAttentionMetadata, VanillaAttentionMetadata
from .interface import AttentionBackend, AttentionMetadata, AttentionType
from .pairwise_attention import (
    PairwiseAttentionCuTeLeftMask,
    PairwiseAttentionCuTeLeftMaskMetadata,
    PairwiseAttentionLeftMaskKernelConfig,
    SDPAPairwiseAttention,
    VanillaPairwiseAttention,
)
from .triangle_attention import (
    CuEquivAttention,
    CuEquivAttentionMetadata,
    SDPATriangleAttention,
    TriangleAttentionCuTeLeftMask,
    TriangleAttentionCuTeLeftMaskMetadata,
    TriangleAttentionLeftMaskKernelConfig,
    VanillaTriangleAttention,
)
from .utils import (
    PrecomputedPairMasks,
    PrecomputedSingleMasks,
    auto_select_pairwise_attention_backend,
    auto_select_triangle_attention_backend,
    create_attention,
    get_attention_backend,
    precompute_pair_masks,
    precompute_single_masks,
    register_precompute_pair_masks,
    register_precompute_single_masks,
)

__all__ = [
    "AttentionMetadata",
    "AttentionBackend",
    "VanillaTriangleAttention",
    "VanillaPairwiseAttention",
    "VanillaAttentionMetadata",
    "SDPAPairwiseAttention",
    "SDPATriangleAttention",
    "SDPAAttentionMetadata",
    "CuEquivAttention",
    "CuEquivAttentionMetadata",
    "PairwiseAttentionCuTeLeftMask",
    "PairwiseAttentionCuTeLeftMaskMetadata",
    "PairwiseAttentionLeftMaskKernelConfig",
    "TriangleAttentionCuTeLeftMask",
    "TriangleAttentionCuTeLeftMaskMetadata",
    "TriangleAttentionLeftMaskKernelConfig",
    "AttentionType",
    "get_attention_backend",
    "create_attention",
    "auto_select_pairwise_attention_backend",
    "auto_select_triangle_attention_backend",
    "PrecomputedPairMasks",
    "precompute_pair_masks",
    "register_precompute_pair_masks",
    "PrecomputedSingleMasks",
    "precompute_single_masks",
    "register_precompute_single_masks",
]
