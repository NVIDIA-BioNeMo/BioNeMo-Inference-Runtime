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
"""Reusable cuDNN operation graphs for eager PyTorch modules."""

from .cache import CudnnGraphCache
from .ops import (
    cudnn_add_add_mask,
    cudnn_add_mask,
    cudnn_linear_mask,
    cudnn_linear_mask_residual,
    cudnn_linear_relu,
    cudnn_linear_residual,
    cudnn_scale_shift_mask,
    prepare_cudnn_linear_mask,
    prepare_cudnn_linear_relu,
)
from .runtime import CudnnGraphModule, can_use_cudnn_graph

__all__ = [
    "CudnnGraphCache",
    "CudnnGraphModule",
    "can_use_cudnn_graph",
    "cudnn_add_add_mask",
    "cudnn_add_mask",
    "cudnn_linear_mask",
    "cudnn_linear_mask_residual",
    "cudnn_linear_relu",
    "cudnn_linear_residual",
    "cudnn_scale_shift_mask",
    "prepare_cudnn_linear_mask",
    "prepare_cudnn_linear_relu",
]
