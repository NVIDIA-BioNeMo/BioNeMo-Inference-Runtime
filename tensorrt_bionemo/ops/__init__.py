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

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from tensorrt_bionemo.ops._loader import load_extension

if TYPE_CHECKING:
    import torch

_C = load_extension()


def generate_deletion_matrix(
    buff: "torch.Tensor",
    offsets: "torch.Tensor",
    lengths: "torch.Tensor",
    output: "torch.Tensor",
    stream: Optional[int] = None,
) -> None:
    """
    Generate deletion matrix.
    
    Args:
        buff: Input buffer tensor (uint8, CUDA, contiguous)
        offsets: Sequence offsets (int32, CUDA, shape: [num_seqs])
        lengths: Sequence lengths (int32, CUDA, shape: [num_seqs])
        output: Pre-allocated output (int32, CUDA, shape: [num_seqs, N])
        stream: Optional CUDA stream pointer
    
    Example:
        >>> import torch
        >>> from tensorrt_bionemo.ops import generate_deletion_matrix
        >>> 
        >>> buff = torch.randint(0, 255, (1000,), dtype=torch.uint8, device="cuda")
        >>> offsets = torch.tensor([0, 100, 300], dtype=torch.int32, device="cuda")
        >>> lengths = torch.tensor([100, 200, 150], dtype=torch.int32, device="cuda")
        >>> output = torch.empty((3, 512), dtype=torch.int32, device="cuda")
        >>> 
        >>> generate_deletion_matrix(buff, offsets, lengths, output)
    """
    _C.generate_deletion_matrix(buff, offsets, lengths, output, stream)
