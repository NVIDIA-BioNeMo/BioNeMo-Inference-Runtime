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

from typing import Optional

import torch

from tensorrt_bionemo.ops._loader import load_extension

_extension_loaded = False
_handle = None
if not _extension_loaded:
    _extension_loaded = True
    _handle = load_extension()
_C = torch.ops._C


def generate_deletion_matrix(buff: torch.Tensor, offsets: torch.Tensor,
                             lengths: torch.Tensor, N: int) -> None:
    """
    Generate deletion matrix.

    Args:
        buff: Input buffer tensor (uint8, CUDA, contiguous)
        offsets: Sequence offsets (int32, CUDA, shape: [num_seqs])
        lengths: Sequence lengths (int32, CUDA, shape: [num_seqs])
        N: Number of columns in the deletion matrix
    Example:
        >>> import torch
        >>> from tensorrt_bionemo.ops import generate_deletion_matrix
        >>>
        >>> buff = torch.randint(0, 255, (1000,), dtype=torch.uint8, device="cuda")
        >>> offsets = torch.tensor([0, 100, 300], dtype=torch.int32, device="cuda")
        >>> lengths = torch.tensor([100, 200, 150], dtype=torch.int32, device="cuda")
        >>> output = torch.empty((3, 512), dtype=torch.int32, device="cuda")
        >>> N = 512
        >>> generate_deletion_matrix(buff, offsets, lengths, N)
    """
    _C.generate_deletion_matrix(buff, offsets, lengths, N)


def x_x_dual_gemm(X: torch.Tensor,
                  W0: torch.Tensor,
                  W1: torch.Tensor,
                  bias0: Optional[torch.Tensor] = None,
                  bias1: Optional[torch.Tensor] = None,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
  Dual GEMM operation.

  Args:
    X: Input tensor (CUDA, contiguous)
    W0: Weight tensor 0 (CUDA, contiguous)
    W1: Weight tensor 1 (CUDA, contiguous)
    bias0: Bias tensor 0 (CUDA, contiguous)
    bias1: Bias tensor 1 (CUDA, contiguous)
    mask: Mask tensor (CUDA, contiguous)

  Returns:
    Output tensor (CUDA, contiguous): X@W0 * sigmoid(X@W1) * mask
  """
    return _C.x_x_dual_gemm(X, W0, W1, bias0, bias1, mask)


def x0_x1_dual_gemm(X0: torch.Tensor,
                    X1: torch.Tensor,
                    W0: torch.Tensor,
                    W1: torch.Tensor,
                    bias0: Optional[torch.Tensor] = None,
                    bias1: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Dual GEMM operation.
    Args:
      X0: Input tensor 0 (CUDA, contiguous)
      X1: Input tensor 1 (CUDA, contiguous)
      W0: Weight tensor 0 (CUDA, contiguous)
      W1: Weight tensor 1 (CUDA, contiguous)
      bias0: Bias tensor 0 (CUDA, contiguous)
      bias1: Bias tensor 1 (CUDA, contiguous)

    Returns:
      Output tensor (CUDA, contiguous): sigmoid(X0@W0) * (X1@W1)
    """
    return _C.x0_x1_dual_gemm(X0, X1, W0, W1, bias0, bias1)


__all__ = [
    "generate_deletion_matrix",
    "x_x_dual_gemm",
    "x0_x1_dual_gemm",
]
