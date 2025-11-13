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

from typing import Optional

import torch
import torch.nn as nn

from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo.config import Mapping


class DistogramModule(nn.Module):
    """Distogram Module."""

    def __init__(self,
                 token_z: int,
                 num_bins: int,
                 num_distograms: int = 1,
                 version: str = "v1",
                 dtype: torch.dtype = torch.float32,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False) -> None:
        """Initialize the distogram module.

        Args:
            token_z : int
                The token pairwise embedding size.
            num_bins : int
                The number of bins.
        """
        super().__init__()
        self.version = version
        self.num_bins = num_bins
        self.num_distograms = num_distograms
        self.distogram = Linear(token_z,
                                num_distograms * num_bins,
                                dtype=dtype,
                                mapping=mapping,
                                tensor_parallel_mode=TensorParallelMode.COLUMN,
                                gather_output=True,
                                skip_create_weights=skip_create_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass.

        Args:
            z : torch.Tensor
                The pairwise embeddings

        Returns:
            torch.Tensor: The predicted distogram.

        """
        z = z + z.transpose(1, 2)
        if self.version == "v1":
            return self.distogram(z)

        return self.distogram(z).reshape(z.shape[0], z.shape[1], z.shape[2],
                                         self.num_distograms, self.num_bins)
