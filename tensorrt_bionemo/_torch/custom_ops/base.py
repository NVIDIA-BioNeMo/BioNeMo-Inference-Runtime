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

import torch

# Define some constants for benchmark scores
MAX_BENCHMARK_SCORE = 100
MID_BENCHMARK_SCORE = 50
MIN_BENCHMARK_SCORE = 0
IGNORE_BENCHMARK_SCORE = -1  # Too slow or error


class CustomOpBase:
    _SUPPORT_DICT = set()
    """Base class for custom operations."""

    @staticmethod
    def apply(*args, **kwargs) -> torch.Tensor:
        """Apply the operation."""
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def is_supported(*args, **kwargs) -> bool:
        """Check if the operation is supported."""
        return True

    @staticmethod
    def get_benchmark_score(*args, **kwargs) -> int:
        """Get benchmark score for the operation."""
        return MAX_BENCHMARK_SCORE
