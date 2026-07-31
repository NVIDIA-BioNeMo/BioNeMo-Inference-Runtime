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


def mismatch_percentage(a: torch.Tensor,
                        b: torch.Tensor,
                        *,
                        atol: float = 1e-8,
                        rtol: float = 1e-5) -> float:
    if a.shape != b.shape:
        raise ValueError("Shape mismatch")

    diff = torch.abs(a - b)
    tol = atol + rtol * torch.abs(b)

    mismatches = diff > tol
    num_mismatch = mismatches.sum().item()
    total = mismatches.numel()

    return 100.0 * num_mismatch / total
