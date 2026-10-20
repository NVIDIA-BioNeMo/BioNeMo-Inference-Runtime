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
"""Identity-keyed child seeds for search rollouts."""

from __future__ import annotations

_GOLDEN = 0x9E3779B97F4A7C15
_MIX = 0xBF58476D1CE4E5B9
_MASK = (1 << 63) - 1


def derive_search_seed(parent: int, *parts: int) -> int:
    """Mix ``parent`` with identity parts into a nonzero 63-bit seed.

    The mix is a function of the parts, not of consumption order, so a
    replica/branch/checkpoint stream is reproducible when those groups are
    processed in a different chunk order.

    Args:
        parent: Root search seed.
        *parts: Nonnegative identity integers such as replica, branch, and
            checkpoint.

    Returns:
        A seed in ``[1, 2**63)``.
    """
    seed = int(parent) & _MASK
    for part in parts:
        seed ^= (int(part) + _GOLDEN) & _MASK
        seed = (seed * _MIX) & _MASK
        seed ^= seed >> 30
    return seed if seed != 0 else 1
