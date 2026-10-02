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
"""SM90 TriMul KF K2: the batched triangle contraction of K1's channel-major ``a`` and ``b``.

``prod[b, d, i, j] = sum_k a[b, d, i, k] b[b, d, j, k]`` outgoing, ``a[b, d, k, i] b[b, d, k, j]``
incoming. A source-free build carries no private kernel adapter; everything re-exported here must
keep working with it absent.
"""

from ._config import TrimulKFK2Tile, shipped_widths
from .ops import TrimulKFK2Op, get_trimul_kf_k2_op

__all__ = ["TrimulKFK2Op", "TrimulKFK2Tile", "get_trimul_kf_k2_op", "shipped_widths"]
