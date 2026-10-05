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
"""SM90 TriMul KF K1: input LayerNorm, gated input projections and the row mask.

``a, b = mask * (LayerNorm(x) @ W_p.T + b_p) * sigmoid(LayerNorm(x) @ W_g.T + b_g)``, written
channel-major for ``trimul_kf_k2``. A source-free build carries no private kernel adapter;
everything re-exported here must keep working with it absent.
"""

from ._config import AB_LAYOUTS, TrimulKFK1Selection, ab_pitch, shipped_shapes
from .cutedsl import TrimulKFK1Output
from .ops import TrimulKFInputFold, TrimulKFK1Op, fold_input_weights, get_trimul_kf_k1_op

__all__ = [
    "AB_LAYOUTS",
    "TrimulKFInputFold",
    "TrimulKFK1Op",
    "TrimulKFK1Output",
    "TrimulKFK1Selection",
    "ab_pitch",
    "fold_input_weights",
    "get_trimul_kf_k1_op",
    "shipped_shapes",
]
