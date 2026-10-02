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
"""SM90 TriMul KF K3: output LayerNorm, output projection and gate, with the optional fused residual.

``update = (LN_out(P) @ W_out.T + b_out) * sigmoid(LN_in(x) @ W_g.T + b_g)`` over ``trimul_kf_k2``'s
product ``P``, or ``(x + update) * mask`` with the residual. A source-free build carries no private
kernel adapter; everything re-exported here must keep working with it absent.
"""

from ._config import TrimulKFK3Selection, shipped_shapes
from .ops import TrimulKFK3Op, TrimulKFOutputFold, fold_output_weights, get_trimul_kf_k3_op

__all__ = [
    "TrimulKFK3Op",
    "TrimulKFK3Selection",
    "TrimulKFOutputFold",
    "fold_output_weights",
    "get_trimul_kf_k3_op",
    "shipped_shapes",
]
