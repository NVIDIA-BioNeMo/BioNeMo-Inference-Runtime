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
"""The conditioned transition block in one SM100 kernel.

``a + sigmoid(g) * MLP(sigmoid(gamma) * LayerNorm(a) + beta)`` for a bf16 2-way SwiGLU MLP, where row ``r`` of ``a``
takes row ``r % A`` of a per-atom table ``[sigmoid(gamma), beta, sigmoid(g)]`` that the caller projects once per
diffusion rollout: the AdaLN apply, both linears, the output gate and the residual read ``a`` once and write once.
:func:`shipped_shapes` lists the ``(width, hidden)`` it serves. A source-free build carries no private kernel adapter;
everything re-exported here must keep working with it absent.
"""

from ._config import shipped_shapes
from .ops import (
    EPS,
    ConditionedTransitionMlpOp,
    conditioned_transition_mlp_reference,
    get_conditioned_transition_mlp_op,
)

__all__ = [
    "EPS",
    "ConditionedTransitionMlpOp",
    "conditioned_transition_mlp_reference",
    "get_conditioned_transition_mlp_op",
    "shipped_shapes",
]
