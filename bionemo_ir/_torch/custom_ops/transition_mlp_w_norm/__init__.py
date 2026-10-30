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
"""SM90 transition MLP with its LayerNorm in one kernel.

``[z +] mask * (act(LayerNorm(z) @ W1.T + b1) @ W2.T + b2)`` over bf16 ``z`` ``[..., C]``, with ``act`` ReLU or
SwiGLU: the LayerNorm folds into the first linear's weights (:func:`pack_weights`), the hidden activation stays on
chip, and the mask and the residual are optional per call. It runs over flat rows or, with per-item lengths, over the
valid rows of square pair maps. :func:`shipped_shapes` lists the ``(C, H, activation)`` it serves and
:func:`candidate_shapes` those built but not yet shown to beat the path they would replace. A source-free build
carries no private kernel adapter; everything re-exported here must keep working with it absent.
"""

from ._config import candidate_shapes, shipped_shapes
from .ops import (
    ACTIVATIONS,
    MAX_BATCH,
    MAX_ROWS,
    TransitionMlpWNormOp,
    TransitionMlpWNormPack,
    get_transition_mlp_w_norm_op,
    pack_weights,
    transition_mlp_w_norm_reference,
)

__all__ = [
    "ACTIVATIONS",
    "MAX_BATCH",
    "MAX_ROWS",
    "TransitionMlpWNormOp",
    "TransitionMlpWNormPack",
    "candidate_shapes",
    "get_transition_mlp_w_norm_op",
    "pack_weights",
    "shipped_shapes",
    "transition_mlp_w_norm_reference",
]
