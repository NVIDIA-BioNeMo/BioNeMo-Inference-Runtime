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
"""Fused transition MLP: ``[residual +] mask * (act(x @ W1.T + b1) @ W2.T + b2)``.

``act`` is ReLU (PairTransition, MSATransition) or a SiLU-gated linear unit (the SwiGLU
Transition). A source-free build carries no private kernel adapter; everything re-exported here
must keep working with it absent.
"""

from ._config import TransitionMlpVariant, shipped_variants
from .ops import get_transition_mlp_op, transition_mlp_reference

__all__ = ["TransitionMlpVariant", "get_transition_mlp_op", "shipped_variants", "transition_mlp_reference"]
