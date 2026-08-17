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
"""Public custom-op getters.

Imports are lazy so a caller that only needs one family — the CUBIN builder
enumerating AdaLN, for example — does not import cuequivariance-backed dual
GEMM ops.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "get_dual_gemm_x_x_op",
    "get_dual_gemm_x0_x1_op",
    "get_adaln_layernorm_sigmoid_op",
    "get_gated_sigmoid_op",
    "get_outer_product_mean_op",
    "get_pair_weighted_averaging_op",
    "LNProjMoveaxisPad",
]

_EXPORTS = {
    "LNProjMoveaxisPad": ".fused_ln_proj_moveaxis_pad",
    "get_adaln_layernorm_sigmoid_op": ".adaln_layernorm_sigmoid",
    "get_dual_gemm_x0_x1_op": ".dual_gemm_x0_x1",
    "get_dual_gemm_x_x_op": ".dual_gemm_x_x",
    "get_gated_sigmoid_op": ".gated_sigmoid",
    "get_outer_product_mean_op": ".outer_product_mean",
    "get_pair_weighted_averaging_op": ".pair_weighted_averaging",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
