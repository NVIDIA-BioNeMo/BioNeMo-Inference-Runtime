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
"""Gated sigmoid custom op: ``out = sigmoid(s @ W.T [+ bias]) * mha_out``.

Public surface of the package. The import path is unchanged from when this was
a flat module, so callers need no edits. A source-free build deletes
``_source.py`` and the private CuTeDSL kernel; everything re-exported here must
keep working with that file absent.
"""

from ._config import (
    _GS_CONFIGS_DIR,
    M_MEDIUM_THRESHOLD,
    M_SHORT_THRESHOLD,
    GatedSigmoidKernelConfig,
    _build_kernel_config,
    _classify_m_range,
    _make_sm80_config,
    _params_from_json,
    get_kernel_config,
    get_m_bucket,
    get_tile_params,
)
from .cutedsl import GatedSigmoidCuTe
from .ops import _invoke_vanilla_gated_sigmoid, get_gated_sigmoid_op

__all__ = [
    "GatedSigmoidCuTe",
    "GatedSigmoidKernelConfig",
    "M_MEDIUM_THRESHOLD",
    "M_SHORT_THRESHOLD",
    "_GS_CONFIGS_DIR",
    "_build_kernel_config",
    "_classify_m_range",
    "_invoke_vanilla_gated_sigmoid",
    "_make_sm80_config",
    "_params_from_json",
    "get_gated_sigmoid_op",
    "get_kernel_config",
    "get_m_bucket",
    "get_tile_params",
]
