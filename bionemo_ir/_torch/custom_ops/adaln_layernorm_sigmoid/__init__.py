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
"""Fused AdaLN-style LayerNorm/RMSNorm + sigmoid gate custom op.

Public surface of the package. The import path is unchanged from when this was
a flat module, so callers need no edits. A source-free build carries no
private kernel adapter; everything re-exported here must keep working with it
absent.
"""

from ._config import (
    _BUCKET_BIG_M,
    _TORCH_TO_CUTLASS_DTYPE,
    SHIPPED_N,
    SUPPORTED_SMS,
    bucket_variants,
    resolve_geometry,
    select_bucket,
)
from .cutedsl import AdaLNLayerNormSigmoidCuTe
from .ops import _invoke_vanilla_adaln_layernorm_sigmoid, get_adaln_layernorm_sigmoid_op

__all__ = [
    "AdaLNLayerNormSigmoidCuTe",
    "SHIPPED_N",
    "SUPPORTED_SMS",
    "_BUCKET_BIG_M",
    "_TORCH_TO_CUTLASS_DTYPE",
    "_invoke_vanilla_adaln_layernorm_sigmoid",
    "bucket_variants",
    "get_adaln_layernorm_sigmoid_op",
    "resolve_geometry",
    "select_bucket",
]
