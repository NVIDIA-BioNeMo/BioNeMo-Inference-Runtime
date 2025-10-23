# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from functools import partial
from typing import Callable

from .cuequiv_custom_ops import (CuEquivFusedSigmoidGatedDualGemm,
                                 CuEquivFusedSigmoidGatedDualGemmDualX)

_OPS_IMPLS_MAP = {
    "fused_sigmoid_gated_dual_gemm": [CuEquivFusedSigmoidGatedDualGemm],
    "fused_sigmoid_gated_dual_gemm_dual_x":
    [CuEquivFusedSigmoidGatedDualGemmDualX],
}


def get_custom_ops_impl(ops_name: str, *args, **kwargs) -> Callable:
    impls = _OPS_IMPLS_MAP.get(ops_name, [])
    for impl in impls:
        if impl.is_supported(*args, **kwargs):
            return partial(impl.apply, *args, **kwargs)
    return None
