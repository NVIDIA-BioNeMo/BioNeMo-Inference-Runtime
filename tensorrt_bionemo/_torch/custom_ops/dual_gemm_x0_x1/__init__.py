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
"""Dual-GEMM ``x0_x1`` public API."""

from __future__ import annotations

from ._config import (
    _CONFIGS_DIR as _CONFIGS_DIR,
)
from ._config import (
    DualGemmX0X1KernelConfig as DualGemmX0X1KernelConfig,
)
from ._config import (
    compute_S as compute_S,
)
from ._config import (
    get_kernel_config as get_kernel_config,
)
from ._config import (
    get_nearest_bucket as get_nearest_bucket,
)
from ._config import (
    kernel_is_sm90 as kernel_is_sm90,
)
from .cutedsl import (
    DualGemmX0X1CuTe as DualGemmX0X1CuTe,
)
from .ops import (
    _invoke_cuequiv_dual_gemm_x0_x1 as _invoke_cuequiv_dual_gemm_x0_x1,
)
from .ops import (
    _invoke_cute_dual_gemm_x0_x1 as _invoke_cute_dual_gemm_x0_x1,
)
from .ops import (
    _invoke_vanilla_dual_gemm_x0_x1 as _invoke_vanilla_dual_gemm_x0_x1,
)
from .ops import (
    get_dual_gemm_x0_x1_op as get_dual_gemm_x0_x1_op,
)

__all__ = [
    "DualGemmX0X1CuTe",
    "DualGemmX0X1KernelConfig",
    "_CONFIGS_DIR",
    "compute_S",
    "get_dual_gemm_x0_x1_op",
    "get_kernel_config",
    "get_nearest_bucket",
    "kernel_is_sm90",
]
