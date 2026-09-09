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
"""Dual-GEMM ``x_x`` public API."""

from __future__ import annotations

from ._config import (
    _CONFIGS_DIR as _CONFIGS_DIR,
)
from ._config import (
    _FALLBACK_SM as _FALLBACK_SM,
)
from ._config import (
    _TUNED_SMS as _TUNED_SMS,
)
from ._config import (
    _VARIANT_KEY_RE as _VARIANT_KEY_RE,
)
from ._config import (
    DualGemmXxKernelConfig as DualGemmXxKernelConfig,
)
from ._config import (
    _fallback_sm as _fallback_sm,
)
from ._config import (
    _nearest_variant as _nearest_variant,
)
from ._config import (
    _variant_key as _variant_key,
)
from ._config import (
    get_kernel_config as get_kernel_config,
)
from ._config import (
    get_nearest_bucket as get_nearest_bucket,
)
from .cutedsl import (
    _TORCH_TO_DTYPE_STR as _TORCH_TO_DTYPE_STR,
)
from .cutedsl import (
    DualGemmXxCuTe as DualGemmXxCuTe,
)
from .cutedsl import (
    _compute_S as _compute_S,
)
from .cutedsl import (
    _dtype_str as _dtype_str,
)
from .ops import (
    _invoke_cuequiv_dual_gemm_x_x as _invoke_cuequiv_dual_gemm_x_x,
)
from .ops import (
    _invoke_cute_dual_gemm_x_x as _invoke_cute_dual_gemm_x_x,
)
from .ops import (
    _invoke_vanilla_dual_gemm_x_x as _invoke_vanilla_dual_gemm_x_x,
)
from .ops import (
    get_dual_gemm_x_x_op as get_dual_gemm_x_x_op,
)
