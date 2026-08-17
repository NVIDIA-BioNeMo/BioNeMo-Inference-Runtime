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
"""Fused pair-weighted-averaging public package facade."""

from __future__ import annotations

from typing import Any

from bionemo_ir._torch._kernel_source_loader import load_source_module

from ._config import (
    _KERNEL_CM as _KERNEL_CM,
)
from ._config import (
    _KERNEL_D as _KERNEL_D,
)
from ._config import (
    _KERNEL_H as _KERNEL_H,
)
from ._config import (
    _PWA_CONFIGS_DIR as _PWA_CONFIGS_DIR,
)
from ._config import (
    _SUPPORTED_DIMS as _SUPPORTED_DIMS,
)
from ._config import (
    _SUPPORTED_SM as _SUPPORTED_SM,
)
from ._config import (
    PWAConfigParams as PWAConfigParams,
)
from ._config import (
    PWAConfigSelection as PWAConfigSelection,
)
from ._config import (
    _parse_key as _parse_key,
)
from ._config import (
    _select_pwa_config_bucket as _select_pwa_config_bucket,
)
from ._config import (
    is_profitable_shape as is_profitable_pwa_shape,
)
from ._config import (
    is_supported_dims as is_supported_pwa_dims,
)
from ._config import (
    select_pwa_config as select_pwa_config,
)
from .cutedsl import (
    _TORCH_TO_DTYPE_STR as _DTYPE_STR,
)
from .cutedsl import (
    PairWeightedAveragingCuTe as PairWeightedAveragingCuTe,
)
from .cutedsl import (
    pwa_cute as pwa_cute,
)
from .ops import (
    _invoke_vanilla_pwa as _invoke_vanilla_pwa,
)
from .ops import (
    _pwa_cute_instance as _pwa_cute_instance,
)
from .ops import (
    get_pair_weighted_averaging_op as get_pair_weighted_averaging_op,
)
from .ops import (
    get_sm_version as get_sm_version,
)

__all__ = [
    "PWAConfigParams",
    "PWAConfigSelection",
    "PairWeightedAveragingCuTe",
    "_DTYPE_STR",
    "_invoke_vanilla_pwa",
    "_select_pwa_config_bucket",
    "compile_kernel",
    "get_pair_weighted_averaging_op",
    "is_profitable_pwa_shape",
    "is_supported_pwa_dims",
    "make_fake_args",
    "pwa_cute",
    "select_pwa_config",
]


def make_fake_args(
    config: object,
    I: int | None = None,
    Jp: int | None = None,
    N: int | None = None,
) -> tuple[object, object, object, object, object]:
    """Lazily build development-time dynamic fake tensors."""
    return load_source_module(__package__).make_fake_args(config, I, Jp, N)


def compile_kernel(
    config: object,
    I: int | None = None,
    Jp: int | None = None,
    N: int | None = None,
) -> object:
    """Lazily compile the development-time dynamic source."""
    return load_source_module(__package__).compile_kernel(config, I, Jp, N)


def __getattr__(name: str) -> Any:
    """Resolve historical source-only attributes without eager private imports."""
    if name in {"PWAConfig", "PWAFused", "_dt"}:
        return getattr(load_source_module(__package__), name)
    if name == "_compile_cache":
        return PairWeightedAveragingCuTe._compiled_cache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
