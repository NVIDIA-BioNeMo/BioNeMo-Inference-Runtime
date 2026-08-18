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
"""Fused Outer-Product-Mean custom op with a PyTorch fallback.

Public surface of the package. The import path is unchanged from when this was
a flat module, so callers need no edits. A source-free build carries no
``_source.py``; everything re-exported here must keep working with that module
absent.
"""

from ._config import (
    _KERNEL_C,
    _KERNEL_CZ,
    _KERNEL_D,
    _OPM_CONFIGS_DIR,
    KernelConfig,
    _parse_key,
    _select_opm_config_bucket,
    config_identity,
    default_config,
    select_opm_config,
)
from .cutedsl import OuterProductMeanCuTe
from .ops import _invoke_vanilla_opm, get_outer_product_mean_op

__all__ = [
    "OuterProductMeanCuTe",
    "KernelConfig",
    "_KERNEL_C",
    "_KERNEL_CZ",
    "_KERNEL_D",
    "_OPM_CONFIGS_DIR",
    "_invoke_vanilla_opm",
    "_parse_key",
    "_select_opm_config_bucket",
    "config_identity",
    "default_config",
    "get_outer_product_mean_op",
    "select_opm_config",
]
