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
"""Shared CuTeDSL kernel helpers: config JSON, source adapters, CUBIN launch."""

from ._cutedsl_kernel_library import (
    CuTeDSLKernelLibraryError,
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelLibraryUnavailable,
    CuTeDSLKernelVariantUnavailable,
    launch_compiled_kernel,
    populate_compiled_cache_from_library,
    require_kernel_backend,
    tensor_s1_d0,
    tensor_s2_d1,
    tensor_s3_d2,
    tensor_s4_d3,
)
from ._kernel_config_loader import (
    KernelConfigBundle,
    get_config_file_name,
    load_kernel_configs,
    resolve_implementation,
)
from ._kernel_source_loader import KernelSourceUnavailable, load_source_module

__all__ = [
    "CuTeDSLKernelLibraryError",
    "CuTeDSLKernelLibraryExecutable",
    "CuTeDSLKernelLibraryUnavailable",
    "CuTeDSLKernelVariantUnavailable",
    "KernelConfigBundle",
    "KernelSourceUnavailable",
    "get_config_file_name",
    "launch_compiled_kernel",
    "load_kernel_configs",
    "load_source_module",
    "populate_compiled_cache_from_library",
    "require_kernel_backend",
    "resolve_implementation",
    "tensor_s1_d0",
    "tensor_s2_d1",
    "tensor_s3_d2",
    "tensor_s4_d3",
]
