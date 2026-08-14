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
"""CUBIN-backed executable adapter for triangle attention."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import torch

from bionemo_ir._torch._cutedsl_kernel_library import (
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelVariantUnavailable,
    tensor_s1_d0,
    tensor_s3_d2,
    tensor_s4_d3,
)


class TriangleAttentionCubinExecutable(CuTeDSLKernelLibraryExecutable):
    """Match the CuTeDSL compiled-function call ABI using the C++ launcher."""

    def __init__(
        self,
        kernel_library: ModuleType,
        launcher: Any,
        target_sm: int,
        head_dim: int,
        bucket: int,
        dtype: torch.dtype,
        qkv_packed: bool,
    ):
        dtype_map = {
            torch.float16: launcher.DType.FLOAT16,
            torch.bfloat16: launcher.DType.BFLOAT16,
        }
        try:
            library_dtype = dtype_map[dtype]
        except KeyError as error:
            raise CuTeDSLKernelVariantUnavailable(f"triangle attention CUBINs do not support {dtype}") from error

        try:
            config = launcher.make_kernel_config(target_sm, head_dim, bucket, library_dtype, qkv_packed)
        except (RuntimeError, TypeError, ValueError) as error:
            raise CuTeDSLKernelVariantUnavailable(
                f"No triangle attention CUBIN for SM{target_sm}, "
                f"head_dim={head_dim}, bucket={bucket}, dtype={dtype}, "
                f"qkv_packed={qkv_packed}"
            ) from error
        if not config.spec.supports_direct_launch:
            raise CuTeDSLKernelVariantUnavailable(
                f"Triangle attention SM{target_sm}, head_dim={head_dim} uses "
                "a native Hopper ABI whose direct launcher is not implemented"
            )

        self._kernel_library = kernel_library
        self._launcher = launcher
        self._config = config

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        actual_s_kv: torch.Tensor,
        output: torch.Tensor,
        lse: torch.Tensor,
        _softmax_scale_log2: float,
        softmax_scale: float,
        i_dim: int,
    ) -> None:
        params = self._launcher.LaunchParams()
        params.q = tensor_s3_d2(self._kernel_library, q)
        params.k = tensor_s3_d2(self._kernel_library, k)
        params.v = tensor_s3_d2(self._kernel_library, v)
        params.actual_s_kv = tensor_s1_d0(self._kernel_library, actual_s_kv)
        params.bias = tensor_s4_d3(self._kernel_library, bias)
        params.output = tensor_s3_d2(self._kernel_library, output)
        params.lse = tensor_s3_d2(self._kernel_library, lse)
        params.softmax_scale = softmax_scale
        params.i_dim = i_dim
        params.stream = torch.cuda.current_stream(q.device).cuda_stream
        self._launcher.launch(self._config, params)
