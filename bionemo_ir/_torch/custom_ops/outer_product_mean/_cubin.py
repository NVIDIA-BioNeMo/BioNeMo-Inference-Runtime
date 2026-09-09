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
"""CUBIN-backed executable adapter for the fused outer-product-mean."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelVariantUnavailable,
    tensor_s1_d0,
    tensor_s2_d1,
    tensor_s3_d2,
)


class OuterProductMeanCubinExecutable(CuTeDSLKernelLibraryExecutable):
    """Match the CuTeDSL compiled-function call ABI using the C++ launcher."""

    def __init__(
        self,
        kernel_library: ModuleType,
        launcher: Any,
        target_sm: int,
        dtype: torch.dtype,
        has_bias: bool,
        norm_before: bool,
        config_identity: str,
    ):
        dtype_map = {
            torch.float16: launcher.DType.FLOAT16,
            torch.bfloat16: launcher.DType.BFLOAT16,
        }
        try:
            library_dtype = dtype_map[dtype]
        except KeyError as error:
            raise CuTeDSLKernelVariantUnavailable(f"outer-product-mean CUBINs do not support {dtype}") from error

        try:
            config = launcher.make_kernel_config(target_sm, library_dtype, has_bias, norm_before, config_identity)
        except (RuntimeError, TypeError, ValueError) as error:
            raise CuTeDSLKernelVariantUnavailable(
                f"No outer-product-mean CUBIN for SM{target_sm}, dtype={dtype}, "
                f"has_bias={has_bias}, norm_before={norm_before}, config={config_identity}"
            ) from error

        self._kernel_library = kernel_library
        self._launcher = launcher
        self._config = config
        self._has_bias = has_bias

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        num_mask: torch.Tensor,
        W_o: torch.Tensor,
        bias: torch.Tensor | None,
        out: torch.Tensor,
    ) -> None:
        if (bias is not None) != self._has_bias:
            raise CuTeDSLKernelVariantUnavailable(
                f"outer-product-mean CUBIN was selected for has_bias={self._has_bias} "
                f"but received bias={'a tensor' if bias is not None else 'None'}"
            )
        params = self._launcher.LaunchParams()
        params.a = tensor_s3_d2(self._kernel_library, a)
        params.b = tensor_s3_d2(self._kernel_library, b)
        params.num_mask = tensor_s3_d2(self._kernel_library, num_mask)
        # W_o and bias lower to bare pointers because C/D/C_z are static. The
        # views still carry extents and a device ordinal so the launcher can
        # check both before discarding the shape.
        params.weight = tensor_s2_d1(self._kernel_library, W_o)
        if bias is not None:
            params.bias = tensor_s1_d0(self._kernel_library, bias)
        params.output = tensor_s3_d2(self._kernel_library, out)
        params.stream = torch.cuda.current_stream(a.device).cuda_stream
        self._launcher.launch(self._config, params)
