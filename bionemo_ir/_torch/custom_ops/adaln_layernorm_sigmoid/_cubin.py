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
"""CUBIN-backed executable adapter for the AdaLN fused kernel."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelVariantUnavailable,
    tensor_s2_d1,
)


class AdaLNLayerNormSigmoidCubinExecutable(CuTeDSLKernelLibraryExecutable):
    """Match the CuTeDSL compiled-function call ABI using the C++ launcher."""

    def __init__(
        self,
        kernel_library: ModuleType,
        launcher: Any,
        target_sm: int,
        dtype: torch.dtype,
        N: int,
        threads_per_row: int,
        num_threads: int,
        rms_norm: bool,
    ):
        dtype_map = {
            torch.float16: launcher.DType.FLOAT16,
            torch.bfloat16: launcher.DType.BFLOAT16,
            torch.float32: launcher.DType.FLOAT32,
        }
        try:
            library_dtype = dtype_map[dtype]
        except KeyError as error:
            raise CuTeDSLKernelVariantUnavailable(f"AdaLN CUBINs do not support {dtype}") from error

        try:
            config = launcher.make_kernel_config(
                target_sm,
                library_dtype,
                N,
                threads_per_row,
                num_threads,
                rms_norm,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            # N is compiled in, so an unshipped feature dimension has no payload
            # at all -- unlike a tile choice, it cannot fall back to a neighbour.
            raise CuTeDSLKernelVariantUnavailable(
                f"No AdaLN CUBIN for SM{target_sm}, dtype={dtype}, N={N}, "
                f"threads_per_row={threads_per_row}, num_threads={num_threads}, "
                f"rms_norm={rms_norm}"
            ) from error

        self._kernel_library = kernel_library
        self._launcher = launcher
        self._config = config

    def __call__(
        self,
        x: torch.Tensor,
        s_scale: torch.Tensor,
        s_bias: torch.Tensor,
        out: torch.Tensor,
        eps: float,
        mult: int,
        inner: int,
    ) -> None:
        params = self._launcher.LaunchParams()
        params.x = tensor_s2_d1(self._kernel_library, x)
        params.s_scale = tensor_s2_d1(self._kernel_library, s_scale)
        params.s_bias = tensor_s2_d1(self._kernel_library, s_bias)
        params.output = tensor_s2_d1(self._kernel_library, out)
        params.eps = float(eps)
        params.mult = int(mult)
        params.inner = int(inner)
        params.stream = torch.cuda.current_stream(x.device).cuda_stream
        self._launcher.launch(self._config, params)
