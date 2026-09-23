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
"""CUBIN-backed executable adapter for the dual-GEMM ``x0_x1`` op."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelVariantUnavailable,
    tensor_s1_d0,
    tensor_s2_d1,
)


class DualGemmX0X1CubinExecutable(CuTeDSLKernelLibraryExecutable):
    """Adapt the Ampere or Hopper source-call ABI to the C++ launcher."""

    def __init__(
        self,
        kernel_library: ModuleType,
        launcher: Any,
        target_sm: int,
        K: int,
        N: int,
        bucket: int,
        dtype: torch.dtype,
        has_bias: bool,
        K1: int | None = None,
        fused_residual: bool = False,
    ):
        K1 = K if K1 is None else K1
        dtype_map = {
            torch.float16: launcher.DType.FLOAT16,
            torch.bfloat16: launcher.DType.BFLOAT16,
        }
        try:
            library_dtype = dtype_map[dtype]
        except KeyError as error:
            raise CuTeDSLKernelVariantUnavailable(f"dual_gemm x0_x1 CUBINs do not support {dtype}") from error

        try:
            config = launcher.make_kernel_config(
                target_sm,
                K,
                N,
                bucket,
                library_dtype,
                has_bias,
                K1=K1,
                fused_residual=fused_residual,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise CuTeDSLKernelVariantUnavailable(
                f"No dual_gemm x0_x1 CUBIN for SM{target_sm}, K0={K}, K1={K1}, N={N}, "
                f"bucket={bucket}, dtype={dtype}, has_bias={has_bias}"
            ) from error

        self._kernel_library = kernel_library
        self._launcher = launcher
        self._config = config
        self._is_sm90 = int(config.spec.kernel_sm) == 90
        self._fused_residual = bool(fused_residual)

    def __call__(self, *args: Any) -> None:
        if self._is_sm90 or self._fused_residual:
            expected = "(X0, X1, W0, W1, bias0, bias1, actual_seqlen, residual, out, I_dim)"
            if len(args) != 10:
                raise CuTeDSLKernelVariantUnavailable(
                    f"dual_gemm x0_x1 selected a CUBIN expecting {expected}; got {len(args)} arguments"
                )
            X0, X1, W0, W1, bias0, bias1, actual_seqlen, residual, out, i_dim = args
            if self._fused_residual != (actual_seqlen is not None and residual is not None):
                raise CuTeDSLKernelVariantUnavailable(
                    "dual_gemm x0_x1 fused-residual operands do not match the selected CUBIN"
                )
        else:
            if len(args) != 7:
                raise CuTeDSLKernelVariantUnavailable(
                    f"dual_gemm x0_x1 selected an Ampere CUBIN expecting "
                    f"(X0, X1, W0, W1, bias0, bias1, out); got {len(args)} arguments"
                )
            X0, X1, W0, W1, bias0, bias1, out = args

        params = self._launcher.LaunchParams()
        params.x0 = tensor_s2_d1(self._kernel_library, X0)
        params.x1 = tensor_s2_d1(self._kernel_library, X1)
        params.w0 = tensor_s2_d1(self._kernel_library, W0)
        params.w1 = tensor_s2_d1(self._kernel_library, W1)
        params.out = tensor_s2_d1(self._kernel_library, out)
        has_bias = bias0 is not None
        if has_bias != self._config.has_bias:
            raise CuTeDSLKernelVariantUnavailable(
                f"dual_gemm x0_x1 CUBIN was built with has_bias={self._config.has_bias} "
                f"but was called with has_bias={has_bias}"
            )
        if has_bias:
            params.bias0 = tensor_s1_d0(self._kernel_library, bias0)
            params.bias1 = tensor_s1_d0(self._kernel_library, bias1)
        if self._fused_residual:
            params.actual_seqlen = tensor_s1_d0(self._kernel_library, actual_seqlen)
            params.residual = tensor_s2_d1(self._kernel_library, residual)
            params.i_dim = i_dim
        params.stream = torch.cuda.current_stream(X0.device).cuda_stream
        self._launcher.launch(self._config, params)
