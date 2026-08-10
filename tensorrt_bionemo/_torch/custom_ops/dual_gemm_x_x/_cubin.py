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
"""CUBIN-backed executable adapter for dual-GEMM ``x_x``."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import torch

from tensorrt_bionemo._torch._cutedsl_kernel_library import (
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelVariantUnavailable,
    tensor_s1_d0,
    tensor_s2_d1,
)


def _same_tensor_reference(x0: torch.Tensor, x1: torch.Tensor) -> bool:
    """Whether two arguments describe the same tensor operand."""
    return x0 is x1 or (
        x0.data_ptr() == x1.data_ptr()
        and x0.shape == x1.shape
        and x0.stride() == x1.stride()
        and x0.dtype == x1.dtype
        and x0.device == x1.device
    )


class DualGemmXxCubinExecutable(CuTeDSLKernelLibraryExecutable):
    """Match both source call ABIs using the direct C++ launcher."""

    def __init__(
        self,
        kernel_library: ModuleType,
        launcher: Any,
        target_sm: int,
        K: int,
        N: int,
        bucket: int,
        dtype: torch.dtype,
        transpose_out: bool,
        has_bias: bool,
        has_mask: bool,
    ):
        dtype_map = {
            torch.float16: launcher.DType.FLOAT16,
            torch.bfloat16: launcher.DType.BFLOAT16,
        }
        try:
            library_dtype = dtype_map[dtype]
        except KeyError as error:
            raise CuTeDSLKernelVariantUnavailable(f"dual_gemm_x_x CUBINs do not support {dtype}") from error

        try:
            config = launcher.make_kernel_config(
                target_sm,
                K,
                N,
                bucket,
                library_dtype,
                transpose_out,
                has_bias,
                has_mask,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise CuTeDSLKernelVariantUnavailable(
                f"No dual_gemm_x_x CUBIN for SM{target_sm}, K={K}, N={N}, "
                f"bucket={bucket}, dtype={dtype}, transpose_out={transpose_out}, "
                f"has_bias={has_bias}, has_mask={has_mask}"
            ) from error

        self._kernel_library = kernel_library
        self._launcher = launcher
        self._config = config
        self._dtype = dtype
        self._transpose_out = transpose_out
        self._has_bias = has_bias
        self._has_mask = has_mask

    def __call__(self, *args: Any) -> None:
        """Launch from either the one-X SM80 or duplicated-X SM90 signature."""
        if len(args) == 8:
            x, w0, w1, bias0, bias1, actual_seqlen, output, i_dim = args
        elif len(args) == 9:
            x, x1, w0, w1, bias0, bias1, actual_seqlen, output, i_dim = args
            if not _same_tensor_reference(x, x1):
                raise ValueError("dual_gemm_x_x SM90 requires x0 and x1 to refer to the same tensor")
        else:
            raise TypeError(f"dual_gemm_x_x executable expects 8 (SM80) or 9 (SM90) arguments; got {len(args)}")

        if (bias0 is None) != (bias1 is None):
            raise ValueError("bias0 and bias1 must both be supplied or both None")
        if self._has_bias != (bias0 is not None):
            raise ValueError(f"dual_gemm_x_x CUBIN was configured with has_bias={self._has_bias}")
        if self._has_mask != (actual_seqlen is not None):
            raise ValueError(f"dual_gemm_x_x CUBIN was configured with has_mask={self._has_mask}")

        for name, tensor in (("x", x), ("w0", w0), ("w1", w1), ("output", output)):
            if tensor.dtype != self._dtype:
                raise TypeError(f"{name} must use {self._dtype}; got {tensor.dtype}")
        for name, tensor in (("bias0", bias0), ("bias1", bias1)):
            if tensor is not None and tensor.dtype != self._dtype:
                raise TypeError(f"{name} must use {self._dtype}; got {tensor.dtype}")
        if actual_seqlen is not None and actual_seqlen.dtype != torch.int32:
            raise TypeError(f"actual_seqlen must use torch.int32; got {actual_seqlen.dtype}")

        params = self._launcher.LaunchParams()
        params.x = tensor_s2_d1(self._kernel_library, x)
        params.w0 = tensor_s2_d1(self._kernel_library, w0)
        params.w1 = tensor_s2_d1(self._kernel_library, w1)
        if bias0 is not None:
            params.bias0 = tensor_s1_d0(self._kernel_library, bias0)
            params.bias1 = tensor_s1_d0(self._kernel_library, bias1)
        if actual_seqlen is not None:
            params.actual_seqlen = tensor_s1_d0(self._kernel_library, actual_seqlen)
        params.output = tensor_s2_d1(
            self._kernel_library,
            output,
            dynamic_stride_dim=1 if self._transpose_out else 0,
        )
        params.i_dim = i_dim
        params.stream = torch.cuda.current_stream(x.device).cuda_stream
        self._launcher.launch(self._config, params)
