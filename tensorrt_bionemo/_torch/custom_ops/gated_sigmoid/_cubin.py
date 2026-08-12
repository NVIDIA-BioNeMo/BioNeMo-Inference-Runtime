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
"""CUBIN-backed executable adapter for the gated sigmoid GEMM."""

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


class GatedSigmoidCubinExecutable(CuTeDSLKernelLibraryExecutable):
    """Match the CuTeDSL compiled-function call ABI using the C++ launcher."""

    def __init__(
        self,
        kernel_library: ModuleType,
        launcher: Any,
        target_sm: int,
        m_bucket: int,
        dtype: torch.dtype,
        has_bias: bool,
        tile_params: dict[str, Any],
    ):
        dtype_map = {
            torch.float16: launcher.DType.FLOAT16,
            torch.bfloat16: launcher.DType.BFLOAT16,
        }
        try:
            library_dtype = dtype_map[dtype]
        except KeyError as error:
            raise CuTeDSLKernelVariantUnavailable(f"gated sigmoid CUBINs do not support {dtype}") from error

        try:
            atom_layout_m, atom_layout_n, atom_layout_k = tile_params["atom_layout_mnk"]
            config = launcher.make_kernel_config(
                target_sm,
                library_dtype,
                has_bias,
                int(tile_params["m_block_size"]),
                int(tile_params["n_block_size"]),
                int(tile_params["k_block_size"]),
                int(tile_params["num_stages"]),
                int(tile_params["raster_factor"]),
                int(atom_layout_m),
                int(atom_layout_n),
                int(atom_layout_k),
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise CuTeDSLKernelVariantUnavailable(
                f"No gated sigmoid CUBIN for SM{target_sm}, dtype={dtype}, "
                f"has_bias={has_bias}, m_bucket={m_bucket}, tile={tile_params}"
            ) from error

        self._kernel_library = kernel_library
        self._launcher = launcher
        self._config = config
        self._has_bias = has_bias

    def __call__(
        self,
        s: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        mha_out: torch.Tensor,
        output: torch.Tensor,
        mult: int,
        inner: int,
    ) -> None:
        if (bias is not None) != self._has_bias:
            raise CuTeDSLKernelVariantUnavailable(
                f"gated sigmoid CUBIN was selected for has_bias={self._has_bias} "
                f"but received bias={'a tensor' if bias is not None else 'None'}"
            )
        params = self._launcher.LaunchParams()
        params.s = tensor_s2_d1(self._kernel_library, s)
        params.weight = tensor_s2_d1(self._kernel_library, weight)
        if bias is not None:
            params.bias = tensor_s1_d0(self._kernel_library, bias)
        params.mha_out = tensor_s2_d1(self._kernel_library, mha_out)
        params.output = tensor_s2_d1(self._kernel_library, output)
        params.mult = int(mult)
        params.inner = int(inner)
        params.stream = torch.cuda.current_stream(s.device).cuda_stream
        self._launcher.launch(self._config, params)
