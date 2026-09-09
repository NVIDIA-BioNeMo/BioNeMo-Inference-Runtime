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
"""CUBIN-backed executable adapter for pair-weighted averaging."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelVariantUnavailable,
    tensor_s2_d1,
    tensor_s4_d3,
)


class PairWeightedAveragingCubinExecutable(CuTeDSLKernelLibraryExecutable):
    """Match the five-tensor CuTeDSL source executable signature."""

    def __init__(
        self,
        kernel_library: ModuleType,
        launcher: Any,
        target_sm: int,
        I: int,
        J: int,
        S: int,
        D: int,
        c_m: int,
        dtype: torch.dtype,
    ):
        dtype_map = {
            torch.float16: launcher.DType.FLOAT16,
            torch.bfloat16: launcher.DType.BFLOAT16,
        }
        try:
            library_dtype = dtype_map[dtype]
        except KeyError as error:
            raise CuTeDSLKernelVariantUnavailable(f"pair_weighted_averaging CUBINs do not support {dtype}") from error

        try:
            config = launcher.make_kernel_config(
                target_sm,
                I,
                J,
                S,
                D,
                c_m,
                library_dtype,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise CuTeDSLKernelVariantUnavailable(
                f"No pair_weighted_averaging CUBIN for SM{target_sm}, I={I}, J={J}, S={S}, "
                f"D={D}, c_m={c_m}, dtype={dtype}"
            ) from error

        self._kernel_library = kernel_library
        self._launcher = launcher
        self._config = config
        self._dtype = dtype

    def __call__(
        self,
        w: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        Wo: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Validate and launch one precompiled PWA variant."""
        for name, tensor in (
            ("w", w),
            ("v", v),
            ("g", g),
            ("Wo", Wo),
            ("out", out),
        ):
            if tensor.dtype != self._dtype:
                raise TypeError(f"{name} must use {self._dtype}; got {tensor.dtype}")
            if tensor.stride(-1) != 1:
                raise ValueError(f"{name} must have a contiguous final dimension")

        params = self._launcher.LaunchParams()
        params.w = tensor_s4_d3(self._kernel_library, w)
        params.v = tensor_s4_d3(self._kernel_library, v)
        params.g = tensor_s4_d3(self._kernel_library, g)
        params.weight = tensor_s2_d1(self._kernel_library, Wo)
        params.output = tensor_s4_d3(self._kernel_library, out)
        params.stream = torch.cuda.current_stream(w.device).cuda_stream
        self._launcher.launch(self._config, params)
