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
"""CUBIN-backed executable adapter for the fused transition MLP."""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelVariantUnavailable,
    current_stream_handle,
    tensor_s1_d0,
    tensor_s2_d1,
)

if TYPE_CHECKING:
    from ._config import TransitionMlpVariant


class TransitionMlpCubinExecutable(CuTeDSLKernelLibraryExecutable):
    """Match the CuTeDSL compiled-function call ABI using the C++ launcher."""

    def __init__(
        self,
        kernel_library: ModuleType,
        launcher: Any,
        target_sm: int,
        dtype: torch.dtype,
        variant: TransitionMlpVariant,
        bucket: int,
    ) -> None:
        if dtype != torch.bfloat16:
            raise CuTeDSLKernelVariantUnavailable(f"transition MLP CUBINs do not support {dtype}")
        try:
            config = launcher.make_kernel_config(
                target_sm,
                launcher.DType.BFLOAT16,
                variant.activation != "relu",
                variant.activation == "silu_gate_3way",
                variant.has_bias,
                variant.has_mask,
                variant.has_residual,
                variant.width,
                variant.hidden,
                bucket,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise CuTeDSLKernelVariantUnavailable(
                f"No transition MLP CUBIN for SM{target_sm}, dtype={dtype}, {variant}, bucket={bucket}"
            ) from error
        self._kernel_library = kernel_library
        self._launcher = launcher
        self._config = config
        self._variant = variant

    def __call__(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
        residual: torch.Tensor | None,
        mask: torch.Tensor | None,
        output: torch.Tensor,
    ) -> None:
        if (
            (b1 is not None) != self._variant.has_bias
            or (mask is not None) != self._variant.has_mask
            or (residual is not None) != self._variant.has_residual
        ):
            raise CuTeDSLKernelVariantUnavailable(f"transition MLP CUBIN operands do not match {self._variant}")
        library = self._kernel_library
        params = self._launcher.LaunchParams()
        params.x = tensor_s2_d1(library, x)
        params.w1 = tensor_s2_d1(library, w1)
        params.w2 = tensor_s2_d1(library, w2)
        if b1 is not None:
            params.b1 = tensor_s1_d0(library, b1)
            params.b2 = tensor_s1_d0(library, b2)
        if residual is not None:
            params.residual = tensor_s2_d1(library, residual)
        if mask is not None:
            params.mask = tensor_s1_d0(library, mask)
        params.output = tensor_s2_d1(library, output)
        params.stream = current_stream_handle(x)
        self._launcher.launch(self._config, params)
