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
"""CUBIN-backed executable adapter for the conditioned transition MLP."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelVariantUnavailable,
    current_stream_handle,
    tensor_s2_d1,
)


class ConditionedTransitionMlpCubinExecutable(CuTeDSLKernelLibraryExecutable):
    """Launch one packaged config through the C++ launcher, with the source executable's call ABI.

    The library keys a shape's images by the activation (the 2-way or 3-way SwiGLU), ``cond_stage`` (the conditioning
    staged in shared memory) and ``samples`` (sample-major tiles of that many samples; 0 for row tiles).
    """

    def __init__(
        self,
        kernel_library: ModuleType,
        launcher: Any,
        target_sm: int,
        width: int,
        hidden: int,
        three_way: bool,
        cond_stage: bool,
        samples: int,
    ) -> None:
        try:
            config = launcher.make_kernel_config(
                target_sm, launcher.DType.BFLOAT16, width, hidden, three_way, cond_stage, samples
            )
        except (IndexError, RuntimeError, TypeError, ValueError) as error:
            raise CuTeDSLKernelVariantUnavailable(
                f"No conditioned transition CUBIN for SM{target_sm}, width={width}, hidden={hidden}, "
                f"three_way={three_way}, cond_stage={cond_stage}, samples={samples}"
            ) from error
        self._kernel_library = kernel_library
        self._launcher = launcher
        self._config = config

    def __call__(
        self, x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, cond: torch.Tensor, output: torch.Tensor
    ) -> None:
        library = self._kernel_library
        params = self._launcher.LaunchParams()
        params.x = tensor_s2_d1(library, x)
        params.w1 = tensor_s2_d1(library, w1)
        params.w2 = tensor_s2_d1(library, w2)
        params.cond = tensor_s2_d1(library, cond)
        params.output = tensor_s2_d1(library, output)
        params.stream = current_stream_handle(x)
        self._launcher.launch(self._config, params)
