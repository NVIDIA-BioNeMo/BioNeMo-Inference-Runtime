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
"""CUBIN-backed executable adapter for the SM90 TriMul KF K2."""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelVariantUnavailable,
    current_stream_handle,
    tensor_s1_d0,
)

if TYPE_CHECKING:
    from ._config import TrimulKFK2Tile


class TrimulKFK2CubinExecutable(CuTeDSLKernelLibraryExecutable):
    """Match the CuTeDSL compiled-function call ABI using the C++ launcher."""

    def __init__(
        self, kernel_library: ModuleType, launcher: Any, target_sm: int, outgoing: bool, tile: TrimulKFK2Tile
    ) -> None:
        try:
            config = launcher.make_kernel_config(
                target_sm,
                launcher.DType.BFLOAT16,
                outgoing,
                int(tile.kernel_variant.removeprefix("K2_")),
                tile.tile_n,
                tile.cluster_m,
                tile.defer_kmin,
                tile.split_epi,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise CuTeDSLKernelVariantUnavailable(
                f"No trimul KF K2 CUBIN for SM{target_sm}, outgoing={outgoing}, {tile}"
            ) from error
        self._kernel_library = kernel_library
        self._launcher = launcher
        self._config = config

    def __call__(self, a: torch.Tensor, b: torch.Tensor, prod: torch.Tensor, n: int, l: int) -> None:  # noqa: E741
        library = self._kernel_library
        params = self._launcher.LaunchParams()
        params.a = tensor_s1_d0(library, a)
        params.b = tensor_s1_d0(library, b)
        params.prod = tensor_s1_d0(library, prod)
        params.n = n
        params.l = l
        params.stream = current_stream_handle(a)
        self._launcher.launch(self._config, params)
