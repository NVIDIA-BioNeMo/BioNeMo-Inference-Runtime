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
    tensor_flat,
)

if TYPE_CHECKING:
    from ._config import TrimulKFK2Tile


class TrimulKFK2CubinExecutable(CuTeDSLKernelLibraryExecutable):
    """Match the CuTeDSL compiled-function call ABI using the C++ launcher.

    A dynamic tile runs its dynamic image, which claims work units from the counter the call passes last. A library
    built before that image existed (no image, or a launcher without the ``dynamic`` axis) runs the tile's static
    image instead, which ignores the counter: the same product, on the static schedule.

    A library built before K2 wrote its product at a row pitch has no launch field for it and writes the product
    dense: a call for a dense product runs there, a call for a pitched one is unavailable.
    """

    def __init__(
        self, kernel_library: ModuleType, launcher: Any, target_sm: int, outgoing: bool, tile: TrimulKFK2Tile
    ) -> None:
        geometry = (
            target_sm,
            launcher.DType.BFLOAT16,
            outgoing,
            int(tile.kernel_variant.removeprefix("K2_")),
            tile.tile_n,
            tile.cluster_m,
            tile.defer_kmin,
            tile.split_epi,
        )
        config = None
        if tile.dynamic:
            try:
                config = launcher.make_kernel_config(*geometry, dynamic=True)
            except (RuntimeError, TypeError, ValueError):
                config = None
        if config is None:
            try:
                config = launcher.make_kernel_config(*geometry)
            except (RuntimeError, TypeError, ValueError) as error:
                raise CuTeDSLKernelVariantUnavailable(
                    f"No trimul KF K2 CUBIN for SM{target_sm}, outgoing={outgoing}, {tile}"
                ) from error
        self._kernel_library = kernel_library
        self._launcher = launcher
        self._config = config
        self.dynamic = bool(getattr(config.spec, "dynamic", False))

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        prod: torch.Tensor,
        n: int,
        l: int,  # noqa: E741
        ab_pitch: int | None = None,
        ab_plane: int | None = None,
        prod_pitch: int | None = None,
        prod_plane: int | None = None,
        counter: torch.Tensor | None = None,
    ) -> None:
        library = self._kernel_library
        params = self._launcher.LaunchParams()
        params.a = tensor_flat(library, a)
        params.b = tensor_flat(library, b)
        params.prod = tensor_flat(library, prod)
        params.n = n
        params.l = l
        params.ab_pitch = n if ab_pitch is None else ab_pitch
        params.ab_plane = n * n if ab_plane is None else ab_plane
        if hasattr(type(params), "prod_pitch"):
            params.prod_pitch = n if prod_pitch is None else prod_pitch
            params.prod_plane = n * n if prod_plane is None else prod_plane
        elif prod_pitch not in (None, n) or prod_plane not in (None, n * n):
            raise CuTeDSLKernelVariantUnavailable("this trimul KF K2 CUBIN library predates pitched products")
        if self.dynamic:
            if counter is None:
                raise ValueError("a dynamic trimul KF K2 CUBIN claims its work from a counter; pass one")
            params.counter = tensor_flat(library, counter)
        params.stream = current_stream_handle(a)
        self._launcher.launch(self._config, params)
