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
"""Source-or-CUBIN interface for the SM90 TriMul KF K2."""

from __future__ import annotations

import contextlib
import functools
from typing import Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryError,
    CuTeDSLKernelLibraryExecutable,
    launch_compiled_kernel,
    load_source_module,
    populate_compiled_cache_from_library,
)
from bionemo_ir.dsl_kernels.cute_cache import FORCE_CUBIN_ENV, CuteKernelCache
from bionemo_ir.logger import logger

from ._config import KERNEL_ABIS, TrimulKFK2Tile, anchors, select, shipped_tiles
from ._cubin import TrimulKFK2CubinExecutable

__all__ = ["TrimulKFK2CuTe"]

_INT32_MAX = 2**31 - 1


@functools.lru_cache(maxsize=256)
def batch_chunks(B: int, per_batch: int) -> tuple[tuple[int, int], ...]:
    """Balanced batch ranges whose flat extents, ``per_batch`` elements a batch, stay within int32."""
    if per_batch > _INT32_MAX:
        raise ValueError(f"trimul KF K2 supports fewer than 2^31 elements per batch, got {per_batch}")
    max_batch = _INT32_MAX // per_batch
    chunks = -(-B // max_batch)
    step = -(-B // chunks)
    return tuple((b, min(B, b + step)) for b in range(0, B, step))


class TrimulKFK2CuTe(CuteKernelCache):
    """Cached backend for K2, the triangle contraction of K1's channel-major ``a`` and ``b``.

    Each call runs the tile its width's configs tune for the anchor nearest the token count.

    Args:
        sm_version: The SM whose configs and kernels to use; defaults to this GPU's.
    """

    _compiled_cache: dict[tuple, Any] = {}

    def __init__(self, sm_version: int | None = None) -> None:
        if sm_version is None:
            major, minor = torch.cuda.get_device_capability()
            sm_version = major * 10 + minor
        self._sm_version = sm_version
        self._num_sms: int | None = None

    def select(self, D: int, n: int) -> tuple[int, TrimulKFK2Tile] | None:
        """The anchor a call over ``n`` tokens selects and the tile that runs it, or ``None`` if ``D`` does not ship."""
        return select(self._sm_version, D, n)

    def ships(self, D: int, outgoing: bool) -> bool:
        """Whether this build can run width ``D`` in one direction, from source or from packaged CUBINs."""
        entries = anchors(self._sm_version, D)
        if not entries:
            return False
        if not self.force_cubin():
            try:
                load_source_module(__package__)
            except ImportError:
                pass
            else:
                return True
        try:
            for tile in shipped_tiles(list(entries.values())):
                self._load_cubin_executable(self._key(outgoing, tile), outgoing, tile)
        except (CuTeDSLKernelLibraryError, RuntimeError):
            return False
        return True

    def _key(self, outgoing: bool, tile: TrimulKFK2Tile) -> tuple:
        return (self._sm_version, outgoing, *tile)

    def _load_cubin_executable(
        self, key: tuple, outgoing: bool, tile: TrimulKFK2Tile, source_error: Exception | None = None
    ) -> Any:
        try:
            executable = populate_compiled_cache_from_library(
                TrimulKFK2CuTe._compiled_cache,
                key,
                "trimul_kf_k2",
                lambda library, launcher: TrimulKFK2CubinExecutable(
                    library, launcher, self._sm_version, outgoing, tile
                ),
            )
        except CuTeDSLKernelLibraryError as library_error:
            if self.force_cubin():
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV}=1 forces the trimul KF K2 CUBIN path, but no CUBIN is available for "
                    f"SM{self._sm_version}, outgoing={outgoing}, {tile}"
                ) from library_error
            raise library_error from source_error
        logger.info(f"CuTeDSL trimul KF K2: using CUBIN kernel for SM{self._sm_version}, outgoing={outgoing}, {tile}")
        return executable

    def _get_or_compile(self, outgoing: bool, tile: TrimulKFK2Tile) -> Any:
        key = self._key(outgoing, tile)
        executable = TrimulKFK2CuTe._compiled_cache.get(key)
        force_cubin = self.force_cubin()
        if executable is not None and (not force_cubin or isinstance(executable, CuTeDSLKernelLibraryExecutable)):
            return executable
        if force_cubin:
            TrimulKFK2CuTe._compiled_cache.pop(key, None)
            return self._load_cubin_executable(key, outgoing, tile)
        try:
            source = load_source_module(__package__)
        except ImportError as source_error:
            return self._load_cubin_executable(key, outgoing, tile, source_error)

        if self._num_sms is None:
            self._num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        disk_key = ("trimul_kf_k2_cute_v1", *key, KERNEL_ABIS[self._sm_version])
        executable = self.load_from_cache(disk_key)
        if executable is None:
            logger.info(f"CuTeDSL trimul KF K2: compiling {tile} outgoing={outgoing} for SM{self._sm_version}")
            executable = source.compile_trimul_kf_k2_source(
                self.compile,
                tile.kernel_variant,
                outgoing,
                tile.tile_n,
                tile.cluster_m,
                tile.defer_kmin,
                tile.split_epi,
                self._num_sms,
            )
            self.save_to_cache(disk_key, executable)
        TrimulKFK2CuTe._compiled_cache[key] = executable
        return executable

    def run(self, a: torch.Tensor, b: torch.Tensor, outgoing: bool, tile: TrimulKFK2Tile) -> torch.Tensor:
        """Contract ``a`` and ``b`` ``[B, D, N, N]`` with ``tile``; the caller has validated the operands."""
        B, D, N, _ = a.shape
        executable = self._get_or_compile(outgoing, tile)
        prod = torch.empty_like(a)
        # A source-backed launch takes its stream from the current device, not the operands'.
        on_device = a.get_device() == torch.cuda.current_device()
        with contextlib.nullcontext() if on_device else torch.cuda.device(a.device):
            for b0, b1 in batch_chunks(B, D * N * N):
                launch_compiled_kernel(
                    executable,
                    a[b0:b1].view(-1),
                    b[b0:b1].view(-1),
                    prod[b0:b1].view(-1),
                    N,
                    (b1 - b0) * D,
                )
        return prod
