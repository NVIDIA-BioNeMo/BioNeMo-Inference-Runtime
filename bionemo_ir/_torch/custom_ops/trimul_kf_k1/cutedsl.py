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
"""Source-or-CUBIN interface for the TriMul KF K1: SM90, SM100 and SM103 from source or packaged CUBINs."""

from __future__ import annotations

import contextlib
import functools
from typing import Any, NamedTuple

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

from ._config import (
    CENTRED_VARIANTS,
    KERNEL_ABIS,
    TrimulKFK1Selection,
    TrimulKFK1Tile,
    anchors,
    layout_rules,
    select,
    tiles,
)
from ._cubin import TrimulKFK1CubinExecutable, TrimulKFK1Sm100CubinExecutable

__all__ = ["TrimulKFK1CuTe", "TrimulKFK1Output"]

#: Rows per K1 tile; a tile never straddles two batches.
TILE_ROWS = 128
#: Most rows, and positions per a/b plane, one launch addresses: TMA coordinates are int32.
MAX_LAUNCH_ROWS = 2**31 - 1 - TILE_ROWS
#: Kernel ABIs the packaged CUBIN library carries; the others run from source only.
CUBIN_ABIS = frozenset({"sm90", "sm100"})
#: Bumped when the compile contract of a source kernel changes outside the hashed kernel sources.
_DISK_CACHE_VERSION = "trimul_kf_k1_cute_v1"


class TrimulKFK1Output(NamedTuple):
    """K1's outputs: channel-major ``a`` and ``b`` ``[B, D, N, N]``, and the row statistics K1_2 hands to K3."""

    a: torch.Tensor
    b: torch.Tensor
    stats: torch.Tensor | None


@functools.lru_cache(maxsize=256)
def batch_chunks(B: int, N: int, pitch: int | None = None) -> tuple[tuple[int, int], ...]:
    """Batch ranges one launch of a kernel that tiles the launch's rows flat covers: a 128-row tile must not
    straddle two batches, and a launch's rows and plane positions stay within :data:`MAX_LAUNCH_ROWS`."""
    if N * max(N, pitch or N) > MAX_LAUNCH_ROWS:
        raise ValueError(f"trimul KF supports N * P <= {MAX_LAUNCH_ROWS} plane positions, got N={N}, P={pitch}")
    if B > 1 and (N * N) % TILE_ROWS:
        return tuple((b, b + 1) for b in range(B))
    max_batch = MAX_LAUNCH_ROWS // (N * N)
    chunks = -(-B // max_batch)
    step = -(-B // chunks)
    return tuple((b, min(B, b + step)) for b in range(0, B, step))


def stats_rows(N: int) -> int:
    """Rows of row statistics one batch holds: ``N * N`` rounded up to whole 128-row tiles."""
    return -(-N * N // TILE_ROWS) * TILE_ROWS


@functools.lru_cache(maxsize=256)
def launch_chunks(B: int, N: int, pitch: int | None = None) -> tuple[tuple[int, int], ...]:
    """Batch ranges one K1 launch covers. K1 tiles each batch on its own, its last tile running past the
    batch's rows, so batches split only where a launch's tile rows (:func:`stats_rows` a batch) or plane
    positions would pass :data:`MAX_LAUNCH_ROWS`."""
    batch_chunks(1, N, pitch)  # rejects planes past int32 coordinates
    max_batch = MAX_LAUNCH_ROWS // stats_rows(N)
    chunks = -(-B // max_batch)
    step = -(-B // chunks)
    return tuple((b, min(B, b + step)) for b in range(0, B, step))


class TrimulKFK1CuTe(CuteKernelCache):
    """Cached backend for K1: ``a``/``b`` from the folded input projections of ``x``.

    Each call runs the variant its ``(C, D)`` configs tune for the anchor nearest the token count,
    compiled from source or loaded from the packaged CUBINs.

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

    @property
    def kernel_abi(self) -> str | None:
        """The kernel ABI this SM runs, or ``None`` when K1 ships none for it."""
        return KERNEL_ABIS.get(self._sm_version)

    @property
    def chunks_batches(self) -> bool:
        """Whether :meth:`run` spreads a call's batches over launches (SM90); SM100 launches once for every batch."""
        return self.kernel_abi == "sm90"

    def select(self, C: int, D: int, n: int) -> TrimulKFK1Selection | None:
        """The anchor and K1 variant a call over ``n`` tokens runs, or ``None`` when ``(C, D)`` does not ship."""
        return select(self._sm_version, C, D, n)

    def centred(self, C: int, D: int) -> bool:
        """Whether a variant ``(C, D)``'s configs select drops the LayerNorm mean term and reads a centred fold."""
        return any(variant in CENTRED_VARIANTS for variant in (anchors(self._sm_version, C, D) or {}).values())

    def _has_cubins(self) -> bool:
        """Whether the packaged library can carry this SM's kernels at all."""
        return self.kernel_abi in CUBIN_ABIS

    def ships(self, C: int, D: int) -> bool:
        """Whether this build can run ``(C, D)``, from source or from a packaged CUBIN.

        A source-free build, or ``CUTEDSL_FORCE_CUBIN``, needs every packaged image the shape's
        configs name: each anchor's variant and, on SM100 and SM103, its tile in every a/b layout its
        rule writes. Without them the caller keeps its own path.
        """
        entries = anchors(self._sm_version, C, D)
        if not entries:
            return False
        if not self.force_cubin():
            try:
                load_source_module(__package__)
            except ImportError:
                pass
            else:
                return True
        if not self._has_cubins():
            return False
        try:
            if self.kernel_abi == "sm90":
                for kernel_variant in sorted(set(entries.values())):
                    self._load_cubin_executable(self._key(C, D, kernel_variant), C, D, kernel_variant)
                return True
            rules, anchor_tiles = layout_rules(self._sm_version, C, D), tiles(self._sm_version, C, D)
            for bucket, kernel_variant in entries.items():
                tile = anchor_tiles[bucket]
                for ab_layout in ("dense",) if rules[bucket] == "dense" else ("dense", "padded"):
                    key = self._key(C, D, kernel_variant, ab_layout, tile)
                    self._load_cubin_executable(key, C, D, kernel_variant, ab_layout=ab_layout, tile=tile)
        except (CuTeDSLKernelLibraryError, RuntimeError):
            return False
        return True

    def _key(
        self, C: int, D: int, kernel_variant: str, ab_layout: str = "dense", tile: TrimulKFK1Tile | None = None
    ) -> tuple:
        key = (self._sm_version, C, D, kernel_variant)
        if ab_layout != "dense":
            key = (*key, ab_layout)
        return key if tile is None else (*key, *tile)

    def supports(self, C: int, D: int, kernel_variant: str, ab_layout: str, tile: TrimulKFK1Tile | None = None) -> bool:
        """Whether this build writes ``ab_layout`` with ``kernel_variant`` (and, for ``K1_B0``, ``tile``)."""
        if ab_layout == "dense":
            return True
        if not self.force_cubin():
            try:
                load_source_module(__package__)
            except ImportError:
                pass
            else:
                return True
        if not self._has_cubins():
            return False
        key = self._key(C, D, kernel_variant, ab_layout, tile)
        try:
            self._load_cubin_executable(key, C, D, kernel_variant, ab_layout=ab_layout, tile=tile)
        except (CuTeDSLKernelLibraryError, RuntimeError):
            return False
        return True

    def _load_cubin_executable(
        self,
        key: tuple,
        C: int,
        D: int,
        kernel_variant: str,
        source_error: Exception | None = None,
        ab_layout: str = "dense",
        tile: TrimulKFK1Tile | None = None,
    ) -> Any:
        if self.kernel_abi == "sm90":

            def make(library: Any, launcher: Any) -> Any:
                return TrimulKFK1CubinExecutable(library, launcher, self._sm_version, C, D, kernel_variant, ab_layout)
        else:

            def make(library: Any, launcher: Any) -> Any:
                return TrimulKFK1Sm100CubinExecutable(
                    library, launcher, self._sm_version, C, D, kernel_variant, ab_layout, tile
                )

        if not isinstance(TrimulKFK1CuTe._compiled_cache.get(key), CuTeDSLKernelLibraryExecutable):
            # A source build cached under the key would pass for the CUBIN; only the library proves that one ships.
            TrimulKFK1CuTe._compiled_cache.pop(key, None)
        tile_text = "" if tile is None else f", {tile}"
        try:
            executable = populate_compiled_cache_from_library(TrimulKFK1CuTe._compiled_cache, key, "trimul_kf_k1", make)
        except CuTeDSLKernelLibraryError as library_error:
            if self.force_cubin():
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV}=1 forces the trimul KF K1 CUBIN path, but no CUBIN is available for "
                    f"SM{self._sm_version}, C={C}, D={D}, {kernel_variant}, {ab_layout} a/b{tile_text}"
                ) from library_error
            raise library_error from source_error
        logger.info(
            f"CuTeDSL trimul KF K1: using CUBIN kernel for SM{self._sm_version}, C={C}, D={D}, {kernel_variant}, "
            f"{ab_layout} a/b{tile_text}"
        )
        return executable

    def _get_or_compile(
        self, C: int, D: int, kernel_variant: str, ab_layout: str = "dense", tile: TrimulKFK1Tile | None = None
    ) -> Any:
        key = self._key(C, D, kernel_variant, ab_layout, tile)
        executable = TrimulKFK1CuTe._compiled_cache.get(key)
        force_cubin = self.force_cubin()
        if executable is not None and (not force_cubin or isinstance(executable, CuTeDSLKernelLibraryExecutable)):
            return executable
        if force_cubin:
            TrimulKFK1CuTe._compiled_cache.pop(key, None)
            return self._load_cubin_executable(key, C, D, kernel_variant, ab_layout=ab_layout, tile=tile)
        try:
            source = load_source_module(__package__)
        except ImportError as source_error:
            return self._load_cubin_executable(key, C, D, kernel_variant, source_error, ab_layout, tile)

        if self._num_sms is None:
            self._num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        # The key carries the ABI and every compile-time axis (an SM100 tile included); the kernel
        # sources are hashed into the cache directory, the host-side compile contract is the version.
        disk_key = (_DISK_CACHE_VERSION, *key, self.kernel_abi)
        executable = self.load_from_cache(disk_key)
        if executable is None:
            logger.info(
                f"CuTeDSL trimul KF K1: compiling {kernel_variant} ({ab_layout} a/b) for SM{self._sm_version}, "
                f"C={C}, D={D}{'' if tile is None else f', {tile}'}"
            )
            executable = source.compile_trimul_kf_k1_source(
                self.compile, kernel_variant, C, D, self._num_sms, ab_layout, tile
            )
            self.save_to_cache(disk_key, executable)
        TrimulKFK1CuTe._compiled_cache[key] = executable
        return executable

    def run(
        self,
        x: torch.Tensor,
        actual_seqlen: torch.Tensor,
        w_in: torch.Tensor,
        w_gate_in: torch.Tensor | None,
        vec_in: torch.Tensor,
        eps: float,
        selection: TrimulKFK1Selection,
    ) -> TrimulKFK1Output:
        """Run ``selection``'s variant on ``x`` ``[B, N, N, C]``; the caller has validated the operands."""
        B, N, _, C = x.shape
        D = vec_in.shape[0] // 8
        executable = self._get_or_compile(C, D, selection.kernel_variant, selection.ab_layout, selection.tile)
        P = selection.ab_pitch(N)
        planes = torch.empty((2, B, D, P, P), dtype=x.dtype, device=x.device)
        a, b = planes[..., :N, :N].unbind(0)
        stats = (
            torch.empty((B, stats_rows(N), 2), dtype=torch.float32, device=x.device) if selection.writes_stats else None
        )
        seqlen = actual_seqlen.view(B * N)
        w_in = w_in.view(-1)
        w_gate_in = None if w_gate_in is None else w_gate_in.view(-1)
        # Images built before K1 tiled each batch on its own tile the launch's rows flat.
        flat_tiles = isinstance(executable, TrimulKFK1CubinExecutable) and executable.flat_tiles
        # A source-backed launch takes its stream from the current device, not the operands'.
        on_device = x.get_device() == torch.cuda.current_device()
        if not self.chunks_batches:
            # One launch for every batch: the tcgen05 kernels tile each batch on its own, and the
            # op accepts no more than MAX_LAUNCH_ROWS positions in all, so int32 tile counts hold.
            # K1_B0 addresses both plane sets from one buffer, K1_B1 takes a and b apart.
            plane_args = (
                (planes.view(-1),) if selection.kernel_variant == "K1_B0" else (planes[0].view(-1), planes[1].view(-1))
            )
            with contextlib.nullcontext() if on_device else torch.cuda.device(x.device):
                launch_compiled_kernel(executable, x.view(-1), seqlen, w_in, w_gate_in, vec_in, *plane_args, N, B, eps)
            return TrimulKFK1Output(a, b, stats)
        with contextlib.nullcontext() if on_device else torch.cuda.device(x.device):
            for b0, b1 in (batch_chunks if flat_tiles else launch_chunks)(B, N, P):
                nb = b1 - b0
                launch_compiled_kernel(
                    executable,
                    x[b0:b1].view(-1),
                    seqlen[b0 * N : b1 * N],
                    w_in,
                    w_gate_in,
                    vec_in,
                    planes[0, b0:b1].view(-1),
                    planes[1, b0:b1].view(-1),
                    None if stats is None else stats[b0:b1].view(-1),
                    nb * N * N,
                    N,
                    nb,
                    eps,
                )
        return TrimulKFK1Output(a, b, stats)
