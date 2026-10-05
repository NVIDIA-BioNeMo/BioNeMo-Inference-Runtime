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
"""Source-or-CUBIN interface for the SM90 TriMul KF K1."""

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

from ._config import KERNEL_ABIS, TrimulKFK1Selection, anchors, select
from ._cubin import TrimulKFK1CubinExecutable

__all__ = ["TrimulKFK1CuTe", "TrimulKFK1Output"]

#: Rows per K1 tile; a tile must not straddle two batches.
TILE_ROWS = 128
#: Most rows, and positions per a/b plane, one launch addresses: TMA coordinates are int32.
MAX_LAUNCH_ROWS = 2**31 - 1 - TILE_ROWS


class TrimulKFK1Output(NamedTuple):
    """K1's outputs: channel-major ``a`` and ``b`` ``[B, D, N, N]``, and the row statistics K1_2 hands to K3."""

    a: torch.Tensor
    b: torch.Tensor
    stats: torch.Tensor | None


@functools.lru_cache(maxsize=256)
def batch_chunks(B: int, N: int, pitch: int | None = None) -> tuple[tuple[int, int], ...]:
    """Batch ranges one launch covers: a 128-row tile must not straddle two batches, and a launch's
    rows and plane positions stay within :data:`MAX_LAUNCH_ROWS`."""
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


class TrimulKFK1CuTe(CuteKernelCache):
    """Cached backend for K1: ``a``/``b`` from the folded input projections of ``x``.

    Each call runs the variant its ``(C, D)`` configs tune for the anchor nearest the token count.

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

    def select(self, C: int, D: int, n: int) -> TrimulKFK1Selection | None:
        """The anchor and K1 variant a call over ``n`` tokens runs, or ``None`` when ``(C, D)`` does not ship."""
        return select(self._sm_version, C, D, n)

    def ships(self, C: int, D: int) -> bool:
        """Whether this build can run ``(C, D)``, from source or from a packaged CUBIN.

        A source-free build, or ``CUTEDSL_FORCE_CUBIN``, needs every packaged image the shape's
        configs name. Without them the caller keeps its own path.
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
        try:
            for kernel_variant in sorted(set(entries.values())):
                self._load_cubin_executable(self._key(C, D, kernel_variant), C, D, kernel_variant)
        except (CuTeDSLKernelLibraryError, RuntimeError):
            return False
        return True

    def _key(self, C: int, D: int, kernel_variant: str, ab_layout: str = "dense") -> tuple:
        key = (self._sm_version, C, D, kernel_variant)
        return key if ab_layout == "dense" else (*key, ab_layout)

    def supports(self, C: int, D: int, kernel_variant: str, ab_layout: str) -> bool:
        """Whether this build writes ``ab_layout`` with ``kernel_variant``."""
        if ab_layout == "dense":
            return True
        if not self.force_cubin():
            try:
                load_source_module(__package__)
            except ImportError:
                pass
            else:
                return True
        key = self._key(C, D, kernel_variant, ab_layout)
        try:
            self._load_cubin_executable(key, C, D, kernel_variant, ab_layout=ab_layout)
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
    ) -> Any:
        try:
            executable = populate_compiled_cache_from_library(
                TrimulKFK1CuTe._compiled_cache,
                key,
                "trimul_kf_k1",
                lambda library, launcher: TrimulKFK1CubinExecutable(
                    library, launcher, self._sm_version, C, D, kernel_variant, ab_layout
                ),
            )
        except CuTeDSLKernelLibraryError as library_error:
            if self.force_cubin():
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV}=1 forces the trimul KF K1 CUBIN path, but no CUBIN is available for "
                    f"SM{self._sm_version}, C={C}, D={D}, {kernel_variant}, {ab_layout} a/b"
                ) from library_error
            raise library_error from source_error
        logger.info(
            f"CuTeDSL trimul KF K1: using CUBIN kernel for SM{self._sm_version}, C={C}, D={D}, {kernel_variant}, "
            f"{ab_layout} a/b"
        )
        return executable

    def _get_or_compile(self, C: int, D: int, kernel_variant: str, ab_layout: str = "dense") -> Any:
        key = self._key(C, D, kernel_variant, ab_layout)
        executable = TrimulKFK1CuTe._compiled_cache.get(key)
        force_cubin = self.force_cubin()
        if executable is not None and (not force_cubin or isinstance(executable, CuTeDSLKernelLibraryExecutable)):
            return executable
        if force_cubin:
            TrimulKFK1CuTe._compiled_cache.pop(key, None)
            return self._load_cubin_executable(key, C, D, kernel_variant, ab_layout=ab_layout)
        try:
            source = load_source_module(__package__)
        except ImportError as source_error:
            return self._load_cubin_executable(key, C, D, kernel_variant, source_error, ab_layout)

        if self._num_sms is None:
            self._num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        disk_key = ("trimul_kf_k1_cute_v1", *key, KERNEL_ABIS[self._sm_version])
        executable = self.load_from_cache(disk_key)
        if executable is None:
            logger.info(
                f"CuTeDSL trimul KF K1: compiling {kernel_variant} ({ab_layout} a/b) for SM{self._sm_version}, "
                f"C={C}, D={D}"
            )
            executable = source.compile_trimul_kf_k1_source(
                self.compile, kernel_variant, C, D, self._num_sms, ab_layout
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
        executable = self._get_or_compile(C, D, selection.kernel_variant, selection.ab_layout)
        P = selection.ab_pitch(N)
        planes = torch.empty((2, B, D, P, P), dtype=x.dtype, device=x.device)
        a, b = planes[..., :N, :N].unbind(0)
        stats = (
            torch.empty((B, stats_rows(N), 2), dtype=torch.float32, device=x.device) if selection.writes_stats else None
        )
        seqlen = actual_seqlen.view(B * N)
        w_in = w_in.view(-1)
        w_gate_in = None if w_gate_in is None else w_gate_in.view(-1)
        # A source-backed launch takes its stream from the current device, not the operands'.
        on_device = x.get_device() == torch.cuda.current_device()
        with contextlib.nullcontext() if on_device else torch.cuda.device(x.device):
            for b0, b1 in batch_chunks(B, N, P):
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
