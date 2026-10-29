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
"""Bounded CPU distance binning for template features."""

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np

_DISTOGRAM_LOCK = Lock()
_DISTOGRAM_POOL: tuple[int, ThreadPoolExecutor] | None = None


def _run_distogram_batch(fill_rows: Callable[[tuple[int, int]], None], bounds: list[tuple[int, int]]) -> None:
    global _DISTOGRAM_POOL
    with _DISTOGRAM_LOCK:
        pid = os.getpid()
        if _DISTOGRAM_POOL is None or _DISTOGRAM_POOL[0] != pid:
            cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
            _DISTOGRAM_POOL = pid, ThreadPoolExecutor(max_workers=min(8, cores), thread_name_prefix="bioir-distogram")
        list(_DISTOGRAM_POOL[1].map(fill_rows, bounds))


def _reset_distogram_pool() -> None:
    """Reset inherited state only in the forked child.

    Parent locks and in-flight work remain unchanged. External C forks must
    invoke Python's at-fork hooks; other libraries retain their fork limits.
    """
    global _DISTOGRAM_LOCK, _DISTOGRAM_POOL
    _DISTOGRAM_LOCK = Lock()
    _DISTOGRAM_POOL = None


def _lock_distogram_pool() -> None:
    _DISTOGRAM_LOCK.acquire()


def _unlock_distogram_pool() -> None:
    _DISTOGRAM_LOCK.release()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_lock_distogram_pool,
        after_in_parent=_unlock_distogram_pool,
        after_in_child=_reset_distogram_pool,
    )


def masked_distogram(
    coords: np.ndarray,
    masks: np.ndarray,
    lower: np.ndarray,
    inf_value: float,
    pair_mask: np.ndarray | None = None,
    output: np.ndarray | None = None,
) -> np.ndarray:
    """Bin masked pair distances with strict edges.

    Args:
        coords: Floating-point coordinates, shaped ``[T, N, 3]``.
        masks: Binary atom presence, shaped ``[T, N]``.
        lower: Increasing squared lower edges, preserving their dtype.
        inf_value: Exclusive upper edge of the final bin.
        pair_mask: Optional binary pair mask, shaped ``[N, N]``.
        output: Optional float32 destination, shaped ``[T, N, N, bins]``.

    Returns:
        Float32 distance bins in the original template slots.
    """

    n_templ, n_tokens, _ = coords.shape
    n_bins = lower.size
    upper = np.concatenate([lower[1:], np.array([inf_value], dtype=lower.dtype)])
    if output is None:
        output = np.zeros((n_templ, n_tokens, n_tokens, n_bins), dtype=np.float32)
    else:
        output.fill(0)
    active = np.flatnonzero(masks.any(axis=1))
    if not active.size:
        return output
    coords, masks = coords[active], masks[active]

    def fill_rows(bounds: tuple[int, int]) -> None:
        start, end = bounds
        allowed = (masks[:, start:end, None] != 0) & (masks[:, None, :] != 0)
        if pair_mask is not None:
            allowed &= pair_mask[start:end] != 0
        ti, ii, ji = np.nonzero(allowed)
        squares = [(coords[ti, ii + start, k] - coords[ti, ji, k]) ** 2 for k in range(3)]
        distances = squares[0] + squares[1] + squares[2]
        del squares
        indices = np.searchsorted(lower, distances, side="left") - 1
        valid = (indices >= 0) & (distances < upper[np.clip(indices, 0, n_bins - 1)])
        output[active[ti[valid]], ii[valid] + start, ji[valid], indices[valid]] = 1.0

    workers = min(8, max(1, n_tokens // 64))
    edges = np.linspace(0, n_tokens, workers + 1, dtype=int)
    bounds = [(int(start), int(end)) for start, end in zip(edges[:-1], edges[1:], strict=True)]
    _run_distogram_batch(fill_rows, bounds)
    return output
