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
"""Vectorized A3M row decoding shared by the MSA featurizers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

_LOWER_A = ord("a")
_LOWER_Z = ord("z")


def code_points(text: str) -> np.ndarray:
    """Unicode code points of ``text``: ``uint8`` when ASCII, else ``uint32``."""
    if text.isascii():
        return np.frombuffer(text.encode("ascii"), dtype=np.uint8)
    return np.frombuffer(text.encode("utf-32-le", "surrogatepass"), dtype=np.uint32)


def map_code_points(
    codes: np.ndarray,
    table: np.ndarray,
    resolve: Callable[[str], int],
    missing: int = -1,
) -> np.ndarray:
    """Map code points through the ASCII lookup ``table``.

    Code points past the table, or whose entry is ``missing``, go through
    ``resolve(char)`` once per distinct code point, in first-seen order, so
    side effects of ``resolve`` (such as warnings) keep their sequence order.
    """
    if codes.dtype == np.uint8:
        values = table[codes]
    else:
        values = np.full(codes.shape, missing, dtype=table.dtype)
        inside = codes < len(table)
        values[inside] = table[codes[inside]]
    pending = values == missing
    if pending.any():
        unique, first, inverse = np.unique(codes[pending], return_index=True, return_inverse=True)
        resolved = np.empty(len(unique), dtype=table.dtype)
        for i in np.argsort(first):
            resolved[i] = resolve(chr(unique[i]))
        values[pending] = resolved[inverse]
    return values


def a3m_columns(rows: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split raw A3M rows into aligned columns and insertion counts.

    Lowercase characters (``str.islower``) are insertions; every other
    character is an aligned column.

    Args:
        rows: Raw A3M rows.

    Returns:
        ``(codes, deletions, counts)``. ``codes`` and ``deletions`` are flat
        over all aligned columns in row order: each column's code point and the
        number of insertions directly before it. ``counts[i]`` is the number of
        aligned columns in ``rows[i]``.
    """
    lengths = np.fromiter(map(len, rows), dtype=np.intp, count=len(rows))
    offsets = np.zeros(len(rows) + 1, dtype=np.intp)
    np.cumsum(lengths, out=offsets[1:])
    codes = code_points("".join(rows))
    if codes.dtype == np.uint8:
        insertion = (codes >= _LOWER_A) & (codes <= _LOWER_Z)
    else:
        unique, inverse = np.unique(codes, return_inverse=True)
        insertion = np.array([chr(c).islower() for c in unique], dtype=bool)[inverse]
    kept = np.flatnonzero(~insertion)
    first = np.searchsorted(kept, offsets)
    counts = np.diff(first)
    deletions = np.diff(kept, prepend=-1) - 1
    # A row's first column counts insertions from the row start.
    starts = first[:-1][counts > 0]
    deletions[starts] = kept[starts] - offsets[:-1][counts > 0]
    return codes[kept], deletions, counts


def ragged_mask(counts: np.ndarray, width: int) -> np.ndarray:
    """``[len(counts), width]`` mask selecting the first ``counts[i]`` columns of row ``i``."""
    return np.arange(width) < counts[:, None]


def truncate_rows(values: np.ndarray, counts: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Keep the first ``limits[i]`` of row ``i``'s ``counts[i]`` flat ``values``."""
    if np.array_equal(counts, limits):
        return values
    starts = np.cumsum(counts) - counts
    column = np.arange(len(values)) - np.repeat(starts, counts)
    return values[column < np.repeat(limits, counts)]
