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
"""Format whole columns of writer tokens as ASCII bytes in NumPy.

A column is a ``uint8`` matrix with one row per token plus each token's length.
A row spells its token in its nonzero bytes, read left to right; zero bytes are
padding and may sit anywhere in the row. Joining rows therefore needs only the
nonzero bytes, which lets every token in a column share one fixed width.
"""

import functools
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

# ``ihm`` wraps CIF loop rows at this many columns; 0 turns wrapping off.
CIF_LINE_LENGTH = 80

_ZERO = ord("0")
# Numbers are spelled four digits at a time, from tables of every four-digit group
# stored as one ``uint32`` cell of four characters each.
_GROUP_DIGITS = 4
_GROUP = 10**_GROUP_DIGITS
_CELL_BYTES = 4
# :func:`grid_tokens` spells a value's fraction as one group of digits.
GRID_DECIMALS = _GROUP_DIGITS
# Every int64 has at most 19 decimal digits.
_POW10 = np.array([10**k for k in range(19)], dtype=np.int64)

# Every float32 in [2**-13, 2**24) prints in positional notation with at most
# eight integer and twelve fractional digits, and scales exactly in float64 by
# the powers of ten its shortest digits need. Other values use ``str``.
_FAST_MIN = 2.0**-13
_FAST_MAX = 2.0**24
_FRACTION_DIGITS = 12
_FLOAT32_SIGNIFICAND_BITS = 24
# Nine significant digits round-trip every float32.
_FLOAT32_MAX_DIGITS = 9
# The scales the fast path multiplies by: 10**0 to 10**_FRACTION_DIGITS.
_POW10_FLOAT = np.array([10.0**k for k in range(_FRACTION_DIGITS + 1)])
# floor(log10(a)) of a fast-path value lies in [-4, 7]; _DECADES[i] is 10**(i + _MIN_DECADE).
_MIN_DECADE = -4
_DECADES = np.array([10.0**k for k in range(_MIN_DECADE, 8)])
_LOG10_2 = np.log10(2.0)
# A float32 token holds a sign cell, two integer cells, a point cell and three fraction cells.
FLOAT32_WIDTH = 7 * _CELL_BYTES
# Rows of a CIF loop joined at once, which bounds the temporaries.
_ROW_CHUNK = 16384

# A grid token fills at most seven bytes of a ``uint64`` cell; the top byte holds its length.
_TOKEN_BYTES = 7
_LENGTH_SHIFT = 8 * _TOKEN_BYTES
_TOKEN_MASK = (1 << _LENGTH_SHIFT) - 1
# JSON text is written in overlapping eight-byte windows, one ``uint64`` each.
_WINDOW_BYTES = 8
_WINDOW_BITS = 8 * _WINDOW_BYTES
_COMMA_SPACE = int.from_bytes(b", ", "little")
# Tokens spelled at once, which keeps the temporaries in cache.
_JSON_CHUNK = 65536


class _Spellings(NamedTuple):
    """Spellings of every four-digit group as ``uint32`` cells, with their lengths."""

    cells: np.ndarray
    lengths: np.ndarray


def _group_table(keep: Callable[[str], str]) -> _Spellings:
    """Spell every four-digit group; ``keep`` maps its zero-padded digits to the characters that stay."""
    kept = [keep(f"{value:0{_GROUP_DIGITS}d}").encode() for value in range(_GROUP)]
    return _Spellings(
        np.array(kept, dtype=f"S{_CELL_BYTES}").view(np.uint32), np.array([len(k) for k in kept], dtype=np.int64)
    )


# Groups spelled whole, without leading zeros (the units group keeps its last
# digit), or without trailing zeros (the tenths group keeps its first digit).
# Empty groups are all padding.
_FULL = _group_table(lambda digits: digits)
_NO_LEADING = _group_table(lambda digits: digits.lstrip("0"))
_UNITS = _group_table(lambda digits: digits.lstrip("0") or "0")
_NO_TRAILING = _group_table(lambda digits: digits.rstrip("0"))
_TENTHS = _group_table(lambda digits: digits.rstrip("0") or "0")
_MINUS = np.frombuffer(b"-\0\0\0", dtype=np.uint32)[0]
_POINT = np.frombuffer(b".\0\0\0", dtype=np.uint32)[0]


def _spell_where(
    condition: np.ndarray, group: np.ndarray, if_true: _Spellings, if_false: _Spellings
) -> tuple[np.ndarray, np.ndarray]:
    """Spell each ``group`` with ``if_true`` where ``condition`` holds, else with ``if_false``."""
    return (
        np.where(condition, if_true.cells[group], if_false.cells[group]),
        np.where(condition, if_true.lengths[group], if_false.lengths[group]),
    )


def _shortest_float32_digits(magnitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the shortest round-trip digits of float32 magnitudes.

    Each magnitude ``a`` in ``[2**-13, 2**24)`` equals ``M * 2**E`` for a 24-bit
    ``M``. For ``p`` significant digits, ``a * 10**k`` with
    ``k = p - 1 - floor(log10(a))`` is exact in float64 because ``M * 5**k``
    stays below ``2**53``; so are its distances to the neighbouring integers and
    the half gaps to the adjacent float32 values. The smallest ``p`` at which a
    neighbour lies strictly inside the rounding interval is the shortest
    length, found by bisection because a longer length always also fits.
    NumPy's Dragon4 keeps the closer neighbour and breaks ties to the even
    digit. Lengths that need ``k < 0`` are skipped: below ``2**24`` such a
    candidate equals ``a`` itself and prints the same as ``k = 0``.

    Args:
        magnitude: float64 copies of float32 magnitudes, shaped ``[n]``.

    Returns:
        Digits ``d`` and scale ``k`` with ``a`` printed as ``d / 10**k``.
    """
    mantissa, exponent = np.frexp(magnitude)
    # Half the gap to the next float32 above, and to the next below, which is
    # half as far at a power of two. Scaling either by 10**k stays exact.
    high_half_gap = np.ldexp(1.0, exponent - _FLOAT32_SIGNIFICAND_BITS - 1)
    low_half_gap = np.where(mantissa == 0.5, 0.5 * high_half_gap, high_half_gap)
    # floor(log10(a)) is the estimate from the binary exponent or one more.
    log10 = np.floor((exponent - 1) * _LOG10_2).astype(np.int64)
    log10 += magnitude >= _DECADES[log10 + 1 - _MIN_DECADE]

    def neighbours(precision: np.ndarray) -> tuple[np.ndarray, ...]:
        k = precision - 1 - log10
        usable = k >= 0
        k = np.maximum(k, 0)
        power = _POW10_FLOAT[k]
        scaled = magnitude * power
        below = np.floor(scaled)
        to_below = scaled - below
        to_above = (below + 1) - scaled
        low_ok = usable & (to_below < power * low_half_gap)
        high_ok = usable & (to_above < power * high_half_gap)
        return k, below, to_below, to_above, low_ok, high_ok

    low = np.ones(magnitude.shape, dtype=np.int64)
    high = np.full(magnitude.shape, _FLOAT32_MAX_DIGITS, dtype=np.int64)
    while (low < high).any():
        middle = (low + high) // 2
        *_, low_ok, high_ok = neighbours(middle)
        fits = low_ok | high_ok
        high = np.where(fits, middle, high)
        low = np.where(fits, low, np.maximum(middle + 1, low))
    k, below, to_below, to_above, low_ok, high_ok = neighbours(high)
    if not (low_ok | high_ok).all():
        raise RuntimeError("float32 digit search did not converge")
    odd = (below.astype(np.int64) & 1) == 1
    up = high_ok & (~low_ok | (to_above < to_below) | ((to_above == to_below) & odd))
    return (below + up).astype(np.int64), k


def _decimal_text(digits: np.ndarray, scale: np.ndarray, negative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spell ``digits / 10**scale`` in positional form, keeping one fractional digit.

    Args:
        digits: Non-negative integers whose integer part ``digits // 10**scale``
            stays below ``10**8``, shaped ``[n]``.
        scale: Fractional digit counts in ``[0, 12]``, shaped ``[n]``.
        negative: Whether to prefix a minus sign, shaped ``[n]``.

    Returns:
        Characters shaped ``[n, FLOAT32_WIDTH]`` (sign, integer digits, the
        point, fraction digits) and lengths shaped ``[n]``.
    """
    # Cells: 0 sign, 1-2 integer groups, 3 point, 4-6 fraction groups.
    cells = np.empty((digits.size, FLOAT32_WIDTH // _CELL_BYTES), dtype=np.uint32)
    cells[:, 0] = np.where(negative, _MINUS, 0)
    whole, fraction = np.divmod(digits, _POW10[scale])
    high, low = np.divmod(whole, _GROUP)
    cells[:, 1], length = _NO_LEADING.cells[high], _NO_LEADING.lengths[high] + negative
    cells[:, 2], part = _spell_where(high > 0, low, _FULL, _UNITS)
    length += part + 1
    cells[:, 3] = _POINT
    # Left-align the fraction in twelve digits; its trailing zeros are dropped,
    # but "x.0" keeps one digit.
    fraction *= _POW10[_FRACTION_DIGITS - scale]
    rest, last = np.divmod(fraction, _GROUP)
    first, middle = np.divmod(rest, _GROUP)
    cells[:, 6], part = _NO_TRAILING.cells[last], _NO_TRAILING.lengths[last]
    length += part
    cells[:, 5], part = _spell_where(last > 0, middle, _FULL, _NO_TRAILING)
    length += part
    cells[:, 4], part = _spell_where((middle > 0) | (last > 0), first, _FULL, _TENTHS)
    return cells.view(np.uint8), length + part


def float32_text(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``str(np.float32(v))`` for every value as a text column.

    Args:
        values: float32 values, any shape; flattened in C order.

    Returns:
        Characters shaped ``[n, FLOAT32_WIDTH]`` and lengths shaped ``[n]``.
    """
    values = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    with np.errstate(invalid="ignore"):  # Signaling NaNs keep their ``str``.
        magnitude = np.abs(values).astype(np.float64)
    fast = (magnitude >= _FAST_MIN) & (magnitude < _FAST_MAX)
    if fast.all():
        return _decimal_text(*_shortest_float32_digits(magnitude), np.signbit(values))
    chars = np.zeros((values.size, FLOAT32_WIDTH), dtype=np.uint8)
    lengths = np.zeros(values.size, dtype=np.int64)
    index = np.flatnonzero(fast)
    if index.size:
        chars[index], lengths[index] = _decimal_text(
            *_shortest_float32_digits(magnitude[index]), np.signbit(values[index])
        )
    # NumPy's float-to-bytes cast spells each value as ``str`` does.
    index = np.flatnonzero(~fast)
    tokens = values[index].astype(f"S{FLOAT32_WIDTH}")
    chars[index] = tokens.view(np.uint8).reshape(index.size, FLOAT32_WIDTH)
    lengths[index] = np.char.str_len(tokens)
    return chars, lengths


def string_text(tokens: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """Return tokens as a text column, or ``None`` if one is not plain ASCII.

    Args:
        tokens: Tokens to encode.

    Returns:
        Characters shaped ``[n, max length]`` and lengths shaped ``[n]``, or
        ``None`` when a token holds a non-ASCII character or a NUL byte, whose
        byte count or padding role would differ from its character count.
    """
    try:
        encoded = [token.encode("ascii") for token in tokens]
    except UnicodeEncodeError:
        return None
    if any(b"\0" in token for token in encoded):
        return None
    width = max(map(len, encoded), default=0) or 1
    chars = np.array(encoded, dtype=f"S{width}").view(np.uint8).reshape(len(encoded), width)
    return chars, np.fromiter(map(len, encoded), dtype=np.int64, count=len(encoded))


def uint_text(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return non-negative integers as a text column.

    Args:
        values: Non-negative integers, shaped ``[n]``.

    Returns:
        Right-aligned digits and lengths, both with ``n`` rows.
    """
    values = np.asarray(values, dtype=np.int64)
    width = len(str(int(values.max()))) if values.size else 1
    chars = np.zeros((values.size, width), dtype=np.uint8)
    for column in range(width):
        place = _POW10[width - 1 - column]
        keep = (values >= place) | (column == width - 1)
        chars[:, column] = np.where(keep, (values // place) % 10 + _ZERO, 0)
    return chars, np.count_nonzero(chars, axis=1)


def wrapped_rows(
    fields: list[tuple[np.ndarray, np.ndarray | int]], rows: int, line_length: int = CIF_LINE_LENGTH
) -> bytes:
    """Join text columns into CIF loop rows, wrapping lines as ``ihm`` does.

    A token goes on the current line after one space unless that would pass
    ``line_length``; then it starts the next line. Every row ends in a newline.

    Args:
        fields: Per key, characters shaped ``[rows, w]`` or ``[1, w]`` and
            lengths shaped ``[rows]`` or a single length for a shared token.
        rows: Number of rows.
        line_length: Longest line before wrapping; 0 keeps every row on one line.

    Returns:
        The rows as ASCII bytes.
    """
    # Byte columns that are padding in every row only cost copying; drop them.
    fields = [
        (chars, lengths) if chars.shape[0] == 1 else (chars[:, np.bitwise_or.reduce(chars, axis=0) != 0], lengths)
        for chars, lengths in fields
    ]
    out = []
    separators = np.array([0, ord(" "), ord("\n")], dtype=np.uint8)
    for start in range(0, rows, _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, rows)
        size = stop - start
        parts = []
        column = np.zeros(size, dtype=np.int64)
        for j, (chars, lengths) in enumerate(fields):
            shared = chars.shape[0] == 1
            length = lengths if isinstance(lengths, int) else lengths[start:stop]
            if j:
                after = column > 0
                wrap = after & (column + length + 1 > line_length) if line_length else np.zeros_like(after)
                parts.append(separators[after.view(np.int8) + wrap.view(np.int8)][:, None])
                column = np.where(after & ~wrap, column + 1 + length, length)
            else:
                column = column + length
            parts.append(np.broadcast_to(chars, (size, chars.shape[1])) if shared else chars[start:stop])
        parts.append(np.full((size, 1), ord("\n"), dtype=np.uint8))
        block = np.concatenate(parts, axis=1)
        out.append(block[block != 0].tobytes())
    return b"".join(out)


@functools.cache
def grid_tokens(whole_max: int) -> np.ndarray:
    """Spell ``i / 10**GRID_DECIMALS`` for ``i`` in ``[0, whole_max * 10**GRID_DECIMALS]`` as ``repr`` does.

    The table is built once per ``whole_max`` and shared read-only.

    Args:
        whole_max: Largest integer part, below 100.

    Returns:
        ``uint64`` cells holding each token's at most seven little-endian bytes,
        with its length in the top byte.
    """
    index = np.arange(whole_max * _GROUP + 1)
    whole, fraction = np.divmod(index, _GROUP)
    whole_length = _UNITS.lengths[whole].astype(np.uint64)
    cells = _UNITS.cells[whole].astype(np.uint64)
    cells |= np.uint64(ord(".")) << (8 * whole_length)
    cells |= _TENTHS.cells[fraction].astype(np.uint64) << (8 * (whole_length + 1))
    cells |= (whole_length + 1 + _TENTHS.lengths[fraction].astype(np.uint64)) << np.uint64(_LENGTH_SHIFT)
    cells.setflags(write=False)
    return cells


def json_grid_text(indices: np.ndarray, cells: np.ndarray) -> str | None:
    """Return ``json.dumps`` of an array whose elements are spelled by ``cells[indices]``.

    Every byte is written by a window of eight known output bytes, so writes
    that overlap agree and their order does not matter: a token, its separator
    and the start of the next token. Separators that windows miss are spaces,
    which the buffer holds before the writes; row boundaries are written last.

    Args:
        indices: Integer array, every dimension positive.
        cells: Tokens of 3 to 7 ASCII bytes from :func:`grid_tokens`.

    Returns:
        The JSON text, or ``None`` if a row boundary does not fit eight bytes.
    """
    shape = indices.shape
    flat = indices.reshape(-1)
    width = shape[-1]
    rows = flat.size // width
    lead = shape[:-1]
    # Opening brackets before each row and closing brackets after it.
    opens = np.ones(rows, dtype=np.int64)
    closes = np.ones(rows, dtype=np.int64)
    first = np.ones(rows, dtype=bool)
    last = np.ones(rows, dtype=bool)
    for axis in range(len(lead) - 1, -1, -1):
        index = np.arange(rows) // int(np.prod(lead[axis + 1 :], dtype=np.int64)) % lead[axis]
        first &= index == 0
        last &= index == lead[axis] - 1
        opens += first
        closes += last
    boundaries = [b"]" * c + b", " + b"[" * o for c, o in zip(closes[:-1].tolist(), opens[1:].tolist(), strict=True)]
    if any(len(b) > _WINDOW_BYTES for b in boundaries):
        return None
    boundary_cells = np.array([int.from_bytes(b, "little") for b in boundaries] + [0], dtype=np.uint64)
    boundary_lengths = np.array([len(b) for b in boundaries] + [0], dtype=np.int64)
    head, tail = int(opens[0]), int(closes[-1])
    # Tokens take at most seven bytes; the text is cut to its length at the end.
    bound = head + _TOKEN_BYTES * flat.size + 2 * (flat.size - rows) + int(boundary_lengths.sum()) + tail
    buffer = np.full(bound + _WINDOW_BYTES, ord(" "), dtype=np.uint8)
    windows = np.ndarray((bound + 1,), dtype=np.uint64, buffer=buffer, strides=(1,))
    base = head
    row_end_starts = np.empty(rows, dtype=np.int64)
    for start in range(0, flat.size, _JSON_CHUNK):
        stop = min(start + _JSON_CHUNK, flat.size)
        size = stop - start
        token = cells[flat[start : stop + 1]]
        length = (token[:size] >> np.uint64(_LENGTH_SHIFT)).astype(np.int64)
        token &= np.uint64(_TOKEN_MASK)
        separator = np.full(size, _COMMA_SPACE, dtype=np.uint64)
        separator_length = np.full(size, 2, dtype=np.int64)
        row = np.arange((width - 1 - start) % width, size, width)
        row_index = (start + row) // width
        separator[row] = boundary_cells[row_index]
        separator_length[row] = boundary_lengths[row_index]
        step = length + separator_length
        offsets = np.cumsum(step)
        starts = offsets - step + base
        base += int(offsets[-1])
        row_end_starts[row_index] = starts[row] + length[row]
        bits = (8 * length).astype(np.uint64)
        window = token[:size] | (separator << bits)
        bits += (8 * separator_length).astype(np.uint64)
        following = np.zeros(size, dtype=np.uint64)
        following[: token.size - 1] = token[1:]
        window |= np.where(
            bits < _WINDOW_BITS, following << np.minimum(bits, np.uint64(_WINDOW_BITS - 1)), np.uint64(0)
        )
        windows[starts] = window
    total = base + tail
    if rows > 1:
        boundary_bytes = boundary_cells[:-1, None] >> (8 * np.arange(_WINDOW_BYTES, dtype=np.uint64))
        written = np.arange(_WINDOW_BYTES) < boundary_lengths[:-1, None]
        positions = row_end_starts[:-1, None] + np.arange(_WINDOW_BYTES)
        buffer[positions[written]] = (boundary_bytes[written] & np.uint64(0xFF)).astype(np.uint8)
    buffer[:head] = ord("[")
    buffer[total - tail : total] = ord("]")
    return str(buffer.data[:total], "ascii")
