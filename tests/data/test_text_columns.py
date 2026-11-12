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
"""Whole-column text spells every token exactly as the per-value formatters do."""

import io
import json

import ihm.format
import numpy as np
import pytest

from bionemo_ir.data.writers import _text

# Every token that :func:`_text.grid_tokens` spells for the score grid in [0, 32].
_GRID_SCALE = 10**_text.GRID_DECIMALS
_GRID_SIZE = 32 * _GRID_SCALE + 1


def _tokens(chars: np.ndarray, lengths: np.ndarray) -> list[str]:
    tokens = [row[row != 0].tobytes().decode("ascii") for row in chars]
    assert [len(token) for token in tokens] == lengths.tolist()
    return tokens


def _float32_edges() -> np.ndarray:
    """Values at the fast range's edges, powers of two and ten, ties and specials."""
    edges = [0.0, -0.0, np.nan, np.inf, -np.inf, 1e-4, 1e16, 1e20, 3.4028235e38, 1e-45, 1.1754944e-38]
    for k in range(-14, 26):
        edges += [2.0**k, np.nextafter(np.float32(2.0**k), np.float32(0)), np.nextafter(np.float32(2.0**k), np.inf)]
    for k in range(-5, 9):
        edges += [10.0**k, np.nextafter(np.float32(10.0**k), np.float32(0)), np.nextafter(np.float32(10.0**k), np.inf)]
    # Halfway cases whose two shortest neighbours are equally close.
    edges += [2110319.75, 280056.375, 0.5, 0.25, 12.25, 1234.5, 8388607.5]
    values = np.array(edges, dtype=np.float32)
    return np.concatenate([values, -values])


def test_float32_text_matches_str() -> None:
    bits = np.random.default_rng(0).integers(0, 2**32, 200_000, dtype=np.uint64).astype(np.uint32)
    # Bias half of the sample to the magnitudes that coordinates and B factors take.
    bits[::2] = np.random.default_rng(1).integers(0x38800000, 0x4B800000, bits[::2].size, dtype=np.uint32)
    values = np.concatenate([bits.view(np.float32), _float32_edges()])

    assert _tokens(*_text.float32_text(values)) == [str(value) for value in values]


def test_float32_text_matches_str_on_fast_path_only() -> None:
    values = np.random.default_rng(2).uniform(-300, 300, 10_000).astype(np.float32)

    assert _tokens(*_text.float32_text(values)) == [str(value) for value in values]


@pytest.mark.parametrize("values", [[0], [7], [9, 10, 11, 99, 100, 101], list(range(1, 12_345, 7))])
def test_uint_text_matches_str(values: list[int]) -> None:
    assert _tokens(*_text.uint_text(np.array(values))) == [str(value) for value in values]


@pytest.mark.parametrize("tokens", [["résidu"], ["a\0b"]])
def test_string_text_rejects_tokens_whose_bytes_differ_from_characters(tokens: list[str]) -> None:
    assert _text.string_text(tokens) is None


@pytest.mark.parametrize("line_wrap", [True, False])
def test_wrapped_rows_match_ihm_loop_rows(line_wrap: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows joined across several chunks match ``ihm``'s own loop writer, wrapped or not."""
    monkeypatch.setattr(_text, "_ROW_CHUNK", 64)
    monkeypatch.setattr(ihm.format.CifWriter, "_line_wrap", line_wrap)
    rng = np.random.default_rng(3)
    alphabet = np.array(list("ABCDEFGHIJ"))
    widths = [1, 4, 9, 12, 30, 79, 80, 81]
    rows = 500
    columns = [["".join(rng.choice(alphabet, rng.integers(1, width + 1))) for _ in range(rows)] for width in widths]
    shared = "1.000"
    fields = [_text.string_text(column) for column in columns]
    fields.insert(3, (_text.string_text([shared])[0], len(shared)))

    actual = _text.wrapped_rows(fields, rows, _text.CIF_LINE_LENGTH if line_wrap else 0).decode("ascii")

    expected = io.StringIO()
    keys = [str(i) for i in range(len(fields))]
    with ihm.format.CifWriter(expected).loop("_x", keys) as loop:
        for i in range(rows):
            tokens = [column[i] for column in columns]
            loop.write(**dict(zip(keys, tokens[:3] + [shared] + tokens[3:], strict=True)))
    lines = expected.getvalue().split("\n")
    # Skip the loop header and its closing "#".
    assert actual == "\n".join(lines[2 + len(fields) : -2]) + "\n"


def test_grid_tokens_match_repr() -> None:
    cells = _text.grid_tokens(32)
    tokens = (cells & np.uint64(_text._TOKEN_MASK)).view(np.uint8).reshape(-1, _text._WINDOW_BYTES)
    lengths = (cells >> np.uint64(_text._LENGTH_SHIFT)).astype(np.int64)

    assert _tokens(tokens, lengths) == [repr(i / _GRID_SCALE) for i in range(_GRID_SIZE)]


def test_grid_tokens_are_built_once_and_read_only() -> None:
    assert _text.grid_tokens(32) is _text.grid_tokens(32)
    assert not _text.grid_tokens(32).flags.writeable


@pytest.mark.parametrize(
    "shape",
    [(1,), (5,), (70_000,), (1, 1), (3, 1), (1, 4), (3, 70_001), (300, 300), (2, 3, 4), (2, 1, 3), (4, 2, 1, 2)],
)
def test_json_grid_text_matches_json(shape: tuple[int, ...]) -> None:
    indices = np.random.default_rng(4).integers(0, _GRID_SIZE, shape)
    indices.flat[:3] = [0, _GRID_SIZE - 1, 5][: indices.size]

    actual = _text.json_grid_text(indices, _text.grid_tokens(32))

    assert actual == json.dumps((indices / _GRID_SCALE).tolist())


def test_json_grid_text_declines_deep_row_boundaries() -> None:
    """A boundary such as "]]]]], [[[[[" no longer fits one eight-byte window."""
    assert _text.json_grid_text(np.zeros((2,) * 6, dtype=np.int64), _text.grid_tokens(32)) is None
