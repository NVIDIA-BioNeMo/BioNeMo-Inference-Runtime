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
"""Atom-name codecs and flat-output layout conversion for data pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from bionemo_ir.data.schemas.basic import AtomTypes

NUM_ATOM_NAME_CHARS = 4
NUM_ATOM_NAME_CHAR_CLASSES = 64
ATOM_NAME_CHAR_OFFSET = 32
ATOM_NAME_TO_FOLDING_INDEX: dict[str, int] = {
    atom_type.name: index for index, atom_type in enumerate(AtomTypes.all_types())
}
NUM_FOLDING_ATOM_TYPES = len(ATOM_NAME_TO_FOLDING_INDEX)


def encode_atom_name_chars(
    atom_name: str,
    *,
    strip: bool = False,
) -> list[int]:
    """Encode an atom name as four ``ord(character) - 32`` values."""
    name = str(atom_name)
    if strip:
        name = name.strip()
    padded = name.ljust(NUM_ATOM_NAME_CHARS)[:NUM_ATOM_NAME_CHARS]
    codes = [ord(character) - ATOM_NAME_CHAR_OFFSET for character in padded]
    if any(code < 0 or code >= NUM_ATOM_NAME_CHAR_CLASSES for code in codes):
        raise ValueError(f"atom name {atom_name!r} contains unsupported characters")
    return codes


def encode_atom_name_chars_one_hot(
    atom_names: Sequence[str],
    *,
    strip: bool = False,
) -> torch.Tensor:
    """One-hot encode atom names to ``[N_atom, 4, 64]`` int32."""
    codes = [encode_atom_name_chars(name, strip=strip) for name in atom_names]
    if not codes:
        return torch.empty(
            (0, NUM_ATOM_NAME_CHARS, NUM_ATOM_NAME_CHAR_CLASSES),
            dtype=torch.int32,
        )
    codes_tensor = torch.tensor(codes, dtype=torch.long)
    return F.one_hot(codes_tensor, NUM_ATOM_NAME_CHAR_CLASSES).to(torch.int32)


def decode_atom_name_chars(
    encoded: torch.Tensor | np.ndarray | None,
    atom_mask: torch.Tensor | np.ndarray,
) -> list[str]:
    """Decode padded integer or one-hot atom-name character tensors."""
    mask = _to_numpy(atom_mask).astype(bool, copy=False)
    if mask.ndim != 1:
        raise ValueError(f"atom_mask must have shape [N_atom], got {mask.shape}")
    if encoded is None:
        return [""] * mask.shape[0]

    chars = _to_numpy(encoded)
    if chars.ndim == 3:
        chars = chars.argmax(axis=-1)
    if chars.shape != (mask.shape[0], NUM_ATOM_NAME_CHARS):
        raise ValueError(f"encoded atom names must have shape [N_atom, 4] or [N_atom, 4, C], got {chars.shape}")

    names: list[str] = []
    for active, char_codes in zip(mask, chars, strict=True):
        if not active:
            names.append("")
            continue
        name = ""
        for code in char_codes:
            code = int(code)
            if code == 0:
                break
            name += chr(code + ATOM_NAME_CHAR_OFFSET)
        names.append(name.strip())
    return names


def scatter_flat_atoms_to_folding_layout(
    positions: torch.Tensor | np.ndarray,
    atom_to_token_index: torch.Tensor | np.ndarray,
    atom_names: Sequence[str],
    atom_mask: torch.Tensor | np.ndarray,
    num_tokens: int,
    *,
    atom_name_to_index: Mapping[str, int] = ATOM_NAME_TO_FOLDING_INDEX,
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter flat atom rows into the canonical ``FoldingOutput`` layout."""
    flat_positions = _to_numpy(positions)
    ownership = _to_numpy(atom_to_token_index)
    active = _to_numpy(atom_mask).astype(bool, copy=False)
    num_atoms = active.shape[0]
    if flat_positions.shape != (num_atoms, 3):
        raise ValueError(f"positions must have shape {(num_atoms, 3)}, got {flat_positions.shape}")
    if ownership.shape != (num_atoms,):
        raise ValueError(f"atom_to_token_index must have shape {(num_atoms,)}, got {ownership.shape}")
    if len(atom_names) != num_atoms:
        raise ValueError(f"atom_names must contain {num_atoms} entries, got {len(atom_names)}")

    num_atom_types = max(atom_name_to_index.values(), default=-1) + 1
    atom_positions = np.zeros((num_tokens, num_atom_types, 3), dtype=np.float32)
    output_mask = np.zeros((num_tokens, num_atom_types), dtype=np.float32)
    for atom_index in np.flatnonzero(active):
        token_index = int(ownership[atom_index])
        if token_index < 0 or token_index >= num_tokens:
            continue
        slot = atom_name_to_index.get(atom_names[atom_index])
        if slot is None:
            continue
        atom_positions[token_index, slot] = flat_positions[atom_index]
        output_mask[token_index, slot] = 1.0
    return atom_positions, output_mask


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
