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
from functools import cache, lru_cache
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from bionemo_ir.data.schemas.basic import AtomTypes
from bionemo_ir.logger import logger

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
) -> np.ndarray:
    """Decode padded integer or one-hot atom-name characters into a string array.

    Args:
        encoded: Character codes shaped ``[N_atom, 4]`` or their one-hot
            ``[N_atom, 4, C]``.
        atom_mask: Atom presence shaped ``[N_atom]``; absent atoms decode to ``""``.
    """
    mask = to_numpy(atom_mask).astype(bool, copy=False)
    if mask.ndim != 1:
        raise ValueError(f"atom_mask must have shape [N_atom], got {mask.shape}")
    if encoded is None:
        return np.zeros(mask.shape, dtype=f"<U{NUM_ATOM_NAME_CHARS}")

    chars = one_hot_indices(encoded) if np.ndim(encoded) == 3 else to_numpy(encoded)
    if chars.shape != (mask.shape[0], NUM_ATOM_NAME_CHARS):
        raise ValueError(f"encoded atom names must have shape [N_atom, 4] or [N_atom, 4, C], got {chars.shape}")

    # Zero terminates the entire remaining name.
    present = np.logical_and.accumulate(chars != 0, axis=-1) & mask[:, None]
    codepoints = np.where(present, chars + ATOM_NAME_CHAR_OFFSET, 0).astype(np.uint32, order="C")
    names = codepoints.view(f"U{NUM_ATOM_NAME_CHARS}").reshape(-1)
    # Codes 1-63 map to "!"-"_", which contain no whitespace to strip.
    if chars.size and (chars.min() < 0 or chars.max() >= NUM_ATOM_NAME_CHAR_CLASSES):
        names = np.asarray([name.strip() for name in names.tolist()], dtype=names.dtype)
    return names


def decode_element_symbols(
    encoded: torch.Tensor | np.ndarray | None,
    atom_mask: torch.Tensor | np.ndarray,
    *,
    index_offset: int = 0,
) -> np.ndarray:
    """Decode a per-atom element one-hot into title-case element symbols.

    Producers one-hot the element by atomic number but disagree on the origin:
    Boltz2 indexes by atomic number directly (``index_offset=0``), OpenFold3 by
    atomic number minus one (``index_offset=1``). Unknown or out-of-table
    indices decode to ``""`` so callers fall back to their own heuristic.

    Args:
        encoded: Element indices shaped ``[N_atom]`` or their one-hot ``[N_atom, C]``.
        atom_mask: Atom presence shaped ``[N_atom]``; absent atoms decode to ``""``.
        index_offset: Atomic number of index zero.
    """
    mask = to_numpy(atom_mask).astype(bool, copy=False)
    if mask.ndim != 1:
        raise ValueError(f"atom_mask must have shape [N_atom], got {mask.shape}")
    symbols = np.zeros(mask.shape, dtype="<U2")
    if encoded is None:
        return symbols

    if np.ndim(encoded) == 2:
        # An all-zero row has no class, though its argmax reads as index 0.
        indices = np.where(_has_class(encoded), one_hot_indices(encoded), -1)
    else:
        indices = to_numpy(encoded)
    if indices.shape != (mask.shape[0],):
        raise ValueError(f"encoded elements must have shape [N_atom] or [N_atom, C], got {indices.shape}")

    numbers = indices + index_offset
    valid = mask & (indices >= 0) & (numbers >= 1) & (numbers <= 118)
    symbols[valid] = _periodic_symbols()[numbers[valid]]
    return symbols


def one_hot_indices(encoded: torch.Tensor | np.ndarray) -> np.ndarray:
    """Index of the first maximum along the last axis, as a host array.

    Device tensors reduce in place so only the indices cross to the host;
    host arrays reduce in NumPy, which outpaces a single-threaded Torch argmax.

    Args:
        encoded: One-hot or score array; the last axis holds the classes.
    """
    if isinstance(encoded, torch.Tensor) and (encoded.device.type != "cpu" or encoded.dtype == torch.bfloat16):
        return encoded.argmax(dim=-1).cpu().numpy()
    return to_numpy(encoded).argmax(axis=-1)


def _has_class(encoded: torch.Tensor | np.ndarray) -> np.ndarray:
    """Whether each row along the last axis has any nonzero entry."""
    if isinstance(encoded, torch.Tensor):
        return encoded.any(dim=-1).cpu().numpy()
    return np.asarray(encoded).any(axis=-1)


@cache
def _periodic_symbols() -> np.ndarray:
    from rdkit.Chem import GetPeriodicTable

    table = GetPeriodicTable()
    return np.asarray([""] + [table.GetElementSymbol(number) for number in range(1, 119)], dtype="<U2")


class ScatteredAtoms(NamedTuple):
    """Dense ``FoldingOutput`` atom layout produced by the shared scatter.

    Attributes:
        positions: ``(num_tokens, num_atom_types, 3)`` float32 coordinates.
        mask: ``(num_tokens, num_atom_types)`` float32 occupancy, 1.0 where an
            atom was placed.
        names: ``(num_tokens, num_atom_types)`` the *source* atom name for each
            placed slot. For atoms whose name is in ``atom_name_to_index`` this
            equals the slot's canonical name; for atoms placed in an overflow
            slot it is the producer's name (e.g. a SMILES ``"C24"``), which the
            canonical slot name would otherwise misreport.
        elements: ``(num_tokens, num_atom_types)`` element symbols, empty
            strings when the producer did not supply ``atom_elements``.
    """

    positions: np.ndarray
    mask: np.ndarray
    names: np.ndarray
    elements: np.ndarray


def scatter_flat_atoms_to_folding_layout(
    positions: torch.Tensor | np.ndarray,
    atom_to_token_index: torch.Tensor | np.ndarray,
    atom_names: Sequence[str] | np.ndarray,
    atom_mask: torch.Tensor | np.ndarray,
    num_tokens: int,
    *,
    atom_name_to_index: Mapping[str, int] = ATOM_NAME_TO_FOLDING_INDEX,
    atom_elements: Sequence[str] | np.ndarray | None = None,
) -> ScatteredAtoms:
    """Scatter flat atom rows into the canonical ``FoldingOutput`` layout.

    Atoms whose name is in ``atom_name_to_index`` keep their canonical slot, so
    the protein/nucleic layout is unchanged. Any other name — a ligand atom
    named by the SMILES builder (``"C24"``), a metal, a halogen — is placed in
    the first free slot of its token's row and its real name is carried out in
    ``ScatteredAtoms.names`` for the writers to emit.

    This fallback exists because ``AtomTypes`` is a fixed vocabulary while
    SMILES atom names are not: they are ``SYMBOL + canonical_rank``, ranked
    over the hydrogen-added molecule, so even a 15-heavy-atom ligand routinely
    produces names past the end of the vocabulary. Dropping them silently
    emitted protein-only structures for successful ligand requests.

    Args:
        positions: Flat atom coordinates shaped ``[N_atom, 3]``.
        atom_to_token_index: Owning token per atom shaped ``[N_atom]``.
        atom_names: Name per atom; selects the canonical slot.
        atom_mask: Atom presence shaped ``[N_atom]``.
        num_tokens: Rows in the output layout.
        atom_name_to_index: Canonical slot per atom name; its largest
            index sets the row width.
        atom_elements: Element symbol per atom, carried to the writers.

    Raises:
        ValueError: if an atom cannot be placed because its token's row is
            already full. That is unrepresentable in this layout, and losing
            the atom quietly is exactly the failure this function must avoid.
    """
    flat_positions = to_numpy(positions)
    ownership = to_numpy(atom_to_token_index)
    active = to_numpy(atom_mask).astype(bool, copy=False)
    num_atoms = active.shape[0]
    if flat_positions.shape != (num_atoms, 3):
        raise ValueError(f"positions must have shape {(num_atoms, 3)}, got {flat_positions.shape}")
    if ownership.shape != (num_atoms,):
        raise ValueError(f"atom_to_token_index must have shape {(num_atoms,)}, got {ownership.shape}")
    if len(atom_names) != num_atoms:
        raise ValueError(f"atom_names must contain {num_atoms} entries, got {len(atom_names)}")
    if atom_elements is not None and len(atom_elements) != num_atoms:
        raise ValueError(f"atom_elements must contain {num_atoms} entries, got {len(atom_elements)}")

    num_atom_types = max(atom_name_to_index.values(), default=-1) + 1
    atom_positions = np.zeros((num_tokens, num_atom_types, 3), dtype=np.float32)
    output_mask = np.zeros((num_tokens, num_atom_types), dtype=np.float32)
    output_names = np.zeros((num_tokens, num_atom_types), dtype=f"<U{NUM_ATOM_NAME_CHARS}")
    output_elements = np.zeros((num_tokens, num_atom_types), dtype="<U2")

    active_indices = np.flatnonzero(active)
    token_of = ownership[active_indices].astype(np.int64, copy=False)
    name_array = np.asarray(atom_names)
    element_array = None if atom_elements is None else np.asarray(atom_elements, dtype="<U2")
    slot_of = _lookup_slots(name_array[active_indices], atom_name_to_index)
    in_range = (token_of >= 0) & (token_of < num_tokens)

    # Pass 1 — canonical slots. Done first so that an overflow atom can never
    # squat on a slot a named atom in the same row still needs.
    canonical = in_range & (slot_of >= 0)
    rows, columns = token_of[canonical], slot_of[canonical]
    source = active_indices[canonical]
    atom_positions[rows, columns] = flat_positions[source]
    output_mask[rows, columns] = 1.0
    output_names[rows, columns] = name_array[source]
    if element_array is not None:
        output_elements[rows, columns] = element_array[source]

    # Pass 2 — the k-th overflow atom of a row, in atom order, takes the row's
    # k-th slot left free by pass 1.
    overflow = in_range & (slot_of < 0)
    if overflow.any():
        source, rows = active_indices[overflow], token_of[overflow]
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        sorted_rank = np.arange(order.shape[0]) - np.searchsorted(sorted_rows, sorted_rows)
        rank = np.empty_like(order)
        rank[order] = sorted_rank
        row_ids = sorted_rows[sorted_rank == 0]
        row_of = np.searchsorted(row_ids, rows)
        free_rows, free_columns = np.nonzero(output_mask[row_ids] == 0.0)
        free_count = np.bincount(free_rows, minlength=row_ids.shape[0])
        full = np.flatnonzero(rank >= free_count[row_of])
        if full.shape[0]:
            atom_index, token_index = int(source[full[0]]), int(rows[full[0]])
            raise ValueError(
                f"atom {str(name_array[atom_index])!r} cannot be placed: token {token_index} already "
                f"holds {num_atom_types} atoms, the width of the FoldingOutput layout"
            )
        columns = free_columns[np.cumsum(free_count)[row_of] - free_count[row_of] + rank]
        atom_positions[rows, columns] = flat_positions[source]
        output_mask[rows, columns] = 1.0
        output_names[rows, columns] = name_array[source]
        if element_array is not None:
            output_elements[rows, columns] = element_array[source]

    out_of_range = int((~in_range).sum())
    if out_of_range:
        logger.warning(
            "scatter_flat_atoms_to_folding_layout: %d active atom(s) reference a token index "
            "outside [0, %d) and were not placed.",
            out_of_range,
            num_tokens,
        )

    return ScatteredAtoms(atom_positions, output_mask, output_names, output_elements)


def _lookup_slots(names: np.ndarray, atom_name_to_index: Mapping[str, int]) -> np.ndarray:
    """Slot of each name in ``atom_name_to_index``, or -1 when absent."""
    keys, slots = _sorted_vocabulary(tuple(atom_name_to_index.items()))
    if names.dtype.kind != "U" or keys.dtype.kind != "U":
        return np.fromiter((atom_name_to_index.get(name, -1) for name in names.tolist()), np.int64, names.shape[0])
    position = np.minimum(np.searchsorted(keys, names), keys.shape[0] - 1)
    return np.where(keys[position] == names, slots[position], -1)


@lru_cache(maxsize=8)
def _sorted_vocabulary(items: tuple[tuple[str, int], ...]) -> tuple[np.ndarray, np.ndarray]:
    """Names sorted for ``np.searchsorted`` and their slots."""
    keys = np.asarray([name for name, _ in items])
    order = np.argsort(keys)
    return keys[order], np.fromiter((slot for _, slot in items), np.int64, len(items))[order]


def to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert inference outputs to NumPy, including bfloat16.

    Args:
        value: Tensor on any device, or an array-like.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    return np.asarray(value)
