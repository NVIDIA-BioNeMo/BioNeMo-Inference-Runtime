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

"""Residue vocabularies a source model's weights are written against.

A vocabulary fixes two things a checkpoint depends on: the order of its residue
classes, which its readout width is indexed by, and which of the 37 atom slots
each class occupies, which bounds the atoms a decoded residue may emit. Both
belong to the checkpoint's contract rather than to any layer, so they are
described here beside the other schemas and resolved when weights are converted.

Each vocabulary is a module under :mod:`.vocabularies` holding its rows as a
literal, and they are looked up by the class count they declare. A release that
grows one adds a module; nothing is parsed or read at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from .vocabularies import aa_rna_dna_glycan_ncaa

#: Atom slots in the atom37 representation.
ATOM37_SIZE = 37

#: Placeholder for a slot a residue does not occupy.
_EMPTY_SLOT = "-"

#: Token kind each vocabulary group decodes as. ``mask`` and ``unknown`` share
#: the unknown kind, which is how they stay unreachable for a specific one.
TOKEN_TYPE_IDS = {"aa": 0, "rna": 1, "dna": 2, "atomized": 3, "unknown": 4, "glycan": 5, "ncaa": 6, "mask": 4}

#: The kind that may decode as anything any other kind may.
UNKNOWN_TOKEN_TYPE = 4


@dataclass(frozen=True)
class ResidueVocabulary:
    """One model's residue classes, in the order its readout is indexed by.

    Attributes:
        name: The vocabulary's file stem, which names it in errors.
        atom_slots: Per residue class, the atom name in each atom37 slot, with
            ``-`` where the residue does not occupy that slot.
        groups: Per residue class, the token group it belongs to.
    """

    name: str
    atom_slots: tuple[tuple[str, ...], ...]
    groups: tuple[str, ...] = ()

    @property
    def num_residue_types(self) -> int:
        """Number of residue classes this vocabulary declares."""
        return len(self.atom_slots)

    @cache  # noqa: B019 -- the instance is frozen, so the table is a constant
    def atom37_occupancy(self) -> tuple[tuple[int, ...], ...]:
        """Occupied atom37 slot indices per residue class."""
        return tuple(tuple(slot for slot, atom in enumerate(atoms) if atom != _EMPTY_SLOT) for atoms in self.atom_slots)

    @cache  # noqa: B019 -- the instance is frozen, so the table is a constant
    def allowed_residue_types(self, num_token_kinds: int) -> tuple[tuple[int, ...], ...]:
        """Residue classes each token kind may decode as.

        A decoder free to pick across the whole vocabulary can land on a class
        its token kind never represents, including the mask class, whose atom
        table is empty and whose token then emits nothing at all. Restricting
        the logits first is what keeps a protein token decoding as an amino
        acid. The unknown kind may decode as anything a specific kind may, and
        classes belonging to no specific kind stay unreachable.
        """
        allowed: list[set[int]] = [set() for _ in range(num_token_kinds)]
        for index, group in enumerate(self.groups):
            kind = TOKEN_TYPE_IDS.get(group)
            if kind is None or kind == UNKNOWN_TOKEN_TYPE or kind >= num_token_kinds:
                continue
            allowed[kind].add(index)
        if UNKNOWN_TOKEN_TYPE < num_token_kinds and allowed:
            allowed[UNKNOWN_TOKEN_TYPE] = set().union(*allowed)
        return tuple(tuple(sorted(classes)) for classes in allowed)


def _from_rows(name: str, rows: tuple[str, ...]) -> ResidueVocabulary:
    """Split ``<group> <residue> <37 slots>`` rows into a vocabulary."""
    groups, slots = [], []
    for row in rows:
        group, _residue, *atoms = row.split()
        groups.append(group)
        slots.append(tuple(atoms))
    return ResidueVocabulary(name=name, atom_slots=tuple(slots), groups=tuple(groups))


@cache
def vendored_residue_vocabularies() -> dict[int, ResidueVocabulary]:
    """Every vocabulary shipped with this build, keyed by class count."""
    vocabularies = (_from_rows("aa_rna_dna_glycan_ncaa", aa_rna_dna_glycan_ncaa.RESIDUES),)
    for vocabulary in vocabularies:
        widest = max(len(atoms) for atoms in vocabulary.atom_slots)
        if widest > ATOM37_SIZE:
            raise ValueError(f"{vocabulary.name}: a residue declares {widest} atom slots, over {ATOM37_SIZE}")
    return {vocabulary.num_residue_types: vocabulary for vocabulary in vocabularies}


def residue_vocabulary(num_residue_types: int) -> ResidueVocabulary | None:
    """The vocabulary of this size, or ``None`` if this build ships none."""
    return vendored_residue_vocabularies().get(num_residue_types)
