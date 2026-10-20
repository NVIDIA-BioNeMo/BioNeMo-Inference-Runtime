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
"""Injectable residue- and atom-index resolution for oracle requests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from bionemo_ir.data.schemas import AtomIdentity, EncodedStructure, ResidueIdentity, ResolvedStructure

ResidueResolverFn = Callable[[EncodedStructure, int], ResidueIdentity]
AtomResolverFn = Callable[[EncodedStructure, int, int, ResidueIdentity], AtomIdentity]
ResidueMappingKey = int | tuple[int, int]
AtomMappingKey = int | tuple[str, int]


class OracleResolutionError(ValueError):
    """A source index cannot be represented in a canonical oracle request."""


class ResidueTypeResolver(Protocol):
    """Resolve one source residue-type index at a token position."""

    def resolve_residue(self, structure: EncodedStructure, token_index: int) -> ResidueIdentity:
        """Return the canonical residue identity for ``token_index``."""


class AtomTypeResolver(Protocol):
    """Resolve one occupied source atom index at a token and slot."""

    def resolve_atom(
        self,
        structure: EncodedStructure,
        token_index: int,
        atom_slot: int,
        residue: ResidueIdentity,
    ) -> AtomIdentity:
        """Return the canonical atom identity for an occupied atom slot."""


@dataclass(frozen=True)
class CallableResidueTypeResolver:
    """Adapt a user function to :class:`ResidueTypeResolver`."""

    function: ResidueResolverFn

    def resolve_residue(self, structure: EncodedStructure, token_index: int) -> ResidueIdentity:
        """Delegate residue resolution to the configured function."""
        identity = self.function(structure, token_index)
        if not isinstance(identity, ResidueIdentity):
            raise TypeError(f"residue resolver must return ResidueIdentity, got {type(identity).__name__}")
        return identity


@dataclass(frozen=True)
class CallableAtomTypeResolver:
    """Adapt a user function to :class:`AtomTypeResolver`."""

    function: AtomResolverFn

    def resolve_atom(
        self,
        structure: EncodedStructure,
        token_index: int,
        atom_slot: int,
        residue: ResidueIdentity,
    ) -> AtomIdentity:
        """Delegate atom resolution to the configured function."""
        identity = self.function(structure, token_index, atom_slot, residue)
        if not isinstance(identity, AtomIdentity):
            raise TypeError(f"atom resolver must return AtomIdentity, got {type(identity).__name__}")
        return identity


class MappingResidueTypeResolver:
    """Resolve residues from integer or ``(token_type, residue_type)`` keys.

    A token-type-qualified entry wins over the plain residue index. This lets
    callers represent vocabularies that reuse an integer index across protein,
    RNA, DNA, or ligand token kinds.
    """

    def __init__(self, mapping: Mapping[ResidueMappingKey, ResidueIdentity]) -> None:
        self._mapping = dict(mapping)

    def resolve_residue(self, structure: EncodedStructure, token_index: int) -> ResidueIdentity:
        """Resolve one source index or raise :class:`OracleResolutionError`."""
        source_index = int(structure.residue_type_indices[token_index])
        token_type = None if structure.token_type_indices is None else int(structure.token_type_indices[token_index])
        qualified_key = None if token_type is None else (token_type, source_index)
        if qualified_key is not None and qualified_key in self._mapping:
            return self._mapping[qualified_key]
        if source_index in self._mapping:
            return self._mapping[source_index]
        raise OracleResolutionError(
            f"no residue identity for source index {source_index}"
            + ("" if token_type is None else f" and token type {token_type}")
        )


class MappingAtomTypeResolver:
    """Resolve atoms from integer or ``(component_id, atom_type)`` keys."""

    def __init__(self, mapping: Mapping[AtomMappingKey, AtomIdentity]) -> None:
        self._mapping = dict(mapping)

    def resolve_atom(
        self,
        structure: EncodedStructure,
        token_index: int,
        atom_slot: int,
        residue: ResidueIdentity,
    ) -> AtomIdentity:
        """Resolve one occupied source atom index."""
        source_index = int(structure.atom_type_indices[token_index, atom_slot])
        qualified_key = (residue.component_id, source_index)
        if qualified_key in self._mapping:
            return self._mapping[qualified_key]
        if source_index in self._mapping:
            return self._mapping[source_index]
        raise OracleResolutionError(
            f"no atom identity for source index {source_index} in residue {residue.component_id}"
        )


def resolve_structure(
    structure: EncodedStructure,
    residue_type_resolver: ResidueTypeResolver,
    atom_type_resolver: AtomTypeResolver,
) -> ResolvedStructure:
    """Resolve every included residue and occupied atom in ``structure``.

    Args:
        structure: Source-model arrays with coordinates in ångströms.
        residue_type_resolver: User-supplied residue vocabulary resolver.
        atom_type_resolver: User-supplied atom vocabulary resolver.

    Returns:
        A shape-aligned structure with canonical identities.
    """
    residues: list[ResidueIdentity] = []
    atoms: list[tuple[AtomIdentity | None, ...]] = []
    occupied = structure.atom_mask.astype(bool, copy=False)
    for token_index in range(structure.coordinates.shape[0]):
        residue = residue_type_resolver.resolve_residue(structure, token_index)
        if not isinstance(residue, ResidueIdentity):
            raise TypeError(f"residue resolver must return ResidueIdentity, got {type(residue).__name__}")
        residues.append(residue)
        atom_row: list[AtomIdentity | None] = []
        for atom_slot in range(structure.coordinates.shape[1]):
            if not occupied[token_index, atom_slot]:
                atom_row.append(None)
                continue
            atom = atom_type_resolver.resolve_atom(structure, token_index, atom_slot, residue)
            if not isinstance(atom, AtomIdentity):
                raise TypeError(f"atom resolver must return AtomIdentity, got {type(atom).__name__}")
            atom_row.append(atom)
        atoms.append(tuple(atom_row))
    return ResolvedStructure(encoded=structure, residues=tuple(residues), atoms=tuple(atoms))


__all__ = [
    "AtomTypeResolver",
    "CallableAtomTypeResolver",
    "CallableResidueTypeResolver",
    "MappingAtomTypeResolver",
    "MappingResidueTypeResolver",
    "OracleResolutionError",
    "ResidueTypeResolver",
    "resolve_structure",
]
