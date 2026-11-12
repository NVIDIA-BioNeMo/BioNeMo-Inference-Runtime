# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import io
import string
from collections import defaultdict
from collections.abc import Callable, Sequence
from itertools import chain, repeat
from typing import NamedTuple

import ihm
import ihm.dumper
import ihm.format
import modelcif
import numpy as np
from modelcif import dumper, model, qa_metric  # noqa: F401

from bionemo_ir.data.schemas.basic import FoldingOutput
from bionemo_ir.data.writers import _text
from bionemo_ir.data.writers.base_writer import (
    _IHM_REMAP,
    _MOL_TYPE_TO_KIND,
    BaseWriter,
    _classify_chain,
    _occupied_atoms,
    _OccupiedAtoms,
)
from bionemo_ir.logger import logger


def _chain_id_from_index(idx: int) -> str:
    """Map a 0-based chain index to a mmCIF asym_id.

    0-25  → 'A'..'Z'
    26-51 → 'a'..'z'
    52+   → 'AA','AB',... ('AAA' for 26*26+26=702 and beyond)

    mmCIF accepts arbitrary multi-character ``label_asym_id`` strings, so
    no hard cap is needed beyond what string.ascii_letters expresses.
    """
    if idx < 0:
        raise ValueError(f"chain index must be non-negative, got {idx}")
    letters = string.ascii_uppercase + string.ascii_lowercase
    base = len(letters)
    if idx < base:
        return letters[idx]
    # Convert to a positional base-N representation using the same alphabet.
    out = ""
    n = idx
    while True:
        out = letters[n % base] + out
        n = n // base - 1
        if n < 0:
            break
    return out


def _pick_polymer_alphabet(kind: str):
    if kind == "rna":
        return ihm.RNAAlphabet
    if kind == "dna":
        return ihm.DNAAlphabet
    return ihm.LPeptideAlphabet


# ihm's nucleotide alphabets hold only the four canonical bases (its peptide
# alphabet does include ``UNK``), so the unknown-nucleotide codes ``_IHM_REMAP``
# emits have nothing to resolve against. ``Entity`` also accepts ChemComp
# instances, so pass these directly; ids follow the PDB CCD. Singletons keep
# Entity dedup stable.
_UNKNOWN_NUCLEOTIDE_COMPS: dict[str, ihm.ChemComp] = {
    "N": ihm.RNAChemComp(id="N", code="N", code_canonical="N", name="UNKNOWN RIBONUCLEOTIDE"),
    "DN": ihm.DNAChemComp(id="DN", code="DN", code_canonical="N", name="UNKNOWN DEOXYRIBONUCLEOTIDE"),
}


# ``modelcif`` pins only itself, so an install outside the lock resolves the
# newest ``ihm``. 2.12 keeps 2.11's CIF writer, range checker and the dumpers
# that modelcif 1.6 runs, and writes the same bytes.
_BULK_IHM_VERSIONS = frozenset({"2.11", "2.12"})
_BULK_MODELCIF_VERSION = "1.6"
_ATOM_SITE_KEYS = [
    "group_PDB",
    "id",
    "type_symbol",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_seq_id",
    "auth_seq_id",
    "pdbx_PDB_ins_code",
    "label_asym_id",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "label_entity_id",
    "auth_asym_id",
    "auth_comp_id",
    "B_iso_or_equiv",
    "pdbx_PDB_model_num",
]


def _bulk_dumper_supported() -> bool:
    """Whether the installed ``ihm`` and ``modelcif`` are the versions whose private hooks the bulk dumper uses.

    The compatibility tests cover the atom-site output of exactly these versions.
    """
    return (
        ihm.__version__ in _BULK_IHM_VERSIONS
        and modelcif.__version__ == _BULK_MODELCIF_VERSION
        and hasattr(modelcif.dumper, "_ModelDumper")
        and hasattr(ihm.dumper, "_RangeChecker")
    )


class _EmittedResidue(NamedTuple):
    """A residue row of the model with its asym unit, ``seq_id`` and record type."""

    row: int
    asym: modelcif.AsymUnit
    seq_id: int
    het: bool


class _ResidueTokens(NamedTuple):
    """One residue's formatted ``_atom_site`` tokens, which its atoms share."""

    group: str
    comp_id: str
    label_seq_id: str
    auth_seq_id: str
    ins_code: str
    asym_id: str
    entity_id: str
    strand_id: str


def _check_atoms(
    scored_model: modelcif.model.Model,
    check: bool,
    residues: list[_EmittedResidue],
    atoms: _OccupiedAtoms,
) -> None:
    """Apply ihm's atom range and duplicate checks without one call per atom.

    The assembly and representation checks depend only on an atom's asym unit
    and ``seq_id``. When an asym unit has one assembly range and one
    representation segment, the ``seq_id`` values that pass form an interval
    (``{1}`` for a non-polymer, possibly empty), so its residues with the
    smallest and largest ``seq_id`` stand for the rest; otherwise every
    residue is checked. The duplicate check becomes one set over every atom.
    If either finds a problem, the per-atom checker runs in emission order to
    raise its own error.

    Args:
        scored_model: Model whose assembly and representation bound the atoms.
        check: The dumper's ``_check`` flag; ``False`` skips every check.
        residues: Emitted residues.
        atoms: Occupied atom slots for every row.
    """
    if not check:
        return
    atom = modelcif.model.Atom(None, None, None, None, 0, 0, 0)
    checker = ihm.dumper._RangeChecker(scored_model, check)
    by_asym: dict[modelcif.AsymUnit, list[tuple[int, int]]] = {}
    keys = []
    for i, asym, seq_id, _ in residues:
        lo, hi = atoms.bounds[i], atoms.bounds[i + 1]
        if lo < hi:
            by_asym.setdefault(asym, []).append((seq_id, i))
            keys.extend(zip(repeat(asym._id), repeat(seq_id), atoms.names[lo:hi]))
    try:
        for asym, members in by_asym.items():
            if len(checker.asmb_asym_ids.get(asym._id, ())) == 1 and len(checker.repr_asym_ids.get(asym._id, ())) == 1:
                members = {min(members), max(members)}
            for seq_id, i in members:
                atom.asym_unit, atom.seq_id, atom.atom_id = asym, seq_id, atoms.names[atoms.bounds[i]]
                checker(atom)
    except ValueError:
        pass
    else:
        if len(set(keys)) == len(keys):
            return
    checker = ihm.dumper._RangeChecker(scored_model, check)
    for i, asym, seq_id, _ in residues:
        atom.asym_unit, atom.seq_id = asym, seq_id
        for k in range(atoms.bounds[i], atoms.bounds[i + 1]):
            atom.atom_id = atoms.names[k]
            checker(atom)


def _residue_tokens(residues: list[_EmittedResidue], quote: Callable[[object], str]) -> list[_ResidueTokens]:
    """Format each residue's shared ``_atom_site`` tokens, quoting each asym unit and component once."""
    asym_tokens: dict[modelcif.AsymUnit, tuple[str, bool, str, str, str]] = {}
    comp_tokens: dict[str, str] = {}
    tokens = []
    for residue in residues:
        asym = residue.asym
        shared = asym_tokens.get(asym)
        if shared is None:
            shared = asym_tokens[asym] = (
                "HETATM" if residue.het else "ATOM",
                asym.entity.is_polymeric(),
                quote(asym._id),
                str(asym.entity._id),
                quote(asym.strand_id),
            )
        group, polymeric, asym_id, entity_id, strand_id = shared
        comp_id = asym.sequence[residue.seq_id - 1].id
        comp = comp_tokens.get(comp_id)
        if comp is None:
            comp = comp_tokens[comp_id] = quote(comp_id)
        auth_seq_id, ins_code = asym._get_auth_seq_id_ins_code(residue.seq_id)
        tokens.append(
            _ResidueTokens(
                group=group,
                comp_id=comp,
                label_seq_id=str(residue.seq_id) if polymeric else ".",
                auth_seq_id=quote(auth_seq_id),
                ins_code=quote(ins_code or ihm.unknown),
                asym_id=asym_id,
                entity_id=entity_id,
                strand_id=strand_id,
            )
        )
    return tokens


def _atom_site_text(
    atoms: _OccupiedAtoms,
    residues: list[_EmittedResidue],
    residue_tokens: list[_ResidueTokens],
    quote: Callable[[object], str],
    model_id: str,
    line_length: int,
) -> str | None:
    """Format every ``_atom_site`` row at once, or return ``None`` for inputs the stock dumper must format.

    The column formatter covers float32 coordinates and B factors with ASCII
    identifiers, which is what the structure predictors emit.

    Args:
        atoms: Occupied atom slots for every residue row.
        residues: Emitted residues.
        residue_tokens: Per emitted residue, its formatted tokens.
        quote: The CIF writer's token formatter.
        model_id: Formatted model number.
        line_length: Columns before a row wraps; 0 keeps rows on one line.

    Returns:
        The rows, or ``None`` when the input needs the stock dumper.
    """
    if atoms.positions.dtype != np.float32 or atoms.b_factors.dtype != np.float32:
        return None
    counts = np.array([atoms.bounds[r.row + 1] - atoms.bounds[r.row] for r in residues], dtype=np.int64)
    total = int(counts.sum())
    if not total:
        return ""
    starts = np.array([atoms.bounds[r.row] for r in residues], dtype=np.int64)
    residue_of_atom = np.repeat(np.arange(len(residues)), counts)
    atom_index = starts[residue_of_atom] + np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)

    def dictionary_column(
        values: Sequence[str], index: np.ndarray, spell: Callable[[str], str] = str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        # Spell each distinct value once, then gather the spellings by ``index``.
        codes = {value: code for code, value in enumerate(dict.fromkeys(values))}
        column = _text.string_text([spell(value) for value in codes])
        if column is None:
            return None
        rows = np.fromiter(map(codes.__getitem__, values), dtype=np.intp, count=len(values))[index]
        return column[0][rows], column[1][rows]

    def constant_column(token: str) -> tuple[np.ndarray, int] | None:
        column = _text.string_text([token])
        return None if column is None else (column[0], int(column[1][0]))

    per_field = _ResidueTokens(*zip(*residue_tokens, strict=True))

    def by_residue(values: Sequence[str]) -> tuple[np.ndarray, np.ndarray] | None:
        return dictionary_column(values, residue_of_atom)

    chars, lengths = _text.float32_text(
        np.concatenate([atoms.positions[atom_index].T, atoms.b_factors[None, atom_index]])
    )
    x, y, z, b_factor = ((chars[i * total : (i + 1) * total], lengths[i * total : (i + 1) * total]) for i in range(4))
    comp = by_residue(per_field.comp_id)
    # One entry per ``_ATOM_SITE_KEYS`` key, in order; the stock dumper writes no alternate location.
    fields = [
        by_residue(per_field.group),
        _text.uint_text(np.arange(1, total + 1)),
        dictionary_column(atoms.elements, atom_index, quote),
        dictionary_column(atoms.names, atom_index, quote),
        constant_column("."),
        comp,
        by_residue(per_field.label_seq_id),
        by_residue(per_field.auth_seq_id),
        by_residue(per_field.ins_code),
        by_residue(per_field.asym_id),
        x,
        y,
        z,
        constant_column(quote(1.0)),
        by_residue(per_field.entity_id),
        by_residue(per_field.strand_id),
        comp,
        b_factor,
        constant_column(model_id),
    ]
    if any(field is None for field in fields):
        return None
    return _text.wrapped_rows(fields, total, line_length).decode("ascii")


class _LoopWriter(ihm.format._CifLoopWriter):
    """``ihm``'s loop writer, remembering each string's CIF token, that also takes preformatted rows."""

    @property
    def line_length(self) -> int:
        """Columns a row may fill before ``ihm`` wraps it; 0 when wrapping is off."""
        return _text.CIF_LINE_LENGTH if self._line_wrap else 0

    def _start(self) -> None:
        if self._empty_loop:
            self.writer.fh.write("#\nloop_\n" + "".join(f"{self.category}.{key}\n" for key in self.keys))
            self._empty_loop = False

    def write_rows(self, text: str) -> None:
        """Write rows already formatted as :meth:`write` would format them."""
        if text:
            self._start()
            self.writer.fh.write(text)

    def write(self, **kwargs: object) -> None:
        self._start()
        writer = self.writer
        token_cache = writer.token_cache
        line_length = self.line_length
        parts = []
        column = 0
        for key in self.python_keys:
            value = kwargs.get(key)
            if isinstance(value, str):
                if "\n" in value:
                    # ``ihm.format._write_multiline`` restarts the line afterwards.
                    parts.append("\n;" + value + ("" if value.endswith("\n") else "\n") + ";\n")
                    column = 0
                    continue
                token = token_cache.get(value)
                if token is None:
                    token = token_cache[value] = writer._repr(value)
            elif value is None:
                token = "."
            elif type(value) is int:
                token = str(value)
            else:
                token = writer._repr(value)
            if column:
                if line_length and column + len(token) + 1 > line_length:
                    parts.append("\n")
                    column = 0
                else:
                    parts.append(" ")
                    column += 1
            parts.append(token)
            column += len(token)
        parts.append("\n")
        writer.fh.write("".join(parts))


class _BulkCifWriter(ihm.format.CifWriter):
    """``ihm``'s CIF writer whose loops cache string tokens and take preformatted rows."""

    def __init__(self, fh: io.TextIOBase) -> None:
        super().__init__(fh)
        self.token_cache: dict[str, str] = {}

    def loop(self, category: str, keys: list[str]) -> _LoopWriter:
        return _LoopWriter(self, category, keys, line_wrap=self._line_wrap)


class _BulkModelDumper(modelcif.dumper._ModelDumper):
    """modelcif's model dumper with the ``_atom_site`` rows formatted as whole columns.

    Validation and category order stay modelcif's; only atom construction and
    text formatting are batched. Inputs the column formatter does not cover go
    to the stock dumper.
    """

    def __init__(self, scored_model: modelcif.model.Model, residues: list[_EmittedResidue], atoms: _OccupiedAtoms):
        super().__init__()
        self._scored_model = scored_model
        self._residues = residues
        self._atoms = atoms

    def dump_atoms(self, system: modelcif.System, writer: _BulkCifWriter, add_ihm: bool = True) -> dict[str, None]:
        scored_model, residues, atoms = self._scored_model, self._residues, self._atoms
        _check_atoms(scored_model, self._check, residues, atoms)
        quote = writer._repr
        line_length = _text.CIF_LINE_LENGTH if writer._line_wrap else 0
        text = _atom_site_text(
            atoms, residues, _residue_tokens(residues, quote), quote, str(scored_model._id), line_length
        )
        if text is None:
            return super().dump_atoms(system, writer, add_ihm)
        with writer.loop("_atom_site", _ATOM_SITE_KEYS) as loop:
            loop.write_rows(text)
        emitting = [r for r in residues if atoms.bounds[r.row] < atoms.bounds[r.row + 1]]
        # As in the stock dumper, an asym unit counts as modeled only if it emits an atom.
        self._assembly_checker.add_model_asyms(scored_model, {r.asym._id for r in emitting})
        return dict.fromkeys(
            chain.from_iterable(atoms.elements[atoms.bounds[r.row] : atoms.bounds[r.row + 1]] for r in emitting)
        )


class _BulkCifVariant(modelcif.dumper.ModelCIFVariant):
    """ModelCIF output through :class:`_BulkCifWriter`, with ``model_dumper`` in place of modelcif's own."""

    def __init__(self, model_dumper: _BulkModelDumper) -> None:
        super().__init__()
        self._model_dumper = model_dumper

    def get_dumpers(self) -> list[ihm.dumper.Dumper]:
        return [self._model_dumper if isinstance(d, modelcif.dumper._ModelDumper) else d for d in super().get_dumpers()]

    def get_system_writer(self, system: modelcif.System, writer_class: type, writer: ihm.format.CifWriter) -> object:
        return super().get_system_writer(system, _BulkCifWriter, _BulkCifWriter(writer.fh))


class CIFWriter(BaseWriter):
    """Writes a multi-chain biomolecular structure to a mmCIF string and file.

    Supports protein, RNA, DNA, and (per-atom-tokenised) non-polymer ligand
    chains. Per-residue pLDDT is sourced from ``FoldingOutput['plddt']`` when
    available; otherwise it falls back to the mean masked-in B-factor for
    each residue (which equals the per-residue pLDDT in pipelines that
    broadcast pLDDT into ``b_factors``, e.g. Boltz2).
    """

    def set_output_path(self, output_path: str) -> None:
        self.output_path = output_path

    def write(
        self,
        folding_output: FoldingOutput,
        system_title: str | None = "BioNeMo Inference Runtime Prediction",
    ) -> str:
        """Serialise ``folding_output`` to a mmCIF string (and optionally to file).

        Args:
            folding_output: Result of the forward pass of a structure
                prediction network (OpenFold2, Boltz1/2, OpenFold3).
            system_title: Title written into the mmCIF data block.

        Returns:
            The mmCIF document as a string.
        """
        residue_types: np.ndarray = folding_output["residue_types"]
        atom_positions: np.ndarray = folding_output["atom_positions"]  # (n_res, n_atoms, 3)
        atom_mask: np.ndarray = folding_output["atom_mask"]  # (n_res, n_atoms)
        residue_indices: np.ndarray = folding_output["residue_indices"]  # 1-based, PDB-style
        b_factors: np.ndarray | None = folding_output.get("b_factors")  # (n_res, n_atoms) or None
        plddt: np.ndarray | None = folding_output.get("plddt")  # (n_res,) or None

        # ``FoldingOutput.b_factors`` is optional, but both the atom emitter and
        # the pLDDT fallback index it per residue. Materialise it once here so a
        # producer that omits it writes zero B-factors instead of raising.
        if b_factors is None:
            if plddt is not None:
                b_factors = np.repeat(np.asarray(plddt, dtype=np.float32)[:, None], atom_mask.shape[1], axis=1)
            else:
                b_factors = np.zeros_like(atom_mask, dtype=np.float32)

        # Optional per-residue identity fields. When the producer fills them
        # we can emit real CCD codes ("TYR", "SAH", "DA"…) on HETATM rows and
        # classify chains by an explicit mol-type rather than the all-X
        # heuristic. Producers that don't carry CCD identity leave both None
        # and fall back to the legacy paths.
        # Per-slot atom identity. ``AtomTypes`` is a fixed vocabulary, so a
        # ligand atom whose name falls outside it lands in an overflow slot
        # whose canonical name says nothing about the atom. When the producer
        # supplies these we emit the real name and element per slot.
        per_slot_atom_names: np.ndarray | None = folding_output.get("atom_names")
        per_slot_elements: np.ndarray | None = folding_output.get("atom_elements")

        residue_names: list[str] | None = folding_output.get("residue_names")
        mol_types: np.ndarray | None = folding_output.get("mol_types")

        chain_indices = folding_output["chain_indices"]
        n = residue_types.shape[0]

        # Both fields are consumed below as residue-aligned arrays. Validate
        # length up-front so an off-by-N producer fails loudly here instead of
        # silently misclassifying residues via wrong indices.
        if residue_names is not None and len(residue_names) != n:
            raise ValueError(f"CIFWriter: residue_names length ({len(residue_names)}) must equal n_res ({n})")
        if mol_types is not None and len(mol_types) != n:
            raise ValueError(f"CIFWriter: mol_types length ({len(mol_types)}) must equal n_res ({n})")

        # ── normalise chain_indices to a numpy array ──────────────────────
        if chain_indices is None:
            chain_indices = np.zeros(n, dtype=np.int64)
        elif not isinstance(chain_indices, np.ndarray):
            raise TypeError(
                "CIFWriter: folding_output['chain_indices'] must be a "
                f"np.ndarray or None, got {type(chain_indices).__name__}"
            )
        else:
            ci_min = int(chain_indices.min()) if chain_indices.size else 0
            if ci_min < 0:
                raise ValueError(f"CIFWriter: chain_indices contains negative values (min={ci_min})")

        # ── ResType-name and atom-name tables (per-index lookups) ─────────
        restypes: list[str] = [x.name for x in self.res_types]
        atom_types: list[str] = [x.name for x in self.atom_types]

        # ── Group residues by chain, preserving encounter order ───────────
        #
        # residue_indices is PDB-style numbering: not necessarily contiguous,
        # not 0-indexed, can repeat across chains. We need a (chain_idx,
        # residue_idx) → 1-indexed local seq_id map for modelcif. Using
        # encounter order (rather than chain-contiguity) means non-contiguous
        # / interleaved chain layouts don't silently corrupt entities.
        #
        # We track two parallel per-chain sequences:
        # * ``chain_to_seq``    — short residue codes from ``self.res_types``,
        #   used by the polymer-alphabet classifier when ``mol_types`` is
        #   absent (legacy path).
        # * ``chain_to_cif``    — the residue codes actually written into the
        #   CIF (``residue_names`` when present, else the remapped short
        #   codes). This is what the writer emits on ATOM/HETATM rows and
        #   what Entity ``ChemComp`` IDs are built from.
        chain_to_seq: dict[int, list[str]] = defaultdict(list)
        chain_to_cif: dict[int, list[str]] = defaultdict(list)
        local_seq_id: dict[tuple[int, int], int] = {}
        chain_pos_counter: defaultdict[int, int] = defaultdict(int)
        for i in range(n):
            c = int(chain_indices[i])
            r = int(residue_indices[i])
            if (c, r) not in local_seq_id:
                chain_pos_counter[c] += 1
                local_seq_id[(c, r)] = chain_pos_counter[c]
                short = restypes[residue_types[i]]
                chain_to_seq[c].append(short)
                if residue_names is not None and i < len(residue_names):
                    chain_to_cif[c].append(residue_names[i])
                else:
                    chain_to_cif[c].append(_IHM_REMAP.get(short, short))

        # ── Drop chains with no renderable atoms ──────────────────────────
        #
        # A residue/token only contributes atoms whose name maps into the
        # ``AtomTypes`` universe; producers (e.g. the Boltz2 postprocessor)
        # silently drop atoms with unknown names. A monatomic-ion ligand whose
        # sole atom name is outside the universe — e.g. a K⁺ chain (CCD "K",
        # atom "K") — therefore ends up with an all-zero ``atom_mask`` and
        # zero modeled atoms. ``modelcif`` raises if such an empty AsymUnit is
        # referenced by the Assembly ("asym IDs … don't have coordinates in
        # any Model"), so we exclude empty chains from the Entities, the
        # AsymUnits, and the Assembly entirely.
        chain_has_atoms: dict[int, bool] = defaultdict(bool)
        for i in range(n):
            c = int(chain_indices[i])
            if not chain_has_atoms[c] and bool(np.any(atom_mask[i] >= 0.5)):
                chain_has_atoms[c] = True
        empty_chains = [c for c in sorted(chain_to_seq) if not chain_has_atoms[c]]
        if empty_chains:
            dropped = ", ".join(_chain_id_from_index(c) for c in empty_chains)
            # If every chain is empty there is nothing to render; modelcif
            # would later fail with an opaque "asym IDs don't have
            # coordinates" error. Fail fast with a clearer message instead.
            if len(empty_chains) == len(chain_to_seq):
                raise ValueError(
                    f"CIFWriter: no renderable atoms in any chain "
                    f"(dropped asym id(s): {dropped}). All atom names fell "
                    "outside the AtomTypes universe."
                )
            logger.warning(
                "CIFWriter: dropping %d chain(s) with no renderable atoms "
                "(asym id(s): %s). This typically happens for monatomic-ion "
                "ligands whose atom name is outside the AtomTypes universe "
                "(e.g. K⁺/Na⁺/Zn²⁺).",
                len(empty_chains),
                dropped,
            )

        # ── Classify each chain once: polymer kind or non-polymer ─────────
        #
        # Prefer the explicit per-residue ``mol_types`` from the producer
        # (Boltz2 / OF3 know the chain type unambiguously). Fall back to the
        # legacy all-X heuristic only when that field is not provided.
        chain_kind: dict[int, str] = {}
        for c, seq in chain_to_seq.items():
            if mol_types is not None:
                # All residues in a chain must agree on mol_type — anything
                # else is a producer bug we want surfaced, not silently
                # masked by picking the first residue's value.
                chain_mol_types = {int(mol_types[i]) for i in range(n) if int(chain_indices[i]) == c}
                if len(chain_mol_types) != 1:
                    raise ValueError(
                        f"CIFWriter: mixed mol_types in chain {_chain_id_from_index(c)}: {sorted(chain_mol_types)}"
                    )
                mt = next(iter(chain_mol_types))
                chain_kind[c] = _MOL_TYPE_TO_KIND.get(mt, "protein")
            else:
                chain_kind[c] = _classify_chain(tuple(seq))

        # ── Build Entities (one per unique polymer sequence, one per unique nonpoly content) ──
        #
        # Polymer chains dedup by (kind, sequence). Non-polymer chains also
        # dedup by their (kind, ccd-content) tuple — two ligand chains with
        # the same atom sequence (e.g. T1124's 2× SAH + 2× TYR) share one
        # Entity each. Without this dedup, IHM's ``modelcif.Entity`` rejects
        # the second "Non-polymer ligand subunit" with a Duplicate entity
        # ValueError.
        polymer_entities: dict[tuple[str, tuple[str, ...]], modelcif.Entity] = {}
        nonpoly_entities: dict[tuple[str, ...], modelcif.Entity] = {}
        entities_map: dict[int, modelcif.Entity] = {}
        for c, seq in chain_to_seq.items():
            if not chain_has_atoms[c]:
                continue
            kind = chain_kind[c]
            cif_seq = chain_to_cif[c]
            if kind == "nonpoly":
                # Per-atom-tokenised ligand. When the producer supplied real
                # CCD codes (Boltz2/OF3 do), one ``NonPolymerChemComp`` per
                # CCD code lets the writer emit "SAH"/"TYR"/etc. on HETATM
                # rows. When no CCD codes are available we fall back to
                # ``UNK`` (the legacy heuristic path).
                key = tuple(cif_seq)
                if key not in nonpoly_entities:
                    chem_comps = [ihm.NonPolymerChemComp(id=name) for name in cif_seq]
                    nonpoly_entities[key] = modelcif.Entity(chem_comps, description="Non-polymer ligand subunit")
                entities_map[c] = nonpoly_entities[key]
                continue
            # Polymer (protein/RNA/DNA) chain. ``ihm.{LPeptide,RNA,DNA}Alphabet``
            # are keyed by SHORT codes — 1-letter for protein/RNA (``A``,
            # ``G``, ``R``, ``T``…), 2-letter for DNA (``DA``, ``DT``…). Use
            # the short codes from ``chain_to_seq`` (already short) with the
            # legacy ``_IHM_REMAP`` for X→UNK / R*-stripping. We do NOT pass
            # the 3-letter CCD codes from ``residue_names`` here even when
            # they're present, because those are not valid alphabet keys
            # (e.g. "ALA" is not in LPeptideAlphabet).
            entity_seq = [_IHM_REMAP.get(r, r) for r in seq]
            key = (kind, tuple(entity_seq))
            if key not in polymer_entities:
                # Protein ``N`` is asparagine; only RNA/DNA may substitute
                # unknown-nucleotide ChemComps (RNA ``N``, DNA ``DN``).
                if kind in ("rna", "dna"):
                    ihm_seq = [_UNKNOWN_NUCLEOTIDE_COMPS.get(code, code) for code in entity_seq]
                else:
                    ihm_seq = entity_seq
                try:
                    polymer_entities[key] = modelcif.Entity(
                        ihm_seq,
                        alphabet=_pick_polymer_alphabet(kind),
                        description=f"Model {kind} subunit",
                    )
                except KeyError as exc:
                    raise ValueError(
                        f"CIFWriter: residue code {exc.args[0]!r} in {kind} chain "
                        f"{_chain_id_from_index(c)} is not in the ihm alphabet; add it "
                        "to _IHM_REMAP or _UNKNOWN_NUCLEOTIDE_COMPS."
                    ) from exc
            entities_map[c] = polymer_entities[key]

        # ── Build AsymUnits and assembly ──────────────────────────────────
        asym_unit_map: dict[int, modelcif.AsymUnit] = {}
        chain_is_het: dict[int, bool] = {}
        for c in sorted(chain_to_seq):
            if not chain_has_atoms[c]:
                continue
            chain_id = _chain_id_from_index(c)
            is_het = chain_kind[c] == "nonpoly"
            chain_is_het[c] = is_het
            asym_unit_map[c] = modelcif.AsymUnit(
                entities_map[c],
                details=("Ligand subunit %s" if is_het else "Model subunit %s") % chain_id,
                id=chain_id,
            )
        modeled_assembly = modelcif.Assembly(asym_unit_map.values(), name="Modeled assembly")

        # ── pLDDT QA metric classes ───────────────────────────────────────
        class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
            name = "pLDDT"
            software = None
            description = "Predicted lddt"

        class _GlobalPLDDT(modelcif.qa_metric.Global, modelcif.qa_metric.PLDDT):
            name = "pLDDT"
            software = None
            description = "Global pLDDT, mean of per-residue pLDDTs"

        # Capture closure state. Atom values stay NumPy scalars: ``ihm``
        # formats those with ``str`` and Python floats with three decimals.
        _atoms = _occupied_atoms(
            atom_types, atom_positions, atom_mask, b_factors, per_slot_atom_names, per_slot_elements
        )
        _atom_mask = atom_mask
        _b_factors = b_factors
        _chain_indices = chain_indices
        _residue_indices = residue_indices
        _local_seq_id = local_seq_id
        _asym_unit_map = asym_unit_map
        _chain_is_het = chain_is_het
        _plddt = plddt

        class _MyModel(modelcif.model.AbInitioModel):
            def get_atoms(self):
                # The bulk dumper reads the arrays directly; only the stock dumper needs per-atom scalars.
                xs, ys, zs = (list(_atoms.positions[:, axis]) for axis in range(3))
                atom_b_factors = list(_atoms.b_factors)
                for i in range(n):
                    c = int(_chain_indices[i])
                    if c not in _asym_unit_map:
                        continue
                    r = int(_residue_indices[i])
                    asym = _asym_unit_map[c]
                    seq_id = _local_seq_id[(c, r)]
                    het = _chain_is_het[c]
                    for k in range(_atoms.bounds[i], _atoms.bounds[i + 1]):
                        yield modelcif.model.Atom(
                            asym_unit=asym,
                            type_symbol=_atoms.elements[k],
                            seq_id=seq_id,
                            atom_id=_atoms.names[k],
                            x=xs[k],
                            y=ys[k],
                            z=zs[k],
                            het=het,
                            biso=atom_b_factors[k],
                            occupancy=1.00,
                        )

            def add_scores(self):
                # Per-residue pLDDT. Prefer the explicit ``plddt`` array
                # (cleanest schema use); fall back to averaging masked-in
                # b_factor entries per residue (works when the producer
                # broadcast pLDDT into b_factors, e.g. Boltz2).
                seen: set[tuple[int, int]] = set()
                values: list[float] = []
                for i in range(n):
                    c = int(_chain_indices[i])
                    if c not in _asym_unit_map:
                        continue
                    r = int(_residue_indices[i])
                    key = (c, r)
                    if key in seen:
                        continue
                    if _plddt is not None:
                        v = float(_plddt[i])
                    else:
                        mask_row = _atom_mask[i]
                        bf_row = _b_factors[i]
                        masked = mask_row >= 0.5
                        if not np.any(masked):
                            continue
                        v = float(bf_row[masked].mean())
                    seen.add(key)
                    values.append(v)
                    self.qa_metrics.append(_LocalPLDDT(_asym_unit_map[c].residue(_local_seq_id[key]), v))
                if values:
                    self.qa_metrics.append(_GlobalPLDDT(float(np.mean(values))))

        system = modelcif.System(title=system_title)
        m = _MyModel(assembly=modeled_assembly, name="Best scoring model")
        m.add_scores()
        system.model_groups.append(modelcif.model.ModelGroup([m], name="All models"))

        fh = io.StringIO()
        if _bulk_dumper_supported() and "\n" not in "".join(
            chain(_atoms.names, _atoms.elements, *chain_to_cif.values())
        ):
            residues = [
                _EmittedResidue(i, asym_unit_map[c], local_seq_id[(c, int(residue_indices[i]))], chain_is_het[c])
                for i in range(n)
                if (c := int(chain_indices[i])) in asym_unit_map
            ]
            modelcif.dumper.write(fh, [system], variant=_BulkCifVariant(_BulkModelDumper(m, residues, _atoms)))
        else:
            modelcif.dumper.write(fh, [system])
        buffer: str = fh.getvalue()

        if self.output_path is not None:
            with open(self.output_path, "w") as f:
                f.write(buffer)
        return buffer
