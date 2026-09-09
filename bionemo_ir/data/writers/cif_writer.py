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

import ihm
import modelcif
import numpy as np
from modelcif import dumper, model, qa_metric  # noqa: F401

from bionemo_ir.data.schemas.basic import FoldingOutput
from bionemo_ir.data.writers.base_writer import _IHM_REMAP, _MOL_TYPE_TO_KIND, BaseWriter, _classify_chain
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

        # Capture closure state.
        _atom_positions = atom_positions
        _atom_mask = atom_mask
        _b_factors = b_factors
        _atom_types = atom_types
        _chain_indices = chain_indices
        _residue_indices = residue_indices
        _local_seq_id = local_seq_id
        _asym_unit_map = asym_unit_map
        _chain_is_het = chain_is_het
        _plddt = plddt

        class _MyModel(modelcif.model.AbInitioModel):
            def get_atoms(self):
                for i in range(n):
                    c = int(_chain_indices[i])
                    if c not in _asym_unit_map:
                        continue
                    r = int(_residue_indices[i])
                    asym = _asym_unit_map[c]
                    seq_id = _local_seq_id[(c, r)]
                    het = _chain_is_het[c]
                    for atom_name, pos, mask, b_factor in zip(
                        _atom_types,
                        _atom_positions[i],
                        _atom_mask[i],
                        _b_factors[i],
                        strict=False,
                    ):
                        if mask < 0.5:
                            continue
                        # Current AtomTypes universe contains only single-
                        # letter element symbols (C, N, O, S, P) suffixed
                        # with digits or primes, so first-char extraction
                        # is exact. If two-letter elements (CL, FE, MG, ZN,
                        # …) ever enter the universe, ``AtomType`` must
                        # gain an explicit ``element`` field and this line
                        # must consult it.
                        element = atom_name[0]
                        yield modelcif.model.Atom(
                            asym_unit=asym,
                            type_symbol=element,
                            seq_id=seq_id,
                            atom_id=atom_name,
                            x=pos[0],
                            y=pos[1],
                            z=pos[2],
                            het=het,
                            biso=b_factor,
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
        modelcif.dumper.write(fh, [system])
        buffer: str = fh.getvalue()

        if self.output_path is not None:
            with open(self.output_path, "w") as f:
                f.write(buffer)
        return buffer
