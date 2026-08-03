# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PDB writer mirroring ``CIFWriter`` semantics in legacy PDB format.

The writer extends :class:`BaseWriter` and accepts the same
``FoldingOutput`` schema as :class:`~tensorrt_bionemo.data.writers.cif_writer.CIFWriter`,
including the optional ``residue_names`` (per-token CCD codes) and
``mol_types`` (per-token polymer-type id) fields. When those are
present, non-polymer chains are emitted with ``HETATM`` record names
and their real CCD codes (``SAH``, ``TYR``, ``NAG``, …) instead of
the all-``UNK`` legacy fallback.

The shared residue-name remap and chain-classification logic lives in
:mod:`base_writer` so this module and ``CIFWriter`` can't drift apart.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tensorrt_bionemo.data.schemas.basic import (AtomType, FoldingOutput,
                                                 ResType)
from tensorrt_bionemo.data.writers.base_writer import (_IHM_REMAP,
                                                       _MOL_TYPE_TO_KIND,
                                                       BaseWriter,
                                                       _classify_chain)

# PDB single-character chain IDs (cf. spec — column 22 is a single byte).
# 62 unique chains max; beyond that PDB format breaks and callers should
# use the CIF writer instead.
PDB_CHAIN_IDS = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                 "abcdefghijklmnopqrstuvwxyz"
                 "0123456789")
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)


class PDBWriter(BaseWriter):
    """Writes a multi-chain biomolecular structure to a legacy PDB string and file.

    Supports protein, RNA, DNA, and (per-atom-tokenised) non-polymer ligand
    chains.

    PDB format constraints worth knowing:

    * Chain ID is a single character (PDB column 22) — max 62 chains.
      Beyond that, callers should switch to :class:`CIFWriter`.
    * Atom names take a 4-char slot (cols 13-16) with right-justification
      for ≤3-char names. Element symbol goes in cols 77-78.
    * Residue name takes 3 chars (cols 18-20), right-justified. RNA
      ``A`` renders as ``"  A"``; DNA ``DA`` renders as ``" DA"``;
      protein ``ALA`` fills the field exactly.
    * Non-polymer rows use ``HETATM`` (6 chars) instead of ``ATOM``
      (4 chars left-aligned in a 6-char field).
    * Lines are padded to 80 chars.
    """

    def __init__(
        self,
        res_type_mapping: dict[int, ResType],
        atom_type_mapping: dict[int, AtomType],
        output_path: str = "output.pdb",
    ):
        super().__init__(
            res_type_mapping=res_type_mapping,
            atom_type_mapping=atom_type_mapping,
            output_path=output_path,
        )

    # ------------------------------------------------------------------
    # Record-section helpers (public; usable independently of ``write``)
    # ------------------------------------------------------------------

    def get_pdb_headers(self) -> list[str]:
        """Return the per-model PDB header lines.

        Currently just emits a single ``PARENT`` line declaring no
        templates were used. The OpenFold/AlphaFold2 export pipelines
        emit one ``PARENT`` per template; we have none.
        """
        return [f"PARENT {' '.join(['N/A'])}"]

    @staticmethod
    def _chain_end(
        atom_index: int,
        end_resname: str,
        chain_name: str,
        residue_index: int,
    ) -> str:
        """Format a ``TER`` line at the end of a chain.

        PDB columns:

        * 1-6   ``TER`` (left-justified in a 6-char field)
        * 7-11  serial number (right-justified)
        * 18-20 residue name (right-justified)
        * 22    chain ID
        * 23-26 residue sequence number (right-justified)
        """
        chain_end = "TER"
        return (f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} "
                f"{chain_name:>1}{residue_index:>4}")

    # ------------------------------------------------------------------
    # Atom-line formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_atom_line(
        record_type: str,
        atom_index: int,
        atom_name: str,
        res_name_3: str,
        chain_tag: str,
        residue_index: int,
        pos: np.ndarray,
        b_factor: float,
        element: str,
        occupancy: float = 1.00,
    ) -> str:
        """Format a single ``ATOM`` / ``HETATM`` line per PDB columnar spec.

        Atom name placement (cols 13-16): four-character names are
        left-aligned at col 13; three-and-fewer-character names start at
        col 14 (one leading space). The slight asymmetry follows the
        legacy AlphaFold writer and matches PDB v3.3 examples for
        residues with hydrogen-bearing positions.
        """
        # 4-char atom names use the full slot; shorter names get one
        # leading space so the element-symbol column stays aligned.
        name = atom_name if len(atom_name) == 4 else f" {atom_name}"
        alt_loc = ""
        insertion_code = ""
        charge = ""
        return (f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_tag:>1}"
                f"{residue_index:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def write(
        self,
        folding_output: FoldingOutput,
    ) -> str:
        """Serialise ``folding_output`` to a PDB string (and optionally to file).

        Reads the same ``FoldingOutput`` fields as
        :meth:`CIFWriter.write` — including the optional
        ``residue_names`` / ``mol_types`` — so the two writers stay
        semantically aligned on chain classification, HETATM
        emission, and CCD-code labelling.

        Args:
            folding_output: Result of the forward pass of a structure
                prediction network (OpenFold2, Boltz1/2, OpenFold3).

        Returns:
            The PDB document as a string.
        """
        residue_types: np.ndarray = folding_output["residue_types"]
        atom_positions: np.ndarray = folding_output["atom_positions"]
        atom_mask: np.ndarray = folding_output["atom_mask"]
        residue_indices: np.ndarray = folding_output["residue_indices"]
        b_factors: np.ndarray = folding_output["b_factors"]
        chain_indices = folding_output.get("chain_indices")

        # Optional per-residue identity fields. When the producer fills
        # them we emit real CCD codes ("TYR", "SAH", "NAG"…) on HETATM
        # rows and classify chains by an explicit mol-type rather than
        # the all-X heuristic. Producers that don't carry CCD identity
        # leave both ``None`` and fall back to the legacy paths.
        residue_names: Optional[list[str]] = folding_output.get(
            "residue_names")
        mol_types: Optional[np.ndarray] = folding_output.get("mol_types")

        n = residue_types.shape[0]

        # Both fields are consumed below as residue-aligned arrays. Validate
        # length up-front so an off-by-N producer fails loudly here instead
        # of silently misclassifying residues via wrong indices. Mirrors the
        # equivalent check in CIFWriter so the two writers stay aligned.
        if residue_names is not None and len(residue_names) != n:
            raise ValueError(
                f"PDBWriter: residue_names length ({len(residue_names)}) "
                f"must equal n_res ({n})")
        if mol_types is not None and len(mol_types) != n:
            raise ValueError(f"PDBWriter: mol_types length ({len(mol_types)}) "
                             f"must equal n_res ({n})")

        # ── normalise chain_indices to a numpy array ──────────────────
        if chain_indices is None:
            chain_indices = np.zeros(n, dtype=np.int64)

        unique_chains = np.unique(chain_indices)  # sorted
        if unique_chains.size and int(unique_chains.max()) >= PDB_MAX_CHAINS:
            raise ValueError(
                f"The PDB format supports at most {PDB_MAX_CHAINS} chains; "
                f"chain index {int(unique_chains.max())} exceeds the limit. "
                "Use CIFWriter for systems with more chains.")

        # ── Fail fast if nothing renders ──────────────────────────────
        # If every atom is masked out (e.g. a single-chain K⁺ ion whose
        # atom name is outside the AtomTypes universe) we'd emit an
        # empty MODEL/ENDMDL block. Surface that as a clear error
        # instead — mirrors CIFWriter's all-empty fail-fast.
        if n > 0 and not bool(np.any(atom_mask >= 0.5)):
            raise ValueError(
                "PDBWriter: no renderable atoms (atom_mask is all zero). "
                "All atom names fell outside the AtomTypes universe.")

        # ── ResType-name and atom-name tables (per-index lookups) ─────
        restypes: list[str] = [x.name for x in self.res_types]
        pdb_restypes: list[str] = [
            _IHM_REMAP.get(x.name, x.canonical_name) for x in self.res_types
        ]
        atom_types: list[str] = [x.name for x in self.atom_types]

        # ── Group residues by chain, preserving encounter order ───────
        # (mirrors CIFWriter so the two stay aligned on chain
        # classification and per-residue cif/pdb code selection.)
        chain_to_seq: dict[int, list[str]] = {}
        chain_to_pdb: dict[int, list[str]] = {}
        chain_to_token_idx: dict[int, list[int]] = {}
        seen_residue: set[tuple[int, int]] = set()
        for i in range(n):
            c = int(chain_indices[i])
            r = int(residue_indices[i])
            if (c, r) in seen_residue:
                continue
            seen_residue.add((c, r))
            short = restypes[residue_types[i]]
            chain_to_seq.setdefault(c, []).append(short)
            if residue_names is not None and i < len(residue_names):
                chain_to_pdb.setdefault(c, []).append(residue_names[i])
            else:
                chain_to_pdb.setdefault(c, []).append(
                    pdb_restypes[residue_types[i]])
            chain_to_token_idx.setdefault(c, []).append(i)

        # ── Classify each chain once: polymer kind or non-polymer ─────
        chain_kind: dict[int, str] = {}
        for c, seq in chain_to_seq.items():
            if mol_types is not None:
                # All residues in a chain must agree on mol_type — anything
                # else is a producer bug we want surfaced, not silently
                # masked by picking the first residue's value. Mirrors the
                # equivalent check in CIFWriter.
                chain_mol_types = {
                    int(mol_types[i])
                    for i in chain_to_token_idx[c]
                }
                if len(chain_mol_types) != 1:
                    raise ValueError(
                        f"PDBWriter: mixed mol_types in chain "
                        f"{PDB_CHAIN_IDS[c]}: {sorted(chain_mol_types)}")
                mt = next(iter(chain_mol_types))
                chain_kind[c] = _MOL_TYPE_TO_KIND.get(mt, "protein")
            else:
                chain_kind[c] = _classify_chain(tuple(seq))

        # Per-token chain ID lookup (encounter order doesn't matter for
        # PDB — column 22 is always the literal chain letter).
        chain_tag_of: dict[int, str] = {
            c: PDB_CHAIN_IDS[c]
            for c in unique_chains.tolist()
        }
        chain_is_het: dict[int, bool] = {
            c: chain_kind[c] == "nonpoly"
            for c in chain_kind
        }

        # ── Emit lines ────────────────────────────────────────────────
        pdb_lines: list[str] = []
        pdb_lines.extend(self.get_pdb_headers())
        pdb_lines.append("MODEL     1")

        atom_index = 1
        last_chain_index: Optional[int] = None
        last_res_pdb_name: Optional[str] = None
        last_res_index: Optional[int] = None
        last_chain_tag: Optional[str] = None

        for i in range(n):
            c = int(chain_indices[i])
            r = int(residue_indices[i])
            chain_tag = chain_tag_of[c]
            is_het = chain_is_het[c]
            record_type = "HETATM" if is_het else "ATOM"

            # Emit a TER row when the chain switches mid-sample.
            if last_chain_index is not None and last_chain_index != c:
                # Polymer chains terminate with TER; non-polymer chains
                # in PDB v3.3 strictly do NOT get a TER row (HETATM
                # groups stand alone). Matching CIFWriter's behaviour
                # which only marks polymer chains as polymer entities.
                if not chain_is_het[last_chain_index]:
                    pdb_lines.append(
                        self._chain_end(
                            atom_index,
                            last_res_pdb_name or "UNK",
                            last_chain_tag or "A",
                            last_res_index or 0,
                        ))
                    atom_index += 1  # Atom serial advances at TER

            # Resolve the 3-char PDB residue name for this token.
            if residue_names is not None and i < len(residue_names):
                res_name_3 = residue_names[i]
            else:
                res_name_3 = pdb_restypes[residue_types[i]]

            # Emit ATOM/HETATM rows for the masked-in atoms.
            for atom_name, pos, mask, b_factor in zip(
                    atom_types,
                    atom_positions[i],
                    atom_mask[i],
                    b_factors[i],
            ):
                if mask < 0.5:
                    continue
                # Current AtomTypes universe contains only single-letter
                # element symbols (C, N, O, S, P) suffixed with digits
                # or primes, so first-char extraction is exact. If
                # two-letter elements (CL, FE, MG, ZN, …) ever enter
                # the universe, AtomType must gain an explicit
                # ``element`` field and this line must consult it.
                element = atom_name[0]
                pdb_lines.append(
                    self._format_atom_line(
                        record_type=record_type,
                        atom_index=atom_index,
                        atom_name=atom_name,
                        res_name_3=res_name_3,
                        chain_tag=chain_tag,
                        residue_index=r,
                        pos=pos,
                        b_factor=float(b_factor),
                        element=element,
                    ))
                atom_index += 1

            last_chain_index = c
            last_res_pdb_name = res_name_3
            last_res_index = r
            last_chain_tag = chain_tag

        # Close the final chain.
        if last_chain_index is not None and not chain_is_het[last_chain_index]:
            pdb_lines.append(
                self._chain_end(
                    atom_index,
                    last_res_pdb_name or "UNK",
                    last_chain_tag or "A",
                    last_res_index or 0,
                ))
            atom_index += 1

        pdb_lines.append("ENDMDL")
        pdb_lines.append("END")

        # Pad all lines to 80 characters (legacy PDB columnar contract).
        pdb_lines = [line.ljust(80) for line in pdb_lines]
        buffer = "\n".join(pdb_lines) + "\n"

        if self.output_path is not None:
            with open(self.output_path, "w") as f:
                f.write(buffer)
        return buffer
