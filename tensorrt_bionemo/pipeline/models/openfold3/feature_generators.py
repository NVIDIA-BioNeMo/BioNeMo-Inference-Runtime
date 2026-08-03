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
"""OpenFold3 feature generators: structure, conformer, MSA, template.

Each generator reads context (built by OpenFold3ContextGenerator) and/or
batch (outputs of prior generators) and returns one group of feature tensors.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
import torch
import torch.nn.functional as F

from tensorrt_bionemo.pipeline.base import FeatureGeneratorBase

from .common import (centre_random_augmentation, compute_deletion_value,
                     encode_atom_name_chars_one_hot, encode_one_hot)
from .const import (DEFAULT_N_TEMPLATES, ELEMENT_ATOMIC_NUMBER, GAP_IDX,
                    MAX_MSA_ROWS, MOL_TYPE_LIGAND, MSA_CHAR_TO_IDX,
                    NUM_ELEMENT_CLASSES, NUM_MSA_CLASSES, NUM_RESTYPE_CLASSES,
                    RESNAME_TO_IDX, TEMPLATE_DISTOGRAM_N_BINS, UNK_IDX)
from .feature_context import (
    _compute_sym_ids,
    _renumber_chain_ids,
    _resolve_msa_char,
)


def _get_row(context: dict[str, Any]) -> dict[str, Any]:
    """Get the context row, handling both _row-wrapped and direct forms."""
    return context.get("_row") or context


class StructureFeatureGenerator(FeatureGeneratorBase):
    """Generates token-level and atom-level structural features.

    Produces: token_index, residue_index, asym_id, entity_id, sym_id,
    restype, is_protein/rna/dna/ligand, is_atomized, token_mask,
    num_atoms_per_token, start_atom_index, atom_mask, atom_to_token_index,
    token_bonds.
    """

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        row = _get_row(context)
        struct = row["structure"]

        n_tokens = struct["n_tokens"]
        n_atoms = struct["n_atoms"]
        resnames = struct["token_resnames"]
        chain_ids = struct["token_chain_ids"]
        entity_ids = struct["token_entity_ids"]
        mol_types = struct["token_mol_types"]
        res_ids = struct["token_res_ids"]
        atoms_per_tok = struct["atoms_per_token"]
        start_atoms = struct["token_start_atoms"]
        atom_tok_idx = struct["atom_token_idx"]

        feats: dict[str, torch.Tensor] = {}

        # token_index: 0-based token indices
        feats["token_index"] = torch.arange(n_tokens, dtype=torch.int32)

        # residue_index: 1-based residue indices within each chain
        feats["residue_index"] = torch.tensor(res_ids, dtype=torch.int32)

        # asym_id: renumbered chain IDs (1-based)
        feats["asym_id"] = torch.tensor(_renumber_chain_ids(chain_ids),
                                        dtype=torch.int32)

        # entity_id
        feats["entity_id"] = torch.tensor(entity_ids, dtype=torch.int32)

        # sym_id
        sym_ids = _compute_sym_ids(entity_ids, chain_ids)
        feats["sym_id"] = torch.tensor(sym_ids, dtype=torch.int32)

        # restype: one-hot [N_tokens, 32]
        restype_idx = [RESNAME_TO_IDX.get(rn, UNK_IDX) for rn in resnames]
        feats["restype"] = encode_one_hot(
            torch.tensor(restype_idx, dtype=torch.long),
            NUM_RESTYPE_CLASSES,
        )

        # molecule type masks
        mt = torch.tensor(mol_types, dtype=torch.int32)
        feats["is_protein"] = (mt == 0).to(torch.int32)
        feats["is_rna"] = (mt == 1).to(torch.int32)
        feats["is_dna"] = (mt == 2).to(torch.int32)
        feats["is_ligand"] = (mt == 3).to(torch.int32)

        # Ligand tokens must set is_atomized=1.
        if len(mol_types) != n_tokens:
            raise ValueError(
                f"token_mol_types length {len(mol_types)} != n_tokens {n_tokens}")
        mol_types_t = torch.tensor(mol_types, dtype=torch.int32)
        feats["is_atomized"] = (mol_types_t == MOL_TYPE_LIGAND).to(torch.int32)

        # token_mask: all valid
        feats["token_mask"] = torch.ones(n_tokens, dtype=torch.float32)

        # num_atoms_per_token
        feats["num_atoms_per_token"] = torch.tensor(atoms_per_tok,
                                                    dtype=torch.int32)

        # start_atom_index
        feats["start_atom_index"] = torch.tensor(start_atoms,
                                                 dtype=torch.int32)

        # atom_mask
        feats["atom_mask"] = torch.ones(n_atoms, dtype=torch.float32)

        # atom_to_token_index
        feats["atom_to_token_index"] = torch.tensor(atom_tok_idx,
                                                    dtype=torch.int32)

        # token_bonds: between atomized-only tokens (OSS
        # filter_fully_atomized_bonds in cleanup.py). For standard protein
        # / nucleotide residues is_atomized=False → no contribution. For
        # atomized ligand chains, every intra-ligand bond becomes a
        # symmetric token_bonds entry between the two atom-tokens
        # involved. We reconstruct this from the RDKit mol attached to
        # each ligand chain's first token.
        token_bonds = torch.zeros(n_tokens, n_tokens, dtype=torch.int32)
        residue_mols_ctx = row["structure"].get("residue_mols", [])
        token_mol_idx_ctx = row["structure"].get("token_mol_idx")
        if token_mol_idx_ctx is not None and residue_mols_ctx:
            # Group atomized tokens by their mol_idx (one ligand chain
            # per group). For each group, read RDKit bonds and convert
            # (atom-local-index pairs) -> (token-index pairs).
            from collections import defaultdict
            mol_idx_to_tokens: dict[int, list[int]] = defaultdict(list)
            for ti, mi in enumerate(token_mol_idx_ctx):
                # Only include atomized tokens (ligands)
                if feats["is_atomized"][ti].item() == 1:
                    mol_idx_to_tokens[mi].append(ti)
            for mi, ti_list in mol_idx_to_tokens.items():
                if len(ti_list) < 2:
                    continue
                first_t = ti_list[0]
                if first_t >= len(residue_mols_ctx):
                    continue
                mol = residue_mols_ctx[first_t]
                if mol is None:
                    continue
                # The atom indices in the mol are the position within the
                # ligand chain (sorted by addition order). The per_atom
                # crop mask in residue_crop_masks tells us which atom
                # in the mol each token represents.
                # Build a map atom_in_mol_idx -> token_idx
                atom_in_mol_to_token: dict[int, int] = {}
                crop_masks_ctx = row["structure"].get(
                    "residue_crop_masks", []
                )
                for ti in ti_list:
                    if ti >= len(crop_masks_ctx):
                        continue
                    mask = crop_masks_ctx[ti]
                    # mask is bool array; the True index is the atom-in-mol
                    true_indices = [i for i, v in enumerate(mask) if v]
                    if len(true_indices) == 1:
                        atom_in_mol_to_token[true_indices[0]] = ti
                # Iterate bonds in the mol and set token_bonds
                try:
                    for bond in mol.GetBonds():
                        a1 = bond.GetBeginAtomIdx()
                        a2 = bond.GetEndAtomIdx()
                        t1 = atom_in_mol_to_token.get(a1)
                        t2 = atom_in_mol_to_token.get(a2)
                        if t1 is not None and t2 is not None and t1 != t2:
                            token_bonds[t1, t2] = 1
                            token_bonds[t2, t1] = 1
                except Exception as e:
                    logger.warning(
                        "token_bonds extraction failed for ligand mol_idx=%s: %s",
                        mi, e, exc_info=True)
                    # Defensive — log but don't crash. The feature is set
                    # to zeros for this chain.
        feats["token_bonds"] = token_bonds

        return feats


class ConformerFeatureGenerator(FeatureGeneratorBase):
    """Generates reference conformer features using Biotite CCD + RDKit.

    Produces: ref_pos, ref_mask, ref_element, ref_charge,
    ref_atom_name_chars, ref_space_uid.

    Uses Biotite CCD for atom identity and RDKit ETKDGv3 for conformer
    coordinates, matching the OSS pipeline. Coordinates are centered
    per-residue with random augmentation (rotation + translation) via
    ``centre_random_augmentation`` (OSS AF3 Algorithm 19). Seed the global
    ``torch`` and ``random`` RNGs before inference if reproducibility is
    required.
    """

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        row = _get_row(context)
        struct = row["structure"]

        n_atoms = struct["n_atoms"]
        atom_names = struct["atom_names"]
        atom_elements = struct["atom_elements"]
        atom_token_idx = struct["atom_token_idx"]
        n_tokens = struct["n_tokens"]
        atoms_per_token = struct["atoms_per_token"]
        residue_mols = struct.get("residue_mols", [None] * n_tokens)
        residue_crop_masks = struct.get("residue_crop_masks", [])
        residue_atom_charges = struct.get("residue_atom_charges",
                                          [[0]] * n_tokens)

        feats: dict[str, torch.Tensor] = {}

        # ref_element: one-hot [N_atoms, 119]
        # Encode using atomic number - 1, matching upstream
        # ``featurize_reference_conformers_of3``, which calls
        # PERIODIC_TABLE.GetAtomicNumber(elem) via RDKit and so supports all
        # 118 elements. RDKit's periodic table is the source of truth here: a
        # short hardcoded element table silently encodes metals (MG, NI, ZN,
        # FE, ...) as carbon (index 5).
        from rdkit.Chem import GetPeriodicTable
        _pt = GetPeriodicTable()
        element_indices = []
        for elem in atom_elements:
            # OSS reserves index 118 for unknown placeholders ("R" symbol)
            sym = elem.strip()
            if sym in ("", "R", "*", "?"):
                element_indices.append(118)
                continue
            # Title-case (e.g., "MG" -> "Mg") for RDKit lookup.
            try:
                anum = _pt.GetAtomicNumber(sym.title())
            except Exception:
                # RDKit raises if symbol unknown; assign to unknown bin
                anum = 119  # → index 118 after -1
            element_indices.append(max(0, anum - 1))  # 0-indexed
        feats["ref_element"] = F.one_hot(
            torch.tensor(element_indices, dtype=torch.long),
            NUM_ELEMENT_CLASSES,
        ).to(torch.int32)

        # ref_charge: from RDKit formal charges
        all_charges: list[float] = []
        for tok_idx in range(n_tokens):
            charges = residue_atom_charges[tok_idx]
            all_charges.extend(charges)
        feats["ref_charge"] = torch.tensor(all_charges, dtype=torch.float32)

        # ref_atom_name_chars: [N_atoms, 4, 64]
        feats["ref_atom_name_chars"] = encode_atom_name_chars_one_hot(
            atom_names)

        # ref_space_uid: per-atom unique conformer instance ID. OSS sets
        # ref_space_uid = mol_idx (the index into processed_ref_mol_list).
        # For non-atomized residues this is per-residue (matches token_idx);
        # for atomized ligand chains all atoms of the chain share one
        # mol_idx (because there's one entry in processed_ref_mol_list per
        # ligand chain). See upstream ``featurize_reference_conformers_of3``
        # in ``core/data/pipelines/featurization/conformer.py``. The structure
        # dict carries token_mol_idx (per-token); we expand it per-atom here.
        token_mol_idx = struct.get("token_mol_idx")
        if token_mol_idx is not None:
            atom_mol_idx = [int(token_mol_idx[t]) for t in atom_token_idx]
            feats["ref_space_uid"] = torch.tensor(atom_mol_idx,
                                                  dtype=torch.int32)
        else:
            # Backward compat for callers that don't supply token_mol_idx:
            # fall back to a per-token uid. This is exact for structures
            # with no atomized (ligand) chains, where mol_idx and token_idx
            # coincide.
            feats["ref_space_uid"] = torch.tensor(atom_token_idx,
                                                  dtype=torch.int32)

        # ref_pos: from RDKit conformer coordinates, centred per residue
        ref_pos = torch.zeros(n_atoms, 3, dtype=torch.float32)
        ref_mask = torch.ones(n_atoms, dtype=torch.int32)

        atom_offset = 0
        for tok_idx in range(n_tokens):
            n_at = atoms_per_token[tok_idx]
            if n_at == 0:
                continue

            mol = residue_mols[tok_idx] if tok_idx < len(
                residue_mols) else None
            crop_mask = (residue_crop_masks[tok_idx]
                         if tok_idx < len(residue_crop_masks) else None)
            tok_pos = ref_pos[atom_offset:atom_offset + n_at]
            tok_mask = ref_mask[atom_offset:atom_offset + n_at]

            if mol is not None and mol.GetNumConformers() > 0:
                conf = mol.GetConformer()
                # Extract coords only for atoms in crop mask (skip OXT)
                out_idx = 0
                for ai in range(mol.GetNumAtoms()):
                    if crop_mask is not None and ai < len(crop_mask):
                        if not crop_mask[ai]:
                            continue
                    if out_idx >= n_at:
                        break
                    pt = conf.GetAtomPosition(ai)
                    tok_pos[out_idx] = torch.tensor([pt.x, pt.y, pt.z],
                                                    dtype=torch.float32)
                    out_idx += 1

            # Apply random centering + rotation + translation
            # (matching OSS centre_random_augmentation / AF3 Algorithm 19)
            augmented = centre_random_augmentation(tok_pos, tok_mask.float())
            ref_pos[atom_offset:atom_offset + n_at] = augmented
            atom_offset += n_at

        feats["ref_pos"] = ref_pos
        feats["ref_mask"] = ref_mask

        return feats


class MsaFeatureGenerator(FeatureGeneratorBase):
    """Generates MSA feature tensors from parsed MSA data.

    Produces: msa, has_deletion, deletion_value, deletion_mean,
    profile, num_paired_seqs, msa_mask.

    Uses a broadcast/max_rows architecture:
    - One MSA matrix of shape [max_rows, n_tokens] shared across all chains.
    - Per-polymer MSA (query + all-paired-rows + unpaired[1:]) is broadcast
      to all chains belonging to that polymer (same MSA entry by identity).
    - profile and deletion_mean are computed over unpaired MSA rows only (not paired)
      (including the query row, which has zero deletions).
    """

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        row = _get_row(context)
        struct = row["structure"]
        msa_per_chain = row.get("msa_per_chain", [])
        paired_msa_per_chain = row.get("paired_msa_per_chain", [])
        chain_sequences = row.get("chain_sequences", [])

        n_tokens = struct["n_tokens"]
        chain_ids = struct["token_chain_ids"]
        token_mol_types = struct["token_mol_types"]

        # Build per-chain token ranges (ordered by first appearance)
        unique_chains: list[str] = []
        chain_token_ranges: dict[str, tuple[int, int]] = {}
        seen_chains: set[str] = set()
        for i, cid in enumerate(chain_ids):
            if cid not in seen_chains:
                seen_chains.add(cid)
                unique_chains.append(cid)
                chain_token_ranges[cid] = (i, i)
            chain_token_ranges[cid] = (chain_token_ranges[cid][0], i + 1)

        # Group chains by polymer: chains sharing the same MSA entry object
        # (identical id()) belong to the same polymer and get the same MSA
        # broadcast to their token positions.
        polymer_groups: list[dict] = []
        msa_id_to_group: dict[int, int] = {}

        for chain_idx, cid in enumerate(unique_chains):
            msa_entry = msa_per_chain[chain_idx] if chain_idx < len(
                msa_per_chain) else None
            paired_entry = paired_msa_per_chain[chain_idx] if chain_idx < len(
                paired_msa_per_chain) else None
            # Group chains by MSA identity. Chains sharing an MSA dict get
            # the same eid (→ same polymer group). Chains with no MSA get a
            # unique eid via id(None) + chain_idx so they don't collapse
            # together.
            eid = id(
                msa_entry) if msa_entry is not None else id(None) + chain_idx

            if eid in msa_id_to_group:
                polymer_groups[msa_id_to_group[eid]]["chains"].append(
                    (chain_idx, cid))
            else:
                msa_id_to_group[eid] = len(polymer_groups)
                polymer_groups.append({
                    "chains": [(chain_idx, cid)],
                    "msa_entry": msa_entry,
                    "paired_entry": paired_entry,
                    "chain_idx": chain_idx,
                })

        # Build per-polymer MSA row lists and compute statistics
        max_rows = 1
        total_n_paired = 0
        polymer_data: list[dict] = []

        for group in polymer_groups:
            first_chain_idx, first_cid = group["chains"][0]
            msa_entry = group["msa_entry"]
            paired_entry = group["paired_entry"]
            start, end = chain_token_ranges[first_cid]
            n_res = end - start
            seq = chain_sequences[first_chain_idx] if first_chain_idx < len(
                chain_sequences) else ""

            # Determine mol_type for this polymer group from the first token
            # in its chain range (all tokens in a chain share the same mol_type).
            # Used by _resolve_msa_char for polymer-type-aware MSA encoding.
            group_mol_type = (token_mol_types[start]
                              if start < len(token_mol_types) else 0)

            poly_rows: list[list[int]] = []
            poly_dels: list[list[int]] = []

            # Track whether THIS polymer group has any MSA content. When False,
            # OSS leaves the chain's token slots at the pre-allocated gap fill
            # AND emits a zero profile + zero deletion_mean for those slots
            # (see ``create_msa_feature_precursor_of3`` in
            # ``core/data/primitives/featurization/msa.py`` — the "else"
            # branch when chain_id_to_query_seq is empty). When True, row 0 is
            # the first row of the MSA file (paired or main), NOT a
            # sequence-derived row.
            has_msa_for_polymer = (
                msa_entry is not None
                and len(msa_entry.get("sequences", [])) > 0
            ) or (
                paired_entry is not None
                and len(paired_entry.get("sequences", [])) > 0
            )

            # Row 0: query — upstream `chain_id_to_query_seq[chain_id]` is
            # populated as `all_msas_per_chain[first_key].msa[0, :]` (see
            # `parse_msas_sample_inference` in
            # `core/data/io/sequence/msa.py`), i.e., the MSA file's first row
            # cropped to n_tokens — NOT the polymer's input query sequence.
            # When the file row 0 differs from the polymer query (e.g. the
            # a3m carries extra prefix/suffix residues, so row 0 is wider
            # than the query), upstream still uses the file row 0 (cropped
            # to n_tokens). We mirror that contract here.
            #
            # When the polymer has no MSA AT ALL, upstream emits a single
            # all-gap row (the `create_msa_feature_precursor_of3` "else"
            # branch). TRT-BNM mirrors that path.
            #
            # The unpaired loop below starts at row 0, not row 1, so the
            # file's row 0 (the query duplicate per a3m convention) is also
            # added to the main MSA section. This matches upstream's behavior
            # of having the query row + the file's row 0 both present in the
            # final MSA.
            if has_msa_for_polymer and msa_entry is not None and msa_entry.get(
                    "sequences"):
                qseq = msa_entry["sequences"][0]
                qrow = [
                    _resolve_msa_char(c, group_mol_type) for c in qseq[:n_res]
                ]
                qrow += [GAP_IDX] * (n_res - len(qrow))
                poly_rows.append(qrow)
                poly_dels.append([0] * n_res)
            elif has_msa_for_polymer:
                # Has paired MSA but no main MSA — use paired row 0 as query
                # (matches OSS's behavior of `all_msas_per_chain[first_key]`
                # ordering: paired comes first in `aln_order`).
                pseq = paired_entry["sequences"][0]
                qrow = [
                    _resolve_msa_char(c, group_mol_type) for c in pseq[:n_res]
                ]
                qrow += [GAP_IDX] * (n_res - len(qrow))
                poly_rows.append(qrow)
                poly_dels.append([0] * n_res)
            else:
                # No MSA for this polymer — emit an all-gap row. The chain's
                # token slots will read as GAP_IDX in the final MSA tensor,
                # matching upstream `create_msa_feature_precursor_of3`.
                poly_rows.append([GAP_IDX] * n_res)
                poly_dels.append([0] * n_res)

            # ------------------------------------------------------------
            # Paired-MSA semantics, matching upstream.
            #
            # The symbol names below refer to upstream OpenFold-3
            # (github.com/aqlaboratory/openfold-3) at revision f16647af,
            # which is the revision this pipeline was validated against.
            # The `3rdparty/openfold-3` submodule is pinned to a different
            # commit, so these symbols may differ or be absent at the
            # current pin. The MSA row-count bookkeeping was refactored
            # upstream after f16647af — `n_rows_paired_subsampled` became
            # `n_rows_paired_cropped` and is now computed from the actual
            # paired-row count rather than left at 0 — so re-validating
            # against a newer upstream requires re-deriving the three
            # consequences listed here.
            # ------------------------------------------------------------
            # Upstream `MsaSampleProcessorInference.create_paired_msa` calls
            # `create_paired_from_precomputed` for precomputed paired MSAs.
            # That function does NOT call `msa_array_collection.set_row_counts(
            # n_rows_paired_subsampled=...)` — only the ONLINE pairing path in
            # `create_paired` does. So whenever the paired MSA is
            # precomputed — which is always the case for this pipeline —
            # `n_rows_paired_subsampled` stays at its default value of 0 (see
            # `MsaArrayCollection`).
            #
            # Consequences:
            #   1. `vstack_pad_msa_arrays`
            #      gates the paired-MSA vstack on `n_rows_paired_subsampled > 0`
            #      — so for precomputed paired the paired rows are NEVER vstacked
            #      into the final per-chain MSA. Only `[query] + [main_filtered]`
            #      end up in the output tensor.
            #   2. In `create_main`, the paired MSA is still used to dedup main
            #      rows via the `is_unique` filter: main rows whose byte-exact
            #      value equals any paired row are dropped from the final main
            #      MSA.
            #   3. The main cap becomes `max_rows - n_rows_paired_subsampled - 1
            #      = max_rows - 1` (since the counter is 0).
            #
            # Profile / deletion_mean are computed from `main_msa_redundant`
            # (the pre-filter, pre-cap, full-width main MSA) and column-indexed
            # to the polymer's res_ids only at featurization time
            # (`map_msas_to_tokens` in
            # `core/data/primitives/featurization/msa.py`). For samples where
            # the file's aligned column count > n_res (say a 136-column a3m
            # for a 117-residue polymer), the full-width-then-crop
            # matters: upstream's `np.repeat` profile uses
            # `block_n_cols * n_symbols` as the bin stride, so the scrambling
            # pattern depends on the FULL file width, not the cropped width.
            # ------------------------------------------------------------

            # Build paired rows at FILE WIDTH (not n_res) so the is_unique
            # comparison matches OSS — OSS compares paired_msa vs
            # main_msa_redundant where BOTH are at file_aligned_width.
            #
            # The paired pool is truncated to MAX_MSA_ROWS_PAIRED before the
            # is_unique filter — matching OSS `create_paired_from_precomputed`
            # which calls `prepaired_msa.truncate(max_rows_paired)` BEFORE the
            # filter is applied. Without this truncation we over-dedup main
            # rows that upstream would have kept
            # (because their byte-exact matches sit beyond paired_idx=2047).
            from .const import MAX_MSA_ROWS_PAIRED
            paired_rows_full: list[list[int]] = []
            paired_full_width = 0
            if paired_entry is not None:
                paired_seqs = paired_entry.get("sequences", [])
                paired_seqs = paired_seqs[:MAX_MSA_ROWS_PAIRED]
                paired_full_width = max(
                    (len(s) for s in paired_seqs), default=0
                )
                for pseq in paired_seqs:
                    prow_full = [GAP_IDX] * paired_full_width
                    for j in range(min(len(pseq), paired_full_width)):
                        prow_full[j] = _resolve_msa_char(
                            pseq[j], group_mol_type
                        )
                    paired_rows_full.append(prow_full)

            # Build main MSA rows at FILE WIDTH for is_unique / profile.
            # `unpaired_rows_full` / `unpaired_dels_full` mirror OSS
            # `main_msa_redundant` exactly (pre-filter, pre-crop).
            unpaired_rows_full: list[list[int]] = []
            unpaired_dels_full: list[list[int]] = []
            main_full_width = 0
            if msa_entry is not None:
                msa_seqs = msa_entry.get("sequences", [])
                msa_raw = msa_entry.get("raw", msa_seqs)
                main_full_width = max(
                    (len(s) for s in msa_seqs), default=0
                )
                for seq_idx in range(len(msa_seqs)):
                    useq = msa_seqs[seq_idx]
                    uraw = msa_raw[seq_idx] if seq_idx < len(
                        msa_raw) else useq
                    del_counts = _extract_deletion_counts(uraw)
                    urow_full = [GAP_IDX] * main_full_width
                    drow_full = [0] * main_full_width
                    for j in range(min(len(useq), main_full_width)):
                        urow_full[j] = _resolve_msa_char(
                            useq[j], group_mol_type
                        )
                        if j < len(del_counts):
                            drow_full[j] = del_counts[j]
                    unpaired_rows_full.append(urow_full)
                    unpaired_dels_full.append(drow_full)

            # `is_unique` filter — drop main rows whose byte-exact encoded
            # value matches any paired row. OSS pads both to the same width
            # implicitly (paired and main come from the same per-chain
            # alignment in OSS's data model). When the two file widths
            # differ we right-pad with GAP_IDX so the comparison is
            # well-defined; this matches OSS's behavior since the
            # `chain_data[aln].msa` arrays for a single chain all have the
            # same column count (the chain's aligned width).
            is_unique_mask: list[bool] | None = None
            if paired_rows_full and unpaired_rows_full:
                # Build numpy arrays at a common width = max(main_w, paired_w).
                common_w = max(main_full_width, paired_full_width)
                main_arr = np.full(
                    (len(unpaired_rows_full), common_w),
                    GAP_IDX, dtype=np.int64,
                )
                for i, r in enumerate(unpaired_rows_full):
                    main_arr[i, :len(r)] = r
                paired_arr = np.full(
                    (len(paired_rows_full), common_w),
                    GAP_IDX, dtype=np.int64,
                )
                for i, r in enumerate(paired_rows_full):
                    paired_arr[i, :len(r)] = r
                # Match OSS `np.isin` on void-view trick: each row becomes
                # a single void item with size n_cols * itemsize.
                main_view = main_arr.view(
                    np.dtype((np.void, main_arr.dtype.itemsize * common_w))
                )
                paired_view = paired_arr.view(
                    np.dtype((np.void, paired_arr.dtype.itemsize * common_w))
                )
                is_unique_arr = np.squeeze(
                    ~np.isin(main_view, paired_view), axis=-1
                )
                is_unique_mask = is_unique_arr.tolist()
            elif unpaired_rows_full:
                is_unique_mask = [True] * len(unpaired_rows_full)

            # Filter main rows for the final MSA output (poly_rows). OSS
            # vstacks `[query] + [main_filtered]` only — paired is NOT
            # vstacked when `n_rows_paired_subsampled == 0` (the
            # precomputed-paired path leaves the counter at 0).
            #
            # Main cap in OSS: `n_rows_main_msa_lim = max(0, max_rows -
            # n_rows_paired_subsampled - 1) = max_rows - 1` for our path.
            main_cap = max(0, MAX_MSA_ROWS - 1)
            n_main_appended = 0
            if msa_entry is not None and unpaired_rows_full and is_unique_mask:
                for seq_idx in range(len(unpaired_rows_full)):
                    if not is_unique_mask[seq_idx]:
                        continue
                    if n_main_appended >= main_cap:
                        break
                    # Crop the filtered row to n_res for the output tensor.
                    # Mirrors upstream featurization-time
                    # `msa_array_vstack.msa[:, msa_column_positions]` in
                    # `map_msas_to_tokens`, where
                    # `msa_column_positions = res_id - 1`.
                    urow_full = unpaired_rows_full[seq_idx]
                    drow_full = unpaired_dels_full[seq_idx]
                    urow_cropped = urow_full[:n_res]
                    drow_cropped = drow_full[:n_res]
                    # Right-pad with GAP_IDX / 0 if file row is shorter.
                    if len(urow_cropped) < n_res:
                        urow_cropped = urow_cropped + [GAP_IDX] * (
                            n_res - len(urow_cropped)
                        )
                        drow_cropped = drow_cropped + [0] * (
                            n_res - len(drow_cropped)
                        )
                    poly_rows.append(urow_cropped)
                    poly_dels.append(drow_cropped)
                    n_main_appended += 1

            n_poly_rows = len(poly_rows)
            if n_poly_rows > max_rows:
                max_rows = n_poly_rows
            # `n_paired_poly` is kept at 0 because OSS does NOT vstack paired
            # into the final MSA when `n_rows_paired_subsampled == 0`.
            total_n_paired = max(total_n_paired, 0)

            # Profile and deletion_mean: computed from the FULL-WIDTH main MSA
            # (`unpaired_rows_full` ↔ OSS `main_msa_redundant`) at the file's
            # native aligned-column width, THEN cropped to n_res. This matches
            # OSS's order of operations:
            #   1. `calculate_profile(main_msa_redundant)` on full file width
            #      (in `MsaSampleProcessor.create_query_seq`).
            #   2. `profile[msa_column_positions, :]` at featurization
            #      (in `calculate_profile_del_mean`) where
            #      `msa_column_positions = res_id - 1`.
            #
            # When the polymer has no MSA at all, OSS emits zero profile and
            # zero deletion_mean (`create_msa_feature_precursor_of3` uses
            # np.zeros for both fields in the no-MSA "else" branch).
            if has_msa_for_polymer and unpaired_rows_full:
                # deletion_mean: full-width then crop. Each chain's polymer
                # has res_ids = [1..n_res] (sequential), so cropping to
                # [0:n_res] matches `del_mean[msa_column_positions]`.
                del_t_full = torch.tensor(
                    unpaired_dels_full, dtype=torch.float32
                )
                deletion_mean_full = del_t_full.mean(dim=0)  # [main_full_width]
                if deletion_mean_full.numel() >= n_res:
                    deletion_mean = deletion_mean_full[:n_res].clone()
                else:
                    deletion_mean = torch.zeros(n_res, dtype=torch.float32)
                    deletion_mean[:deletion_mean_full.numel()] = (
                        deletion_mean_full
                    )

                # Intentionally reproduces upstream `calculate_profile`
                # (`core/data/primitives/sequence/msa.py`), which uses
                # np.repeat where np.tile would be required for a row-major
                # ravel. The result is a column-permuted background-frequency
                # profile rather than a per-column distribution. The released
                # weights were trained against that permuted profile, so
                # matching it bit-for-bit is required for parity — do not
                # "fix" this to np.tile.
                #
                # The profile is computed on the FULL file width and only then
                # cropped to n_res. Computing it on already-cropped rows
                # yields different scrambling whenever
                # file_aligned_len != polymer_len.
                msa_idx_arr = np.asarray(
                    unpaired_rows_full, dtype=np.int64
                )
                n_rows_msa, n_cols_full = msa_idx_arr.shape
                n_symbols = NUM_MSA_CLASSES
                # Upstream chunk_size = 1000 (in `create_query_seq`)
                chunk_size = 1000
                counts_full = np.zeros(
                    (n_cols_full, n_symbols), dtype=np.int64
                )
                col_start = 0
                while col_start < n_cols_full:
                    col_end = min(col_start + chunk_size, n_cols_full)
                    msa_chunk = msa_idx_arr[:, col_start:col_end]
                    block_n_cols = col_end - col_start
                    val_indices = msa_chunk.ravel()  # row-major
                    # np.repeat (not np.tile) is deliberate here — see the
                    # profile-parity note above.
                    col_indices_local = np.repeat(
                        np.arange(block_n_cols), n_rows_msa
                    )
                    to_count_local = (
                        col_indices_local * n_symbols + val_indices
                    )
                    chunk_counts_1d = np.bincount(
                        to_count_local,
                        minlength=block_n_cols * n_symbols,
                    )
                    chunk_counts_2d = chunk_counts_1d.reshape(
                        block_n_cols, n_symbols
                    )
                    counts_full[col_start:col_end, :] += chunk_counts_2d
                    col_start = col_end
                profile_full = counts_full / n_rows_msa
                # Crop to n_res (col indices [0..n_res-1] = res_id - 1).
                if profile_full.shape[0] >= n_res:
                    profile_cropped = profile_full[:n_res, :]
                else:
                    profile_cropped = np.zeros(
                        (n_res, n_symbols), dtype=profile_full.dtype
                    )
                    profile_cropped[
                        :profile_full.shape[0], :
                    ] = profile_full
                profile = torch.from_numpy(
                    profile_cropped.astype(np.float32)
                )
            else:
                deletion_mean = torch.zeros(n_res, dtype=torch.float32)
                profile = torch.zeros(
                    n_res, NUM_MSA_CLASSES, dtype=torch.float32
                )

            polymer_data.append({
                "chains": group["chains"],
                "poly_rows": poly_rows,
                "poly_dels": poly_dels,
                "n_rows": n_poly_rows,
                "deletion_mean": deletion_mean,
                "profile": profile,
                "n_res": n_res,
            })

        # Build the global [max_rows, n_tokens] MSA matrix.
        #
        # msa_mask semantics: `global_mask` is 1.0 everywhere by default and
        # is zeroed only where the token itself is padding. It is NOT gated on
        # each polymer's actual row count — masking rows beyond a chain's row
        # count would diverge from upstream, where
        # `create_msa_feature_precursor_of3`
        # (`core/data/primitives/featurization/msa.py`) initializes `msa_mask`
        # to all 1s and zeros it only via the token-validity mask
        # (`token_mask[np.newaxis, :]`).
        global_msa = torch.full((max_rows, n_tokens),
                                GAP_IDX,
                                dtype=torch.long)
        global_del = torch.zeros(max_rows, n_tokens, dtype=torch.long)
        # Upstream initialization: msa_mask = 1.0 everywhere.
        global_mask = torch.ones(max_rows, n_tokens, dtype=torch.float32)
        global_profile = torch.zeros(n_tokens,
                                     NUM_MSA_CLASSES,
                                     dtype=torch.float32)
        global_del_mean = torch.zeros(n_tokens, dtype=torch.float32)

        for pd in polymer_data:
            r = pd["n_rows"]
            rows_t = torch.tensor(pd["poly_rows"],
                                  dtype=torch.long)  # [r, n_res]
            dels_t = torch.tensor(pd["poly_dels"],
                                  dtype=torch.long)  # [r, n_res]

            # Broadcast to ALL chains of this polymer
            for _, cid in pd["chains"]:
                s, e = chain_token_ranges[cid]
                global_msa[:r, s:e] = rows_t
                global_del[:r, s:e] = dels_t
                global_profile[s:e] = pd["profile"]
                global_del_mean[s:e] = pd["deletion_mean"]

        feats: dict[str, torch.Tensor] = {}
        feats["msa"] = encode_one_hot(global_msa,
                                      NUM_MSA_CLASSES).to(torch.int32)
        feats["has_deletion"] = (global_del != 0).to(torch.float32)
        feats["deletion_value"] = compute_deletion_value(global_del)
        feats["deletion_mean"] = global_del_mean
        feats["profile"] = global_profile
        # At upstream OpenFold-3 (github.com/aqlaboratory/openfold-3)
        # revision f16647af — the revision this pipeline was validated
        # against, which is not the `3rdparty/openfold-3` submodule pin —
        # n_rows_paired_subsampled is only updated when
        # ONLINE paired-MSA pairing runs. With PRECOMPUTED paired MSAs — the
        # path taken here — the counter stays at its default 0, so upstream
        # num_paired_seqs reads as `0 + 1 = 1`. We mirror that contract here:
        # always emit 1 regardless of the actual loaded paired-row count.
        # See the paired-MSA semantics note earlier in this method — this
        # hardcoded 1 is tied to that revision's row-count bookkeeping.
        feats["num_paired_seqs"] = torch.tensor([1], dtype=torch.int32)
        feats["msa_mask"] = global_mask

        return feats


def _extract_deletion_counts(raw_seq: str) -> list[int]:
    """Extract per-position deletion counts from a raw A3M sequence."""
    counts = []
    del_count = 0
    for char in raw_seq:
        if char.islower():
            del_count += 1
        else:
            counts.append(del_count)
            del_count = 0
    return counts


class TemplateFeatureGenerator(FeatureGeneratorBase):
    """Generates OF3 template features (``featurize_template_structures_of3``).

    Two paths, both producing ``n_templ = DEFAULT_N_TEMPLATES`` slots:

    * **No templates supplied** (common case): the no-template placeholder —
      restype one-hot at the GAP class, all-zero masks/coords/distogram/
      unit-vector, byte-identical to OSS no-template inference.
    * **Direct-CIF templates supplied** (protein only): parse each CIF, align its
      best chain to the query via kalign, build per-token pseudo-beta /
      backbone-frame precursors, then apply the OSS featurization math with
      inter/intra-chain masking. See ``template_logic.py``.
    """

    def is_enabled(self) -> bool:
        return True  # Always produces template features (placeholder or real)

    def _no_template_feats(self, n_tokens: int) -> dict[str, torch.Tensor]:
        n_templ = DEFAULT_N_TEMPLATES
        feats: dict[str, torch.Tensor] = {}
        template_restype = torch.zeros(n_templ,
                                       n_tokens,
                                       NUM_RESTYPE_CLASSES,
                                       dtype=torch.int32)
        template_restype[..., GAP_IDX] = 1
        feats["template_restype"] = template_restype
        feats["template_pseudo_beta_mask"] = torch.zeros(n_templ,
                                                         n_tokens,
                                                         dtype=torch.float32)
        feats["template_backbone_frame_mask"] = torch.zeros(
            n_templ, n_tokens, dtype=torch.float32)
        feats["template_distogram"] = torch.zeros(n_templ,
                                                  n_tokens,
                                                  n_tokens,
                                                  TEMPLATE_DISTOGRAM_N_BINS,
                                                  dtype=torch.float32)
        feats["template_unit_vector"] = torch.zeros(n_templ,
                                                    n_tokens,
                                                    n_tokens,
                                                    3,
                                                    dtype=torch.float32)
        return feats

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        n_tokens = batch["token_index"].shape[0]
        row = _get_row(context)
        templates_per_chain = row.get("templates_per_chain") or {}
        if not templates_per_chain:
            return self._no_template_feats(n_tokens)

        from .common import (create_template_distogram, create_template_restype,
                             create_template_unit_vector)
        from .const import (MOL_TYPE_PROTEIN, TEMPLATE_CIF_DIRECT_MIN_SCORE,
                            TEMPLATE_DISTOGRAM_INF_VALUE,
                            TEMPLATE_DISTOGRAM_MAX_BIN,
                            TEMPLATE_DISTOGRAM_MIN_BIN,
                            TEMPLATE_MIN_TOKENS_PER_CHAIN, TEMPLATE_TAKE_TOP_K)
        from .template_logic import (fill_precursor_for_chain,
                                     resolve_template_idx_map,
                                     select_template_for_cif)

        struct = row["structure"]
        token_chain_ids = struct["token_chain_ids"]
        token_res_ids = struct["token_res_ids"]
        token_mol_types = struct["token_mol_types"]
        template_query_seq = row.get("template_query_seq") or {}

        n_templ = DEFAULT_N_TEMPLATES
        res_names = np.full((n_templ, n_tokens), "GAP", dtype=np.dtype("U3"))
        pb_coords = np.full((n_templ, n_tokens, 3), np.nan, dtype=np.float64)
        frame_coords = np.full((n_templ, n_tokens, 3, 3),
                               np.nan,
                               dtype=np.float64)

        # Group protein tokens by their original chain_id, preserving order.
        for cid in dict.fromkeys(token_chain_ids):
            token_pos = [
                i for i in range(n_tokens)
                if token_chain_ids[i] == cid
                and token_mol_types[i] == MOL_TYPE_PROTEIN
            ]
            if len(token_pos) < TEMPLATE_MIN_TOKENS_PER_CHAIN:
                continue
            templates = templates_per_chain.get(cid)
            if not templates:
                continue
            query_seq = template_query_seq.get(cid, "")
            token_pos_by_res_id = {
                int(token_res_ids[i]): i
                for i in token_pos
            }

            selected = []
            for tmpl in templates:
                content = tmpl.get("content")
                if content is None:
                    continue
                sel = select_template_for_cif(
                    query_seq=query_seq,
                    content=content,
                    fmt=tmpl.get("format", "cif"),
                    specified_chain_id=tmpl.get("chain_id"),
                    min_score=TEMPLATE_CIF_DIRECT_MIN_SCORE,
                )
                if sel is not None:
                    selected.append(sel)

            # Inference takes the top-k templates by alignment score.
            selected.sort(key=lambda s: s.score, reverse=True)
            if not TEMPLATE_TAKE_TOP_K:  # pragma: no cover - inference is top-k
                pass
            topk = selected[:n_templ]

            # Keep/drop per OSS ``map_token_pos_to_template_residues`` (see
            # ``resolve_template_idx_map``). OSS takes top-k first, then drops —
            # dropped templates are not backfilled, so survivors pack into the
            # lowest free slots in score order.
            chain_len = len(token_pos)
            slot = 0
            for sel in topk:
                if slot >= n_templ:
                    break
                eff_idx = resolve_template_idx_map(sel, chain_len)
                if eff_idx is None:
                    continue  # dropped: consumes no template slot
                if eff_idx.shape[0] > 0:
                    fill_precursor_for_chain(sel, slot, eff_idx,
                                             token_pos_by_res_id, res_names,
                                             pb_coords, frame_coords)
                # A kept template still occupies a slot (all-GAP if empty).
                slot += 1

        # A pseudo-beta / backbone frame is present iff its coords are not NaN.
        pb_mask = torch.tensor(~np.isnan(pb_coords).any(axis=-1),
                               dtype=torch.float32)
        bb_mask = torch.tensor(~np.isnan(frame_coords).any(axis=(-2, -1)),
                               dtype=torch.float32)

        # Inter/intra-chain pair mask from asym_id, shaped [1, N, N, 1] to
        # broadcast over templates and the last feature dim.
        asym = np.asarray(_renumber_chain_ids(token_chain_ids))
        mc_pair = torch.tensor(
            (asym[:, None] == asym[None, :]).astype(np.float32),
            dtype=torch.float32)[None, :, :, None]

        feats: dict[str, torch.Tensor] = {}
        feats["template_restype"] = create_template_restype(
            res_names, pb_mask, RESNAME_TO_IDX, UNK_IDX, NUM_RESTYPE_CLASSES)
        feats["template_pseudo_beta_mask"] = pb_mask
        feats["template_backbone_frame_mask"] = bb_mask
        feats["template_distogram"] = create_template_distogram(
            pb_coords,
            pb_mask,
            mc_pair,
            TEMPLATE_DISTOGRAM_MIN_BIN,
            TEMPLATE_DISTOGRAM_MAX_BIN,
            TEMPLATE_DISTOGRAM_N_BINS,
            TEMPLATE_DISTOGRAM_INF_VALUE,
        )
        feats["template_unit_vector"] = create_template_unit_vector(
            frame_coords, bb_mask, mc_pair)
        return feats
