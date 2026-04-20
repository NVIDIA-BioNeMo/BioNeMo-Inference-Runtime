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

from typing import Any

import torch
import torch.nn.functional as F

from tensorrt_bionemo.pipeline.base import FeatureGeneratorBase

from .common import (centre_random_augmentation, compute_deletion_value,
                     encode_atom_name_chars_one_hot, encode_one_hot)
from .const import (DEFAULT_N_TEMPLATES, ELEMENT_ATOMIC_NUMBER, GAP_IDX,
                    MSA_CHAR_TO_IDX, NUM_ELEMENT_CLASSES, NUM_MSA_CLASSES,
                    NUM_RESTYPE_CLASSES, RESNAME_TO_IDX,
                    TEMPLATE_DISTOGRAM_N_BINS, UNK_IDX)
from .feature_context import _compute_sym_ids, _renumber_chain_ids


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

        # is_atomized: 0 for all standard protein residues
        feats["is_atomized"] = torch.zeros(n_tokens, dtype=torch.int32)

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

        # token_bonds: all zeros for standard proteins.
        # OSS filters to atomized-only bonds via filter_fully_atomized_bonds().
        # Since all standard protein residues have is_atomized=False, ALL bonds
        # are filtered out, producing an all-zeros matrix.
        feats["token_bonds"] = torch.zeros(n_tokens,
                                           n_tokens,
                                           dtype=torch.int32)

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
        # Encode using atomic number - 1 (matching OSS PERIODIC_TABLE logic)
        element_indices = []
        for elem in atom_elements:
            anum = ELEMENT_ATOMIC_NUMBER.get(elem.upper(), 6)  # default C
            element_indices.append(anum - 1)  # 0-indexed
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

        # ref_space_uid: token index for each atom
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

    Uses a broadcast/max_rows architecture matching the ductr reference:
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

            poly_rows: list[list[int]] = []
            poly_dels: list[list[int]] = []

            # Row 0: query (from unpaired MSA first row if available, else sequence)
            if msa_entry is not None:
                msa_seqs = msa_entry.get("sequences", [])
                if msa_seqs:
                    qseq = msa_seqs[0]
                    qrow = [
                        MSA_CHAR_TO_IDX.get(c, UNK_IDX) for c in qseq[:n_res]
                    ]
                    qrow += [GAP_IDX] * (n_res - len(qrow))
                    poly_rows.append(qrow)
                    poly_dels.append([0] * n_res)
                else:
                    qrow = [
                        MSA_CHAR_TO_IDX.get(c, UNK_IDX) for c in seq[:n_res]
                    ]
                    qrow += [GAP_IDX] * (n_res - len(qrow))
                    poly_rows.append(qrow)
                    poly_dels.append([0] * n_res)
            else:
                qrow = [MSA_CHAR_TO_IDX.get(c, UNK_IDX) for c in seq[:n_res]]
                qrow += [GAP_IDX] * (n_res - len(qrow))
                poly_rows.append(qrow)
                poly_dels.append([0] * n_res)

            # Paired rows: ALL rows from the paired file (including its own query row)
            n_paired_poly = 0
            if paired_entry is not None:
                paired_seqs = paired_entry.get("sequences", [])
                paired_raw = paired_entry.get("raw", paired_seqs)
                for seq_idx in range(len(paired_seqs)):
                    prow = [GAP_IDX] * n_res
                    draw = [0] * n_res
                    praw = paired_raw[seq_idx] if seq_idx < len(
                        paired_raw) else paired_seqs[seq_idx]
                    del_counts = _extract_deletion_counts(praw)
                    pseq = paired_seqs[seq_idx]
                    for j in range(min(len(pseq), n_res)):
                        prow[j] = MSA_CHAR_TO_IDX.get(pseq[j], UNK_IDX)
                        if j < len(del_counts):
                            draw[j] = del_counts[j]
                    poly_rows.append(prow)
                    poly_dels.append(draw)
                    n_paired_poly += 1

            # Unpaired non-query rows from the unpaired MSA file
            unpaired_rows: list[list[int]] = [poly_rows[0]]
            unpaired_dels: list[list[int]] = [poly_dels[0]]
            if msa_entry is not None:
                msa_seqs = msa_entry.get("sequences", [])
                msa_raw = msa_entry.get("raw", msa_seqs)
                for seq_idx in range(1, len(msa_seqs)):
                    urow = [GAP_IDX] * n_res
                    draw = [0] * n_res
                    uraw = msa_raw[seq_idx] if seq_idx < len(
                        msa_raw) else msa_seqs[seq_idx]
                    del_counts = _extract_deletion_counts(uraw)
                    useq = msa_seqs[seq_idx]
                    for j in range(min(len(useq), n_res)):
                        urow[j] = MSA_CHAR_TO_IDX.get(useq[j], UNK_IDX)
                        if j < len(del_counts):
                            draw[j] = del_counts[j]
                    poly_rows.append(urow)
                    poly_dels.append(draw)
                    unpaired_rows.append(urow)
                    unpaired_dels.append(draw)

            n_poly_rows = len(poly_rows)
            if n_poly_rows > max_rows:
                max_rows = n_poly_rows
            total_n_paired = max(total_n_paired, n_paired_poly)

            # Profile and deletion_mean: unpaired MSA only (matches OSS/backup)
            del_t = torch.tensor(unpaired_dels, dtype=torch.float32)
            deletion_mean = del_t.mean(dim=0)  # [n_res]

            msa_idx_t = torch.tensor(unpaired_rows, dtype=torch.long)
            msa_oh = encode_one_hot(msa_idx_t, NUM_MSA_CLASSES).float()
            profile = msa_oh.mean(dim=0)  # [n_res, 32]

            polymer_data.append({
                "chains": group["chains"],
                "poly_rows": poly_rows,
                "poly_dels": poly_dels,
                "n_rows": n_poly_rows,
                "deletion_mean": deletion_mean,
                "profile": profile,
                "n_res": n_res,
            })

        # Build the global [max_rows, n_tokens] MSA matrix
        # msa_mask is 0 for positions beyond this polymer's row count
        global_msa = torch.full((max_rows, n_tokens),
                                GAP_IDX,
                                dtype=torch.long)
        global_del = torch.zeros(max_rows, n_tokens, dtype=torch.long)
        global_mask = torch.zeros(max_rows, n_tokens, dtype=torch.float32)
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
                global_mask[:r, s:e] = 1.0
                global_profile[s:e] = pd["profile"]
                global_del_mean[s:e] = pd["deletion_mean"]

        feats: dict[str, torch.Tensor] = {}
        feats["msa"] = encode_one_hot(global_msa,
                                      NUM_MSA_CLASSES).to(torch.int32)
        feats["has_deletion"] = (global_del != 0).to(torch.float32)
        feats["deletion_value"] = compute_deletion_value(global_del)
        feats["deletion_mean"] = global_del_mean
        feats["profile"] = global_profile
        feats["num_paired_seqs"] = torch.tensor([total_n_paired + 1],
                                                dtype=torch.int32)
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
    """Generates dummy template features matching OSS ``featurize_templates_dummy_of3``.

    Templates are not wired through the TRT-BNM parser/schema yet. This
    generator produces one-filled tensors in the expected shapes and dtypes,
    matching the OSS no-template inference path byte-for-byte.
    """

    def is_enabled(self) -> bool:
        return True  # Always produce dummy features

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        n_tokens = batch["token_index"].shape[0]
        n_templ = DEFAULT_N_TEMPLATES

        # OSS featurize_templates_dummy_of3() fills with all-ones.
        feats: dict[str, torch.Tensor] = {}
        feats["template_restype"] = torch.ones(n_templ,
                                               n_tokens,
                                               NUM_RESTYPE_CLASSES,
                                               dtype=torch.int32)
        feats["template_pseudo_beta_mask"] = torch.ones(n_templ,
                                                        n_tokens,
                                                        dtype=torch.float32)
        feats["template_backbone_frame_mask"] = torch.ones(n_templ,
                                                           n_tokens,
                                                           dtype=torch.float32)
        feats["template_distogram"] = torch.ones(n_templ,
                                                 n_tokens,
                                                 n_tokens,
                                                 TEMPLATE_DISTOGRAM_N_BINS,
                                                 dtype=torch.int32)
        feats["template_unit_vector"] = torch.ones(n_templ,
                                                   n_tokens,
                                                   n_tokens,
                                                   3,
                                                   dtype=torch.float32)

        return feats
