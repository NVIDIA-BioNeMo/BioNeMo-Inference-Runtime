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
"""Boltz2 PostProcessor: convert raw model output into :class:`FoldingOutput`.

Maps Boltz2's atom-level predictions to the standard ``FoldingOutput`` schema
used across the pipeline.  Atom coordinates are remapped from Boltz2's
variable per-token layout into the standard 37-atom-type scheme
(``AtomTypes.all_types()``) so that downstream writers (``PDBWriter``,
``CIFWriter``) work without modification.
"""

from typing import Any

import numpy as np
import torch
from pydantic import BaseModel

from bionemo_ir.data.schemas import FoldingOutput
from bionemo_ir.data.schemas.basic import MOL_TYPE_DNA, MOL_TYPE_LIGAND, MOL_TYPE_PROTEIN, MOL_TYPE_RNA, AtomTypes
from bionemo_ir.pipeline.base import PostProcessorBase
from bionemo_ir.pipeline.models.boltz2.const import chain_type_ids, tokens

NUM_ATOM_TYPES = len(AtomTypes.all_types())  # 37

_ATOM_NAME_TO_IDX: dict[str, int] = {at.name: i for i, at in enumerate(AtomTypes.all_types())}

# Remap Boltz2's internal chain_type_ids (PROTEIN=0, DNA=1, RNA=2, NONPOLYMER=3)
# to FoldingOutput's canonical mol-type convention (PROTEIN=0, RNA=1, DNA=2,
# LIGAND=3). The two encodings differ in the DNA / RNA slot ordering, plus
# Boltz2 calls ligands ``NONPOLYMER``.
_BOLTZ_TO_FOLDING_MOL_TYPE = np.full(max(chain_type_ids.values()) + 1, MOL_TYPE_PROTEIN, dtype=np.int64)
_BOLTZ_TO_FOLDING_MOL_TYPE[chain_type_ids["PROTEIN"]] = MOL_TYPE_PROTEIN
_BOLTZ_TO_FOLDING_MOL_TYPE[chain_type_ids["RNA"]] = MOL_TYPE_RNA
_BOLTZ_TO_FOLDING_MOL_TYPE[chain_type_ids["DNA"]] = MOL_TYPE_DNA
_BOLTZ_TO_FOLDING_MOL_TYPE[chain_type_ids["NONPOLYMER"]] = MOL_TYPE_LIGAND

# Map Boltz2 ``res_type`` argmax index → 3-letter residue name. Index 0 is
# Boltz2's "<pad>", index 1 is the gap "-"; both produce "UNK" so the CIF
# writer doesn't render bogus residue codes for padded positions.
_BOLTZ_RES_NAMES: list[str] = list(tokens)
_BOLTZ_RES_NAMES[0] = "UNK"
_BOLTZ_RES_NAMES[1] = "UNK"


class PostProcessorConfig(BaseModel):
    """Configuration for the Boltz2 postprocessor."""


class PostProcessor(PostProcessorBase):
    """Convert raw Boltz2 model output into :class:`FoldingOutput`.

    The processor:
    1. Selects the best diffusion sample by ``confidence_score``.
    2. Decodes atom names from ``ref_atom_name_chars`` in the batch.
    3. Remaps flat atom coordinates into the standard 37-atom-type layout
       using the decoded names so that ``PDBWriter`` / ``CIFWriter`` work
       directly.
    4. Populates standard ``FoldingOutput`` fields and appends Boltz2-specific
       confidence metrics as extra dict keys.
    """

    def __init__(self, config: BaseModel | None = None) -> None:
        super().__init__(config)
        if self.config is None:
            self.config = PostProcessorConfig()

    def __call__(
        self,
        batch: dict[str, Any],
        output: dict[str, Any],
    ) -> FoldingOutput:
        # --- Masks --------------------------------------------------------------
        atom_pad_mask = _cpu(output["masks"])[0]  # (N_atoms_pad,)
        token_pad_mask = _cpu(output["token_masks"])[0]  # (N_tokens_pad,)
        atom_mask_bool = atom_pad_mask.bool().numpy()
        token_mask_bool = token_pad_mask.bool().numpy()
        n_tokens = int(token_mask_bool.sum())

        # --- Best sample selection ----------------------------------------------
        conf_score = _cpu(output["confidence_score"])  # (B, S)
        best_idx = int(conf_score[0].argmax())
        n_samples = int(_cpu(output["coords"]).shape[1])

        # --- Coordinates (best sample, flat) ------------------------------------
        best_coords_flat = _cpu(output["coords"])[0, best_idx].numpy()  # (N_atoms_pad, 3)

        # --- Atom-to-token mapping ----------------------------------------------
        atom_to_token = _cpu(batch["atom_to_token"])[0]  # (N_atoms_pad, N_tokens_pad)
        token_per_atom = atom_to_token.argmax(dim=-1).numpy()  # (N_atoms_pad,)

        # --- Decode atom names --------------------------------------------------
        flat_atom_names = _decode_flat_atom_names(batch, atom_mask_bool)

        # --- Remap into 37-atom-type layout -------------------------------------
        atom_positions = np.zeros((n_tokens, NUM_ATOM_TYPES, 3), dtype=np.float32)
        atom_mask_out = np.zeros((n_tokens, NUM_ATOM_TYPES), dtype=np.float32)

        for ai in np.where(atom_mask_bool)[0]:
            t = token_per_atom[ai]
            if t >= n_tokens:
                continue
            name = flat_atom_names[ai]
            slot = _ATOM_NAME_TO_IDX.get(name)
            if slot is None:
                continue
            atom_positions[t, slot] = best_coords_flat[ai]
            atom_mask_out[t, slot] = 1.0

        # --- Residue metadata from batch ----------------------------------------
        res_type_onehot = _cpu(batch["res_type"])[0].numpy()  # (N_tokens_pad, C)
        residue_types = res_type_onehot[:n_tokens].argmax(axis=-1).astype(np.int64)
        # Boltz2's residue_types index into the full 33-entry token table
        # (PAD, GAP, 20 amino acids, X, RA..RX, DA..DX). Writers built from
        # ``get_all_residue_types("boltz-2")`` have the same 33-entry
        # ``self.res_types`` so they can recover RA/DA/etc. natively.
        residue_types_raw = residue_types

        residue_indices = _cpu(batch["residue_index"])[0].numpy()[:n_tokens].astype(np.int64) + 1
        chain_indices = _cpu(batch["asym_id"])[0].numpy()[:n_tokens].astype(np.int64)

        # --- Per-residue CCD codes + mol-type (canonical encoding) --------------
        # ``residue_names`` carries the 3-letter CCD code per residue so the
        # CIF writer can emit "TYR"/"SAH"/"DA"/etc. on ligand/HETATM rows
        # instead of falling back to "UNK". ``mol_types`` lets the writer
        # classify chains explicitly (protein/RNA/DNA/ligand) without relying
        # on the all-X heuristic.
        residue_names: list[str] = [
            _BOLTZ_RES_NAMES[i] if 0 <= i < len(_BOLTZ_RES_NAMES) else "UNK" for i in residue_types_raw.tolist()
        ]
        if "mol_type" in batch:
            boltz_mol_types = _cpu(batch["mol_type"])[0].numpy()[:n_tokens].astype(np.int64)
            # Clamp out-of-range values (defensive — should not happen) before
            # indexing the remap table.
            boltz_mol_types = np.clip(boltz_mol_types, 0, len(_BOLTZ_TO_FOLDING_MOL_TYPE) - 1)
            mol_types_out = _BOLTZ_TO_FOLDING_MOL_TYPE[boltz_mol_types].astype(np.int64)
        else:
            mol_types_out = None

        # --- Per-token confidence (best sample) ---------------------------------
        plddt = _cpu(output["plddt"])[0, best_idx].numpy()[:n_tokens]

        b_factors = np.repeat(plddt[:, None], NUM_ATOM_TYPES, axis=-1) * atom_mask_out

        # --- PAE / PDE ----------------------------------------------------------
        pae = _extract_pair_matrix(output, "pae", best_idx, n_tokens)
        pde = _extract_pair_matrix(output, "pde", best_idx, n_tokens)
        max_pae = float(np.max(pae)) if pae is not None else None

        # --- Scalar confidence metrics ------------------------------------------
        ptm = _scalar(output, "ptm", best_idx)
        iptm = _scalar(output, "iptm", best_idx)

        result = FoldingOutput(
            atom_positions=atom_positions,
            residue_types=residue_types,
            atom_mask=atom_mask_out,
            residue_indices=residue_indices,
            b_factors=b_factors,
            chain_indices=chain_indices,
            plddt=plddt,
            ptm=ptm,
            iptm=iptm,
            pae=pae,
            max_pae=max_pae,
            residue_names=residue_names,
            mol_types=mol_types_out,
        )

        # --- Boltz2-specific extras ---------------------------------------------
        result["confidence_score"] = _scalar(output, "confidence_score", best_idx)
        result["complex_plddt"] = _scalar(output, "complex_plddt", best_idx)
        result["complex_iplddt"] = _scalar(output, "complex_iplddt", best_idx)
        result["complex_pde"] = _scalar(output, "complex_pde", best_idx)
        result["complex_ipde"] = _scalar(output, "complex_ipde", best_idx)
        result["ligand_iptm"] = _scalar(output, "ligand_iptm", best_idx)
        result["protein_iptm"] = _scalar(output, "protein_iptm", best_idx)
        result["pde"] = pde
        result["n_samples"] = n_samples
        result["best_sample_idx"] = best_idx

        raw_pci = output.get("pair_chains_iptm", {})
        pair_chains_iptm: dict[str, dict[str, float]] = {}
        for c1, inner in raw_pci.items():
            k1 = str(c1)
            pair_chains_iptm[k1] = {}
            for c2, val in inner.items():
                if isinstance(val, torch.Tensor):
                    v = val.cpu().squeeze()
                    pair_chains_iptm[k1][str(c2)] = float(v[best_idx] if v.dim() > 0 else v)
                else:
                    pair_chains_iptm[k1][str(c2)] = float(val)
        result["pair_chains_iptm"] = pair_chains_iptm

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cpu(t: Any) -> torch.Tensor:
    """Move a tensor to CPU; pass through if already on CPU or not a tensor."""
    if isinstance(t, torch.Tensor):
        return t.cpu()
    return torch.as_tensor(t)


def _scalar(output: dict, key: str, best_idx: int) -> float:
    """Extract a scalar confidence metric for the best sample."""
    v = output.get(key)
    if v is None:
        return float("nan")
    if isinstance(v, torch.Tensor):
        v = v.cpu()
        if v.dim() >= 2:
            return float(v[0, best_idx])
        if v.dim() == 1:
            return float(v[0])
        return float(v)
    return float(v)


def _extract_pair_matrix(
    output: dict,
    key: str,
    best_idx: int,
    n_tokens: int,
) -> np.ndarray | None:
    """Extract a (N_tokens, N_tokens) pairwise matrix for the best sample."""
    raw = output.get(key)
    if raw is None:
        return None
    if isinstance(raw, torch.Tensor):
        raw = raw.cpu().numpy()
    mat = raw[0, best_idx][:n_tokens, :n_tokens]
    return np.round(mat, 3)


def _decode_flat_atom_names(
    batch: dict[str, Any],
    atom_mask_bool: np.ndarray,
) -> list[str]:
    """Decode atom name strings for every padded atom position.

    The OSS featurizer encodes each atom name as 4 integers via
    ``ord(c) - 32``, then one-hot encodes into 64 classes.  We reverse
    that here.

    Returns:
        List of length ``N_atoms_pad`` with decoded names (empty string
        for padded positions).
    """
    raw = batch.get("ref_atom_name_chars")
    n_atoms_pad = atom_mask_bool.shape[0]
    if raw is None:
        return [""] * n_atoms_pad

    chars = _cpu(raw)[0]  # (N_atoms_pad, 4, 64)
    char_indices = chars.argmax(dim=-1).numpy()  # (N_atoms_pad, 4)

    names: list[str] = []
    for ai in range(n_atoms_pad):
        if not atom_mask_bool[ai]:
            names.append("")
            continue
        name = ""
        for c in range(4):
            v = char_indices[ai, c]
            if v == 0:
                break
            name += chr(v + 32)
        names.append(name)
    return names
