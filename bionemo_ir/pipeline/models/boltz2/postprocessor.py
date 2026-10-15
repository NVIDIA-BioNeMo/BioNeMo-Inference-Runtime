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
variable per-token layout into the shared writer atom-type scheme
(``AtomTypes.all_types()``) so that downstream writers (``PDBWriter``,
``CIFWriter``) work without modification.
"""

from typing import Any

import numpy as np
import torch
from pydantic import BaseModel

from bionemo_ir.data.schemas import FoldingOutput
from bionemo_ir.data.schemas.basic import MOL_TYPE_DNA, MOL_TYPE_LIGAND, MOL_TYPE_PROTEIN, MOL_TYPE_RNA
from bionemo_ir.pipeline.base import PostProcessorBase
from bionemo_ir.pipeline.models.boltz2.const import chain_type_ids, tokens
from bionemo_ir.pipeline.utils.atom import (
    NUM_FOLDING_ATOM_TYPES,
    decode_atom_name_chars,
    decode_element_symbols,
    one_hot_indices,
    scatter_flat_atoms_to_folding_layout,
)

NUM_ATOM_TYPES = NUM_FOLDING_ATOM_TYPES

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

# Boltz2-specific per-sample ``(B, S)`` scores, beside ``ptm`` and ``iptm``.
_EXTRA_SCORE_KEYS = (
    "confidence_score",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
    "ligand_iptm",
    "protein_iptm",
)


class PostProcessorConfig(BaseModel):
    """Configuration for the Boltz2 postprocessor."""


class PostProcessor(PostProcessorBase):
    """Convert raw Boltz2 model output into :class:`FoldingOutput`.

    The processor:
    1. Selects the best diffusion sample by ``confidence_score``.
    2. Decodes atom names from ``ref_atom_name_chars`` in the batch.
    3. Remaps flat atom coordinates into the shared writer atom-type layout
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
        coords = torch.as_tensor(output["coords"])
        n_samples = int(coords.shape[1])

        # --- Coordinates (best sample, flat) ------------------------------------
        best_coords_flat = _cpu(coords[0, best_idx]).numpy()  # (N_atoms_pad, 3)

        # --- Atom-to-token mapping ----------------------------------------------
        token_per_atom = one_hot_indices(torch.as_tensor(batch["atom_to_token"])[0])  # (N_atoms_pad,)

        # --- Decode atom names --------------------------------------------------
        raw_atom_names = batch.get("ref_atom_name_chars")
        if raw_atom_names is not None:
            raw_atom_names = torch.as_tensor(raw_atom_names)[0]
        flat_atom_names = decode_atom_name_chars(raw_atom_names, atom_mask_bool)

        # --- Decode elements ----------------------------------------------------
        # Boltz2 one-hots ``ref_element`` by atomic number directly (see
        # ``boltz2/const.py`` ``num_elements = 128``), so no index offset.
        raw_elements = batch.get("ref_element")
        if raw_elements is not None:
            raw_elements = torch.as_tensor(raw_elements)[0]
        flat_atom_elements = decode_element_symbols(raw_elements, atom_mask_bool, index_offset=0)

        # --- Remap into the canonical atom layout -------------------------------
        atom_positions, atom_mask_out, per_slot_atom_names, per_slot_elements = scatter_flat_atoms_to_folding_layout(
            best_coords_flat,
            token_per_atom,
            flat_atom_names,
            atom_mask_bool,
            n_tokens,
            atom_elements=flat_atom_elements,
        )

        # --- Residue metadata from batch ----------------------------------------
        residue_types = one_hot_indices(torch.as_tensor(batch["res_type"])[0, :n_tokens]).astype(np.int64)
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
        token_resnames = batch.get("token_resnames")
        if token_resnames is not None and len(token_resnames) >= n_tokens:
            residue_names = list(token_resnames[:n_tokens])
        else:
            residue_names = [
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
        # Copy every score to the host at once; separate reads each synchronize.
        score_keys = ("ptm", "iptm", *_EXTRA_SCORE_KEYS)
        raw_pci = output.get("pair_chains_iptm", {})
        selected = [_best_sample_score(output.get(key), best_idx) for key in score_keys]
        selected += [_pair_chain_score(val, best_idx) for inner in raw_pci.values() for val in inner.values()]
        values = iter(_to_floats(selected))
        scores = {key: next(values) for key in score_keys}
        pair_chains_iptm = {str(c1): {str(c2): next(values) for c2 in inner} for c1, inner in raw_pci.items()}

        result = FoldingOutput(
            atom_positions=atom_positions,
            residue_types=residue_types,
            atom_mask=atom_mask_out,
            residue_indices=residue_indices,
            b_factors=b_factors,
            chain_indices=chain_indices,
            plddt=plddt,
            ptm=scores["ptm"],
            iptm=scores["iptm"],
            pae=pae,
            max_pae=max_pae,
            residue_names=residue_names,
            mol_types=mol_types_out,
            atom_names=per_slot_atom_names,
            atom_elements=per_slot_elements,
        )

        # --- Boltz2-specific extras ---------------------------------------------
        for key in _EXTRA_SCORE_KEYS:
            result[key] = scores[key]
        result["pde"] = pde
        result["n_samples"] = n_samples
        result["best_sample_idx"] = best_idx
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


def _best_sample_score(value: Any, best_idx: int) -> Any:
    """Select the best sample of a ``(B, S)`` score without leaving its device."""
    if isinstance(value, torch.Tensor):
        if value.dim() >= 2:
            return value[0, best_idx]
        if value.dim() == 1:
            return value[0]
    return value


def _pair_chain_score(value: Any, best_idx: int) -> Any:
    """Select the best sample of a ``pair_chains_iptm`` entry without leaving its device."""
    if isinstance(value, torch.Tensor):
        value = value.squeeze()
        return value[best_idx] if value.dim() > 0 else value
    return value


def _to_floats(values: list[Any]) -> list[float]:
    """Convert one-element scores to floats, ``None`` to NaN, in one host copy."""
    tensors = [value.reshape(()) for value in values if isinstance(value, torch.Tensor)]
    if len({(t.device, t.dtype) for t in tensors}) > 1:
        tensors = [t.to(tensors[0].device, torch.float64) for t in tensors]
    host = iter(torch.stack(tensors).tolist() if tensors else ())
    return [
        float(next(host)) if isinstance(value, torch.Tensor) else float("nan") if value is None else float(value)
        for value in values
    ]


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
    mat = torch.as_tensor(raw)[0, best_idx, :n_tokens, :n_tokens]
    if mat.is_cuda and mat.dtype == torch.float32:
        # Equals ``np.round(x, 3)`` for every float32 bit pattern, NaN
        # payloads aside, and leaves the host only the copy.
        return mat.round(decimals=3).cpu().numpy()
    return np.round(_cpu(mat).numpy(), 3)
