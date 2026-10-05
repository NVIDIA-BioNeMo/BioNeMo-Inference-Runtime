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
"""OpenFold3 PostProcessor: convert raw model output into FoldingOutput.

Maps OF3's atom-level predictions to the standard FoldingOutput schema.
Atom coordinates are remapped from OF3's variable per-token layout into
the standard 37-atom-type scheme so downstream writers work correctly.
"""

from typing import Any

import numpy as np
import torch
from pydantic import BaseModel

from bionemo_ir._torch.modules.openfold3.utils.confidence import bin_centers, reduce_pae_logits
from bionemo_ir.data.schemas import FoldingOutput
from bionemo_ir.pipeline.base import PostProcessorBase
from bionemo_ir.pipeline.utils.atom import (
    NUM_FOLDING_ATOM_TYPES,
    decode_atom_name_chars,
    scatter_flat_atoms_to_folding_layout,
)

NUM_ATOM_TYPES = NUM_FOLDING_ATOM_TYPES


class PostProcessorConfig(BaseModel):
    """Configuration for the OpenFold3 postprocessor."""


class PostProcessor(PostProcessorBase):
    """Convert raw OpenFold3 model output into FoldingOutput.

    The processor:
    1. Extracts atom positions from the model output (selects best sample
       by mean pLDDT if multiple diffusion samples are present).
    2. Computes confidence scores (pLDDT, pTM, ipTM, PAE) from logits.
    3. Remaps flat atom coordinates into the standard 37-atom-type layout.
    4. Populates standard FoldingOutput fields.
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
        # --- Extract masks ---
        token_mask = _cpu(batch["token_mask"]).squeeze(0)  # (N_tokens,)
        atom_mask = _cpu(batch["atom_mask"]).squeeze(0)  # (N_atoms,)
        n_tokens = int(token_mask.sum())
        atom_mask_bool = atom_mask.bool().numpy()

        # --- Per-atom pLDDT ---
        # Drives both the diffusion-sample choice and the reported pLDDT.
        plddt_per_atom = _plddt_per_atom(output)

        # --- Atom positions ---
        # atom_positions_predicted: (B, S, N_atoms, 3) or (B, N_atoms, 3)
        raw_pos = torch.as_tensor(output["atom_positions_predicted"])
        if raw_pos.dim() == 4:
            # Multiple diffusion samples — select best by mean pLDDT
            best_idx = _select_best_sample(plddt_per_atom)
            best_pos = raw_pos[0, best_idx]
        elif raw_pos.dim() == 3:
            best_idx = 0
            best_pos = raw_pos[0]
        else:
            best_idx = 0
            best_pos = raw_pos
        # Copy only the chosen sample, not all of them
        best_pos = best_pos.cpu().numpy()  # (N_atoms, 3)

        # --- Atom-to-token mapping ---
        atom_to_token = _cpu(batch["atom_to_token_index"]).squeeze(0).numpy()  # (N_atoms,)

        # --- Decode atom names ---
        raw_atom_names = batch.get("ref_atom_name_chars")
        if raw_atom_names is not None:
            raw_atom_names = _cpu(raw_atom_names).squeeze(0)
        flat_atom_names = decode_atom_name_chars(raw_atom_names, atom_mask_bool)

        # --- Remap into the shared atom layout ---
        # The universe covers protein backbone+sidechain, nucleic backbone,
        # nucleobases, and common ligand atom labels. Ligand tokens are
        # atomized (one atom = one token), so
        # writing each atom to its own (token, atom-name-slot) does not
        # collide with neighbouring tokens: there's no shared vocabulary
        # problem here because each ligand atom owns its own row.
        atom_positions, atom_mask_out = scatter_flat_atoms_to_folding_layout(
            best_pos,
            atom_to_token,
            flat_atom_names,
            atom_mask_bool,
            n_tokens,
        )

        # --- Residue metadata ---
        restype_onehot = _cpu(batch["restype"]).squeeze(0).numpy()  # (N_tokens, 32)
        residue_types = restype_onehot[:n_tokens].argmax(axis=-1).astype(np.int64)
        residue_indices = _cpu(batch["residue_index"]).squeeze(0).numpy()[:n_tokens].astype(np.int64)
        # OF3 asym_id is 1-indexed per OSS contract (see feature_context.py
        # _renumber_chain_ids → chains numbered 1..N alphabetically). The
        # FoldingOutput / CIF-writer chain_indices contract expects 0-indexed
        # chain IDs (chain_tags[0] == 'A'). Subtract 1 to convert so chain A
        # writes as 'A' rather than 'B' in the produced CIF.
        # (Surfaced by Plan 07 e2e — OST_CMD chain_mapping was offset by 1.)
        chain_indices = _cpu(batch["asym_id"]).squeeze(0).numpy()[:n_tokens].astype(np.int64)
        chain_indices = chain_indices - 1

        # --- Confidence scores from logits ---
        # Reduce logits on their current device and copy only the results.
        # pTM, ipTM and PAE share the selected, cropped PAE logits and one
        # softmax per row block.
        plddt = _compute_plddt(plddt_per_atom, best_idx, n_tokens, atom_to_token, atom_mask_bool)
        if "pae" in output:
            pae, ptm, iptm = _compact_confidence(output, best_idx, n_tokens)
        else:
            pae_logits = _pae_logits(output, best_idx, n_tokens)
            has_frame = _frame_mask(output, best_idx, n_tokens)
            pae, ptm, iptm = _reduce_confidence(pae_logits, n_tokens, chain_indices, has_frame)
        max_pae = float(np.max(pae)) if pae is not None else None

        b_factors = np.repeat(plddt[:, None], NUM_ATOM_TYPES, axis=-1) * atom_mask_out

        # Per-residue identity for the comprehensive CIF writer: 3-letter CCD
        # code per token and the canonical mol-type id. Surfaces ligand
        # ("SAH"/"TYR"/etc.) and nucleotide ("DA"/"A"/...) on HETATM rows
        # rather than the protein-letter heuristic.
        tnames = batch.get("token_resnames")
        tmtypes = batch.get("token_mol_types")
        struct = batch.get("_row", batch).get("structure") if isinstance(batch, dict) else None
        if struct is not None:
            if tnames is None:
                tnames = struct.get("token_resnames")
            if tmtypes is None:
                tmtypes = struct.get("token_mol_types")
        residue_names = list(tnames[:n_tokens]) if tnames is not None and len(tnames) >= n_tokens else None
        mol_types_out = (
            np.asarray(tmtypes[:n_tokens], dtype=np.int64) if tmtypes is not None and len(tmtypes) >= n_tokens else None
        )

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

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cpu(t: Any) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        return t.cpu()
    return torch.as_tensor(t)


def _select_sample(value: torch.Tensor, best_idx: int, trailing_ndim: int) -> torch.Tensor:
    """Drop the ``(B, S)`` axes a per-sample model output may carry in front of ``trailing_ndim`` axes."""
    if value.dim() == trailing_ndim + 2:
        return value[0, best_idx]
    if value.dim() == trailing_ndim + 1:
        return value[0]
    return value


def _frame_mask(output: dict, best_idx: int, n_tokens: int) -> torch.Tensor | None:
    """``has_frame`` for the pTM / ipTM maximum, as emitted by the confidence head.

    The head computes it from the sampled coordinates, so it carries the diffusion
    sample axis and arrives as 0/1 floats in the output dtype. Absent when the PAE
    head is disabled, in which case there is no pTM to restrict. Stays on its
    original device, alongside the logits it will gate.
    """
    mask = output.get("valid_frame_mask")
    if mask is None:
        return None
    return _select_sample(torch.as_tensor(mask), best_idx, 1)[:n_tokens].bool()


def _pae_logits(output: dict, best_idx: int, n_tokens: int) -> torch.Tensor | None:
    """The selected sample's PAE logits, cropped to the real tokens.

    Left on its original device: ``torch.as_tensor`` preserves it, unlike
    ``_cpu``. The engine calls the post-processor with the model's own output, so
    this is normally GPU memory, and every consumer reduces it.
    """
    logits = output.get("pae_logits")
    if logits is None:
        return None
    return _select_sample(torch.as_tensor(logits), best_idx, 3)[:n_tokens, :n_tokens]


def _plddt_per_atom(output: dict) -> torch.Tensor | None:
    """Per-atom pLDDT on a 0-100 scale, as ``(n_samples, N_atom)``.

    Both the diffusion-sample choice and the reported per-token score are this
    same expectation, so it runs once here instead of once in each. Stays on the
    logits' device; only the selected row is ever copied to the host.
    """
    logits = output.get("plddt_logits")
    if logits is None:
        return None
    logits = torch.as_tensor(logits)
    if logits.dim() == 4:
        logits = logits[0]  # (B, S, N_atom, n_bins) -> (S, N_atom, n_bins)
    elif logits.dim() == 3:
        logits = logits[:1]  # (B, N_atom, n_bins) -> a single sample
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)  # (N_atom, n_bins) -> a single sample
    probs = torch.softmax(logits.float(), dim=-1)
    centers = bin_centers(0.0, 1.0, probs.shape[-1]).to(device=probs.device)
    return (probs * centers).sum(dim=-1) * 100.0


def _select_best_sample(plddt_per_atom: torch.Tensor | None) -> int:
    """Select best diffusion sample by mean pLDDT.

    Reduces on the device and brings back only the winning index.
    """
    if plddt_per_atom is None:
        return 0
    return int(plddt_per_atom.mean(dim=-1).argmax())


def _compute_plddt(
    plddt_per_atom: torch.Tensor | None,
    best_idx: int,
    n_tokens: int,
    atom_to_token: np.ndarray,
    atom_mask_bool: np.ndarray,
) -> np.ndarray:
    """Average the selected sample's per-atom pLDDT into per-token pLDDT.

    Only the selected sample's ``(N_atom,)`` row crosses to the host. The
    atom-to-token mean then runs as a bincount there: a scatter-add on the device
    would be faster still, but CUDA's unordered ``atomicAdd`` makes it
    non-reproducible run to run, and this value is reported and written into the
    B-factors.
    """
    if plddt_per_atom is None:
        return np.full(n_tokens, 50.0, dtype=np.float32)
    per_atom = plddt_per_atom[best_idx].cpu().numpy()

    # Mean over the present atoms of each token; tokens with no atom stay at 0
    scored = atom_mask_bool & (atom_to_token < n_tokens)
    token_of_atom = atom_to_token[scored]
    totals = np.bincount(token_of_atom, weights=per_atom[scored], minlength=n_tokens)
    counts = np.bincount(token_of_atom, minlength=n_tokens)
    return (totals[:n_tokens] / np.maximum(counts[:n_tokens], 1)).astype(np.float32)


def _reduce_confidence(
    logits: torch.Tensor | None,
    n_tokens: int,
    chain_indices: np.ndarray,
    has_frame: torch.Tensor | None,
) -> tuple[np.ndarray | None, float, float]:
    """PAE matrix, pTM and ipTM of the selected sample, or their absent values.

    The reducer normalizes one row block at a time on the logits' device, so
    the host receives an ``(N_token, N_token)`` matrix and two scalars rather
    than the logits. PAE keeps its three-decimal rounding; pTM and ipTM are NaN
    without the PAE head, without an eligible frame, or, for ipTM, without a
    second chain, which ``FoldingOutput.get_scores()`` reports as ``None``.
    """
    if logits is None:
        return None, float("nan"), float("nan")
    reduced = reduce_pae_logits(logits, n_tokens, torch.as_tensor(chain_indices), has_frame)
    ptm, iptm = reduced.scores.tolist()
    return np.round(reduced.pae.cpu().numpy(), 3), float(ptm), float(iptm)


def _compact_confidence(output: dict, best_idx: int, n_tokens: int) -> tuple[np.ndarray, float, float]:
    """The selected sample's producer-reduced PAE, pTM and ipTM.

    The confidence head already reduced every sample on its device, so only
    the selected ``(N_token, N_token)`` matrix and two scalars cross to the
    host. The matrix is rounded exactly as the logits path rounds it.
    """
    pae = _select_sample(torch.as_tensor(output["pae"]), best_idx, 2)[:n_tokens, :n_tokens]
    scores = torch.stack([_select_sample(torch.as_tensor(output[name]), best_idx, 0) for name in ("ptm", "iptm")])
    ptm, iptm = scores.tolist()
    return np.round(pae.float().cpu().numpy(), 3), float(ptm), float(iptm)
