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

from bionemo_ir.data.schemas import FoldingOutput
from bionemo_ir.data.schemas.basic import AtomTypes
from bionemo_ir.pipeline.base import PostProcessorBase

NUM_ATOM_TYPES = len(AtomTypes.all_types())

_ATOM_NAME_TO_IDX: dict[str, int] = {at.name: i for i, at in enumerate(AtomTypes.all_types())}


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

        # --- Atom positions ---
        # atom_positions_predicted: (B, S, N_atoms, 3) or (B, N_atoms, 3)
        raw_pos = _cpu(output["atom_positions_predicted"])
        if raw_pos.dim() == 4:
            # Multiple diffusion samples — select best by mean pLDDT
            best_idx = _select_best_sample(output)
            best_pos = raw_pos[0, best_idx].numpy()  # (N_atoms, 3)
        elif raw_pos.dim() == 3:
            best_idx = 0
            best_pos = raw_pos[0].numpy()  # (N_atoms, 3)
        else:
            best_idx = 0
            best_pos = raw_pos.numpy()

        # --- Atom-to-token mapping ---
        atom_to_token = _cpu(batch["atom_to_token_index"]).squeeze(0).numpy()  # (N_atoms,)

        # --- Decode atom names ---
        flat_atom_names = _decode_flat_atom_names(batch, atom_mask_bool)

        # --- Remap into the shared atom layout ---
        # The universe covers protein backbone+sidechain, nucleic backbone,
        # nucleobases, and common ligand atom labels. Ligand tokens are
        # atomized (one atom = one token), so
        # writing each atom to its own (token, atom-name-slot) does not
        # collide with neighbouring tokens: there's no shared vocabulary
        # problem here because each ligand atom owns its own row.
        atom_positions = np.zeros((n_tokens, NUM_ATOM_TYPES, 3), dtype=np.float32)
        atom_mask_out = np.zeros((n_tokens, NUM_ATOM_TYPES), dtype=np.float32)

        for ai in np.where(atom_mask_bool)[0]:
            t = atom_to_token[ai]
            if t >= n_tokens:
                continue
            name = flat_atom_names[ai]
            slot = _ATOM_NAME_TO_IDX.get(name)
            if slot is None:
                continue
            atom_positions[t, slot] = best_pos[ai]
            atom_mask_out[t, slot] = 1.0

        # --- Residue metadata ---
        restype_onehot = _cpu(batch["restype"]).squeeze(0).numpy()  # (N_tokens, 32)
        residue_types = restype_onehot[:n_tokens].argmax(axis=-1).astype(np.int64)
        residue_indices = _cpu(batch["residue_index"]).squeeze(0).numpy()[:n_tokens].astype(np.int64)
        # OF3 asym_id is 1-indexed per the upstream contract (see
        # ``_renumber_chain_ids`` in feature_context.py → chains numbered
        # 1..N alphabetically). The FoldingOutput / CIF-writer chain_indices
        # contract expects 0-indexed chain IDs (chain_tags[0] == 'A').
        # Subtract 1 to convert so chain A writes as 'A' rather than 'B' in
        # the produced CIF.
        chain_indices = _cpu(batch["asym_id"]).squeeze(0).numpy()[:n_tokens].astype(np.int64)
        chain_indices = chain_indices - 1

        # --- Confidence scores from logits ---
        plddt = _compute_plddt(output, best_idx, n_tokens, atom_to_token, atom_mask_bool)
        has_frame = _aligned_token_mask(batch, n_tokens)
        ptm = _compute_ptm(output, best_idx, n_tokens, has_frame=has_frame)
        iptm = _compute_iptm(output, best_idx, n_tokens, chain_indices, has_frame=has_frame)
        pae = _compute_pae(output, best_idx, n_tokens)
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
        # Convention of the aligned-token (frame) mask behind ptm / iptm, carried
        # into get_scores(): "polymer_tokens" = interim ~is_atomized mask,
        # "none" = every token eligible (is_atomized absent from the batch).
        result["ptm_frame_mask"] = "none" if has_frame is None else "polymer_tokens"

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cpu(t: Any) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        return t.cpu()
    return torch.as_tensor(t)


def _bin_centers(n_bins: int, bin_min: float, bin_max: float) -> torch.Tensor:
    """Midpoints of ``n_bins`` equal-width bins on ``[bin_min, bin_max]``.

    Matches AF3 and upstream OpenFold3 (``openfold3/core/metrics/confidence.py::
    get_bin_centers``): 0.25, 0.75, ..., 31.75 Å for the 64-bin PAE head and
    0.01, 0.03, ..., 0.99 for the 50-bin pLDDT head -- not the end-point-inclusive
    positions of ``torch.linspace(bin_min, bin_max, n_bins)``.
    """
    width = (bin_max - bin_min) / n_bins
    return bin_min + width * (torch.arange(n_bins, dtype=torch.float32) + 0.5)


def _aligned_token_mask(batch: dict[str, Any], n_tokens: int) -> torch.Tensor | None:
    """Tokens eligible as the aligned token ``i`` in the pTM / ipTM max (``has_frame``).

    Interim stand-in for the coordinate-based frame validity of upstream OpenFold3
    (``openfold3/core/utils/atomize_utils.py::get_token_frame_atoms``): polymer
    tokens are eligible; atomized tokens (``batch["is_atomized"]``: ligand atoms,
    ions) are scored as ``j`` but never used as aligned tokens. Upstream additionally
    admits atomized tokens whose nearest-neighbour local frame is valid and requires
    the backbone frame atoms of polymer residues to be present. Returns ``None``
    (every token eligible) only when the flag is absent from the batch; an input
    without polymer tokens (ligand-only query) yields an all-False mask, for which
    pTM / ipTM are reported as NaN (see ``_tm_score_from_pae_logits``).
    """
    is_atomized = batch.get("is_atomized")
    if is_atomized is None:
        return None
    return ~_cpu(is_atomized).reshape(-1)[:n_tokens].bool()


def _select_best_sample(output: dict) -> int:
    """Select best diffusion sample by mean pLDDT."""
    logits = output.get("plddt_logits")
    if logits is None:
        return 0
    logits = _cpu(logits)
    if logits.dim() >= 3:
        # (B, S, N_atoms, 50) → compute mean pLDDT per sample
        probs = torch.softmax(logits[0], dim=-1)
        n_bins = probs.shape[-1]
        bin_centers = torch.linspace(0, 1, n_bins)
        plddt_per_atom = (probs * bin_centers).sum(dim=-1)  # (S, N_atoms)
        mean_plddt = plddt_per_atom.mean(dim=-1)  # (S,)
        return int(mean_plddt.argmax())
    return 0


def _compute_plddt(
    output: dict, best_idx: int, n_tokens: int, atom_to_token: np.ndarray, atom_mask_bool: np.ndarray
) -> np.ndarray:
    """Compute per-token pLDDT from per-atom pLDDT logits."""
    logits = output.get("plddt_logits")
    if logits is None:
        return np.full(n_tokens, 50.0, dtype=np.float32)
    logits = _cpu(logits)
    if logits.dim() == 4:
        logits = logits[0, best_idx]  # (N_atoms, 50)
    elif logits.dim() == 3:
        logits = logits[0]
    probs = torch.softmax(logits, dim=-1)
    n_bins = probs.shape[-1]
    bin_centers = torch.linspace(0, 1, n_bins)
    plddt_per_atom = (probs * bin_centers).sum(dim=-1).numpy() * 100.0

    # Aggregate to per-token
    plddt = np.zeros(n_tokens, dtype=np.float32)
    counts = np.zeros(n_tokens, dtype=np.float32)
    for ai in np.where(atom_mask_bool)[0]:
        t = atom_to_token[ai]
        if t < n_tokens:
            plddt[t] += plddt_per_atom[ai]
            counts[t] += 1
    counts = np.maximum(counts, 1)
    return plddt / counts


def _compute_ptm(
    output: dict,
    best_idx: int,
    n_tokens: int,
    has_frame: torch.Tensor | None = None,
) -> float:
    """Compute predicted TM-score from PAE logits."""
    logits = output.get("pae_logits")
    if logits is None:
        return float("nan")
    logits = _cpu(logits)
    if logits.dim() == 5:
        logits = logits[0, best_idx]  # (N_tokens, N_tokens, 64)
    elif logits.dim() == 4:
        logits = logits[0]
    logits = logits[:n_tokens, :n_tokens]
    return float(_tm_score_from_pae_logits(logits, n_tokens, has_frame=has_frame))


def _compute_iptm(
    output: dict,
    best_idx: int,
    n_tokens: int,
    chain_indices: np.ndarray,
    has_frame: torch.Tensor | None = None,
) -> float:
    """Compute interface pTM from PAE logits (inter-chain pairs only)."""
    logits = output.get("pae_logits")
    if logits is None:
        return float("nan")
    unique_chains = np.unique(chain_indices)
    if len(unique_chains) < 2:
        return float("nan")
    logits = _cpu(logits)
    if logits.dim() == 5:
        logits = logits[0, best_idx]
    elif logits.dim() == 4:
        logits = logits[0]
    logits = logits[:n_tokens, :n_tokens]

    # Mask to inter-chain pairs
    ci = torch.tensor(chain_indices, dtype=torch.long)
    inter_mask = (ci.unsqueeze(0) != ci.unsqueeze(1)).float()
    if inter_mask.sum() == 0:
        return float("nan")

    return float(_tm_score_from_pae_logits(logits, n_tokens, mask=inter_mask, has_frame=has_frame))


def _tm_score_from_pae_logits(
    logits: torch.Tensor,
    n_tokens: int,
    mask: torch.Tensor | None = None,
    has_frame: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute pTM / ipTM from PAE logits (AF3 SI §5.9.1, eqs. 17-18).

    For every aligned token ``i`` the expected pairwise TM term
    ``E[1 / (1 + (e_ij / d0)^2)]`` is averaged over the scored tokens ``j``
    (all tokens for pTM; tokens of other chains for ipTM, selected through
    ``mask[i, j]``), and the score is the *maximum* of these per-aligned-token
    means over ``i`` -- the reduction used by AlphaFold2/3 and by upstream
    OpenFold3 (``openfold3/core/metrics/confidence.py::compute_ptm``). A mean
    over ``i`` (or over all pairs) is a lower bound of that value and is not
    comparable with AF3-calibrated pTM / ipTM thresholds.

    Args:
        logits: (N, N, n_bins) PAE logits; row ``i`` is the aligned token.
        n_tokens: number of tokens N used for ``d0`` (full complex).
        mask: optional (N, N) 0/1 mask of scored pairs; ``None`` scores all pairs (pTM).
        has_frame: optional (N,) bool mask restricting the max over ``i`` to
            tokens with a valid frame (upstream derives it from the predicted
            coordinates); ``None`` treats every token as a valid aligned token.

    Returns NaN when no aligned token is eligible (``has_frame`` given with no True
    entry, e.g. a ligand-only query under the interim mask, or no scored pair):
    there is no frame-eligible token to align on. Upstream OpenFold3 returns 0.0 in
    that case (``masked_fill`` then ``max``) and Protenix returns zeros; NaN is used
    here so that ``FoldingOutput.get_scores()`` reports ``None`` rather than a
    misleading 0, as it already does for the ipTM of single-chain inputs.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    n_bins = probs.shape[-1]
    bin_centers = _bin_centers(n_bins, 0.0, 32.0)  # 64 bins on [0, 32] Å -> 0.25 ... 31.75

    # d0 = 1.24 * (max(N, 19) - 15)^(1/3) - 1.8
    d0 = 1.24 * (max(n_tokens, 19) - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.01)

    # TM-score term per pair: E_bins[1 / (1 + (e/d0)^2)]
    tm_per_bin = 1.0 / (1.0 + (bin_centers / d0) ** 2)
    tm_per_pair = (probs * tm_per_bin).sum(dim=-1)  # (N, N)

    if mask is None:
        mask = torch.ones_like(tm_per_pair)
    mask = mask.to(dtype=tm_per_pair.dtype)

    # Mean over scored tokens j for each aligned token i, then max over i.
    n_scored = mask.sum(dim=-1)  # (N,)
    tm_per_aligned = (tm_per_pair * mask).sum(dim=-1) / n_scored.clamp(min=1)
    valid = n_scored > 0
    if has_frame is not None:
        valid = valid & has_frame.to(device=valid.device, dtype=torch.bool)
    if not bool(valid.any()):
        return torch.tensor(float("nan"))
    return tm_per_aligned[valid].max()


def _compute_pae(output: dict, best_idx: int, n_tokens: int) -> np.ndarray | None:
    """Compute PAE matrix from PAE logits."""
    logits = output.get("pae_logits")
    if logits is None:
        return None
    logits = _cpu(logits)
    if logits.dim() == 5:
        logits = logits[0, best_idx]
    elif logits.dim() == 4:
        logits = logits[0]
    logits = logits[:n_tokens, :n_tokens]
    probs = torch.softmax(logits, dim=-1)
    n_bins = probs.shape[-1]
    bin_centers = torch.linspace(0, 32, n_bins)
    pae = (probs * bin_centers).sum(dim=-1).numpy()
    return np.round(pae, 3)


def _decode_flat_atom_names(
    batch: dict[str, Any],
    atom_mask_bool: np.ndarray,
) -> list[str]:
    """Decode atom name strings from ref_atom_name_chars.

    The featurizer encodes each atom name as 4 integers via ord(c) - 32,
    then one-hot encodes into 64 classes. We reverse that here.
    """
    raw = batch.get("ref_atom_name_chars")
    n_atoms = atom_mask_bool.shape[0]
    if raw is None:
        return [""] * n_atoms

    chars = _cpu(raw).squeeze(0)  # (N_atoms, 4, 64)
    char_indices = chars.argmax(dim=-1).numpy()  # (N_atoms, 4)

    names: list[str] = []
    for ai in range(n_atoms):
        if not atom_mask_bool[ai]:
            names.append("")
            continue
        name = ""
        for c in range(4):
            v = char_indices[ai, c]
            if v == 0:
                break
            name += chr(v + 32)
        names.append(name.strip())
    return names
