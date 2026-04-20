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

from typing import Any, Optional

import numpy as np
import torch
from pydantic import BaseModel

from tensorrt_bionemo.data.schemas import FoldingOutput
from tensorrt_bionemo.data.schemas.basic import AtomTypes
from tensorrt_bionemo.pipeline.base import PostProcessorBase

NUM_ATOM_TYPES = len(AtomTypes.all_types())  # 37

_ATOM_NAME_TO_IDX: dict[str, int] = {
    at.name: i for i, at in enumerate(AtomTypes.all_types())
}


class PostProcessorConfig(BaseModel):
    """Configuration for the OpenFold3 postprocessor."""
    pass


class PostProcessor(PostProcessorBase):
    """Convert raw OpenFold3 model output into FoldingOutput.

    The processor:
    1. Extracts atom positions from the model output (selects best sample
       by mean pLDDT if multiple diffusion samples are present).
    2. Computes confidence scores (pLDDT, pTM, ipTM, PAE) from logits.
    3. Remaps flat atom coordinates into the standard 37-atom-type layout.
    4. Populates standard FoldingOutput fields.
    """

    def __init__(self, config: Optional[BaseModel] = None) -> None:
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
        atom_to_token = _cpu(
            batch["atom_to_token_index"]).squeeze(0).numpy()  # (N_atoms,)

        # --- Decode atom names ---
        flat_atom_names = _decode_flat_atom_names(batch, atom_mask_bool)

        # --- Remap into 37-atom-type layout ---
        atom_positions = np.zeros(
            (n_tokens, NUM_ATOM_TYPES, 3), dtype=np.float32)
        atom_mask_out = np.zeros(
            (n_tokens, NUM_ATOM_TYPES), dtype=np.float32)

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
        restype_onehot = _cpu(
            batch["restype"]).squeeze(0).numpy()  # (N_tokens, 32)
        residue_types = restype_onehot[:n_tokens].argmax(axis=-1).astype(
            np.int64)
        residue_indices = _cpu(
            batch["residue_index"]).squeeze(0).numpy()[:n_tokens].astype(
            np.int64)
        chain_indices = _cpu(
            batch["asym_id"]).squeeze(0).numpy()[:n_tokens].astype(np.int64)

        # --- Confidence scores from logits ---
        plddt = _compute_plddt(output, best_idx, n_tokens, atom_to_token,
                                atom_mask_bool)
        ptm = _compute_ptm(output, best_idx, n_tokens)
        iptm = _compute_iptm(output, best_idx, n_tokens, chain_indices)
        pae = _compute_pae(output, best_idx, n_tokens)
        max_pae = float(np.max(pae)) if pae is not None else None

        b_factors = np.repeat(
            plddt[:, None], NUM_ATOM_TYPES, axis=-1) * atom_mask_out

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
        )

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cpu(t: Any) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        return t.cpu()
    return torch.as_tensor(t)


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


def _compute_plddt(output: dict, best_idx: int, n_tokens: int,
                    atom_to_token: np.ndarray,
                    atom_mask_bool: np.ndarray) -> np.ndarray:
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


def _compute_ptm(output: dict, best_idx: int, n_tokens: int) -> float:
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
    return float(_tm_score_from_pae_logits(logits, n_tokens))


def _compute_iptm(output: dict, best_idx: int, n_tokens: int,
                   chain_indices: np.ndarray) -> float:
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

    return float(_tm_score_from_pae_logits(logits, n_tokens, mask=inter_mask))


def _tm_score_from_pae_logits(
    logits: torch.Tensor,
    n_tokens: int,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Compute TM-score from PAE logits using AF2/AF3 formula."""
    probs = torch.softmax(logits, dim=-1)
    n_bins = probs.shape[-1]
    bin_centers = torch.linspace(0, 32, n_bins)  # 64 bins, 0-32 Å

    # d0 = 1.24 * (max(N, 19) - 15)^(1/3) - 1.8
    d0 = 1.24 * (max(n_tokens, 19) - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.01)

    # TM-score per pair: 1 / (1 + (d/d0)^2)
    tm_per_bin = 1.0 / (1.0 + (bin_centers / d0) ** 2)
    tm_per_pair = (probs * tm_per_bin).sum(dim=-1)  # (N, N)

    if mask is not None:
        return (tm_per_pair * mask).sum() / mask.sum().clamp(min=1)
    return tm_per_pair.mean()


def _compute_pae(output: dict, best_idx: int,
                  n_tokens: int) -> Optional[np.ndarray]:
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
