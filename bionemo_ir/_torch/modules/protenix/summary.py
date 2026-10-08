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
"""Protenix confidence summary / postprocess bridge.

Converts distogram / confidence head logits into OSS-format
``summary_confidence`` + ``full_data`` (inference path only).

The chain bookkeeping (which tokens and atoms each chain and chain pair
holds, frame positions, ligand and polymer flags) comes from two host
copies, one of the token rows and one of the atom rows (:class:`ChainIndex`). The scores gather with its
ascending index lists, the elements and order boolean masks select, so each
score is OSS's arithmetic on the same values without a host synchronization
per chain or chain pair.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bionemo_ir.configs import BaseConfig


def get_bin_centers(min_bin: float, max_bin: float, no_bins: int) -> torch.Tensor:
    """Bin centers for a ``[min_bin, max_bin]`` range split into ``no_bins``."""
    bin_width = (max_bin - min_bin) / no_bins
    boundaries = torch.linspace(min_bin, max_bin - bin_width, no_bins)
    return boundaries + 0.5 * bin_width


# (device, dtype, default dtype, min_bin, max_bin, no_bins) -> get_bin_centers(...) on that device: copied
# once, not once per call (a pageable host-to-device copy waits for the stream). get_bin_centers computes in
# the default dtype, so the key holds it.
_DEVICE_BIN_CENTERS: dict[tuple, torch.Tensor] = {}


def _bin_centers_on(
    device: torch.device, dtype: torch.dtype, min_bin: float, max_bin: float, no_bins: int
) -> torch.Tensor:
    """``get_bin_centers(min_bin, max_bin, no_bins).to(device, dtype)``, cached."""
    key = (device, dtype, torch.get_default_dtype(), min_bin, max_bin, no_bins)
    centers = _DEVICE_BIN_CENTERS.get(key)
    if centers is None:
        centers = get_bin_centers(min_bin, max_bin, no_bins).to(device, dtype)
        _DEVICE_BIN_CENTERS[key] = centers
    return centers


def logits_to_score(logits: torch.Tensor, min_bin: float, max_bin: float, no_bins: int, return_prob: bool = False):
    """Expected value of the bin centers under ``softmax(logits)``."""
    prob = F.softmax(logits, dim=-1)
    bin_centers = _bin_centers_on(logits.device, logits.dtype, min_bin, max_bin, no_bins)
    score = prob @ bin_centers
    return (score, prob) if return_prob else score


def compute_contact_prob(
    distogram_logits: torch.Tensor, min_bin: float, max_bin: float, no_bins: int, thres: float = 8.0
) -> torch.Tensor:
    """Contact probability = summed distogram probability below ``thres`` Å."""
    prob = F.softmax(distogram_logits, dim=-1)
    bin_centers = get_bin_centers(min_bin, max_bin, no_bins)
    thres_idx = int((bin_centers < thres).sum())
    return prob[..., :thres_idx].sum(-1)


def _calculate_normalization(n: int) -> float:
    """TM-score length normalization constant ``d0``."""
    return 1.24 * (max(n, 19) - 15) ** (1 / 3) - 1.8


class _TokenSet:
    """Ascending device token indices of a chain or chain pair, and the positions of its frame tokens."""

    def __init__(self, tokens: torch.Tensor, frames: torch.Tensor, n_frame: int):
        self.tokens = tokens
        self.frames = frames
        self.n_frame = n_frame


class ChainIndex:
    """Chain bookkeeping of one structure, built on the host from one copy of its token rows and one of its atom rows.

    Chains are numbered ``0..K-1`` in ascending ``asym_id`` order (``torch.unique``'s order). Each index
    list is ascending, as a boolean mask selects; all of them reach the device in one copy.

    Args:
        asym_id, has_frame, token_is_ligand: ``[N_token]``
        atom_to_token_idx, is_polymer: ``[N_atom]``
    """

    def __init__(
        self,
        asym_id: torch.Tensor,
        has_frame: torch.Tensor,
        token_is_ligand: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
        is_polymer: torch.Tensor,
    ) -> None:
        device = asym_id.device
        # Flags as the boolean masks read them (nonzero), before the integer copy: a fractional flag stays set.
        tokens = torch.stack([asym_id.long(), (has_frame != 0).long(), (token_is_ligand != 0).long()]).cpu().numpy()
        atoms = torch.stack([atom_to_token_idx.long(), (is_polymer != 0).long()]).cpu().numpy()
        chains = np.unique(tokens[0])
        asym = np.searchsorted(chains, tokens[0])  # contiguous 0..K-1, ascending asym_id
        frame, ligand = tokens[1] != 0, tokens[2] != 0
        atom_asym = asym[atoms[0]]
        polymer = atoms[1] != 0
        self.n_chain = n_chain = len(chains)

        lists: list[np.ndarray] = [asym]
        spans: list[tuple[int, int]] = []

        def push(index: np.ndarray) -> int:
            lists.append(np.asarray(index, dtype=np.int64))
            return len(lists) - 1

        def token_set(selected: np.ndarray) -> tuple[int, int, int]:
            token_ids = np.flatnonzero(selected)
            frames = np.flatnonzero(frame[token_ids])
            return push(token_ids), push(frames), len(frames)

        chain_tokens = [asym == a for a in range(n_chain)]
        all_frames = np.flatnonzero(frame)
        self._all = (None, push(all_frames), len(all_frames))
        self._chains = [token_set(chain_tokens[a]) for a in range(n_chain)]
        self._pairs = {
            (a1, a2): token_set(chain_tokens[a1] | chain_tokens[a2])
            for a1 in range(n_chain)
            for a2 in range(a1 + 1, n_chain)
        }
        self.chain_has_frame = [bool(frame[chain_tokens[a]].any()) for a in range(n_chain)]
        self.chain_is_ligand = [
            bool(ligand[chain_tokens[a]].sum() >= chain_tokens[a].sum() // 2) for a in range(n_chain)
        ]
        chain_atoms = [atom_asym == a for a in range(n_chain)]
        self.chain_is_polymer = [bool(polymer[chain_atoms[a]].any()) for a in range(n_chain)]
        self.chain_atom_count = [int(chain_atoms[a].sum()) for a in range(n_chain)]
        self._chain_atoms = [push(np.flatnonzero(chain_atoms[a])) for a in range(n_chain)]
        self._pair_atoms = {
            pair: push(np.flatnonzero(chain_atoms[pair[0]] | chain_atoms[pair[1]])) for pair in self._pairs
        }

        offset = 0
        for index in lists:
            spans.append((offset, offset + len(index)))
            offset += len(index)
        blob = torch.from_numpy(np.concatenate(lists)).to(device)
        self._views = [blob[start:end] for start, end in spans]
        self.asym = self._views[0]

    def _token_set(self, entry: tuple[int | None, int, int]) -> _TokenSet:
        tokens, frames, n_frame = entry
        return _TokenSet(None if tokens is None else self._views[tokens], self._views[frames], n_frame)

    def all_tokens(self) -> _TokenSet:
        return self._token_set(self._all)

    def chain(self, a: int) -> _TokenSet:
        return self._token_set(self._chains[a])

    def pair(self, a1: int, a2: int) -> _TokenSet:
        return self._token_set(self._pairs[(min(a1, a2), max(a1, a2))])

    def chain_atoms(self, a: int) -> torch.Tensor:
        return self._views[self._chain_atoms[a]]

    def pair_atoms(self, a1: int, a2: int) -> torch.Tensor:
        return self._views[self._pair_atoms[(min(a1, a2), max(a1, a2))]]


def _token_token_ptm(
    pae_prob: torch.Tensor, selection: _TokenSet, n_token: int, min_bin: float, max_bin: float, no_bins: int
) -> torch.Tensor:
    """Per token pair TM term over the selected tokens (pae_prob restricted to them)."""
    if selection.tokens is not None:
        pae_prob = pae_prob.index_select(-3, selection.tokens).index_select(-2, selection.tokens)
    ptm_norm = _calculate_normalization(n_token)
    # In the default dtype, as get_bin_centers(...).to(device) kept it.
    bin_center = _bin_centers_on(pae_prob.device, torch.get_default_dtype(), min_bin, max_bin, no_bins)
    per_bin_weight = 1 / (1 + (bin_center / ptm_norm) ** 2)
    return (pae_prob * per_bin_weight).sum(dim=-1)


def calculate_ptm(
    pae_prob: torch.Tensor, selection: _TokenSet, n_token: int, min_bin: float, max_bin: float, no_bins: int
) -> torch.Tensor:
    """pTM (AF2/AF3) over the selected tokens: max over frame tokens of the mean per-token TM term."""
    if selection.n_frame == 0:
        return torch.zeros(pae_prob.shape[:-3], device=pae_prob.device)
    token_token_ptm = _token_token_ptm(pae_prob, selection, n_token, min_bin, max_bin, no_bins)
    return token_token_ptm.mean(dim=-1).index_select(-1, selection.frames).max(dim=-1).values


def calculate_iptm(
    pae_prob: torch.Tensor,
    asym_id: torch.Tensor,
    selection: _TokenSet,
    n_token: int,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Interface pTM over the selected tokens: per-token mean over *other-chain* tokens only."""
    if selection.n_frame == 0:
        return torch.zeros(pae_prob.shape[:-3], device=pae_prob.device)
    token_token_ptm = _token_token_ptm(pae_prob, selection, n_token, min_bin, max_bin, no_bins)
    if selection.tokens is not None:
        asym_id = asym_id.index_select(0, selection.tokens)
    is_diff_chain = asym_id[None, :] != asym_id[:, None]
    iptm = (token_token_ptm * is_diff_chain).sum(dim=-1) / (eps + is_diff_chain.sum(dim=-1))
    return iptm.index_select(-1, selection.frames).max(dim=-1).values


def calculate_chain_based_ptm(
    pae_prob: torch.Tensor, index: ChainIndex, min_bin: float, max_bin: float, no_bins: int
) -> dict:
    """Per-chain pTM / ipTM and chain-pair ipTM (+ ligand-aware global)."""
    n_chain = index.n_chain
    batch = pae_prob.shape[:-3]
    device = pae_prob.device
    args = (min_bin, max_bin, no_bins)

    def n_token(selection: _TokenSet) -> int:
        return selection.tokens.shape[0]

    chain_pair_iptm = torch.zeros(batch + (n_chain, n_chain), device=device)
    for a1 in range(n_chain):
        for a2 in range(n_chain):
            if a1 == a2:
                continue
            if a1 > a2:
                chain_pair_iptm[..., a1, a2] = chain_pair_iptm[..., a2, a1]
                continue
            pair = index.pair(a1, a2)
            chain_pair_iptm[..., a1, a2] = calculate_iptm(pae_prob, index.asym, pair, n_token(pair), *args)

    chain_ptm = torch.zeros(batch + (n_chain,), device=device)
    for a in range(n_chain):
        chain = index.chain(a)
        chain_ptm[..., a] = calculate_ptm(pae_prob, chain, n_token(chain), *args)

    chain_iptm = torch.zeros(batch + (n_chain,), device=device)
    for a in range(n_chain):
        pairs = [
            (i, j)
            for i in range(n_chain)
            for j in range(n_chain)
            if (i == a or j == a) and i != j and index.chain_has_frame[i]
        ]
        if pairs:
            chain_iptm[..., a] = torch.stack([chain_pair_iptm[..., i, j] for i, j in pairs], dim=-1).mean(dim=-1)

    chain_pair_iptm_global = torch.zeros(batch + (n_chain, n_chain), device=device)
    for a1 in range(n_chain):
        for a2 in range(n_chain):
            if a1 == a2:
                continue
            if index.chain_is_ligand[a1]:
                chain_pair_iptm_global[..., a1, a2] = chain_iptm[..., a1]
            elif index.chain_is_ligand[a2]:
                chain_pair_iptm_global[..., a1, a2] = chain_iptm[..., a2]
            else:
                chain_pair_iptm_global[..., a1, a2] = (chain_iptm[..., a1] + chain_iptm[..., a2]) * 0.5
    return {
        "chain_ptm": chain_ptm,
        "chain_iptm": chain_iptm,
        "chain_pair_iptm": chain_pair_iptm,
        "chain_pair_iptm_global": chain_pair_iptm_global,
    }


def calculate_chain_based_gpde(
    token_pair_pde: torch.Tensor, contact_probs: torch.Tensor, index: ChainIndex, eps: float = 1e-8
) -> dict:
    """Contact-weighted PDE within each chain (gPDE) and between chain pairs."""
    n_chain = index.n_chain
    batch = token_pair_pde.shape[:-2]
    device = token_pair_pde.device

    def _gpde(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        cp = contact_probs.index_select(-2, t1).index_select(-1, t2)
        pde = token_pair_pde.index_select(-2, t1).index_select(-1, t2)
        return (pde * cp).sum(dim=(-1, -2)) / (cp.sum(dim=(-1, -2)) + eps)

    chain_gpde = torch.zeros(batch + (n_chain,), device=device)
    for a in range(n_chain):
        tokens = index.chain(a).tokens
        chain_gpde[..., a] = _gpde(tokens, tokens)
    chain_pair_gpde = torch.zeros(batch + (n_chain, n_chain), device=device)
    for a1 in range(n_chain):
        for a2 in range(n_chain):
            if a1 == a2:
                continue
            if a2 < a1:
                chain_pair_gpde[..., a1, a2] = chain_pair_gpde[..., a2, a1]
                continue
            chain_pair_gpde[..., a1, a2] = _gpde(index.chain(a1).tokens, index.chain(a2).tokens)
    return {"chain_gpde": chain_gpde, "chain_pair_gpde": chain_pair_gpde}


def calculate_chain_based_plddt(atom_plddt: torch.Tensor, index: ChainIndex) -> dict:
    """Per-chain and chain-pair mean atom pLDDT."""
    n_chain = index.n_chain
    batch = atom_plddt.shape[:-1]
    device = atom_plddt.device

    chain_plddt = torch.zeros(batch + (n_chain,), device=device)
    for a in range(n_chain):
        chain_plddt[..., a] = atom_plddt.index_select(-1, index.chain_atoms(a)).mean(-1)
    chain_pair_plddt = torch.zeros(batch + (n_chain, n_chain), device=device)
    for a1 in range(n_chain):
        for a2 in range(a1 + 1, n_chain):
            # Both orders select the same atoms.
            pair_plddt = atom_plddt.index_select(-1, index.pair_atoms(a1, a2)).mean(-1)
            chain_pair_plddt[..., a1, a2] = pair_plddt
            chain_pair_plddt[..., a2, a1] = pair_plddt
    return {"chain_plddt": chain_plddt, "chain_pair_plddt": chain_pair_plddt}


def calculate_clash(pred_coordinate: torch.Tensor, index: ChainIndex, threshold: float) -> torch.Tensor:
    """AF3 steric clash between polymer-chain pairs (ligand chains excluded)."""
    n_sample = pred_coordinate.shape[0]
    n_chain = index.n_chain
    has_clash = torch.zeros(n_sample, n_chain, n_chain, device=pred_coordinate.device)
    for s in range(n_sample):
        for i in range(n_chain):
            if not index.chain_is_polymer[i]:
                continue
            ci = pred_coordinate[s].index_select(0, index.chain_atoms(i))
            for j in range(i + 1, n_chain):
                if not index.chain_is_polymer[j]:
                    continue
                cj = pred_coordinate[s].index_select(0, index.chain_atoms(j))
                total = (torch.cdist(ci, cj) < threshold).sum()
                # OSS: total > 100 or total / min(n_i, n_j) > 0.5; for integer counts
                # total / m > 0.5 exactly when 2 * total > m, so the flag stays on the device.
                smaller = min(index.chain_atom_count[i], index.chain_atom_count[j])
                flag = ((total > 100) | (2 * total > smaller)).float()
                has_clash[s, i, j] = flag
                has_clash[s, j, i] = flag
    return has_clash.reshape(n_sample, -1).max(dim=-1).values


class ProtenixConfidenceSummary(nn.Module):
    """Confidence summary bridge (OSS ``compute_full_data_and_summary``).

    Parameter-free. Converts distogram / confidence logits into OSS-format
    ``summary_confidence`` (list of per-sample score dicts) and optional
    ``full_data`` (list of per-sample logit/coord dicts).
    """

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        self.config = config

    def contact_probs(self, distogram_logits: torch.Tensor) -> torch.Tensor:
        """Contact probability from distogram logits (sample-independent).

        Args:
            distogram_logits: ``[N_token, N_token, no_bins]`` (unbatched)

        Returns:
            ``[N_token, N_token]`` contact probabilities
        """
        return compute_contact_prob(
            distogram_logits.float(), *tuple(self.config.distogram_bins), thres=self.config.contact_threshold
        )

    @staticmethod
    def token_is_ligand(
        asym_id: torch.Tensor, atom_to_token_idx: torch.Tensor, is_polymer: torch.Tensor
    ) -> torch.Tensor:
        """Per-token ligand flag (a token is ligand if any of its atoms is)."""
        atom_is_ligand = (1 - is_polymer).long()
        return torch.zeros_like(asym_id).scatter_add(0, atom_to_token_idx, atom_is_ligand) > 0

    @staticmethod
    def chain_index(
        asym_id: torch.Tensor,
        has_frame: torch.Tensor,
        token_is_ligand: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
        is_polymer: torch.Tensor,
    ) -> ChainIndex:
        """The structure's chain bookkeeping, shared by its samples (one host round trip)."""
        return ChainIndex(asym_id, has_frame, token_is_ligand, atom_to_token_idx, is_polymer)

    @torch.no_grad()
    def summary_one_sample(
        self,
        contact_probs: torch.Tensor,
        plddt_logits: torch.Tensor,
        pae_logits: torch.Tensor,
        pde_logits: torch.Tensor,
        coordinate: torch.Tensor,
        asym_id: torch.Tensor,
        has_frame: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
        is_polymer: torch.Tensor,
        token_is_ligand: torch.Tensor,
        num_recycles: int,
        return_full_data: bool = True,
        chain_index: ChainIndex | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Summary (+ full data) for one diffusion sample (RAII over heavy logits).

        Args:
            contact_probs: ``[N_token, N_token]``
            plddt_logits: ``[N_atom, b_plddt]``
            pae_logits: ``[N_token, N_token, b_pae]``
            pde_logits: ``[N_token, N_token, b_pde]``
            coordinate: ``[N_atom, 3]``
            chain_index: :meth:`chain_index` of the structure, when the caller shares it
                across samples; built here otherwise.

        Returns:
            ``(summary_dict, full_data_dict_or_None)`` — summary holds scalar /
            chain metrics (plddt, ptm, iptm, ranking_score, ...); full_data
            holds atom/token maps when ``return_full_data``.
        """
        cfg = self.config
        pae_bins = tuple(cfg.pae_bins)
        pde_bins = tuple(cfg.pde_bins)
        plddt_bins = tuple(cfg.plddt_bins)
        if chain_index is None:
            chain_index = self.chain_index(asym_id, has_frame, token_is_ligand, atom_to_token_idx, is_polymer)
        n_token = asym_id.shape[-1]

        atom_plddt = logits_to_score(plddt_logits.float(), *plddt_bins)
        token_pair_pde = logits_to_score(pde_logits.float(), *pde_bins)
        token_pair_pae, pae_prob = logits_to_score(pae_logits.float(), *pae_bins, return_prob=True)

        summary: dict[str, Any] = {}
        summary["plddt"] = atom_plddt.mean(dim=-1) * 100
        summary["gpde"] = (token_pair_pde * contact_probs).sum(dim=[-1, -2]) / contact_probs.sum(dim=[-1, -2])
        everything = chain_index.all_tokens()
        summary["ptm"] = calculate_ptm(pae_prob, everything, n_token, *pae_bins)
        summary["iptm"] = calculate_iptm(pae_prob, chain_index.asym, everything, n_token, *pae_bins)
        summary.update(calculate_chain_based_gpde(token_pair_pde, contact_probs, chain_index))
        summary.update(calculate_chain_based_ptm(pae_prob, chain_index, *pae_bins))
        summary.update(calculate_chain_based_plddt(atom_plddt, chain_index))
        summary["has_clash"] = calculate_clash(coordinate.unsqueeze(0), chain_index, cfg.af3_clash_threshold)[0]
        summary["disorder"] = torch.zeros_like(summary["ptm"])
        summary["num_recycles"] = torch.full((), num_recycles, dtype=torch.long, device=coordinate.device)
        summary["ranking_score"] = (
            cfg.iptm_weight * summary["iptm"]
            + cfg.ptm_weight * summary["ptm"]
            + cfg.disorder_weight * summary["disorder"]
            - cfg.clash_penalty * summary["has_clash"]
        )

        full = None
        if return_full_data:
            full = {
                "atom_plddt": atom_plddt,
                "token_pair_pde": token_pair_pde,
                "token_pair_pae": token_pair_pae,
                "atom_coordinate": coordinate,
                "contact_probs": contact_probs,
                "token_has_frame": has_frame,
                "token_asym_id": asym_id,
                "atom_to_token_idx": atom_to_token_idx,
                "atom_is_polymer": is_polymer,
            }
        return summary, full

    @torch.no_grad()
    def forward(
        self,
        distogram_logits: torch.Tensor,
        plddt_logits: torch.Tensor,
        pae_logits: torch.Tensor,
        pde_logits: torch.Tensor,
        coordinate: torch.Tensor,
        asym_id: torch.Tensor,
        has_frame: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
        is_polymer: torch.Tensor,
        num_recycles: int,
        return_full_data: bool = True,
    ) -> dict[str, Any]:
        """Batched entry: loop :meth:`summary_one_sample` (one-sample lifetimes).

        Args:
            distogram_logits: ``[N_token, N_token, no_bins]``
            plddt_logits: ``[N_sample, N_atom, b_plddt]``
            pae_logits / pde_logits: ``[N_sample, N_token, N_token, b_*]``
            coordinate: ``[N_sample, N_atom, 3]``

        Returns:
            Dict with ``summary_confidence`` (list[dict]), optional ``full_data``
            (list[dict]), plus ``coordinate`` / ``contact_probs``.
        """
        contact_probs = self.contact_probs(distogram_logits)
        asym_id = asym_id.long()
        atom_to_token_idx = atom_to_token_idx.long()
        token_is_ligand = self.token_is_ligand(asym_id, atom_to_token_idx, is_polymer)
        chain_index = self.chain_index(asym_id, has_frame, token_is_ligand, atom_to_token_idx, is_polymer)

        summary_list: list[dict[str, Any]] = []
        full_list: list[dict[str, Any]] = []
        for i in range(pae_logits.shape[0]):
            summary_i, full_i = self.summary_one_sample(
                contact_probs,
                plddt_logits[i],
                pae_logits[i],
                pde_logits[i],
                coordinate[i],
                asym_id,
                has_frame,
                atom_to_token_idx,
                is_polymer,
                token_is_ligand,
                num_recycles,
                return_full_data,
                chain_index=chain_index,
            )
            summary_list.append(summary_i)
            if full_i is not None:
                full_list.append(full_i)

        result = {
            "coordinate": coordinate,
            "contact_probs": contact_probs,
            "summary_confidence": summary_list,
        }
        if return_full_data:
            result["full_data"] = full_list
        return result
