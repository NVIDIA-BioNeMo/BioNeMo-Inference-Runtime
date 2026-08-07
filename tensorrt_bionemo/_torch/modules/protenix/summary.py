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
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_bionemo.configs import BaseConfig


def get_bin_centers(min_bin: float, max_bin: float, no_bins: int) -> torch.Tensor:
    """Bin centers for a ``[min_bin, max_bin]`` range split into ``no_bins``."""
    bin_width = (max_bin - min_bin) / no_bins
    boundaries = torch.linspace(min_bin, max_bin - bin_width, no_bins)
    return boundaries + 0.5 * bin_width


def logits_to_score(logits: torch.Tensor, min_bin: float, max_bin: float, no_bins: int, return_prob: bool = False):
    """Expected value of the bin centers under ``softmax(logits)``."""
    prob = F.softmax(logits, dim=-1)
    bin_centers = get_bin_centers(min_bin, max_bin, no_bins).to(logits.device, logits.dtype)
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


def _remap_contiguous(asym_id: torch.Tensor) -> torch.Tensor:
    """Remap ``asym_id`` to a contiguous ``0..N_chain-1`` range."""
    asym_id = asym_id.long()
    unique = torch.unique(asym_id)
    if len(unique) != asym_id.max() + 1:
        remap = {old.item(): new for new, old in enumerate(unique)}
        asym_id = torch.tensor([remap[int(x)] for x in asym_id], dtype=torch.long, device=asym_id.device)
    return asym_id


def _prepare_ptm(
    pae_prob: torch.Tensor,
    has_frame: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    token_mask: torch.Tensor | None = None,
    asym_id: torch.Tensor | None = None,
):
    """Shared PTM prep: apply token_mask, empty-frame early exit, bin weights.

    Returns ``(None, empty_zeros)`` when ``has_frame`` is empty, else
    ``((token_token_ptm, has_frame, asym_id), None)``.
    """
    has_frame = has_frame.bool()
    if token_mask is not None:
        token_mask = token_mask.bool()
        pae_prob = pae_prob[..., token_mask, :, :][..., :, token_mask, :]
        has_frame = has_frame[token_mask]
        if asym_id is not None:
            asym_id = asym_id[token_mask]
    if has_frame.sum() == 0:
        return None, torch.zeros(pae_prob.shape[:-3], device=pae_prob.device)
    ptm_norm = _calculate_normalization(has_frame.shape[-1])
    bin_center = get_bin_centers(min_bin, max_bin, no_bins).to(pae_prob.device)
    per_bin_weight = 1 / (1 + (bin_center / ptm_norm) ** 2)
    token_token_ptm = (pae_prob * per_bin_weight).sum(dim=-1)
    return (token_token_ptm, has_frame, asym_id), None


def calculate_ptm(
    pae_prob: torch.Tensor,
    has_frame: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """pTM (AF2/AF3): max over frame tokens of the mean per-token TM term."""
    prepared, early = _prepare_ptm(pae_prob, has_frame, min_bin, max_bin, no_bins, token_mask)
    if prepared is None:
        return early
    token_token_ptm, has_frame, _ = prepared
    return token_token_ptm.mean(dim=-1)[..., has_frame].max(dim=-1).values


def calculate_iptm(
    pae_prob: torch.Tensor,
    has_frame: torch.Tensor,
    asym_id: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    token_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Interface pTM: per-token mean over *other-chain* tokens only."""
    prepared, early = _prepare_ptm(pae_prob, has_frame, min_bin, max_bin, no_bins, token_mask, asym_id)
    if prepared is None:
        return early
    token_token_ptm, has_frame, asym_id = prepared
    is_diff_chain = asym_id[None, :] != asym_id[:, None]
    iptm = (token_token_ptm * is_diff_chain).sum(dim=-1) / (eps + is_diff_chain.sum(dim=-1))
    return iptm[..., has_frame].max(dim=-1).values


def calculate_chain_based_ptm(
    pae_prob: torch.Tensor,
    has_frame: torch.Tensor,
    asym_id: torch.Tensor,
    token_is_ligand: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
) -> dict:
    """Per-chain pTM / ipTM and chain-pair ipTM (+ ligand-aware global)."""
    has_frame = has_frame.bool()
    asym_id = _remap_contiguous(asym_id)
    masks = {int(a): asym_id == a for a in torch.unique(asym_id)}
    n_chain = len(masks)
    chain_is_ligand = {a: token_is_ligand[m].sum() >= m.sum() // 2 for a, m in masks.items()}
    batch = pae_prob.shape[:-3]
    device = pae_prob.device
    args = (min_bin, max_bin, no_bins)

    chain_pair_iptm = torch.zeros(batch + (n_chain, n_chain), device=device)
    for a1 in range(n_chain):
        for a2 in range(n_chain):
            if a1 == a2:
                continue
            if a1 > a2:
                chain_pair_iptm[..., a1, a2] = chain_pair_iptm[..., a2, a1]
                continue
            chain_pair_iptm[..., a1, a2] = calculate_iptm(
                pae_prob, has_frame, asym_id, *args, token_mask=masks[a1] + masks[a2]
            )

    chain_ptm = torch.zeros(batch + (n_chain,), device=device)
    for a, m in masks.items():
        chain_ptm[..., a] = calculate_ptm(pae_prob, has_frame, *args, token_mask=m)

    chain_has_frame = [bool((masks[i] * has_frame).any()) for i in range(n_chain)]
    chain_iptm = torch.zeros(batch + (n_chain,), device=device)
    for a in range(n_chain):
        pairs = [
            (i, j)
            for i in range(n_chain)
            for j in range(n_chain)
            if (i == a or j == a) and i != j and chain_has_frame[i]
        ]
        if pairs:
            chain_iptm[..., a] = torch.stack([chain_pair_iptm[..., i, j] for i, j in pairs], dim=-1).mean(dim=-1)

    chain_pair_iptm_global = torch.zeros(batch + (n_chain, n_chain), device=device)
    for a1 in range(n_chain):
        for a2 in range(n_chain):
            if a1 == a2:
                continue
            if chain_is_ligand[a1]:
                chain_pair_iptm_global[..., a1, a2] = chain_iptm[..., a1]
            elif chain_is_ligand[a2]:
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
    token_pair_pde: torch.Tensor, contact_probs: torch.Tensor, asym_id: torch.Tensor, eps: float = 1e-8
) -> dict:
    """Contact-weighted PDE within each chain (gPDE) and between chain pairs."""
    asym_id = _remap_contiguous(asym_id)
    n_chain = int(asym_id.max()) + 1
    batch = token_pair_pde.shape[:-2]
    device = token_pair_pde.device

    def _gpde(m1, m2):
        cp = contact_probs[..., m1, :][..., m2]
        pde = token_pair_pde[..., m1, :][..., m2]
        return (pde * cp).sum(dim=(-1, -2)) / (cp.sum(dim=(-1, -2)) + eps)

    chain_gpde = torch.zeros(batch + (n_chain,), device=device)
    for a in range(n_chain):
        chain_gpde[..., a] = _gpde(asym_id == a, asym_id == a)
    chain_pair_gpde = torch.zeros(batch + (n_chain, n_chain), device=device)
    for a1 in range(n_chain):
        for a2 in range(n_chain):
            if a1 == a2:
                continue
            if a2 < a1:
                chain_pair_gpde[..., a1, a2] = chain_pair_gpde[..., a2, a1]
                continue
            chain_pair_gpde[..., a1, a2] = _gpde(asym_id == a1, asym_id == a2)
    return {"chain_gpde": chain_gpde, "chain_pair_gpde": chain_pair_gpde}


def calculate_chain_based_plddt(
    atom_plddt: torch.Tensor, asym_id: torch.Tensor, atom_to_token_idx: torch.Tensor
) -> dict:
    """Per-chain and chain-pair mean atom pLDDT."""
    asym_id = _remap_contiguous(asym_id)
    masks = {int(a): asym_id == a for a in torch.unique(asym_id)}
    n_chain = len(masks)
    batch = atom_plddt.shape[:-1]
    device = atom_plddt.device

    def _mean(token_mask):
        return atom_plddt[..., token_mask[atom_to_token_idx]].mean(-1)

    chain_plddt = torch.zeros(batch + (n_chain,), device=device)
    for a, m in masks.items():
        chain_plddt[..., a] = _mean(m)
    chain_pair_plddt = torch.zeros(batch + (n_chain, n_chain), device=device)
    for a1 in masks:
        for a2 in masks:
            if a1 == a2:
                continue
            chain_pair_plddt[..., a1, a2] = _mean(masks[a1] + masks[a2])
    return {"chain_plddt": chain_plddt, "chain_pair_plddt": chain_pair_plddt}


def calculate_clash(
    pred_coordinate: torch.Tensor,
    asym_id: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    is_polymer: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """AF3 steric clash between polymer-chain pairs (ligand chains excluded)."""
    n_sample = pred_coordinate.shape[0]
    asym_id = _remap_contiguous(asym_id)
    n_chain = int(asym_id.max()) + 1
    atom_asym = asym_id[atom_to_token_idx]
    masks = {c: atom_asym == c for c in range(n_chain)}
    is_poly = {c: bool(is_polymer[masks[c]].any()) for c in range(n_chain)}
    has_clash = torch.zeros(n_sample, n_chain, n_chain, device=pred_coordinate.device)
    for s in range(n_sample):
        for i in range(n_chain):
            if not is_poly[i]:
                continue
            ci = pred_coordinate[s, masks[i]]
            n_i = ci.shape[0]
            for j in range(i + 1, n_chain):
                if not is_poly[j]:
                    continue
                cj = pred_coordinate[s, masks[j]]
                total = (torch.cdist(ci, cj) < threshold).sum().item()
                relative = total / min(n_i, cj.shape[0])
                flag = float(total > 100 or relative > 0.5)
                has_clash[s, i, j] = flag
                has_clash[s, j, i] = flag
    return has_clash.reshape(n_sample, -1).max(dim=-1).values


def break_down_to_per_sample_dict(input_dict: dict, shared_keys: list) -> list:
    """Split ``[N_sample, ...]`` tensors into per-sample dicts; copy ``shared_keys``."""
    per_sample = [k for k in input_dict if k not in shared_keys]
    n_sample = input_dict[per_sample[0]].size(0)
    out = []
    for i in range(n_sample):
        d = {k: input_dict[k][i] for k in per_sample}
        d.update({k: input_dict[k] for k in shared_keys})
        out.append(d)
    return out


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
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Summary (+ full data) for one diffusion sample (RAII over heavy logits).

        Args:
            contact_probs: ``[N_token, N_token]``
            plddt_logits: ``[N_atom, b_plddt]``
            pae_logits: ``[N_token, N_token, b_pae]``
            pde_logits: ``[N_token, N_token, b_pde]``
            coordinate: ``[N_atom, 3]``

        Returns:
            ``(summary_dict, full_data_dict_or_None)`` — summary holds scalar /
            chain metrics (plddt, ptm, iptm, ranking_score, ...); full_data
            holds atom/token maps when ``return_full_data``.
        """
        cfg = self.config
        pae_bins = tuple(cfg.pae_bins)
        pde_bins = tuple(cfg.pde_bins)
        plddt_bins = tuple(cfg.plddt_bins)

        atom_plddt = logits_to_score(plddt_logits.float(), *plddt_bins)
        token_pair_pde = logits_to_score(pde_logits.float(), *pde_bins)
        token_pair_pae, pae_prob = logits_to_score(pae_logits.float(), *pae_bins, return_prob=True)

        summary: dict[str, Any] = {}
        summary["plddt"] = atom_plddt.mean(dim=-1) * 100
        summary["gpde"] = (token_pair_pde * contact_probs).sum(dim=[-1, -2]) / contact_probs.sum(dim=[-1, -2])
        summary["ptm"] = calculate_ptm(pae_prob, has_frame, *pae_bins)
        summary["iptm"] = calculate_iptm(pae_prob, has_frame, asym_id, *pae_bins)
        summary.update(calculate_chain_based_gpde(token_pair_pde, contact_probs, asym_id))
        summary.update(calculate_chain_based_ptm(pae_prob, has_frame, asym_id, token_is_ligand, *pae_bins))
        summary.update(calculate_chain_based_plddt(atom_plddt, asym_id, atom_to_token_idx))
        summary["has_clash"] = calculate_clash(
            coordinate.unsqueeze(0), asym_id, atom_to_token_idx, is_polymer, cfg.af3_clash_threshold
        )[0]
        summary["disorder"] = torch.zeros_like(summary["ptm"])
        summary["num_recycles"] = torch.tensor(num_recycles, device=coordinate.device)
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
