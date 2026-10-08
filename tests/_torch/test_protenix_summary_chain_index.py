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
"""Protenix confidence summary: host chain bookkeeping against boolean masks and the OSS reference."""

import os

import pytest
import torch
from ml_collections.config_dict import ConfigDict

from bionemo_ir._torch.modules.protenix import ProtenixConfidenceSummary
from bionemo_ir._torch.modules.protenix.summary import ChainIndex
from bionemo_ir.models.protenix.config import ConfidenceSummaryConfig

_OSS_CONFIGS = ConfigDict(
    {
        "loss": {
            "plddt": {"min_bin": 0, "max_bin": 1.0, "no_bins": 50},
            "pde": {"min_bin": 0, "max_bin": 32, "no_bins": 64},
            "pae": {"min_bin": 0, "max_bin": 32, "no_bins": 64},
            "distogram": {"min_bin": 2.3125, "max_bin": 21.6875, "no_bins": 64},
        },
        "metrics": {"clash": {"af3_clash_threshold": 1.1}},
    }
)


def _irregular_structure(device: torch.device, seed: int = 0) -> dict:
    """Chains with non-contiguous, unsorted asym ids; a polymer chain without frames; a ligand chain."""
    generator = torch.Generator().manual_seed(seed)
    # asym 12: polymer with frames; 3: polymer without frames; 7: ligand; 9: polymer with frames
    chain_ids = [12, 3, 7, 9]
    lengths = [5, 3, 4, 6]
    asym_id = torch.cat([torch.full((n,), a) for a, n in zip(chain_ids, lengths, strict=True)])
    # Float flags with fractional values: the old masks tested nonzero, so 0.5 counts as set.
    has_frame = torch.tensor([0.5 if a == 9 else float(a == 12) for a in asym_id.tolist()])
    atoms_per_token = torch.randint(1, 4, (asym_id.numel(),), generator=generator)
    atom_to_token_idx = torch.repeat_interleave(torch.arange(asym_id.numel()), atoms_per_token)
    is_ligand = (asym_id[atom_to_token_idx] == 7).long()
    return {
        "asym_id": asym_id.to(device),
        "has_frame": has_frame.to(device),
        "atom_to_token_idx": atom_to_token_idx.to(device),
        "is_ligand": is_ligand.to(device),
        "generator": generator,
    }


def test_chain_index_lists_equal_the_boolean_masks():
    """Every index list selects the elements and order that the boolean mask of the old code selected."""
    s = _irregular_structure(torch.device("cpu"))
    asym_id, has_frame, atom_to_token_idx = s["asym_id"], s["has_frame"], s["atom_to_token_idx"]
    is_polymer = 1 - s["is_ligand"]
    token_is_ligand = ProtenixConfidenceSummary.token_is_ligand(asym_id, atom_to_token_idx, is_polymer)
    index = ChainIndex(asym_id, has_frame, token_is_ligand, atom_to_token_idx, is_polymer)

    chains = torch.unique(asym_id)  # the old code numbered chains in this order
    assert index.n_chain == chains.numel()
    assert torch.equal(index.asym, torch.searchsorted(chains, asym_id))
    frame = has_frame.bool()
    assert torch.equal(index.all_tokens().frames, frame.nonzero().flatten())
    atom_asym = index.asym[atom_to_token_idx]
    for a in range(index.n_chain):
        mask = index.asym == a
        chain = index.chain(a)
        assert torch.equal(chain.tokens, mask.nonzero().flatten())
        assert torch.equal(chain.frames, frame[mask].nonzero().flatten())
        assert chain.n_frame == int(frame[mask].sum())
        assert index.chain_has_frame[a] == bool(frame[mask].any())
        assert index.chain_is_ligand[a] == bool(token_is_ligand[mask].sum() >= mask.sum() // 2)
        atom_mask = atom_asym == a
        assert torch.equal(index.chain_atoms(a), atom_mask.nonzero().flatten())
        assert index.chain_is_polymer[a] == bool(is_polymer[atom_mask].any())
        assert index.chain_atom_count[a] == int(atom_mask.sum())
        for b in range(index.n_chain):
            if a == b:
                continue
            pair = index.pair(a, b)
            pair_mask = mask | (index.asym == b)
            assert torch.equal(pair.tokens, pair_mask.nonzero().flatten())
            assert torch.equal(pair.frames, frame[pair_mask].nonzero().flatten())
            assert torch.equal(index.pair_atoms(a, b), (atom_mask | (atom_asym == b)).nonzero().flatten())


def test_summary_on_irregular_chains_matches_oss():
    """Non-contiguous asym ids, a frameless polymer chain, a ligand chain and clashing chains, against OSS."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # The OSS reference needs the Protenix submodule; the boolean-mask test above does not.
    from tests.common.test_utils.protenix.ref_layers_from_oss import (
        oss_compute_contact_prob,
        oss_compute_full_data_and_summary,
    )

    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    s = _irregular_structure(device, seed=1)
    generator = s["generator"]
    n_token, n_atom, n_sample = s["asym_id"].numel(), s["atom_to_token_idx"].numel(), 2

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).to(device)

    coordinate = randn(n_sample, n_atom, 3) * 4
    # Pile two polymer chains onto one point in the second sample so the clash flag is exercised.
    atom_asym = s["asym_id"][s["atom_to_token_idx"]]
    coordinate[1, (atom_asym == 12) | (atom_asym == 9)] = 0.0
    inputs = {
        "plddt_logits": randn(n_sample, n_atom, 50),
        "pae_logits": randn(n_sample, n_token, n_token, 64),
        "pde_logits": randn(n_sample, n_token, n_token, 64),
        "coordinate": coordinate,
    }
    distogram_logits = randn(n_token, n_token, 64)
    contact_probs = oss_compute_contact_prob(distogram_logits, min_bin=2.3125, max_bin=21.6875, no_bins=64)
    oss_summary, _ = oss_compute_full_data_and_summary(
        configs=_OSS_CONFIGS,
        pae_logits=inputs["pae_logits"],
        plddt_logits=inputs["plddt_logits"],
        pde_logits=inputs["pde_logits"],
        contact_probs=contact_probs,
        token_asym_id=s["asym_id"],
        token_has_frame=s["has_frame"],
        atom_coordinate=inputs["coordinate"],
        atom_to_token_idx=s["atom_to_token_idx"],
        atom_is_polymer=1 - s["is_ligand"],
        N_recycle=3,
        return_full_data=False,
    )
    result = ProtenixConfidenceSummary(ConfidenceSummaryConfig()).to(device)(
        distogram_logits=distogram_logits,
        **inputs,
        asym_id=s["asym_id"],
        has_frame=s["has_frame"],
        atom_to_token_idx=s["atom_to_token_idx"],
        is_polymer=1 - s["is_ligand"],
        num_recycles=3,
        return_full_data=False,
    )
    assert bool(result["summary_confidence"][1]["has_clash"]) and bool(oss_summary[1]["has_clash"])
    for ours, oss in zip(result["summary_confidence"], oss_summary, strict=True):
        assert set(ours) == set(oss)
        for key, expected in oss.items():
            actual = ours[key]
            if not torch.is_tensor(expected) or not expected.is_floating_point():
                assert torch.equal(actual.long(), expected.long()), key
                continue
            torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-4, atol=2e-4, msg=key)
