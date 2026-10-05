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

import pytest
import torch

from bionemo_ir._torch.modules.boltz.confidence_utils import (
    NUM_CONTACT_BINS,
    compute_contact_prob,
    repeat_with_multiplicity,
)
from bionemo_ir._torch.utils import ChunkPolicy


def _reference_contact_prob(logits: torch.Tensor, num_contact_bins: int = NUM_CONTACT_BINS) -> torch.Tensor:
    """The pre-reduction confidence-head expression: softmax over bins, mask, sum.

    Mirrors the original ``(softmax(logits, -1) * contacts).sum(-1)`` with ``contacts`` a full-width
    0/1 mask, which is what ``compute_contact_prob`` must reproduce.
    """
    prob = torch.softmax(logits.float(), dim=-1)
    contacts = torch.zeros(logits.shape[-1], dtype=torch.float32, device=logits.device)
    contacts[:num_contact_bins] = 1.0
    return (prob * contacts).sum(-1)


@pytest.mark.parametrize("n_tokens", [7, 64, 129])
@pytest.mark.parametrize("num_bins", [64, 38])
def test_matches_head_reduction(n_tokens, num_bins):
    torch.manual_seed(0)
    logits = torch.randn(2, n_tokens, n_tokens, num_bins)

    torch.testing.assert_close(
        compute_contact_prob(logits),
        _reference_contact_prob(logits),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("chunk_size", [1, 3, 16, 512])
def test_row_chunking_is_exact(chunk_size):
    """Row-chunking must not perturb the result: softmax is independent per pair row."""
    torch.manual_seed(0)
    logits = torch.randn(1, 33, 33, 64)

    dense = compute_contact_prob(logits, policy=ChunkPolicy(chunk_size=chunk_size, min_size=10**9, dim=1))
    chunked = compute_contact_prob(logits, policy=ChunkPolicy(chunk_size=chunk_size, min_size=0, dim=1))

    torch.testing.assert_close(chunked, dense, rtol=0, atol=0)
    assert chunked.shape == (1, 33, 33)


def test_reduces_pair_dim_and_is_a_probability():
    torch.manual_seed(0)
    logits = torch.randn(1, 12, 12, 64)
    out = compute_contact_prob(logits)

    assert out.shape == (1, 12, 12)
    assert out.dtype == torch.float32
    assert ((out >= 0) & (out <= 1)).all()


def test_num_contact_bins_selects_leading_bins():
    """All mass in a bin inside/outside the cutoff drives the probability to 1/0."""
    logits = torch.full((1, 2, 2, 64), -1e4)
    logits[..., 0] = 1e4  # nearest bin -> a contact
    torch.testing.assert_close(compute_contact_prob(logits), torch.ones(1, 2, 2))

    logits = torch.full((1, 2, 2, 64), -1e4)
    logits[..., NUM_CONTACT_BINS] = 1e4  # first bin beyond the cutoff -> not a contact
    torch.testing.assert_close(compute_contact_prob(logits), torch.zeros(1, 2, 2))


@pytest.mark.parametrize("multiplicity", [1, 3])
def test_reduce_then_repeat_matches_repeat_then_reduce(multiplicity):
    """The heads now repeat an already-reduced ``prob_contact`` instead of repeating the logits.

    ``repeat_interleave`` copies exactly, so hoisting the reduction ahead of the sample repeat is
    value-preserving -- and it drops the per-sample ``[B, mult, N, N, num_bins]`` softmax entirely.
    """
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 5, 64)

    reduce_then_repeat = repeat_with_multiplicity(compute_contact_prob(logits), multiplicity)
    repeat_then_reduce = _reference_contact_prob(repeat_with_multiplicity(logits, multiplicity))

    torch.testing.assert_close(reduce_then_repeat, repeat_then_reduce, rtol=0, atol=0)
    assert reduce_then_repeat.shape == (2, multiplicity, 5, 5)


def test_bf16_logits_reduce_in_fp32():
    """bf16 logits are promoted before the softmax, so the result keeps fp32 resolution."""
    torch.manual_seed(0)
    logits = torch.randn(1, 8, 8, 64).bfloat16()
    out = compute_contact_prob(logits)

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, _reference_contact_prob(logits), rtol=0, atol=0)


def _reference_frame_pred(pred_atom_coords, frames_idx_true, feats):
    """The previous ``compute_frame_pred``: per-chain ``unique``/``item`` loop with boolean gathers."""
    from bionemo_ir.pipeline.models.boltz2.const import chain_type_ids

    asym_id_token = feats["asym_id"]
    asym_id_atom = torch.bmm(feats["atom_to_token"].float(), asym_id_token.unsqueeze(-1).float()).squeeze(-1)
    _, multiplicity, _, _ = pred_atom_coords.shape
    frames_idx_pred = repeat_with_multiplicity(frames_idx_true, multiplicity)
    for i, pred_atom_coord in enumerate(pred_atom_coords):
        token_idx = 0
        atom_idx = 0
        for chain_id in torch.unique(asym_id_token[i]):
            mask_chain_token = (asym_id_token[i] == chain_id) * feats["token_pad_mask"][i]
            mask_chain_atom = (asym_id_atom[i] == chain_id) * feats["atom_pad_mask"][i]
            num_tokens = int(mask_chain_token.sum().item())
            num_atoms = int(mask_chain_atom.sum().item())
            if feats["mol_type"][i, token_idx] != chain_type_ids["NONPOLYMER"] or num_atoms < 3:
                token_idx += num_tokens
                atom_idx += num_atoms
                continue
            chain_coords = pred_atom_coord[:, mask_chain_atom.bool()]
            dist_mat = ((chain_coords[:, None, :, :] - chain_coords[:, :, None, :]) ** 2).sum(-1) ** 0.5
            indices = torch.sort(dist_mat, dim=2).indices
            frames = torch.cat([indices[:, :, 1:2], indices[:, :, 0:1], indices[:, :, 2:3]], dim=2) + atom_idx
            frames_idx_pred[i, :, token_idx : token_idx + num_atoms, :] = frames
            token_idx += num_tokens
            atom_idx += num_atoms
    return frames_idx_pred


def _frame_feats(device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Protein chain (3 tokens x 4 atoms), ligand (5 atoms), 2-atom ligand (skipped), then padding."""
    from bionemo_ir.pipeline.models.boltz2.const import chain_type_ids

    n_tokens, n_atoms = 12, 24
    token_chain = [0] * 3 + [1] * 5 + [2] * 2 + [0] * 2  # padded tokens carry asym_id 0
    token_pad = [1] * 10 + [0] * 2
    mol_type = [chain_type_ids["PROTEIN"]] * 3 + [chain_type_ids["NONPOLYMER"]] * 7 + [chain_type_ids["PROTEIN"]] * 2
    atom_token = [0] * 4 + [1] * 4 + [2] * 4 + [3, 4, 5, 6, 7] + [8, 9]
    atom_to_token = torch.zeros(1, n_atoms, n_tokens, device=device)
    atom_to_token[0, torch.arange(len(atom_token)), torch.tensor(atom_token)] = 1.0
    atom_pad = torch.zeros(1, n_atoms, device=device)
    atom_pad[0, : len(atom_token)] = 1.0
    feats = {
        "asym_id": torch.tensor([token_chain], device=device),
        "token_pad_mask": torch.tensor([token_pad], device=device, dtype=torch.float32),
        "mol_type": torch.tensor([mol_type], device=device),
        "atom_to_token": atom_to_token,
        "atom_pad_mask": atom_pad,
    }
    frames_idx_true = torch.randint(0, len(atom_token), (1, n_tokens, 3), device=device)
    return feats, frames_idx_true


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
@pytest.mark.parametrize("multiplicity", [1, 3])
def test_compute_frame_pred_matches_reference(device: str, multiplicity: int) -> None:
    from bionemo_ir._torch.modules.boltz.confidence_utils import chain_table, compute_frame_pred

    torch.manual_seed(0)
    device = torch.device(device)
    feats, frames_idx_true = _frame_feats(device)
    pred = torch.randn(1, multiplicity, feats["atom_pad_mask"].shape[1], 3, device=device)
    chains = chain_table(feats)
    assert chains.asym_ids == (0, 1, 2)
    assert chains.frame_spans == (((3, 12, 5),),)
    frames, collinear = compute_frame_pred(pred, frames_idx_true, feats, chains=chains)
    frames_default, collinear_default = compute_frame_pred(pred, frames_idx_true, feats)
    expected = _reference_frame_pred(pred, frames_idx_true, feats)
    torch.testing.assert_close(frames, expected, atol=0, rtol=0)
    torch.testing.assert_close(frames_default, expected, atol=0, rtol=0)
    torch.testing.assert_close(collinear, collinear_default, atol=0, rtol=0)
    assert collinear.shape == (1, multiplicity, feats["asym_id"].shape[1])
