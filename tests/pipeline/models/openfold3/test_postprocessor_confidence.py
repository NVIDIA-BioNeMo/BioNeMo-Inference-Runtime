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
"""Confidence-score math in the OpenFold3 postprocessor: bins, pTM, ipTM.

Two things here are easy to get subtly wrong and impossible to notice from the
output alone, since both failures return a plausible number in [0, 1]:

  * **Bin centers.** A binned head predicts a distribution over bins, so its
    expectation weights each bin by that bin's *midpoint*. Weighting by
    ``linspace(bin_min, bin_max, n_bins)`` takes the bin edges instead and
    biases every pLDDT, PAE and pTM low.
  * **The pTM reduction.** pTM averages over the scored partners ``j`` of an
    aligned token and then takes the **maximum** over aligned tokens ``i``
    (AF3 SI 5.9.1); it is the score from the structure's best alignment frame.
    A mean over ``i``, or over all pairs at once, is a lower bound on it and is
    not comparable with AF3-calibrated thresholds.

Only a token with a valid frame may be the aligned token, so the maximum is
restricted to ``valid_frame_mask`` from the confidence head. These tests pin
the reduction, the restriction, and parity against the vendored ``openfold-3``.
"""

import numpy as np
import pytest
import torch

from bionemo_ir.pipeline.models.openfold3.postprocessor import (
    _bin_centers,
    _compute_iptm,
    _compute_plddt,
    _compute_ptm,
    _frame_mask,
    _pae_logits,
    _plddt_per_atom,
    _select_best_sample,
    _tm_score_from_pae_logits,
)

N_PAE_BINS = 64
PAE_RANGE = (0.0, 32.0)


def _pae_output(logits: torch.Tensor, valid_frame_mask: torch.Tensor | None = None) -> dict:
    """A model output dict with the (B, S, ...) axes the postprocessor expects."""
    output = {"pae_logits": logits[None, None]}
    if valid_frame_mask is not None:
        # The head casts every aux output to the model dtype, so the mask is 0/1 floats.
        output["valid_frame_mask"] = valid_frame_mask.float()[None, None]
    return output


def _confident_for_token(n_tokens: int, token: int) -> torch.Tensor:
    """PAE logits where every pair looks bad except the rows/cols of ``token``."""
    logits = torch.full((n_tokens, n_tokens, N_PAE_BINS), -8.0)
    logits[..., 40] = 4.0  # mass on a far bin => low TM term
    logits[token, :, 40] = -8.0
    logits[token, :, 0] = 8.0  # mass on the nearest bin => high TM term
    return logits


def test_bin_centers_are_midpoints_not_edges():
    """The documented convention, and the one the rest of BioIR already uses."""
    pae = _bin_centers(*PAE_RANGE, N_PAE_BINS)
    assert pae.shape == (N_PAE_BINS,)
    assert pae[0] == pytest.approx(0.25)
    assert pae[-1] == pytest.approx(31.75)

    plddt = _bin_centers(0.0, 1.0, 50)
    assert plddt[0] == pytest.approx(0.01)
    assert plddt[-1] == pytest.approx(0.99)

    # Midpoints lie strictly inside the range, one bin width apart. The linspace
    # grid instead touches both end points, stretching the spacing to cover them.
    assert 0.0 < float(pae[0]) and float(pae[-1]) < PAE_RANGE[1]
    assert torch.allclose(pae.diff(), torch.full((N_PAE_BINS - 1,), 0.5))

    edges = torch.linspace(*PAE_RANGE, N_PAE_BINS)
    assert float(edges[0]) == PAE_RANGE[0] and float(edges[-1]) == PAE_RANGE[1]
    assert not torch.allclose(pae, edges)


def test_ptm_is_a_max_over_aligned_tokens_not_a_mean():
    """One well-aligned token carries pTM; a mean over pairs would bury it."""
    n_tokens = 24
    logits = _confident_for_token(n_tokens, token=3)

    ptm = _compute_ptm(_pae_logits(_pae_output(logits), 0, n_tokens), n_tokens)

    # The per-pair mean of the same quantity, i.e. the reduction to avoid.
    d0 = 1.24 * (max(n_tokens, 19) - 15) ** (1.0 / 3.0) - 1.8
    tm_per_bin = 1.0 / (1.0 + (_bin_centers(*PAE_RANGE, N_PAE_BINS) / d0) ** 2)
    per_pair_mean = float((torch.softmax(logits, dim=-1) * tm_per_bin).sum(-1).mean())

    assert ptm > 0.8
    assert per_pair_mean < 0.1
    # The rows of the one good token are 1/24 of the matrix, so the mean is far off.
    assert ptm > 5 * per_pair_mean


def test_has_frame_restricts_the_aligned_token():
    """Excluding the only well-aligned token must lower the score."""
    n_tokens = 24
    logits = _confident_for_token(n_tokens, token=3)

    unrestricted = _compute_ptm(_pae_logits(_pae_output(logits), 0, n_tokens), n_tokens)

    frames = torch.ones(n_tokens, dtype=torch.bool)
    frames[3] = False
    output = _pae_output(logits, valid_frame_mask=frames)
    restricted = _compute_ptm(_pae_logits(output, 0, n_tokens), n_tokens, has_frame=_frame_mask(output, 0, n_tokens))

    assert restricted < unrestricted


def test_no_frame_eligible_token_gives_nan():
    """NaN, not 0.0: the score is unavailable rather than bad.

    ``FoldingOutput.get_scores()`` maps NaN to ``None``, which is how this module
    already reports the ipTM of a single-chain input. Upstream returns 0.0 here.
    """
    n_tokens = 8
    logits = _confident_for_token(n_tokens, token=0)
    frames = torch.zeros(n_tokens, dtype=torch.bool)
    output = _pae_output(logits, valid_frame_mask=frames)

    ptm = _compute_ptm(_pae_logits(output, 0, n_tokens), n_tokens, has_frame=_frame_mask(output, 0, n_tokens))
    assert np.isnan(ptm)


def test_iptm_scores_only_inter_chain_pairs_and_is_nan_for_one_chain():
    """ipTM is the interface score, so intra-chain pairs must not contribute."""
    n_tokens = 8
    chains_two = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    chains_one = np.zeros(n_tokens, dtype=np.int64)

    # Confident only within chain 0: a real interface score should stay low.
    logits = torch.full((n_tokens, n_tokens, N_PAE_BINS), -8.0)
    logits[..., 40] = 4.0
    logits[:4, :4, 40] = -8.0
    logits[:4, :4, 0] = 8.0
    pae_logits = _pae_logits(_pae_output(logits), 0, n_tokens)

    iptm = _compute_iptm(pae_logits, n_tokens, chain_indices=chains_two)
    ptm = _compute_ptm(pae_logits, n_tokens)
    assert iptm < 0.1 < ptm

    assert np.isnan(_compute_iptm(pae_logits, n_tokens, chain_indices=chains_one))
    # Absent PAE head: no logits, so no scores.
    assert np.isnan(_compute_ptm(None, n_tokens))
    assert np.isnan(_compute_iptm(None, n_tokens, chain_indices=chains_two))


def test_frame_mask_reader_handles_sample_axis_padding_and_dtype():
    """The head emits (B, S, N_token) floats; the reader crops and converts."""
    n_tokens, n_padded, n_samples = 5, 8, 3
    per_sample = torch.zeros(n_samples, n_padded)
    per_sample[1, :n_tokens] = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])

    mask = _frame_mask({"valid_frame_mask": per_sample[None]}, best_idx=1, n_tokens=n_tokens)
    assert mask is not None
    assert mask.dtype == torch.bool
    assert mask.tolist() == [True, False, True, True, False]

    # Without a sample axis, and absent entirely when the PAE head is off.
    flat = _frame_mask({"valid_frame_mask": per_sample[1][None]}, best_idx=0, n_tokens=n_tokens)
    assert flat.tolist() == [True, False, True, True, False]
    assert _frame_mask({}, best_idx=0, n_tokens=n_tokens) is None


def test_pae_logit_extraction_selects_the_sample_and_crops_padding():
    """pTM, ipTM and PAE share one extraction, so it has to pick the right block."""
    n_tokens, n_padded, n_samples = 4, 7, 3
    logits = torch.zeros(1, n_samples, n_padded, n_padded, N_PAE_BINS)
    logits[0, 2, :n_tokens, :n_tokens] = 5.0  # mark the sample and region we want

    got = _pae_logits({"pae_logits": logits}, best_idx=2, n_tokens=n_tokens)
    assert got.shape == (n_tokens, n_tokens, N_PAE_BINS)
    assert torch.all(got == 5.0)

    # Without a sample axis, and absent when the PAE head is off.
    flat = _pae_logits({"pae_logits": logits[:, 2]}, best_idx=0, n_tokens=n_tokens)
    assert flat.shape == (n_tokens, n_tokens, N_PAE_BINS)
    assert _pae_logits({}, best_idx=0, n_tokens=n_tokens) is None


def test_one_plddt_expectation_feeds_both_selection_and_score():
    """Sample choice and reported pLDDT are the same expectation, computed once."""
    n_atom, n_bins, n_tokens, n_samples = 6, 50, 3, 4
    logits = torch.zeros(1, n_samples, n_atom, n_bins)
    logits[0, :, :, 0] = 20.0  # every sample confident of the *lowest* pLDDT bin
    logits[0, 2, :, 0] = 0.0
    logits[0, 2, :, -1] = 20.0  # except sample 2, which should therefore win

    per_atom = _plddt_per_atom({"plddt_logits": logits})
    assert per_atom.shape == (n_samples, n_atom)
    assert _select_best_sample(per_atom) == 2
    # Midpoints run 0.01 ... 0.99, reported on a 0-100 scale.
    assert float(per_atom[2].mean()) > 90.0
    assert float(per_atom[0].mean()) < 10.0

    # Per-token mean over present atoms. Atom 3 is absent and atom 5 maps past
    # the real tokens, so token 2 ends up with no atoms at all.
    atom_to_token = np.array([0, 0, 1, 1, 1, 9], dtype=np.int64)
    atom_mask_bool = np.array([True, True, True, False, True, True])
    plddt = _compute_plddt(per_atom, 2, n_tokens, atom_to_token, atom_mask_bool)
    assert plddt.shape == (n_tokens,)
    assert plddt[0] > 90.0 and plddt[1] > 90.0
    assert plddt[2] == 0.0

    # Shape dispatch, and the fallbacks when the pLDDT head is absent.
    assert _plddt_per_atom({"plddt_logits": logits[:, 0]}).shape == (1, n_atom)
    assert _plddt_per_atom({"plddt_logits": logits[0, 0]}).shape == (1, n_atom)
    assert _plddt_per_atom({}) is None
    assert _select_best_sample(None) == 0
    assert np.all(_compute_plddt(None, 0, n_tokens, atom_to_token, atom_mask_bool) == 50.0)


_OPENFOLD3_PTM_FIXTURES = [
    (0, 0.0771128386259079, 0.07978863269090652),
    (1, 0.030543696135282516, 0.02889089658856392),
    (2, 0.08182767778635025, 0.09414934366941452),
    (3, 0.07244866341352463, 0.07187891006469727),
    (4, 0.06491488963365555, 0.0688648521900177),
]


@pytest.mark.parametrize(("seed", "expected_ptm", "expected_iptm"), _OPENFOLD3_PTM_FIXTURES)
def test_matches_vendored_openfold3_compute_ptm(seed, expected_ptm, expected_iptm):
    """Parity with upstream, which is the definition of these scores.

    Expected values come from the pinned OpenFold3 0.4.3 ``compute_ptm`` in
    float32. Keeping them here makes the parity check independent of whether the
    vendored submodule is present.
    """
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    n_tokens = int(rng.integers(4, 40))
    chain_indices = rng.choice(rng.choice(np.arange(1, 9), size=3, replace=False), size=n_tokens).astype(np.int64)
    logits = torch.randn(n_tokens, n_tokens, N_PAE_BINS, generator=g) * float(rng.uniform(0.5, 3.0))
    frames = torch.rand(n_tokens, generator=g) < 0.7
    frames[int(rng.integers(n_tokens))] = True  # keep at least one eligible

    assert _tm_score_from_pae_logits(logits, n_tokens, has_frame=frames) == pytest.approx(expected_ptm, abs=1e-6)

    ci = torch.as_tensor(chain_indices, dtype=torch.long)
    pair_mask = (ci.unsqueeze(-1) != ci.unsqueeze(-2)).float()
    got_iptm = _tm_score_from_pae_logits(logits, n_tokens, pair_mask=pair_mask, has_frame=frames)
    assert got_iptm == pytest.approx(expected_iptm, abs=1e-6)
