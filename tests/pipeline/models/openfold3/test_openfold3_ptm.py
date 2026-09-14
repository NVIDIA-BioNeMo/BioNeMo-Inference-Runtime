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
"""Confidence post-processing of the OpenFold3 pipeline: pTM / ipTM reduction and bin centers.

AF3 SI §5.9.1 (eqs. 17-18), as AF2: pTM = max_i mean_j E[f(e_ij)], ipTM the
same with j restricted to tokens of other chains, f(e) = 1 / (1 + (e/d0)^2),
d0(N) = 1.24 (max(N, 19) - 15)^(1/3) - 1.8, E[.] over the 64 aligned-error bins
with midpoints 0.25 ... 31.75 Å. The outer reduction is a *max over the
aligned token i*, not a mean over tokens or token pairs. Expected PAE / pLDDT
use the same bin midpoints (pLDDT: 50 bins on [0, 1] -> 0.01 ... 0.99).
"""

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import tests
from bionemo_ir.data.schemas.basic import FoldingOutput
from bionemo_ir.pipeline.models.openfold3.postprocessor import (
    _aligned_token_mask,
    _compute_iptm,
    _compute_pae,
    _compute_plddt,
    _compute_ptm,
    _select_best_sample,
    _tm_score_from_pae_logits,
)
from tests.common.test_utils.basic import path_for_package_in_repo, require_vendored_submodule

N_BINS = 64
BIN_WIDTH = 32.0 / N_BINS
N_PLDDT_BINS = 50


def _d0(n_tokens: int) -> float:
    return 1.24 * (max(n_tokens, 19) - 15) ** (1.0 / 3.0) - 1.8


def _f(err: float | np.ndarray, n_tokens: int) -> float | np.ndarray:
    return 1.0 / (1.0 + (np.asarray(err, dtype=np.float64) / _d0(n_tokens)) ** 2)


def _one_hot_logits(bin_idx: np.ndarray, sharpness: float = 60.0, n_bins: int = N_BINS) -> torch.Tensor:
    """Logits whose softmax puts (numerically) all mass in ``bin_idx[...]``."""
    logits = torch.full((*bin_idx.shape, n_bins), -sharpness, dtype=torch.float32)
    idx = torch.as_tensor(bin_idx, dtype=torch.long)
    logits.scatter_(-1, idx.unsqueeze(-1), sharpness)
    return logits


def _bin_of(err_angstrom: float) -> int:
    """Index of the 0.5 Å aligned-error bin whose midpoint is closest to ``err``."""
    return int(np.clip(round(err_angstrom / BIN_WIDTH - 0.5), 0, N_BINS - 1))


# --- pTM / ipTM reduction ---------------------------------------------------


def test_ptm_is_max_over_aligned_tokens_not_mean() -> None:
    """One well-aligned token must dominate pTM (max over i), AF3 eq. 17."""
    n = 40
    good_bin, bad_bin = _bin_of(0.75), _bin_of(15.25)  # 0.75 Å and 15.25 Å midpoints
    bins = np.full((n, n), bad_bin)
    bins[0, :] = good_bin  # aligned on token 0, every token is placed within 0.75 Å
    logits = _one_hot_logits(bins)

    ptm = float(_tm_score_from_pae_logits(logits, n))

    expected_max = float(_f(0.75, n))  # row 0 mean
    expected_grand_mean = float((_f(0.75, n) + (n - 1) * _f(15.25, n)) / n)
    assert ptm == pytest.approx(expected_max, abs=1e-5)
    assert ptm > expected_grand_mean + 0.5  # a mean over token pairs returns the grand mean


def test_iptm_is_max_over_aligned_tokens_of_interchain_row_means() -> None:
    """ipTM = max_i mean_{j in other chains} f(e_ij) (AF3 eq. 18); intra-chain pairs never enter."""
    n_a, n_b = 25, 15
    n = n_a + n_b
    chain = np.array([0] * n_a + [1] * n_b)
    inter = chain[:, None] != chain[None, :]

    bins = np.full((n, n), _bin_of(20.25))
    bins[~inter] = _bin_of(0.25)  # perfect intra-chain blocks must not leak into ipTM
    bins[3, chain == 1] = _bin_of(2.25)  # token 3 (chain A) places chain B at 2.25 Å
    bins[30, chain == 0] = _bin_of(4.75)  # token 30 (chain B) places chain A at 4.75 Å
    logits = _one_hot_logits(bins)
    mask = torch.as_tensor(inter, dtype=torch.float32)

    iptm = float(_tm_score_from_pae_logits(logits, n, mask=mask))

    assert iptm == pytest.approx(float(_f(2.25, n)), abs=1e-5)  # best aligned token wins
    # pTM on the same logits scores all pairs of the best row (row 3: 25 intra at 0.25 Å, 15 inter at 2.25 Å)
    ptm = float(_tm_score_from_pae_logits(logits, n))
    assert ptm == pytest.approx(float((n_a * _f(0.25, n) + n_b * _f(2.25, n)) / n), abs=1e-5)


def test_tm_term_uses_bin_midpoints_and_af3_d0() -> None:
    """All mass in bin k -> pTM == f(0.25 + 0.5 k; d0(N)) exactly (bin midpoints, AF3 d0)."""
    n = 100
    for k in (0, 1, 7, 63):
        logits = _one_hot_logits(np.full((n, n), k))
        ptm = float(_tm_score_from_pae_logits(logits, n))
        assert ptm == pytest.approx(float(_f(0.25 + 0.5 * k, n)), abs=1e-6), k
    # d0 is clipped at N = 19 tokens
    logits = _one_hot_logits(np.full((5, 5), 3))
    assert float(_tm_score_from_pae_logits(logits, 5)) == pytest.approx(float(_f(1.75, 19)), abs=1e-6)
    assert _d0(5) == pytest.approx(1.24 * (19 - 15) ** (1 / 3) - 1.8)


def test_has_frame_restricts_the_aligned_tokens() -> None:
    """Tokens without a valid frame may be scored (j) but not aligned on (i); none eligible -> NaN."""
    n = 30
    bins = np.full((n, n), _bin_of(10.25))
    bins[0, :] = _bin_of(0.25)  # best row, but token 0 has no frame
    bins[1, :] = _bin_of(3.25)  # best row among tokens with a frame
    logits = _one_hot_logits(bins)
    has_frame = torch.ones(n, dtype=torch.bool)
    has_frame[0] = False

    assert float(_tm_score_from_pae_logits(logits, n)) == pytest.approx(float(_f(0.25, n)), abs=1e-5)
    assert float(_tm_score_from_pae_logits(logits, n, has_frame=has_frame)) == pytest.approx(
        float(_f(3.25, n)), abs=1e-5
    )
    assert math.isnan(float(_tm_score_from_pae_logits(logits, n, has_frame=torch.zeros(n, dtype=torch.bool))))


def test_compute_ptm_iptm_from_output_dict_selects_sample_and_crops_padding() -> None:
    """_compute_ptm/_compute_iptm: (B, S, N_pad, N_pad, 64) logits, best sample, padded tokens dropped."""
    n, n_pad, best = 24, 32, 1
    chain = np.array([0] * 10 + [1] * 14)
    rng = np.random.default_rng(0)
    logits = torch.as_tensor(rng.normal(size=(1, 2, n_pad, n_pad, N_BINS)), dtype=torch.float32)
    output = {"pae_logits": logits}

    ptm = _compute_ptm(output, best, n)
    iptm = _compute_iptm(output, best, n, chain)

    cropped = logits[0, best, :n, :n]
    inter = torch.as_tensor(chain[:, None] != chain[None, :], dtype=torch.float32)
    assert ptm == pytest.approx(float(_tm_score_from_pae_logits(cropped, n)), abs=1e-6)
    assert iptm == pytest.approx(float(_tm_score_from_pae_logits(cropped, n, mask=inter)), abs=1e-6)
    assert math.isnan(_compute_iptm(output, best, n, np.zeros(n, dtype=np.int64)))  # single chain
    assert math.isnan(_compute_ptm({}, best, n))  # no PAE head


# --- interim aligned-token (frame) mask at the call site ----------------------


def test_aligned_token_mask_excludes_atomized_tokens() -> None:
    """Interim has_frame at the call site: atomized tokens are scored (j) but not aligned on (i)."""
    n = 20
    is_atomized = torch.zeros(1, n, dtype=torch.int32)
    is_atomized[0, 15:] = 1  # e.g. a 5-atom ligand after a 15-residue chain
    has_frame = _aligned_token_mask({"is_atomized": is_atomized}, n)
    assert has_frame.dtype == torch.bool and has_frame.tolist() == [True] * 15 + [False] * 5
    assert _aligned_token_mask({}, n) is None  # flag absent -> every token eligible (has_frame=None path)

    # the best row belongs to a ligand atom: it sets pTM / ipTM only when atomized tokens are eligible
    chain = np.array([0] * 15 + [1] * 5)
    bins = np.full((n, n), _bin_of(12.25))
    bins[17, :] = _bin_of(0.25)  # ligand atom 17
    bins[4, :] = _bin_of(2.75)  # residue 4
    output = {"pae_logits": _one_hot_logits(bins)[None, None]}
    assert _compute_ptm(output, 0, n) == pytest.approx(float(_f(0.25, n)), abs=1e-5)
    assert _compute_ptm(output, 0, n, has_frame=has_frame) == pytest.approx(float(_f(2.75, n)), abs=1e-5)
    assert _compute_iptm(output, 0, n, chain, has_frame=has_frame) == pytest.approx(float(_f(2.75, n)), abs=1e-5)


def test_all_atomized_input_gives_nan_ptm_iptm() -> None:
    """Ligand-only query (every token atomized): no frame-eligible token -> pTM / ipTM NaN -> None."""
    n = 31  # e.g. one ATP: 31 heavy-atom tokens, all is_atomized
    two_ligands = np.array([0] * 20 + [1] * 11)
    has_frame = _aligned_token_mask({"is_atomized": torch.ones(1, n, dtype=torch.int32)}, n)
    assert has_frame is not None and not bool(has_frame.any())

    rng = np.random.default_rng(5)
    output = {"pae_logits": torch.as_tensor(rng.normal(size=(1, 1, n, n, N_BINS)), dtype=torch.float32)}
    assert math.isnan(_compute_ptm(output, 0, n, has_frame=has_frame))
    assert math.isnan(_compute_iptm(output, 0, n, two_ligands, has_frame=has_frame))
    # the has_frame=None path (flag absent) is unchanged and finite on the same logits
    assert 0.0 < _compute_ptm(output, 0, n) < 1.0
    assert 0.0 < _compute_iptm(output, 0, n, two_ligands) < 1.0


def test_get_scores_carries_the_frame_mask_convention() -> None:
    """``ptm_frame_mask`` set by the post-processor is serialised by get_scores(); absent -> key absent."""
    n = 3
    kwargs = {
        "atom_positions": np.zeros((n, 4, 3), dtype=np.float32),
        "residue_types": np.zeros(n, dtype=np.int64),
        "atom_mask": np.ones((n, 4), dtype=np.float32),
        "residue_indices": np.arange(n),
        "ptm": float("nan"),
        "iptm": 0.5,
    }
    plain = FoldingOutput(**kwargs)
    assert "ptm_frame_mask" not in plain.get_scores()
    assert plain.get_scores()["ptm"] is None and plain.get_scores()["iptm"] == pytest.approx(0.5)
    tagged = FoldingOutput(**kwargs)
    tagged["ptm_frame_mask"] = "polymer_tokens"
    assert tagged.get_scores()["ptm_frame_mask"] == "polymer_tokens"


# --- expected PAE / pLDDT at bin midpoints -----------------------------------


def test_pae_uses_bin_midpoints() -> None:
    """All mass in aligned-error bin k -> PAE_ij == 0.25 + 0.5 k Å (64 bins on [0, 32] Å)."""
    n = 12
    k = np.arange(n * n).reshape(n, n) % N_BINS
    pae = _compute_pae({"pae_logits": _one_hot_logits(k)[None, None]}, 0, n)
    np.testing.assert_allclose(pae, 0.25 + 0.5 * k, atol=1e-3)
    assert pae.min() == pytest.approx(0.25, abs=1e-3)  # was 0.0 with linspace(0, 32, 64)
    assert _compute_pae({"pae_logits": _one_hot_logits(np.full((n, n), N_BINS - 1))[None]}, 0, n).max() == (
        pytest.approx(31.75, abs=1e-3)  # was 32.0
    )


def test_plddt_uses_bin_midpoints() -> None:
    """All mass in pLDDT bin k -> pLDDT == 100 (k + 0.5) / 50 = 2k + 1 (50 bins on [0, 1], x100)."""
    n_atoms = N_PLDDT_BINS
    k = np.arange(n_atoms)  # atom a has all mass in bin a
    logits = _one_hot_logits(k, n_bins=N_PLDDT_BINS)[None, None]  # (B=1, S=1, N_atoms, 50)
    atom_to_token = np.arange(n_atoms)  # one atom per token
    plddt = _compute_plddt({"plddt_logits": logits}, 0, n_atoms, atom_to_token, np.ones(n_atoms, dtype=bool))
    np.testing.assert_allclose(plddt, 2.0 * k + 1.0, atol=1e-4)  # 1, 3, ..., 99 (was 0 ... 100)
    # per-token mean over atoms: token 0 <- atoms in bins 0 and 1, ...
    two_per_token = np.repeat(np.arange(n_atoms // 2), 2)
    plddt2 = _compute_plddt({"plddt_logits": logits}, 0, n_atoms // 2, two_per_token, np.ones(n_atoms, dtype=bool))
    np.testing.assert_allclose(plddt2, 0.5 * ((2.0 * k[0::2] + 1) + (2.0 * k[1::2] + 1)), atol=1e-4)


def test_best_sample_selection_is_invariant_under_the_bin_centre_change() -> None:
    """Midpoints are an increasing affine map of the old linspace positions, so the argmax is unchanged."""
    rng = np.random.default_rng(7)
    n_samples, n_atoms = 6, 40
    logits = torch.as_tensor(rng.normal(scale=3.0, size=(1, n_samples, n_atoms, N_PLDDT_BINS)), dtype=torch.float32)
    probs = torch.softmax(logits[0], dim=-1)
    old = (probs * torch.linspace(0, 1, N_PLDDT_BINS)).sum(-1).mean(-1)  # end-point convention
    new = (probs * (torch.arange(N_PLDDT_BINS) + 0.5) / N_PLDDT_BINS).sum(-1).mean(-1)
    torch.testing.assert_close(new, old * (N_PLDDT_BINS - 1) / N_PLDDT_BINS + 0.5 / N_PLDDT_BINS)
    assert int(old.argmax()) == int(new.argmax()) == _select_best_sample({"plddt_logits": logits})
    assert _select_best_sample({}) == 0


# --- parity with the vendored OSS OpenFold3 ------------------------------------


def _oss_confidence():
    """Vendored OSS ``openfold3/core/metrics/confidence.py`` loaded by path (torch-only); skips when absent."""
    root = require_vendored_submodule(path_for_package_in_repo(tests).parent / "3rdparty/openfold-3")
    spec = importlib.util.spec_from_file_location(
        "_oss_openfold3_confidence", Path(root) / "openfold3/core/metrics/confidence.py"
    )
    confidence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(confidence)
    return confidence


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_oss_openfold3_compute_ptm(seed: int) -> None:
    """Parity with upstream OpenFold3 ``compute_ptm`` (pTM, ipTM, with and without a frame mask)."""
    compute_ptm = _oss_confidence().compute_ptm
    rng = np.random.default_rng(seed)
    sizes = [(13, 9, 5), (40, 1, 22), (7, 7, 7)][seed]
    chain = np.concatenate([np.full(s, c) for c, s in enumerate(sizes)])
    n = len(chain)
    logits = torch.as_tensor(rng.normal(scale=2.0, size=(n, n, N_BINS)), dtype=torch.float32)
    has_frame = torch.as_tensor(rng.random(n) > 0.3)
    asym_id = torch.as_tensor(chain + 1)
    token_mask = torch.ones(n, dtype=torch.bool)
    inter = (asym_id[:, None] != asym_id[None, :]).float()

    for frame_mask in (torch.ones(n, dtype=torch.bool), has_frame):
        kwargs = {
            "logits": logits[None],
            "has_frame": frame_mask[None],
            "bin_min": 0,
            "bin_max": 32,
            "no_bins": N_BINS,
            "mask_i": token_mask,
            "asym_id": asym_id,
        }
        ref_ptm = float(compute_ptm(interface=False, **kwargs)[0])
        ref_iptm = float(compute_ptm(interface=True, **kwargs)[0])
        ours = {} if bool(frame_mask.all()) else {"has_frame": frame_mask}
        assert float(_tm_score_from_pae_logits(logits, n, **ours)) == pytest.approx(ref_ptm, abs=1e-5)
        assert float(_tm_score_from_pae_logits(logits, n, mask=inter, **ours)) == pytest.approx(ref_iptm, abs=1e-5)


def test_matches_oss_openfold3_expected_pae_and_plddt() -> None:
    """Parity of PAE / pLDDT with upstream ``probs_to_expected_error`` (64 bins, 0-32 Å) / ``compute_plddt``."""
    confidence = _oss_confidence()
    rng = np.random.default_rng(3)
    n, n_atoms = 17, 23
    pae_logits = torch.as_tensor(rng.normal(scale=2.0, size=(n, n, N_BINS)), dtype=torch.float32)
    ref_pae = confidence.probs_to_expected_error(torch.softmax(pae_logits, -1), bin_min=0, bin_max=32, no_bins=64)
    np.testing.assert_allclose(_compute_pae({"pae_logits": pae_logits[None]}, 0, n), ref_pae.numpy(), atol=1e-3)

    plddt_logits = torch.as_tensor(rng.normal(scale=2.0, size=(n_atoms, N_PLDDT_BINS)), dtype=torch.float32)
    ref_plddt = confidence.compute_plddt(plddt_logits).numpy() * 100.0
    ours = _compute_plddt({"plddt_logits": plddt_logits[None]}, 0, n_atoms, np.arange(n_atoms), np.ones(n_atoms, bool))
    np.testing.assert_allclose(ours, ref_plddt, atol=1e-4)


# --- report helper (not an assertion): per-sample values on a real multi-sample logits file ---

_REPORT_ENV = "BIOIR_OF3_CONFIDENCE_REPORT_NPZ"


def _pair_mean_tm(logits: torch.Tensor, n_tokens: int, mask: torch.Tensor | None = None) -> float:
    """The 0.1.0 reduction (mean over token pairs, linspace end points), for the before/after report only."""
    probs = torch.softmax(logits.float(), dim=-1)
    centers = torch.linspace(0, 32, probs.shape[-1])
    tm = (probs * (1.0 / (1.0 + (centers / max(_d0(n_tokens), 0.01)) ** 2))).sum(-1)
    if mask is None:
        return float(tm.mean())
    return float((tm * mask).sum() / mask.sum().clamp(min=1))


@pytest.mark.skipif(_REPORT_ENV not in os.environ, reason=f"set {_REPORT_ENV}=<file.npz> to print the report")
def test_report_per_sample_scores_before_and_after() -> None:
    """Prints 0.1.0-vs-patched pTM / ipTM per diffusion sample and the resulting sample order.

    Input ``.npz``: ``pae_logits`` (S, N, N, 64) [required]; optional ``asym_id`` or
    ``chain_index`` (N,), ``is_atomized`` (N,), ``plddt_logits`` (S, N_atoms, 50).
    Reports, per sample: pTM / ipTM with the 0.1.0 reduction and with this file's,
    ``0.8 ipTM + 0.2 pTM`` for both, and which sample each criterion ranks first
    (the pipeline's own choice is the mean-pLDDT argmax, unchanged by either fix).
    """
    data = np.load(os.environ[_REPORT_ENV])
    pae_logits = torch.as_tensor(data["pae_logits"], dtype=torch.float32)
    assert pae_logits.dim() == 4 and pae_logits.shape[-1] == N_BINS, pae_logits.shape
    n_samples, n = pae_logits.shape[0], pae_logits.shape[1]
    chain = None
    for key in ("asym_id", "chain_index"):
        if key in data.files:
            chain = np.asarray(data[key]).reshape(-1)[:n].astype(np.int64)
    has_frame = None
    if "is_atomized" in data.files:
        has_frame = _aligned_token_mask({"is_atomized": torch.as_tensor(np.asarray(data["is_atomized"]))}, n)
    inter = None if chain is None else torch.as_tensor(chain[:, None] != chain[None, :], dtype=torch.float32)

    rows = []
    for s in range(n_samples):
        out = {"pae_logits": pae_logits[s][None]}
        old_ptm = _pair_mean_tm(pae_logits[s], n)
        old_iptm = float("nan") if inter is None else _pair_mean_tm(pae_logits[s], n, inter)
        new_ptm = _compute_ptm(out, 0, n, has_frame=has_frame)
        new_iptm = float("nan") if chain is None else _compute_iptm(out, 0, n, chain, has_frame=has_frame)
        rows.append((s, old_ptm, new_ptm, old_iptm, new_iptm))

    def _rank(p: float, i: float) -> float:
        return p if math.isnan(i) else 0.8 * i + 0.2 * p

    print(
        f"\n{'sample':>6} {'pTM 0.1.0':>10} {'pTM new':>8} {'ipTM 0.1.0':>11} {'ipTM new':>9} {'rank 0.1.0':>11} {'rank new':>9}"
    )
    for s, op, np_, oi, ni in rows:
        print(f"{s:>6} {op:>10.4f} {np_:>8.4f} {oi:>11.4f} {ni:>9.4f} {_rank(op, oi):>11.4f} {_rank(np_, ni):>9.4f}")
    first_old = max(rows, key=lambda r: _rank(r[1], r[3]))[0]
    first_new = max(rows, key=lambda r: _rank(r[2], r[4]))[0]
    print(f"first by 0.8 ipTM + 0.2 pTM: 0.1.0 -> sample {first_old}; new -> sample {first_new}")
    if "plddt_logits" in data.files:
        best = _select_best_sample({"plddt_logits": torch.as_tensor(data["plddt_logits"], dtype=torch.float32)[None]})
        print(f"pipeline selection (mean pLDDT argmax, unchanged by this fix): sample {best}")
    print(f"aligned-token mask: {'none' if has_frame is None else 'polymer_tokens'}; tokens {n}; samples {n_samples}")
