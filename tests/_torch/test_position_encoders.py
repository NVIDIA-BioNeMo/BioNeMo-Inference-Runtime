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

from tensorrt_bionemo._torch.layers.position_encoders import RelativePositionEncoder

# Boltz2's real RelativePositionEncoder configuration
# (models/boltz2/modeling.py builds it; models/boltz2/config.py supplies the flags).
_BOLTZ2_TOKEN_Z = 128
_BOLTZ2_R_MAX = 32  # RelativePositionEncoder default (Boltz2 does not override)
_BOLTZ2_S_MAX = 2


def _boltz2_rpe(seed: int = 0) -> RelativePositionEncoder:
    """A Boltz2-configured RPE (embedding-gather path) with non-zero ("real") weights."""
    torch.manual_seed(seed)
    rpe = (
        RelativePositionEncoder(
            token_z=_BOLTZ2_TOKEN_Z,
            r_max=_BOLTZ2_R_MAX,
            s_max=_BOLTZ2_S_MAX,
            fix_sym_check=True,  # Boltz2 config default
            cyclic_pos_enc=True,  # Boltz2 config default
            period_broadcast=False,  # hardcoded False for Boltz2
            dtype=torch.float32,
            skip_create_weights=False,
        )
        .to("cpu")
        .eval()
    )  # CPU: the gather vs one-hot GEMM are bit-identical there
    # The Linear default-inits to zero, which would make the equivalence vacuous (0 == 0). Fill with
    # a realistic small-normal weight, standing in for the model's trained rel_pos.linear.
    with torch.no_grad():
        rpe.linear.weight.normal_(0.0, 0.1)
    return rpe


def _long(rows) -> torch.Tensor:
    return torch.tensor([rows], dtype=torch.long)


def _boltz2_inputs(cyclic: bool) -> dict:
    """A single complex with 3 chains exercising the relative-position branches.

    Chains A(res 0:15) and C(res 30:40) are symmetric copies of entity 0 (sym_id 0 vs 1); chain
    B(15:30) is entity 1. This covers same-chain / cross-chain, the same-entity sym check, and --
    when ``cyclic`` -- the cyclic-period branch on chain A.
    """
    asym = [0] * 15 + [1] * 15 + [2] * 10
    entity = [0] * 15 + [1] * 15 + [0] * 10
    sym = [0] * 15 + [0] * 15 + [1] * 10
    residue = list(range(15)) + list(range(15)) + list(range(10))
    token = list(range(40))
    cyclic_period = ([15] * 15 + [0] * 25) if cyclic else [0] * 40
    return {
        "asym_id": _long(asym),
        "residue_index": _long(residue),
        "entity_id": _long(entity),
        "cyclic_period": _long(cyclic_period),
        "token_index": _long(token),
        "sym_id": _long(sym),
    }


def _random_inputs(seed: int, n: int = 64) -> dict:
    """Random (but valid) rel-pos indices, single batch, to fuzz the slice/concat mapping."""
    g = torch.Generator().manual_seed(seed)
    randint = lambda hi: torch.randint(0, hi, (1, n), generator=g, dtype=torch.long)
    # ~half the tokens get a real cyclic period, the rest 0 (acyclic).
    cyclic_period = torch.where(randint(2).bool(), randint(19) + 1, torch.zeros(1, n, dtype=torch.long))
    return {
        "asym_id": randint(4),
        "residue_index": randint(50),
        "entity_id": randint(3),
        "cyclic_period": cyclic_period,
        "token_index": randint(n),
        "sym_id": randint(3),
    }


def _gather_vs_onehot(rpe: RelativePositionEncoder, inputs: dict):
    """Run the gather path and the one-hot reference on identical weights/inputs."""
    with torch.no_grad():
        out_gather = rpe(**inputs)  # forward() -> embedding-gather
        # Explicit one-hot + cat + Linear reference over the same weights/inputs:
        # ``generate_relp`` materializes the concatenated feature and ``forward(relp=...)``
        # projects it with a full matmul.
        out_onehot = rpe(relp=rpe.generate_relp(**inputs))
    return out_gather, out_onehot


@pytest.mark.parametrize("cyclic", [False, True], ids=["acyclic", "cyclic"])
def test_rpe_embedding_gather_bit_identical_to_onehot(cyclic):
    rpe = _boltz2_rpe()
    out_gather, out_onehot = _gather_vs_onehot(rpe, _boltz2_inputs(cyclic))

    assert out_gather.shape == (1, 40, 40, _BOLTZ2_TOKEN_Z)
    # Non-vacuous: non-zero weights -> non-trivial encoding (guards against a 0 == 0 pass).
    assert out_gather.abs().max() > 0
    # The gather's weight-row slices must match the one-hot concat order exactly.
    torch.testing.assert_close(out_gather, out_onehot, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [0, 1, 7, 123])
def test_rpe_embedding_gather_bit_identical_fuzz(seed):
    rpe = _boltz2_rpe(seed=seed)
    out_gather, out_onehot = _gather_vs_onehot(rpe, _random_inputs(seed))
    assert out_gather.abs().max() > 0
    torch.testing.assert_close(out_gather, out_onehot, rtol=0, atol=0)


def test_rpe_slice_order_matters():
    """Sanity: the encoding depends on the weight-row *layout*, so the bit-identical test above has
    teeth. Permuting the concat-order blocks of the weight must change the one-hot output."""
    rpe = _boltz2_rpe()
    inputs = _boltz2_inputs(cyclic=False)
    with torch.no_grad():
        ref = rpe(**inputs)
        # Swap the d_residue and d_token weight-row blocks (each n_pos wide). If the gather sliced
        # in this wrong order it would produce exactly this (different) result.
        n_pos = 2 * _BOLTZ2_R_MAX + 2
        w = rpe.linear.weight  # [token_z, K], columns are the concat blocks
        w[:, 0:n_pos], w[:, n_pos : 2 * n_pos] = (w[:, n_pos : 2 * n_pos].clone(), w[:, 0:n_pos].clone())
        permuted = rpe(**inputs)
    assert not torch.allclose(ref, permuted)
