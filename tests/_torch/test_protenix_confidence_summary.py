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
"""OSS equivalence test for Protenix confidence post-processing."""

import os
from dataclasses import dataclass

import pytest
import torch
from ml_collections.config_dict import ConfigDict

from tensorrt_bionemo._torch.modules.protenix import ProtenixConfidenceSummary
from tensorrt_bionemo.models.protenix.config import ConfidenceSummaryConfig
from tests.common.test_utils.protenix.ref_layers_from_oss import (
    oss_compute_contact_prob,
    oss_compute_full_data_and_summary,
)


@dataclass(kw_only=True, frozen=True)
class Scenario:
    tokens_per_chain: int = 4
    atoms_per_token: int = 3
    n_sample: int = 2
    # Polymer chains have frames; the ligand does not.
    n_poly_chains: int = 2
    n_lig_chains: int = 1


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


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / (b.abs().max() + 1e-8)).item()


def _make_case(sc: Scenario, device: torch.device) -> dict:
    torch.manual_seed(0)
    n_chain = sc.n_poly_chains + sc.n_lig_chains
    n_token = n_chain * sc.tokens_per_chain
    asym_id = torch.repeat_interleave(torch.arange(n_chain), sc.tokens_per_chain).to(device)
    has_frame = (asym_id < sc.n_poly_chains).long()
    atom_to_token_idx = torch.repeat_interleave(torch.arange(n_token), sc.atoms_per_token).to(device)
    n_atom = n_token * sc.atoms_per_token
    is_ligand = (asym_id[atom_to_token_idx] >= sc.n_poly_chains).long()

    S = sc.n_sample
    return {
        "distogram_logits": torch.randn(n_token, n_token, 64, device=device),
        "plddt_logits": torch.randn(S, n_atom, 50, device=device),
        "pae_logits": torch.randn(S, n_token, n_token, 64, device=device),
        "pde_logits": torch.randn(S, n_token, n_token, 64, device=device),
        "coordinate": torch.randn(S, n_atom, 3, device=device),
        "asym_id": asym_id,
        "has_frame": has_frame,
        "atom_to_token_idx": atom_to_token_idx,
        "is_ligand": is_ligand,
    }


def test_protenix_confidence_summary():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    sc = Scenario()
    c = _make_case(sc, device)
    n_recycle = 3

    contact_probs = oss_compute_contact_prob(c["distogram_logits"], min_bin=2.3125, max_bin=21.6875, no_bins=64)
    oss_summary, oss_full = oss_compute_full_data_and_summary(
        configs=_OSS_CONFIGS,
        pae_logits=c["pae_logits"],
        plddt_logits=c["plddt_logits"],
        pde_logits=c["pde_logits"],
        contact_probs=contact_probs,
        token_asym_id=c["asym_id"],
        token_has_frame=c["has_frame"],
        atom_coordinate=c["coordinate"],
        atom_to_token_idx=c["atom_to_token_idx"],
        atom_is_polymer=1 - c["is_ligand"],
        N_recycle=n_recycle,
        return_full_data=True,
    )

    bridge = ProtenixConfidenceSummary(ConfidenceSummaryConfig()).to(device)
    result = bridge(
        distogram_logits=c["distogram_logits"],
        plddt_logits=c["plddt_logits"],
        pae_logits=c["pae_logits"],
        pde_logits=c["pde_logits"],
        coordinate=c["coordinate"],
        asym_id=c["asym_id"],
        has_frame=c["has_frame"],
        atom_to_token_idx=c["atom_to_token_idx"],
        is_polymer=1 - c["is_ligand"],
        num_recycles=n_recycle,
    )

    assert len(result["summary_confidence"]) == sc.n_sample
    assert len(result["full_data"]) == sc.n_sample

    def _cmp(bucket: str, oss_list: list, tol: float = 2e-4):
        for i in range(sc.n_sample):
            oss_i, act_i = oss_list[i], result[bucket][i]
            assert set(act_i) == set(oss_i), (bucket, set(act_i) ^ set(oss_i))
            for key, exp in oss_i.items():
                act = act_i[key]
                if not torch.is_tensor(exp) or not exp.is_floating_point():
                    assert torch.equal(act.long(), exp.long()), (bucket, key)
                    continue
                r = _rel_err(act, exp)
                assert r < tol, f"{bucket}[{i}].{key} rel_err={r:.2e} (tol {tol:.0e})"

    _cmp("summary_confidence", oss_summary)
    _cmp("full_data", oss_full)

    summary_only = bridge(
        distogram_logits=c["distogram_logits"],
        plddt_logits=c["plddt_logits"],
        pae_logits=c["pae_logits"],
        pde_logits=c["pde_logits"],
        coordinate=c["coordinate"],
        asym_id=c["asym_id"],
        has_frame=c["has_frame"],
        atom_to_token_idx=c["atom_to_token_idx"],
        is_polymer=1 - c["is_ligand"],
        num_recycles=n_recycle,
        return_full_data=False,
    )
    assert "full_data" not in summary_only
    for compact, full in zip(summary_only["summary_confidence"], result["summary_confidence"], strict=True):
        assert compact.keys() == full.keys()
        for key in compact:
            torch.testing.assert_close(compact[key], full[key])
