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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact confidence preserves the pipeline output contract."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from bionemo_ir._torch.modules.protenix import summary as summary_module
from bionemo_ir._torch.modules.protenix.confidence import ProtenixConfidenceHead
from bionemo_ir._torch.modules.protenix.summary import ProtenixConfidenceSummary
from bionemo_ir._torch.utils import ChunkPolicy
from bionemo_ir.configs import PairformerConfig
from bionemo_ir.models.protenix.config import ConfidenceHeadConfig, ConfidenceSummaryConfig
from bionemo_ir.models.protenix.modeling import Protenix
from bionemo_ir.pipeline.models.protenix.postprocessor import PostProcessor
from bionemo_ir.pipeline.utils.atom import encode_atom_name_chars_one_hot
from tests._torch.test_compact_confidence import initialize_heads

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("case", ["monomer", "multimer", "ligand", "no-frame"])
@pytest.mark.parametrize("return_full_data", [False, True])
@torch.inference_mode()
def test_compact_pipeline_contract(
    chunked: bool, case: str, return_full_data: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(47)
    config = ConfidenceHeadConfig(
        c_s=8,
        c_z=8,
        c_s_inputs=8,
        max_atoms_per_token=2,
        graph_optimization_config=None,
        pairformer_config=PairformerConfig(
            token_s=8,
            token_z=8,
            num_blocks=1,
            num_heads=2,
            pairwise_head_width=8,
            pairwise_num_heads=1,
            enable_token_pad=False,
            triangle_attention_backend="VANILLA",
            pairwise_attention_backend="SDPA",
            version="v1",
        ),
    )
    head = initialize_heads(ProtenixConfidenceHead(config), "cuda", torch.float32)
    head.plddt_weight.normal_(std=0.1)
    head.resolved_weight.normal_(std=0.1)
    head.pair_logits_chunk_policy = ChunkPolicy(chunk_size=3, min_size=1) if chunked else None
    summary = ProtenixConfidenceSummary(ConfidenceSummaryConfig())
    model = SimpleNamespace(confidence_head=head, confidence_summary=summary)
    n, samples = 7, 5
    asym = torch.zeros(n, dtype=torch.long, device="cuda") if case == "monomer" else torch.arange(n, device="cuda") % 2
    ligand = (asym == 1).long() if case == "ligand" else torch.zeros_like(asym)
    features = {
        "asym_id": asym[None],
        "atom_to_token_idx": torch.arange(n, device="cuda")[None],
        "atom_to_tokatom_idx": torch.zeros(1, n, dtype=torch.long, device="cuda"),
        "distogram_rep_atom_mask": torch.ones(1, n, device="cuda"),
        "has_frame": (torch.zeros_like(asym) if case == "no-frame" else 1 - ligand)[None],
        "is_ligand": ligand[None],
    }
    held = {
        "s_inputs": torch.randn(1, n, 8, device="cuda"),
        "s": torch.randn(1, n, 8, device="cuda"),
        "z": torch.randn(1, n, n, 8, device="cuda"),
        "distogram_logits": torch.randn(1, n, n, 64, device="cuda"),
    }
    coordinates = torch.randn(1, samples, n, 3, device="cuda") * 5
    coordinates[:, 0].zero_()  # Include a clashing candidate.
    raw = {}
    Protenix._streaming_confidence(model, dict(features), raw, dict(held), coordinates, 3, False, return_full_data)

    unused = Mock(side_effect=AssertionError("Unused confidence computation ran"))
    for module in (head.pde_ln, head.linear_no_bias_pde, head.resolved_ln):
        monkeypatch.setattr(module, "forward", unused)
    monkeypatch.setattr(summary, "contact_probs", unused)
    for name in ("calculate_chain_based_gpde", "calculate_chain_based_ptm", "calculate_chain_based_plddt"):
        monkeypatch.setattr(summary_module, name, unused)
    compact = {}
    compact_held = {key: value for key, value in held.items() if key != "distogram_logits"}
    Protenix._streaming_confidence(model, dict(features), compact, compact_held, coordinates, 3, True, return_full_data)
    assert compact_held == {}
    assert len(compact["summary_confidence"]) == samples
    assert ("full_data" in compact) == return_full_data
    for index, scores in enumerate(compact["summary_confidence"]):
        assert scores.keys() == {"ptm", "iptm", "has_clash", "ranking_score"}
        for key, value in scores.items():
            assert torch.equal(value, raw["summary_confidence"][index][key])
    unused.assert_not_called()
    if not return_full_data:
        return
    for index, data in enumerate(compact["full_data"]):
        assert data.keys() == {"atom_plddt", "token_pair_pae"}
        for key, value in data.items():
            assert torch.equal(value, raw["full_data"][index][key])

    batch = {
        **features,
        "atom_mask": torch.ones(1, n),
        "token_mask": torch.ones(1, n),
        "ref_atom_name_chars": encode_atom_name_chars_one_hot(["CA"] * n)[None],
        "ref_element": torch.nn.functional.one_hot(torch.full((1, n), 5), 128),
        "restype": torch.nn.functional.one_hot(torch.zeros(1, n, dtype=torch.long), 32),
        "residue_index": torch.arange(n)[None],
        "token_resnames": ["ALA"] * n,
        "token_mol_types": [0] * n,
    }
    expected, actual = (PostProcessor()(batch, output) for output in (raw, compact))
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        if isinstance(value, np.ndarray):
            assert value.dtype == expected[key].dtype
            assert value.shape == expected[key].shape
            assert value.tobytes() == expected[key].tobytes()
        else:
            assert value == expected[key]
