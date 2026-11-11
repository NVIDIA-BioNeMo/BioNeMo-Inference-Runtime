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
"""MSA support omission, graph keys, and the public OpenFold2 row contract."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import CUDAGraphOptimizationTracker
from bionemo_ir.models.openfold2.config import _trunk_graph_config
from bionemo_ir.models.openfold2.modeling import OpenFold2, _compact_msa_features, _template_msa_row_count


def _features(depth: int, *, rows: int = 64, batch: tuple[int, ...] = ()) -> dict[str, torch.Tensor]:
    mask = torch.zeros(*batch, rows, 7)
    mask[..., :depth, :] = 1
    return {
        "msa_feat": torch.randn(*batch, rows, 7, 8),
        "msa_mask": mask,
        "seq_mask": torch.ones(*batch, 7),
    }


@pytest.mark.parametrize("depth,rows", [(1, 16), (16, 16), (17, 32), (31, 32), (53, 64), (64, 64)])
def test_support_buckets_preserve_inputs(depth: int, rows: int) -> None:
    feats = _features(depth)
    original = {key: value.clone() for key, value in feats.items()}
    compact = _compact_msa_features(feats)
    assert compact["msa_feat"].shape[-3] == rows
    torch.testing.assert_close(compact["msa_feat"], original["msa_feat"][..., :rows, :, :])
    for key in feats:
        torch.testing.assert_close(feats[key], original[key])


def test_support_uses_last_row_and_union_across_batch() -> None:
    feats = _features(1, batch=(2,))
    feats["msa_mask"][1, 35, 3] = 1
    compact = _compact_msa_features(feats)
    assert compact["msa_mask"].shape == (2, 48, 7)
    torch.testing.assert_close(compact["msa_mask"], feats["msa_mask"][:, :48])


@pytest.mark.parametrize("depth", [0, 1])
def test_empty_active_column_keeps_finite_softmax_semantics(depth: int) -> None:
    feats = _features(depth)
    feats["msa_mask"][..., 3] = 0
    feats["template_torsion_angles_mask"] = torch.zeros(4, 7, 7)
    assert _compact_msa_features(feats) is feats
    assert _template_msa_row_count(feats, is_multimer=False) is None


def test_unpadded_and_masked_residues() -> None:
    feats = _features(17, rows=23)
    feats["seq_mask"][-2:] = 0
    feats["msa_mask"][..., -2:] = 0
    assert _compact_msa_features(feats) is feats
    feats = _features(1, rows=23)
    assert _compact_msa_features(feats)["msa_feat"].shape[-3] == 16


def test_graph_key_reuses_bucket_and_separates_depths() -> None:
    tracker = CUDAGraphOptimizationTracker(_trunk_graph_config())
    keys = [tracker.input_key_for_this_call(feats=_compact_msa_features(_features(depth))) for depth in (17, 31, 33)]
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]


@pytest.mark.parametrize("template_rows", [0, 1, 4])
def test_iteration_restores_cluster_rows_without_template_leakage(
    monkeypatch: pytest.MonkeyPatch, template_rows: int
) -> None:
    feats = _features(17)
    feats["aatype"] = torch.zeros(7, dtype=torch.long)
    feats["atom37_atom_exists"] = torch.ones(7, 37)
    feats["template_torsion_angles_mask"] = torch.zeros(4, 7, 7)
    feats["template_torsion_angles_mask"][:template_rows, :, 2] = 1
    model = OpenFold2.__new__(OpenFold2)
    nn.Module.__init__(model)
    model.is_multimer = False
    model.config = SimpleNamespace(enable_template=True)
    feats["template_mask"] = torch.ones(4)
    seen = []

    def trunk(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = kwargs["feats"]
        assert kwargs["template_mask_any"] is True
        seen.append(batch)
        # Distinct template activations must never enter the public MSA.
        m = torch.cat((batch["msa_feat"], torch.full((kwargs["template_msa_rows"], 7, 8), 99.0)), dim=-3)
        return m, torch.zeros(7, 7, 4), torch.zeros(7, 4)

    model.trunk_graph = trunk
    model.structure_module = Mock(return_value={"positions": [torch.zeros(7, 14, 3)], "frames": [torch.zeros(7, 4, 4)]})
    monkeypatch.setattr("bionemo_ir.models.openfold2.modeling.atom14_to_atom37", lambda pos, _: pos)
    prevs = [torch.zeros(7, 8), torch.zeros(7, 7, 4), torch.zeros(7, 37, 3)]
    out, m_prev, _, _, _ = model.iteration(feats, prevs)
    assert seen[0]["msa_feat"].shape[-3] == 32
    assert out["msa"].shape == feats["msa_feat"].shape
    torch.testing.assert_close(out["msa"][:32], feats["msa_feat"][:32])
    assert torch.count_nonzero(out["msa"][32:]) == 0
    torch.testing.assert_close(m_prev, feats["msa_feat"][0])


@pytest.mark.parametrize("is_multimer", [False, True])
def test_template_prefix_preserves_holes_and_batch_union(is_multimer: bool) -> None:
    feats = {
        "template_all_atom_mask": torch.zeros(2, 4, 7, 37),
        "template_torsion_angles_mask": torch.zeros(2, 4, 7, 7),
    }
    assert _template_msa_row_count(feats, is_multimer=is_multimer) == 0
    feats["template_all_atom_mask"][1, 2, 3, 5] = 1
    feats["template_torsion_angles_mask"][1, 2, 3, 2] = 1
    assert _template_msa_row_count(feats, is_multimer=is_multimer) == 3
    assert _template_msa_row_count({}, is_multimer=is_multimer) is None


def test_template_keys_do_not_enter_the_length_mask_prefix() -> None:
    feats = _features(16)
    compact = _compact_msa_features(feats, template_msa_rows=1)
    # The optimized column path selects 17 prefix keys on main: 16 real
    # clusters and a padded cluster. Preserve that selection after moving
    # the template from row 64 to the compact cluster boundary.
    assert compact["msa_feat"].shape[-3] == 32
    torch.testing.assert_close(compact["msa_feat"][:17], feats["msa_feat"][:17])


@pytest.mark.parametrize("opm_first", [False, True])
def test_compaction_matches_fp32_evoformer_with_templates(opm_first: bool) -> None:
    from test_utils.openfold.ref_layers import RefEvoformerBlock

    torch.manual_seed(23)
    feats = _features(17, batch=(2,))
    # Unequal batch depths, internal holes, and a partially supported row.
    feats["msa_mask"][0, 8] = 0
    feats["msa_mask"][1, 5:16] = 0
    feats["msa_mask"][1, 16, 3:] = 0
    compact = _compact_msa_features(feats)
    template_m = torch.randn(2, 4, 7, 8)
    template_mask = torch.zeros(2, 4, 7)
    template_mask[:, 0] = 1
    original_m = torch.cat((feats["msa_feat"], template_m), dim=-3)
    original_mask = torch.cat((feats["msa_mask"], template_mask), dim=-2)
    compact_m = torch.cat((compact["msa_feat"], template_m[:, :1]), dim=-3)
    compact_mask = torch.cat((compact["msa_mask"], template_mask[:, :1]), dim=-2)
    z = torch.randn(2, 7, 7, 8)
    block = RefEvoformerBlock(
        c_m=8,
        c_z=8,
        c_hidden_msa_att=4,
        c_hidden_opm=4,
        c_hidden_mul=8,
        c_hidden_pair_att=4,
        no_heads_msa=2,
        no_heads_pair=2,
        transition_n=2,
        no_column_attention=False,
        opm_first=opm_first,
    ).eval()
    full_z, compact_z = z.clone(), z.clone()
    with torch.no_grad():
        for _ in range(3):
            original_m, full_z = block(original_m, full_z, original_mask, torch.ones(2, 7, 7))
            compact_m, compact_z = block(compact_m, compact_z, compact_mask, torch.ones(2, 7, 7))
    torch.testing.assert_close(compact_z, full_z, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(compact_m[:, :32], original_m[:, :32], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(compact_m[:, -1:], original_m[:, -4:-3], rtol=2e-5, atol=2e-5)
