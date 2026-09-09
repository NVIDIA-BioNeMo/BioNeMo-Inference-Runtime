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

import logging
from pathlib import Path

import numpy as np
import pytest

from bionemo_ir.pipeline.models.openfold2 import const as rc
from bionemo_ir.pipeline.models.openfold2 import template_logic
from bionemo_ir.pipeline.models.openfold2.template_logic import (
    ChainTemplateData,
    align_query_to_template_chain,
    build_template_feats,
    extract_template_chains,
    select_template_for_cif,
    stable_top_k,
)

_FIXTURE = Path(__file__).with_name("data") / "minimal_template.cif"


@pytest.fixture
def cif_content() -> str:
    return _FIXTURE.read_text()


def test_alignment_keeps_only_mutually_aligned_positions(monkeypatch):
    monkeypatch.setattr(template_logic, "run_kalign", lambda sequences: ["AC-D", "A-CD"])
    chain = ChainTemplateData(
        block_name="alignment",
        chain_id="A",
        canonical_seq="ACD",
        res_name_by_pos={},
        coords_by_pos={},
    )

    idx_map, sequence_identity, coverage = align_query_to_template_chain("ACD", chain)

    np.testing.assert_array_equal(idx_map, np.asarray([[1, 1], [3, 3]], dtype=np.int64))
    assert sequence_identity == pytest.approx(2 / 3)
    assert coverage == pytest.approx(2 / 3)


def test_extract_uses_author_chain_ids_and_corrects_atoms(cif_content):
    chains = extract_template_chains(cif_content)

    assert list(chains) == ["A", "B"]
    assert "L1" not in chains
    assert chains["A"].block_name == "MINI_TEMPLATE"
    assert chains["A"].canonical_seq == "MR"
    assert chains["B"].canonical_seq == "AA"

    mse_atoms = chains["A"].coords_by_pos[1]
    assert "SE" not in mse_atoms
    assert "SD" in mse_atoms

    arg_atoms = chains["A"].coords_by_pos[2]
    nh1_distance = np.linalg.norm(arg_atoms["NH1"] - arg_atoms["CD"])
    nh2_distance = np.linalg.norm(arg_atoms["NH2"] - arg_atoms["CD"])
    assert nh1_distance < nh2_distance

    all_coordinates = np.stack(
        [coordinate for atoms in chains["A"].coords_by_pos.values() for coordinate in atoms.values()]
    )
    np.testing.assert_allclose(all_coordinates.mean(axis=0), np.zeros(3), atol=1e-6, rtol=0)


def test_coordinate_centering_uses_canonical_atom37_order():
    coordinates = {
        1: {
            "C": np.asarray([-1e8, 0.0, 0.0], dtype=np.float32),
            "N": np.asarray([1e8, 0.0, 0.0], dtype=np.float32),
            "CA": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        }
    }

    centered = template_logic._center_atom_coordinates(coordinates)
    canonical_center = np.stack([coordinates[1]["N"], coordinates[1]["CA"], coordinates[1]["C"]]).mean(axis=0)

    for atom_name in coordinates[1]:
        np.testing.assert_array_equal(
            centered[1][atom_name],
            coordinates[1][atom_name] - canonical_center,
        )


def test_explicit_author_chain_has_no_score_rejection(cif_content):
    selected = select_template_for_cif("VV", cif_content, specified_chain_id="B")

    assert selected.chain_id == "B"
    assert selected.score == 0.0
    assert selected.idx_map.shape == (2, 2)


def test_automatic_selection_picks_best_chain(cif_content):
    selected = select_template_for_cif("MR", cif_content)

    assert selected.chain_id == "A"
    assert selected.sequence_identity == 1.0
    assert selected.query_coverage == 1.0
    assert selected.score == 1.0


def test_automatic_selection_warns_but_keeps_weak_match(cif_content, caplog):
    with caplog.at_level(logging.WARNING, logger=template_logic.__name__):
        selected = select_template_for_cif("VV", cif_content)

    assert selected.chain_id == "A"
    assert selected.score == 0.0
    assert "Automatically selected weak template match" in caplog.text


def test_missing_explicit_author_chain_is_invalid(cif_content):
    with pytest.raises(ValueError, match="author chain 'Z'.*available chains"):
        select_template_for_cif("MR", cif_content, specified_chain_id="Z")


def test_non_finite_coordinates_are_rejected(cif_content):
    malformed = cif_content.replace(
        "ATOM   1  N  N   . MSE L1 1 1 ? 0.0  0.0 0.0",
        "ATOM   1  N  N   . MSE L1 1 1 ? nan  0.0 0.0",
    )

    with pytest.raises(ValueError, match="non-finite coordinate.*atom 'N'"):
        build_template_feats("MR", [{"content": malformed, "chain_id": "A"}])


def test_fewer_than_five_aligned_atoms_are_rejected(cif_content):
    sparse = "\n".join(
        line
        for line in cif_content.splitlines()
        if not ((line.startswith(("ATOM", "HETATM"))) and " L1 " in line and not line.startswith("ATOM   1  "))
    )

    with pytest.raises(ValueError, match="fewer than 5 aligned atom37"):
        build_template_feats("MR", [{"content": sparse, "chain_id": "A"}])


def test_implausible_adjacent_c_alpha_distance_is_rejected(cif_content):
    malformed = cif_content.replace(
        "ATOM   8  C  CA  . ARG L1 1 2 ? 5.0  0.0 0.0",
        "ATOM   8  C  CA  . ARG L1 1 2 ? 1000.0  0.0 0.0",
    )

    with pytest.raises(ValueError, match="adjacent C-alpha distance.*150"):
        build_template_feats("MR", [{"content": malformed, "chain_id": "A"}])


def test_stable_top_k_sorts_scores_and_preserves_ties():
    items = [("low", 0.1), ("first-high", 0.8), ("second-high", 0.8), ("middle", 0.5)]

    result = stable_top_k(items, 3, key=lambda item: item[1])

    assert result == [items[1], items[2], items[3]]


def test_build_template_feats_packs_atom37_and_openfold_metadata(cif_content):
    features = build_template_feats("MR", [{"content": cif_content, "format": "cif", "chain_id": "A"}])

    assert features["template_aatype"].shape == (1, 2, 22)
    assert features["template_all_atom_positions"].shape == (1, 2, 37, 3)
    assert features["template_all_atom_mask"].shape == (1, 2, 37)
    assert features["template_sum_probs"].shape == (1, 1)
    assert features["template_aatype"].dtype == np.int64
    assert all(
        features[key].dtype == np.float32
        for key in {"template_all_atom_positions", "template_all_atom_mask", "template_sum_probs"}
    )

    np.testing.assert_array_equal(
        features["template_aatype"].argmax(axis=-1), np.asarray([[rc.HHBLITS_AA_TO_ID["M"], rc.HHBLITS_AA_TO_ID["R"]]])
    )
    np.testing.assert_array_equal(features["template_aatype"].sum(axis=-1), np.ones((1, 2), dtype=np.int64))
    np.testing.assert_array_equal(features["template_sum_probs"], np.ones((1, 1), dtype=np.float32))
    np.testing.assert_array_equal(features["template_domain_names"], np.asarray([b"mini_template_A"], dtype=object))
    np.testing.assert_array_equal(features["template_sequence"], np.asarray([b"MR"], dtype=object))

    mask = features["template_all_atom_mask"].astype(bool)
    positions = features["template_all_atom_positions"]
    assert mask[0, 0, rc.atom_order["SD"]]
    np.testing.assert_array_equal(positions[~mask], np.zeros_like(positions[~mask]))
    np.testing.assert_allclose(positions[mask].mean(axis=0), np.zeros(3), atol=1e-6, rtol=0)

    arg = positions[0, 1]
    nh1_distance = np.linalg.norm(arg[rc.atom_order["NH1"]] - arg[rc.atom_order["CD"]])
    nh2_distance = np.linalg.norm(arg[rc.atom_order["NH2"]] - arg[rc.atom_order["CD"]])
    assert nh1_distance < nh2_distance


def test_build_template_feats_accepts_materialized_content(cif_content):

    class MaterializedTemplate:
        content = cif_content
        format = "cif"
        chain_id = "A"

        def get_content(self):
            raise AssertionError("materialized templates must not perform I/O")

    features = build_template_feats("MR", [MaterializedTemplate()])

    assert features["template_aatype"].shape == (1, 2, 22)


def test_build_template_feats_rejects_get_content_only_input(cif_content):

    class LazyTemplate:
        def get_content(self):
            return cif_content

    with pytest.raises(ValueError, match="contain.*CIF content"):
        build_template_feats("MR", [LazyTemplate()])


def test_build_template_feats_rejects_path_only_input():
    with pytest.raises(ValueError, match="contain.*CIF content"):
        build_template_feats("MR", [{"path": _FIXTURE, "chain_id": "A"}])


def test_build_template_feats_emits_hhblits_gap_class(cif_content, monkeypatch):
    monkeypatch.setattr(template_logic, "run_kalign", lambda sequences: ["MQR", "M-R"])

    features = build_template_feats("MQR", [{"content": cif_content, "chain_id": "A"}])

    np.testing.assert_array_equal(
        features["template_aatype"].argmax(axis=-1),
        np.asarray([[rc.HHBLITS_AA_TO_ID["M"], rc.HHBLITS_AA_TO_ID["-"], rc.HHBLITS_AA_TO_ID["R"]]]),
    )
    np.testing.assert_array_equal(features["template_sequence"], np.asarray([b"M-R"], dtype=object))


def test_build_template_feats_ranks_by_score_before_top_k(cif_content):
    features = build_template_feats(
        "MR",
        [
            {"content": cif_content, "chain_id": "B"},
            {"content": cif_content, "chain_id": "A"},
        ],
        max_templates=2,
    )

    np.testing.assert_array_equal(
        features["template_domain_names"], np.asarray([b"mini_template_A", b"mini_template_B"], dtype=object)
    )
    np.testing.assert_array_equal(features["template_sum_probs"], np.ones((2, 1), dtype=np.float32))


def test_build_template_feats_preserves_input_order_for_score_ties(cif_content):
    contents = [cif_content.replace("data_MINI_TEMPLATE", f"data_TIE_{index}") for index in range(3)]
    templates = [{"content": content, "chain_id": "A"} for content in contents]

    features = build_template_feats("MR", templates, max_templates=2)

    np.testing.assert_array_equal(features["template_domain_names"], np.asarray([b"tie_0_A", b"tie_1_A"], dtype=object))


def test_empty_templates_match_existing_no_template_shapes():
    features = build_template_feats("MR", [])

    assert features["template_aatype"].shape == (0, 2, 22)
    assert features["template_all_atom_positions"].shape == (0, 2, 37, 3)
    assert features["template_all_atom_mask"].shape == (0, 2, 37)
    assert features["template_sum_probs"].shape == (0, 1)
    np.testing.assert_array_equal(features["template_domain_names"], np.asarray([b""], dtype=object))
    np.testing.assert_array_equal(features["template_sequence"], np.asarray([b""], dtype=object))


def test_invalid_supplied_template_reports_its_input_index(cif_content):
    with pytest.raises(ValueError, match="Invalid supplied template at index 0"):
        build_template_feats("MR", [{"content": cif_content, "format": "pdb"}])


def test_build_template_feats_warns_on_excess_input_templates(cif_content, caplog):
    # Far more templates than can be kept: warn for visibility (no truncation),
    # still featurize the top ``max_templates``.
    templates = [{"content": cif_content, "chain_id": "A"}] * 5
    with caplog.at_level(logging.WARNING, logger=template_logic.__name__):
        features = build_template_feats("MR", templates, max_templates=1)

    assert features["template_aatype"].shape[0] == 1
    assert "max_templates=1" in caplog.text
