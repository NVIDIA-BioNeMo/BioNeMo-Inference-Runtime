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

from pathlib import Path

import numpy as np
import pytest
import torch

from bionemo_ir.pipeline.models.openfold3.const import MOL_TYPE_PROTEIN
from bionemo_ir.pipeline.models.openfold3.feature_generators import TemplateFeatureGenerator
from bionemo_ir.pipeline.models.openfold3.template_logic import extract_template_chains, select_template_for_cif

_FIXTURE = Path(__file__).parents[1] / "openfold2/data/minimal_template.cif"


@pytest.fixture
def pdb_content() -> str:
    return _FIXTURE.with_suffix(".pdb").read_text()


@pytest.mark.parametrize("fmt", ["pdb", "PDB"])
def test_pdb_cif_chains(pdb_content: str, fmt: str) -> None:
    actual = extract_template_chains(pdb_content, fmt)
    expected = extract_template_chains(_FIXTURE.read_text())

    assert list(actual) == ["A", "B"]
    for author, label in [("A", "L1"), ("B", "L2")]:
        assert actual[author].canonical_seq == expected[label].canonical_seq
        assert actual[author].res_name_by_pos == expected[label].res_name_by_pos
        assert actual[author].coords_by_pos.keys() == expected[label].coords_by_pos.keys()
        for position, atoms in expected[label].coords_by_pos.items():
            assert actual[author].coords_by_pos[position].keys() == atoms.keys()
            for name, coordinate in atoms.items():
                np.testing.assert_array_equal(actual[author].coords_by_pos[position][name], coordinate)


@pytest.mark.parametrize("author,label,query", [(None, None, "MRGGG"), ("A", "L1", "MRGGG"), ("B", "L2", "AAGGG")])
def test_pdb_cif_features(pdb_content: str, author: str | None, label: str | None, query: str) -> None:
    generator = TemplateFeatureGenerator(config=None, name="t")
    row = {
        "template_query_seq": {"Q": query},
        "structure": {
            "token_chain_ids": ["Q"] * len(query),
            "token_res_ids": list(range(1, len(query) + 1)),
            "token_mol_types": [MOL_TYPE_PROTEIN] * len(query),
        },
    }
    batch = {"token_index": torch.arange(len(query))}
    expected = generator(
        batch,
        {"_row": {**row, "templates_per_chain": {"Q": [{"content": _FIXTURE.read_text(), "chain_id": label}]}}},
    )
    actual = generator(
        batch,
        {
            "_row": {
                **row,
                "templates_per_chain": {"Q": [{"content": pdb_content, "format": "pdb", "chain_id": author}]},
            }
        },
    )

    assert actual["template_pseudo_beta_mask"].sum() == 2
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, equal_nan=True)


def test_pdb_unresolved_seqres(pdb_content: str) -> None:
    content = "SEQRES   1 A    3  MSE GLY ARG\n" + pdb_content.replace("ARG A  11", "ARG A  12")

    chain = extract_template_chains(content, "pdb")["A"]

    assert chain.canonical_seq == "MGR"
    assert chain.res_name_by_pos == {1: "MSE", 2: "GLY", 3: "ARG"}
    assert list(chain.coords_by_pos) == [1, 3]
    selected = select_template_for_cif("MGR", content, "pdb", "A")
    assert selected is not None
    np.testing.assert_array_equal(selected.idx_map, [[1, 1], [2, 2], [3, 3]])


def test_pdb_insertions(pdb_content: str) -> None:
    lines = []
    for line in pdb_content.splitlines():
        if line[21:22] == "A":
            insertion = " " if line[17:20] == "MSE" else "A"
            line = line[:22] + f"{-3:4d}" + insertion + line[27:]
        lines.append(line)

    chain = extract_template_chains("\n".join(lines), "pdb")["A"]

    assert chain.canonical_seq == "MR"
    assert list(chain.coords_by_pos) == [1, 2]


def test_pdb_altloc_occupancy(pdb_content: str) -> None:
    lines = []
    for line in pdb_content.splitlines():
        if line[12:16].strip() == "CA" and line[17:20] == "MSE":
            lines.append(line[:16] + "A" + line[17:30] + f"{99.0:8.3f}" + line[38:54] + "  0.25" + line[60:])
            line = line[:16] + "B" + line[17:54] + "  0.75" + line[60:]
        lines.append(line)

    actual = extract_template_chains("\n".join(lines), "pdb")["A"]
    expected = extract_template_chains(pdb_content, "pdb")["A"]

    for position, atoms in expected.coords_by_pos.items():
        for name, coordinate in atoms.items():
            np.testing.assert_array_equal(actual.coords_by_pos[position][name], coordinate)


def test_pdb_first_model(pdb_content: str) -> None:
    content = f"MODEL        1\n{pdb_content}ENDMDL\nMODEL        2\n{pdb_content.replace('ARG', 'GLY')}ENDMDL\n"

    assert extract_template_chains(content, "pdb")["A"].canonical_seq == "MR"


@pytest.mark.parametrize("content", ["", "not a PDB", "END\n"])
def test_invalid_pdb(content: str) -> None:
    with pytest.raises(ValueError, match="template PDB|Template PDB"):
        extract_template_chains(content, "pdb")


def test_pdb_without_polymer() -> None:
    content = "HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00 20.00           O  \nEND\n"

    with pytest.raises(ValueError, match="Template PDB contains no polymer sequence"):
        extract_template_chains(content, "pdb")


def test_pdb_empty_polymer(pdb_content: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from bionemo_ir.pipeline.models.openfold3 import template_logic

    monkeypatch.setattr(template_logic, "_canonical_seq_from_poly_scheme", lambda *_: {})

    with pytest.raises(ValueError, match="Template PDB contains no polymer sequence"):
        extract_template_chains(pdb_content, "pdb")


def test_invalid_format(pdb_content: str) -> None:
    with pytest.raises(ValueError, match="Unsupported template format"):
        extract_template_chains(pdb_content, "bcif")
