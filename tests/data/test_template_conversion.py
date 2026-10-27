# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import io
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from biotite.structure.io.pdbx import CIFFile, get_structure

from bionemo_ir.data.utils import pdb_to_cif, pdb_to_cif_file

_FIXTURE = Path(__file__).parents[1] / "pipeline/models/openfold2/data/minimal_template.pdb"


@pytest.mark.parametrize("variant", ["minimal", "metadata", "seqres", ".", "?", "models", "water"])
def test_conversion_parity(variant: str) -> None:
    content = _FIXTURE.read_text()
    if variant == "metadata":
        content = "TITLE     A 'QUOTED' TITLE WITH \"DOUBLE QUOTES\"\n" + content
    elif variant == "seqres":
        content = "SEQRES   1 A    3  MSE GLY ARG\n" + content.replace("ARG A  11", "ARG A  12")
    elif variant in {".", "?"}:
        content = "\n".join(line[:21] + variant + line[22:] for line in content.splitlines())
    elif variant == "models":
        content = f"MODEL        1\n{content}ENDMDL\nMODEL        2\n{content}ENDMDL\n"
    elif variant == "water":
        content = "HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00 20.00           O  \nEND\n"

    expected = CIFFile.read(io.StringIO(pdb_to_cif(content)))
    actual = pdb_to_cif_file(content)

    assert list(actual) == list(expected)
    assert set(actual.block) == set(expected.block)
    for name, category in expected.block.items():
        assert set(actual.block[name]) == set(category)
        for key, column in category.items():
            np.testing.assert_array_equal(actual.block[name][key].as_array(), column.as_array())
            np.testing.assert_array_equal(
                actual.block[name][key].as_array(masked_value="NULL"), column.as_array(masked_value="NULL")
            )
    assert get_structure(actual, model=1, use_author_fields=False, altloc="occupancy") == get_structure(
        expected, model=1, use_author_fields=False, altloc="occupancy"
    )


@pytest.mark.parametrize("convert", [pdb_to_cif, pdb_to_cif_file])
def test_blank_chain_id_is_rejected(convert: Callable[[str], object]) -> None:
    # gemmi assigns no entity to a blank-ID chain, which CASP model files use.
    content = "\n".join(line[:21] + " " + line[22:] for line in _FIXTURE.read_text().splitlines())
    with pytest.raises(ValueError, match="blank chain ID"):
        convert(content)
