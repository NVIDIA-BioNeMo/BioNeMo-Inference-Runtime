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

from __future__ import annotations

import numpy as np
import pytest

from bionemo_ir.data.schemas import (
    AtomIdentity,
    AtomTypes,
    EncodedStructure,
    FoldingOutput,
    InputRequest,
    OracleRequest,
    OracleResponse,
    OracleStatus,
    Polymer,
    PolymerType,
    ResidueIdentity,
    ResTypes,
)
from bionemo_ir.oracles import (
    MappingAtomTypeResolver,
    MappingResidueTypeResolver,
    resolve_structure,
)
from bionemo_ir.oracles.providers import CallableOracleProvider, OracleProvider


def _resolved_request(request_id: str) -> OracleRequest:
    encoded = EncodedStructure(
        coordinates=np.zeros((1, 1, 3), dtype=np.float32),
        atom_mask=np.ones((1, 1), dtype=bool),
        residue_type_indices=np.array([1], dtype=np.int64),
        atom_type_indices=np.array([[1]], dtype=np.int64),
        source_token_indices=np.array([0], dtype=np.int64),
    )
    structure = resolve_structure(
        encoded,
        MappingResidueTypeResolver(
            {
                1: ResidueIdentity(
                    source_index=1,
                    residue_type=ResTypes.A,
                    one_letter_code="A",
                    polymer_type=PolymerType.PROTEIN,
                )
            }
        ),
        MappingAtomTypeResolver({1: AtomIdentity(source_index=1, atom_type=AtomTypes.CA, element="C")}),
    )
    return OracleRequest(request_id=request_id, structure=structure)


def test_resolvers_support_token_and_residue_qualified_indices() -> None:
    encoded = EncodedStructure(
        coordinates=np.zeros((2, 3, 3), dtype=np.float32),
        atom_mask=np.array([[True, True, False], [False, True, True]]),
        residue_type_indices=np.array([0, 0], dtype=np.int64),
        atom_type_indices=np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
        source_token_indices=np.array([4, 7], dtype=np.int64),
        token_type_indices=np.array([0, 1], dtype=np.int64),
        chain_indices=np.array([3, 4], dtype=np.int64),
    )
    resolved = resolve_structure(
        encoded,
        MappingResidueTypeResolver(
            {
                (0, 0): ResidueIdentity(0, ResTypes.A, "A", PolymerType.PROTEIN),
                (1, 0): ResidueIdentity(0, ResTypes.RA, "A", PolymerType.RNA),
            }
        ),
        MappingAtomTypeResolver(
            {
                0: AtomIdentity(0, AtomTypes.N, "N"),
                1: AtomIdentity(1, AtomTypes.CA, "C"),
                ("A", 2): AtomIdentity(2, AtomTypes.N9, "N"),
            }
        ),
    )
    assert resolved.sequence == "AA"
    assert resolved.residues[0].component_id == "ALA"
    assert resolved.residues[1].polymer_type is PolymerType.RNA
    assert resolved.atoms[0][2] is None
    assert resolved.atoms[1][2] == AtomIdentity(2, AtomTypes.N9, "N")


def test_oracle_schemas_reuse_pipeline_input_and_output_types() -> None:
    input_request = InputRequest(
        input_id="first",
        polymers=[Polymer(polymer_type=PolymerType.PROTEIN, chain_id="A", sequence="A")],
    )
    folding_output = FoldingOutput(
        atom_positions=np.zeros((1, 1, 3), dtype=np.float32),
        residue_types=np.zeros(1, dtype=np.int64),
        atom_mask=np.ones((1, 1), dtype=np.float32),
        residue_indices=np.ones(1, dtype=np.int64),
    )
    request = OracleRequest(
        request_id="first",
        structure=_resolved_request("source").structure,
        input_request=input_request,
    )
    response = OracleResponse(
        request_id=request.request_id,
        status=OracleStatus.OK,
        folding_output=folding_output,
    )

    assert request.structure.residues[0].residue_type is ResTypes.A
    assert request.structure.atoms[0][0].atom_type is AtomTypes.CA
    assert request.input_request is input_request
    assert response.folding_output is folding_output


def test_oracle_provider_base_validates_and_reorders_user_responses() -> None:
    requests = [_resolved_request("first"), _resolved_request("second")]

    provider = CallableOracleProvider(
        "custom",
        lambda rows: [
            OracleResponse(rows[1].request_id, OracleStatus.OK, {"custom.score": 2.0}),
            OracleResponse(rows[0].request_id, OracleStatus.OK, {"custom.score": 1.0}),
        ],
    )
    responses = provider.evaluate(requests)
    assert isinstance(provider, OracleProvider)
    assert [response.request_id for response in responses] == ["first", "second"]

    invalid = CallableOracleProvider(
        "invalid",
        lambda rows: [OracleResponse(rows[0].request_id, OracleStatus.OK)],
    )
    with pytest.raises(ValueError, match="missing=.*second"):
        invalid.evaluate(requests)
    with pytest.raises(ValueError, match="name"):
        CallableOracleProvider("", lambda rows: []).evaluate(requests)
