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

from collections.abc import Mapping, Sequence

import numpy as np
import pytest
import torch

from bionemo_ir._torch.modules.rfproteina.contracts import (
    RFProteinaPredictionOutput,
    RFProteinaSamplingOutput,
)
from bionemo_ir._torch.search import CompositeMetricEvaluator, MetricEvaluationContext
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
from bionemo_ir.oracles.adapters import OracleMetricEvaluator, RFProteinaOracleRequestAdapter
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


class _StubRequestAdapter:
    def batch_size(self, prediction: object) -> int:
        if not isinstance(prediction, int):
            raise TypeError("stub prediction must be an integer batch size")
        return prediction

    def build_requests(
        self,
        prediction: object,
        static_features: Mapping[str, object],
        context: MetricEvaluationContext,
    ) -> Sequence[OracleRequest | OracleResponse]:
        del static_features
        if not isinstance(prediction, int):
            raise TypeError("stub prediction must be an integer batch size")
        return [
            OracleRequest(
                request_id=context.request_ids[index],
                structure=_resolved_request(f"source-{index}").structure,
                tags=context.tags,
            )
            for index in range(prediction)
        ]


def test_oracle_metric_evaluator_gates_tags_and_preserves_failed_rows() -> None:
    calls: list[tuple[OracleRequest, ...]] = []

    def oracle(requests: Sequence[OracleRequest]) -> Sequence[OracleResponse]:
        calls.append(tuple(requests))
        return [
            OracleResponse(requests[0].request_id, OracleStatus.OK, {"oracle.confidence": 0.8}),
            OracleResponse(requests[1].request_id, OracleStatus.FAILED, error="oracle failed"),
        ]

    evaluator = CompositeMetricEvaluator(
        [
            lambda prediction, _features: [{"local": float(index)} for index in range(prediction)],
            OracleMetricEvaluator(
                CallableOracleProvider("custom", oracle),
                _StubRequestAdapter(),
                required_tags_any={"refolding"},
            ),
        ]
    )
    skipped = evaluator.evaluate(
        2,
        {},
        MetricEvaluationContext(checkpoint=2, tags=("local",), request_ids=("a", "b")),
    )
    assert skipped == [{"local": 0.0}, {"local": 1.0}]
    assert not calls

    evaluated = evaluator.evaluate(
        2,
        {},
        MetricEvaluationContext(checkpoint=4, tags=("refolding",), request_ids=("a", "b")),
    )
    assert len(calls) == 1
    assert calls[0][0].tags == ("refolding",)
    assert evaluated == [{"local": 0.0, "oracle.confidence": 0.8}, {"local": 1.0}]


def _rfproteina_prediction() -> RFProteinaPredictionOutput:
    batch, tokens, atom_slots = 1, 3, 37
    state = {
        "bb_ca": torch.zeros(batch, tokens, 3),
        "local_latents": torch.zeros(batch, tokens, 2),
    }
    sampling = RFProteinaSamplingOutput(
        state=state,
        clean_prediction={key: value.clone() for key, value in state.items()},
        insertion_logits=None,
        trajectory=[],
        seed=1,
    )
    coordinates = torch.full((batch, tokens, atom_slots, 3), 0.2)
    atom_mask = torch.zeros(batch, tokens, atom_slots, dtype=torch.bool)
    atom_mask[:, :, 1] = True
    return RFProteinaPredictionOutput(
        sampling=sampling,
        coordinates=coordinates,
        sequence_logits=torch.zeros(batch, tokens, 20),
        residue_types=torch.tensor([[1, 2, 3]]),
        atom_mask=atom_mask,
    )


def test_rfproteina_request_adapter_uses_user_indices_and_converts_units() -> None:
    prediction = _rfproteina_prediction()
    custom_atom_indices = torch.zeros(1, 3, 37, dtype=torch.long)
    custom_atom_indices[:, :, 1] = 99
    element = torch.zeros(1, 3, 37, dtype=torch.long)
    element[:, :, 1] = 6
    features = {
        "padding_mask": torch.tensor([[True, True, False]]),
        "token_type": torch.tensor([[0, 0, 0]]),
        "chain_iid": torch.tensor([[7, 8, 9]]),
        "element": element,
        "custom_atom_indices": custom_atom_indices,
    }
    adapter = RFProteinaOracleRequestAdapter(
        MappingResidueTypeResolver(
            {
                1: ResidueIdentity(1, ResTypes.A, "A", PolymerType.PROTEIN),
                2: ResidueIdentity(2, ResTypes.C, "C", PolymerType.PROTEIN),
            }
        ),
        MappingAtomTypeResolver({99: AtomIdentity(99, AtomTypes.CA, "C")}),
        atom_type_indices_key="custom_atom_indices",
    )
    items = adapter.build_requests(
        prediction,
        features,
        MetricEvaluationContext(checkpoint=4, tags=("oracle",), request_ids=("candidate-0",)),
    )
    assert len(items) == 1
    request = items[0]
    assert isinstance(request, OracleRequest)
    assert request.request_id == "candidate-0"
    assert request.structure.sequence == "AC"
    assert request.structure.encoded.coordinates.shape == (2, 37, 3)
    assert request.structure.encoded.coordinates[0, 1, 0] == pytest.approx(2.0)
    assert request.structure.encoded.source_token_indices.tolist() == [0, 1]
    assert request.structure.encoded.chain_indices.tolist() == [7, 8]
    assert request.structure.encoded.metadata["element"].shape == (3, 37)
    assert request.structure.atoms[0][1] == AtomIdentity(99, AtomTypes.CA, "C")


def test_rfproteina_request_adapter_marks_unknown_indices_unsupported() -> None:
    prediction = _rfproteina_prediction()
    adapter = RFProteinaOracleRequestAdapter(
        MappingResidueTypeResolver({1: ResidueIdentity(1, ResTypes.A, "A", PolymerType.PROTEIN)}),
        MappingAtomTypeResolver({1: AtomIdentity(1, AtomTypes.CA, "C")}),
    )
    items = adapter.build_requests(
        prediction,
        {"padding_mask": torch.tensor([[True, True, False]])},
        MetricEvaluationContext(request_ids=("candidate-0",)),
    )
    assert isinstance(items[0], OracleResponse)
    assert items[0].status is OracleStatus.UNSUPPORTED
    assert "source index 2" in items[0].error
