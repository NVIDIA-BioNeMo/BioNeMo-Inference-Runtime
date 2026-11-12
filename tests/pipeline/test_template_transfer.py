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
"""Template staging preserves payload bytes and producer absence metadata."""

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bionemo_ir.configs.base import DeviceConfig
from bionemo_ir.pipeline import engine as engine_module
from bionemo_ir.pipeline.engine import FoldingEngine
from bionemo_ir.pipeline.models.openfold3.feature_collators import OpenFold3FinalFeatureCollator
from bionemo_ir.pipeline.models.openfold3.feature_generators import TemplateFeatureGenerator as OpenFold3Templates
from bionemo_ir.pipeline.models.protenix.feature_collators import ProtenixFeatureCollator
from bionemo_ir.pipeline.models.protenix.templates import TemplateFeatureGenerator as ProtenixTemplates
from bionemo_ir.pipeline.models.protenix.templates import template_geometry
from bionemo_ir.pipeline.utils._template_transfer import (
    TEMPLATE_TRANSFER_COLUMN,
    attach_template_slots,
    record_template_slots,
    transfer_template_tensor,
)


def _assert_bytes_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.cpu().contiguous().view(torch.uint8), expected.cpu().contiguous().view(torch.uint8))


def _features(active: tuple[int, ...]) -> torch.Tensor:
    values = torch.zeros(4, 256, 256, 9, dtype=torch.float32)
    for slot in active:
        values[slot].fill_(slot + 0.5)
        values[slot, 0, 0, 0] = -0.0
        values[slot, 0, 0, 1] = float("nan")
    return values


@pytest.mark.parametrize("active", [(), (0, 2), (0, 1, 2, 3)])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_template_transfer_preserves_bytes(active: tuple[int, ...], batched: bool, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    value = _features(active)
    if batched:
        value = value.unsqueeze(0)
    original = value.clone()
    actual = transfer_template_tensor(value, device, (int(batched), active, tuple(value.shape)))
    assert actual.device.type == device
    _assert_bytes_equal(actual, original)
    _assert_bytes_equal(value, original)


@pytest.mark.parametrize("readonly", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_numpy_template_transfer(readonly: bool, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    expected = _features((1, 3))
    value = expected.numpy().copy()
    value.flags.writeable = not readonly
    original = value.tobytes()
    actual = transfer_template_tensor(value, device, (0, (1, 3), value.shape))
    _assert_bytes_equal(actual, expected)
    assert value.tobytes() == original
    assert value.flags.writeable is not readonly


def test_stale_template_shape_is_rejected() -> None:
    value = torch.zeros(4, 3, 3, 9)
    with pytest.raises(ValueError):
        transfer_template_tensor(value, "cpu", (0, (), (4, 4, 4, 9)))


def test_recollation_discards_stale_template_metadata() -> None:
    value = _features(())
    context: dict[str, object] = {}
    record_template_slots(context, "template_distogram", value, ())
    output = {"template_distogram": value}
    attach_template_slots(output, context, add_batch_dim=False)
    assert TEMPLATE_TRANSFER_COLUMN in output
    output["template_distogram"] = value.clone().fill_(1)
    attach_template_slots(output, context, add_batch_dim=False)
    assert TEMPLATE_TRANSFER_COLUMN not in output


@pytest.mark.parametrize("active", [(0, 0), (-1,), (4,), (1, 4)])
def test_invalid_template_slots_are_rejected(active: tuple[int, ...]) -> None:
    value = torch.zeros(4, 3, 3, 9)
    with pytest.raises(ValueError, match="populated template slots"):
        transfer_template_tensor(value, "cpu", (0, active, tuple(value.shape)))


@pytest.mark.parametrize("add_batch_dim", [False, True])
def test_template_metadata_tracks_batch_axis(add_batch_dim: bool) -> None:
    value = _features((0, 2))
    context: dict[str, object] = {}
    record_template_slots(context, "template_distogram", value, (0, 2))
    output = {"template_distogram": value.unsqueeze(0) if add_batch_dim else value}
    attach_template_slots(output, context, add_batch_dim=add_batch_dim)
    axis, active, shape = output[TEMPLATE_TRANSFER_COLUMN]["template_distogram"]
    assert axis == int(add_batch_dim)
    assert tuple(active) == (0, 2)
    assert tuple(shape) == tuple(output["template_distogram"].shape)


@pytest.mark.parametrize("add_batch_dim", [False, True])
def test_openfold3_collator_preserves_template_metadata(add_batch_dim: bool) -> None:
    value = _features(())
    context: dict[str, object] = {}
    record_template_slots(context, "template_distogram", value, ())
    collator = OpenFold3FinalFeatureCollator(add_batch_dim=add_batch_dim)
    output = collator({"template_distogram": value}, context)
    axis, active, shape = output[TEMPLATE_TRANSFER_COLUMN]["template_distogram"]
    assert axis == int(add_batch_dim)
    assert not active
    assert tuple(shape) == tuple(output["template_distogram"].shape)


@pytest.mark.parametrize("case", ["full", "replaced"])
def test_ineligible_template_metadata_is_omitted(case: str) -> None:
    value = _features(())
    active = (0, 1, 2, 3) if case == "full" else ()
    context: dict[str, object] = {}
    record_template_slots(context, "template_distogram", value, active)
    output = {"template_distogram": value.clone() if case == "replaced" else value}
    attach_template_slots(output, context, add_batch_dim=False)
    assert TEMPLATE_TRANSFER_COLUMN not in output


@pytest.mark.parametrize("model", ["openfold3", "protenix"])
def test_absent_template_producer_survives_collation_and_serialization(model: str) -> None:
    n_tokens = 256
    if model == "openfold3":
        context = {"_row": {"templates_per_chain": {}}}
        features = OpenFold3Templates()({"token_index": torch.arange(n_tokens)}, context)
        output = OpenFold3FinalFeatureCollator()(features, context)
    else:
        from bionemo_ir.models.protenix.config import ProtenixConfig

        atoms = SimpleNamespace(
            res_name=np.full(n_tokens, "ALA"),
            mol_type=np.zeros(n_tokens, dtype=int),
            hetero=np.zeros(n_tokens, dtype=bool),
        )
        structure = {
            "atoms": atoms,
            "tokens": [{"center": index} for index in range(n_tokens)],
            "chains": [{"chain_id": "A"}],
        }
        context = {"_row": {"structure": structure}}
        features = ProtenixTemplates(config=ProtenixConfig(use_templates=False))({}, context)
        output = ProtenixFeatureCollator()(features, context)

    # Ray can serialize the batch between its producer and the engine worker.
    restored = pickle.loads(pickle.dumps(output))
    specs = restored[TEMPLATE_TRANSFER_COLUMN]
    assert "template_distogram" in specs
    assert all(not spec[1] for spec in specs.values())
    assert not {"template_aatype", "template_restype"}.intersection(specs)
    for name, expected in output.items():
        if isinstance(expected, torch.Tensor):
            value = restored[name]
            actual = transfer_template_tensor(value, "cpu", specs[name]) if name in specs else torch.as_tensor(value)
            _assert_bytes_equal(actual, expected)


_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("annotated", [False, True])
def test_engine_removes_template_metadata_and_preserves_legacy_batches(
    annotated: bool, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def spy(*args: object) -> torch.Tensor:
        calls.append(args[2])
        return transfer_template_tensor(*args)

    monkeypatch.setattr(engine_module, "transfer_template_tensor", spy)
    engine = object.__new__(FoldingEngine)
    engine.device_config = DeviceConfig(device=device)
    value = _features((1,))
    batch: dict[str, object] = {"template_distogram": value, "name": "sample", "indices": np.arange(3)}
    if annotated:
        batch[TEMPLATE_TRANSFER_COLUMN] = {"template_distogram": (0, (1,), tuple(value.shape))}
    result = engine.transfer_batch_to_device(batch)
    assert TEMPLATE_TRANSFER_COLUMN not in result
    assert result["name"] == "sample"
    assert torch.equal(result["indices"].cpu(), torch.arange(3))
    assert result["template_distogram"].device.type == device
    _assert_bytes_equal(result["template_distogram"], value)
    assert (TEMPLATE_TRANSFER_COLUMN in batch) == annotated
    # Only a CUDA destination takes the slot path.
    assert len(calls) == int(annotated and device == "cuda")


@pytest.mark.parametrize("device", _DEVICES)
def test_engine_copies_densely_when_provenance_mismatches(device: str) -> None:
    engine = object.__new__(FoldingEngine)
    engine.device_config = DeviceConfig(device=device)
    value = _features((1,))
    batch = {"template_distogram": value, TEMPLATE_TRANSFER_COLUMN: {"template_distogram": (0, (1,), (4, 4, 4, 9))}}
    result = engine.transfer_batch_to_device(batch)
    assert result["template_distogram"].device.type == device
    _assert_bytes_equal(result["template_distogram"], value)


_TEMPLATE_FIXTURES = Path(__file__).resolve().parents[2] / "examples" / "data" / "samples" / "templates"


def _openfold3_template_output(target: str) -> tuple[dict[str, object], dict[str, object]]:
    """OpenFold3 template features and their collated output for a template fixture, without its MSA."""
    from bionemo_ir.data.schemas.basic import InputParsed, PolymerParsed, PolymerType, TemplateParsed
    from bionemo_ir.pipeline.models.openfold3.feature_context import OpenFold3ContextGenerator
    from bionemo_ir.pipeline.models.openfold3.feature_generators import StructureFeatureGenerator

    doc = json.loads((_TEMPLATE_FIXTURES / f"{target}.json").read_text())[0]
    polymers = []
    for polymer in doc["polymers"]:
        templates = []
        for entry in polymer.get("templates") or []:
            content = (_TEMPLATE_FIXTURES / entry["path"]).read_text()
            if content.startswith("version https://git-lfs"):
                pytest.skip("template fixtures are Git LFS pointers")
            templates.append(
                TemplateParsed(content=content, format=entry.get("format", "cif"), chain_id=entry.get("chain_id"))
            )
        chain_id = polymer["chain_id"]
        polymers.append(
            PolymerParsed(
                polymer_type=PolymerType.PROTEIN,
                chain_id=chain_id[0] if isinstance(chain_id, list) else chain_id,
                sequence=polymer.get("sequence"),
                msas=None,
                paired_msas=None,
                templates=templates or None,
            )
        )
    context = {"_row": OpenFold3ContextGenerator(config=None)(InputParsed(input_id=target, polymers=polymers))}
    tokens = StructureFeatureGenerator(config=None, name="s")({}, context)["token_index"]
    features = OpenFold3Templates()({"token_index": tokens}, context)
    return features, OpenFold3FinalFeatureCollator()(features, context)


# 7tpu: 616 tokens, the native distogram path; 7qsj: 375 tokens, padded slots. Both populate one of four slots.
@pytest.mark.parametrize("target", ["7tpu", "7qsj"])
@pytest.mark.parametrize("device", _DEVICES)
def test_openfold3_template_hits_transfer_byte_exact(target: str, device: str) -> None:
    _, output = _openfold3_template_output(target)
    specs = output[TEMPLATE_TRANSFER_COLUMN]
    assert {"template_distogram", "template_unit_vector"} <= set(specs)
    for name, (axis, active, shape) in specs.items():
        assert 0 < len(active) < shape[axis]
        _assert_bytes_equal(transfer_template_tensor(output[name], device, specs[name]), output[name])


@pytest.mark.parametrize("device", _DEVICES)
def test_protenix_template_geometry_keeps_absent_slots_zero(device: str) -> None:
    """Protenix certifies the slots with any resolved atom; its pair features hold +0 in every other slot."""
    generator = np.random.default_rng(775)
    count, n_tokens = 4, 40
    restypes = np.full((count, n_tokens), 31, dtype=np.int64)
    restypes[0] = generator.integers(0, 20, n_tokens)
    positions = np.zeros((count, n_tokens, 24, 3), dtype=np.float32)
    positions[0] = generator.normal(scale=8.0, size=(n_tokens, 24, 3))
    masks = np.zeros((count, n_tokens, 24), dtype=np.float32)
    masks[0, :, :5] = 1
    masks[0, ::7] = 0
    features = template_geometry(restypes, positions, masks)
    active = tuple(int(slot) for slot in np.flatnonzero(masks.any(axis=(1, 2))))
    assert active == (0,)
    for name, tensor in features.items():
        if name == "template_aatype":
            continue
        assert tensor[0].count_nonzero() > 0, name
        _assert_bytes_equal(transfer_template_tensor(tensor, device, (0, active, tuple(tensor.shape))), tensor)


def test_openfold3_native_distogram_keeps_absent_slots_byte_exact() -> None:
    n_tokens = 512
    context = {
        "_row": {
            # A supplied list that yields no usable hit retains all four slots.
            "templates_per_chain": {"unused": []},
            "structure": {
                "token_chain_ids": ["A"] * n_tokens,
                "token_res_ids": list(range(n_tokens)),
                "token_mol_types": [0] * n_tokens,
            },
        }
    }
    features = OpenFold3Templates()({"token_index": torch.arange(n_tokens)}, context)
    output = OpenFold3FinalFeatureCollator()(features, context)
    spec = output[TEMPLATE_TRANSFER_COLUMN]["template_distogram"]
    assert spec[1] == ()
    assert output["template_distogram"].shape[1] == 4
    actual = transfer_template_tensor(output["template_distogram"], "cpu", spec)
    _assert_bytes_equal(actual, output["template_distogram"])
