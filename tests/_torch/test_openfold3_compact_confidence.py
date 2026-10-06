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
"""Compact OpenFold3 confidence outputs: the head reduces PAE per sample and skips unused heads."""

import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from bionemo_ir._torch.layers.linear import Linear
from bionemo_ir._torch.modules.openfold3 import confidence as confidence_module
from bionemo_ir._torch.modules.openfold3.confidence import (
    AuxiliaryHeadsAllAtom,
    DistogramHead,
    ExperimentallyResolvedHeadAllAtom,
    PerResidueLDDTAllAtom,
    PredictedAlignedErrorHead,
    PredictedDistanceErrorHead,
)
from bionemo_ir._torch.modules.openfold3.utils.confidence import reduce_pae_logits
from bionemo_ir._torch.utils import ChunkPolicy
from bionemo_ir.models.openfold3.config import AuxiliaryHeadsConfig
from bionemo_ir.pipeline.models.openfold3.postprocessor import (
    _compact_confidence,
    _plddt_per_atom,
    _reduce_confidence,
    _select_best_sample,
)
from tests._torch.test_openfold3_confidence_memory import _FakePairformerEmbedding

TOKENS, CHANNELS = 8, 8


def _heads_fixture(samples: int, batch_size: int = 1) -> tuple[AuxiliaryHeadsAllAtom, dict, torch.Tensor, dict]:
    module = AuxiliaryHeadsAllAtom.__new__(AuxiliaryHeadsAllAtom)
    nn.Module.__init__(module)
    module.config = AuxiliaryHeadsConfig(max_atoms_per_token=1)
    module.max_atoms_per_token = 1
    module.dtype = torch.float32
    module.apply_per_sample = True
    module.offload_pairformer_outputs = True
    module.compact_output = False
    module.pairformer_embedding = _FakePairformerEmbedding()
    module.pae = PredictedAlignedErrorHead(CHANNELS, 64, skip_create_weights=True)
    module.pde = PredictedDistanceErrorHead(CHANNELS, 64, skip_create_weights=True)
    module.pde.projection_chunk_policy = ChunkPolicy(chunk_size=3, min_size=1)
    module.distogram = DistogramHead(CHANNELS, 64, skip_create_weights=True)
    module.plddt = PerResidueLDDTAllAtom(CHANNELS, 50, 1, skip_create_weights=True)
    module.experimentally_resolved = ExperimentallyResolvedHeadAllAtom(CHANNELS, 2, 1, skip_create_weights=True)
    for child in module.modules():
        if isinstance(child, Linear):
            child.weight = nn.Parameter(torch.empty(child.out_features, child.in_features, dtype=child.dtype))
            child.register_parameter("bias", None)
            child._weights_created = True
    module.eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0, std=0.1)
    mask = torch.ones(batch_size, 1, TOKENS)
    batch = {
        "token_mask": mask,
        "atom_mask": mask.clone(),
        "num_atoms_per_token": torch.ones_like(mask, dtype=torch.long),
        "asym_id": (torch.arange(TOKENS) % 2 + 1).expand(batch_size, 1, TOKENS).clone(),
        "start_atom_index": torch.arange(TOKENS).expand(batch_size, 1, TOKENS).clone(),
        "restype": torch.nn.functional.one_hot(torch.zeros_like(mask, dtype=torch.long), 32).float(),
        "is_protein": mask.clone(),
        "is_atomized": mask.clone(),
        "is_dna": torch.zeros_like(mask),
        "is_rna": torch.zeros_like(mask),
    }
    si_input = torch.randn(batch_size, 1, TOKENS, 7)
    output = {
        "si_trunk": torch.randn(batch_size, 1, TOKENS, CHANNELS),
        "zij_trunk": torch.randn(batch_size, 1, TOKENS, TOKENS, CHANNELS),
        "atom_positions_predicted": torch.randn(batch_size, samples, TOKENS, 3),
    }
    return module, batch, si_input, output


def _raw_reference(expected: dict, batch: dict, b: int, s: int, n_tokens: int) -> tuple[np.ndarray, float, float]:
    """PAE, pTM and ipTM from the raw logits through the postprocessor's own reduction."""
    logits = expected["pae_logits"][b, s, :n_tokens, :n_tokens]
    frames = expected["valid_frame_mask"][b, s, :n_tokens].bool()
    chains = batch["asym_id"][b, 0, :n_tokens].numpy()
    return _reduce_confidence(logits, n_tokens, chains, frames)


@pytest.mark.parametrize("samples", [1, 5])
@pytest.mark.parametrize("pae_enabled", [False, True])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16])
def test_compact_forward_contract(samples, pae_enabled, output_dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compact mode keeps pLDDT and frames exactly, reduces every sample, and never runs the skipped heads."""
    torch.manual_seed(305)
    module, batch, si_input, output = _heads_fixture(samples, batch_size=2)
    module.config.pae.enabled = pae_enabled
    output["atom_positions_predicted"] = output["atom_positions_predicted"].to(output_dtype)
    originals = {name: value.clone() for name, value in output.items()}
    keys = set(module.state_dict())
    with torch.inference_mode():
        expected = module(batch, si_input, output)
        module.compact_output = True
        rng_before = torch.get_rng_state()

        def reject_unused(*_args, **_kwargs) -> torch.Tensor:
            raise AssertionError("a skipped confidence head ran in compact mode")

        for head in (module.pde, module.distogram, module.experimentally_resolved):
            monkeypatch.setattr(head, "forward", reject_unused)
        monkeypatch.setattr(module.pae, "forward", reject_unused)
        actual = module(batch, si_input, output)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert set(module.state_dict()) == keys
    assert torch.equal(actual["plddt_logits"], expected["plddt_logits"])
    assert actual["plddt_logits"].dtype == output_dtype
    if not pae_enabled:
        assert set(actual) == {"plddt_logits"}
    else:
        assert set(actual) == {"plddt_logits", "valid_frame_mask", "pae", "ptm", "iptm"}
        assert torch.equal(actual["valid_frame_mask"], expected["valid_frame_mask"])
        assert actual["pae"].shape == (2, samples, TOKENS, TOKENS)
        assert actual["pae"].dtype == actual["ptm"].dtype == actual["iptm"].dtype == torch.float32
        assert actual["ptm"].shape == actual["iptm"].shape == (2, samples)
        for b in range(2):
            for s in range(samples):
                pae, ptm, iptm = _raw_reference(expected, batch, b, s, TOKENS)
                np.testing.assert_allclose(np.round(actual["pae"][b, s].numpy(), 3), pae, atol=1e-3)
                torch.testing.assert_close(
                    actual["pae"][b, s], _reduce_pae_unrounded(expected, b, s), atol=1e-6, rtol=0
                )
                assert float(actual["ptm"][b, s]) == pytest.approx(ptm, abs=1e-6)
                assert float(actual["iptm"][b, s]) == pytest.approx(iptm, abs=1e-6)
        assert _select_best_sample(_plddt_per_atom(actual)) == _select_best_sample(_plddt_per_atom(expected))
    for name, original in originals.items():
        assert torch.equal(output[name], original)


def _reduce_pae_unrounded(expected: dict, b: int, s: int) -> torch.Tensor:
    logits = expected["pae_logits"][b, s]
    return reduce_pae_logits(logits, TOKENS, torch.zeros(TOKENS, dtype=torch.long), None).pae


def test_compact_output_requires_per_sample_mode() -> None:
    """Compact mode only exists on the per-sample path; asking for both is a configuration error, not a silent fallback."""
    config = AuxiliaryHeadsConfig(
        max_atoms_per_token=1, memory_efficient_mode=False, compact_output=True, skip_create_weights=True
    )
    config.set_dtype("float32")
    with pytest.raises(ValueError, match="memory_efficient_mode"):
        AuxiliaryHeadsAllAtom(config)


@pytest.mark.parametrize("guard", ["default", "training", "grad", "batched"])
def test_compact_guards_preserve_raw_outputs(guard: str) -> None:
    """Outside per-sample inference the raw contract is returned unchanged."""
    module, batch, si_input, output = _heads_fixture(2)
    module.compact_output = guard != "default"
    if guard == "training":
        module.train()
    elif guard == "batched":
        module.apply_per_sample = False
    with torch.set_grad_enabled(guard == "grad"):
        assert not module._should_compact(output["zij_trunk"])
        actual = module(batch, si_input, output)
        module.compact_output = False
        expected = module(batch, si_input, output)
    assert actual.keys() == expected.keys()
    assert "pae" not in actual
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0, equal_nan=True)


def test_capture_keeps_the_raw_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CUDA graph capture cannot host the per-sample host syncs, so the raw path serves it."""
    module, _, _, _ = _heads_fixture(1)
    module.compact_output = True
    probe = SimpleNamespace(is_cuda=True)
    with torch.inference_mode():
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        assert not module._should_compact(probe)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        assert module._should_compact(probe)


def test_compact_crops_each_batch_element_and_releases_samples() -> None:
    """Padding is scored out, the padded matrix stays zero, and no projected block outlives its reduction."""
    module, batch, si_input, output = _heads_fixture(3, batch_size=2)
    n_tokens = 5
    batch["token_mask"][:, 0, n_tokens:] = 0
    batch["asym_id"][1] = 1  # the second element is a monomer: no interface, NaN ipTM
    frames = torch.ones(2, 3, TOKENS)
    frames[0, 1, 0] = 0.0
    arguments = {
        "si_input": si_input,
        "si": output["si_trunk"],
        "zij": output["zij_trunk"],
        "x_pred": output["atom_positions_predicted"],
        "single_mask": batch["token_mask"].expand(2, 3, TOKENS),
        "pair_mask": batch["token_mask"][..., None] * batch["token_mask"][..., None, :],
    }
    released: list[weakref.ReferenceType[torch.Tensor]] = []
    original_project = module._project_pae_rows

    def project(zij_rows: torch.Tensor) -> torch.Tensor:
        assert all(reference() is None for reference in released)
        logits = original_project(zij_rows)
        released.append(weakref.ref(logits))
        return logits

    module._project_pae_rows = project
    with torch.inference_mode():
        _, expected = module._stream_pair_heads(
            **arguments, output_device=torch.device("cpu"), pair_output_dtype=torch.float32
        )
        si_output, actual = module._stream_compact_heads(**arguments, batch=batch, frames=frames)
    assert released and all(reference() is None for reference in released)
    assert si_output.shape == (2, 3, TOKENS, CHANNELS)
    assert torch.isnan(actual["iptm"][1]).all() and torch.isfinite(actual["iptm"][0]).all()
    for b in range(2):
        for s in range(3):
            logits = expected["pae_logits"][b, s, :n_tokens, :n_tokens]
            pae, ptm, iptm = _reduce_confidence(
                logits, n_tokens, batch["asym_id"][b, 0, :n_tokens].numpy(), frames[b, s, :n_tokens].bool()
            )
            np.testing.assert_allclose(np.round(actual["pae"][b, s, :n_tokens, :n_tokens].numpy(), 3), pae, atol=1e-3)
            assert torch.count_nonzero(actual["pae"][b, s, n_tokens:]) == 0
            assert torch.count_nonzero(actual["pae"][b, s, :, n_tokens:]) == 0
            assert float(actual["ptm"][b, s]) == pytest.approx(ptm, abs=1e-6)
            assert float(actual["iptm"][b, s]) == pytest.approx(iptm, abs=1e-6, nan_ok=True)


def test_compact_monomer_and_no_frame_keep_nan_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    module, batch, si_input, output = _heads_fixture(2)
    module.compact_output = True
    batch["asym_id"][...] = 1
    with torch.inference_mode():
        actual = module(batch, si_input, output)
    assert torch.isnan(actual["iptm"]).all()
    assert torch.isfinite(actual["ptm"]).all()

    module, batch, si_input, output = _heads_fixture(2)
    module.compact_output = True
    monkeypatch.setattr(
        confidence_module, "get_token_frame_mask", lambda batch, x, atom_mask: torch.zeros(1, 2, TOKENS, dtype=x.dtype)
    )
    with torch.inference_mode():
        actual = module(batch, si_input, output)
    assert torch.count_nonzero(actual["valid_frame_mask"]) == 0
    assert torch.isnan(actual["ptm"]).all() and torch.isnan(actual["iptm"]).all()


def test_compact_reader_selects_rounds_and_transfers_once(monkeypatch: pytest.MonkeyPatch) -> None:
    pae = torch.arange(3 * 7 * 7).reshape(1, 3, 7, 7).float() / 7
    output = {"pae": pae, "ptm": torch.tensor([[0.1, 0.2, 0.3]]), "iptm": torch.tensor([[0.4, 0.5, float("nan")]])}
    actual_pae, ptm, iptm = _compact_confidence(output, 2, 4)
    np.testing.assert_array_equal(actual_pae, np.round(pae[0, 2, :4, :4].numpy(), 3))
    assert actual_pae.dtype == np.float32
    assert ptm == pytest.approx(0.3)
    assert np.isnan(iptm)
    # Without batch and sample axes, as a single-sample caller may pass them.
    flat = {"pae": pae[0, 1], "ptm": output["ptm"][0, 1], "iptm": output["iptm"][0, 1]}
    np.testing.assert_array_equal(_compact_confidence(flat, 0, 4)[0], np.round(pae[0, 1, :4, :4].numpy(), 3))


@pytest.mark.parametrize("sample_mode", ["tie", "nan"])
def test_compact_preserves_sample_selection(sample_mode: str) -> None:
    logits = torch.zeros(1, 5, 8, 50)
    if sample_mode == "nan":
        logits[0, 3, 0, 0] = float("nan")
    raw = {"plddt_logits": logits}
    compact = {"plddt_logits": logits.clone(), "pae": torch.zeros(1, 5, 8, 8)}
    expected = 0 if sample_mode == "tie" else 3
    assert _select_best_sample(_plddt_per_atom(raw)) == expected
    assert _select_best_sample(_plddt_per_atom(compact)) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("offload", [False, True])
def test_cuda_compact_outputs_stay_on_device(offload: bool) -> None:
    """Compact results live on the coordinate device whatever the raw offload policy says."""
    torch.manual_seed(305)
    module, batch, si_input, output = _heads_fixture(5)
    module.to(device="cuda", dtype=torch.bfloat16)
    module.dtype = torch.bfloat16
    module.offload_pairformer_outputs = offload
    batch = {name: value.cuda() for name, value in batch.items()}
    output = {name: value.cuda() for name, value in output.items()}
    si_input = si_input.cuda()
    with torch.inference_mode():
        expected = module(batch, si_input, output)
        module.compact_output = True
        actual = module(batch, si_input, output)
    assert torch.equal(actual["plddt_logits"], expected["plddt_logits"])
    for name in ("pae", "ptm", "iptm", "valid_frame_mask"):
        assert actual[name].device.type == "cuda"
    expected_cpu = {name: value.cpu() for name, value in expected.items()}
    batch_cpu = {name: value.cpu() for name, value in batch.items()}
    for s in range(5):
        pae, ptm, iptm = _raw_reference(expected_cpu, batch_cpu, 0, s, TOKENS)
        np.testing.assert_allclose(np.round(actual["pae"][0, s].cpu().numpy(), 3), pae, atol=1e-3)
        assert float(actual["ptm"][0, s]) == pytest.approx(ptm, abs=1e-5)
        assert float(actual["iptm"][0, s]) == pytest.approx(iptm, abs=1e-5)
