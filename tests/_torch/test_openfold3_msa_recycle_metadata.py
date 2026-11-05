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

from typing import Literal

import pytest
import torch

from bionemo_ir._torch.modules.openfold3.embedders import MSAModuleEmbedder
from bionemo_ir._torch.utils.common import commit_graph_safe_generator, make_graph_safe_generator
from bionemo_ir.models.openfold3.config import MSAModuleEmbedderConfig

Pattern = Literal["all", "none", "mixed", "fractional"]
DEPTH = 4


class _MsaSelectionProbe:
    prepare_msa = MSAModuleEmbedder.prepare_msa
    prepare_msa_indices = MSAModuleEmbedder.prepare_msa_indices
    _draw_all_msa_depth = MSAModuleEmbedder._draw_all_msa_depth
    _msa_valid_indices = staticmethod(MSAModuleEmbedder._msa_valid_indices)
    _select_all_msa_rows = staticmethod(MSAModuleEmbedder._select_all_msa_rows)
    _msa_features = staticmethod(MSAModuleEmbedder._msa_features)
    _subsample_all_msa = staticmethod(MSAModuleEmbedder._subsample_all_msa)
    _apply_subsample_fn_batch = MSAModuleEmbedder._apply_subsample_fn_batch
    subsample_main_msa = False
    subsample_all_msa = True
    max_subsampled_all_msa = DEPTH

    def __init__(self, *, fixed: bool, assume_valid: bool) -> None:
        self.min_subsampled_all_msa = DEPTH if fixed else 2
        self.assume_valid_msa_rows = assume_valid


def _legacy_prepare_msa(
    batch: dict[str, torch.Tensor], generator: torch.Generator, low: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-MSA selection as on main before row metadata reuse."""
    feat = torch.cat([batch["msa"], batch["has_deletion"][..., None], batch["deletion_value"][..., None]], dim=-1)
    mask = batch["msa_mask"]
    count = torch.randint(low, DEPTH + 1, (1,), device=feat.device, generator=generator).item()

    def select(feat: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = (mask.sum(dim=-1) > 0).squeeze()
        valid = valid.unsqueeze(0) if valid.ndim == 0 else valid
        valid_idx = valid.nonzero().squeeze()
        invalid_idx = (~valid).nonzero().squeeze()
        valid_idx = valid_idx.unsqueeze(0) if valid_idx.ndim == 0 else valid_idx
        invalid_idx = invalid_idx.unsqueeze(0) if invalid_idx.ndim == 0 else invalid_idx
        if valid_idx.numel() >= count:
            perm = torch.randperm(valid_idx.numel(), device=feat.device, generator=generator)
            selected = valid_idx[perm][:count]
        elif invalid_idx.numel() > 0:
            perm = torch.randperm(invalid_idx.numel(), device=feat.device, generator=generator)
            selected = torch.cat([valid_idx, invalid_idx[perm][: count - valid_idx.numel()]])
        else:
            selected = valid_idx
        return feat.index_select(-3, selected), mask.index_select(-2, selected)

    if feat.shape[0] == 1:
        return select(feat, mask)
    picked = [select(f, m) for f, m in zip(feat, mask, strict=True)]
    rows = max(f.shape[-3] for f, _ in picked)
    return (
        torch.stack([torch.nn.functional.pad(f, (0, 0, 0, 0, 0, rows - f.shape[-3])) for f, _ in picked]),
        torch.stack([torch.nn.functional.pad(m, (0, 0, 0, rows - m.shape[-2])) for _, m in picked]),
    )


def _batch(device: str, batch_size: int, rows: int, pattern: Pattern) -> dict[str, torch.Tensor]:
    tokens, channels = 5, 3
    mask = torch.ones(batch_size, rows, tokens, device=device)
    if pattern == "none":
        mask.zero_()
    elif pattern == "mixed":
        mask[:, 1::2] = 0
    elif pattern == "fractional":
        mask[..., 0] = -4
        mask[:, ::2, 1] = 0.25
    shape = (batch_size, rows, tokens)
    return {
        "msa": torch.arange(batch_size * rows * tokens * channels, device=device).reshape(*shape, channels).int(),
        "has_deletion": torch.rand(shape, device=device),
        "deletion_value": torch.rand(shape, device=device),
        "msa_mask": mask,
    }


def _cuda_or_skip(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")


def test_rows_are_valid_by_default() -> None:
    assert MSAModuleEmbedderConfig().assume_valid_msa_rows


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("assume_valid", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "rows,pattern",
    [(1, "all"), (3, "all"), (9, "all"), (9, "none"), (9, "mixed"), (9, "fractional")],
)
def test_prepared_msa_matches_legacy_bytes_and_rng(
    device: str, assume_valid: bool, fixed: bool, batch_size: int, rows: int, pattern: Pattern
) -> None:
    _cuda_or_skip(device)
    if assume_valid and batch_size == 1 and pattern != "all":
        pytest.skip("empty rows break the all-valid contract")
    probe = _MsaSelectionProbe(fixed=fixed, assume_valid=assume_valid)
    batch = _batch(device, batch_size, rows, pattern)
    original = {name: value.clone() for name, value in batch.items()}
    legacy_generator = torch.Generator(device=device).manual_seed(17)
    prepared_generator = torch.Generator(device=device).manual_seed(17)
    indices = probe.prepare_msa_indices(batch)
    assert (indices is not None) == (batch_size == 1)

    for _ in range(4):
        expected = _legacy_prepare_msa(batch, legacy_generator, probe.min_subsampled_all_msa)
        actual = probe.prepare_msa(batch, generator=prepared_generator, prepared_indices=indices)
        assert all(a.dtype == b.dtype and torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
        assert torch.equal(legacy_generator.get_state(), prepared_generator.get_state())
    assert batch.keys() == original.keys()
    assert all(torch.equal(batch[name], value) for name, value in original.items())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("rows", [3, 9])
def test_prepared_msa_preserves_default_rng(device: str, rows: int) -> None:
    _cuda_or_skip(device)
    probe = _MsaSelectionProbe(fixed=True, assume_valid=True)
    batch = _batch(device, 1, rows, "all")
    indices = probe.prepare_msa_indices(batch)
    initial = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()

    def rng_state() -> torch.Tensor:
        return torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()

    def legacy(_: object) -> tuple[torch.Tensor, torch.Tensor]:
        generator = make_graph_safe_generator(torch.device(device))
        output = _legacy_prepare_msa(batch, generator, DEPTH)
        commit_graph_safe_generator(generator, torch.device(device))
        return output

    def prepared(prepared_indices: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor]:
        return probe.prepare_msa(batch, prepared_indices=prepared_indices)

    runs = []
    try:
        for run, arg in ((legacy, None), (prepared, indices), (prepared, None)):
            torch.cuda.set_rng_state(initial) if device == "cuda" else torch.set_rng_state(initial)
            runs.append([(run(arg), rng_state()) for _ in range(4)])
        for (expected, expected_state), *others in zip(*runs, strict=True):
            for actual, state in others:
                assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
                assert torch.equal(state, expected_state)
    finally:
        torch.cuda.set_rng_state(initial) if device == "cuda" else torch.set_rng_state(initial)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "assume_valid,fixed,batch_size,expected",
    [
        (True, True, 1, {"item": 0, "nonzero": 0, "randint": 4}),
        (False, True, 1, {"item": 0, "nonzero": 2, "randint": 4}),
        (True, False, 1, {"item": 4, "nonzero": 0, "randint": 4}),
        (True, True, 2, {"item": 0, "nonzero": 16, "randint": 4}),
        (True, False, 2, {"item": 4, "nonzero": 16, "randint": 4}),
    ],
)
def test_prepared_msa_blocking_sites(
    device: str,
    assume_valid: bool,
    fixed: bool,
    batch_size: int,
    expected: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cuda_or_skip(device)
    probe = _MsaSelectionProbe(fixed=fixed, assume_valid=assume_valid)
    batch = _batch(device, batch_size, 9, "all")
    counts = dict.fromkeys(expected, 0)
    original_item, original_nonzero, original_randint = torch.Tensor.item, torch.Tensor.nonzero, torch.randint

    def item(tensor: torch.Tensor, *args: object, **kwargs: object) -> int | float | bool | complex:
        counts["item"] += 1
        return original_item(tensor, *args, **kwargs)

    def nonzero(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        counts["nonzero"] += 1
        return original_nonzero(tensor, *args, **kwargs)

    def randint(*args: object, **kwargs: object) -> torch.Tensor:
        counts["randint"] += 1
        return original_randint(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", item)
    monkeypatch.setattr(torch.Tensor, "nonzero", nonzero)
    monkeypatch.setattr(torch, "randint", randint)
    indices = probe.prepare_msa_indices(batch)
    generator = torch.Generator(device=device).manual_seed(17)
    for _ in range(4):
        probe.prepare_msa(batch, generator=generator, prepared_indices=indices)
    assert counts == expected


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_identity_selection_keeps_request_mask(device: str) -> None:
    _cuda_or_skip(device)
    probe = _MsaSelectionProbe(fixed=True, assume_valid=True)
    batch = _batch(device, 1, DEPTH - 1, "all")
    generator = torch.Generator(device=device).manual_seed(17)
    msa_feat, msa_mask = probe.prepare_msa(batch, generator=generator)
    assert msa_mask is batch["msa_mask"]
    assert msa_feat.shape == (*batch["msa"].shape[:-1], batch["msa"].shape[-1] + 2)


@pytest.mark.parametrize("main,all_msa", [(True, True), (False, False)])
def test_other_msa_modes_do_not_prepare_indices(main: bool, all_msa: bool) -> None:
    probe = _MsaSelectionProbe(fixed=True, assume_valid=True)
    probe.subsample_main_msa = main
    probe.subsample_all_msa = all_msa
    assert probe.prepare_msa_indices({}) is None
