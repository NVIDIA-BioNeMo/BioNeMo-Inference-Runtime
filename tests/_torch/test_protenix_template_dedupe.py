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
"""Identical templates run the Protenix template pair stack once."""

import pytest
import torch

from bionemo_ir._torch.modules.protenix import ProtenixTemplateEmbedder
from bionemo_ir.models.protenix.config import TemplateEmbedderConfig
from tests._torch import run_post_load_weights

N_TOKEN = 24
N_TEMPLATES = 4


def _embedder(n_blocks: int) -> ProtenixTemplateEmbedder:
    config = TemplateEmbedderConfig(
        c=64,
        c_z=128,
        n_blocks=n_blocks,
        pairwise_head_width=32,
        pairwise_num_heads=2,
        dtype="float32",
        pairformer_dtype="bfloat16",
        triangle_attention_backend="VANILLA",
    )
    model = ProtenixTemplateEmbedder(config).cuda().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.05)
    return run_post_load_weights(model)


def _features(order: list[int]) -> dict[str, torch.Tensor]:
    """Batched features whose templates copy ``N_TEMPLATES`` random ones in ``order``."""
    g = torch.Generator(device="cpu").manual_seed(7)
    N, T = N_TOKEN, N_TEMPLATES
    templates = {
        "template_distogram": torch.randn(T, N, N, 39, generator=g),
        "template_pseudo_beta_mask": (torch.randn(T, N, N, generator=g) > 0).float(),
        "template_aatype": torch.randint(0, 32, (T, N), generator=g),
        "template_unit_vector": torch.randn(T, N, N, 3, generator=g),
        "template_backbone_frame_mask": (torch.randn(T, N, N, generator=g) > 0).float(),
    }
    features = {name: value[order].unsqueeze(0).cuda() for name, value in templates.items()}
    features["asym_id"] = (torch.arange(N) // (N // 2)).unsqueeze(0).cuda()
    return features


@pytest.mark.parametrize(
    ("order", "expected"),
    [([0, 0, 0, 0], [0, 0, 0, 0]), ([0, 1, 0, 1], [0, 1, 0, 1]), ([0, 1, 2, 3], [0, 1, 2, 3]), ([2], [0])],
)
def test_template_representatives(order: list[int], expected: list[int]):
    assert _embedder(n_blocks=0).template_representatives(_features(order)) == expected


@pytest.mark.parametrize("order", [[0, 0, 0, 0], [0, 1, 0, 1]], ids=["identical", "pairs"])
def test_template_dedupe_is_bitwise_exact(order: list[int]):
    """Evaluating each distinct template once matches evaluating every template."""
    torch.manual_seed(3)
    model = _embedder(n_blocks=2)
    features = _features(order)
    z = torch.randn(1, N_TOKEN, N_TOKEN, 128, device="cuda")

    with torch.inference_mode():
        deduped = model(features, z)
        every = model(features, z, representatives=list(range(len(order))))

    assert torch.equal(deduped, every)


def _with_empty_slots(hits: int) -> dict[str, torch.Tensor]:
    """Features whose slots from ``hits`` on carry no template, as the featurizer emits them."""
    features = _features([0, 1, 2, 3])
    for name in ProtenixTemplateEmbedder.pair_features:
        features[name][:, hits:] = 0
    features["template_aatype"][:, hits:] = 0
    return features


@pytest.mark.parametrize("hits", [0, 1, 2, 4])
@pytest.mark.parametrize("owned", [False, True])
def test_dropping_trailing_empty_slots_is_bitwise_exact(hits: int, owned: bool):
    """Shortened pair features embed exactly as the dense zero slots, with the same representatives."""
    torch.manual_seed(5)
    model = _embedder(n_blocks=2)
    dense = _with_empty_slots(hits)
    compact = dict(dense)
    ProtenixTemplateEmbedder.drop_trailing_empty_slots(compact, owned=owned)
    for name in ProtenixTemplateEmbedder.pair_features:
        assert compact[name].shape[1] == hits
        if hits:
            assert (compact[name].data_ptr() == dense[name].data_ptr()) == (not owned or hits == N_TEMPLATES)
    assert compact["template_aatype"] is dense["template_aatype"]
    # Compacting again, including an input that already has no slots, changes nothing.
    again = dict(compact)
    ProtenixTemplateEmbedder.drop_trailing_empty_slots(again, owned=owned)
    assert all(again[name] is compact[name] for name in compact)
    assert model.template_representatives(compact) == model.template_representatives(dense)
    z = torch.randn(1, N_TOKEN, N_TOKEN, 128, device="cuda")
    with torch.inference_mode():
        expected = model(dense, z)
        actual = model(compact, z)
    assert torch.equal(actual, expected)


def test_interior_empty_slot_is_kept():
    """Only trailing empty slots are dropped; an empty slot before a hit stays dense."""
    features = _features([0, 1, 2, 3])
    for name in ProtenixTemplateEmbedder.pair_features:
        features[name][:, 1] = 0
        features[name][:, 3] = 0
    ProtenixTemplateEmbedder.drop_trailing_empty_slots(features, owned=True)
    assert all(features[name].shape[1] == 3 for name in ProtenixTemplateEmbedder.pair_features)
