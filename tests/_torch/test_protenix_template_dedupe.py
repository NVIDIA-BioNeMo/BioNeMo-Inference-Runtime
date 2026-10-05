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
    return model


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
