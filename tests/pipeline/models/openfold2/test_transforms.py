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

from dataclasses import dataclass

import pytest
import torch

from bionemo_ir.pipeline.models.openfold2.transforms import (
    CastTo64BitInts,
    CorrectMsaRestypes,
    FixTemplatesAatype,
    RandomlyReplaceMsaWithUnknown,
    SqueezeFeatures,
)


@dataclass
class MockConfig:
    enable_template: bool = True


@pytest.fixture
def mock_config():
    return MockConfig()


@pytest.fixture
def sample_batch():
    n_res = 64
    n_msa = 128
    n_templ = 4
    return {
        "aatype": torch.nn.functional.one_hot(torch.randint(0, 21, (n_res,)), 21).float(),
        "msa": torch.randint(0, 22, (n_msa, n_res), dtype=torch.int64),
        "deletion_matrix": torch.rand(n_msa, n_res, 1),
        "residue_index": torch.arange(n_res, dtype=torch.int32).unsqueeze(-1),
        "between_segment_residues": torch.zeros(n_res, 1, dtype=torch.int32),
        "seq_length": torch.tensor([n_res] * n_res, dtype=torch.int32),
        "num_alignments": torch.tensor([n_msa] * n_res, dtype=torch.int32),
        "template_aatype": torch.nn.functional.one_hot(torch.randint(0, 22, (n_templ, n_res)), 22).float(),
        "template_all_atom_mask": torch.ones(n_templ, n_res, 37, 1),
    }


class TestCastTo64BitInts:
    def test_converts_int32_to_int64(self, mock_config):
        batch = {
            "int32_tensor": torch.tensor([1, 2, 3], dtype=torch.int32),
            "float_tensor": torch.tensor([1.0, 2.0, 3.0]),
        }
        transform = CastTo64BitInts(config=mock_config)
        result = transform(batch)

        assert result["int32_tensor"].dtype == torch.int64
        assert result["float_tensor"].dtype == torch.float32

    def test_preserves_values(self, mock_config):
        original = torch.tensor([1, 2, 3], dtype=torch.int32)
        batch = {"tensor": original.clone()}
        transform = CastTo64BitInts(config=mock_config)
        result = transform(batch)

        assert torch.equal(result["tensor"], original.to(torch.int64))


class TestCorrectMsaRestypes:
    def test_reorders_msa(self, mock_config, sample_batch):
        transform = CorrectMsaRestypes(config=mock_config)
        result = transform(sample_batch)

        assert "msa" in result
        assert result["msa"].shape == sample_batch["msa"].shape

    def test_reorders_profile_if_present(self, mock_config, sample_batch):
        sample_batch["hhblits_profile"] = torch.rand(64, 22)
        transform = CorrectMsaRestypes(config=mock_config)
        result = transform(sample_batch)

        assert "hhblits_profile" in result
        assert result["hhblits_profile"].shape == (64, 22)


class TestSqueezeFeatures:
    def test_converts_aatype_from_onehot_to_indices(self, mock_config, sample_batch):
        transform = SqueezeFeatures(config=mock_config)
        result = transform(sample_batch)

        assert result["aatype"].dim() == 1
        assert result["aatype"].shape[0] == 64

    def test_squeezes_singleton_dimensions(self, mock_config, sample_batch):
        transform = SqueezeFeatures(config=mock_config)
        result = transform(sample_batch)

        assert result["deletion_matrix"].dim() == 2
        assert result["residue_index"].dim() == 1
        assert result["between_segment_residues"].dim() == 1

    def test_reduces_seq_length_to_scalar(self, mock_config, sample_batch):
        transform = SqueezeFeatures(config=mock_config)
        result = transform(sample_batch)

        assert result["seq_length"].dim() == 0 or result["seq_length"].shape == ()


class TestRandomlyReplaceMsaWithUnknown:
    def test_no_replacement_when_proportion_zero(self, mock_config, sample_batch):
        original_msa = sample_batch["msa"].clone()
        transform = RandomlyReplaceMsaWithUnknown(config=mock_config, replace_proportion=0.0)
        result = transform(sample_batch)

        assert torch.equal(result["msa"], original_msa)

    def test_replaces_some_when_proportion_positive(self, mock_config, sample_batch):
        original_msa = sample_batch["msa"].clone()
        transform = RandomlyReplaceMsaWithUnknown(config=mock_config, replace_proportion=0.5)
        result = transform(sample_batch)

        x_idx = 20
        has_replacements = (result["msa"] == x_idx).any()
        assert has_replacements or torch.equal(result["msa"], original_msa)

    def test_does_not_replace_gaps(self, mock_config, sample_batch):
        gap_idx = 21
        sample_batch["msa"][:, 0] = gap_idx
        transform = RandomlyReplaceMsaWithUnknown(config=mock_config, replace_proportion=1.0)
        result = transform(sample_batch)

        assert torch.all(result["msa"][:, 0] == gap_idx)


class TestFixTemplatesAatype:
    def test_converts_onehot_to_indices(self, mock_config, sample_batch):
        transform = FixTemplatesAatype(config=mock_config)
        result = transform(sample_batch)

        assert result["template_aatype"].dim() == 2

    def test_is_enabled_when_template_enabled(self, mock_config):
        transform = FixTemplatesAatype(config=mock_config)
        assert transform.is_enabled() is True

    def test_is_disabled_when_template_disabled(self):
        config = MockConfig(enable_template=False)
        transform = FixTemplatesAatype(config=config)
        assert transform.is_enabled() is False

    def test_reorders_aatype_indices(self, mock_config, sample_batch):
        transform = FixTemplatesAatype(config=mock_config)
        result = transform(sample_batch)

        assert result["template_aatype"].dtype == torch.int64
        assert result["template_aatype"].max() <= 21
