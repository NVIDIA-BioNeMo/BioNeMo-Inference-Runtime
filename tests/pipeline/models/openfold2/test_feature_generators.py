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

from bionemo_ir.pipeline.models.openfold2.feature_generators import (
    Atom37ToTorsionAngles,
    MakeAtom14Masks,
    MakeHhblitsProfile,
    MakeMsaMask,
    MakeSequenceMask,
    MakeTemplateMask,
    MakeTemplatePseudoBeta,
    UseClampedFape,
)


@dataclass
class MockConfig:
    max_recycling_iters: int = 3
    enable_template: bool = True
    use_template_torsion_angles: bool = True


@pytest.fixture
def mock_config():
    return MockConfig()


@pytest.fixture
def sample_batch():
    n_res = 64
    n_msa = 128
    n_templ = 4
    return {
        "aatype": torch.randint(0, 20, (n_res,)),
        "msa": torch.randint(0, 22, (n_msa, n_res)),
        "template_aatype": torch.randint(0, 21, (n_templ, n_res)),
        "template_all_atom_positions": torch.randn(n_templ, n_res, 37, 3),
        "template_all_atom_mask": torch.ones(n_templ, n_res, 37),
    }


@pytest.fixture
def context():
    return {}


class TestUseClampedFape:
    def test_generates_use_clamped_fape(self, mock_config, sample_batch, context):
        generator = UseClampedFape(config=mock_config)
        result = generator(sample_batch, context)

        assert "use_clamped_fape" in result
        assert result["use_clamped_fape"].shape[0] == mock_config.max_recycling_iters + 1

    def test_use_clamped_fape_values(self, mock_config, sample_batch, context):
        generator = UseClampedFape(config=mock_config)
        result = generator(sample_batch, context)

        assert torch.all(result["use_clamped_fape"] == 0.0)

    def test_use_clamped_fape_dtype(self, mock_config, sample_batch, context):
        generator = UseClampedFape(config=mock_config)
        result = generator(sample_batch, context)

        assert result["use_clamped_fape"].dtype == torch.float32


class TestMakeSequenceMask:
    def test_generates_seq_mask(self, mock_config, sample_batch, context):
        generator = MakeSequenceMask(config=mock_config)
        result = generator(sample_batch, context)

        assert "seq_mask" in result
        assert result["seq_mask"].shape == sample_batch["aatype"].shape

    def test_seq_mask_all_ones(self, mock_config, sample_batch, context):
        generator = MakeSequenceMask(config=mock_config)
        result = generator(sample_batch, context)

        assert torch.all(result["seq_mask"] == 1.0)

    def test_seq_mask_dtype(self, mock_config, sample_batch, context):
        generator = MakeSequenceMask(config=mock_config)
        result = generator(sample_batch, context)

        assert result["seq_mask"].dtype == torch.float32


class TestMakeMsaMask:
    def test_generates_msa_mask(self, mock_config, sample_batch, context):
        generator = MakeMsaMask(config=mock_config)
        result = generator(sample_batch, context)

        assert "msa_mask" in result
        assert "msa_row_mask" in result

    def test_msa_mask_shape(self, mock_config, sample_batch, context):
        generator = MakeMsaMask(config=mock_config)
        result = generator(sample_batch, context)

        assert result["msa_mask"].shape == sample_batch["msa"].shape
        assert result["msa_row_mask"].shape[0] == sample_batch["msa"].shape[0]

    def test_msa_mask_all_ones(self, mock_config, sample_batch, context):
        generator = MakeMsaMask(config=mock_config)
        result = generator(sample_batch, context)

        assert torch.all(result["msa_mask"] == 1.0)
        assert torch.all(result["msa_row_mask"] == 1.0)


class TestMakeTemplateMask:
    def test_generates_template_mask(self, mock_config, sample_batch, context):
        generator = MakeTemplateMask(config=mock_config)
        result = generator(sample_batch, context)

        assert "template_mask" in result

    def test_template_mask_shape(self, mock_config, sample_batch, context):
        generator = MakeTemplateMask(config=mock_config)
        result = generator(sample_batch, context)

        n_templ = sample_batch["template_aatype"].shape[0]
        assert result["template_mask"].shape[0] == n_templ

    def test_is_enabled_when_template_enabled(self, mock_config):
        generator = MakeTemplateMask(config=mock_config)
        assert generator.is_enabled() is True

    def test_is_disabled_when_template_disabled(self):
        config = MockConfig(enable_template=False)
        generator = MakeTemplateMask(config=config)
        assert generator.is_enabled() is False


class TestMakeTemplatePseudoBeta:
    def test_generates_pseudo_beta(self, mock_config, sample_batch, context):
        generator = MakeTemplatePseudoBeta(config=mock_config)
        result = generator(sample_batch, context)

        assert "template_pseudo_beta" in result
        assert "template_pseudo_beta_mask" in result

    def test_pseudo_beta_shape(self, mock_config, sample_batch, context):
        generator = MakeTemplatePseudoBeta(config=mock_config)
        result = generator(sample_batch, context)

        n_templ, n_res = sample_batch["template_aatype"].shape
        assert result["template_pseudo_beta"].shape == (n_templ, n_res, 3)
        assert result["template_pseudo_beta_mask"].shape == (n_templ, n_res)

    def test_is_enabled_when_template_enabled(self, mock_config):
        generator = MakeTemplatePseudoBeta(config=mock_config)
        assert generator.is_enabled() is True


class TestMakeHhblitsProfile:
    def test_generates_hhblits_profile(self, mock_config, sample_batch, context):
        generator = MakeHhblitsProfile(config=mock_config)
        result = generator(sample_batch, context)

        assert "hhblits_profile" in result

    def test_hhblits_profile_shape(self, mock_config, sample_batch, context):
        generator = MakeHhblitsProfile(config=mock_config)
        result = generator(sample_batch, context)

        n_res = sample_batch["msa"].shape[1]
        assert result["hhblits_profile"].shape == (n_res, 22)

    def test_skips_if_already_present(self, mock_config, sample_batch, context):
        sample_batch["hhblits_profile"] = torch.randn(64, 22)
        generator = MakeHhblitsProfile(config=mock_config)
        result = generator(sample_batch, context)

        assert len(result) == 0

    def test_hhblits_profile_sums_to_one(self, mock_config, sample_batch, context):
        generator = MakeHhblitsProfile(config=mock_config)
        result = generator(sample_batch, context)

        sums = result["hhblits_profile"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestMakeAtom14Masks:
    def test_generates_atom14_masks(self, mock_config, sample_batch, context):
        generator = MakeAtom14Masks(config=mock_config)
        result = generator(sample_batch, context)

        assert "atom14_atom_exists" in result
        assert "residx_atom14_to_atom37" in result
        assert "residx_atom37_to_atom14" in result
        assert "atom37_atom_exists" in result

    def test_atom14_atom_exists_shape(self, mock_config, sample_batch, context):
        generator = MakeAtom14Masks(config=mock_config)
        result = generator(sample_batch, context)

        n_res = sample_batch["aatype"].shape[0]
        assert result["atom14_atom_exists"].shape == (n_res, 14)

    def test_atom37_atom_exists_shape(self, mock_config, sample_batch, context):
        generator = MakeAtom14Masks(config=mock_config)
        result = generator(sample_batch, context)

        n_res = sample_batch["aatype"].shape[0]
        assert result["atom37_atom_exists"].shape == (n_res, 37)

    def test_residx_atom14_to_atom37_shape(self, mock_config, sample_batch, context):
        generator = MakeAtom14Masks(config=mock_config)
        result = generator(sample_batch, context)

        n_res = sample_batch["aatype"].shape[0]
        assert result["residx_atom14_to_atom37"].shape == (n_res, 14)

    def test_residx_atom37_to_atom14_shape(self, mock_config, sample_batch, context):
        generator = MakeAtom14Masks(config=mock_config)
        result = generator(sample_batch, context)

        n_res = sample_batch["aatype"].shape[0]
        assert result["residx_atom37_to_atom14"].shape == (n_res, 37)


class TestAtom37ToTorsionAngles:
    def test_is_enabled_check(self, mock_config):
        generator = Atom37ToTorsionAngles(config=mock_config, prefix="template_")
        assert generator.is_enabled() is True

    def test_is_disabled_when_template_disabled(self):
        config = MockConfig(enable_template=False)
        generator = Atom37ToTorsionAngles(config=config, prefix="template_")
        assert generator.is_enabled() is False

    def test_is_disabled_when_torsion_angles_disabled(self):
        config = MockConfig(use_template_torsion_angles=False)
        generator = Atom37ToTorsionAngles(config=config, prefix="template_")
        assert generator.is_enabled() is False
