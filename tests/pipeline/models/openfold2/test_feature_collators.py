# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch

from tensorrt_bionemo.pipeline.models.openfold2.feature_collators import (
    CropExtraMsa, DeleteExtraMsa, MakeFixedSize, MakeMaskedMsa, MakeMsaFeat,
    NearestNeighborClusters, RandomCropToSize, SampleMsa, SelectFeat,
    SummarizeClusters)


@dataclass
class MockConfig:
    max_msa_clusters: int = 128
    max_extra_msa: int = 1024
    max_templates: int = 4
    resample_msa_in_recycling: bool = False
    msa_cluster_features: bool = True
    is_multimer: bool = False


@pytest.fixture
def mock_config():
    return MockConfig()


@pytest.fixture
def sample_features():
    n_res = 64
    n_msa = 256
    n_templ = 4
    return {
        "msa": torch.randint(0, 22, (n_msa, n_res)),
        "deletion_matrix": torch.rand(n_msa, n_res),
        "msa_mask": torch.ones(n_msa, n_res),
        "msa_row_mask": torch.ones(n_msa),
        "aatype": torch.randint(0, 21, (n_res, )),
        "between_segment_residues": torch.zeros(n_res),
        "hhblits_profile": torch.rand(n_res, 22),
        "seq_length": torch.tensor(n_res),
        "template_aatype": torch.randint(0, 21, (n_templ, n_res)),
        "template_all_atom_mask": torch.ones(n_templ, n_res, 37),
        "template_all_atom_positions": torch.randn(n_templ, n_res, 37, 3),
        "template_mask": torch.ones(n_templ),
    }


@pytest.fixture
def context():
    return {"ensemble_seed": 42}


class TestSampleMsa:

    def test_samples_msa_to_max_clusters(self, mock_config, sample_features,
                                         context):
        collator = SampleMsa(config=mock_config, keep_extra=True)
        result = collator(sample_features, context)

        assert result["msa"].shape[0] == mock_config.max_msa_clusters

    def test_keeps_extra_msa(self, mock_config, sample_features, context):
        original_msa_count = sample_features["msa"].shape[0]
        collator = SampleMsa(config=mock_config, keep_extra=True)
        result = collator(sample_features, context)

        expected_extra = original_msa_count - mock_config.max_msa_clusters
        assert result["extra_msa"].shape[0] == expected_extra

    def test_discards_extra_msa_when_keep_extra_false(self, mock_config,
                                                      sample_features,
                                                      context):
        collator = SampleMsa(config=mock_config, keep_extra=False)
        result = collator(sample_features, context)

        assert "extra_msa" not in result

    def test_preserves_first_sequence(self, mock_config, sample_features,
                                      context):
        first_seq = sample_features["msa"][0].clone()
        collator = SampleMsa(config=mock_config, keep_extra=True)
        result = collator(sample_features, context)

        assert torch.equal(result["msa"][0], first_seq)


class TestMakeMaskedMsa:

    def test_creates_bert_mask(self, mock_config, sample_features, context):
        collator = MakeMaskedMsa(config=mock_config)
        result = collator(sample_features, context)

        assert "bert_mask" in result
        assert result["bert_mask"].shape == sample_features["msa"].shape

    def test_creates_true_msa(self, mock_config, sample_features, context):
        original_msa = sample_features["msa"].clone()
        collator = MakeMaskedMsa(config=mock_config)
        result = collator(sample_features, context)

        assert "true_msa" in result
        assert torch.equal(result["true_msa"], original_msa)

    def test_modifies_msa(self, mock_config, sample_features, context):
        original_msa = sample_features["msa"].clone()
        collator = MakeMaskedMsa(config=mock_config)
        result = collator(sample_features, context)

        assert not torch.equal(result["msa"], original_msa)

    def test_is_enabled_when_all_params_set(self, mock_config):
        collator = MakeMaskedMsa(config=mock_config)
        assert collator.is_enabled() is True

    def test_is_disabled_when_params_none(self, mock_config):
        collator = MakeMaskedMsa(config=mock_config, profile_prob=None)
        assert collator.is_enabled() is False


class TestNearestNeighborClusters:

    def test_creates_cluster_assignment(self, mock_config, sample_features,
                                        context):
        sample_features["extra_msa"] = torch.randint(0, 22, (512, 64))
        sample_features["extra_msa_mask"] = torch.ones(512, 64)

        collator = NearestNeighborClusters(config=mock_config)
        result = collator(sample_features, context)

        assert "extra_cluster_assignment" in result

    def test_cluster_assignment_shape(self, mock_config, sample_features,
                                      context):
        n_extra = 512
        sample_features["extra_msa"] = torch.randint(0, 22, (n_extra, 64))
        sample_features["extra_msa_mask"] = torch.ones(n_extra, 64)

        collator = NearestNeighborClusters(config=mock_config)
        result = collator(sample_features, context)

        assert result["extra_cluster_assignment"].shape[0] == n_extra

    def test_is_enabled_when_cluster_features_enabled(self, mock_config):
        collator = NearestNeighborClusters(config=mock_config)
        assert collator.is_enabled() is True

    def test_is_disabled_when_cluster_features_disabled(self):
        config = MockConfig(msa_cluster_features=False)
        collator = NearestNeighborClusters(config=config)
        assert collator.is_enabled() is False


class TestSummarizeClusters:

    def test_creates_cluster_profile(self, mock_config, sample_features,
                                     context):
        n_msa = 128
        n_extra = 512
        n_res = 64
        sample_features["msa"] = torch.randint(0, 22, (n_msa, n_res))
        sample_features["msa_mask"] = torch.ones(n_msa, n_res)
        sample_features["deletion_matrix"] = torch.rand(n_msa, n_res)
        sample_features["extra_msa"] = torch.randint(0, 22, (n_extra, n_res))
        sample_features["extra_msa_mask"] = torch.ones(n_extra, n_res)
        sample_features["extra_deletion_matrix"] = torch.rand(n_extra, n_res)
        sample_features["extra_cluster_assignment"] = torch.randint(
            0, n_msa, (n_extra, ))

        collator = SummarizeClusters(config=mock_config)
        result = collator(sample_features, context)

        assert "cluster_profile" in result
        assert "cluster_deletion_mean" in result

    def test_cluster_profile_shape(self, mock_config, sample_features,
                                   context):
        n_msa = 128
        n_extra = 512
        n_res = 64
        sample_features["msa"] = torch.randint(0, 22, (n_msa, n_res))
        sample_features["msa_mask"] = torch.ones(n_msa, n_res)
        sample_features["deletion_matrix"] = torch.rand(n_msa, n_res)
        sample_features["extra_msa"] = torch.randint(0, 22, (n_extra, n_res))
        sample_features["extra_msa_mask"] = torch.ones(n_extra, n_res)
        sample_features["extra_deletion_matrix"] = torch.rand(n_extra, n_res)
        sample_features["extra_cluster_assignment"] = torch.randint(
            0, n_msa, (n_extra, ))

        collator = SummarizeClusters(config=mock_config)
        result = collator(sample_features, context)

        assert result["cluster_profile"].shape == (n_msa, n_res, 23)
        assert result["cluster_deletion_mean"].shape == (n_msa, n_res)


class TestCropExtraMsa:

    def test_crops_extra_msa(self, mock_config, sample_features, context):
        n_extra = 2048
        sample_features["extra_msa"] = torch.randint(0, 22, (n_extra, 64))

        collator = CropExtraMsa(config=mock_config)
        result = collator(sample_features, context)

        assert result["extra_msa"].shape[0] == mock_config.max_extra_msa

    def test_is_enabled_when_max_extra_msa_set(self, mock_config):
        collator = CropExtraMsa(config=mock_config)
        assert collator.is_enabled() is True

    def test_is_disabled_when_max_extra_msa_none(self):
        config = MockConfig(max_extra_msa=None)
        collator = CropExtraMsa(config=config)
        assert collator.is_enabled() is False


class TestDeleteExtraMsa:

    def test_deletes_extra_msa_features(self, mock_config, sample_features,
                                        context):
        sample_features["extra_msa"] = torch.randint(0, 22, (512, 64))
        sample_features["extra_deletion_matrix"] = torch.rand(512, 64)
        sample_features["extra_msa_mask"] = torch.ones(512, 64)

        config = MockConfig(max_extra_msa=None)
        collator = DeleteExtraMsa(config=config)
        result = collator(sample_features, context)

        assert "extra_msa" not in result
        assert "extra_deletion_matrix" not in result
        assert "extra_msa_mask" not in result

    def test_is_enabled_when_max_extra_msa_none(self):
        config = MockConfig(max_extra_msa=None)
        collator = DeleteExtraMsa(config=config)
        assert collator.is_enabled() is True

    def test_is_disabled_when_max_extra_msa_set(self, mock_config):
        collator = DeleteExtraMsa(config=mock_config)
        assert collator.is_enabled() is False


class TestMakeMsaFeat:

    def test_creates_msa_feat(self, mock_config, sample_features, context):
        collator = MakeMsaFeat(config=mock_config)
        result = collator(sample_features, context)

        assert "msa_feat" in result
        assert "target_feat" in result

    def test_msa_feat_shape(self, mock_config, sample_features, context):
        n_msa = sample_features["msa"].shape[0]
        n_res = sample_features["msa"].shape[1]

        collator = MakeMsaFeat(config=mock_config)
        result = collator(sample_features, context)

        assert result["msa_feat"].shape[0] == n_msa
        assert result["msa_feat"].shape[1] == n_res

    def test_target_feat_shape(self, mock_config, sample_features, context):
        n_res = sample_features["aatype"].shape[0]

        collator = MakeMsaFeat(config=mock_config)
        result = collator(sample_features, context)

        assert result["target_feat"].shape[0] == n_res
        assert result["target_feat"].shape[1] == 22


class TestSelectFeat:

    def test_excludes_specified_features(self, mock_config, sample_features,
                                         context):
        exclude = ["msa", "deletion_matrix"]
        collator = SelectFeat(config=mock_config, exclude_feats=exclude)
        result = collator(sample_features, context)

        for key in exclude:
            assert key not in result

    def test_includes_only_specified_features(self, mock_config,
                                              sample_features, context):
        include = ["aatype", "seq_length"]
        collator = SelectFeat(config=mock_config, include_feats=include)
        result = collator(sample_features, context)

        assert set(result.keys()) == set(include)


class TestRandomCropToSize:

    def test_crops_templates(self, mock_config, sample_features, context):
        n_templ = 8
        sample_features["template_aatype"] = torch.randint(
            0, 21, (n_templ, 64))
        sample_features["template_mask"] = torch.ones(n_templ)

        collator = RandomCropToSize(config=mock_config,
                                    subsample_templates=False)
        result = collator(sample_features, context)

        assert result["template_aatype"].shape[0] <= mock_config.max_templates


class TestMakeFixedSize:

    def test_pads_msa_features(self, mock_config, sample_features, context):
        n_msa = 64
        sample_features["msa_mask"] = torch.ones(n_msa, 64)
        sample_features["msa_row_mask"] = torch.ones(n_msa)

        collator = MakeFixedSize(config=mock_config)
        result = collator(sample_features, context)

        assert result["msa_mask"].shape[0] == mock_config.max_msa_clusters
        assert result["msa_row_mask"].shape[0] == mock_config.max_msa_clusters

    def test_pads_template_features(self, mock_config, sample_features,
                                    context):
        n_templ = 2
        sample_features["template_mask"] = torch.ones(n_templ)

        collator = MakeFixedSize(config=mock_config)
        result = collator(sample_features, context)

        assert result["template_mask"].shape[0] == mock_config.max_templates
