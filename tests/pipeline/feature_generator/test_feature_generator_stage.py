# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import MagicMock

import pytest

from tensorrt_bionemo.pipeline.stages.configs import FeatureGeneratorStageConfig
from tensorrt_bionemo.pipeline.stages.feature_generator_stage import (
    FeatureGeneratorStage,
    FeatureGeneratorUDF,
)


class TestFeatureGeneratorStageConfiguration:

    def test_stage_has_correct_fn_class(self):
        stage = FeatureGeneratorStage(fn=FeatureGeneratorUDF, fn_constructor_kwargs={
            "feature_generators": [],
        })
        assert stage.fn == FeatureGeneratorUDF

    def test_stage_initialization_with_defaults(self):
        stage = FeatureGeneratorStage(fn=FeatureGeneratorUDF, fn_constructor_kwargs={
            "feature_generators": [],
        })

        assert stage.fn == FeatureGeneratorUDF
        assert stage.compute_by_rows is True
        assert stage.update_row is False

    def test_stage_update_row_default_false(self):
        stage = FeatureGeneratorStage(fn=FeatureGeneratorUDF, fn_constructor_kwargs={
            "feature_generators": [],
        })

        assert stage.update_row is False

    def test_stage_configuration_with_custom_compute(self):
        stage = FeatureGeneratorStage(
            fn=FeatureGeneratorUDF,
            fn_constructor_kwargs={"feature_generators": []},
            map_batches_kwargs={"concurrency": 4},
        )

        assert stage.map_batches_kwargs["concurrency"] == 4


class TestFeatureGeneratorStageConfig:

    def test_default_config(self):
        config = FeatureGeneratorStageConfig()

        assert config.compute is None
        assert config.compute_by_rows is True
        assert config.enabled is True

    def test_config_with_custom_compute(self):
        config = FeatureGeneratorStageConfig(compute=8)

        assert config.compute == 8

    def test_config_with_compute_by_rows_false(self):
        config = FeatureGeneratorStageConfig(compute_by_rows=False)

        assert config.compute_by_rows is False

    def test_config_with_drop_keys(self):
        config = FeatureGeneratorStageConfig(drop_keys=["parsed", "tokenized"])

        assert config.drop_keys == ["parsed", "tokenized"]

    def test_config_disabled(self):
        config = FeatureGeneratorStageConfig(enabled=False)

        assert config.enabled is False

    def test_config_with_batch_size(self):
        config = FeatureGeneratorStageConfig(batch_size=64)

        assert config.batch_size == 64


class TestFeatureGeneratorStageGetDatasetKwargs:

    def test_get_dataset_map_batches_kwargs(self):
        stage = FeatureGeneratorStage(
            fn=FeatureGeneratorUDF,
            fn_constructor_kwargs={"feature_generators": []},
            compute_by_rows=True,
            drop_keys=["parsed"],
        )

        kwargs = stage.get_dataset_map_batches_kwargs(batch_size=32)

        assert kwargs["batch_size"] == 32
        assert kwargs["fn_constructor_kwargs"]["compute_by_rows"] is True
        assert kwargs["fn_constructor_kwargs"]["drop_keys"] == ["parsed"]

    def test_kwargs_include_feature_generators(self):
        mock_generators = [MagicMock(), MagicMock()]

        stage = FeatureGeneratorStage(
            fn=FeatureGeneratorUDF,
            fn_constructor_kwargs={
                "feature_generators": mock_generators,
                "feature_collators": [],
            },
        )

        kwargs = stage.get_dataset_map_batches_kwargs(batch_size=16)

        assert "feature_generators" in kwargs["fn_constructor_kwargs"]
        assert kwargs["fn_constructor_kwargs"]["feature_generators"] == mock_generators

    def test_kwargs_include_feature_collators(self):
        mock_collators = [MagicMock()]

        stage = FeatureGeneratorStage(
            fn=FeatureGeneratorUDF,
            fn_constructor_kwargs={
                "feature_generators": [],
                "feature_collators": mock_collators,
            },
        )

        kwargs = stage.get_dataset_map_batches_kwargs(batch_size=16)

        assert "feature_collators" in kwargs["fn_constructor_kwargs"]
        assert kwargs["fn_constructor_kwargs"]["feature_collators"] == mock_collators

    def test_kwargs_include_pre_init(self):
        def mock_pre_init(context):
            return context

        stage = FeatureGeneratorStage(
            fn=FeatureGeneratorUDF,
            fn_constructor_kwargs={
                "feature_generators": [],
                "pre_init": mock_pre_init,
            },
        )

        kwargs = stage.get_dataset_map_batches_kwargs(batch_size=16)

        assert "pre_init" in kwargs["fn_constructor_kwargs"]
        assert kwargs["fn_constructor_kwargs"]["pre_init"] == mock_pre_init

    def test_kwargs_include_features_merger_func(self):
        def custom_merger(features):
            return features

        stage = FeatureGeneratorStage(
            fn=FeatureGeneratorUDF,
            fn_constructor_kwargs={
                "feature_generators": [],
                "features_merger_func": custom_merger,
            },
        )

        kwargs = stage.get_dataset_map_batches_kwargs(batch_size=16)

        assert "features_merger_func" in kwargs["fn_constructor_kwargs"]


class TestFeatureGeneratorStageWithMockComponents:

    def test_stage_with_mock_feature_generators(self):
        mock_gen1 = MagicMock()
        mock_gen1.name = "generator1"
        mock_gen1.is_enabled.return_value = True

        mock_gen2 = MagicMock()
        mock_gen2.name = "generator2"
        mock_gen2.is_enabled.return_value = True

        generators = [mock_gen1, mock_gen2]

        stage = FeatureGeneratorStage(
            fn=FeatureGeneratorUDF,
            fn_constructor_kwargs={
                "feature_generators": generators,
            },
        )

        assert len(stage.fn_constructor_kwargs["feature_generators"]) == 2

    def test_stage_with_mock_collators(self):
        mock_collator = MagicMock()
        mock_collator.name = "collator1"
        mock_collator.is_enabled.return_value = True

        stage = FeatureGeneratorStage(
            fn=FeatureGeneratorUDF,
            fn_constructor_kwargs={
                "feature_generators": [],
                "feature_collators": [mock_collator],
            },
        )

        assert len(stage.fn_constructor_kwargs["feature_collators"]) == 1


class TestFeatureGeneratorStageUpdateRowBehavior:

    def test_update_row_false_means_replace_mode(self):
        stage = FeatureGeneratorStage(
            fn=FeatureGeneratorUDF,
            fn_constructor_kwargs={"feature_generators": []},
            update_row=False,
        )

        kwargs = stage.get_dataset_map_batches_kwargs(batch_size=16)

        assert kwargs["fn_constructor_kwargs"]["update_row"] is False

    def test_update_row_can_be_overridden(self):
        stage = FeatureGeneratorStage(
            fn=FeatureGeneratorUDF,
            fn_constructor_kwargs={"feature_generators": []},
            update_row=True,
        )

        kwargs = stage.get_dataset_map_batches_kwargs(batch_size=16)

        assert kwargs["fn_constructor_kwargs"]["update_row"] is True
