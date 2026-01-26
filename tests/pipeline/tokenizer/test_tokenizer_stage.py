# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from tensorrt_bionemo.pipeline.stages.configs import TokenizerStageConfig
from tensorrt_bionemo.pipeline.stages.tokenizer_stage import (
    TokenizerStage,
    TokenizerUDF,
)


class TestTokenizerStageConfiguration:

    def test_stage_has_correct_fn_class(self):
        stage = TokenizerStage(fn=TokenizerUDF, fn_constructor_kwargs={
            "context_generators": {},
        })
        assert stage.fn == TokenizerUDF

    def test_required_input_keys(self):
        stage = TokenizerStage(fn=TokenizerUDF, fn_constructor_kwargs={
            "context_generators": {},
        })
        required_keys = stage.get_required_input_keys()

        assert "parsed" in required_keys
        assert isinstance(required_keys["parsed"], str)

    def test_stage_initialization_with_defaults(self):
        stage = TokenizerStage(fn=TokenizerUDF, fn_constructor_kwargs={
            "context_generators": {},
        })

        assert stage.fn == TokenizerUDF
        assert stage.compute_by_rows is True

    def test_stage_configuration_with_custom_compute(self):
        stage = TokenizerStage(
            fn=TokenizerUDF,
            fn_constructor_kwargs={"context_generators": {}},
            map_batches_kwargs={"concurrency": 4},
        )

        assert stage.map_batches_kwargs["concurrency"] == 4


class TestTokenizerStageConfig:

    def test_default_config(self):
        config = TokenizerStageConfig()

        assert config.compute is None
        assert config.compute_by_rows is True
        assert config.enabled is True

    def test_config_with_custom_compute(self):
        config = TokenizerStageConfig(compute=8)

        assert config.compute == 8

    def test_config_with_compute_by_rows_false(self):
        config = TokenizerStageConfig(compute_by_rows=False)

        assert config.compute_by_rows is False

    def test_config_with_drop_keys(self):
        config = TokenizerStageConfig(drop_keys=["temp_data", "debug_info"])

        assert config.drop_keys == ["temp_data", "debug_info"]

    def test_config_disabled(self):
        config = TokenizerStageConfig(enabled=False)

        assert config.enabled is False


class TestTokenizerStageGetDatasetKwargs:

    def test_get_dataset_map_batches_kwargs(self):
        stage = TokenizerStage(
            fn=TokenizerUDF,
            fn_constructor_kwargs={"context_generators": {}},
            compute_by_rows=True,
            drop_keys=["temp_data"],
        )

        kwargs = stage.get_dataset_map_batches_kwargs(batch_size=32)

        assert kwargs["batch_size"] == 32
        assert kwargs["fn_constructor_kwargs"]["compute_by_rows"] is True
        assert kwargs["fn_constructor_kwargs"]["drop_keys"] == ["temp_data"]

    def test_get_dataset_map_batches_kwargs_with_expected_keys(self):
        stage = TokenizerStage(
            fn=TokenizerUDF,
            fn_constructor_kwargs={"context_generators": {}},
        )

        kwargs = stage.get_dataset_map_batches_kwargs(batch_size=16)

        assert "expected_input_keys" in kwargs["fn_constructor_kwargs"]
        assert kwargs["fn_constructor_kwargs"]["expected_input_keys"] == ["parsed"]

    def test_kwargs_include_context_generators(self):
        from unittest.mock import MagicMock
        mock_generators = {"primary": MagicMock(), "msa": MagicMock()}

        stage = TokenizerStage(
            fn=TokenizerUDF,
            fn_constructor_kwargs={
                "context_generators": mock_generators,
                "transform_funcs": [],
            },
        )

        kwargs = stage.get_dataset_map_batches_kwargs(batch_size=16)

        assert "context_generators" in kwargs["fn_constructor_kwargs"]
        assert kwargs["fn_constructor_kwargs"]["context_generators"] == mock_generators


class TestTokenizerStageWithContextGenerators:

    def test_stage_with_mock_context_generators(self):
        from unittest.mock import MagicMock

        mock_primary = MagicMock()
        mock_primary.required_kwargs = []

        mock_msa = MagicMock()
        mock_msa.required_kwargs = ["parsed"]

        generators = {"primary": mock_primary, "msa": mock_msa}

        stage = TokenizerStage(
            fn=TokenizerUDF,
            fn_constructor_kwargs={
                "context_generators": generators,
            },
        )

        assert stage.fn_constructor_kwargs["context_generators"] == generators

