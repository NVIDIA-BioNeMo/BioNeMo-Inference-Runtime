# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from tensorrt_bionemo.hubs import FoldingSupportMatrix as SupMat
from tensorrt_bionemo.registry import (ModelRegistry, get_model_class,
                                       register_all_factories)


class MockModel:
    """Mock model class for testing."""


class TestModelRegistry:
    """Test suite for MODEL_REGISTRY functionality - four essential tests."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test - clears the registry."""
        original_registry = ModelRegistry._factories.copy()
        ModelRegistry._factories.clear()
        yield
        ModelRegistry._factories.clear()
        ModelRegistry._factories.update(original_registry)

    def test_register_and_retrieve_model(self):
        """Test basic model registration and retrieval."""
        class MockFactory:
            @classmethod
            def get_model_class(cls):
                return MockModel
        ModelRegistry.register("test_model", MockFactory)

        assert "test_model" in ModelRegistry._factories
        retrieved_class = get_model_class("test_model")
        assert retrieved_class == MockModel

    def test_get_model_class_raises_error_for_nonexistent_model(self):
        """Test that retrieving a non-existing model raises ValueError."""
        with pytest.raises(
                ValueError,
                match="Model nonexistent_model not registered"):
            get_model_class("nonexistent_model")

    def test_register_default_models(self):
        """Test that register_all_factories populates registry correctly."""
        register_all_factories()

        # Verify expected models are registered
        expected_models = [
            SupMat.Boltz1,
            SupMat.Boltz2,
            SupMat.Boltz2Affinity,
            SupMat.OpenFold2_FT2,
            SupMat.AlphaFold2_1,
        ]

        for model_name in expected_models:
            assert model_name in ModelRegistry._factories

    def test_default_models_have_correct_types(self):
        """Test that default models are registered with correct class types."""
        register_all_factories()

        from tensorrt_bionemo.models.boltz1 import Boltz1
        from tensorrt_bionemo.models.boltz2 import Boltz2, Boltz2Affinity
        from tensorrt_bionemo.models.openfold2 import OpenFold2

        assert get_model_class(SupMat.Boltz1) == Boltz1
        assert get_model_class(SupMat.Boltz2) == Boltz2
        assert get_model_class(SupMat.Boltz2Affinity) == Boltz2Affinity
        assert get_model_class(SupMat.OpenFold2_FT2) == OpenFold2
        assert get_model_class(SupMat.AlphaFold2_1) == OpenFold2
