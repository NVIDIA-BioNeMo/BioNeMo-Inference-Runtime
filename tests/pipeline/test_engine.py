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

from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn as nn

from tensorrt_bionemo.pipeline.engine import FoldingEngine

# Determine device based on CUDA availability
TEST_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MockEngineConfig:
    """Mock EngineConfig for testing."""

    def __init__(self):
        self.name = "test_model"
        self.model = Mock()
        self.device = Mock(device=TEST_DEVICE)
        self.postprocessor = Mock()
        self.accelerated = None
        self.profile_inference = False


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self, config=None, model_name=None):
        super().__init__()
        self.config = config
        self.model_name = model_name
        self.linear = nn.Linear(10, 10)

    def forward(self, batch):
        return {"output": torch.randn(2, 3, 10)}

    def optimize(self, accelerated_configs, context_memory_allocator=None):
        return self, None


class MockPostProcessor:
    """Mock postprocessor for testing."""

    def __init__(self, config):
        self.config = config

    def __call__(self, batch, output):
        return {"processed": output}


class TestFoldingEngine:
    """Test suite for FoldingEngine - four essential tests."""

    def test_initialization_creates_model_and_postprocessor(self):
        """Test that FoldingEngine initializes model and postprocessor correctly."""
        config = MockEngineConfig()

        with patch.object(FoldingEngine, 'create_model') as mock_create_model, \
             patch.object(FoldingEngine, 'create_postprocessor') as mock_create_postprocessor:
            engine = FoldingEngine(config, MockModel, MockPostProcessor)

            assert engine.config == config
            assert engine.model_cls == MockModel
            assert engine.postprocessor_cls == MockPostProcessor
            mock_create_model.assert_called_once()
            mock_create_postprocessor.assert_called_once()

    def test_transfer_batch_to_device(self):
        """Test that transfer_batch_to_device correctly moves data to device."""
        config = MockEngineConfig()
        engine = FoldingEngine(config, MockModel)

        # Test with mixed batch data
        batch = {
            "tensor_data": torch.randn(2, 3),
            "numpy_data": np.array([1, 2, 3]),
            "string_data": "test",
            "int_data": 42
        }

        device_batch = engine.transfer_batch_to_device(batch)

        # Verify tensors/arrays are converted
        assert isinstance(device_batch["tensor_data"], torch.Tensor)
        assert isinstance(device_batch["numpy_data"], torch.Tensor)
        assert device_batch["tensor_data"].device.type == TEST_DEVICE

        # Verify non-tensor data is preserved
        assert device_batch["string_data"] == "test"
        assert device_batch["int_data"] == 42

    def test_execute_runs_inference_pipeline(self):
        """Test that execute runs the full inference pipeline."""
        config = MockEngineConfig()
        engine = FoldingEngine(config, MockModel, MockPostProcessor)

        # Create test batch
        batch = {"input": torch.randn(2, 10), "metadata": "test_metadata"}

        # Execute inference
        output = engine.execute(batch)

        # Verify output structure
        assert "processed" in output
        assert "output" in output["processed"]

    def test_create_model_with_accelerated_configs(self):
        """Test model creation with acceleration enabled."""
        config = MockEngineConfig()
        config.accelerated = Mock()  # Enable acceleration

        with patch(
                'tensorrt_bionemo.pipeline.engine.OnDemandContextMemoryManager'
        ) as mock_allocator:
            mock_allocator_instance = Mock()
            mock_allocator.return_value = mock_allocator_instance

            engine = FoldingEngine(config, MockModel)

            # Verify model is created and on correct device
            assert engine.model is not None
            assert next(engine.model.parameters()).device.type == TEST_DEVICE

            # Verify memory allocator was created
            mock_allocator.assert_called_once()
            assert engine.context_memory_allocator == mock_allocator_instance
