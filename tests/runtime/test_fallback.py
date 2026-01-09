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

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import tensorrt_llm_lite
import torch
import torch.nn as nn
from tensorrt_llm_lite import Tensor
from tensorrt_llm_lite._utils import str_dtype_to_trt
from tensorrt_llm_lite.layers.linear import Linear
from tensorrt_llm_lite.logger import logger

from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.runtime.allocator import (BaseContextMemoryManager,
                                                SimpleContextMemoryManager)
from tensorrt_bionemo.runtime.backend import BackendBase


class FallbackConfig(BaseConfig):
    """Configuration for fallback test with need_fallback callable"""
    in_features: int = 128
    out_features: int = 256
    fallback_threshold: int = 128  # If input size exceeds this, fallback to torch

    def get_input_names(self):
        """Return the input tensor names for this config"""
        return ["input"]

    def get_output_names(self):
        """Return the output tensor names for this config"""
        return ["output"]


class TorchLinearModule(nn.Module):
    """Simple PyTorch Linear module for fallback"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.torch_execution_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.torch_execution_count += 1
        return self.linear(x)


class LinearBackendWithFallback(BackendBase):
    """Backend that can fallback to PyTorch based on input size"""
    CONFIG_CLASS = FallbackConfig

    def __init__(
            self,
            config: FallbackConfig,
            context_memory_allocator: Optional[BaseContextMemoryManager] = None
    ):
        super().__init__(config,
                         context_memory_allocator=context_memory_allocator)
        self.trt_dtype = str_dtype_to_trt(config.dtype)
        self.trt_execution_count = 0

    def forward_udf(self, x: torch.Tensor) -> torch.Tensor:
        """TensorRT forward implementation"""
        self.trt_execution_count += 1
        logger.info(
            f"Executing TensorRT forward (count: {self.trt_execution_count})")

        inputs = {"input": x}
        allocator = self._context_memory_allocator
        outputs = allocator.forward(self, inputs)
        return outputs["output"]


def need_fallback_fn(x: torch.Tensor, threshold: int = 16) -> bool:
    """
    Callable that determines if fallback to torch is needed.
    Falls back if batch size exceeds threshold.
    """
    should_fallback = x.shape[0] > threshold
    if should_fallback:
        logger.info(
            f"Fallback triggered: input shape {x.shape[0]} > threshold {threshold}"
        )
    return should_fallback


def create_linear_engine(config: FallbackConfig, engine_name: str):
    """Create a simple TensorRT Linear engine"""
    logger.info(f"Creating TensorRT Linear engine: {engine_name}")
    builder = tensorrt_llm_lite.Builder()
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()
    builder_config = builder.create_builder_config(precision="float32")

    with tensorrt_llm_lite.net_guard(net):
        # Create input tensor
        input = Tensor(name='input',
                       shape=(1, config.in_features),
                       dtype=tensorrt_llm_lite.torch_dtype_to_trt(
                           torch.float32))

        # Create a simple linear layer
        layer = Linear(in_features=config.in_features,
                       out_features=config.out_features)
        output = layer(input)
        output.mark_output("output",
                           tensorrt_llm_lite.torch_dtype_to_trt(torch.float32))

        # Set random weights
        layer.weight.value = np.random.randn(
            config.out_features, config.in_features).astype(np.float32)

    logger.info("Building TensorRT engine...")
    engine_buffer = builder.build_engine(net, builder_config)
    assert engine_buffer is not None
    logger.info("TensorRT engine built successfully")

    # Save engine and config
    engine_dir = f"./tmp/{engine_name}"
    os.makedirs(engine_dir, exist_ok=True)

    engine_path = os.path.join(engine_dir, "rank0.engine")
    with open(engine_path, 'wb') as f:
        f.write(engine_buffer)

    config_path = os.path.join(engine_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config.model_dump(), f)

    logger.info(f"Saved engine to: {engine_dir}")
    return engine_dir


def test_fallback_mechanism():
    """Test that backend correctly falls back to PyTorch when needed"""
    logger.info("=" * 80)
    logger.info("Testing fallback mechanism")
    logger.info("=" * 80)

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    # Configuration
    in_features = 128
    out_features = 256
    fallback_threshold = 1  # Will fallback if batch size > 1

    # Create config with fallback function
    config = FallbackConfig(in_features=in_features,
                            out_features=out_features,
                            fallback_threshold=fallback_threshold)

    # Set the need_fallback callable
    config.need_fallback = lambda x: need_fallback_fn(
        x, threshold=fallback_threshold)

    # Create TensorRT engine
    engine_dir = create_linear_engine(config, "linear_with_fallback")

    # Load config from disk
    config_path = os.path.join(engine_dir, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = FallbackConfig().copy_and_validate(**config_dict)
    config.need_fallback = lambda x: need_fallback_fn(
        x, threshold=fallback_threshold)

    # Create allocator and backend
    allocator = SimpleContextMemoryManager()
    backend = LinearBackendWithFallback.load_weights(
        Path(engine_dir),
        context_memory_allocator=allocator,
        loaded_by_manager=True)
    backend.config = config  # Update config with fallback function

    # Create PyTorch fallback module
    torch_module = TorchLinearModule(in_features, out_features).cuda()
    backend.set_fallback_module(torch_module)

    # Load the TensorRT engine
    allocator.load()
    logger.info("TensorRT engine loaded successfully")

    # Test 1: Small input (should use TensorRT)
    logger.info("\n" + "=" * 80)
    logger.info("Test 1: Small input (batch size = 1) - should use TensorRT")
    logger.info("=" * 80)
    small_input = torch.randn(1, in_features).cuda()
    output_trt = backend(small_input)

    assert output_trt is not None, "TensorRT forward failed"
    assert output_trt.shape == (
        1, out_features), f"Wrong output shape: {output_trt.shape}"
    assert backend.trt_execution_count == 1, f"Expected 1 TensorRT execution, got {backend.trt_execution_count}"
    logger.info(f"✓ TensorRT execution successful! Shape: {output_trt.shape}")
    logger.info(f"  TensorRT executions: {backend.trt_execution_count}")
    logger.info(f"  Torch executions: {torch_module.torch_execution_count}")

    # Test 2: Large input (should fallback to PyTorch)
    logger.info("\n" + "=" * 80)
    logger.info(
        "Test 2: Large input (batch size = 32) - should fallback to PyTorch")
    logger.info("=" * 80)
    large_input = torch.randn(32, 128).cuda()
    output_torch = backend(large_input)

    assert output_torch is not None, "PyTorch forward failed"
    # Output shape will be different since PyTorch module was initialized independently
    assert backend.trt_execution_count == 1, f"TensorRT should not execute again, count: {backend.trt_execution_count}"
    assert torch_module.torch_execution_count == 1, f"PyTorch should execute once, count: {torch_module.torch_execution_count}"
    logger.info(f"✓ PyTorch fallback successful! Shape: {output_torch.shape}")
    logger.info(f"  TensorRT executions: {backend.trt_execution_count}")
    logger.info(f"  Torch executions: {torch_module.torch_execution_count}")

    # Test 3: Edge case - exactly at threshold (should use TensorRT)
    logger.info("\n" + "=" * 80)
    logger.info(
        "Test 3: Edge case (batch_size = threshold) - should use TensorRT")
    logger.info("=" * 80)
    threshold_input = torch.randn(fallback_threshold, in_features).cuda()
    # Create a compatible torch module for this size
    backend.set_fallback_module(torch_module)

    output_threshold = backend(threshold_input)
    assert output_threshold is not None, "Threshold input forward failed"
    assert backend.trt_execution_count == 2, f"Expected 3 TensorRT executions, got {backend.trt_execution_count}"
    assert torch_module.torch_execution_count == 1, f"PyTorch should not execute again, count: {torch_module.torch_execution_count}"
    logger.info(
        f"✓ TensorRT execution successful (at threshold)! Shape: {output_threshold.shape}"
    )
    logger.info(f"  TensorRT executions: {backend.trt_execution_count}")
    logger.info(f"  Torch executions: {torch_module.torch_execution_count}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)
    logger.info(f"Total TensorRT executions: {backend.trt_execution_count}")
    logger.info(
        f"Total Torch fallback executions: {torch_module.torch_execution_count}"
    )
    logger.info("✓ All fallback mechanism tests passed!")

    # Cleanup
    allocator.get_handles().clear()


def test_fallback_without_fallback_module():
    """Test that backend raises error when fallback is needed but no fallback module is set"""
    logger.info("=" * 80)
    logger.info(
        "Testing fallback without fallback module (should fail gracefully)")
    logger.info("=" * 80)

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    # Configuration
    in_features = 256
    out_features = 256
    fallback_threshold = 16

    config = FallbackConfig(in_features=in_features,
                            out_features=out_features,
                            fallback_threshold=fallback_threshold)
    config.need_fallback = lambda x: need_fallback_fn(
        x, threshold=fallback_threshold)

    # Create TensorRT engine
    engine_dir = create_linear_engine(config, "linear_no_fallback_module")

    # Load config
    config_path = os.path.join(engine_dir, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = FallbackConfig().copy_and_validate(**config_dict)
    config.need_fallback = lambda x: need_fallback_fn(
        x, threshold=fallback_threshold)

    # Create backend WITHOUT setting fallback module
    allocator = SimpleContextMemoryManager()
    backend = LinearBackendWithFallback.load_weights(
        Path(engine_dir),
        context_memory_allocator=allocator,
        loaded_by_manager=True)
    backend.config = config
    # NOTE: NOT setting fallback module here: backend.set_fallback_module(...)

    allocator.load()

    # Small input should work (uses TensorRT)
    small_input = torch.randn(1, in_features).cuda()
    output = backend(small_input)
    assert output is not None
    logger.info("✓ Small input worked with TensorRT")

    # Large input should fail (needs fallback but no module set)
    large_input = torch.randn(32, 128).cuda()
    try:
        backend(large_input)
        assert False, "Should have raised error when fallback module is None"
    except (TypeError, AttributeError) as e:
        logger.info(
            f"✓ Correctly raised error when fallback needed but module not set: {type(e).__name__}"
        )

    # Cleanup
    allocator.get_handles().clear()


if __name__ == "__main__":
    test_fallback_mechanism()
    test_fallback_without_fallback_module()
