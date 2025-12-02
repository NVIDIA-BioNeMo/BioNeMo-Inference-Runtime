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

import gc
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import tensorrt_llm
import torch
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.layers.linear import Linear
from tensorrt_llm.profiler import device_memory_info, host_memory_info

from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.runtime.allocator import (BaseContextMemoryManager,
                                                OnDemandContextMemoryManager,
                                                SharedContextMemoryManager,
                                                SimpleContextMemoryManager)
from tensorrt_bionemo.runtime.backend import BackendBase

# Set up logger for tests
logger = logging.getLogger(__name__)


def get_memory_usage():
    pid = os.getpid()
    host_used, _, _ = host_memory_info(pid)
    device_used, _, _ = device_memory_info()
    # Convert bytes to MB for easier reading
    host_used_mb = host_used / (1024 * 1024)
    device_used_mb = device_used / (1024 * 1024)
    return host_used_mb, device_used_mb


def save_engine_buffer_to_disk(engine_buffer, config, engine_dir):
    """Helper function to save engine buffer and config to disk"""
    logger.info(f"Creating directory: {engine_dir}")
    os.makedirs(engine_dir, exist_ok=True)
    logger.info(f"Directory created: {os.path.exists(engine_dir)}")

    # Save engine
    engine_path = os.path.join(engine_dir, "rank0.engine")
    logger.info(f"Saving engine to: {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(engine_buffer)
    logger.info(f"Engine saved: {os.path.exists(engine_path)}")

    # Save config
    config_path = os.path.join(engine_dir, "config.json")
    logger.info(f"Saving config to: {config_path}")
    with open(config_path, 'w') as f:
        json.dump(config.model_dump(), f)
    logger.info(f"Config saved: {os.path.exists(config_path)}")

    logger.info(f"Returning engine_dir: {engine_dir}")
    return engine_dir


class DummyConfig(BaseConfig):
    in_features: int = 256
    out_features: int = 256

    def get_input_names(self):
        """Return the input tensor names for this config"""
        return ["input"]

    def get_output_names(self):
        """Return the output tensor names for this config"""
        return ["output"]


class DummyBackend(BackendBase):
    CONFIG_CLASS = DummyConfig

    def __init__(
            self,
            config: BaseConfig,
            context_memory_allocator: Optional[BaseContextMemoryManager] = None
    ):
        # Pass context_memory_allocator to parent, handling None case

        super().__init__(config,
                         context_memory_allocator=context_memory_allocator)
        self.trt_dtype = str_dtype_to_trt(config.dtype)

    def forward(self, x: torch.Tensor):
        # Use the allocator from the base class
        inputs = {"input": x}
        # Cast to memory manager since it has the forward method
        allocator = self._context_memory_allocator

        outputs = allocator.forward(self, inputs)
        return outputs


def create_dummy_engine(config: DummyConfig, layer_configs: list,
                        engine_name: str):
    """ Create a dummy TensorRT engine with the specified layer configuration """
    logger.info(f"Starting create_dummy_engine: {engine_name}...")
    builder = tensorrt_llm.Builder()
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()
    builder_config = builder.create_builder_config(precision="float32")

    with tensorrt_llm.net_guard(net):
        # Create input tensor
        input = Tensor(name='input',
                       shape=(1, config.in_features),
                       dtype=tensorrt_llm.torch_dtype_to_trt(torch.float32))

        # Create layers based on configuration
        layers = []
        for i, (in_features, out_features) in enumerate(layer_configs):
            layer = Linear(in_features=in_features, out_features=out_features)
            layers.append(layer)

        # Chain the layers
        x = input
        for layer in layers:
            x = layer(x)
        output = x

        output.mark_output("output",
                           tensorrt_llm.torch_dtype_to_trt(torch.float32))

        # Set random weights for each layer
        for i, (layer, (in_features,
                        out_features)) in enumerate(zip(layers, layer_configs)):
            layer.weight.value = np.random.randn(out_features,
                                                 in_features).astype(np.float32)

    logger.info("Building engine...")
    engine_buffer = builder.build_engine(net, builder_config)
    assert engine_buffer is not None
    logger.info("Engine built successfully")

    # Save engine and config
    engine_dir = f"./tmp/{engine_name}"
    return save_engine_buffer_to_disk(engine_buffer, config, engine_dir)


def create_dummy_net0(config: DummyConfig):
    """Create a simple 2-layer network"""
    layer_configs = [
        (config.in_features, config.out_features),  # layer_0
        (config.out_features, 1)  # layer_1
    ]
    return create_dummy_engine(config, layer_configs, "net0")


def create_dummy_net1(config: DummyConfig):
    """Create a simple 3-layer network"""
    layer_configs = [
        (config.in_features, config.out_features),  # layer_0
        (config.out_features, config.out_features // 2),  # layer_1
        (config.out_features // 2, 1)  # layer_2
    ]
    return create_dummy_engine(config, layer_configs, "net1")


def create_dummy_net2(config: DummyConfig):
    """Create a complex network with many layers and large dimensions"""
    # Use much larger dimensions to require substantial memory
    large_dim = config.out_features * 4  # 4x larger dimensions

    layer_configs = [
        (config.in_features, large_dim),  # layer_0: 256 -> 1024
        (large_dim, large_dim),  # layer_1: 1024 -> 1024
        (large_dim, large_dim),  # layer_2: 1024 -> 1024
        (large_dim, large_dim),  # layer_3: 1024 -> 1024
        (large_dim, large_dim),  # layer_4: 1024 -> 1024
        (large_dim, large_dim),  # layer_5: 1024 -> 1024
        (large_dim, large_dim),  # layer_6: 1024 -> 1024
        (large_dim, large_dim),  # layer_7: 1024 -> 1024
        (large_dim, large_dim),  # layer_8: 1024 -> 1024
        (large_dim, large_dim),  # layer_9: 1024 -> 1024
        (large_dim, large_dim // 2),  # layer_10: 1024 -> 512
        (large_dim // 2, large_dim // 4),  # layer_11: 512 -> 256
        (large_dim // 4, 1)  # layer_12: 256 -> 1
    ]
    return create_dummy_engine(config, layer_configs, "net2")


def run_allocator_test(allocator_class: BaseContextMemoryManager,
                       allocator_name: str):
    """ Generic test function for both SimpleContextMemoryManager and SharedContextMemoryManager """
    # 0. Get memory usage at start
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    start_host_memory, start_device_memory = get_memory_usage()
    logger.info(
        f"{allocator_name} Test - Memory usage at start - Host: {start_host_memory} MB, Device: {start_device_memory} MB"
    )

    # 1. Create multiple engines with different configurations
    logger.info(
        f"Creating multiple dummy engines for {allocator_name.lower()} test...")

    # Create engines with different configurations
    engine_dir_0 = create_dummy_net0(
        DummyConfig(in_features=512, out_features=512))  # 2-layer network
    engine_dir_1 = create_dummy_net1(
        DummyConfig(in_features=512, out_features=512))  # 3-layer network
    engine_dir_2 = create_dummy_net2(
        DummyConfig(in_features=512, out_features=512))  # 13-layer network

    # Load configs for all engines
    configs = []
    engine_dirs = [engine_dir_0, engine_dir_1, engine_dir_2]

    for i, engine_dir in enumerate(engine_dirs):
        config_path = os.path.join(engine_dir, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = DummyConfig().copy_and_validate(**config_dict)
        configs.append(config)
        logger.info(f"Loaded config for engine {i}")

    # Create allocator and backends, register all backends
    allocator = allocator_class()
    backends = []

    for i, (config, engine_dir) in enumerate(zip(configs, engine_dirs)):
        backend = DummyBackend.load_weights(Path(engine_dir),
                                            context_memory_allocator=allocator,
                                            loaded_by_manager=True)
        backends.append(backend)
        logger.info(
            f"Registered backend {i} with {allocator_name.lower()} allocator")

    # Load all engines by manager
    logger.info(
        f"Loading all engines with {allocator_name.lower()} memory manager...")
    allocator.load()
    logger.info(
        f"Loaded {len(allocator.get_handles())} engines with {allocator_name.lower()} memory"
    )

    # Assert that engines were loaded successfully
    assert len(allocator.get_handles()
               ) == 3, f"Expected 3 engines, got {len(allocator.get_handles())}"

    # Test forward pass through each engine
    logger.info(
        f"Testing forward passes through all engines with {allocator_name.lower()} memory..."
    )
    for i, backend in enumerate(backends):
        input_tensor = torch.randn(1, 512).cuda()
        logger.info(
            f"Running forward pass for engine {i} with {allocator_name.lower()} memory"
        )
        outputs = allocator.forward(backend, {"input": input_tensor})

        # Assert that forward pass was successful
        assert outputs is not None, f"Engine {i} forward pass returned None"
        assert "output" in outputs, f"Engine {i} output missing 'output' key, got keys: {list(outputs.keys())}"
        assert outputs["output"].shape == (
            1, 1
        ), f"Engine {i} output shape incorrect, expected (1,1), got {outputs['output'].shape}"

        logger.info(
            f"Engine {i} forward pass successful! Output keys: {list(outputs.keys())}"
        )
    torch.cuda.synchronize()
    # Get memory usage after loading engines
    end_host_memory, end_device_memory = get_memory_usage()
    logger.info(
        f"{allocator_name} Test - Memory usage after loading - Host: {end_host_memory} MB, Device: {end_device_memory} MB"
    )
    # Cleanup
    logger.info(f"Cleaning up {allocator_name.lower()} test...")
    allocator.get_handles().clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"{allocator_name} test cleanup successful!")


def test_simple_context_memory_manager():
    run_allocator_test(SimpleContextMemoryManager, "Simple")


def test_shared_context_memory_manager():
    run_allocator_test(SharedContextMemoryManager, "Shared")


def test_ondemand_context_memory_manager():
    """Test the OnDemandContextMemoryManager with memory allocation on demand"""
    logger.info("Testing OnDemand memory usage...")

    # Get initial memory usage
    start_host_memory, start_device_memory = get_memory_usage()
    logger.info(
        f"OnDemand Test - Memory usage at start - Host: {start_host_memory} MB, Device: {start_device_memory} MB"
    )

    # Create multiple engines with different configurations
    engine_dir_0 = create_dummy_net0(
        DummyConfig(in_features=512, out_features=512))
    engine_dir_1 = create_dummy_net1(
        DummyConfig(in_features=512, out_features=512))
    engine_dir_2 = create_dummy_net2(
        DummyConfig(in_features=512, out_features=512))

    # Load configs for all engines
    configs = []
    engine_dirs = [engine_dir_0, engine_dir_1, engine_dir_2]
    for engine_dir in engine_dirs:
        config_path = os.path.join(engine_dir, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = DummyConfig().copy_and_validate(**config_dict)
        configs.append(config)

    # Create allocator and backends
    allocator = OnDemandContextMemoryManager()
    backends = []
    for config, engine_dir in zip(configs, engine_dirs):
        backend = DummyBackend.load_weights(Path(engine_dir),
                                            context_memory_allocator=allocator,
                                            loaded_by_manager=True)
        backends.append(backend)

    # Load engines (should NOT allocate memory)
    allocator.load()

    # Track peak memory usage during forward passes
    peak_memory_usage = 0
    input_tensor = torch.randn(1, 512).cuda()

    # Test forward pass through each engine and measure peak memory
    for i, backend in enumerate(backends):
        logger.info(f"Running forward pass for engine {i}")

        # Measure memory before forward pass
        memory_before_bytes = device_memory_info()[0]
        memory_before = memory_before_bytes / (1024 * 1024)  # Convert to MB

        # Run forward pass (this should allocate → use → deallocate)
        outputs = allocator.forward(backend, {"input": input_tensor})

        # Synchronize CUDA to ensure deallocation completes
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Clear PyTorch cache to see actual memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Measure memory after forward pass
        memory_after_bytes = device_memory_info()[0]
        memory_after = memory_after_bytes / (1024 * 1024)  # Convert to MB

        # Calculate memory used during this forward pass (all values are in MB)
        memory_used = memory_after - start_device_memory
        peak_memory_usage = max(peak_memory_usage, memory_used)

        logger.info(
            f"Engine {i} - Memory before: {memory_before} MB, Memory after: {memory_after} MB, Memory used: {memory_used} MB"
        )

        # Verify outputs
        assert outputs is not None, f"Engine {i} forward pass returned None"
        assert "output" in outputs, f"Engine {i} output missing 'output' key"
        assert outputs["output"].shape == (
            1, 1), f"Engine {i} output shape incorrect"

    # Final cleanup and synchronization
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Get final memory usage
    end_host_memory, end_device_memory = get_memory_usage()
    final_memory_diff = end_device_memory - start_device_memory

    logger.info(
        f"OnDemand Test - Final memory usage - Host: {end_host_memory} MB, Device: {end_device_memory} MB"
    )
    logger.info(
        f"OnDemand peak memory usage during execution: {peak_memory_usage} MB")
    logger.info(f"OnDemand final memory difference: {final_memory_diff} MB")

    # Assert that peak memory usage is reasonable (should be > 0 but less than sum of all engines)
    assert peak_memory_usage > 0, f"OnDemand should use some memory during execution, got: {peak_memory_usage} MB"

    # Cleanup
    logger.info("Cleaning up OnDemand test...")
    allocator.get_handles().clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    logger.info("OnDemand test cleanup successful!")
    logger.info(f"OnDemand peak memory usage: {peak_memory_usage} MB")


def test_custom_stream():
    """Test using Custom CUDA streams with the allocators"""
    logger.info("Testing Custom CUDA streams...")

    # Create custom CUDA streams for each engine
    stream_0 = torch.cuda.Stream()
    stream_1 = torch.cuda.Stream()
    stream_2 = torch.cuda.Stream()
    logger.info(f"Created custom streams: {stream_0}, {stream_1}, {stream_2}")

    # Get the CUDA stream handles (integers) for TensorRT
    stream_handle_0 = stream_0.cuda_stream
    stream_handle_1 = stream_1.cuda_stream
    stream_handle_2 = stream_2.cuda_stream
    logger.info(
        f"CUDA stream handles: {stream_handle_0}, {stream_handle_1}, {stream_handle_2}"
    )

    # Get initial memory usage
    start_host_memory, start_device_memory = get_memory_usage()
    logger.info(
        f"Custom Stream Test - Memory usage at start - Host: {start_host_memory} MB, Device: {start_device_memory} MB"
    )

    # Create all three engines
    engine_dir_0 = create_dummy_net0(
        DummyConfig(in_features=512, out_features=512))
    engine_dir_1 = create_dummy_net1(
        DummyConfig(in_features=512, out_features=512))
    engine_dir_2 = create_dummy_net2(
        DummyConfig(in_features=512, out_features=512))

    # Load configs for all engines
    configs = []
    engine_dirs = [engine_dir_0, engine_dir_1, engine_dir_2]
    streams = [stream_0, stream_1, stream_2]
    stream_handles = [stream_handle_0, stream_handle_1, stream_handle_2]

    for i, engine_dir in enumerate(engine_dirs):
        config_path = os.path.join(engine_dir, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = DummyConfig().copy_and_validate(**config_dict)
        configs.append(config)
        logger.info(f"Loaded config for engine {i}")

    # Test with SharedContextMemoryManager and custom streams
    allocator = SharedContextMemoryManager()
    backends = []

    # Create backends with different streams
    for i, (config, engine_dir, stream, stream_handle) in enumerate(
            zip(configs, engine_dirs, streams, stream_handles)):
        # Pass the custom stream handle to the allocator
        backend = DummyBackend.load_weights(Path(engine_dir),
                                            context_memory_allocator=allocator,
                                            stream=stream_handle,
                                            loaded_by_manager=True)
        backends.append(backend)
        logger.info(
            f"Registered backend {i} with custom stream handle {stream_handle}")

    # Load all engines
    allocator.load()
    logger.info(
        f"Loaded {len(allocator.get_handles())} engines with custom streams")

    # Assert that engines were loaded successfully
    assert len(allocator.get_handles()
               ) == 3, f"Expected 3 engines, got {len(allocator.get_handles())}"

    # Test forward pass through each engine with its custom stream
    input_tensor = torch.randn(1, 512).cuda()

    for i, (backend, stream,
            stream_handle) in enumerate(zip(backends, streams, stream_handles)):
        logger.info(
            f"Running forward pass for engine {i} with custom stream handle {stream_handle}"
        )

        # Run forward pass with custom stream using the allocator
        with torch.cuda.stream(stream):
            outputs = allocator.forward(backend, {"input": input_tensor})

        # Synchronize the custom stream
        stream.synchronize()

        # Verify outputs
        assert outputs is not None, f"Engine {i} forward pass returned None"
        assert "output" in outputs, f"Engine {i} output missing 'output' key"
        assert outputs["output"].shape == (
            1, 1
        ), f"Engine {i} output shape incorrect, expected (1,1), got {outputs['output'].shape}"

        logger.info(
            f"Engine {i} forward pass successful with stream handle {stream_handle}"
        )

    # Get final memory usage
    end_host_memory, end_device_memory = get_memory_usage()
    device_memory_diff = end_device_memory - start_device_memory

    logger.info(
        f"Custom Stream Test - Final memory usage - Host: {end_host_memory} MB, Device: {end_device_memory} MB"
    )
    logger.info(
        f"Custom Stream Test - Memory difference: {device_memory_diff} MB")

    # Assert that allocator allocated memory
    assert device_memory_diff > 0, f"Custom Stream allocator should allocate some device memory, got: {device_memory_diff} MB"

    # Cleanup
    logger.info("Cleaning up Custom stream test...")
    allocator.get_handles().clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    logger.info("Custom stream test completed successfully!")
    logger.info(f"Custom stream test memory usage: {device_memory_diff} MB")


def test_optimization_profile_switching():
    """Test auto-switching optimization profiles functionality"""
    torch.cuda.set_device(0)
    logger.info("Testing optimization profile switching...")

    # Create a simple engine with multiple optimization profiles
    config = DummyConfig(in_features=256, out_features=256)
    builder = tensorrt_llm.Builder()
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()
    builder_config = builder.create_builder_config(precision="float32")

    with tensorrt_llm.net_guard(net):
        # Create input tensor with dynamic shape
        input = Tensor(name='input',
                       shape=(1, -1),
                       dtype=tensorrt_llm.torch_dtype_to_trt(torch.float32))

        output = input + 0.0  # addition to create a new tensor
        output.mark_output("output",
                           tensorrt_llm.torch_dtype_to_trt(torch.float32))

    # Create optimization profiles for different sequence lengths
    for profile_idx in range(3):
        profile = builder.trt_builder.create_optimization_profile()

        # Define different sequence length ranges for each profile
        min_seqlen = 64 + profile_idx * 32  # 64, 96, 128
        max_seqlen = 96 + profile_idx * 32  # 96, 128, 160

        # Set input shapes for this profile
        profile.set_shape("input",
                          min=(1, min_seqlen),
                          opt=(1, max_seqlen),
                          max=(1, max_seqlen))
        builder_config.trt_builder_config.add_optimization_profile(profile)

    # Build engine
    engine_buffer = builder.build_engine(net, builder_config)
    assert engine_buffer is not None

    # Save engine
    engine_dir = "./tmp/opt_profile_test"
    engine_path = os.path.join(engine_dir, "rank0.engine")
    os.makedirs(engine_dir, exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(engine_buffer)

    # Save config
    config_path = os.path.join(engine_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f)

    # Test with SimpleContextMemoryManager
    allocator = SimpleContextMemoryManager()
    backend = DummyBackend.load_weights(Path(engine_dir),
                                        context_memory_allocator=allocator,
                                        loaded_by_manager=True)
    allocator.load()

    # Test different sequence lengths to trigger profile switching
    test_cases = [
        (70, "short sequence"),  # Should use profile 0 (64-96)
        (110, "medium sequence"),  # Should use profile 1 (96-128)
        (140, "long sequence"),  # Should use profile 2 (128-160)
    ]

    for seqlen, description in test_cases:
        logger.info(f"Testing {description} with length {seqlen}")
        input_tensor = torch.randn(1, seqlen).cuda()

        # Get profile info before forward pass
        profile_info_before = allocator.get_opt_profile_info(backend)
        logger.info(f"Profile before: {profile_info_before['current_profile']}")
        logger.info(
            f"Available profiles: {profile_info_before['num_profiles']}")

        # Run forward pass (should auto-switch profile)
        outputs = allocator.forward(backend, {"input": input_tensor})

        # Get profile info after forward pass
        profile_info_after = allocator.get_opt_profile_info(backend)
        logger.info(f"Profile after: {profile_info_after['current_profile']}")

        # Verify output
        assert outputs is not None, f"Forward pass failed for {description}"
        assert "output" in outputs, f"Missing output for {description}"
        assert outputs["output"].shape == (
            1, seqlen
        ), f"Wrong output shape for {description}, expected (1, {seqlen}), got {outputs['output'].shape}"

        logger.info(f"✓ {description} test passed")

    # Cleanup
    allocator.get_handles().clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Optimization profile switching test completed successfully!")
