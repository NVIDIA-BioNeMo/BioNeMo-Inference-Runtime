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
"""Tests for the fallback machinery: AutoFallback, TRTFallbackStrategy,
TorchFallbackStrategy, and the BackendBase.forward() routing.

The file is organised into:
  1. Shared test fixtures (mock allocator, configs, backends).
  2. CPU-only unit tests for TRTFallbackStrategy internals.
  3. CPU-only unit tests for AutoFallback dispatcher + BackendBase routing.
  4. GPU integration tests that build real TRT engines (run via __main__).
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.runtime.allocator import (BaseContextMemoryManager,
                                                SimpleContextMemoryManager)
from tensorrt_bionemo.runtime.backend import (AutoFallback, BackendBase,
                                              TRTFallbackStrategy)

# ═════════════════════════════════════════════════════════════════════════
# 1. Shared test fixtures
# ═════════════════════════════════════════════════════════════════════════


class MockHandle:
    """Minimal stand-in for TRTEngineHandle with an optimization profile map."""

    def __init__(self, opt_profile_map=None):
        if opt_profile_map is not None:
            self._opt_profile_map = opt_profile_map


class MockAllocator(BaseContextMemoryManager):
    """Allocator that skips engine deserialization and allows direct handle injection."""

    def __init__(self):
        super().__init__(auto_load=False)

    def load(self):
        pass

    def build_opt_profile_map(self, handle):
        if not hasattr(handle, '_opt_profile_map'):
            handle._opt_profile_map = {}

    def set_deserialized(self, backend, handle):
        self._deserialized_handles[backend] = handle


class SimpleConfig(BaseConfig):
    pass


class TorchFallbackModule(nn.Module):
    """Counts how many times it is invoked as a fallback."""

    def __init__(self):
        super().__init__()
        self.call_count = 0

    def forward(self, *args, **kwargs):
        self.call_count += 1
        return args[0] if args else None


class EvoformerLikeBackend(BackendBase):
    """Mimics EvoformerStackTRT's forward_udf signature (m, z, msa_mask, pair_mask)."""

    CONFIG_CLASS = SimpleConfig

    def __init__(self, config, context_memory_allocator=None):
        super().__init__(config,
                         context_memory_allocator=context_memory_allocator)
        self.udf_call_count = 0

    def forward_udf(self, m: torch.Tensor, z: torch.Tensor,
                    msa_mask: torch.Tensor, pair_mask: torch.Tensor, **kwargs):
        self.udf_call_count += 1
        return m, z


class SingleInputBackend(BackendBase):
    """Backend with a single tensor input named 'x'."""

    CONFIG_CLASS = SimpleConfig

    def __init__(self, config, context_memory_allocator=None):
        super().__init__(config,
                         context_memory_allocator=context_memory_allocator)
        self.udf_call_count = 0

    def forward_udf(self, x: torch.Tensor, **kwargs):
        self.udf_call_count += 1
        return x


def make_profile(min_shape, max_shape, profile_idx=0):
    return {
        'profile_idx': profile_idx,
        'min_shape': min_shape,
        'max_shape': max_shape
    }


SMALL_PROFILE_M = make_profile((1, 8, 2, 4), (1, 8, 16, 4))
SMALL_PROFILE_Z = make_profile((1, 2, 2, 4), (1, 16, 16, 4))
SMALL_PROFILE_MSA_MASK = make_profile((1, 8, 2), (1, 8, 16))
SMALL_PROFILE_PAIR_MASK = make_profile((1, 2, 2), (1, 16, 16))


def _make_evoformer_backend(profile_map=None, backend_type="trt"):
    """Helper: create an EvoformerLikeBackend wired to a MockAllocator + handle."""
    allocator = MockAllocator()
    config = SimpleConfig(backend=backend_type)
    backend = EvoformerLikeBackend(config, context_memory_allocator=allocator)
    if profile_map is not None:
        handle = MockHandle(opt_profile_map=profile_map)
        allocator.set_deserialized(backend, handle)
    return backend, allocator


def _make_trt_strategy(backend, allocator, profile_map):
    """Bind a TRTFallbackStrategy to a backend with a pre-set profile map."""
    handle = MockHandle(opt_profile_map=profile_map)
    allocator.set_deserialized(backend, handle)
    strategy = TRTFallbackStrategy()
    strategy.bind(backend)
    strategy._ensure_profiles()
    return strategy


def _small_tensors(seqlen):
    """Create a set of tiny tensors matching the evoformer-like signature."""
    return (torch.randn(1, 8, seqlen, 4), torch.randn(1, seqlen, seqlen, 4),
            torch.randn(1, 8, seqlen), torch.randn(1, seqlen, seqlen))


# ═════════════════════════════════════════════════════════════════════════
# 2. TRTFallbackStrategy unit tests
# ═════════════════════════════════════════════════════════════════════════

# ── Binding and param name caching ───────────────────────────────────────


def test_trt_strategy_bind():
    """bind() stores backend, allocator, and resolves param names."""
    backend, alloc = _make_evoformer_backend()
    strategy = TRTFallbackStrategy()
    ret = strategy.bind(backend)
    assert ret is strategy
    assert strategy._backend is backend
    assert strategy._allocator is alloc
    assert strategy._param_names == ["m", "z", "msa_mask", "pair_mask"]


def test_trt_strategy_bind_single_input():
    """Param names from a single-input backend."""
    allocator = MockAllocator()
    config = SimpleConfig(backend="trt")
    backend = SingleInputBackend(config, context_memory_allocator=allocator)
    strategy = TRTFallbackStrategy()
    strategy.bind(backend)
    assert strategy._param_names == ["x"]


def test_trt_strategy_rebind_clears_profiles():
    """Rebinding to a different backend resets the cached profile map."""
    alloc = MockAllocator()
    config = SimpleConfig(backend="trt")
    backend_a = EvoformerLikeBackend(config, context_memory_allocator=alloc)
    backend_b = SingleInputBackend(config, context_memory_allocator=alloc)

    strategy = TRTFallbackStrategy()
    strategy.bind(backend_a)
    strategy._opt_profile_map = {"cached": True}

    strategy.bind(backend_b)
    assert strategy._opt_profile_map is None
    assert strategy._param_names == ["x"]


def test_trt_strategy_rebind_same_backend_keeps_cache():
    """Rebinding to the same backend is a no-op."""
    backend, alloc = _make_evoformer_backend()
    strategy = TRTFallbackStrategy()
    strategy.bind(backend)
    strategy._opt_profile_map = {"cached": True}

    strategy.bind(backend)
    assert strategy._opt_profile_map == {"cached": True}


# ── _shape_fits_profiles ─────────────────────────────────────────────────


def test_shape_fits_profiles_direct_match():
    """Shape within profile bounds -> True."""
    backend, alloc = _make_evoformer_backend()
    pm = {"m": [make_profile((1, 8, 2, 4), (1, 8, 16, 4))]}
    strategy = _make_trt_strategy(backend, alloc, pm)
    assert strategy._shape_fits_profiles("m", (1, 8, 10, 4)) is True


def test_shape_fits_profiles_at_min_boundary():
    """Shape exactly at the min boundary -> True."""
    backend, alloc = _make_evoformer_backend()
    pm = {"m": [make_profile((1, 8, 2, 4), (1, 8, 16, 4))]}
    strategy = _make_trt_strategy(backend, alloc, pm)
    assert strategy._shape_fits_profiles("m", (1, 8, 2, 4)) is True


def test_shape_fits_profiles_at_max_boundary():
    """Shape exactly at the max boundary -> True."""
    backend, alloc = _make_evoformer_backend()
    pm = {"m": [make_profile((1, 8, 2, 4), (1, 8, 16, 4))]}
    strategy = _make_trt_strategy(backend, alloc, pm)
    assert strategy._shape_fits_profiles("m", (1, 8, 16, 4)) is True


def test_shape_fits_profiles_exceeds_max():
    """Shape exceeding max profile -> False."""
    backend, alloc = _make_evoformer_backend()
    pm = {"m": [make_profile((1, 8, 2, 4), (1, 8, 16, 4))]}
    strategy = _make_trt_strategy(backend, alloc, pm)
    assert strategy._shape_fits_profiles("m", (1, 8, 20, 4)) is False


def test_shape_fits_profiles_below_min():
    """Shape below min profile -> False."""
    backend, alloc = _make_evoformer_backend()
    pm = {"m": [make_profile((1, 8, 2, 4), (1, 8, 16, 4))]}
    strategy = _make_trt_strategy(backend, alloc, pm)
    assert strategy._shape_fits_profiles("m", (1, 8, 1, 4)) is False


def test_shape_fits_profiles_multiple_profiles():
    """Shape fits a later profile when the first one doesn't match."""
    backend, alloc = _make_evoformer_backend()
    pm = {
        "m": [
            make_profile((1, 8, 2, 4), (1, 8, 8, 4), profile_idx=0),
            make_profile((1, 8, 9, 4), (1, 8, 16, 4), profile_idx=1),
        ]
    }
    strategy = _make_trt_strategy(backend, alloc, pm)
    assert strategy._shape_fits_profiles("m", (1, 8, 5, 4)) is True
    assert strategy._shape_fits_profiles("m", (1, 8, 12, 4)) is True
    assert strategy._shape_fits_profiles("m", (1, 8, 20, 4)) is False


def test_shape_fits_profiles_unknown_tensor():
    """Tensor name absent from profile map -> True (no constraint)."""
    backend, alloc = _make_evoformer_backend()
    pm = {"m": [make_profile((1, 2, 4), (1, 16, 4))]}
    strategy = _make_trt_strategy(backend, alloc, pm)
    assert strategy._shape_fits_profiles("unknown", (99, 99, 99)) is True


def test_shape_fits_profiles_empty_profiles():
    """Empty profile list for the tensor -> True."""
    backend, alloc = _make_evoformer_backend()
    pm = {"m": []}
    strategy = _make_trt_strategy(backend, alloc, pm)
    assert strategy._shape_fits_profiles("m", (1, 10, 4)) is True


def test_shape_fits_profiles_batch_prepend():
    """ndim(tensor) == ndim(profile) - 1: try prepending batch dim = 1."""
    backend, alloc = _make_evoformer_backend()
    pm = {"m": [make_profile((1, 8, 2, 4), (1, 8, 16, 4))]}
    strategy = _make_trt_strategy(backend, alloc, pm)
    assert strategy._shape_fits_profiles("m", (8, 10, 4)) is True
    assert strategy._shape_fits_profiles("m", (8, 20, 4)) is False


def test_shape_fits_profiles_batch_strip():
    """ndim(tensor) == ndim(profile) + 1: try stripping leading batch dim."""
    backend, alloc = _make_evoformer_backend()
    pm = {"m": [make_profile((8, 2, 4), (8, 16, 4))]}
    strategy = _make_trt_strategy(backend, alloc, pm)
    assert strategy._shape_fits_profiles("m", (1, 8, 10, 4)) is True
    assert strategy._shape_fits_profiles("m", (1, 8, 20, 4)) is False


# ── should_fallback ──────────────────────────────────────────────────────


def test_should_fallback_no_deserialized_handle():
    """Returns False when the engine handle has not been deserialized yet."""
    backend, _ = _make_evoformer_backend(profile_map=None)
    strategy = TRTFallbackStrategy()
    strategy.bind(backend)
    assert strategy.should_fallback(torch.randn(1, 8, 10, 4),
                                    torch.randn(1, 10, 10, 4)) is False


def test_should_fallback_empty_profile_map():
    """Returns False when profile map is empty (engine has no input tensors)."""
    backend, _ = _make_evoformer_backend(profile_map={})
    strategy = TRTFallbackStrategy()
    strategy.bind(backend)
    assert strategy.should_fallback(torch.randn(1, 8, 10, 4),
                                    torch.randn(1, 10, 10, 4)) is False


def test_should_fallback_shape_within_profiles():
    """Returns False when all inputs fit within profiles (TRT path)."""
    profile_map = {
        "m": [SMALL_PROFILE_M],
        "z": [SMALL_PROFILE_Z],
        "msa_mask": [SMALL_PROFILE_MSA_MASK],
        "pair_mask": [SMALL_PROFILE_PAIR_MASK],
    }
    backend, alloc = _make_evoformer_backend()
    strategy = _make_trt_strategy(backend, alloc, profile_map)
    assert strategy.should_fallback(*_small_tensors(10)) is False


def test_should_fallback_shape_exceeds_profiles():
    """Returns True when at least one input exceeds all profiles."""
    profile_map = {
        "m": [SMALL_PROFILE_M],
        "z": [SMALL_PROFILE_Z],
    }
    backend, alloc = _make_evoformer_backend()
    strategy = _make_trt_strategy(backend, alloc, profile_map)
    assert strategy.should_fallback(*_small_tensors(20)) is True


def test_should_fallback_one_input_exceeds():
    """Returns True if even a single input exceeds (short-circuits)."""
    profile_map = {
        "m": [SMALL_PROFILE_M],
        "z": [SMALL_PROFILE_Z],
    }
    backend, alloc = _make_evoformer_backend()
    strategy = _make_trt_strategy(backend, alloc, profile_map)

    m = torch.randn(1, 8, 20, 4)
    z = torch.randn(1, 10, 10, 4)
    msa_mask = torch.randn(1, 8, 10)
    pair_mask = torch.randn(1, 10, 10)
    assert strategy.should_fallback(m, z, msa_mask, pair_mask) is True


def test_should_fallback_with_kwargs():
    """Tensor keyword arguments are checked against profiles too."""
    profile_map = {
        "m": [SMALL_PROFILE_M],
        "z": [SMALL_PROFILE_Z],
    }
    backend, alloc = _make_evoformer_backend()
    strategy = _make_trt_strategy(backend, alloc, profile_map)

    m = torch.randn(1, 8, 10, 4)
    assert strategy.should_fallback(m, z=torch.randn(1, 20, 20, 4)) is True


def test_should_fallback_with_batch_dim_adjustment():
    """Profile expects 4-D but forward arg is 3-D: batch dim prepended."""
    profile_map = {"m": [SMALL_PROFILE_M]}
    backend, alloc = _make_evoformer_backend()
    strategy = _make_trt_strategy(backend, alloc, profile_map)

    m_3d = torch.randn(8, 10, 4)
    z = torch.randn(1, 10, 10, 4)
    msa_mask = torch.randn(1, 8, 10)
    pair_mask = torch.randn(1, 10, 10)
    assert strategy.should_fallback(m_3d, z, msa_mask, pair_mask) is False


# ═════════════════════════════════════════════════════════════════════════
# 3. AutoFallback dispatcher + BackendBase.forward() routing tests
# ═════════════════════════════════════════════════════════════════════════


def test_auto_fallback_init():
    """AutoFallback starts unbound with no strategy."""
    af = AutoFallback()
    assert af._backend is None
    assert af._strategy is None


def test_auto_fallback_unbound_returns_false():
    """Unbound AutoFallback must return False (never block the forward path)."""
    af = AutoFallback()
    assert af(torch.randn(1, 2, 3, 4)) is False


def test_auto_fallback_bind_trt():
    """bind() with a TRT backend selects TRTFallbackStrategy."""
    af = AutoFallback()
    backend, _ = _make_evoformer_backend(backend_type="trt")
    ret = af.bind(backend)
    assert af._backend is backend
    assert ret is af
    assert isinstance(af._strategy, TRTFallbackStrategy)


def test_auto_fallback_bind_torch():
    """bind() with a torch backend selects TorchFallbackStrategy (never fallback)."""
    af = AutoFallback()
    allocator = MockAllocator()
    config = SimpleConfig(backend="torch")
    backend = EvoformerLikeBackend(config, context_memory_allocator=allocator)
    af.bind(backend)
    assert af._strategy is not None
    assert af(*_small_tensors(999)) is False


def test_auto_fallback_bind_same_backend_keeps_strategy():
    """Rebinding to the same backend is a no-op."""
    af = AutoFallback()
    backend, _ = _make_evoformer_backend(backend_type="trt")
    af.bind(backend)
    strategy = af._strategy

    af.bind(backend)
    assert af._strategy is strategy


def test_auto_fallback_rebind_different_backend():
    """Rebinding to a different backend creates a new strategy."""
    af = AutoFallback()
    backend_a, _ = _make_evoformer_backend(backend_type="trt")
    af.bind(backend_a)
    strategy_a = af._strategy

    allocator = MockAllocator()
    config = SimpleConfig(backend="trt")
    backend_b = SingleInputBackend(config, context_memory_allocator=allocator)
    af.bind(backend_b)
    assert af._strategy is not strategy_a


# ── BackendBase.forward() routing ────────────────────────────────────────


def test_forward_routes_to_udf_when_shapes_fit():
    """forward() calls forward_udf (TRT) when all shapes fit profiles."""
    profile_map = {"m": [SMALL_PROFILE_M]}
    backend, _ = _make_evoformer_backend(profile_map, backend_type="trt")
    backend.config.need_fallback = AutoFallback()
    fallback = TorchFallbackModule()
    backend.set_fallback_module(fallback)

    backend(*_small_tensors(10))
    assert backend.udf_call_count == 1
    assert fallback.call_count == 0


def test_forward_routes_to_fallback_when_shapes_exceed():
    """forward() calls the fallback module when shapes exceed profiles."""
    profile_map = {"m": [SMALL_PROFILE_M]}
    backend, _ = _make_evoformer_backend(profile_map, backend_type="trt")
    backend.config.need_fallback = AutoFallback()
    fallback = TorchFallbackModule()
    backend.set_fallback_module(fallback)

    backend(*_small_tensors(20))
    assert backend.udf_call_count == 0
    assert fallback.call_count == 1


def test_forward_auto_binds_on_first_call():
    """forward() lazily binds the AutoFallback to the backend."""
    profile_map = {"m": [SMALL_PROFILE_M]}
    backend, _ = _make_evoformer_backend(profile_map, backend_type="trt")
    af = AutoFallback()
    backend.config.need_fallback = af

    assert af._backend is None
    backend(*_small_tensors(10))
    assert af._backend is backend


def test_forward_mixed_trt_and_fallback_sequence():
    """Multiple forward calls alternate between TRT and fallback correctly."""
    profile_map = {"m": [SMALL_PROFILE_M]}
    backend, _ = _make_evoformer_backend(profile_map, backend_type="trt")
    backend.config.need_fallback = AutoFallback()
    fallback = TorchFallbackModule()
    backend.set_fallback_module(fallback)

    backend(*_small_tensors(10))
    assert backend.udf_call_count == 1
    assert fallback.call_count == 0

    backend(*_small_tensors(20))
    assert backend.udf_call_count == 1
    assert fallback.call_count == 1

    backend(*_small_tensors(5))
    assert backend.udf_call_count == 2
    assert fallback.call_count == 1

    backend(*_small_tensors(16))
    assert backend.udf_call_count == 3
    assert fallback.call_count == 1

    backend(*_small_tensors(17))
    assert backend.udf_call_count == 3
    assert fallback.call_count == 2


def test_forward_default_auto_fallback_when_none():
    """When need_fallback is None, forward() uses default AutoFallback."""
    backend, _ = _make_evoformer_backend(profile_map={}, backend_type="trt")
    backend.config.need_fallback = None

    backend(*_small_tensors(10))
    assert backend.udf_call_count == 1


def test_forward_custom_callable_still_works():
    """Existing custom need_fallback callables still work alongside AutoFallback."""
    backend, _ = _make_evoformer_backend(profile_map={}, backend_type="trt")
    fallback = TorchFallbackModule()
    backend.set_fallback_module(fallback)
    backend.config.need_fallback = lambda m, *args, **kwargs: m.shape[-2] > 16

    backend(*_small_tensors(10))
    assert backend.udf_call_count == 1
    assert fallback.call_count == 0

    backend(*_small_tensors(20))
    assert backend.udf_call_count == 1
    assert fallback.call_count == 1


# ═════════════════════════════════════════════════════════════════════════
# 4. GPU integration tests (real TRT engine)
# ═════════════════════════════════════════════════════════════════════════


class FallbackConfig(BaseConfig):
    in_features: int = 128
    out_features: int = 256
    fallback_threshold: int = 128

    def get_input_names(self):
        return ["input"]

    def get_output_names(self):
        return ["output"]


class TorchLinearModule(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.torch_execution_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.torch_execution_count += 1
        return self.linear(x)


class LinearBackendWithFallback(BackendBase):
    CONFIG_CLASS = FallbackConfig

    def __init__(
            self,
            config: FallbackConfig,
            context_memory_allocator: Optional[BaseContextMemoryManager] = None
    ):
        super().__init__(config,
                         context_memory_allocator=context_memory_allocator)
        from tensorrt_llm_lite._utils import str_dtype_to_trt
        self.trt_dtype = str_dtype_to_trt(config.dtype)
        self.trt_execution_count = 0

    def forward_udf(self, x: torch.Tensor) -> torch.Tensor:
        self.trt_execution_count += 1
        inputs = {"input": x}
        allocator = self._context_memory_allocator
        outputs = allocator.forward(self, inputs)
        return outputs["output"]


def need_fallback_fn(x: torch.Tensor, threshold: int = 16) -> bool:
    should_fallback = x.shape[0] > threshold
    return should_fallback


def create_linear_engine(config: FallbackConfig, engine_name: str):
    import tensorrt_llm_lite
    from tensorrt_llm_lite import Tensor
    from tensorrt_llm_lite.layers.linear import Linear

    builder = tensorrt_llm_lite.Builder()
    net = builder.create_network()
    net.plugin_config.to_legacy_setting()
    builder_config = builder.create_builder_config(precision="float32")

    with tensorrt_llm_lite.net_guard(net):
        input = Tensor(name='input',
                       shape=(1, config.in_features),
                       dtype=tensorrt_llm_lite.torch_dtype_to_trt(
                           torch.float32))
        layer = Linear(in_features=config.in_features,
                       out_features=config.out_features)
        output = layer(input)
        output.mark_output("output",
                           tensorrt_llm_lite.torch_dtype_to_trt(torch.float32))
        layer.weight.value = np.random.randn(
            config.out_features, config.in_features).astype(np.float32)

    engine_buffer = builder.build_engine(net, builder_config)
    assert engine_buffer is not None

    engine_dir = f"./tmp/{engine_name}"
    os.makedirs(engine_dir, exist_ok=True)

    with open(os.path.join(engine_dir, "rank0.engine"), 'wb') as f:
        f.write(engine_buffer)
    with open(os.path.join(engine_dir, "config.json"), 'w') as f:
        json.dump(config.model_dump(), f)

    return engine_dir


def test_gpu_fallback_mechanism():
    """GPU test: backend correctly falls back to PyTorch when needed."""
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    in_features = 128
    out_features = 256
    fallback_threshold = 1

    config = FallbackConfig(in_features=in_features,
                            out_features=out_features,
                            fallback_threshold=fallback_threshold)
    config.need_fallback = lambda x: need_fallback_fn(
        x, threshold=fallback_threshold)

    engine_dir = create_linear_engine(config, "linear_with_fallback")

    config_path = os.path.join(engine_dir, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = FallbackConfig().copy_and_validate(**config_dict)
    config.need_fallback = lambda x: need_fallback_fn(
        x, threshold=fallback_threshold)

    allocator = SimpleContextMemoryManager()
    backend = LinearBackendWithFallback.load_weights(
        Path(engine_dir),
        context_memory_allocator=allocator,
        loaded_by_manager=True)
    backend.config = config

    torch_module = TorchLinearModule(in_features, out_features).cuda()
    backend.set_fallback_module(torch_module)
    allocator.load()

    # Small input -> TRT
    small_input = torch.randn(1, in_features).cuda()
    output_trt = backend(small_input)
    assert output_trt is not None
    assert output_trt.shape == (1, out_features)
    assert backend.trt_execution_count == 1

    # Large input -> fallback to torch
    large_input = torch.randn(32, 128).cuda()
    output_torch = backend(large_input)
    assert output_torch is not None
    assert backend.trt_execution_count == 1
    assert torch_module.torch_execution_count == 1

    # At threshold -> TRT
    threshold_input = torch.randn(fallback_threshold, in_features).cuda()
    output_threshold = backend(threshold_input)
    assert output_threshold is not None
    assert backend.trt_execution_count == 2
    assert torch_module.torch_execution_count == 1

    allocator.get_handles().clear()


def test_gpu_fallback_without_fallback_module():
    """GPU test: backend raises error when fallback needed but no module set."""
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

    in_features = 256
    out_features = 256
    fallback_threshold = 16

    config = FallbackConfig(in_features=in_features,
                            out_features=out_features,
                            fallback_threshold=fallback_threshold)
    config.need_fallback = lambda x: need_fallback_fn(
        x, threshold=fallback_threshold)

    engine_dir = create_linear_engine(config, "linear_no_fallback_module")

    config_path = os.path.join(engine_dir, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = FallbackConfig().copy_and_validate(**config_dict)
    config.need_fallback = lambda x: need_fallback_fn(
        x, threshold=fallback_threshold)

    allocator = SimpleContextMemoryManager()
    backend = LinearBackendWithFallback.load_weights(
        Path(engine_dir),
        context_memory_allocator=allocator,
        loaded_by_manager=True)
    backend.config = config
    allocator.load()

    small_input = torch.randn(1, in_features).cuda()
    output = backend(small_input)
    assert output is not None

    large_input = torch.randn(32, 128).cuda()
    try:
        backend(large_input)
        assert False, "Should have raised error when fallback module is None"
    except (TypeError, AttributeError):
        pass

    allocator.get_handles().clear()
