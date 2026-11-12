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
"""Compiler lifetime tests for direct and pipeline model construction on CPU."""

import contextlib
import inspect
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bionemo_ir.dsl_kernels import triton_cache
from bionemo_ir.dsl_kernels.triton_cache import CachedKernel, TritonKernelCache
from bionemo_ir.models.optimize_module_setter import OptimizedModuleSetterMixin
from bionemo_ir.pipeline.engine import FoldingEngine

_ECHO_WORKER = "import sys\nfor line in sys.stdin: print('ok', flush=True)\n"


class _FakeCache(TritonKernelCache):
    def compile(self, jit_fn, dummy_args, grid, constexpr_kwargs, stream=None) -> CachedKernel:  # noqa: ANN001
        return CachedKernel(SimpleNamespace(args=dummy_args), stream=stream, enable_driver=False)


def _compile_cold(name: str) -> tuple[CachedKernel, list[subprocess.Popen]]:
    """Queue one cold configuration in the active session; return its kernel and the session's workers."""
    jit_fn = SimpleNamespace(fn=SimpleNamespace(__name__=name, __module__="fake"))
    kernel = _FakeCache().compile_for_dtypes(jit_fn, [torch.float32], lambda dtype: (1,), BLOCK=16)[torch.float32]
    assert kernel.driver is None
    return kernel, list(triton_cache._warm_session.pool._processes)


@pytest.fixture
def cold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", _ECHO_WORKER)
    monkeypatch.setattr(triton_cache, "TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(triton_cache, "_start_triton_driver", lambda: None)
    monkeypatch.setattr(triton_cache, "_triton_target", lambda: {"backend": "cuda", "arch": 100, "warp_size": 32})
    monkeypatch.setattr(triton_cache, "_triton_cache_was_cold", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: "stream")
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())


@pytest.mark.usefixtures("cold")
def test_new_model_and_subclass_share_workers_until_outer_constructor_returns() -> None:
    pools = []
    processes = []

    class Model(nn.Module, OptimizedModuleSetterMixin):
        def __init__(self, value: float = 3.0) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(value))
            pools.append(triton_cache._warm_session.pool)
            processes.extend(_compile_cold("model")[1])

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            assert not hasattr(triton_cache._warm_session, "pool")
            return value * self.weight

    class Child(Model):
        def __init__(self, value: float = 3.0) -> None:
            super().__init__(value)
            pools.append(triton_cache._warm_session.pool)
            _compile_cold("child")
            assert all(process.poll() is None for process in processes)

    model = Child(value=4.0)
    assert pools[0] is pools[1]
    assert processes and all(process.poll() is not None for process in processes)
    assert not hasattr(triton_cache._warm_session, "pool")
    torch.testing.assert_close(model(torch.tensor(2.0)), torch.tensor(8.0), rtol=0, atol=0)
    assert str(inspect.signature(Child)) == "(value: float = 3.0) -> None"
    assert set(model.state_dict()) == {"weight"}


def test_inherited_constructor_on_warm_cache_starts_no_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_process(*args: object, **kwargs: object) -> None:
        pytest.fail("A warm constructor must not start a compiler")

    monkeypatch.setattr(triton_cache.subprocess, "Popen", unexpected_process)
    monkeypatch.setattr(triton_cache, "_triton_cache_was_cold", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: "stream")

    class Model(nn.Module, OptimizedModuleSetterMixin):
        def __init__(self) -> None:
            super().__init__()
            jit_fn = SimpleNamespace(fn=SimpleNamespace(__name__="warm", __module__="fake"))
            _FakeCache().compile_for_dtypes(jit_fn, [torch.float32], lambda dtype: (1,), BLOCK=16)
            assert triton_cache._warm_session.pool._processes == []

    class Child(Model):
        pass

    Child()
    assert not hasattr(triton_cache._warm_session, "pool")


def test_constructor_inherited_from_outside_the_mixin_is_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(triton_cache, "_triton_cache_was_cold", lambda: False)
    pools = []

    class Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            pools.append(getattr(triton_cache._warm_session, "pool", None))

    class Model(Base, OptimizedModuleSetterMixin):
        pass

    Model()
    Base()
    assert pools[0] is not None and pools[1] is None
    assert not hasattr(triton_cache._warm_session, "pool")


@pytest.mark.usefixtures("cold")
def test_new_model_constructor_failure_releases_workers() -> None:
    processes = []

    class Model(nn.Module, OptimizedModuleSetterMixin):
        def __init__(self) -> None:
            super().__init__()
            processes.extend(_compile_cold("failing")[1])
            raise ValueError("model failed")

    with pytest.raises(ValueError, match="model failed"):
        Model()
    assert processes and all(process.poll() is not None for process in processes)
    assert not hasattr(triton_cache._warm_session, "pool")


@pytest.mark.usefixtures("cold")
@pytest.mark.parametrize("fail", [False, True])
def test_engine_construction_opens_one_session_through_the_mixin(fail: bool) -> None:
    pools = []
    processes = []

    class Model(nn.Module, OptimizedModuleSetterMixin):
        def __init__(self, config: object, model_name: str) -> None:
            super().__init__()
            pools.append(triton_cache._warm_session.pool)
            processes.extend(_compile_cold("engine")[1])
            if fail:
                raise ValueError("model failed")

        def forward(self, batch: dict) -> dict:
            assert not hasattr(triton_cache._warm_session, "pool")
            return batch

    engine = FoldingEngine.__new__(FoldingEngine)
    engine.model_cls = Model
    engine.model_name = "future-model"
    engine.model_config = None
    engine.device_config = SimpleNamespace(device="cpu")
    engine.accelerated_configs = None
    if fail:
        with pytest.raises(ValueError, match="model failed"):
            engine.create_model()
    else:
        model = engine.create_model()
        value = torch.tensor(2.0)
        assert model({"value": value})["value"] is value
    assert len(pools) == 1
    assert processes and all(process.poll() is not None for process in processes)
    assert not hasattr(triton_cache._warm_session, "pool")
