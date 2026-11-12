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
"""CPU regressions for the constructor-scoped compiler workers."""

import contextlib
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bionemo_ir.dsl_kernels import triton_cache
from bionemo_ir.dsl_kernels.triton_cache import CachedKernel, TritonKernelCache

_ECHO_WORKER = "import sys\nfor line in sys.stdin: print('ok', flush=True)\n"
# Records which worker served each configuration, after a delay that keeps every worker busy.
_LOGGING_WORKER = (
    "import os, sys, time\n"
    "for line in sys.stdin:\n"
    "    time.sleep(0.3)\n"
    "    open(os.environ['FAKE_WORKER_LOG'], 'a').write(f'{os.getpid()}\\n')\n"
    "    print('ok', flush=True)\n"
)

# Replies by kernel name: "bad" fails to compile, "hung" never replies, "garbled"
# breaks the protocol, "crash" exits, and every other name compiles.
_SCRIPTED_WORKER = (
    "import json, os, sys, time\n"
    "for line in sys.stdin:\n"
    "    name = json.loads(line)['fn_name']\n"
    "    open(os.environ['FAKE_WORKER_LOG'], 'a').write(f'{os.getpid()} {name}\\n')\n"
    "    if name == 'hung': time.sleep(60)\n"
    "    if name == 'crash': os._exit(1)\n"
    "    reply = {'bad': 'failed', 'garbled': 'compiled'}.get(name, 'ok')\n"
    "    print(reply, flush=True)\n"
)

# A stand-in kernel for the real worker script: it writes to fd 1 from C and from a
# child process, as compiler code and the tools it starts may, then logs who served it.
_FD1_NOISE_KERNEL = """\
import ctypes, os, subprocess


class _Kernel:
    def warmup(self, *args, grid, **kwargs):
        ctypes.CDLL(None).write(1, b"native note\\n", 12)
        subprocess.run(["echo", "child note"], check=True)
        with open(os.environ["FAKE_WORKER_LOG"], "a") as log:
            log.write(f"{os.getpid()} {kwargs['BLOCK']}\\n")


noisy = _Kernel()
"""
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the worker loads Triton's CUDA driver")


class _FakeCache(TritonKernelCache):
    """Loads a kernel per call; a build without ``maxnreg`` uses 200 registers."""

    def __init__(self) -> None:
        self.loads: list[tuple[str, dict]] = []

    def compile(self, jit_fn, dummy_args, grid, constexpr_kwargs, stream=None) -> CachedKernel:  # noqa: ANN001
        self.loads.append((jit_fn.fn.__name__, dict(constexpr_kwargs)))
        compiled = SimpleNamespace(n_regs=constexpr_kwargs.get("maxnreg", 200), args=dummy_args)
        return CachedKernel(compiled, stream=stream, enable_driver=False)


def _jit(name: str) -> SimpleNamespace:
    return SimpleNamespace(fn=SimpleNamespace(__name__=name, __module__="fake"))


def _compile(cache: _FakeCache, name: str, block: int = 16, **kwargs: object) -> dict:
    return cache.compile_for_dtypes(
        _jit(name), [torch.float32], lambda dtype: (block,), grid=(0,), BLOCK=block, **kwargs
    )


@pytest.fixture(autouse=True)
def _own_request_pool(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give each test its own request-time compiler and stop it afterwards."""
    monkeypatch.setattr(triton_cache, "_request_pool_instance", None)
    yield
    pool = triton_cache._request_pool_instance
    if pool is not None:
        pool.close(load=False)


@pytest.fixture
def cold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An empty Triton cache in a CPU-only process."""
    monkeypatch.setattr(triton_cache, "TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(triton_cache, "_start_triton_driver", lambda: None)
    monkeypatch.setattr(triton_cache, "_triton_target", lambda: {"backend": "cuda", "arch": 100, "warp_size": 32})
    monkeypatch.setattr(triton_cache, "_triton_cache_was_cold", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: "stream")
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    return tmp_path


def _scripted_workers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, limit: int):  # noqa: ANN202
    """Run ``limit`` fake workers that reply by kernel name, and return a reader of what they served."""
    log = tmp_path / "served.log"
    monkeypatch.setenv("FAKE_WORKER_LOG", str(log))
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", _SCRIPTED_WORKER)
    monkeypatch.setattr(triton_cache, "_session_worker_limit", lambda: limit)
    return lambda: [tuple(line.split()) for line in log.read_text().splitlines()] if log.exists() else []


def _processes() -> list[subprocess.Popen]:
    return list(triton_cache._warm_session.pool._processes)


def _markers(cache_dir: Path) -> list[Path]:
    return [path for path in (cache_dir / "bioir_warm").iterdir() if path.suffix != ".lock"]


@contextlib.contextmanager
def _recording_one_shot(monkeypatch: pytest.MonkeyPatch, returncode: int = 0) -> Iterator[list[list[str]]]:
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode, stderr="failed")

    monkeypatch.setattr(subprocess, "run", run)
    yield calls


def test_session_starts_no_process_for_a_warm_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_process(*args: object, **kwargs: object) -> None:
        pytest.fail("A warm cache must not start a compiler")

    monkeypatch.setattr(triton_cache.subprocess, "Popen", unexpected_process)
    monkeypatch.setattr(triton_cache, "_triton_cache_was_cold", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: "stream")
    with triton_cache.triton_warmup_session():
        kernel = _compile(_FakeCache(), "warm")[torch.float32]
        assert type(kernel) is CachedKernel
    assert not hasattr(triton_cache._warm_session, "pool")


def test_cold_configs_compile_in_parallel_and_load_on_first_use(
    cold: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = tmp_path / "workers.log"
    monkeypatch.setenv("FAKE_WORKER_LOG", str(log))
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", _LOGGING_WORKER)
    monkeypatch.setattr(triton_cache, "_session_worker_limit", lambda: 3)
    cache = _FakeCache()
    with triton_cache.triton_warmup_session():
        kernels = [_compile(cache, f"k{index}")[torch.float32] for index in range(4)]
        # The constructor goes on while the workers compile.
        assert all(type(kernel) is triton_cache._PendingKernel for kernel in kernels)
        assert cache.loads == []
        assert kernels[2].compiled.args == (16,)
        assert type(kernels[2]) is CachedKernel
        assert [name for name, _ in cache.loads] == ["k2"]
        pool = triton_cache._warm_session.pool
    # Closing loads the rest in submission order and stops every worker.
    assert [name for name, _ in cache.loads] == ["k2", "k0", "k1", "k3"]
    assert all(type(kernel) is CachedKernel for kernel in kernels)
    assert len(pool._processes) == 3 and all(process.poll() is not None for process in pool._processes)
    assert len(set(log.read_text().split())) == 3
    assert len(_markers(cold)) == 4
    assert not hasattr(triton_cache._warm_session, "pool")


def test_warmed_marker_skips_the_worker(cold: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", _ECHO_WORKER)
    with triton_cache.triton_warmup_session():
        _compile(_FakeCache(), "once")
    # A worker that is still starting would hold the session open; one that served nothing is killed.
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", "import time\ntime.sleep(60)\n")
    start = time.monotonic()
    with triton_cache.triton_warmup_session():
        assert type(_compile(_FakeCache(), "once")[torch.float32]) is CachedKernel
        pool = triton_cache._warm_session.pool
        assert pool._jobs == {}
    assert time.monotonic() - start < 30
    assert pool._processes and all(process.poll() is not None for process in pool._processes)


def test_a_queued_config_compiles_once(cold: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", _ECHO_WORKER)
    cache = _FakeCache()
    with triton_cache.triton_warmup_session():
        first = _compile(cache, "same")[torch.float32]
        second = _compile(cache, "same")[torch.float32]
        assert type(second) is CachedKernel and first is not second
        assert len(triton_cache._warm_session.pool._jobs) == 1


def test_nested_sessions_share_and_release_the_workers(cold: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", _ECHO_WORKER)
    cache = _FakeCache()
    with triton_cache.triton_warmup_session():
        pool = triton_cache._warm_session.pool
        assert _compile(cache, "outer")[torch.float32].driver is None
        with triton_cache.triton_warmup_session():
            assert triton_cache._warm_session.pool is pool
            inner = _compile(cache, "inner")[torch.float32]
        assert type(inner) is triton_cache._PendingKernel
        processes = _processes()
        assert processes and all(process.poll() is None for process in processes)
    assert type(inner) is CachedKernel
    assert all(process.poll() is not None for process in processes)


def test_constructor_error_stops_workers_and_kernels_still_load(cold: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A worker that never replies: the failed constructor must not wait for it.
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", "import sys\nsys.stdin.read()\n")
    cache = _FakeCache()
    kernels = []
    with pytest.raises(ValueError, match="constructor failed"), triton_cache.triton_warmup_session():
        kernels.append(_compile(cache, "orphan")[torch.float32])
        pool = triton_cache._warm_session.pool
        raise ValueError("constructor failed")
    assert pool._processes and all(process.poll() is not None for process in pool._processes)
    assert cache.loads == []
    # A kernel kept past the failure loads through the one-shot path.
    with _recording_one_shot(monkeypatch) as calls:
        assert kernels[0].compiled.args == (16,)
    assert len(calls) == 1 and type(kernels[0]) is CachedKernel


def test_constructor_error_stops_a_worker_waiting_for_a_marker_lock(
    cold: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Another process compiles the same configuration and holds its marker lock throughout.
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", _ECHO_WORKER)
    monkeypatch.setattr(triton_cache, "_WARM_LOCK_TIMEOUT", 30)
    _, marker = triton_cache._warm_spec(_jit("held"), {torch.float32: (16,)}, (0,), {"BLOCK": 16})
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    start = time.monotonic()
    with triton_cache.FileLock(Path(f"{marker}.lock"), exclusive=True):
        with pytest.raises(ValueError, match="constructor failed"), triton_cache.triton_warmup_session():
            _compile(_FakeCache(), "held")
            job = triton_cache._warm_session.pool._jobs[marker]
            while not job.started:
                time.sleep(0.01)
            raise ValueError("constructor failed")
        assert time.monotonic() - start < 10
        assert job.done.is_set() and not os.path.exists(marker)


@pytest.mark.parametrize("started", [0, 1])
def test_a_worker_thread_that_cannot_start_leaves_a_working_session(
    started: int, cold: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The host runs out of threads after ``started`` compiler threads.
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", _ECHO_WORKER)
    monkeypatch.setattr(triton_cache, "_session_worker_limit", lambda: 3)
    thread_start = threading.Thread.start
    attempts = []

    def start(thread: threading.Thread) -> None:
        if thread.name == "bioir-triton-compiler":
            attempts.append(thread)
            if len(attempts) > started:
                raise RuntimeError("can't start new thread")
        thread_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    cache = _FakeCache()
    with _recording_one_shot(monkeypatch) as calls, triton_cache.triton_warmup_session():
        pool = triton_cache._warm_session.pool
        kernel = _compile(cache, "first")[torch.float32]
        _compile(cache, "second")
    assert not hasattr(triton_cache._warm_session, "pool")
    assert len(pool._threads) == started and not any(thread.is_alive() for thread in pool._threads)
    # A failed start is not retried; without any worker, each configuration compiles in a one-shot process.
    # A construction that queued work then starts one request-time worker, which fails here too.
    assert len(attempts) == (started + 2 if started else 1)
    assert len(calls) == (0 if started else 2)
    assert type(kernel) is CachedKernel and [name for name, _ in cache.loads] == ["first", "second"]


@pytest.mark.parametrize("limit", [1, 2])
def test_failed_reply_keeps_the_worker_and_retries_once(
    limit: int, cold: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    served = _scripted_workers(monkeypatch, tmp_path, limit)
    cache = _FakeCache()
    with _recording_one_shot(monkeypatch) as calls, triton_cache.triton_warmup_session():
        bad = _compile(cache, "bad")[torch.float32]
        assert bad.driver is None
        # The failed configuration retries once in a one-shot process, for its diagnostics.
        assert len(calls) == 1
        good = [_compile(cache, f"good{index}")[torch.float32] for index in range(3)]
        assert all(type(kernel) is triton_cache._PendingKernel for kernel in good)
        pool = triton_cache._warm_session.pool
    assert len(calls) == 1
    assert sorted(name for _, name in served()) == ["bad", "good0", "good1", "good2"]
    if limit == 1:
        # The worker that replied "failed" served every later configuration.
        assert len({pid for pid, _ in served()}) == 1
    assert len(pool._processes) == limit


@pytest.mark.parametrize("limit", [1, 2])
def test_timed_out_worker_stops_without_a_one_shot_retry(
    limit: int, cold: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(triton_cache, "_WARM_TIMEOUT", 0.5)
    served = _scripted_workers(monkeypatch, tmp_path, limit)
    cache = _FakeCache()
    start = time.monotonic()
    with _recording_one_shot(monkeypatch) as calls, triton_cache.triton_warmup_session():
        hung = _compile(cache, "hung")[torch.float32]
        # A timed-out configuration compiles in this process, without another timeout in a one-shot process.
        assert hung.compiled.args == (16,)
        assert calls == []
        # Asking for it again shares that fallback, though no worker may be left.
        assert _compile(cache, "hung")[torch.float32].compiled.args == (16,)
        assert calls == []
        pool = triton_cache._warm_session.pool
        # The timed-out worker is stopped at once, not when the session closes.
        deadline = time.monotonic() + 10
        while not any(process.poll() is not None for process in pool._processes) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert sum(process.poll() is not None for process in pool._processes) == 1
        after = _compile(cache, "after")[torch.float32]
        assert after.driver is None
    assert time.monotonic() - start < 30
    if limit == 1:
        # No worker is left, so the rest of the construction compiles one-shot.
        assert len(calls) == 1 and [name for _, name in served()] == ["hung"]
    else:
        # The other worker keeps serving.
        assert calls == [] and sorted(name for _, name in served()) == ["after", "hung"]


def test_a_kernel_read_between_adopt_and_its_class_switch_is_loaded(cold: Path) -> None:
    """Another thread may read a pending kernel after ``_adopt`` replaced its job but before the class switch."""
    cache = _FakeCache()
    job = triton_cache._CompileJob(cache, _jit("race"), lambda dtype: (16,), {torch.float32: (16,)}, (0,), None, {})
    kernel = job.kernels[torch.float32]
    loaded = cache.compile(_jit("race"), (16,), (0,), {})
    # The state _adopt leaves just before it switches the class.
    kernel._stream, kernel._driver = loaded._stream, loaded._driver
    kernel._kernel = loaded._kernel
    assert type(kernel) is triton_cache._PendingKernel
    assert kernel.compiled.args == (16,)
    assert kernel.driver is None
    assert triton_cache._loaded(kernel) is kernel


@pytest.mark.parametrize("name", ["garbled", "crash"])
def test_broken_worker_retries_once_and_stops(
    name: str, cold: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    served = _scripted_workers(monkeypatch, tmp_path, 1)
    cache = _FakeCache()
    with _recording_one_shot(monkeypatch) as calls, triton_cache.triton_warmup_session():
        assert _compile(cache, name)[torch.float32].driver is None
        assert len(calls) == 1
        (process,) = triton_cache._warm_session.pool._processes
        assert process.wait(timeout=10) is not None
        # With its only worker stopped, the rest of this construction compiles one-shot.
        assert type(_compile(cache, "after")[torch.float32]) is CachedKernel
        assert len(calls) == 2
    assert [name for _, name in served()] == [name]


@pytest.mark.parametrize("returncode", [0, 1])
def test_worker_failure_uses_one_shot_and_marks_only_success(
    returncode: int, cold: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", "raise SystemExit(1)")
    with _recording_one_shot(monkeypatch, returncode) as calls, triton_cache.triton_warmup_session():
        _compile(_FakeCache(), "fake")
    assert len(calls) == 1
    assert len(_markers(cold)) == (returncode == 0)


@pytest.mark.parametrize(("session", "was_cold"), [(False, False), (False, True), (True, True)])
def test_register_cap_rebuilds_heavy_kernels(
    session: bool, was_cold: bool, cold: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", _ECHO_WORKER)
    monkeypatch.setattr(triton_cache, "_triton_cache_was_cold", lambda: was_cold)
    cache = _FakeCache()
    with triton_cache.triton_warmup_session() if session else contextlib.nullcontext():
        light = _compile(cache, "light", max_registers=255)[torch.float32]
        heavy = _compile(cache, "heavy", max_registers=128)[torch.float32]
    assert light.compiled.n_regs == 200
    assert heavy.compiled.n_regs == 128
    assert cache.loads == [
        ("light", {"BLOCK": 16}),
        ("heavy", {"BLOCK": 16}),
        ("heavy", {"BLOCK": 16, "maxnreg": 128}),
    ]
    if was_cold and not session:
        # The capped rebuild queues behind its parent on the same request-time worker.
        assert len(_markers(cold)) == 3 and triton_cache._request_pool_instance._jobs == {}


def _request_processes() -> list[subprocess.Popen]:
    return list(triton_cache._request_pool_instance._processes)


def _wait_for(condition, timeout: float = 10) -> None:  # noqa: ANN001
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def test_request_time_configs_share_one_worker(cold: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log = tmp_path / "workers.log"
    monkeypatch.setenv("FAKE_WORKER_LOG", str(log))
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", _LOGGING_WORKER)
    cache = _FakeCache()
    with _recording_one_shot(monkeypatch) as calls:
        kernels = [_compile(cache, f"r{index}")[torch.float32] for index in range(3)]
    # Outside a session each call returns loaded kernels, all compiled by one worker.
    assert all(type(kernel) is CachedKernel for kernel in kernels)
    assert [name for name, _ in cache.loads] == ["r0", "r1", "r2"]
    assert calls == []
    assert len(log.read_text().split()) == 3 and len(set(log.read_text().split())) == 1
    assert len(_markers(cold)) == 3
    # Loaded jobs are dropped; their markers answer any later request.
    assert triton_cache._request_pool_instance._jobs == {}


def test_a_forked_child_starts_its_own_request_worker(cold: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", _ECHO_WORKER)
    _compile(_FakeCache(), "parent")
    parent_pool = triton_cache._request_pool_instance
    (worker,) = _request_processes()
    pid = os.fork()
    if pid == 0:
        ok = triton_cache._request_pool_instance is None
        # The lock is fresh, and the inherited pipe to the parent's worker is gone.
        ok = ok and triton_cache._request_pool_lock.acquire(timeout=5)
        ok = ok and os.path.realpath(f"/proc/self/fd/{worker.stdin.fileno()}") == os.devnull
        os._exit(0 if ok else 1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert triton_cache._request_pool_instance is parent_pool


def test_cold_construction_leaves_one_request_worker(cold: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", _ECHO_WORKER)
    with triton_cache.triton_warmup_session():
        _compile(_FakeCache(), "built")
        session = triton_cache._warm_session.pool
    assert all(process.poll() is not None for process in session._processes)
    # Its thread starts the process.
    _wait_for(_request_processes)
    processes = _request_processes()
    assert len(processes) == 1 and processes[0].poll() is None
    # The first request-time configuration uses the worker that is already up.
    _compile(_FakeCache(), "first-request")
    assert _request_processes() == processes


def test_a_serial_caller_keeps_one_request_worker(cold: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log = tmp_path / "workers.log"
    monkeypatch.setenv("FAKE_WORKER_LOG", str(log))
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", _SCRIPTED_WORKER)
    monkeypatch.setattr(triton_cache, "_session_worker_limit", lambda: 4)
    for index in range(20):
        _compile(_FakeCache(), f"serial{index}")
    # The worker counts as idle before its caller wakes, so no call finds every worker busy.
    assert len({line.split()[0] for line in log.read_text().splitlines()}) == 1


def test_a_broken_request_worker_is_reaped_at_once(cold: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _scripted_workers(monkeypatch, tmp_path, 1)
    with _recording_one_shot(monkeypatch) as calls:
        assert type(_compile(_FakeCache(), "garbled")[torch.float32]) is CachedKernel
    (worker,) = _request_processes()
    _wait_for(lambda: worker.returncode is not None)
    assert len(calls) == 1


def test_concurrent_request_time_configs_compile_in_parallel(
    cold: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = tmp_path / "workers.log"
    monkeypatch.setenv("FAKE_WORKER_LOG", str(log))
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", _LOGGING_WORKER)
    monkeypatch.setattr(triton_cache, "_session_worker_limit", lambda: 2)
    barrier = threading.Barrier(2, timeout=10)

    def request(index: int) -> None:
        barrier.wait()
        _compile(_FakeCache(), f"c{index}")

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(request, range(2)))
    assert len(set(log.read_text().split())) == 2


def test_session_without_cold_configs_leaves_no_request_worker(cold: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", _ECHO_WORKER)
    with triton_cache.triton_warmup_session():
        pass
    assert triton_cache._request_pool_instance is None


def test_a_worker_thread_error_falls_back_instead_of_hanging(cold: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_popen(*args: object, **kwargs: object) -> None:
        raise ValueError("unexpected")

    monkeypatch.setattr(triton_cache.subprocess, "Popen", broken_popen)
    with _recording_one_shot(monkeypatch) as calls:
        first = _compile(_FakeCache(), "first")[torch.float32]
        second = _compile(_FakeCache(), "second")[torch.float32]
    assert type(first) is CachedKernel and type(second) is CachedKernel
    assert len(calls) == 2


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_a_worker_dies_with_its_parent(tmp_path: Path) -> None:
    pid_file = tmp_path / "worker.pid"
    worker = tmp_path / "worker.py"
    worker.write_text(
        triton_cache._PARENT_DEATH_SCRIPT
        + f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\nimport time\ntime.sleep(60)\n"
    )
    # The parent starts a worker that would sleep for a minute, then exits as soon as it runs.
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import os, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, {str(worker)!r}, str(os.getpid())])\n"
        f"while not os.path.exists({str(pid_file)!r}): time.sleep(0.01)\n"
    )
    subprocess.run([sys.executable, str(parent)], check=True, timeout=30)
    pid = int(pid_file.read_text())

    def gone() -> bool:
        try:
            return Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z"
        except OSError:
            return True

    _wait_for(gone)


def test_idle_request_worker_stops_and_the_next_config_starts_another(
    cold: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", _ECHO_WORKER)
    monkeypatch.setattr(triton_cache, "_REQUEST_IDLE_TIMEOUT", 1.0)
    _compile(_FakeCache(), "early")
    (first,) = _request_processes()
    _wait_for(lambda: not _request_processes() and first.poll() is not None)
    assert type(_compile(_FakeCache(), "late")[torch.float32]) is CachedKernel
    (second,) = _request_processes()
    assert second.pid != first.pid


def test_sessions_on_different_threads_have_separate_workers(cold: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(triton_cache, "_WORKER_SCRIPT", _ECHO_WORKER)
    barrier = threading.Barrier(2, timeout=10)

    def construct(index: int) -> int:
        with triton_cache.triton_warmup_session():
            assert _compile(_FakeCache(), f"thread{index}")[torch.float32].driver is None
            pid = _processes()[0].pid
            barrier.wait()
        return pid

    with ThreadPoolExecutor(max_workers=2) as pool:
        pids = list(pool.map(construct, range(2)))
    assert len(set(pids)) == 2


@pytest.fixture
def noisy_kernel(cold: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """The fd-1 writing stand-in, importable by one real worker that logs to ``tmp_path / "worker.log"``."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "fd1_noise_kernel.py").write_text(_FD1_NOISE_KERNEL)
    monkeypatch.setenv("PYTHONPATH", f"{source}:{os.environ.get('PYTHONPATH', '')}")
    monkeypatch.setenv("FAKE_WORKER_LOG", str(tmp_path / "worker.log"))
    monkeypatch.setattr(triton_cache, "_session_worker_limit", lambda: 1)
    return SimpleNamespace(fn=SimpleNamespace(__name__="noisy", __module__="fd1_noise_kernel"))


@needs_cuda
def test_real_worker_replies_past_fd1_output_and_serves_every_config(
    cold: Path, noisy_kernel: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def one_shot(command: list[str], **kwargs: object) -> None:
        pytest.fail("the session worker must compile every configuration itself")

    monkeypatch.setattr(subprocess, "run", one_shot)
    jit_fn = noisy_kernel
    log = tmp_path / "worker.log"
    cache = _FakeCache()
    with triton_cache.triton_warmup_session():
        for block in (16, 32):
            cache.compile_for_dtypes(jit_fn, [torch.float32], lambda dtype, b=block: (b,), BLOCK=block)
        pool = triton_cache._warm_session.pool
    (process,) = pool._processes
    served = [line.split() for line in log.read_text().splitlines()]
    assert served == [[str(process.pid), "16"], [str(process.pid), "32"]]
    assert len(_markers(cold)) == 2
    assert [name for name, _ in cache.loads] == ["noisy", "noisy"]


@needs_cuda
def test_real_worker_keeps_serving_after_a_failed_compile(
    cold: Path, noisy_kernel: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = SimpleNamespace(fn=SimpleNamespace(__name__="kernel", __module__="no_such_kernel_module"))
    cache = _FakeCache()
    with _recording_one_shot(monkeypatch) as calls, triton_cache.triton_warmup_session():
        failed = cache.compile_for_dtypes(missing, [torch.float32], lambda dtype: (16,), BLOCK=16)[torch.float32]
        assert failed.compiled.args == (16,)
        assert len(calls) == 1
        cache.compile_for_dtypes(noisy_kernel, [torch.float32], lambda dtype: (16,), BLOCK=16)
        pool = triton_cache._warm_session.pool
    assert len(calls) == 1
    (process,) = pool._processes
    assert [job.outcome for job in pool._jobs.values()] == [triton_cache._FAILED, triton_cache._WARMED]
    assert (tmp_path / "worker.log").read_text().split() == [str(process.pid), "16"]
