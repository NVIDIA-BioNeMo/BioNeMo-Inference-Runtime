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

"""Compile Triton kernels once and launch them with low Python overhead.

``CachedKernel`` prefers a direct ``cuda.bindings`` CUBIN launcher and falls
back to ``CompiledKernel.run``. ``TritonKernelCache`` isolates the disk cache
and warms a cold cache in a subprocess.

Usage::

    class MyKernel(TritonKernelCache):
        def __init__(self):
            self.kernels = self.compile_for_dtypes(
                my_triton_jit_fn,
                dtypes=[torch.bfloat16],
                make_dummy_args=lambda dtype: (
                    torch.empty(1, dtype=dtype, device="cuda"),
                ),
                grid=(0,),
                BLOCK_SIZE=128,
            )

    kernel = MyKernel().kernels[torch.bfloat16]
    if kernel.driver is not None:
        kernel.driver.launch_with((tensor.data_ptr(),), 1)
    else:
        kernel.launch((1,), tensor, 128)
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import select
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from .cache_base import DiskCache, DriverLauncher, FileLock, KernelCacheBase, make_driver_launcher

# Resolve the isolated cache path before importing Triton.


def _resolve_triton_cache_dir() -> str:
    """Resolve the Triton cache directory."""
    explicit = os.getenv("BIOIR_TRITON_CACHE_DIR")
    if explicit:
        return explicit
    if os.getenv("TRITON_CACHE_DIR"):
        return os.environ["TRITON_CACHE_DIR"]
    return str(DiskCache.get_cache_dir() / "triton")


TRITON_CACHE_DIR: str = _resolve_triton_cache_dir()
os.environ.setdefault("TRITON_CACHE_DIR", TRITON_CACHE_DIR)

import torch  # noqa: E402
import triton  # noqa: E402
from triton.runtime.jit import JITFunction, compute_cache_key  # noqa: E402

logger = logging.getLogger(__name__)

# Triton 3.6 moved ``knobs`` to the top-level package.
_knobs = None
try:
    _knobs = triton.knobs  # triton >= 3.6
except AttributeError:
    try:
        from triton.runtime import knobs as _knobs  # triton 3.5
    except Exception:
        _knobs = None

if _knobs is not None:
    ENTER_HOOK = getattr(_knobs.runtime, "launch_enter_hook", None)
    EXIT_HOOK = getattr(_knobs.runtime, "launch_exit_hook", None)
else:
    ENTER_HOOK = EXIT_HOOK = None

# The direct launcher reads Triton's private CompiledKernel ABI, so only validated
# versions get it; keep in step with the ``triton`` bound in pyproject.toml. An
# omitted version still runs, just without the fast path.
_SUPPORTED_TRITON_VERSIONS: tuple[tuple[int, int], ...] = ((3, 6), (3, 7))


def _triton_supports_driver() -> bool:
    """True only when the installed Triton exposes a validated CompiledKernel ABI."""
    raw = getattr(triton, "__version__", "0.0.0")
    try:
        major, minor = (int(p) for p in raw.split(".")[:2])
    except (ValueError, AttributeError):
        return False
    return (major, minor) in _SUPPORTED_TRITON_VERSIONS


_DRIVER_TRITON_OK: bool = _triton_supports_driver()


def value_specialized_params(jit_fn: JITFunction, dummy_args: Sequence[Any]) -> list[str] | None:
    """Return scalar parameters Triton may specialize on their compile-time value.

    ``DriverLauncher`` reuses one CUBIN for every runtime shape, so the CUBIN must
    not depend on the values the dummy arguments held. Triton specializes a
    non-constexpr scalar two ways, both fatal for a reused CUBIN: ``value % 16 == 0``
    adds a divisible_by_16 alignment assumption that a non-multiple runtime value
    violates, and ``value == 1`` folds the argument away entirely, shifting every
    later ``params[i]`` assignment. ``do_not_specialize`` suppresses both. Pointers
    are exempt -- their marker reflects allocator alignment, which always holds.

    Args:
        jit_fn: The ``@triton.jit`` function being compiled.
        dummy_args: Positional compile arguments; tensors mark pointer parameters.

    Returns:
        Names of unprotected scalar parameters, empty when the CUBIN is safe to
        reuse, or ``None`` when the parameters cannot be introspected.
    """
    params = getattr(jit_fn, "params", None)
    if not params or len(params) < len(dummy_args):
        return None
    unprotected: list[str] = []
    for param, arg in zip(params, dummy_args, strict=False):
        if param.is_constexpr or isinstance(arg, torch.Tensor):
            continue
        if not getattr(param, "do_not_specialize", False):
            unprotected.append(param.name)
    return unprotected


class CachedKernel:
    """Compiled Triton kernel with direct-driver, ``.run()``, and JIT launch paths."""

    __slots__ = ("_kernel", "_stream", "_driver")

    def __init__(self, compiled_kernel: Any, stream: torch.cuda.Stream | None = None, enable_driver: bool = True):
        self._kernel = compiled_kernel
        self._stream = stream or torch.cuda.current_stream()
        self._driver: DriverLauncher | None = None

        # Fall back safely when Triton's private ABI differs.
        if enable_driver and not _DRIVER_TRITON_OK:
            enable_driver = False

        if enable_driver:
            try:
                from .cache_base import _HAS_CUDA_BINDINGS, _drv

                cu_stream = None
                if _HAS_CUDA_BINDINGS:
                    cu_stream = _drv.CUstream(self._stream.cuda_stream)
                self._driver = make_driver_launcher(compiled_kernel, cu_stream=cu_stream)
            except Exception:
                self._driver = None

    @property
    def driver(self) -> DriverLauncher | None:
        """``DriverLauncher`` for the cuda.bindings fast path, or None."""
        return self._driver

    @property
    def compiled(self) -> Any:
        """The underlying Triton ``CompiledKernel``."""
        return self._kernel

    def launch(self, grid: tuple[int, ...], *args: Any) -> None:
        """Launch the kernel.

        A ``JITFunction`` goes through Triton's dispatch, which recompiles per
        specialization. A compiled CUBIN uses Triton's C-level ``.run()``.

        Args:
            grid: (grid_x,) or (grid_x, grid_y) or (grid_x, grid_y, grid_z).
            *args: Kernel arguments in declaration order, including constexprs.
        """
        kernel = self._kernel
        if isinstance(kernel, JITFunction):
            kernel[grid](*args)
            return
        gs = len(grid)
        gx = grid[0]
        gy = grid[1] if gs > 1 else 1
        gz = grid[2] if gs > 2 else 1
        # Use the active stream, including during CUDA graph capture.
        stream = torch.cuda.current_stream().cuda_stream
        lm = kernel.launch_metadata(grid, stream, *args)
        kernel.run(gx, gy, gz, stream, kernel.function, kernel.packed_metadata, lm, ENTER_HOOK, EXIT_HOOK, *args)


_was_cold: bool | None = None


def _triton_cache_was_cold() -> bool:
    """Return the process-start cache state, latched on first use."""
    global _was_cold
    if _was_cold is None:
        cache_dir = TRITON_CACHE_DIR
        if not os.path.isdir(cache_dir):
            _was_cold = True
        else:
            try:
                _was_cold = len(os.listdir(cache_dir)) == 0
            except OSError:
                _was_cold = True
        if _was_cold:
            logger.info(
                "Triton disk cache is cold (%s) — kernel compilations "
                "will use subprocess to avoid in-process degradation",
                cache_dir,
            )
    return _was_cold


_WARM_TIMEOUT = 120
# A waiter outlasts one peer's compile; a worker or one-shot process holds the lock for one.
_WARM_LOCK_TIMEOUT = _WARM_TIMEOUT + 60

_WARM_SCRIPT = textwrap.dedent("""\
import functools, json, os, sys

@functools.cache
def compile_only_driver():
    # Warmup compiles and returns before any launch, so this process needs no
    # device, stream or CUDA context. It compiles for the target the parent
    # names and never initializes CUDA, which would contend with the parent.
    from triton.backends.nvidia.driver import CudaDriver
    from triton.runtime import driver

    cuda = CudaDriver()
    cuda.get_current_stream = lambda device=None: 0
    driver.set_active(cuda)
    return cuda

def warm(spec):
    # Match the parent process's cache before importing Triton.
    os.environ["TRITON_CACHE_DIR"] = spec["triton_cache_dir"]
    import importlib, torch
    from triton.backends.compiler import GPUTarget

    cuda = compile_only_driver()
    if spec["target"] is not None:
        target = GPUTarget(**spec["target"])
        cuda.get_current_target = lambda: target
        # Triton keys its per-device caches, target included, by this id.
        cuda.get_current_device = lambda: target.arch

    mod = importlib.import_module(spec["module"])
    jit_fn = getattr(mod, spec["fn_name"])
    grid = tuple(spec["grid"])
    ckw = spec["constexpr_kwargs"]

    for variant in spec["variants"]:
        args = []
        for a in variant:
            if a["type"] == "tensor":
                # Warmup takes a dtype as an aligned placeholder tensor, as the parent's allocations are.
                args.append(getattr(torch, a["dtype"]))
            else:
                args.append(a["value"])
        jit_fn.warmup(*args, grid=grid, **ckw)
""")

_SUBPROCESS_SCRIPT = _WARM_SCRIPT + textwrap.dedent("""\
with open(sys.argv[1]) as f:
    warm(json.load(f))
""")

# A worker outlives its last configuration only while its parent lives. Stdin EOF
# ends it between configurations; on Linux the kernel also kills it mid-compile
# when the parent thread that started it exits. ``argv[1]`` is the parent's pid.
_PARENT_DEATH_SCRIPT = textwrap.dedent("""\
import os, sys

if sys.platform.startswith("linux"):
    import ctypes, signal

    ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    if os.getppid() != int(sys.argv[1]):
        os._exit(0)
""")

_WORKER_SCRIPT = (
    _PARENT_DEATH_SCRIPT
    + _WARM_SCRIPT
    + textwrap.dedent("""\
import traceback

# Reply on a private copy of fd 1 and point fd 1 at stderr: native compiler
# code and the tools it starts write to fd 1, and must not reach the parent.
reply = os.fdopen(os.dup(1), "wb", buffering=0)
os.dup2(2, 1)

# Load Triton's driver while the parent constructs the model.
compile_only_driver()

for line in sys.stdin:
    try:
        warm(json.loads(line))
    except Exception:
        traceback.print_exc(file=sys.stderr)
        reply.write(b"failed\\n")
    else:
        reply.write(b"ok\\n")
# Every reply follows a complete cache write, so finalization has nothing to save.
os._exit(0)
""")
)

# Each worker is a full interpreter with Torch and Triton. Cold first output, OpenFold3/Protenix, 1/4/27 workers:
# B200 23.4/19.0/21.9 and 26.4/22.9/25.0 s, B300 43.6/39.5/43.2 and 44.8/39.1/44.0 s; 2 or 8 land within noise of 4.
_MAX_WORKERS = 4


def _session_worker_limit() -> int:
    """Workers one construction may run, leaving a CPU for the constructor."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    return max(1, min(_MAX_WORKERS, cpus - 1))


# Outcomes of one worker request. A worker that replies "failed" keeps serving;
# a timed-out or broken one stops. A failed or broken compile retries once in a
# one-shot process for its diagnostics; a timed-out one would only time out again.
_WARMED, _FAILED, _TIMED_OUT, _BROKEN = "warmed", "failed", "timed out", "broken"


def _request_warm(process: subprocess.Popen[bytes], spec: dict) -> str:
    """Send one configuration to a worker and wait for its bounded reply; return the outcome."""
    name = f"{spec.get('module')}.{spec.get('fn_name')}"
    try:
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("its pipes are unavailable")
        process.stdin.write(json.dumps(spec).encode() + b"\n")
        process.stdin.flush()
        deadline = time.monotonic() + _WARM_TIMEOUT
        response = b""
        while b"\n" not in response:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                logger.warning("Triton compiler worker timed out after %s s on %s; stopping it", _WARM_TIMEOUT, name)
                return _TIMED_OUT
            chunk = os.read(process.stdout.fileno(), 32)
            if not chunk or len(response) + len(chunk) > 32:
                raise RuntimeError("its reply is unavailable")
            response += chunk
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("Triton compiler worker stopped on %s; using a one-shot subprocess: %s", name, exc)
        return _BROKEN
    if response == b"ok\n":
        return _WARMED
    if response == b"failed\n":
        logger.warning("Triton compiler worker failed to compile %s; retrying in a one-shot subprocess", name)
        return _FAILED
    logger.warning("Triton compiler worker sent an unreadable reply on %s: %r; stopping it", name, response)
    return _BROKEN


def _stop_worker(process: subprocess.Popen[bytes], timeout: float = 5) -> None:
    """Wait for a worker whose stdin is closed, killing it after ``timeout``."""
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    finally:
        if process.stdout is not None:
            process.stdout.close()


class _CompileJob:
    """One cold configuration: compiled by a session worker, loaded by the constructor on first use."""

    def __init__(
        self,
        cache: TritonKernelCache,
        jit_fn: JITFunction,
        make_dummy_args: Callable[[torch.dtype], tuple],
        dummy_args_by_dtype: dict[torch.dtype, tuple],
        grid: tuple[int, ...],
        max_registers: int | None,
        constexpr_kwargs: dict,
    ) -> None:
        self.cache = cache
        self.jit_fn = jit_fn
        self.make_dummy_args = make_dummy_args
        self.dummy_args_by_dtype = dummy_args_by_dtype
        self.grid = grid
        self.max_registers = max_registers
        self.constexpr_kwargs = constexpr_kwargs
        self.spec: dict = {}
        self.marker = ""
        self.device = torch.cuda.current_device()
        self.stream = torch.cuda.current_stream()
        self.kernels = {dtype: _PendingKernel(self) for dtype in dummy_args_by_dtype}
        self.pool: _CompilerPool | None = None
        self.started = False
        self.outcome = _FAILED
        self.done = threading.Event()
        self._load_lock = threading.RLock()
        self._loaded = False

    def finish(self, outcome: str) -> None:
        self.outcome = outcome
        self.done.set()

    def load(self) -> None:
        """Wait for the worker, then load each dtype from the disk cache in this process."""
        with self._load_lock:
            if self._loaded:
                return
            pool = self.pool
            if pool is not None:
                pool.wait(self)
            if self.outcome in (_FAILED, _BROKEN):
                # The worker could not warm it: take the path a constructor outside a session takes.
                _compile_in_subprocess(self.jit_fn, self.dummy_args_by_dtype, self.grid, self.constexpr_kwargs)
            # A timed-out configuration compiles below in this process, as one whose one-shot compile times out does.
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                loaded = {
                    dtype: self.cache.compile(self.jit_fn, dummy_args, self.grid, self.constexpr_kwargs, self.stream)
                    for dtype, dummy_args in self.dummy_args_by_dtype.items()
                }
                if self.max_registers is not None:
                    self.cache._cap_registers(
                        loaded, self.jit_fn, self.make_dummy_args, self.grid, self.max_registers, self.constexpr_kwargs
                    )
            for dtype, pending in self.kernels.items():
                pending._adopt(loaded[dtype])
            self._loaded = True
            self.dummy_args_by_dtype = {}


def _loaded(kernel: CachedKernel) -> CachedKernel:
    """Return ``kernel`` after loading it if a session worker still holds its compile."""
    if isinstance(kernel, _PendingKernel):
        kernel._wait()
    return kernel


class _PendingKernel(CachedKernel):
    """A ``CachedKernel`` whose cold compile is still in a session worker.

    The first use loads it and turns it into a plain ``CachedKernel``, so a
    loaded kernel costs launches nothing; the session loads any left unused
    before the constructor returns. ``_kernel`` holds the job until then.
    """

    __slots__ = ()

    def __init__(self, job: _CompileJob) -> None:
        self._kernel = job
        self._stream = job.stream
        self._driver = None

    def _adopt(self, loaded: CachedKernel) -> None:
        # Replace the job last: a thread that finds a loaded kernel in _kernel before the class switch below
        # also finds its stream and driver.
        self._stream, self._driver = loaded._stream, loaded._driver
        self._kernel = loaded._kernel
        # Same slots, so the loaded object is exactly a CachedKernel from here on.
        self.__class__ = CachedKernel

    def _wait(self) -> None:
        """Load the kernel, unless another thread's ``_adopt`` has already replaced the job."""
        job = self._kernel  # read once: _adopt replaces it before it switches the class
        if isinstance(job, _CompileJob):
            job.load()

    @property
    def driver(self) -> DriverLauncher | None:
        """``DriverLauncher`` for the cuda.bindings fast path, or None."""
        self._wait()
        return self._driver

    @property
    def compiled(self) -> Any:
        """The underlying Triton ``CompiledKernel``."""
        self._wait()
        return self._kernel

    def launch(self, grid: tuple[int, ...], *args: Any) -> None:
        """Load the kernel, then launch it as a ``CachedKernel``."""
        self._wait()
        CachedKernel.launch(self, grid, *args)


class _CompilerPool:
    """Isolated compiler processes shared by one thread's model construction.

    A constructor queues each cold configuration and continues; up to
    ``limit`` workers compile the queue in parallel, each holding the shared
    marker lock of the configuration it compiles. A configuration's kernels
    load when first used, or when the session closes.

    A worker that times out or breaks its reply protocol is stopped and not
    replaced, so a session starts at most ``limit`` workers; once none is
    left, the queue falls back to one-shot compiles. With ``idle_timeout``, a
    worker that waits that many seconds for work stops, and the next
    configuration starts a new one.
    """

    def __init__(self, limit: int, idle_timeout: float | None = None) -> None:
        self._limit = limit
        self._idle_timeout = idle_timeout
        self._cond = threading.Condition()
        self._queue: deque[_CompileJob] = deque()
        self._jobs: dict[str, _CompileJob] = {}
        self._threads: list[threading.Thread] = []
        self._processes: list[subprocess.Popen[bytes]] = []
        self._idle = 0
        self._live = 0
        self._failed = False
        self._closed = False
        # Set when a failed constructor stops the pool: workers give up waiting for a marker lock.
        self._cancelled = threading.Event()

    def submit(self, job: _CompileJob) -> _CompileJob | None:
        """Queue ``job``, or return the queued job for the same marker; else ``None`` after a failure."""
        with self._cond:
            # Even after a failure: the queued job already took its fallback, which a timed-out compile
            # would otherwise repeat in a one-shot process.
            queued = self._jobs.get(job.marker)
            if queued is not None:
                return queued
            if self._failed or self._closed:
                return None
            job.pool = self
            self._jobs[job.marker] = job
            self._queue.append(job)
            if self._idle == 0:
                self._spawn()
            self._cond.notify()
        return job

    def forget(self, job: _CompileJob) -> None:
        """Drop a loaded ``job``: its marker now answers later requests for the same configuration."""
        with self._cond:
            if self._jobs.get(job.marker) is job:
                del self._jobs[job.marker]

    def start(self, count: int | None = None) -> None:
        """Start ``count`` workers, all by default, so their startup overlaps other work."""
        with self._cond:
            while (count is None or len(self._threads) < count) and self._spawn():
                pass

    def _spawn(self) -> bool:
        # Hold self._cond.
        if len(self._threads) >= self._limit:
            return False
        thread = threading.Thread(target=self._serve, name="bioir-triton-compiler", daemon=True)
        try:
            thread.start()
        except RuntimeError as exc:
            # The host has no thread to spare: keep the workers running, and fall back without any.
            logger.warning("Triton compiler worker unavailable; using one-shot subprocess: %s", exc)
            self._limit = len(self._threads)
            if not self._threads:
                self._failed = True
                self._fail_queue()
            return False
        # The thread waits for self._cond, which the caller holds, before it can retire.
        self._threads.append(thread)
        self._live += 1
        return True

    def wait(self, job: _CompileJob) -> None:
        """Move ``job`` ahead of the queue if no worker took it yet, then wait for it."""
        with self._cond:
            if not job.started and job in self._queue:
                self._queue.remove(job)
                self._queue.appendleft(job)
        job.done.wait()

    def _fail_queue(self) -> None:
        # Hold self._cond. Queued configurations fall back when loaded.
        while self._queue:
            self._queue.popleft().finish(_FAILED)

    def _retire(self) -> None:
        # Hold self._cond. The last worker to stop leaves the queue to one-shot compiles.
        self._live -= 1
        if self._live == 0:
            self._failed = True
            self._fail_queue()
            self._cond.notify_all()

    def _serve(self) -> None:
        process = None
        expired = killed = False
        try:
            env = {**os.environ, "TRITON_CACHE_DIR": TRITON_CACHE_DIR}
            # No stderr pipe: compiler diagnostics cannot fill it and block a
            # response. The one-shot fallback keeps them for failures.
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", _WORKER_SCRIPT, str(os.getpid())],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            with self._cond:
                self._processes.append(process)
            served = False
            ready = False
            while True:
                with self._cond:
                    if not ready:
                        self._idle += 1
                    ready = False
                    while not self._queue and not self._closed and not self._failed:
                        if not self._cond.wait(self._idle_timeout) and not self._queue:
                            expired = True
                            break
                    self._idle -= 1
                    if expired:
                        # Free the slot, so the next configuration spawns a worker; close() still reaps the process.
                        self._threads.remove(threading.current_thread())
                        self._live -= 1
                        return
                    if not self._queue:
                        if not served:
                            # It wrote nothing: stop it without waiting for its startup.
                            process.kill()
                        return
                    job = self._queue.popleft()
                    job.started = True
                served = True
                outcome = _BROKEN
                try:
                    outcome = self._warm(process, job)
                finally:
                    if outcome not in (_TIMED_OUT, _BROKEN):
                        # Count it idle before its caller wakes, so the caller's next configuration finds it.
                        with self._cond:
                            self._idle += 1
                        ready = True
                    job.finish(outcome)
                if outcome in (_TIMED_OUT, _BROKEN):
                    # Stop it now: a hung compile would otherwise run until the session closes.
                    process.kill()
                    killed = True
                    with self._cond:
                        self._retire()
                    return
        except Exception as exc:
            # Retire on any error, so queued configurations fall back instead of waiting on a dead thread.
            logger.warning("Triton compiler worker unavailable; using one-shot subprocess: %s", exc)
            if process is not None:
                with suppress(OSError):
                    process.kill()
                killed = True
            with self._cond:
                self._retire()
        finally:
            if process is not None and process.stdin is not None:
                with suppress(OSError):
                    process.stdin.close()
            if expired or killed:
                # Reap it now: a pool that lives as long as the process would keep a zombie and its pipe.
                _stop_worker(process)
            if expired:
                with self._cond:
                    self._processes.remove(process)

    def _warm(self, process: subprocess.Popen[bytes], job: _CompileJob) -> str:
        try:
            os.makedirs(os.path.dirname(job.marker), exist_ok=True)
            lock = FileLock(
                Path(f"{job.marker}.lock"), exclusive=True, timeout=_WARM_LOCK_TIMEOUT, cancel=self._cancelled
            )
            lock.__enter__()
        except (OSError, RuntimeError):
            if self._cancelled.is_set():
                # close() killed the worker; a kernel kept past the failure loads through the one-shot path.
                return _BROKEN
            # No shared marker: warm this process's own copy.
            return _request_warm(process, job.spec)
        try:
            if os.path.exists(job.marker):
                return _WARMED
            outcome = _request_warm(process, job.spec)
            if outcome == _WARMED:
                with suppress(OSError):
                    Path(job.marker).touch()
            return outcome
        finally:
            lock.__exit__(None, None, None)

    def close(self, load: bool) -> None:
        """Load every queued configuration when ``load``, then stop the workers."""
        try:
            if load:
                for job in list(self._jobs.values()):
                    job.load()
        finally:
            with self._cond:
                self._closed = True
                self._fail_queue()
                self._cond.notify_all()
                processes = list(self._processes)
                threads = list(self._threads)
            if not load:
                # A failed constructor needs no compile in flight; Triton writes cache entries atomically.
                self._cancelled.set()
                for process in processes:
                    with suppress(OSError):
                        process.kill()
            for thread in threads:
                thread.join()
            # A worker that was still starting has joined the list by now.
            with self._cond:
                processes = list(self._processes)
            for process in processes:
                _stop_worker(process)


_warm_session = threading.local()

# A worker idle this long after its last request-time compile stops; the next one starts another. An idle worker holds
# ~600 MiB RSS and 28 threads; a restart slows the next new-shape request (OpenFold3 8FZA 30 s vs 150 s after the first
# request: B200 3.2 vs 5.1 s, B300 4.7 vs 7.2 s).
_REQUEST_IDLE_TIMEOUT = 120
_request_pool_lock = threading.Lock()
_request_pool_instance: _CompilerPool | None = None


def _request_pool() -> _CompilerPool:
    """The process's compilers for cold configurations first compiled outside model construction.

    Each caller waits for its own configuration; a worker serves one after
    another, so a single thread pays the interpreter startup once rather than
    per configuration. Concurrent callers start more workers, up to the session
    limit. Like session workers, a worker keeps the environment it started with.
    """
    global _request_pool_instance
    with _request_pool_lock:
        if _request_pool_instance is None:
            _request_pool_instance = _CompilerPool(_session_worker_limit(), idle_timeout=_REQUEST_IDLE_TIMEOUT)
        return _request_pool_instance


@atexit.register
def _close_request_pool() -> None:
    pool = _request_pool_instance
    if pool is not None:
        pool.close(load=False)


def _forget_request_pool() -> None:
    """Give a forked child no compiler: it has none of the parent's threads, and starts its own on demand."""
    global _request_pool_instance, _request_pool_lock
    pool, _request_pool_instance = _request_pool_instance, None
    _request_pool_lock = threading.Lock()
    if pool is not None:
        # Point the inherited pipe ends at /dev/null, so a parent's worker sees EOF when the parent closes its
        # stdin. Rebinding the descriptors takes none of the pipes' locks, which a parent thread may have held.
        devnull = os.open(os.devnull, os.O_RDWR)
        try:
            for process in pool._processes:
                for pipe in (process.stdin, process.stdout):
                    if pipe is not None:
                        with suppress(OSError, ValueError):
                            os.dup2(devnull, pipe.fileno())
        finally:
            os.close(devnull)


os.register_at_fork(after_in_child=_forget_request_pool)


def _start_triton_driver() -> None:
    """Start Triton's CUDA driver in this process, which a cold cache makes it build.

    The build lands in the shared cache, so workers starting meanwhile load it
    instead of each building their own. This process needs the driver for its
    first kernel load anyway.
    """
    try:
        from triton.runtime import driver

        driver.active.get_current_target()
    except (ImportError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        logger.debug("Triton driver unavailable before the first kernel load: %s", exc)


@contextmanager
def triton_warmup_session() -> Iterator[None]:
    """Compile cold Triton configurations in parallel isolated workers during model construction.

    Nested sessions share their thread's workers. Each configuration's kernels
    load in this process on first use, and the outermost session loads the rest
    and stops its workers before it exits. A warm disk cache starts no worker.
    After a construction that compiled cold configurations, one worker stays up
    for those the first requests compile; it stops after
    ``_REQUEST_IDLE_TIMEOUT`` seconds idle.
    """
    if getattr(_warm_session, "pool", None) is not None:
        yield
        return
    pool = _CompilerPool(_session_worker_limit())
    _warm_session.pool = pool
    loaded = False
    compiled = False
    try:
        if _triton_cache_was_cold():
            pool.start()
            _start_triton_driver()
        yield
        loaded = True
        compiled = bool(pool._jobs)
    finally:
        try:
            # Loading may recompile under a register cap, through the same workers.
            pool.close(load=loaded)
        finally:
            del _warm_session.pool
    if compiled:
        # Start it now, so its startup overlaps weight loading rather than the first request.
        _request_pool().start(1)


def _triton_target() -> dict | None:
    """The GPU target Triton compiles for on this device, or ``None`` when it has no CUDA driver."""
    try:
        from triton.runtime import driver

        target = driver.active.get_current_target()
    except (ImportError, OSError, RuntimeError, subprocess.SubprocessError):
        return None
    return {"backend": target.backend, "arch": target.arch, "warp_size": target.warp_size}


def _warm_spec(
    jit_fn: JITFunction,
    dummy_args_by_dtype: dict[torch.dtype, tuple],
    grid: tuple[int, ...],
    constexpr_kwargs: dict,
) -> tuple[dict, str]:
    """Describe one kernel config (all dtypes) for a compiler process, with its warm marker path."""
    variants = []
    for dummy_args in dummy_args_by_dtype.values():
        variant = []
        for arg in dummy_args:
            if isinstance(arg, torch.Tensor):
                variant.append(
                    {
                        "type": "tensor",
                        "shape": list(arg.shape),
                        "dtype": str(arg.dtype).removeprefix("torch."),
                    }
                )
            elif isinstance(arg, float):
                # Triton types a float argument fp32; int() would warm an i32 variant.
                variant.append({"type": "float", "value": arg})
            else:
                variant.append({"type": "int", "value": int(arg)})
        variants.append(variant)

    ckw_ser = {}
    for name, value in constexpr_kwargs.items():
        ckw_ser[name] = value if isinstance(value, (bool, float)) else int(value)

    spec = {
        "triton_cache_dir": TRITON_CACHE_DIR,
        "target": _triton_target(),
        "module": jit_fn.fn.__module__,
        "fn_name": jit_fn.fn.__name__,
        "grid": list(grid),
        "constexpr_kwargs": ckw_ser,
        "variants": variants,
    }
    warm_dir = os.path.join(TRITON_CACHE_DIR, "bioir_warm")
    marker = os.path.join(warm_dir, hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest())
    return spec, marker


def _compile_in_subprocess(
    jit_fn: JITFunction,
    dummy_args_by_dtype: dict[torch.dtype, tuple],
    grid: tuple[int, ...],
    constexpr_kwargs: dict,
) -> None:
    """Compile one kernel config (all dtypes) in a subprocess.

    A marker in the Triton cache records each warmed config, so concurrent
    processes sharing the cache warm it once.
    """
    spec, marker = _warm_spec(jit_fn, dummy_args_by_dtype, grid, constexpr_kwargs)
    try:
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        with FileLock(Path(f"{marker}.lock"), exclusive=True, timeout=_WARM_LOCK_TIMEOUT):
            if not os.path.exists(marker):
                _run_warm_subprocess(jit_fn, spec, marker)
    except (OSError, RuntimeError):
        # No shared marker: warm this process's own copy.
        _run_warm_subprocess(jit_fn, spec, None)


def _run_warm_subprocess(jit_fn: JITFunction, spec: dict, marker: str | None) -> None:
    spec_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(spec, f)
            spec_path = f.name

        result = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_SCRIPT, spec_path],
            capture_output=True,
            text=True,
            timeout=_WARM_TIMEOUT,
        )
        if result.returncode != 0:
            logger.warning(
                "Subprocess compile failed for %s.%s: %s",
                jit_fn.fn.__module__,
                jit_fn.fn.__name__,
                result.stderr[-500:],
            )
        elif marker is not None:
            Path(marker).touch()
    except Exception as e:
        logger.warning("Subprocess compile error for %s: %s", jit_fn.fn.__name__, e)
    finally:
        if spec_path:
            try:
                os.unlink(spec_path)
            except OSError:
                pass


def _looks_like_compiled_kernel(obj: Any) -> bool:
    """Return whether *obj* exposes the required ``CompiledKernel`` ABI."""
    return obj is not None and hasattr(obj, "function") and hasattr(obj, "packed_metadata") and hasattr(obj, "run")


class TritonKernelCache(KernelCacheBase):
    """Triton cache with subprocess warmup and direct-driver launch support."""

    def compile(
        self,
        jit_fn: JITFunction,
        dummy_args: tuple,
        grid: tuple[int, ...],
        constexpr_kwargs: dict,
        stream: torch.cuda.Stream | None = None,
    ) -> CachedKernel:
        """Compile and wrap a Triton kernel.

        Args:
            jit_fn: A ``@triton.jit`` decorated function.
            dummy_args: Dummy positional arguments.
            grid: Grid dimensions for compilation.
            constexpr_kwargs: Keyword arguments for ``tl.constexpr`` params.
            stream: Optional launch stream.

        Returns:
            The compiled kernel wrapper.
        """
        # ``warmup`` compiles and specializes exactly as a launch does, but never
        # runs the kernel, so the uninitialized dummy buffers are only addressed.
        compiled = jit_fn.warmup(*dummy_args, grid=grid, **constexpr_kwargs)

        if not _looks_like_compiled_kernel(compiled):
            compiled = self._lookup_compiled(jit_fn, dummy_args, constexpr_kwargs)

        # A value-specialized CUBIN is only valid for the dummy arguments. Do not
        # wrap it: DriverLauncher and CachedKernel.launch()/.run() would both reuse
        # it. Hand the JIT function through so launch() recompiles per specialization.
        name = f"{jit_fn.fn.__module__}.{jit_fn.fn.__name__}"
        unprotected = value_specialized_params(jit_fn, dummy_args)
        if unprotected is None:
            logger.warning(
                "Cannot introspect parameters of %s; disabling the direct CUBIN launcher because value "
                "specialization cannot be ruled out",
                name,
            )
        elif unprotected:
            logger.warning(
                "%s has scalar parameter(s) %s without do_not_specialize; disabling the direct CUBIN "
                "launcher because the compiled kernel is only valid for the compile-time values",
                name,
                ", ".join(unprotected),
            )
        enable_driver = unprotected == []
        return CachedKernel(compiled if enable_driver else jit_fn, stream=stream, enable_driver=enable_driver)

    @staticmethod
    def _lookup_compiled(
        jit_fn: JITFunction,
        dummy_args: tuple,
        constexpr_kwargs: dict,
    ) -> Any:
        """Read a compiled kernel from Triton's legacy per-device cache."""
        device = torch.cuda.current_device()
        cache, key_cache, _, _, binder = jit_fn.device_caches[device]
        runtime_kwargs: dict[str, Any] = {"debug": bool(jit_fn.debug)}
        if _knobs is not None:
            if getattr(_knobs.runtime, "debug", False):
                runtime_kwargs["debug"] = True
            if hasattr(_knobs.compilation, "instrumentation_mode"):
                runtime_kwargs["instrumentation_mode"] = _knobs.compilation.instrumentation_mode
        ba, spec, opts = binder(*dummy_args, **constexpr_kwargs, **runtime_kwargs)
        key = compute_cache_key(key_cache, spec, opts)
        return cache[key]

    def compile_for_dtypes(
        self,
        jit_fn: JITFunction,
        dtypes: Sequence[torch.dtype],
        make_dummy_args,
        *,
        grid: tuple[int, ...] = (0,),
        max_registers: int | None = None,
        **constexpr_kwargs: Any,
    ) -> dict[torch.dtype, CachedKernel]:
        """Compile one kernel for each dtype.

        During a :func:`triton_warmup_session` on a cold cache, the kernels
        compile in the session's workers and each loads on first use, so the
        constructor goes on while they compile.

        Args:
            jit_fn: A ``@triton.jit`` decorated function.
            dtypes: Sequence of dtypes to compile for.
            make_dummy_args: Callable returning positional arguments by dtype.
            grid: Compilation grid. An empty grid, the default, compiles and loads
                the kernel without launching it, here and in the cold-cache
                compiler process, which passes each tensor dummy's dtype.
            max_registers: Registers per thread a kernel may use. A dtype whose
                kernel uses more is recompiled with ``maxnreg`` set to it.
            **constexpr_kwargs: Constexpr keyword arguments.

        Returns:
            Compiled kernels keyed by dtype.
        """
        unique_dtypes = list(dict.fromkeys(dtypes))
        dummy_args_by_dtype = {dtype: make_dummy_args(dtype) for dtype in unique_dtypes}

        if _triton_cache_was_cold():
            session_pool = getattr(_warm_session, "pool", None)
            pool = session_pool if session_pool is not None else _request_pool()
            spec, marker = _warm_spec(jit_fn, dummy_args_by_dtype, grid, constexpr_kwargs)
            if not os.path.exists(marker):
                job = _CompileJob(
                    self, jit_fn, make_dummy_args, dummy_args_by_dtype, grid, max_registers, constexpr_kwargs
                )
                job.spec, job.marker = spec, marker
                queued = pool.submit(job)
                if queued is job:
                    if pool is not session_pool:
                        # Outside a session the caller needs the kernels now.
                        try:
                            job.load()
                        finally:
                            pool.forget(job)
                    return dict(job.kernels)
                if queued is not None:
                    # The same configuration is already queued: share its compile, or its fallback, and load
                    # separately. A one-shot compile would only repeat a failure or a timeout.
                    queued.load()
                else:
                    _compile_in_subprocess(jit_fn, dummy_args_by_dtype, grid, constexpr_kwargs)
            else:
                _compile_in_subprocess(jit_fn, dummy_args_by_dtype, grid, constexpr_kwargs)

        stream = torch.cuda.current_stream()
        kernels = {}
        for dt in unique_dtypes:
            dummy_args = dummy_args_by_dtype[dt]
            kernels[dt] = self.compile(jit_fn, dummy_args, grid, constexpr_kwargs, stream=stream)
        if max_registers is not None:
            self._cap_registers(kernels, jit_fn, make_dummy_args, grid, max_registers, constexpr_kwargs)
        return kernels

    def _cap_registers(
        self,
        kernels: dict[torch.dtype, CachedKernel],
        jit_fn: JITFunction,
        make_dummy_args: Callable[[torch.dtype], tuple],
        grid: tuple[int, ...],
        max_registers: int,
        constexpr_kwargs: dict,
    ) -> None:
        """Replace each loaded kernel that uses more than ``max_registers`` with a ``maxnreg`` build."""
        heavy = [dtype for dtype, kernel in kernels.items() if getattr(kernel.compiled, "n_regs", 0) > max_registers]
        if heavy:
            capped = self.compile_for_dtypes(
                jit_fn, heavy, make_dummy_args, grid=grid, maxnreg=max_registers, **constexpr_kwargs
            )
            kernels.update({dtype: _loaded(kernel) for dtype, kernel in capped.items()})

    def save_to_cache(self, key: tuple, artifact: Any) -> None:
        """No-op — Triton manages its own disk cache."""

    def load_from_cache(self, key: tuple) -> CachedKernel | None:
        """Return ``None`` because Triton manages cache lookup internally."""
        return None

    def get_or_compile(self, key: tuple, *args: Any, **kwargs: Any) -> Any:
        """Compile through Triton's cache.

        Args:
            key: Unused interface-compatible cache key.
            *args, **kwargs: Forwarded to :meth:`compile`.

        Returns:
            The compiled kernel wrapper.
        """
        return self.compile(*args, **kwargs)
