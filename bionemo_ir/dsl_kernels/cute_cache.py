# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Portions of this file are adapted from Quack:
# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# Licensed under Apache-2.0.
# Source: https://github.com/Dao-AILab/quack/blob/c8ec3170057987da0ec99883736f381ea1937cf3/quack/cache/jit.py

"""Persistent .o cache for CuTe DSL compiled kernels.

Compiled kernels are exported as object files (.o) via ``export_to_c``.
On subsequent runs the .o is loaded instead of regenerating IR and re-JIT'ing.

An entry is host code, not just device code: the .o embeds the CUBIN image and
wraps it in native code the compiler emitted for the machine that built it. The
CUBIN half is reproducible across hosts — the same kernel and target SM give the
same image bytes wherever it compiles — but the wrapper half is only as portable
as the CPU that produced it. A cache directory is routinely shared (one volume
per node pool, one worktree cache across several checkouts), so the disk key
carries every property a replayed .o could disagree with: kernel sources, the
CuTe DSL version, the Python ABI, the device SM count, and the host CPU ISA.
See :func:`_host_cpu_isa_signature` and :func:`_device_sm_count` for why the
last two are in there.

Inherits from :class:`~.cache_base.KernelCacheBase` for the unified
compile / save / load interface shared with the Triton backend.

The CuTeDSL compiler is imported on first compile or cache load, so a
source-free install that only launches packaged CUBINs never needs it.

Usage::

    class MyKernel(CuteKernelCache):
        def my_compile(self, ...):
            exe = self.compile(kernel, *fakes)
            self.save_to_cache(disk_key, exe)
            ...
        def my_load(self, ...):
            exe = self.load_from_cache(disk_key)
            ...

Controls:
  BIOIR_KERNEL_CACHE_ENABLED=0  — disable persistent .o cache (default: enabled)
  BIOIR_KERNEL_CACHE_DIR=path   — override default cache directory
                                  (must be owned by the runtime user)
  CUTEDSL_FORCE_CUBIN=1           — resolve executables from the packaged CUBIN
                                    library instead of compiling from source

"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import hashlib
import inspect
import logging
import os
import pickle
import platform
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import CodeType, FrameType, ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cutlass.cutlass_dsl.tvm_ffi_provider import TVMFFIJitCompiledFunction

from .cache_base import DiskCache, FileLock, KernelCacheBase

__all__ = [
    "CuteKernelCache",
    "CACHE_ENABLED",
    "CACHE_DIR",
    "EXTRA_SOURCE_DIRS",
    "FORCE_CUBIN_ENV",
    "get_cache_dir",
    "FileLock",
    "memoized_op_locations",
]

CACHE_ENABLED: bool = DiskCache.ENABLED
CACHE_DIR: str | None = DiskCache._CACHE_DIR

FORCE_CUBIN_ENV = "CUTEDSL_FORCE_CUBIN"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

EXTRA_SOURCE_DIRS: list[Path] = []

EXPORT_FUNC_NAME = "kernel"
LOCK_TIMEOUT = 60
# A cold compile takes up to a few minutes on a contended host.
COMPILE_WAIT_TIMEOUT = 600

# Compile locks this process holds, by key hash, from a cache miss to its save.
_compile_claims: dict[str, int] = {}


def _cute() -> ModuleType:
    """Import the CuTeDSL compiler, which only the source path needs."""
    import cutlass.cute as cute

    return cute


class _SiteMemoizedInspect:
    """``inspect`` with ``getframeinfo`` memoized per call site; every other name is the real module's."""

    def __init__(self) -> None:
        # Keyed by code object identity: equal code objects from different files compare equal. Each entry
        # holds its code object, so the id is not reused while the entry lives.
        self._sites: dict[tuple[int, int, int, int], tuple[CodeType, inspect.Traceback]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(inspect, name)

    def getframeinfo(self, frame: FrameType, context: int = 1) -> inspect.Traceback:
        """Return ``inspect.getframeinfo(frame, context)``, computed once per code object and instruction."""
        if not inspect.isframe(frame):
            return inspect.getframeinfo(frame, context)
        code = frame.f_code
        key = (id(code), frame.f_lasti, frame.f_lineno, context)
        site = self._sites.get(key)
        if site is None:
            site = self._sites[key] = (code, inspect.getframeinfo(frame, context))
        return site[1]


_memo_lock = threading.Lock()
# Open memo scopes per thread.
_memo_scopes: dict[int, int] = {}
# 4.6 moved the helper and changed location capture; newer releases also cache
# source lookups upstream. Recheck locations and error diagnostics before adding
# a version here rather than assuming that a private API remains compatible.
_OP_LOCATION_MEMO_VERSIONS = frozenset({"4.5.2", "4.5.3"})
_memo_warning_logged = False
_logger = logging.getLogger(__name__)


def _reset_memo_after_fork() -> None:
    """Keep only the forking thread's memo scopes in a forked child, with a fresh lock."""
    global _memo_lock
    _memo_lock = threading.Lock()
    forking = threading.get_ident()
    for thread in [t for t in _memo_scopes if t != forking]:
        del _memo_scopes[thread]
    dsl_op = sys.modules.get("cutlass.base_dsl._mlir_helpers.op")
    if not _memo_scopes and isinstance(getattr(dsl_op, "inspect", None), _SiteMemoizedInspect):
        dsl_op.inspect = inspect


os.register_at_fork(after_in_child=_reset_memo_after_fork)


@contextlib.contextmanager
def memoized_op_locations() -> Iterator[None]:
    """Memoize the CuTeDSL front end's source-location lookup for one compile.

    ``dsl_user_op`` (``cutlass/base_dsl/_mlir_helpers/op.py``) calls
    ``inspect.getframeinfo`` on its caller for every IR op it builds. Each
    call decodes the code object's position table up to the current
    instruction and, on Python 3.12, also scans the source back to the
    enclosing ``def``; one kernel builds tens of thousands of ops from about a
    thousand call sites, so uncached these lookups are most of a cold compile.
    A site's answer depends only on its code object, instruction and source
    file, none of which change during a compile. The DSL re-executes the
    kernel source for every compile, so the memo lives until the last open
    compile in any thread returns; a forked child counts only the forking
    thread's. Untested versions and changed helpers use the upstream compiler
    unchanged and log one warning per process so maintainers can recheck this
    workaround. ``CuteKernelCache.compile`` and the CUBIN pack builders enter
    this scope; cache hits and CUBIN launches never do.
    """
    global _memo_warning_logged
    dsl_op = None
    dsl_version = "unknown"
    skip_reason = "CuTeDSL is unavailable"
    try:
        import cutlass
    except ImportError:
        pass
    else:
        dsl_version = getattr(cutlass, "__version__", "unknown")
        if dsl_version not in _OP_LOCATION_MEMO_VERSIONS:
            skip_reason = "this version has not been validated for the private helper"
        else:
            try:
                from cutlass.base_dsl._mlir_helpers import op as dsl_op
            except ImportError:
                skip_reason = "the private location helper is unavailable"
            else:
                skip_reason = "the private helper's inspect reference changed"
    thread = threading.get_ident()
    warn = False
    with _memo_lock:
        memo = getattr(dsl_op, "inspect", None)
        active = isinstance(memo, _SiteMemoizedInspect) or memo is inspect
        if active:
            if not isinstance(memo, _SiteMemoizedInspect):
                memo = dsl_op.inspect = _SiteMemoizedInspect()
            _memo_scopes[thread] = _memo_scopes.get(thread, 0) + 1
        elif not _memo_warning_logged:
            _memo_warning_logged = warn = True
    if warn:
        _logger.warning(
            "CuTeDSL %s: BioIR source-location memoization disabled: %s. "
            "Using the upstream compiler unchanged; cold compilation may be slower. "
            "Maintainers: recheck source locations and compile-error diagnostics, then update "
            "_OP_LOCATION_MEMO_VERSIONS in bionemo_ir/dsl_kernels/cute_cache.py if compatible.",
            dsl_version,
            skip_reason,
        )
    try:
        yield
    finally:
        if active:
            with _memo_lock:
                _memo_scopes[thread] -= 1
                if not _memo_scopes[thread]:
                    del _memo_scopes[thread]
                if not _memo_scopes and dsl_op.inspect is memo:
                    dsl_op.inspect = inspect


def get_cache_dir() -> Path:
    """Return (and create) the root cache directory."""
    return _secure_cache_dir(DiskCache.get_cache_dir())


def _secure_cache_dir(path: Path) -> Path:
    """Create a private cache directory owned by this user.

    The ``mkdir`` mode is masked by the process umask, so the mode is also set
    explicitly. Both the ownership check and that fix-up go through a single
    descriptor: a separate ``lstat`` and ``chmod`` pair names the path twice,
    which lets a peer swap in a different target between the two calls, and
    ``O_NOFOLLOW`` refuses a symlink standing in for the directory.

    Args:
        path: Cache directory to create and lock down.

    Returns:
        The same path, now mode ``0o700`` and owned by the runtime user.

    Raises:
        PermissionError: The path is not a directory this user owns.
    """
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as e:
        raise PermissionError("Kernel cache directory is not private") from e
    try:
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
            raise PermissionError("Kernel cache directory is not private")
        os.fchmod(fd, 0o700)
    finally:
        os.close(fd)
    return path


def _is_owned_object(path: Path) -> bool:
    """Return whether path is a user-owned regular file."""
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()


# ---------------------------------------------------------------------------
# Source fingerprinting (CuTe-specific — includes cutlass version)
# ---------------------------------------------------------------------------


def _hash_source_dir(h: hashlib._Hash, root: Path) -> None:
    """Hash all Python sources under *root* into *h*."""
    for src in sorted(root.rglob("*.py")):
        if not src.is_file():
            continue
        h.update(src.relative_to(root).as_posix().encode())
        content = src.read_bytes()
        h.update(len(content).to_bytes(8, "little"))
        h.update(content)


@functools.lru_cache(maxsize=1)
def _compute_source_fingerprint() -> str:
    """Hash kernel source dirs plus runtime ABI stamps into a fingerprint."""
    import cutlass

    h = hashlib.sha256()
    h.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    h.update(f"cutlass={cutlass.__version__}".encode() if hasattr(cutlass, "__version__") else b"cutlass=unknown")
    # The exported .o wraps device code in host code built for this CPU, so a
    # shared cache must not cross ISAs — see _host_cpu_isa_signature.
    h.update(f"host_isa={_host_cpu_isa_signature()}".encode())
    # Separate SKUs that share a compute capability but differ in SM count.
    h.update(f"sm_count={_device_sm_count()}".encode())

    dsl_kernels_dir = Path(__file__).resolve().parent
    _hash_source_dir(h, dsl_kernels_dir)

    for extra_dir in EXTRA_SOURCE_DIRS:
        _hash_source_dir(h, Path(extra_dir).resolve())

    return h.hexdigest()


_CPUINFO_ISA_FIELDS = frozenset(
    {
        "cpu architecture",
        "cpu family",
        "cpu implementer",
        "cpu part",
        "cpu revision",
        "cpu variant",
        "features",
        "flags",
        "model",
        "stepping",
        "vendor_id",
    }
)
_CPUINFO_FEATURE_FIELDS = frozenset({"features", "flags"})
_HOST_CPU_FALLBACK_NONCE = os.urandom(16).hex()


def _read_cpuinfo() -> str | None:
    try:
        return Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _cpuinfo_isa_profiles(cpuinfo: str, affinity: set[int] | None) -> tuple[str, ...]:
    """Return the distinct ISA profiles for CPUs on which this process may run."""
    records: list[dict[str, str]] = []
    record: dict[str, str] = {}
    for line in cpuinfo.splitlines():
        if not line.strip():
            if record:
                records.append(record)
                record = {}
            continue
        key, separator, value = line.partition(":")
        if separator:
            record[key.strip().lower()] = value.strip().lower()
    if record:
        records.append(record)

    indexed_records: list[tuple[int, dict[str, str]]] = []
    for current in records:
        try:
            indexed_records.append((int(current["processor"]), current))
        except (KeyError, ValueError):
            indexed_records = []
            break
    if affinity is not None and indexed_records:
        records = [current for cpu, current in indexed_records if cpu in affinity]

    profiles: set[str] = set()
    for current in records:
        if not any(current.get(field) for field in _CPUINFO_FEATURE_FIELDS):
            return ()
        fields: dict[str, str] = {}
        for key in _CPUINFO_ISA_FIELDS:
            value = current.get(key)
            if not value:
                continue
            if key in _CPUINFO_FEATURE_FIELDS:
                value = " ".join(sorted(set(value.split())))
            fields[key] = value
        profiles.add("|".join(f"{key}={fields[key]}" for key in sorted(fields)))
    return tuple(sorted(profiles))


def _host_cpu_isa_signature() -> str:
    """Return a safe cache partition for the host instructions an ELF may use.

    The GPU side of a cached entry is host-independent: the same kernel compiled
    for the same target SM yields the same CUBIN image on any builder, which is
    what lets the shipped CUBIN packs be content-addressed without recording
    where they were compiled. The host wrapper around it is not. ``export_to_c``
    runs a native compiler that may use whatever ISA extensions the building CPU
    advertises, so two runners sharing one cache
    directory can hold a key that matches in every kernel-visible way and still
    produce an object the loading CPU cannot execute — a SIGILL in code whose
    device half was never in question, and the reason this belongs in the key
    rather than in a comment about being careful with shared volumes.

    The signature is the CPU model and feature set exactly as ``/proc/cpuinfo``
    reports it, canonicalized (feature flags sorted, fields ordered) so hosts
    that differ only in reporting order share entries, and scoped to this
    process's affinity mask because a cgroup may pin it to one socket of a
    heterogeneous machine. When the features cannot be read, or the affinity
    spans CPUs with different profiles, no persistent value is trustworthy: the
    signature then carries a process-unique nonce, which keeps in-process reuse
    and makes the entry unshareable.
    """
    machine = platform.machine().strip().lower()
    try:
        affinity = set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None

    cpuinfo = _read_cpuinfo()
    profiles = _cpuinfo_isa_profiles(cpuinfo, affinity) if cpuinfo is not None else ()
    if len(profiles) == 1:
        return f"machine={machine}|{profiles[0]}"

    # No single trustworthy profile: fall back to a process-unique key.
    profile_summary = "||".join(profiles) if profiles else "unavailable"
    process_nonce = f"{os.getpid()}-{_HOST_CPU_FALLBACK_NONCE}"
    return f"machine={machine}|profiles={profile_summary}|nonshareable={process_nonce}"


def _device_sm_count() -> int:
    """SM count of the active CUDA device (0 if unavailable).

    Folded into the cache fingerprint so a kernel compiled for one SKU isn't
    reused on another with a different SM count: CuTe persistent kernels bake
    launch geometry (grid / cluster sizing) from the runtime SM count at compile
    time. Compute capability alone cannot separate those SKUs; SM count can.
    Assumes one device per process (the fingerprint is lru_cached on first use).
    """
    try:
        import torch

        return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    except Exception:
        return 0


def _key_to_hash(key: tuple) -> str:
    return hashlib.sha256(pickle.dumps(key)).hexdigest()


def _claim_compile(cache_path: Path, sha: str) -> bool:
    """Lock ``sha`` for compiling; return whether a peer held it first.

    The lock lasts until ``save_to_cache`` or process exit. A waiter that
    times out compiles without it.
    """
    fd = os.open(str(cache_path / f"{sha}.compile"), os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _compile_claims[sha] = fd
        return False
    except OSError:
        pass
    deadline = time.monotonic() + COMPILE_WAIT_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(0.2)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _compile_claims[sha] = fd
            return True
        except OSError:
            continue
    os.close(fd)
    return True


def _release_compile(sha: str) -> None:
    fd = _compile_claims.pop(sha, None)
    if fd is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ---------------------------------------------------------------------------
# CuteKernelCache — KernelCacheBase implementation for CuTe DSL
# ---------------------------------------------------------------------------


class CuteKernelCache(KernelCacheBase):
    """CuTe DSL kernel cache backed by ``.o`` object files.

    Implements the :class:`~.cache_base.KernelCacheBase` interface:

    * :meth:`compile` — calls ``cute.compile`` with TVM FFI.
    * :meth:`save_to_cache` — exports the compiled kernel via ``export_to_c``.
    * :meth:`load_from_cache` — loads the ``.o`` via ``cute.runtime.load_module``.
    """

    @staticmethod
    def force_cubin() -> bool:
        """Whether ``CUTEDSL_FORCE_CUBIN`` demands the packaged CUBIN path.

        Lets a source-enabled build exercise the CUBINs it ships even though the
        kernel sources are importable, which is otherwise only reachable in a
        source-free build. Read per call rather than at import so a process can
        flip it; only ops that have a CUBIN launcher honor it.

        Accepts ``1``, ``true``, ``yes``, or ``on`` (case-insensitive).
        """
        return os.getenv(FORCE_CUBIN_ENV, "").strip().lower() in _TRUTHY

    def compile(
        self, kernel_callable: Any, *fake_tensors: Any, options: str = "--enable-tvm-ffi", **kwargs: Any
    ) -> TVMFFIJitCompiledFunction:
        """Compile a CuTe DSL kernel with fake tensors.

        Args:
            kernel_callable: A ``@cute.jit``-decorated class or function.
            *fake_tensors: Fake tensors describing the kernel signature.
            options: Compile options string (default includes TVM FFI).

        Returns:
            The compiled ``TVMFFIJitCompiledFunction``.
        """
        cute = _cute()
        with memoized_op_locations():
            return cute.compile(kernel_callable, *fake_tensors, options=options, **kwargs)

    def save_to_cache(self, key: tuple, artifact: Any) -> None:
        """Export compiled kernel as ``.o`` to disk cache.

        The object file is exported to a unique temp file in the cache dir and
        then atomically renamed into place (``os.replace``). Exporting straight
        to the canonical ``{sha}.o`` is not crash-safe: under memory pressure
        a writer can be killed mid-``export_to_c``, leaving a *truncated*
        ``.o`` at the path every other process probes. A later reader then
        loads that partial object and executes it, dying with SIGILL ("Fatal
        Python error: Illegal instruction"). The atomic rename means the
        canonical path only ever names a fully-exported object; a killed
        writer leaves at most a stray temp file, never a poisoned cache
        entry.

        Args:
            key: Hashable tuple identifying the kernel variant.
            artifact: Object returned by ``cute.compile`` (has ``export_to_c``).
        """
        if not CACHE_ENABLED:
            return

        sha = _key_to_hash(key)
        try:
            cache_path = _secure_cache_dir(get_cache_dir() / _compute_source_fingerprint())
            o_path = cache_path / f"{sha}.o"
            lock_path = cache_path / f"{sha}.lock"
            with FileLock(lock_path, exclusive=True, timeout=LOCK_TIMEOUT):
                if o_path.exists() or o_path.is_symlink():
                    if _is_owned_object(o_path):
                        return
                    o_path.unlink(missing_ok=True)
                # Same directory as o_path so os.replace() is a same-filesystem
                # atomic rename (a cross-fs rename would fall back to a
                # non-atomic copy, reopening the torn-write window).
                fd, tmp_name = tempfile.mkstemp(dir=str(cache_path), prefix=f"{sha}.", suffix=".o.tmp")
                os.close(fd)
                try:
                    artifact.export_to_c(
                        object_file_path=tmp_name,
                        function_name=EXPORT_FUNC_NAME,
                    )
                    os.replace(tmp_name, o_path)
                finally:
                    # Removes the temp on export failure; a no-op once the
                    # rename has consumed it.
                    Path(tmp_name).unlink(missing_ok=True)
        except Exception as e:
            from bionemo_ir.logger import logger

            logger.warning(f"bioir kernel cache: export failed for key {sha}: {e}")
        finally:
            _release_compile(sha)

    def load_from_cache(self, key: tuple) -> Any | None:
        """Load a compiled CuTe DSL kernel from the disk cache.

        A miss claims the key's compile lock for the caller, who compiles and
        then calls :meth:`save_to_cache`. While another process holds that
        lock, this waits for its save and loads the result.

        Args:
            key: Same hashable tuple used in :meth:`save_to_cache`.

        Returns:
            The loaded callable if a cache hit, or ``None``.
        """
        if not CACHE_ENABLED:
            return None

        try:
            sha = _key_to_hash(key)
            cache_path = _secure_cache_dir(get_cache_dir() / _compute_source_fingerprint())
        except Exception:
            return None
        executable = self._load_object(cache_path, sha)
        if executable is not None or sha in _compile_claims:
            return executable
        try:
            peer_compiled = _claim_compile(cache_path, sha)
        except OSError:
            return None
        if peer_compiled:
            executable = self._load_object(cache_path, sha)
            if executable is not None:
                _release_compile(sha)
        return executable

    @staticmethod
    def _load_object(cache_path: Path, sha: str) -> Any | None:
        try:
            o_path = cache_path / f"{sha}.o"
            lock_path = cache_path / f"{sha}.lock"
            with FileLock(lock_path, exclusive=False, timeout=LOCK_TIMEOUT):
                if not _is_owned_object(o_path):
                    return None
                try:
                    m = _cute().runtime.load_module(str(o_path), enable_tvm_ffi=True)
                except Exception:
                    # A .o that fails to load is corrupt (e.g. a truncated
                    # artifact left by an older build predating the atomic-write
                    # save path). Purge it so the caller recompiles instead of
                    # every worker re-tripping over the same bad file; missing_ok
                    # tolerates a peer racing the unlink. Scoped to load_module
                    # so lock-acquisition or transient FS errors don't delete a
                    # healthy artifact.
                    try:
                        o_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    return None
                return m[EXPORT_FUNC_NAME]
        except Exception:
            # Lock timeout or other transient failure: leave the artifact in
            # place and let the caller recompile for this attempt.
            pass
        return None
