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
  CUTEDSL_FORCE_CUBIN=1           — resolve executables from the packaged CUBIN
                                    library instead of compiling from source

"""

from __future__ import annotations

import functools
import hashlib
import os
import pickle
import platform
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cutlass
import cutlass.cute as cute

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
]

CACHE_ENABLED: bool = DiskCache.ENABLED
CACHE_DIR: str | None = DiskCache._CACHE_DIR

FORCE_CUBIN_ENV = "CUTEDSL_FORCE_CUBIN"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

EXTRA_SOURCE_DIRS: list[Path] = []

EXPORT_FUNC_NAME = "kernel"
LOCK_TIMEOUT = 60


def get_cache_dir() -> Path:
    """Return (and create) the root cache directory."""
    return DiskCache.get_cache_dir()


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
        cache_path = get_cache_dir() / _compute_source_fingerprint()
        cache_path.mkdir(parents=True, exist_ok=True)
        o_path = cache_path / f"{sha}.o"
        lock_path = cache_path / f"{sha}.lock"

        try:
            with FileLock(lock_path, exclusive=True, timeout=LOCK_TIMEOUT):
                if o_path.exists():
                    return
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

    def load_from_cache(self, key: tuple) -> Any | None:
        """Load a compiled CuTe DSL kernel from the disk cache.

        Args:
            key: Same hashable tuple used in :meth:`save_to_cache`.

        Returns:
            The loaded callable if a cache hit, or ``None``.
        """
        if not CACHE_ENABLED:
            return None

        sha = _key_to_hash(key)
        cache_path = get_cache_dir() / _compute_source_fingerprint()
        o_path = cache_path / f"{sha}.o"
        lock_path = cache_path / f"{sha}.lock"

        try:
            with FileLock(lock_path, exclusive=False, timeout=LOCK_TIMEOUT):
                if not o_path.exists():
                    return None
                try:
                    m = cute.runtime.load_module(str(o_path), enable_tvm_ffi=True)
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
