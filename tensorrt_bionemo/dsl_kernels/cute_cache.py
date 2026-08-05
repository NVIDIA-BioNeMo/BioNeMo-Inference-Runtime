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

"""Persistent .o cache for CuTe DSL compiled kernels.

Compiled kernels are exported as object files (.o) via ``export_to_c``.
On subsequent runs the .o is loaded (~1 ms) instead of re-generating
IR + re-JIT'ing (~100 ms per kernel).

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
  BIONEMO_KERNEL_CACHE_ENABLED=0  — disable persistent .o cache (default: enabled)
  BIONEMO_KERNEL_CACHE_DIR=path   — override default cache directory

Adapted from quack.cache_utils (Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao).
"""

from __future__ import annotations

import functools
import hashlib
import os
import pickle
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
    "get_cache_dir",
    "FileLock",
]

CACHE_ENABLED: bool = DiskCache.ENABLED
CACHE_DIR: str | None = DiskCache._CACHE_DIR

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
    # Separate SKUs that share a compute capability but differ in SM count
    # (H20's 78 vs H100/H200's 114-144) — see _device_sm_count.
    h.update(f"sm_count={_device_sm_count()}".encode())

    dsl_kernels_dir = Path(__file__).resolve().parent
    _hash_source_dir(h, dsl_kernels_dir)

    for extra_dir in EXTRA_SOURCE_DIRS:
        _hash_source_dir(h, Path(extra_dir).resolve())

    return h.hexdigest()


def _device_sm_count() -> int:
    """SM count of the active CUDA device (0 if unavailable).

    Folded into the cache fingerprint so a kernel compiled for one SKU isn't
    reused on another with a different SM count: CuTe persistent kernels bake
    launch geometry (grid / cluster sizing) from the runtime SM count at compile
    time, so an H20 (78 SMs) replaying an H100/H200 (114-144) kernel breaks bf16
    parity. Compute capability (sm90) can't separate them; SM count can. Assumes
    one device per process (the fingerprint is lru_cached on first use).
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
            from tensorrt_bionemo.logger import logger

            logger.warning(f"bionemo kernel cache: export failed for key {sha}: {e}")

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
