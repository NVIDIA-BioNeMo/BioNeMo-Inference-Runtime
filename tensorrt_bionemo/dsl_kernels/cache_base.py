# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified compilation and caching infrastructure for DSL kernels.

Provides shared abstractions used by both Triton and CuTe DSL backends:

* :class:`KernelCacheBase` — abstract base class with ``compile``,
  ``save_to_cache``, and ``load_from_cache`` methods.
* :class:`FileLock` — advisory file locking for concurrent cache access.
* :class:`DiskCache` — persistent artifact cache with file locking.
* :class:`DriverLauncher` — ``cuda.bindings`` ``cuLaunchKernel`` for
  minimal Python dispatch overhead (~17 µs vs ~25 µs for Triton's ``.run()``).
* :func:`parse_ptx_params` — auto-discover kernel parameter layouts from PTX.
* :func:`make_driver_launcher` — factory that builds a
  :class:`DriverLauncher` from a Triton ``CompiledKernel``.

Concrete implementations:

* :mod:`~.cute_cache` — CuTe DSL ``.o`` caching via ``export_to_c``/``load_module``.
* :mod:`~.triton_cache` — Triton CUBIN caching with cold-cache subprocess safety.
"""

from __future__ import annotations

import abc
import atexit
import ctypes
import fcntl
import hashlib
import logging
import os
import pickle
import re
import tempfile
import time
from getpass import getuser
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional cuda.bindings import
# ---------------------------------------------------------------------------

try:
    from cuda.bindings import driver as _drv
    _HAS_CUDA_BINDINGS = True
except ImportError:
    _drv = None  # type: ignore[assignment]
    _HAS_CUDA_BINDINGS = False

_cuda_initialized = False
_loaded_cu_modules: set = set()


def _ensure_cuda_init():
    global _cuda_initialized
    if _cuda_initialized or not _HAS_CUDA_BINDINGS:
        return
    (err, ) = _drv.cuInit(0)
    if err == _drv.CUresult.CUDA_SUCCESS:
        _cuda_initialized = True


def _unload_all_cu_modules():
    """Unload every CUmodule tracked by :func:`make_driver_launcher`."""
    for mod in list(_loaded_cu_modules):
        try:
            _drv.cuModuleUnload(mod)
        except Exception:
            pass
    _loaded_cu_modules.clear()


atexit.register(_unload_all_cu_modules)


def has_cuda_bindings() -> bool:
    """Return True if cuda.bindings is available."""
    return _HAS_CUDA_BINDINGS


# ---------------------------------------------------------------------------
# File locking (shared by Triton and CuTe DSL caches)
# ---------------------------------------------------------------------------

_LOCK_TIMEOUT = 60


class FileLock:
    """Advisory file lock using ``fcntl.flock`` with timeout."""

    def __init__(self, lock_path: Path, exclusive: bool, timeout: float = 15):
        self.lock_path = lock_path
        self.exclusive = exclusive
        self.timeout = timeout
        self._fd: int = -1

    def __enter__(self) -> FileLock:
        flags = (os.O_WRONLY | os.O_CREAT if self.exclusive else os.O_RDONLY
                 | os.O_CREAT)
        lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        self._fd = os.open(str(self.lock_path), flags)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                fcntl.flock(self._fd, lock_type | fcntl.LOCK_NB)
                return self
            except OSError:
                time.sleep(0.1)
        os.close(self._fd)
        self._fd = -1
        raise RuntimeError(f"Timed out waiting for lock: {self.lock_path}")

    def __exit__(self, *exc) -> None:
        if self._fd >= 0:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = -1


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


class DiskCache:
    """Persistent disk cache for compiled kernel artifacts.

    Both Triton and CuTe DSL backends share this infrastructure.
    The cache is keyed by hashable tuples and stores arbitrary binary
    files with file-lock-protected concurrent access.

    Controls:
        BIONEMO_KERNEL_CACHE_ENABLED=0  — disable (default: enabled)
        BIONEMO_KERNEL_CACHE_DIR=path   — override default directory
    """

    ENABLED: bool = os.getenv("BIONEMO_KERNEL_CACHE_ENABLED", "1") == "1"
    _CACHE_DIR: str | None = os.getenv("BIONEMO_KERNEL_CACHE_DIR", None)

    @staticmethod
    def get_cache_dir() -> Path:
        """Return (and create) the root cache directory."""
        if DiskCache._CACHE_DIR is not None:
            d = Path(DiskCache._CACHE_DIR)
        else:
            d = (Path(tempfile.gettempdir()) / getuser() /
                 "bionemo_kernel_cache")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def key_to_hash(key: tuple) -> str:
        return hashlib.sha256(pickle.dumps(key)).hexdigest()

    @classmethod
    def artifact_dir(cls, fingerprint: str | None = None) -> Path:
        """Return the directory for artifacts, optionally scoped by fingerprint."""
        base = cls.get_cache_dir()
        if fingerprint:
            base = base / fingerprint
        base.mkdir(parents=True, exist_ok=True)
        return base


# ---------------------------------------------------------------------------
# Abstract base class — compile / save / load contract
# ---------------------------------------------------------------------------


class KernelCacheBase(abc.ABC):
    """Abstract base for kernel compilation and caching backends.

    Subclasses implement the three core operations:

    1. :meth:`compile` — build a kernel from source / IR.
    2. :meth:`save_to_cache` — persist compiled artifact to disk.
    3. :meth:`load_from_cache` — reload artifact from disk.

    The convenience method :meth:`get_or_compile` chains these:
    load → (miss) → compile → save.
    """

    @abc.abstractmethod
    def compile(self, *args: Any, **kwargs: Any) -> Any:
        """Compile a kernel and return the executable artifact."""
        ...

    @abc.abstractmethod
    def save_to_cache(self, key: tuple, artifact: Any) -> None:
        """Persist *artifact* under *key* in the disk cache."""
        ...

    @abc.abstractmethod
    def load_from_cache(self, key: tuple) -> Optional[Any]:
        """Load artifact for *key* from disk cache, or ``None`` on miss."""
        ...

    def get_or_compile(self, key: tuple, *args: Any, **kwargs: Any) -> Any:
        """Load from cache or compile, save, and return.

        Subclasses may override for backend-specific logic (e.g. Triton's
        subprocess compilation on cold cache).
        """
        artifact = self.load_from_cache(key)
        if artifact is not None:
            return artifact
        artifact = self.compile(*args, **kwargs)
        self.save_to_cache(key, artifact)
        return artifact


# ---------------------------------------------------------------------------
# PTX param introspection
# ---------------------------------------------------------------------------

_PTX_PARAM_RE = re.compile(r'\.param\s+\.(\w+)')


def parse_ptx_params(ptx: str, kernel_name: str) -> list[str]:
    """Extract parameter types from a PTX ``.entry`` block.

    Parses ``.param .u64 ...`` / ``.param .u32 ...`` directives inside
    the kernel's entry point and returns an ordered list of type strings
    (e.g. ``['u64', 'u32', 'u64', 'u32', 'u64', 'u64']``).

    This auto-discovers the exact CUBIN parameter layout so that
    :class:`DriverLauncher` can pack ``kernelParams`` correctly without
    manual param-spec maintenance.
    """
    escaped = re.escape(kernel_name)

    for pattern in [
            # Exact match
            rf'\.entry\s+{escaped}\s*\((.*?)\)',
            # Triton adds specialization suffixes like _0d1d2d3d4c5c6c
            rf'\.entry\s+{escaped}\w*\s*\((.*?)\)',
            # Broad fallback: any entry containing the kernel name
            rf'\.entry\s+\w*{escaped}\w*\s*\((.*?)\)',
    ]:
        match = re.search(pattern, ptx, re.DOTALL)
        if match:
            break
    else:
        return []

    params_block = match.group(1)
    types = []
    for line in params_block.split('\n'):
        m = _PTX_PARAM_RE.search(line)
        if m:
            types.append(m.group(1))
    return types


# ---------------------------------------------------------------------------
# CUDA Driver Launcher (cuda.bindings)
# ---------------------------------------------------------------------------

_CTYPES_MAP = {
    'u64': ctypes.c_uint64,
    'u32': ctypes.c_uint32,
    'u16': ctypes.c_uint16,
    'u8': ctypes.c_uint8,
    'i64': ctypes.c_int64,
    'i32': ctypes.c_int32,
    'f32': ctypes.c_float,
    'f64': ctypes.c_double,
    'b64': ctypes.c_uint64,
    'b32': ctypes.c_uint32,
}


class DriverLauncher:
    """Low-overhead kernel launcher using ``cuda.bindings`` ``cuLaunchKernel``.

    Pre-allocates ``ctypes`` parameter storage at construction time.
    On each :meth:`launch`, only the parameter *values* are updated before
    calling ``cuLaunchKernel`` — achieving ~17 µs total dispatch overhead
    vs ~25 µs for Triton's C-level ``.run()`` path.

    Parameters are exposed via the :attr:`params` list for direct value
    assignment on the hot path::

        launcher.params[0].value = tensor.data_ptr()   # pointer
        launcher.params[1].value = tensor.stride(0)     # stride (u32)
        launcher.launch(grid_x, grid_y)

    Requires ``pip install cuda-python`` (the ``cuda.bindings`` package).
    """

    __slots__ = ('_func', '_module', '_block_x', '_shmem', '_stream',
                 '_stream_handle', 'params', '_kp', '_n_params')

    def __init__(
        self,
        cu_function: Any,
        num_warps: int,
        shared_mem: int,
        param_types: list[str],
        cu_stream: Any = None,
        cu_module: Any = None,
    ):
        if not _HAS_CUDA_BINDINGS:
            raise ImportError("cuda.bindings required for DriverLauncher "
                              "(pip install cuda-python)")

        self._func = cu_function
        self._module = cu_module
        self._block_x = num_warps * 32
        self._shmem = shared_mem

        if shared_mem > 48 * 1024:
            attr = (_drv.CUfunction_attribute.
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)
            (err, ) = _drv.cuFuncSetAttribute(cu_function, attr, shared_mem)
            if err != _drv.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"cuFuncSetAttribute({attr.name}, {shared_mem}) "
                    f"failed for CUfunction {cu_function}: {err.name} "
                    f"({int(err)})")

        if cu_stream is None:
            import torch
            cu_stream = _drv.CUstream(torch.cuda.current_stream().cuda_stream)
        self._stream = cu_stream
        # Raw handle backing self._stream. launch() compares the current
        # stream's handle against this to detect a stream switch and refresh
        # self._stream, so the kernel always runs on the active stream rather
        # than a stream cached at construction. Initialized to None so the
        # first launch always re-reads the current stream.
        self._stream_handle = None

        self.params: list = []
        for pt in param_types:
            ct = _CTYPES_MAP.get(pt)
            if ct is None:
                raise ValueError(f"Unknown param type: {pt!r}")
            self.params.append(ct(0))

        self._n_params = len(self.params)
        self._kp = (ctypes.c_void_p * self._n_params)(*(ctypes.addressof(p)
                                                        for p in self.params))

    def launch(self, grid_x: int, grid_y: int = 1, grid_z: int = 1) -> None:
        """Launch the kernel.  Caller must set ``params[i].value`` first."""
        # Honor the CURRENT stream at launch time. Only rebuild the CUstream
        # wrapper when the active stream actually changes, so the common
        # single-stream case stays cheap while side/capture streams (CUDA-graph
        # warmup/capture) are still respected.
        import torch
        cur = torch.cuda.current_stream().cuda_stream
        if cur != self._stream_handle:
            self._stream = _drv.CUstream(cur)
            self._stream_handle = cur
        (err, ) = _drv.cuLaunchKernel(
            self._func,
            grid_x,
            grid_y,
            grid_z,
            self._block_x,
            1,
            1,
            self._shmem,
            self._stream,
            self._kp,
            0,
        )
        if err != _drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLaunchKernel failed: {err}")


# ---------------------------------------------------------------------------
# Factory: Triton CompiledKernel → DriverLauncher
# ---------------------------------------------------------------------------


def make_driver_launcher(
    compiled_triton_kernel: Any,
    param_types: list[str] | None = None,
    cu_stream: Any = None,
) -> DriverLauncher | None:
    """Create a :class:`DriverLauncher` from a Triton ``CompiledKernel``.

    Loads the CUBIN via ``cuModuleLoadData`` and extracts the
    ``CUfunction``.  Parameter types are auto-discovered from the PTX
    if *param_types* is not provided.

    Returns ``None`` when ``cuda.bindings`` is not available or the
    kernel cannot be loaded.
    """
    if not _HAS_CUDA_BINDINGS:
        return None

    _ensure_cuda_init()

    ck = compiled_triton_kernel
    cubin = ck.asm.get("cubin")
    name = ck.metadata.name
    if cubin is None:
        logger.debug("No CUBIN in compiled kernel for %s", name)
        return None

    err, cu_module = _drv.cuModuleLoadData(cubin)
    if err != _drv.CUresult.CUDA_SUCCESS:
        logger.warning("cuModuleLoadData failed for %s: %s", name, err)
        return None

    err, cu_function = _drv.cuModuleGetFunction(cu_module,
                                                name.encode("utf-8"))
    if err != _drv.CUresult.CUDA_SUCCESS:
        logger.warning("cuModuleGetFunction failed for %s: %s", name, err)
        _drv.cuModuleUnload(cu_module)
        return None

    if param_types is None:
        ptx = ck.asm.get("ptx", "")
        if ptx:
            param_types = parse_ptx_params(ptx, name)
        if not param_types:
            logger.debug("Cannot auto-discover param types for %s", name)
            _drv.cuModuleUnload(cu_module)
            return None

    _loaded_cu_modules.add(cu_module)

    logger.debug("DriverLauncher for %s: %d params %s", name, len(param_types),
                 param_types)

    import torch
    if cu_stream is None:
        cu_stream = _drv.CUstream(torch.cuda.current_stream().cuda_stream)

    return DriverLauncher(
        cu_function,
        num_warps=ck.metadata.num_warps,
        shared_mem=ck.metadata.shared,
        param_types=param_types,
        cu_stream=cu_stream,
        cu_module=cu_module,
    )
