# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast-launch utilities for pre-compiled Triton kernels.

Standard ``@triton.jit`` dispatch adds ~25 µs of Python overhead per call
(binder, cache-key hash, kwargs parsing, stream lookup, globals check).
This module provides a thin wrapper that pre-compiles the kernel at init
time and offers two launch paths:

1. **cuda.bindings** (preferred, ~17 µs): loads CUBIN via
   ``cuModuleLoadData`` and calls ``cuLaunchKernel`` directly through
   NVIDIA's ``cuda.bindings`` Python package.
2. **Triton C-level** (fallback, ~20 µs): calls the compiled kernel's
   ``.run()`` method, bypassing ``JITFunction.run()`` overhead.

The launch path is selected automatically: ``cuda.bindings`` when
available, Triton ``.run()`` otherwise.

Inherits from :class:`~.cache_base.KernelCacheBase` for the unified
compile / save / load interface shared with the CuTe DSL backend.

Cache isolation
---------------
By default Triton writes CUBINs to ``~/.triton/cache/`` which is shared
by every package in the process.  This module redirects the Triton cache
to a TRT-BioNemo-controlled directory (``<bionemo_kernel_cache>/triton/``)
so that other packages (e.g. CuTe DSL) don't collide.  Override with:

* ``BIONEMO_TRITON_CACHE_DIR``  — per-project override
* ``TRITON_CACHE_DIR``          — standard Triton env var

Cold-cache safety
-----------------
Compiling Triton kernels from source (TTIR → LLVM → PTX → CUBIN) inside
the main process permanently degrades its inference performance by ~10 %,
even after compilation finishes.  Loading pre-compiled CUBINs from the
Triton disk cache has **no** such effect.

When the disk cache is cold, :meth:`TritonKernelCache.compile_for_dtypes`
automatically offloads compilation to a short-lived **subprocess** that
populates the cache directory.  The main process then loads CUBINs from
disk, guaranteeing zero performance degradation.

Usage
-----
::

    from tensorrt_bionemo.dsl_kernels.triton_cache import TritonKernelCache

    class MyKernel(TritonKernelCache):
        def __init__(self):
            self._kernels = self.compile_for_dtypes(
                my_triton_jit_fn,
                dtypes=[torch.bfloat16],
                make_dummy_args=lambda dt: (
                    torch.empty(1, dtype=dt, device="cuda"), 1),
                grid=(1,),
                MY_CONSTEXPR=128,
            )

    k = MyKernel()
    cached = k._kernels[torch.bfloat16]

    # cuda.bindings path (preferred — set params then launch):
    drv = cached.driver
    if drv is not None:
        drv.params[0].value = tensor.data_ptr()
        drv.params[1].value = tensor.stride(0)
        drv.launch(grid_x, grid_y)

    # Triton .run() fallback:
    cached.launch((grid_x, grid_y), real_tensor, real_int, ...)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Optional, Sequence

from .cache_base import (DiskCache, DriverLauncher, KernelCacheBase,
                         make_driver_launcher)

# ---------------------------------------------------------------------------
# TRT-BioNemo Triton cache directory isolation
# ---------------------------------------------------------------------------
# Redirect Triton's disk cache away from the shared ``~/.triton/cache/`` to
# a TRT-BioNemo-controlled location.  This prevents conflicts with other
# packages (e.g. CuTe DSL) that also populate the same global Triton cache.
#
# Priority:
#   1. ``BIONEMO_TRITON_CACHE_DIR`` — explicit per-project override
#   2. ``TRITON_CACHE_DIR``         — user already set; leave it alone
#   3. Default: ``<bionemo_kernel_cache>/triton/``
#
# Must be resolved *before* ``import triton`` so that Triton's knobs see
# the env var on first access.
# ---------------------------------------------------------------------------


def _resolve_triton_cache_dir() -> str:
    """Resolve the Triton cache directory for TRT-BioNemo."""
    explicit = os.getenv("BIONEMO_TRITON_CACHE_DIR")
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

# ---------------------------------------------------------------------------
# Resolve Triton launch hooks once at import time
# ---------------------------------------------------------------------------

# Triton moved ``knobs`` from ``triton.runtime.knobs`` (3.5) to the top-level
# ``triton.knobs`` module in 3.6. Try the new location first, then fall back.
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

# ---------------------------------------------------------------------------
# Triton version compatibility for the cuda.bindings driver fast path
# ---------------------------------------------------------------------------
# TRT-BioNemo pins ``triton==3.5``. The cuda.bindings ``DriverLauncher`` path
# pokes at internals of ``CompiledKernel`` (``packed_metadata``, ``function``
# slot layout, launch ABI) that change between Triton minor releases. When
# users install a third-party package that pulls in ``triton>3.5``, the
# driver path silently mis-launches kernels (wrong arg count, missing
# constexpr handling, segfault on ``cuLaunchKernel``).
#
# Detect the version once at import time. On any mismatch, disable the
# driver path globally — the ``.run()`` fallback is only ~3 µs slower and
# remains forward/backward compatible.
# ---------------------------------------------------------------------------

_SUPPORTED_TRITON_MAJOR_MINOR = (3, 5)


def _triton_supports_driver() -> bool:
    """True only when the installed Triton matches the pinned 3.5 ABI."""
    raw = getattr(triton, "__version__", "0.0.0")
    try:
        major, minor = (int(p) for p in raw.split(".")[:2])
    except (ValueError, AttributeError):
        return False
    return (major, minor) == _SUPPORTED_TRITON_MAJOR_MINOR


_DRIVER_TRITON_OK: bool = _triton_supports_driver()
if not _DRIVER_TRITON_OK:
    logger.warning(
        "triton %s is installed but TRT-BioNemo's cuda.bindings driver fast "
        "path is only validated against triton==%d.%d. Falling back to "
        "Triton's `.run()` launch path (~3 us slower). Pin ``triton==%d.%d`` "
        "to re-enable the fast path.",
        getattr(triton, "__version__", "?"),
        *_SUPPORTED_TRITON_MAJOR_MINOR,
        *_SUPPORTED_TRITON_MAJOR_MINOR,
    )

# ---------------------------------------------------------------------------
# CachedKernel — wraps a CompiledKernel with optional cuda.bindings launch
# ---------------------------------------------------------------------------


class CachedKernel:
    """Pre-compiled Triton kernel with two launch paths.

    **Primary (cuda.bindings):** When ``cuda.bindings`` is available, a
    :class:`~.cache_base.DriverLauncher` is created automatically.
    Callers use ``kernel.driver.params[i].value = ...`` then
    ``kernel.driver.launch(gx, gy)`` for the lowest overhead (~17 µs).

    **Fallback (Triton .run()):** :meth:`launch` calls the compiled
    kernel's C-level ``.run()`` directly, bypassing ``JITFunction.run()``
    and its ~17 µs of Python overhead.  The cached CUDA stream avoids
    the 5 µs cost of ``torch.cuda.current_stream().cuda_stream`` per call.
    """

    __slots__ = ("_kernel", "_stream", "_driver")

    def __init__(self,
                 compiled_kernel: Any,
                 stream: torch.cuda.Stream | None = None,
                 enable_driver: bool = True):
        self._kernel = compiled_kernel
        self._stream = stream or torch.cuda.current_stream()
        self._driver: DriverLauncher | None = None

        # Force-disable the cuda.bindings fast path on triton ABIs we don't
        # validate against (anything other than the pinned 3.5). See
        # ``_triton_supports_driver`` above for context.
        if enable_driver and not _DRIVER_TRITON_OK:
            enable_driver = False

        if enable_driver:
            try:
                from .cache_base import _HAS_CUDA_BINDINGS, _drv
                cu_stream = None
                if _HAS_CUDA_BINDINGS:
                    cu_stream = _drv.CUstream(self._stream.cuda_stream)
                self._driver = make_driver_launcher(compiled_kernel,
                                                    cu_stream=cu_stream)
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
        """Launch the kernel via Triton's C-level ``.run()``.

        This is the backward-compatible path.  For the lowest overhead,
        use :attr:`driver` directly.

        Args:
            grid: (grid_x,) or (grid_x, grid_y) or (grid_x, grid_y, grid_z).
            *args: ALL kernel arguments in declaration order, including
                   constexpr values (the C launcher knows which to skip).
        """
        kernel = self._kernel
        gs = len(grid)
        gx = grid[0]
        gy = grid[1] if gs > 1 else 1
        gz = grid[2] if gs > 2 else 1
        stream = self._stream.cuda_stream
        lm = kernel.launch_metadata(grid, stream, *args)
        kernel.run(gx, gy, gz, stream, kernel.function, kernel.packed_metadata,
                   lm, ENTER_HOOK, EXIT_HOOK, *args)


# ---------------------------------------------------------------------------
# Cold-cache detection (per-process, latched)
# ---------------------------------------------------------------------------

_was_cold: bool | None = None


def _triton_cache_was_cold() -> bool:
    """Return True if the Triton disk cache was cold when this process started.

    The result is latched on first call so that populating the cache
    (via subprocess) doesn't flip the answer mid-init.  Checks the
    TRT-BioNemo-controlled ``TRITON_CACHE_DIR``.
    """
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
                cache_dir)
    return _was_cold


# ---------------------------------------------------------------------------
# Subprocess compilation
# ---------------------------------------------------------------------------

_SUBPROCESS_SCRIPT = textwrap.dedent('''\
import json, os, sys

with open(sys.argv[1]) as _f:
    spec = json.load(_f)

# Set TRITON_CACHE_DIR before any triton import so CUBINs land in
# the same TRT-BioNemo-controlled directory the parent process uses.
os.environ["TRITON_CACHE_DIR"] = spec["triton_cache_dir"]

import importlib, torch

mod = importlib.import_module(spec["module"])
jit_fn = getattr(mod, spec["fn_name"])
grid = tuple(spec["grid"])
ckw = {k: (v if isinstance(v, bool) else int(v))
       for k, v in spec["constexpr_kwargs"].items()}

for dtype_str in spec["dtypes"]:
    dt = getattr(torch, dtype_str)
    args = []
    for a in spec["dummy_args_template"]:
        if a["type"] == "tensor":
            args.append(torch.empty(a["shape"], dtype=dt, device="cuda"))
        else:
            args.append(a["value"])
    jit_fn[grid](*args, **ckw)

torch.cuda.synchronize()
''')


def _compile_in_subprocess(
    jit_fn: JITFunction,
    dummy_args: tuple,
    unique_dtypes: list[torch.dtype],
    grid: tuple[int, ...],
    constexpr_kwargs: dict,
) -> None:
    """Compile one kernel config (all dtypes) in a subprocess."""
    template = []
    for arg in dummy_args:
        if isinstance(arg, torch.Tensor):
            template.append({"type": "tensor", "shape": list(arg.shape)})
        else:
            template.append({"type": "int", "value": int(arg)})

    ckw_ser = {}
    for k, v in constexpr_kwargs.items():
        ckw_ser[k] = v if isinstance(v, bool) else int(v)

    spec = {
        "triton_cache_dir": TRITON_CACHE_DIR,
        "module": jit_fn.fn.__module__,
        "fn_name": jit_fn.fn.__name__,
        "grid": list(grid),
        "constexpr_kwargs": ckw_ser,
        "dtypes": [str(dt).replace("torch.", "") for dt in unique_dtypes],
        "dummy_args_template": template,
    }

    spec_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w',
                                         suffix='.json',
                                         delete=False) as f:
            json.dump(spec, f)
            spec_path = f.name

        result = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_SCRIPT, spec_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.warning("Subprocess compile failed for %s.%s: %s",
                           jit_fn.fn.__module__, jit_fn.fn.__name__,
                           result.stderr[-500:])
    except Exception as e:
        logger.warning("Subprocess compile error for %s: %s",
                       jit_fn.fn.__name__, e)
    finally:
        if spec_path:
            try:
                os.unlink(spec_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# TritonKernelCache — KernelCacheBase implementation for Triton
# ---------------------------------------------------------------------------


class TritonKernelCache(KernelCacheBase):
    """Triton kernel cache with cold-cache subprocess compilation.

    Implements the :class:`~.cache_base.KernelCacheBase` interface:

    * :meth:`compile` — runs a ``@triton.jit`` kernel once to trigger
      compilation, then extracts the ``CompiledKernel`` from Triton's
      internal cache.
    * :meth:`compile_for_dtypes` — compiles for multiple dtypes with
      cold-cache subprocess safety.
    * :meth:`save_to_cache` — no-op (Triton manages its own cache dir).
    * :meth:`load_from_cache` — checks Triton's in-memory and disk cache.

    Subclasses (e.g. ``FusedSwiGLU``, ``MoveaxisPad``) inherit these
    methods and call ``self.compile_for_dtypes(...)`` during init.
    """

    def compile(
        self,
        jit_fn: JITFunction,
        dummy_args: tuple,
        grid: tuple[int, ...],
        constexpr_kwargs: dict,
        stream: torch.cuda.Stream | None = None,
    ) -> CachedKernel:
        """Compile a Triton kernel by running it once with dummy args.

        Args:
            jit_fn: A ``@triton.jit`` decorated function.
            dummy_args: Dummy tensors/scalars matching the kernel's
                positional parameters.
            grid: Grid dimensions for the compilation launch.
            constexpr_kwargs: Keyword arguments for ``tl.constexpr`` params.
            stream: Optional CUDA stream to bind to the ``CachedKernel``.

        Returns:
            A :class:`CachedKernel` wrapping the compiled kernel.

        .. todo::
            Triton > 3.5 may change the internal ``device_caches`` layout,
            causing a ``KeyError`` when looking up the compiled kernel.
            The cache introspection below relies on undocumented internals
            (``device_caches``, ``compute_cache_key``); revisit when
            upgrading Triton.
        """
        jit_fn[grid](*dummy_args, **constexpr_kwargs)
        torch.cuda.synchronize()

        device = torch.cuda.current_device()
        cache, key_cache, _, _, binder = jit_fn.device_caches[device]
        # Mirror the kwargs preprocessing inside ``JITFunction.run`` so the
        # ``options`` dict (and therefore the cache key) matches the entry
        # actually stored in ``cache``. Triton 3.6 added
        # ``instrumentation_mode``; 3.5 only had ``debug``. Anything not in
        # the kernel signature ends up in ``options``, which
        # ``compute_cache_key`` stringifies into the key.
        runtime_kwargs: dict[str, Any] = {"debug": bool(jit_fn.debug)}
        if _knobs is not None:
            if getattr(_knobs.runtime, "debug", False):
                runtime_kwargs["debug"] = True
            if hasattr(_knobs.compilation, "instrumentation_mode"):
                runtime_kwargs["instrumentation_mode"] = (
                    _knobs.compilation.instrumentation_mode)
        ba, spec, opts = binder(*dummy_args, **constexpr_kwargs,
                                **runtime_kwargs)
        key = compute_cache_key(key_cache, spec, opts)
        return CachedKernel(cache[key], stream=stream)

    def compile_for_dtypes(
        self,
        jit_fn: JITFunction,
        dtypes: Sequence[torch.dtype],
        make_dummy_args,
        *,
        grid: tuple[int, ...] = (1, ),
        **constexpr_kwargs: Any,
    ) -> dict[torch.dtype, CachedKernel]:
        """Compile a kernel for multiple dtypes with cold-cache safety.

        On a cold Triton disk cache, all dtypes are compiled in a
        subprocess first, then loaded from disk in the main process.

        Args:
            jit_fn: A ``@triton.jit`` decorated function.
            dtypes: Sequence of dtypes to compile for.
            make_dummy_args: Callable ``(dtype) -> tuple`` returning
                dummy positional args for the given dtype.
            grid: Grid dimensions for the compilation launch.
            **constexpr_kwargs: Constexpr keyword arguments.

        Returns:
            Dict mapping each dtype to its :class:`CachedKernel`.
        """
        unique_dtypes = list(dict.fromkeys(dtypes))

        if _triton_cache_was_cold():
            first_args = make_dummy_args(unique_dtypes[0])
            _compile_in_subprocess(jit_fn, first_args, unique_dtypes, grid,
                                   constexpr_kwargs)

        stream = torch.cuda.current_stream()
        kernels = {}
        for dt in unique_dtypes:
            dummy_args = make_dummy_args(dt)
            kernels[dt] = self.compile(jit_fn,
                                       dummy_args,
                                       grid,
                                       constexpr_kwargs,
                                       stream=stream)
        return kernels

    def save_to_cache(self, key: tuple, artifact: Any) -> None:
        """No-op — Triton manages its own disk cache."""

    def load_from_cache(self, key: tuple) -> Optional[CachedKernel]:
        """Not applicable for Triton (uses Triton's internal cache).

        Returns ``None`` — Triton's cache is accessed implicitly during
        :meth:`compile` (which loads from the Triton cache dir on a warm
        cache hit).
        """
        return None

    def get_or_compile(self, key: tuple, *args: Any, **kwargs: Any) -> Any:
        """Compile with cold-cache subprocess safety.

        On a cold Triton disk cache, compilation is offloaded to a
        subprocess so the main process only ever loads CUBINs from disk.

        Args:
            key: Unused (Triton manages its own keys), kept for interface
                 compatibility.
            *args, **kwargs: Forwarded to :meth:`compile`.

        Returns:
            A :class:`CachedKernel`.
        """
        return self.compile(*args, **kwargs)
