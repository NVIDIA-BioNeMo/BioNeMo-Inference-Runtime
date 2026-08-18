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
                grid=(1,),
                BLOCK_SIZE=128,
            )

    kernel = MyKernel().kernels[torch.bfloat16]
    if kernel.driver is not None:
        kernel.driver.params[0].value = tensor.data_ptr()
        kernel.driver.launch(1)
    else:
        kernel.launch((1,), tensor, 128)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Sequence
from typing import Any

from .cache_base import DiskCache, DriverLauncher, KernelCacheBase, make_driver_launcher

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

# The direct launcher depends on Triton's private CompiledKernel ABI.

_SUPPORTED_TRITON_MAJOR_MINOR = (3, 6)


def _triton_supports_driver() -> bool:
    """True only when the installed Triton matches the pinned 3.6 ABI."""
    raw = getattr(triton, "__version__", "0.0.0")
    try:
        major, minor = (int(p) for p in raw.split(".")[:2])
    except (ValueError, AttributeError):
        return False
    return (major, minor) == _SUPPORTED_TRITON_MAJOR_MINOR


_DRIVER_TRITON_OK: bool = _triton_supports_driver()


class CachedKernel:
    """Compiled Triton kernel with direct-driver and ``.run()`` launch paths."""

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
        """Launch through Triton's C-level ``.run()`` fallback.

        Args:
            grid: (grid_x,) or (grid_x, grid_y) or (grid_x, grid_y, grid_z).
            *args: Kernel arguments in declaration order, including constexprs.
        """
        kernel = self._kernel
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


_SUBPROCESS_SCRIPT = textwrap.dedent("""\
import json, os, sys

with open(sys.argv[1]) as _f:
    spec = json.load(_f)

# Match the parent process's cache before importing Triton.
os.environ["TRITON_CACHE_DIR"] = spec["triton_cache_dir"]

import importlib, torch

mod = importlib.import_module(spec["module"])
jit_fn = getattr(mod, spec["fn_name"])
grid = tuple(spec["grid"])
ckw = {k: (v if isinstance(v, bool) else int(v))
       for k, v in spec["constexpr_kwargs"].items()}

for variant in spec["variants"]:
    args = []
    for a in variant:
        if a["type"] == "tensor":
            args.append(torch.empty(a["shape"], dtype=getattr(torch, a["dtype"]), device="cuda"))
        else:
            args.append(a["value"])
    jit_fn[grid](*args, **ckw)

torch.cuda.synchronize()
""")


def _compile_in_subprocess(
    jit_fn: JITFunction,
    dummy_args_by_dtype: dict[torch.dtype, tuple],
    grid: tuple[int, ...],
    constexpr_kwargs: dict,
) -> None:
    """Compile one kernel config (all dtypes) in a subprocess."""
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
            else:
                variant.append({"type": "int", "value": int(arg)})
        variants.append(variant)

    ckw_ser = {}
    for k, v in constexpr_kwargs.items():
        ckw_ser[k] = v if isinstance(v, bool) else int(v)

    spec = {
        "triton_cache_dir": TRITON_CACHE_DIR,
        "module": jit_fn.fn.__module__,
        "fn_name": jit_fn.fn.__name__,
        "grid": list(grid),
        "constexpr_kwargs": ckw_ser,
        "variants": variants,
    }

    spec_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(spec, f)
            spec_path = f.name

        result = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_SCRIPT, spec_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.warning(
                "Subprocess compile failed for %s.%s: %s",
                jit_fn.fn.__module__,
                jit_fn.fn.__name__,
                result.stderr[-500:],
            )
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
            grid: Grid dimensions for the compilation launch.
            constexpr_kwargs: Keyword arguments for ``tl.constexpr`` params.
            stream: Optional launch stream.

        Returns:
            The compiled kernel wrapper.
        """
        compiled = jit_fn[grid](*dummy_args, **constexpr_kwargs)
        torch.cuda.synchronize()

        if not _looks_like_compiled_kernel(compiled):
            compiled = self._lookup_compiled(jit_fn, dummy_args, constexpr_kwargs)
        return CachedKernel(compiled, stream=stream)

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
        grid: tuple[int, ...] = (1,),
        **constexpr_kwargs: Any,
    ) -> dict[torch.dtype, CachedKernel]:
        """Compile one kernel for each dtype.

        Args:
            jit_fn: A ``@triton.jit`` decorated function.
            dtypes: Sequence of dtypes to compile for.
            make_dummy_args: Callable returning positional arguments by dtype.
            grid: Compilation grid.
            **constexpr_kwargs: Constexpr keyword arguments.

        Returns:
            Compiled kernels keyed by dtype.
        """
        unique_dtypes = list(dict.fromkeys(dtypes))
        dummy_args_by_dtype = {dtype: make_dummy_args(dtype) for dtype in unique_dtypes}

        if _triton_cache_was_cold():
            _compile_in_subprocess(jit_fn, dummy_args_by_dtype, grid, constexpr_kwargs)

        stream = torch.cuda.current_stream()
        kernels = {}
        for dt in unique_dtypes:
            dummy_args = dummy_args_by_dtype[dt]
            kernels[dt] = self.compile(jit_fn, dummy_args, grid, constexpr_kwargs, stream=stream)
        return kernels

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
