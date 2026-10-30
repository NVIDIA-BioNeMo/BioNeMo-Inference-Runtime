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
"""Source-or-CUBIN interface for the SM90 transition with its LayerNorm."""

from __future__ import annotations

import contextlib
from typing import Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryError,
    CuTeDSLKernelLibraryExecutable,
    launch_compiled_kernel,
    load_source_module,
    populate_compiled_cache_from_library,
)
from bionemo_ir.dsl_kernels.cute_cache import FORCE_CUBIN_ENV, CuteKernelCache
from bionemo_ir.logger import logger

from . import _config
from ._cubin import TransitionMlpWNormCubinExecutable

__all__ = ["TransitionMlpWNormCuTe"]


class TransitionMlpWNormCuTe(CuteKernelCache):
    """Cached backend for ``[z +] mask * (act(LayerNorm(z) @ W1.T + b1) @ W2.T + b2)`` in one kernel.

    Each shape has two images. One takes every launch layout (:func:`._config.choose_layout`); the persistent one
    takes the full tile at split 1 only, with that layout fixed while tracing, so its code is the kernel's from before
    it took a layout. A call whose layout is the full tile at split 1 runs the persistent image.

    Args:
        sm_version: The SM whose configs and kernel to use; defaults to this GPU's.
    """

    _compiled_cache: dict[tuple, Any] = {}
    # Per shape: the source kernel's (rows per full tile, warpgroups per CTA, layout mask)
    _source_geometry: dict[tuple, tuple[int, int, int]] = {}
    # Per shape, path, device and row count: the call's (tile_wgs, split, persistent image)
    _layout_cache: dict[tuple, tuple[int, int, bool]] = {}
    # Per device index: its SM count
    _device_sms: dict[int, int] = {}

    def __init__(self, sm_version: int | None = None) -> None:
        if sm_version is None:
            major, minor = torch.cuda.get_device_capability()
            sm_version = major * 10 + minor
        self._sm_version = sm_version
        self._num_sms: int | None = None

    @property
    def sm_version(self) -> int:
        """The SM whose configs and kernel this backend uses."""
        return self._sm_version

    def ships(self, width: int, hidden: int, activation: str, *, candidates: bool = False) -> bool:
        """Whether this build can run ``(width, hidden, activation)``, from source or from both of its packaged
        CUBINs; with ``candidates``, a shape whose config entry is a candidate counts too."""
        if not _config.ships(self._sm_version, width, hidden, activation, candidates=candidates):
            return False
        if not self.force_cubin() and self._source_available():
            return True
        try:
            for persistent in (False, True):
                key = self._key(width, hidden, activation, persistent)
                self._load_cubin_executable(key, width, hidden, activation, persistent)
        except (CuTeDSLKernelLibraryError, RuntimeError):
            return False
        return True

    @staticmethod
    def _source_available() -> bool:
        try:
            load_source_module(__package__)
        except ImportError:
            return False
        return True

    def _key(self, width: int, hidden: int, activation: str, persistent: bool = False) -> tuple:
        return (self._sm_version, width, hidden, activation, persistent)

    def _describe(self, width: int, hidden: int, activation: str, persistent: bool = False) -> str:
        image = "persistent" if persistent else "layouts"
        return f"SM{self._sm_version}, width={width}, hidden={hidden}, activation={activation}, image={image}"

    def _load_cubin_executable(
        self,
        key: tuple,
        width: int,
        hidden: int,
        activation: str,
        persistent: bool = False,
        source_error: Exception | None = None,
    ) -> Any:
        if not isinstance(TransitionMlpWNormCuTe._compiled_cache.get(key), CuTeDSLKernelLibraryExecutable):
            # A source build cached under the key would pass for the CUBIN; only the library proves that one ships.
            TransitionMlpWNormCuTe._compiled_cache.pop(key, None)
        try:
            executable = populate_compiled_cache_from_library(
                TransitionMlpWNormCuTe._compiled_cache,
                key,
                "transition_mlp_w_norm",
                lambda library, launcher: TransitionMlpWNormCubinExecutable(
                    library, launcher, self._sm_version, width, hidden, activation, persistent
                ),
            )
        except CuTeDSLKernelLibraryError as library_error:
            if self.force_cubin():
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV}=1 forces the fused transition CUBIN path, but no CUBIN is available for "
                    f"{self._describe(width, hidden, activation, persistent)}"
                ) from library_error
            raise library_error from source_error
        description = self._describe(width, hidden, activation, persistent)
        logger.info(f"CuTeDSL fused transition: using CUBIN kernel for {description}")
        return executable

    def _get_or_compile(self, width: int, hidden: int, activation: str, persistent: bool = False) -> Any:
        """The image that takes every layout, or the persistent one, from source or from the library."""
        return self._executable(width, hidden, activation, persistent, self.force_cubin())

    def _executable(self, width: int, hidden: int, activation: str, persistent: bool, force_cubin: bool) -> Any:
        key = self._key(width, hidden, activation, persistent)
        executable = TransitionMlpWNormCuTe._compiled_cache.get(key)
        if executable is not None and (not force_cubin or isinstance(executable, CuTeDSLKernelLibraryExecutable)):
            return executable
        if force_cubin:
            TransitionMlpWNormCuTe._compiled_cache.pop(key, None)
            return self._load_cubin_executable(key, width, hidden, activation, persistent)
        try:
            source = load_source_module(__package__)
        except ImportError as source_error:
            return self._load_cubin_executable(key, width, hidden, activation, persistent, source_error)

        if self._num_sms is None:
            self._num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        disk_key = ("transition_mlp_w_norm_cute_v4", *key, _config.KERNEL_ABIS[self._sm_version])
        executable = self.load_from_cache(disk_key)
        if executable is None:
            description = self._describe(width, hidden, activation, persistent)
            logger.info(f"CuTeDSL fused transition: compiling for {description}")
            executable = source.compile_transition_mlp_w_norm_source(
                self.compile, width, hidden, self._num_sms, activation, persistent=persistent
            )
            self.save_to_cache(disk_key, executable)
        TransitionMlpWNormCuTe._compiled_cache[key] = executable
        return executable

    def geometry(self, width: int, hidden: int, activation: str) -> tuple[int, int, int]:
        """``(tile_m, num_wg, layouts)`` of the image that takes every layout: rows per full tile, warpgroups per CTA
        and the mask of the ``(tile_wgs, split)`` pairs it launches with (:func:`._config.layout_mask`)."""
        force_cubin = self.force_cubin()
        library = force_cubin or not self._source_available()
        return self._geometry(width, hidden, activation, library, force_cubin)

    def _geometry(
        self, width: int, hidden: int, activation: str, library: bool, force_cubin: bool
    ) -> tuple[int, int, int]:
        if library:
            return self._executable(width, hidden, activation, False, force_cubin).geometry
        key = self._key(width, hidden, activation)
        geometry = TransitionMlpWNormCuTe._source_geometry.get(key)
        if geometry is None:
            fields = load_source_module(__package__).launch_geometry(width, hidden, activation)
            geometry = (fields["tile_m"], fields["num_threads"] // 128, fields["layouts"])
            TransitionMlpWNormCuTe._source_geometry[key] = geometry
        return geometry

    def launch_layout(self, width: int, hidden: int, activation: str, rows: int, device: int) -> tuple[int, int]:
        """``(tile_wgs, split)`` for a call of ``rows`` rows on CUDA device ``device``, from the config's layouts
        (:func:`._config.choose_layout`)."""
        force_cubin = self.force_cubin()
        library = force_cubin or not self._source_available()
        return self._layout(width, hidden, activation, rows, device, library, force_cubin)[:2]

    def _layout(
        self, width: int, hidden: int, activation: str, rows: int, device: int, library: bool, force_cubin: bool
    ) -> tuple[int, int, bool]:
        # A packaged image and the source kernel may differ in geometry: each path keeps its own choices.
        key = (self._sm_version, width, hidden, activation, library, device, rows)
        layout = TransitionMlpWNormCuTe._layout_cache.get(key)
        if layout is None:
            num_sms = TransitionMlpWNormCuTe._device_sms.get(device)
            if num_sms is None:
                num_sms = torch.cuda.get_device_properties(device).multi_processor_count
                TransitionMlpWNormCuTe._device_sms[device] = num_sms
            tile_m, num_wg, allowed = self._geometry(width, hidden, activation, library, force_cubin)
            layouts = _config.layouts(self._sm_version, width, hidden, activation)
            tile_wgs, split = _config.choose_layout(rows, tile_m, num_wg, num_sms, layouts, allowed)
            layout = (tile_wgs, split, (tile_wgs, split) == (num_wg, 1))
            TransitionMlpWNormCuTe._layout_cache[key] = layout
        return layout

    def run(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None,
        image: torch.Tensor,
        vec: torch.Tensor,
        n_valid: torch.Tensor | None,
        rows: int,
        n: int,
        n_batch: int,
        eps: float,
        residual: bool,
        out: torch.Tensor,
        width: int,
        hidden: int,
        activation: str,
        layout: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Run the kernel into ``out``; the caller has validated the operands.

        ``z``, ``mask``, ``image`` and ``out`` are flat. Without ``n_valid`` every one of the ``rows`` rows is
        computed, and ``n`` and ``n_batch`` are 0; with it, ``z`` is ``n_batch`` pair maps of side ``n``.
        ``layout``: the launch's ``(tile_wgs, split)``, one of the kernel's (:meth:`geometry`); by default
        :meth:`launch_layout`'s. The full tile at split 1 runs the persistent image, every other layout the image
        that takes them all.
        """
        # Compiling, loading and launching all use the current device's context, and a source-backed launch its
        # stream, so make z's device current. Switch only when they differ: an unconditional guard costs several
        # microseconds a call.
        on_device = z.get_device() == torch.cuda.current_device()
        with contextlib.nullcontext() if on_device else torch.cuda.device(z.device):
            force_cubin = self.force_cubin()
            library = force_cubin or not self._source_available()
            if layout is None:
                tile_wgs, split, persistent = self._layout(
                    width, hidden, activation, rows, z.get_device(), library, force_cubin
                )
            else:
                tile_wgs, split = layout
                _, num_wg, allowed = self._geometry(width, hidden, activation, library, force_cubin)
                if not (tile_wgs <= num_wg and allowed >> _config.layout_bit(tile_wgs, split) & 1):
                    raise ValueError(f"the fused transition kernel takes no layout tile_wgs={tile_wgs}, split={split}")
                persistent = (tile_wgs, split) == (num_wg, 1)
            executable = self._executable(width, hidden, activation, persistent, force_cubin)
            flag = int(residual)
            if isinstance(executable, CuTeDSLKernelLibraryExecutable):
                # The launcher checks the spans behind the mask and the item lengths, so it takes the tensors.
                executable(z, mask, image, vec, out, n_valid, rows, n, n_batch, eps, flag, split, tile_wgs)
                return out
            mask_address = 0 if mask is None else mask.data_ptr()
            n_valid_address = 0 if n_valid is None else n_valid.data_ptr()
            launch_compiled_kernel(
                executable,
                z,
                mask_address,
                image,
                vec,
                out,
                n_valid_address,
                rows,
                n,
                n_batch,
                eps,
                flag,
                split,
                tile_wgs,
            )
        return out
