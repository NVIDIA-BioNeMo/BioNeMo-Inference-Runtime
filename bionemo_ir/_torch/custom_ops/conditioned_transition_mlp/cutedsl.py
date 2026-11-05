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
"""Source-or-CUBIN interface for the conditioned transition MLP."""

from __future__ import annotations

import contextlib
import functools
from typing import Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryError,
    CuTeDSLKernelLibraryExecutable,
    launch_compiled_kernel,
    load_source_module,
    populate_compiled_cache_from_library,
)
from bionemo_ir.dsl_kernels.cute_cache import CuteKernelCache
from bionemo_ir.logger import logger

from . import _config
from ._cubin import ConditionedTransitionMlpCubinExecutable

__all__ = ["ConditionedTransitionMlpCuTe"]


@functools.cache
def _source_available() -> bool:
    """Whether this build carries the kernel source. Checked once: a failed import costs about 10 us a call."""
    try:
        load_source_module(__package__)
    except ImportError:
        return False
    return True


class ConditionedTransitionMlpCuTe(CuteKernelCache):
    """Cached backend for ``a + sigmoid(g) * MLP(sigmoid(gamma) * LayerNorm(a) + beta)`` in one kernel.

    Each call runs the config :func:`._config.choose_config` picks among its activation's tiles for its rows, its atoms
    and the device's CTA pairs, from source or from the packaged CUBIN. A call whose kernel this build lacks returns ``None``, so the caller keeps
    its unfused path; ``CUTEDSL_FORCE_CUBIN`` does not make that an error, as a library that predates the family has
    no image to run.

    Args:
        sm_version: The SM whose configs and kernel to use.
    """

    _compiled_cache: dict[tuple, Any] = {}

    def __init__(self, sm_version: int) -> None:
        self._sm_version = sm_version
        # Per (switch, width, hidden, activation, tile): the configs that failed to compile or load.
        self._unavailable: set[tuple] = set()
        # Per (device, switch, width, hidden, activation, rows, atoms): the executable of the call's config.
        self._executables: dict[tuple, Any] = {}
        # Per device index: its CTA pairs.
        self._pairs: dict[int, int] = {}

    def ships(self, width: int, hidden: int, activation: str = "silu_gate") -> bool:
        """Whether this build can run ``(width, hidden, activation)``: from source, or from a packaged CUBIN of every
        config."""
        configs = _config.configs(self._sm_version, width, hidden, activation)
        if not configs:
            return False
        if not self.force_cubin() and _source_available():
            return True
        try:
            for params in configs.values():
                self._load_cubin_executable(width, hidden, activation, params)
        except CuTeDSLKernelLibraryError:
            return False
        return True

    def _load_cubin_executable(self, width: int, hidden: int, activation: str, params: dict[str, int]) -> Any:
        three_way = activation == "silu_gate_3way"
        cond_stage, samples = _config.image_key(params)
        key = (self._sm_version, width, hidden, three_way, cond_stage, samples, "cubin")
        return populate_compiled_cache_from_library(
            ConditionedTransitionMlpCuTe._compiled_cache,
            key,
            "conditioned_transition_mlp",
            lambda library, launcher: ConditionedTransitionMlpCubinExecutable(
                library, launcher, self._sm_version, width, hidden, three_way, cond_stage, samples
            ),
        )

    def _get_or_compile(self, width: int, hidden: int, activation: str, tile: str, force_cubin: bool) -> Any:
        """The executable of ``activation``'s config ``tile``: the packaged CUBIN under ``CUTEDSL_FORCE_CUBIN`` or
        without the source, otherwise compiled from source."""
        params = _config.configs(self._sm_version, width, hidden, activation)[tile]
        if force_cubin or not _source_available():
            return self._load_cubin_executable(width, hidden, activation, params)
        disk_key = (
            "conditioned_transition_mlp_cute_v2",
            self._sm_version,
            width,
            hidden,
            activation,
            *sorted(params.items()),
        )
        executable = ConditionedTransitionMlpCuTe._compiled_cache.get(disk_key) or self.load_from_cache(disk_key)
        if executable is None:
            logger.info(f"CuTeDSL conditioned transition: compiling {activation} {tile} for SM{self._sm_version}")
            source = load_source_module(__package__)
            executable = source.compile_conditioned_transition_mlp_source(
                self.compile, params, width, hidden, activation
            )
            self.save_to_cache(disk_key, executable)
        ConditionedTransitionMlpCuTe._compiled_cache[disk_key] = executable
        return executable

    def _executable(self, width: int, hidden: int, activation: str, rows: int, atoms: int, device: int) -> Any:
        """The executable a call of ``rows`` rows over ``atoms`` atoms on CUDA device ``device`` runs, or ``None``."""
        force_cubin = self.force_cubin()
        key = (device, force_cubin, width, hidden, activation, rows, atoms)
        executable = self._executables.get(key)
        if executable is not None:
            return executable
        pairs = self._pairs.get(device)
        if pairs is None:
            pairs = self._pairs[device] = torch.cuda.get_device_properties(device).multi_processor_count // 2
        configs = _config.configs(self._sm_version, width, hidden, activation)
        samples = next((_config.image_key(params)[1] for params in configs.values() if _config.image_key(params)[1]), 0)
        tile = _config.choose_config(rows, atoms, pairs, samples, configs)
        failed = (force_cubin, width, hidden, activation, tile)
        if failed in self._unavailable:
            return None
        try:
            executable = self._get_or_compile(width, hidden, activation, tile, force_cubin)
        except Exception as error:  # noqa: BLE001 - a kernel this build lacks or cannot compile keeps the caller's path
            logger.warning(
                f"CuTeDSL conditioned transition {activation} {tile} unavailable; using the unfused path: {error}"
            )
            self._unavailable.add(failed)
            return None
        self._executables[key] = executable
        return executable

    def run(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        cond: torch.Tensor,
        out: torch.Tensor,
        width: int,
        hidden: int,
        activation: str = "silu_gate",
    ) -> bool:
        """Launch on ``x`` and ``out`` ``[rows, width]`` and ``cond`` ``[atoms, 3 * width]``, which the caller has
        validated; ``False`` when this build has no kernel for the call."""
        device = x.get_device()
        executable = self._executable(width, hidden, activation, x.shape[0], cond.shape[0], device)
        if executable is None:
            return False
        if isinstance(executable, CuTeDSLKernelLibraryExecutable):
            executable(x, w1, w2, cond, out)
            return True
        # A source-backed launch takes its stream from the current device; switch only when the operands' differs.
        with contextlib.nullcontext() if device == torch.cuda.current_device() else torch.cuda.device(device):
            launch_compiled_kernel(executable, x, w1, w2, cond, out)
        return True
