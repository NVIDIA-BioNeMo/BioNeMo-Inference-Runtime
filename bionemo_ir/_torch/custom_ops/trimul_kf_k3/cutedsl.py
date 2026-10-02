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
"""Source-or-CUBIN interface for the SM90 TriMul KF K3."""

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

from ..trimul_kf_k1.cutedsl import batch_chunks
from ._config import KERNEL_ABIS, TrimulKFK3Selection, anchors, select
from ._cubin import TrimulKFK3CubinExecutable

__all__ = ["TrimulKFK3CuTe"]


class TrimulKFK3CuTe(CuteKernelCache):
    """Cached backend for K3: output LayerNorm, projection and gate, with the optional fused residual.

    Each call runs the variant its ``(C, D)`` configs tune for the anchor nearest the token count.

    Args:
        sm_version: The SM whose configs and kernels to use; defaults to this GPU's.
    """

    _compiled_cache: dict[tuple, Any] = {}

    def __init__(self, sm_version: int | None = None) -> None:
        if sm_version is None:
            major, minor = torch.cuda.get_device_capability()
            sm_version = major * 10 + minor
        self._sm_version = sm_version
        self._num_sms: int | None = None

    def select(self, C: int, D: int, n: int) -> TrimulKFK3Selection | None:
        """The anchor and K3 variant a call over ``n`` tokens runs, or ``None`` when ``(C, D)`` does not ship."""
        return select(self._sm_version, C, D, n)

    def ships(self, C: int, D: int, residual: bool) -> bool:
        """Whether this build can run ``(C, D)`` with or without the residual, from source or packaged CUBINs."""
        entries = anchors(self._sm_version, C, D)
        if not entries:
            return False
        if not self.force_cubin():
            try:
                load_source_module(__package__)
            except ImportError:
                pass
            else:
                return True
        try:
            for kernel_variant in sorted(set(entries.values())):
                self._load_cubin_executable(self._key(C, D, kernel_variant, residual), C, D, kernel_variant, residual)
        except (CuTeDSLKernelLibraryError, RuntimeError):
            return False
        return True

    def _key(self, C: int, D: int, kernel_variant: str, residual: bool) -> tuple:
        return (self._sm_version, C, D, kernel_variant, residual)

    def _load_cubin_executable(
        self,
        key: tuple,
        C: int,
        D: int,
        kernel_variant: str,
        residual: bool,
        source_error: Exception | None = None,
    ) -> Any:
        try:
            executable = populate_compiled_cache_from_library(
                TrimulKFK3CuTe._compiled_cache,
                key,
                "trimul_kf_k3",
                lambda library, launcher: TrimulKFK3CubinExecutable(
                    library, launcher, self._sm_version, C, D, kernel_variant, residual
                ),
            )
        except CuTeDSLKernelLibraryError as library_error:
            if self.force_cubin():
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV}=1 forces the trimul KF K3 CUBIN path, but no CUBIN is available for "
                    f"SM{self._sm_version}, C={C}, D={D}, {kernel_variant}, residual={residual}"
                ) from library_error
            raise library_error from source_error
        logger.info(
            f"CuTeDSL trimul KF K3: using CUBIN kernel for SM{self._sm_version}, C={C}, D={D}, {kernel_variant}, "
            f"residual={residual}"
        )
        return executable

    def _get_or_compile(self, C: int, D: int, kernel_variant: str, residual: bool) -> Any:
        key = self._key(C, D, kernel_variant, residual)
        executable = TrimulKFK3CuTe._compiled_cache.get(key)
        force_cubin = self.force_cubin()
        if executable is not None and (not force_cubin or isinstance(executable, CuTeDSLKernelLibraryExecutable)):
            return executable
        if force_cubin:
            TrimulKFK3CuTe._compiled_cache.pop(key, None)
            return self._load_cubin_executable(key, C, D, kernel_variant, residual)
        try:
            source = load_source_module(__package__)
        except ImportError as source_error:
            return self._load_cubin_executable(key, C, D, kernel_variant, residual, source_error)

        if self._num_sms is None:
            self._num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        disk_key = ("trimul_kf_k3_cute_v1", *key, KERNEL_ABIS[self._sm_version])
        executable = self.load_from_cache(disk_key)
        if executable is None:
            logger.info(
                f"CuTeDSL trimul KF K3: compiling {kernel_variant} residual={residual} for SM{self._sm_version}, "
                f"C={C}, D={D}"
            )
            executable = source.compile_trimul_kf_k3_source(self.compile, kernel_variant, C, D, residual, self._num_sms)
            self.save_to_cache(disk_key, executable)
        TrimulKFK3CuTe._compiled_cache[key] = executable
        return executable

    def run(
        self,
        prod: torch.Tensor,
        x: torch.Tensor,
        w_out: torch.Tensor,
        w_gate_out: torch.Tensor,
        vec_out: torch.Tensor,
        stats: torch.Tensor | None,
        actual_seqlen: torch.Tensor | None,
        eps: float,
        selection: TrimulKFK3Selection,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Run ``selection``'s variant into ``out``; the caller has validated the operands."""
        B, N, _, C = x.shape
        D = prod.shape[1]
        residual = actual_seqlen is not None
        executable = self._get_or_compile(C, D, selection.kernel_variant, residual)
        seqlen = None if actual_seqlen is None else actual_seqlen.view(B * N)
        w_out = w_out.view(-1)
        w_gate_out = w_gate_out.view(-1)
        # A source-backed launch takes its stream from the current device, not the operands'.
        on_device = x.get_device() == torch.cuda.current_device()
        with contextlib.nullcontext() if on_device else torch.cuda.device(x.device):
            for b0, b1 in batch_chunks(B, N, max(C, D)):
                nb = b1 - b0
                launch_compiled_kernel(
                    executable,
                    prod[b0:b1].view(-1),
                    x[b0:b1].view(-1),
                    w_out,
                    w_gate_out,
                    vec_out,
                    None if stats is None else stats[b0:b1].view(-1),
                    None if seqlen is None else seqlen[b0 * N : b1 * N],
                    out[b0:b1].view(-1),
                    nb * N * N,
                    N,
                    nb,
                    eps,
                )
        return out
