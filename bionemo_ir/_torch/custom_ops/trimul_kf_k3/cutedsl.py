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
"""Source-or-CUBIN interface for the TriMul KF K3: SM90 from source or CUBINs, SM100/SM103 from source."""

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

from ..trimul_kf_k1.cutedsl import MAX_LAUNCH_ROWS, batch_chunks
from ._config import KERNEL_ABIS, PITCHED_VARIANTS, TrimulKFK3Selection, anchors, select
from ._cubin import TrimulKFK3CubinExecutable

__all__ = ["TrimulKFK3CuTe"]

#: Kernel ABIs the packaged CUBIN library carries; the others run from source only.
CUBIN_ABIS = frozenset({"sm90"})
#: Disk-cache tag per kernel ABI. The compiled kernel's instantiation lives in ``_source``, outside the
#: cache's source fingerprint, so a change there bumps the tag.
_DISK_CACHE_TAGS = {"sm90": "trimul_kf_k3_cute_v3", "sm100": "trimul_kf_k3_cute_sm100_v1"}


def one_launch_chunks(B: int, N: int) -> tuple[tuple[int, int], ...]:
    """Batch ranges of a variant whose row tiles may straddle batch items: only int32 row coordinates split."""
    batch_chunks(1, N)  # rejects planes past int32 coordinates
    max_batch = MAX_LAUNCH_ROWS // (N * N)
    chunks = -(-B // max_batch)
    step = -(-B // chunks)
    return tuple((b, min(B, b + step)) for b in range(0, B, step))


class TrimulKFK3CuTe(CuteKernelCache):
    """Cached backend for K3: output LayerNorm, projection and gate, with the optional fused residual.

    Each call runs the variant its ``(C, D)`` configs tune for the anchor nearest the token count. SM90
    runs from source or the packaged CUBINs; SM100 and SM103 run from source alone, so under
    ``CUTEDSL_FORCE_CUBIN=1`` they ship nothing.

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
        if KERNEL_ABIS[self._sm_version] not in CUBIN_ABIS:
            return False
        try:
            for kernel_variant in sorted(set(entries.values())):
                self._load_cubin_executable(self._key(C, D, kernel_variant, residual), C, D, kernel_variant, residual)
        except (CuTeDSLKernelLibraryError, RuntimeError):
            return False
        return True

    def _key(self, C: int, D: int, kernel_variant: str, residual: bool, pitched: bool = False) -> tuple:
        key = (self._sm_version, C, D, kernel_variant, residual)
        return (*key, "pitched") if pitched else key

    def _load_cubin_executable(
        self,
        key: tuple,
        C: int,
        D: int,
        kernel_variant: str,
        residual: bool,
        source_error: Exception | None = None,
        pitched: bool = False,
    ) -> Any:
        if KERNEL_ABIS[self._sm_version] not in CUBIN_ABIS:
            if self.force_cubin():
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV}=1 forces the trimul KF K3 CUBIN path, but SM{self._sm_version} runs it from "
                    "source only"
                ) from source_error
            raise RuntimeError(
                f"trimul KF K3 runs from source only on SM{self._sm_version}, and this build has no kernel sources"
            ) from source_error
        product = "pitched" if pitched else "dense"
        try:
            executable = populate_compiled_cache_from_library(
                TrimulKFK3CuTe._compiled_cache,
                key,
                "trimul_kf_k3",
                lambda library, launcher: TrimulKFK3CubinExecutable(
                    library, launcher, self._sm_version, C, D, kernel_variant, residual, pitched
                ),
            )
        except CuTeDSLKernelLibraryError as library_error:
            if self.force_cubin():
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV}=1 forces the trimul KF K3 CUBIN path, but no CUBIN is available for "
                    f"SM{self._sm_version}, C={C}, D={D}, {kernel_variant}, residual={residual}, {product} product"
                ) from library_error
            raise library_error from source_error
        logger.info(
            f"CuTeDSL trimul KF K3: using CUBIN kernel for SM{self._sm_version}, C={C}, D={D}, {kernel_variant}, "
            f"residual={residual}, {product} product"
        )
        return executable

    def supports_pitch(self, C: int, D: int, kernel_variant: str, residual: bool) -> bool:
        """Whether this build reads a pitched product with ``kernel_variant``, from source or packaged CUBINs."""
        if kernel_variant not in PITCHED_VARIANTS:
            return False
        if not self.force_cubin():
            try:
                load_source_module(__package__)
            except ImportError:
                pass
            else:
                return True
        key = self._key(C, D, kernel_variant, residual, pitched=True)
        try:
            self._load_cubin_executable(key, C, D, kernel_variant, residual, pitched=True)
        except (CuTeDSLKernelLibraryError, RuntimeError):
            return False
        return True

    def _get_or_compile(self, C: int, D: int, kernel_variant: str, residual: bool, pitched: bool = False) -> Any:
        key = self._key(C, D, kernel_variant, residual, pitched)
        executable = TrimulKFK3CuTe._compiled_cache.get(key)
        force_cubin = self.force_cubin()
        if executable is not None and (not force_cubin or isinstance(executable, CuTeDSLKernelLibraryExecutable)):
            return executable
        if force_cubin:
            TrimulKFK3CuTe._compiled_cache.pop(key, None)
            return self._load_cubin_executable(key, C, D, kernel_variant, residual, pitched=pitched)
        try:
            source = load_source_module(__package__)
        except ImportError as source_error:
            return self._load_cubin_executable(key, C, D, kernel_variant, residual, source_error, pitched)

        if self._num_sms is None:
            self._num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        kernel_abi = KERNEL_ABIS[self._sm_version]
        disk_key = (_DISK_CACHE_TAGS[kernel_abi], *key, kernel_abi)
        executable = self.load_from_cache(disk_key)
        if executable is None:
            logger.info(
                f"CuTeDSL trimul KF K3: compiling {kernel_variant} residual={residual} for SM{self._sm_version}, "
                f"C={C}, D={D}, {'pitched' if pitched else 'dense'} product"
            )
            executable = source.compile_trimul_kf_k3_source(
                self.compile, kernel_variant, C, D, residual, self._num_sms, pitched
            )
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
        """Run ``selection``'s variant into ``out``; the caller has validated the operands.

        ``prod`` is dense or, for one of the :data:`PITCHED_VARIANTS`, a view of K2's ``[B, D, N, P]`` buffer: rows
        ``P`` elements apart, channels ``N`` rows apart. A pitched product runs the variant's per-row image.
        """
        B, N, _, C = x.shape
        D = prod.shape[1]
        residual = actual_seqlen is not None
        pitch, plane = prod.stride(2), prod.stride(1)
        pitched = pitch != N
        sm100 = KERNEL_ABIS[self._sm_version] == "sm100"
        if sm100 and pitched:
            raise ValueError(f"trimul KF K3 {selection.kernel_variant} reads a dense product; got row pitch {pitch}")
        executable = self._get_or_compile(C, D, selection.kernel_variant, residual, pitched)
        seqlen = None if actual_seqlen is None else actual_seqlen.view(B * N)
        w_out = w_out.view(-1)
        w_gate_out = w_gate_out.view(-1)
        # A source-backed launch takes its stream from the current device, not the operands'.
        on_device = x.get_device() == torch.cuda.current_device()
        with contextlib.nullcontext() if on_device else torch.cuda.device(x.device):
            if sm100:
                # Tiles never straddle batches and every offset past a batch is 64-bit: one launch
                # covers the whole batch. The kernel reduces its own row statistics.
                launch_compiled_kernel(
                    executable,
                    prod.view(-1),
                    x.view(-1),
                    w_out,
                    w_gate_out,
                    vec_out,
                    seqlen,
                    out.view(-1),
                    N,
                    B,
                    eps,
                )
                return out
            for b0, b1 in (one_launch_chunks if selection.one_launch else batch_chunks)(B, N):
                nb = b1 - b0
                if pitched:
                    span = (nb * D - 1) * plane + (N - 1) * pitch + N
                    flat_prod = prod[b0:b1].as_strided((span,), (1,))
                    prod_strides = (pitch, plane)
                else:
                    flat_prod = prod[b0:b1].view(-1)
                    # a variant that can read a pitch takes none for a dense product
                    prod_strides = (None, None) if selection.kernel_variant in PITCHED_VARIANTS else ()
                launch_compiled_kernel(
                    executable,
                    flat_prod,
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
                    *prod_strides,
                )
        return out
