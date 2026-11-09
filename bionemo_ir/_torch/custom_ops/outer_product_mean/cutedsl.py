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
"""Source-or-CUBIN CuTeDSL interface for fused outer-product-mean."""

from __future__ import annotations

from typing import Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryError,
    CuTeDSLKernelLibraryExecutable,
    get_config_file_name,
    launch_compiled_kernel,
    load_kernel_configs,
    load_source_module,
    populate_compiled_cache_from_library,
)
from bionemo_ir.dsl_kernels.cute_cache import FORCE_CUBIN_ENV, CuteKernelCache
from bionemo_ir.dsl_kernels.triton.opm_prologue import empty_sequence_last
from bionemo_ir.logger import logger

from ._config import (
    _KERNEL_C,
    _KERNEL_CZ,
    _KERNEL_D,
    _OPM_CONFIGS_DIR,
    _SM100_FAMILY,
    _SUPPORTED_CZ,
    OpmConfig,
    Sm100KernelConfig,
    _parse_key,
    _select_opm_config_bucket,
    config_identity,
    select_opm_config,
)
from ._cubin import OuterProductMeanCubinExecutable, OuterProductMeanCubinMissing
from .ops import _invoke_triton_opm, _invoke_vanilla_opm

__all__ = [
    "OuterProductMeanCuTe",
    "_KERNEL_C",
    "_KERNEL_CZ",
    "_KERNEL_D",
    "_OPM_CONFIGS_DIR",
    "_invoke_vanilla_opm",
    "_parse_key",
    "_select_opm_config_bucket",
    "config_identity",
    "select_opm_config",
]

# SMs with shipped CUBINs. SM86/89 use a smaller tile to fit their shared-memory limit.
_SUPPORTED_SM = (80, 86, 89, 90, 100, 103)
# SMs whose CUBINs shipped after earlier kernel libraries: a library without
# them runs the call on Triton rather than failing.
_TRITON_FALLBACK_SMS = frozenset({100, 103})
_DTYPE_STR = {torch.float16: "fp16", torch.bfloat16: "bf16"}


def _token_blocks(blocked: torch.Tensor, flat: torch.Tensor, I: int):
    """Pair views of ``blocked [B, S, blocks, T, G, cc]`` and ``flat [B, S, I, G, cc]``: whole blocks, then the tail."""
    T = blocked.shape[3]
    full, tail = divmod(I, T)
    if full:
        yield blocked[:, :, :full], flat[:, :, : full * T].unflatten(2, (full, T))
    if tail:
        yield blocked[:, :, full, :tail], flat[:, :, full * T :]


def _relayout_a(a: torch.Tensor, cc: int) -> torch.Tensor:
    """Copy ``a [B, S, I, C]`` into the sequence-last layout the SM100 kernel loads.

    The layer's fused prologue writes that layout itself; this copy serves callers that pass ``a`` as
    ``[B, S, I, C]``.
    """
    B, S, I, C = a.shape
    a_t = empty_sequence_last(B, S, I, C, cc, a.dtype, a.device)
    for blocked, flat in _token_blocks(a_t.permute(0, 5, 2, 4, 1, 3), a.unflatten(-1, (C // cc, cc)), I):
        blocked.copy_(flat)
    return a_t


def _plain_a(a: torch.Tensor, I: int) -> torch.Tensor:
    """Undo the sequence-last layout: ``[B, G, ceil(I / T), cc, T, S]`` -> ``[B, S, I, G * cc]``."""
    B, G, _, cc, _, S = a.shape
    plain = torch.empty(B, S, I, G * cc, dtype=a.dtype, device=a.device)
    for blocked, flat in _token_blocks(a.permute(0, 5, 2, 4, 1, 3), plain.unflatten(-1, (G, cc)), I):
        flat.copy_(blocked)
    return plain


def _in_place(t: torch.Tensor, align: int = 16) -> torch.Tensor:
    """``t`` if the kernels read it as it lies, else a packed copy.

    The kernels (``_source.make_fake_args``) take a unit last stride, the stride before it equal to the last
    extent, and the other strides and the base on ``align``-byte boundaries. ``.contiguous()`` would hand a
    contiguous view that starts mid-allocation back unchanged.
    """
    strides, itemsize = t.stride(), t.element_size()
    if (
        strides[-1] == 1
        and (t.dim() == 1 or strides[-2] == t.shape[-1])
        and t.data_ptr() % align == 0
        and all(stride > 0 and stride * itemsize % align == 0 for stride in strides[:-2])
    ):
        return t
    return t.clone(memory_format=torch.contiguous_format)


class OuterProductMeanCuTe(CuteKernelCache):
    """Cached CuTeDSL backend for the fused outer-product-mean projection."""

    _compiled_cache: dict[tuple, Any] = {}
    _triton_fallback_keys: set[tuple] = set()

    def __init__(self):
        major, minor = torch.cuda.get_device_capability()
        self._sm_version = major * 10 + minor
        self._last_resolution: tuple | None = None

    def _disk_cache_key(self, key: tuple) -> tuple:
        return ("opm_cute",) + key

    def _load_cubin_executable(
        self,
        key: tuple,
        config: OpmConfig,
        dtype: torch.dtype,
        has_bias: bool,
        norm_before: bool,
        source_error: Exception | None = None,
    ):
        if key in OuterProductMeanCuTe._triton_fallback_keys:
            return None
        identity = config_identity(config)
        try:
            executable = populate_compiled_cache_from_library(
                OuterProductMeanCuTe._compiled_cache,
                key,
                "outer_product_mean",
                lambda library, launcher: OuterProductMeanCubinExecutable(
                    library,
                    launcher,
                    self._sm_version,
                    dtype,
                    has_bias,
                    norm_before,
                    identity,
                ),
            )
        except CuTeDSLKernelLibraryError as library_error:
            if self._sm_version in _TRITON_FALLBACK_SMS and isinstance(library_error, OuterProductMeanCubinMissing):
                OuterProductMeanCuTe._triton_fallback_keys.add(key)
                logger.warning(
                    f"CuTeDSL OPM: the kernel library has no CUBIN for SM{self._sm_version}, "
                    f"dtype={dtype}, has_bias={has_bias}, norm_before={norm_before}, config={identity}; "
                    f"running on Triton: {library_error}"
                )
                return None
            if self.force_cubin():
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV}=1 forces the outer-product-mean CUBIN "
                    f"path, but no CUBIN is available for SM{self._sm_version}, "
                    f"dtype={dtype}, has_bias={has_bias}, "
                    f"norm_before={norm_before}, config={identity}"
                ) from library_error
            raise library_error from source_error
        logger.info(
            f"CuTeDSL OPM: using CUBIN kernel for SM{self._sm_version}, "
            f"dtype={dtype}, has_bias={has_bias}, norm_before={norm_before}"
        )
        return executable

    @staticmethod
    def _resolve_source(config: OpmConfig, has_bias: bool, norm_before: bool):
        source = load_source_module(__package__)

        needed = source.dynamic_smem_bytes(config)
        limit = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block_optin
        if needed > limit:
            raise RuntimeError(
                f"outer-product-mean needs {needed} B dynamic shared memory, but the device allows {limit} B"
            )
        return source.make_kernel(config, has_bias, norm_before), source.compile_opm_source

    def _load_or_compile_source(self, key: tuple, kernel: Any, compile_source: Any, config: OpmConfig, has_bias: bool):
        disk_key = self._disk_cache_key(key)
        executable = self.load_from_cache(disk_key)
        if executable is not None:
            logger.info(f"CuTeDSL OPM: loaded cached kernel for SM{self._sm_version}, key={key}")
            OuterProductMeanCuTe._compiled_cache[key] = executable
            return executable

        logger.info(f"CuTeDSL OPM: compiling kernel for SM{self._sm_version}, key={key}")
        executable = compile_source(self.compile, kernel, config, has_bias)
        OuterProductMeanCuTe._compiled_cache[key] = executable
        self.save_to_cache(disk_key, executable)
        return executable

    def _get_or_compile(
        self,
        config: OpmConfig,
        dtype: torch.dtype,
        has_bias: bool,
        norm_before: bool,
        key: tuple,
    ):
        executable = OuterProductMeanCuTe._compiled_cache.get(key)
        force_cubin = self.force_cubin()
        if executable is not None and (not force_cubin or isinstance(executable, CuTeDSLKernelLibraryExecutable)):
            return executable

        if force_cubin:
            OuterProductMeanCuTe._compiled_cache.pop(key, None)
            return self._load_cubin_executable(key, config, dtype, has_bias, norm_before)

        try:
            kernel, compile_source = self._resolve_source(config, has_bias, norm_before)
        except ImportError as source_error:
            return self._load_cubin_executable(key, config, dtype, has_bias, norm_before, source_error)
        return self._load_or_compile_source(key, kernel, compile_source, config, has_bias)

    def is_supported(self, a: torch.Tensor, b: torch.Tensor, W_o: torch.Tensor) -> bool:
        """Whether the CuTeDSL kernel can run this problem (else vanilla)."""
        if self._sm_version not in _SUPPORTED_SM:
            return False
        if a.dtype not in _DTYPE_STR or b.dim() != 4:
            return False
        if a.dim() != 4 and not (a.dim() == 6 and self._sm_version in _SM100_FAMILY):
            return False
        C = a.shape[1] * a.shape[3] if a.dim() == 6 else a.shape[-1]
        D, C_z = b.shape[-1], W_o.shape[0]
        return C == _KERNEL_C and D == _KERNEL_D and C_z in _SUPPORTED_CZ

    def _resolve(
        self, S: int, I: int, J: int, C_z: int, norm_before: bool, has_bias: bool, dtype: torch.dtype
    ) -> tuple[OpmConfig, Any]:
        """The config one call runs and its executable, ``None`` where Triton runs it instead.

        The layer asks twice per forward (for ``a``'s layout, then to launch), and every layer of a model asks with
        the same shape, so the last answer is kept. A new config bundle (another tuned-config folder), selector or
        executable cache misses it.
        """
        sources = (
            load_kernel_configs(_OPM_CONFIGS_DIR, get_config_file_name(self._sm_version)),
            _select_opm_config_bucket,
            OuterProductMeanCuTe._compiled_cache,
        )
        key = (S, I, J, C_z, norm_before, has_bias, dtype, self.force_cubin())
        last = self._last_resolution
        if last is not None and last[0] == key and all(old is new for old, new in zip(last[1], sources, strict=True)):
            return last[2]
        selector = sources[1]
        config, bucket_configs = selector(self._sm_version, I, J, S, norm_before, has_bias, _DTYPE_STR[dtype], C_z=C_z)
        executable = None
        # Load or compile the selected config's whole N bucket at once.
        for variant in bucket_configs:
            variant_key = (self._sm_version, variant.config_key(), has_bias, norm_before)
            loaded = self._get_or_compile(variant, dtype, has_bias, norm_before, variant_key)
            if variant.config_key() == config.config_key():
                executable = loaded
        self._last_resolution = (key, sources, (config, executable))
        return config, executable

    def a_chunk(
        self, S: int, I: int, J: int, C_z: int, norm_before: bool, has_bias: bool, dtype: torch.dtype
    ) -> int | None:
        """Channels per chunk when this call takes ``a`` sequence-last, else ``None`` (``a`` as ``[B, S, I, C]``).

        Only the SM100 kernel loads ``a`` sequence-last (see ``empty_sequence_last``); a producer that writes that
        layout directly saves the relayout pass. A call that runs on Triton takes ``a`` plain.
        """
        if self._sm_version not in _SM100_FAMILY or dtype not in _DTYPE_STR or S == 0 or I * J == 0:
            return None
        config, executable = self._resolve(S, I, J, C_z, norm_before, has_bias, dtype)
        return config.cc if isinstance(config, Sm100KernelConfig) and executable is not None else None

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        num_mask: torch.Tensor,
        W_o: torch.Tensor,
        bias: torch.Tensor | None = None,
        norm_before: bool = True,
    ) -> torch.Tensor:
        """Run fused OPM, falling back to PyTorch when unsupported.

        Args:
            a: Left input with shape ``[B, S, I, C]``, or on SM100/SM103 already sequence-last as
                ``[B, C / cc, ceil(I / T), cc, T, S]`` with the chunk :meth:`a_chunk` returns.
            b: Right input with shape ``[B, S, J, D]``.
            num_mask: Normalization counts with shape ``[B, I, J]``.
            W_o: Projection weight with shape ``[C_z, C * D]``.
            bias: Optional projection bias with shape ``[C_z]``.
            norm_before: Apply normalization before the projection.

        Returns:
            Tensor with shape ``[B, I, J, C_z]``.
        """
        if not self.is_supported(a, b, W_o):
            if a.dim() == 6:
                a = _plain_a(a, num_mask.shape[1])
            return _invoke_vanilla_opm(a, b, num_mask, W_o, bias, norm_before)

        B, S, J = b.shape[:3]
        I = num_mask.shape[1]
        C_z = W_o.shape[0]
        out = torch.empty(B, I, J, C_z, dtype=a.dtype, device=a.device)
        if out.numel() == 0:
            return out
        if S == 0:
            # Every CuTe kernel needs at least one sequence row: the SM100 relayout and TMA maps have no zero
            # extent, and the SM90 mainloop never completes an empty K loop.
            return _invoke_triton_opm(a, b, num_mask, W_o, bias, norm_before)

        config, exe = self._resolve(S, I, J, C_z, norm_before, bias is not None, a.dtype)
        if exe is None:
            if a.dim() == 6:
                a = _plain_a(a, I)
            return _invoke_triton_opm(a, b, num_mask, W_o, bias, norm_before)

        if isinstance(config, Sm100KernelConfig):
            if a.dim() == 4:
                a = _relayout_a(a, config.cc)
            elif a.shape[3] != config.cc:
                raise ValueError(f"a is sequence-last with chunk {a.shape[3]}; this call needs {config.cc}")
        else:
            a = _in_place(a)
        # The layer's operands already conform; only a direct caller's misaligned or strided ones are copied.
        b, W_o = _in_place(b), _in_place(W_o)
        num_mask = _in_place(num_mask.float(), align=4)
        t_bias = None if bias is None else _in_place(bias)
        launch_compiled_kernel(exe, a, b, num_mask, W_o, t_bias, out)
        return out
