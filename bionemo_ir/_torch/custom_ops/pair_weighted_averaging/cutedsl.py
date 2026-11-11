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
"""Source-or-CUBIN CuTeDSL backend for pair-weighted averaging."""

from __future__ import annotations

import functools
from types import ModuleType
from typing import Any

import torch

from bionemo_ir._torch.utils.kernel import (
    CuTeDSLKernelLibraryError,
    CuTeDSLKernelLibraryExecutable,
    CuTeDSLKernelVariantUnavailable,
    launch_compiled_kernel,
    load_source_module,
    populate_compiled_cache_from_library,
)
from bionemo_ir._torch.utils.kernel._cutedsl_kernel_library import _load_kernel_library
from bionemo_ir.dsl_kernels.cute_cache import FORCE_CUBIN_ENV, CuteKernelCache
from bionemo_ir.logger import logger

from ._config import (
    _SM100_SPLIT_TILE_S,
    _SUPPORTED_SM,
    PWAConfigParams,
    PWAConfigSelection,
    _select_pwa_config_selection_bucket,
    default_params,
    is_profitable_shape,
    loads_transposed_value,
)
from ._cubin import PairWeightedAveragingCubinExecutable

_TORCH_TO_DTYPE_STR = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}

_CompileKey = tuple[int, tuple[object, ...], str, str | None, torch.dtype]
_TMA_ABIS = ("sm90", "sm100")
# From this many tokens an SM90 launch covers a bounded block of MSA rows. Over hundreds of waves, the CTAs that
# share a w or v tile drift apart, and L2 evicts the shared tile between their reads; each launch starts them
# together again.
_SM90_ROW_SPLIT_TOKENS = 6144
# Token tiles the D=32 kernel launches together. A group with all of a launch's MSA-row tiles must fit one wave.
_SM90_D32_GROUP_I = 16
# MSA rows one D=8 launch covers: four clusters of four MSA-row tiles.
_SM90_D8_ROWS_PER_LAUNCH = 128
# A split-tile launch covers as many MSA rows as keep its gated value Z [B, rows, I, H*D] within this many bytes.
_SPLIT_Z_BYTES = 4 << 30


def _dtype_str(dtype: torch.dtype) -> str:
    """Return the source config token for one supported torch dtype."""
    try:
        return _TORCH_TO_DTYPE_STR[dtype]
    except KeyError as error:
        raise TypeError(f"Pair-weighted averaging expects float16 or bfloat16; got {dtype}") from error


def _expected_packaged_tile(selected: PWAConfigSelection) -> tuple[int, int, int, int]:
    """Return ``(tile_i, tile_s, tile_j, num_threads)`` the selected kernel compiles with."""
    params = selected.params
    tile = (selected.kernel_abi, selected.kernel_variant, params.TILE_I, params.num_threads)
    if tile == ("sm90", None, 32, 64):
        return 128, 2, 128, 384
    if tile == ("sm90", "d8", 64, 64):
        return 128, 8, 64, 384
    if tile == ("sm100", None, 32, 64):
        return 128, 4, 64, 192
    return params.TILE_I, params.TILE_S, params.TILE_J, params.num_threads


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _sm90_rows_per_launch(selected: PWAConfigSelection, I: int, J: int, S: int, device: torch.device) -> int:
    """MSA rows one launch covers; every launch but the last covers exactly this many."""
    if selected.kernel_abi != "sm90" or max(I, J) < _SM90_ROW_SPLIT_TOKENS:
        return S
    if selected.kernel_variant == "d8":
        return min(S, _SM90_D8_ROWS_PER_LAUNCH)
    tile_i, tile_s, _, _ = _expected_packaged_tile(selected)
    sms = _sm_count(device.index if device.index is not None else torch.cuda.current_device())
    i_tiles, s_tiles = -(-I // tile_i), -(-S // tile_s)

    def waves(tiles: int) -> int:
        # Each launch ends on a partial wave that idles the remaining SMs.
        full, rest = divmod(s_tiles, tiles)
        return full * -(-i_tiles * tiles // sms) + -(-i_tiles * rest // sms)

    most = max(1, sms // _SM90_D32_GROUP_I)
    tiles = min(range(max(1, most // 2), most + 1), key=lambda tiles: (waves(tiles), -tiles))
    return min(S, tiles * tile_s)


def _same_packaged_tile(config: object, selected: PWAConfigSelection) -> bool:
    """Whether one packaged image is the tile the Python config selected.

    fp16 defaults carry no anchors, so an unset anchor matches any packaged anchor; the tile fields decide. A D=8
    tile's cluster belongs to it: two images of one tile and cluster size can group their tiles differently.
    """
    tile_i, tile_s, tile_j, num_threads = _expected_packaged_tile(selected)
    return (
        (selected.n_anchor is None or config.n_anchor == selected.n_anchor)
        and (selected.s_anchor is None or config.s_anchor == selected.s_anchor)
        and config.tile_i == tile_i
        and config.tile_s == tile_s
        and config.tile_j == tile_j
        and config.num_threads == num_threads
        and (config.cluster_i, config.cluster_s) == tuple(selected.params.cluster)
    )


def _compile_cache_key(
    sm_version: int,
    params: PWAConfigParams,
    kernel_abi: str,
    dtype: torch.dtype,
    kernel_variant: str | None = None,
) -> _CompileKey:
    """Key one always-predicated executable.

    ``kernel_abi`` and ``kernel_variant`` select the source class. The same tile on SM100 compiles the tcgen05
    kernel or the Ampere kernel, and those executables must not share a cache entry.
    """
    return sm_version, params.machine_key(), kernel_abi, kernel_variant, dtype


def _selection_cache_key(sm_version: int, selection: PWAConfigSelection, dtype: torch.dtype) -> _CompileKey:
    return _compile_cache_key(sm_version, selection.params, selection.kernel_abi, dtype, selection.kernel_variant)


def _tma_value_ready(v: torch.Tensor) -> bool:
    """Whether ``v`` is a layout CUDA TMA can encode.

    TMA needs a 16-byte aligned base, a contiguous last dimension, and every outer byte stride a multiple of 16.
    A view such as ``buffer[..., :256]`` of a contiguous ``[..., 257]`` tensor keeps a unit last stride and a
    514-byte token stride, which the descriptor rejects.
    """
    if v.stride(-1) != 1 or v.data_ptr() % 16:
        return False
    element_size = v.element_size()
    return all(
        stride > 0 and stride * element_size % 16 == 0
        for stride, size in zip(v.stride()[:-1], v.shape[:-1], strict=True)
        if size > 1
    )


def _pitched_empty(*shape: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """An uninitialized tensor whose last-dimension pitch is a multiple of 8 elements, as TMA rows need."""
    *outer, last = shape
    return torch.empty(*outer, (last + 7) // 8 * 8, dtype=dtype, device=device)[..., :last]


def _kernel_operand(t: torch.Tensor, tma: bool) -> torch.Tensor:
    """Return ``t`` when the kernel can load it as is, else a pitch-padded copy.

    Every kernel needs a unit last stride. TMA also needs a 16-byte base and outer strides that are multiples of 16
    bytes, which a contiguous ``[B, S, H*D, N]`` v^T with ``N % 8 != 0`` does not have.
    """
    if _tma_value_ready(t) if tma else t.stride(-1) == 1:
        return t
    return _pitched_empty(*t.shape, dtype=t.dtype, device=t.device).copy_(t)


def transposed_value_buffer(
    B: int, S: int, hidden: int, N: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """An uninitialized v^T ``[B, S, hidden, N]`` that TMA loads as is.

    A producer fills it and passes ``buffer.transpose(-1, -2)``, which is ``v`` as the op takes it.
    """
    return _pitched_empty(B, S, hidden, N, dtype=dtype, device=device)


class PairWeightedAveragingCuTe(CuteKernelCache):
    """Cached source-or-CUBIN backend for fused pair-weighted averaging."""

    # Source and CUBIN executables intentionally share this one cache.
    _compiled_cache: dict[_CompileKey, Any] = {}

    def __init__(self) -> None:
        major, minor = torch.cuda.get_device_capability()
        self._sm_version = major * 10 + minor

    @property
    def sm_version(self) -> int:
        """Compute capability of the device this backend launches on, as ``major * 10 + minor``."""
        return self._sm_version

    def _disk_cache_key(self, key: _CompileKey) -> tuple[object, ...]:
        """Version the always-predicated dynamic signature."""
        return ("pwa_cute_always_predicated_v2", *key)

    def _load_cubin_executable(
        self,
        key: _CompileKey,
        selected: PWAConfigSelection,
        I: int,
        J: int,
        S: int,
        dtype: torch.dtype,
        source_error: ImportError | None = None,
    ) -> PairWeightedAveragingCubinExecutable:
        """Populate the shared cache from the packaged PWA family.

        An image whose tile is not the selected config is not a match. ``CUTEDSL_FORCE_CUBIN`` still raises.
        A source-free call lets that error reach ``__call__``, which runs PyTorch instead of the stale image.
        """
        params = selected.params
        try:
            executable = populate_compiled_cache_from_library(
                type(self)._compiled_cache,
                key,
                "pair_weighted_averaging",
                lambda library, launcher: PairWeightedAveragingCubinExecutable(
                    library,
                    launcher,
                    self._sm_version,
                    I,
                    J,
                    S,
                    params.D,
                    params.c_m,
                    dtype,
                ),
            )
            if not _same_packaged_tile(executable._config, selected):
                type(self)._compiled_cache.pop(key, None)
                raise CuTeDSLKernelVariantUnavailable(
                    f"packaged pair_weighted_averaging image does not match the selected tile "
                    f"on SM{self._sm_version}, D={params.D}, c_m={params.c_m}"
                )
        except CuTeDSLKernelLibraryError as library_error:
            if source_error is None:
                raise RuntimeError(
                    f"{FORCE_CUBIN_ENV} is set but the precompiled kernel library "
                    f"cannot provide this variant: {library_error}"
                ) from library_error
            raise

        logger.info(
            f"CuTeDSL PWA: using precompiled CUBIN for SM{self._sm_version}, I={I}, J={J}, S={S}, "
            f"D={params.D}, c_m={params.c_m}, dtype={dtype}"
        )
        return executable

    def _load_or_compile_source(
        self,
        key: _CompileKey,
        selection: PWAConfigSelection,
        source_module: ModuleType,
        source: Any,
    ) -> Any:
        """Load one persistent dynamic source object or compile and save it."""
        executable = type(self)._compiled_cache.get(key)
        if executable is not None:
            return executable

        disk_key = self._disk_cache_key(key)
        executable = self.load_from_cache(disk_key)
        if executable is not None:
            logger.info(f"CuTeDSL PWA: loaded cached kernel for SM{self._sm_version}, key={key}")
            type(self)._compiled_cache[key] = executable
            return executable

        logger.info(f"CuTeDSL PWA: compiling kernel for SM{self._sm_version}, key={key}, params={selection.params}")
        executable = source_module.compile_pair_weighted_averaging_source(self.compile, source)
        type(self)._compiled_cache[key] = executable
        self.save_to_cache(disk_key, executable)
        logger.info("CuTeDSL PWA: compilation done")
        return executable

    def _get_bucket_executable(
        self,
        selected: PWAConfigSelection,
        bucket: tuple[PWAConfigSelection, ...],
        I: int,
        J: int,
        S: int,
        dtype: torch.dtype,
    ) -> Any:
        """Resolve the selected executable, preloading all source bucket configs."""
        selected_key = _selection_cache_key(self._sm_version, selected, dtype)
        cached = type(self)._compiled_cache.get(selected_key)
        # Only a source-free or forced call caches a packaged image, and it preloads no bucket: retrying the source
        # import for the rest of the bucket would cost every call a failed import.
        if isinstance(cached, CuTeDSLKernelLibraryExecutable):
            return cached

        if self.force_cubin():
            # A source executable cached before the flag was set must not
            # defeat the explicit request for a packaged CUBIN.
            type(self)._compiled_cache.pop(selected_key, None)
            return self._load_cubin_executable(selected_key, selected, I, J, S, dtype)

        bucket_variants = list(bucket)
        if cached is not None and all(
            _selection_cache_key(self._sm_version, selection, dtype) in type(self)._compiled_cache
            for selection in bucket_variants
        ):
            return cached

        try:
            source_module = load_source_module(__package__)

            unresolved_sources = [
                (
                    selection,
                    source_module.resolve_pwa_source(selection, _dtype_str(dtype)),
                )
                for selection in bucket_variants
                if _selection_cache_key(self._sm_version, selection, dtype) not in type(self)._compiled_cache
            ]
        except ImportError as source_error:
            if cached is not None:
                return cached
            return self._load_cubin_executable(selected_key, selected, I, J, S, dtype, source_error)

        # Compilation stays outside the source-resolution try block so a real
        # compiler failure is never hidden behind the CUBIN fallback.
        for selection, source in unresolved_sources:
            key = _selection_cache_key(self._sm_version, selection, dtype)
            self._load_or_compile_source(key, selection, source_module, source)
        return type(self)._compiled_cache[selected_key]

    @staticmethod
    def _validate_launch_dtypes(
        dtype: torch.dtype,
        v: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        """Require every source/CUBIN tensor operand to use one scalar dtype."""
        for name, tensor in (("v", v), ("g", g), ("weight", weight)):
            if tensor.dtype != dtype:
                raise TypeError(f"{name} must use {dtype}; got {tensor.dtype}")

    def _launch_selection(
        self,
        w: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
        selected: PWAConfigSelection,
        bucket: tuple[PWAConfigSelection, ...],
    ) -> torch.Tensor:
        """Allocate output, resolve one dynamic variant, and launch it."""
        B, _, I, _ = w.shape
        S, J = v.shape[1], v.shape[2]
        dtype = w.dtype
        self._validate_launch_dtypes(dtype, v, g, weight)
        executable = self._get_bucket_executable(selected, bucket, I, J, S, dtype)
        tma = selected.kernel_abi in _TMA_ABIS
        # A kernel that loads v^T takes [B, S, H*D, N]; the transposed view of a v^T producer's buffer needs no copy.
        v = _kernel_operand(v.transpose(-1, -2) if selected.transposed_v else v, tma)
        g = _kernel_operand(g, tma)
        w, weight = w.contiguous(), weight.contiguous()
        if selected.writes_gated_value:
            return self._launch_gated_value(executable, w, v, g, weight)
        output = torch.empty(B, S, I, weight.shape[0], dtype=dtype, device=w.device)
        rows = _sm90_rows_per_launch(selected, I, J, S, w.device)
        if rows >= S:
            launch_compiled_kernel(executable, w, v, g, weight, output)
            return output
        # The CUBIN launcher checks every stride of a contiguous output, including the batch stride of a single
        # entry, so each block is indexed from one batch entry and gets a fresh batch dimension.
        for b in range(B):
            for s0 in range(0, S, rows):
                rs = slice(s0, min(S, s0 + rows))
                launch_compiled_kernel(
                    executable,
                    w[b].unsqueeze(0),
                    v[b, rs].unsqueeze(0),
                    g[b, rs].unsqueeze(0),
                    weight,
                    output[b, rs].unsqueeze(0),
                )
        return output

    @staticmethod
    def _launch_gated_value(
        executable: Any,
        w: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Launch the split tile into Z [B, rows, I, H*D] one block of MSA rows at a time and project it with Wo."""
        B, _, I, _ = w.shape
        S, hidden, c_m = v.shape[1], weight.shape[1], weight.shape[0]
        tile = _SM100_SPLIT_TILE_S
        rows = min(S, max(tile, _SPLIT_Z_BYTES // (B * I * hidden * w.element_size()) // tile * tile))
        z = torch.empty(B, rows, I, hidden, dtype=w.dtype, device=w.device)
        if rows == S:
            launch_compiled_kernel(executable, w, v, g, weight, z)
            return torch.matmul(z, weight.t())
        output = torch.empty(B, S, I, c_m, dtype=w.dtype, device=w.device)
        for s0 in range(0, S, rows):
            n = min(rows, S - s0)
            # The CUBIN launcher takes a contiguous output, so a short last block views the start of the buffer.
            block = z if n == rows else z.view(-1)[: B * n * I * hidden].view(B, n, I, hidden)
            rs = slice(s0, s0 + n)
            launch_compiled_kernel(executable, w, v[:, rs], g[:, rs], weight, block)
            for b in range(B):
                torch.mm(block[b].view(-1, hidden), weight.t(), out=output[b, rs].view(-1, c_m))
        return output

    def is_supported(self, w: torch.Tensor, v: torch.Tensor, Wo: torch.Tensor) -> bool:
        """Check dtype, registered dimensions, SM, and w-padding constraints."""
        if self._sm_version not in _SUPPORTED_SM:
            return False
        if w.dtype not in _TORCH_TO_DTYPE_STR or w.dim() != 4 or v.dim() != 4 or Wo.dim() != 2:
            return False
        H, I, Jp = w.shape[1], w.shape[2], w.shape[3]
        N, hidden = v.shape[2], v.shape[3]
        D = hidden // H if H else 0
        c_m = Wo.shape[0]
        return (
            hidden == H * D
            and Jp >= N
            and Jp % 8 == 0
            and is_profitable_shape(
                H, D, c_m, max(I, N), sm_version=self._sm_version, dtype_str=_TORCH_TO_DTYPE_STR[w.dtype]
            )
        )

    def _source_importable(self) -> bool:
        """Whether this checkout can compile the kernel. A wheel without ``_source`` cannot."""
        try:
            load_source_module(__package__)
        except ImportError:
            return False
        return True

    def _packaged_image_matches(
        self,
        selected: PWAConfigSelection,
        I: int,
        J: int,
        S: int,
        dtype: torch.dtype,
    ) -> bool:
        """Whether the committed CUBIN for this shape is the selected tile. A miss is not a launch."""
        try:
            library = _load_kernel_library()
        except CuTeDSLKernelLibraryError:
            return False
        launcher = getattr(library, "pair_weighted_averaging", None)
        if launcher is None:
            return False
        try:
            library_dtype = {torch.float16: launcher.DType.FLOAT16, torch.bfloat16: launcher.DType.BFLOAT16}[dtype]
            config = launcher.make_kernel_config(
                self._sm_version,
                I,
                J,
                S,
                selected.params.D,
                selected.params.c_m,
                library_dtype,
            )
        except (RuntimeError, TypeError, ValueError, KeyError):
            return False
        return _same_packaged_tile(config, selected)

    def launch_available(
        self,
        I: int,
        J: int,
        S: int,
        dtype: torch.dtype,
        H: int,
        D: int,
        c_m: int,
    ) -> bool:
        """Whether the layer should enter the fused path for this shape.

        Source can compile the selected tile. A source-free build can enter only when the packaged image is
        that tile, so a stale or missing CUBIN takes the chunked eager path instead of raising. Setting
        ``CUTEDSL_FORCE_CUBIN`` still enters and raises.
        """
        if self.force_cubin() or self._source_importable():
            return True
        dtype_str = _TORCH_TO_DTYPE_STR.get(dtype)
        if dtype_str is None:
            return False
        selected, _bucket = _select_pwa_config_selection_bucket(
            self._sm_version,
            I,
            J,
            S,
            dtype_str,
            H=H,
            D=D,
            c_m=c_m,
        )
        return self._packaged_image_matches(selected, I, J, S, dtype)

    def loads_transposed_value(self, D: int, c_m: int) -> bool:
        """Whether this SM's ``(D, c_m)`` kernel loads v^T, ``[B, S, H*D, N]`` with j contiguous.

        A producer that writes v^T into :func:`transposed_value_buffer` and passes its ``transpose(-1, -2)`` view
        spares the launch a relayout copy.
        """
        return loads_transposed_value(self._sm_version, D, c_m)

    def __call__(
        self,
        w: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        Wo: torch.Tensor,
    ) -> torch.Tensor:
        """Run fused PWA or use the vanilla fallback outside its fixed envelope.

        ``v`` is ``[B, S, N, H*D]`` with any strides; see :meth:`loads_transposed_value` for the layout that saves a
        copy.
        """
        if not self.is_supported(w, v, Wo):
            from .ops import _invoke_vanilla_pwa

            return _invoke_vanilla_pwa(w, v, g, Wo)

        _, H, I, _ = w.shape
        S, J = v.shape[1], v.shape[2]
        D = v.shape[3] // H
        c_m = Wo.shape[0]
        dtype_str = _dtype_str(w.dtype)
        selected, bucket = _select_pwa_config_selection_bucket(
            self._sm_version,
            I,
            J,
            S,
            dtype_str,
            H=H,
            D=D,
            c_m=c_m,
        )
        try:
            return self._launch_selection(w, v, g, Wo, selected, bucket)
        except CuTeDSLKernelLibraryError:
            if self.force_cubin():
                raise
            from .ops import _invoke_vanilla_pwa

            return _invoke_vanilla_pwa(w, v, g, Wo)

    def _call_explicit(
        self,
        w: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        Wo: torch.Tensor,
        config: object | None,
    ) -> torch.Tensor:
        """Compatibility launch with one explicit or default source config."""
        if w.dim() != 4 or v.dim() != 4 or Wo.dim() != 2:
            raise ValueError("PWA expects rank-4 w/v and rank-2 output weight tensors")
        _, H, _, Jp = w.shape
        N = v.shape[2]
        if Jp < N or Jp % 8 != 0:
            raise ValueError(f"PWA: w j-extent Jp={Jp} must cover N={N} and be a multiple of 8")
        dtype_str = _dtype_str(w.dtype)
        D = v.shape[-1] // H
        c_m = Wo.shape[0]
        if config is None:
            params = default_params(dtype_str, H, D, c_m)
        else:
            config_dtype = config.ab_dtype
            if config_dtype != dtype_str:
                raise TypeError(f"PWA config dtype {config_dtype!r} does not match tensor dtype {w.dtype}")
            params = PWAConfigParams.from_config(config)
        selection = PWAConfigSelection(params=params)
        return self._launch_selection(w, v, g, Wo, selection, (selection,))


def pwa_cute(
    w: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    Wo: torch.Tensor,
    config: object | None = None,
) -> torch.Tensor:
    """Launch PWA with an explicit source-compatible config or its defaults."""
    return PairWeightedAveragingCuTe()._call_explicit(w, v, g, Wo, config)
