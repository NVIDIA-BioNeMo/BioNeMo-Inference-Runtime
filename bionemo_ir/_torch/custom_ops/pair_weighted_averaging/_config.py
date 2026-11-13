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
"""Source-independent configuration lookup for pair-weighted averaging."""

from __future__ import annotations

import functools
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType

from bionemo_ir._torch.utils.kernel import KernelConfigBundle, get_config_file_name, load_kernel_configs

_KERNEL_H = 8

# All current models use H=8, so variants are keyed by (D, c_m).
# A new H needs its own filename, configs, defaults, and CUBINs.
_SUPPORTED_DIMS: tuple[tuple[int, int], ...] = (
    (32, 64),  # Boltz1/Boltz2 trunk, Boltz1 confidence
    (8, 64),  # OpenFold3
    (8, 128),  # Protenix
)
_KERNEL_D, _KERNEL_CM = _SUPPORTED_DIMS[0]

# Defaults for a dtype its tuning file does not list, or a missing tuning file.
# At D=8, each cp.async row is one 128-bit vector, so TILE_I and TILE_J
# must be multiples of num_threads.
_DEFAULT_PARAMS_BY_DIMS: dict[tuple[int, int], dict[str, object]] = {
    (32, 64): {},
    (8, 64): {"TILE_I": 64, "TILE_J": 64, "TILE_S": 4, "KO_TILE": 32, "num_stages": 3, "atom_p": (2, 1, 1)},
    (8, 128): {"TILE_I": 64, "TILE_J": 64, "TILE_S": 4, "KO_TILE": 32, "num_stages": 3},
}

# Default limit when a tuning file sets none for the dtype. Measured limits live in the file, next to the
# measurements. A dtype the file does not list was not swept, so it never takes a limit higher than this default.
_MAX_FUSED_TOKENS_BY_DIMS: dict[tuple[int, int], int | None] = {
    (32, 64): 1536,
    (8, 64): 2048,
    (8, 128): 2048,
}
_FUSED_LIMIT_DTYPES = ("bf16", "fp16")

_SUPPORTED_SM = (80, 90, 100, 103, 107, 120, 121)

# Kernels that load v transposed, [B, S, H*D, N] with j contiguous, so the value GEMM's B operand arrives in 128-byte
# rows. Every other kernel loads [B, S, N, H*D].
_TRANSPOSED_V_KERNELS = frozenset({("sm100", "d8")})
# The SM100 D=32 split tile: its 8-row value MMA fills TMEM, so the kernel writes the gated value Z [B, S, I, H*D] and
# the caller applies Wo as one GEMM.
_SM100_SPLIT_TILE_S = 8


def is_supported_dims(H: int, D: int, c_m: int) -> bool:
    """Whether this (H, D, c_m) tuple has tuned configs and embedded CUBINs."""
    return H == _KERNEL_H and (D, c_m) in _SUPPORTED_DIMS


def max_fused_tokens(
    D: int,
    c_m: int,
    sm_version: int | None = None,
    dtype_str: str = "bf16",
) -> int | None:
    """Token count above which the chunked eager fallback is the supported path, or ``None`` for no limit.

    A tuning file's ``max_fused_tokens`` is either one bf16 limit or a mapping from dtype to limit, each
    measured on that SM. A dtype the file does not list keeps the default for ``(D, c_m)``, or the file's bf16
    limit when that is lower. With no ``sm_version`` and no file, the default applies.
    """
    default = _MAX_FUSED_TOKENS_BY_DIMS.get((D, c_m))
    if sm_version is None:
        return default
    file_limits = _max_fused_tokens_in_file(sm_version, D, c_m)
    if dtype_str in file_limits:
        return file_limits[dtype_str]
    bf16_limit = file_limits.get("bf16")
    if default is None or bf16_limit is None:
        return default
    return min(default, bf16_limit)


def is_profitable_shape(
    H: int,
    D: int,
    c_m: int,
    tokens: int,
    sm_version: int | None = None,
    dtype_str: str = "bf16",
) -> bool:
    """Whether fusing this shape is worth it, from dimensions, dtype, and the SM.

    Callers that own an eager fallback should consult this *before* building kernel inputs: declining inside
    the op only reaches its own unchunked reference. The limit is the largest measured size, not a claim that
    the fused kernel wins at every size past it.
    """
    if not is_supported_dims(H, D, c_m):
        return False
    token_limit = max_fused_tokens(D, c_m, sm_version, dtype_str)
    return token_limit is None or tokens <= token_limit


def config_file_name(sm_version: int, D: int, c_m: int) -> str:
    """Return the tuning filename for one SM and (D, c_m) tuple."""
    return get_config_file_name(sm_version, D=D, cm=c_m)


_PWA_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


@functools.cache
def _max_fused_tokens_in_file(sm_version: int, D: int, c_m: int) -> Mapping[str, int | None]:
    """Read one tuning file's token limits by dtype. A bare limit is bf16's; a missing file or field lists none."""
    path = os.path.join(_PWA_CONFIGS_DIR, config_file_name(sm_version, D, c_m))
    if not os.path.exists(path):
        return MappingProxyType({})
    with open(path) as handle:
        raw = json.load(handle)
    limits = raw.get("max_fused_tokens", {})
    if not isinstance(limits, dict):
        limits = {"bf16": limits}
    for dtype_str, limit in limits.items():
        if dtype_str not in _FUSED_LIMIT_DTYPES:
            raise ValueError(f"max_fused_tokens in {path} lists {dtype_str!r}; expected one of {_FUSED_LIMIT_DTYPES}")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
            raise ValueError(
                f"{dtype_str} max_fused_tokens in {path} must be a positive integer or null, got {limit!r}"
            )
    return MappingProxyType(dict(limits))


@dataclass(frozen=True, slots=True)
class PWAConfigParams:
    """Immutable, source-free representation of ``PWAConfig`` parameters."""

    ab_dtype: str = "bf16"
    H: int = _KERNEL_H
    D: int = _KERNEL_D
    c_m: int = _KERNEL_CM
    TILE_I: int = 32
    TILE_S: int = 2
    TILE_J: int = 32
    KO_TILE: int = 64
    num_threads: int = 64
    atom_v: tuple[int, int, int] = (2, 1, 1)
    atom_p: tuple[int, int, int] = (1, 2, 1)
    num_stages: int = 4
    num_stages_w: int = 2
    swizzle: bool = True
    cluster: tuple[int, int] = (1, 1)

    @classmethod
    def from_dict(cls, params: Mapping[str, object]) -> PWAConfigParams:
        """Parse one raw JSON ``params`` mapping into immutable values."""
        known = {field.name for field in fields(cls)}
        values = {name: value for name, value in params.items() if name in known}
        for name in ("atom_v", "atom_p", "cluster"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)

    @classmethod
    def from_config(cls, config: object, *, ab_dtype: str | None = None) -> PWAConfigParams:
        """Copy a source ``PWAConfig``-compatible object without importing it."""
        values = {field.name: getattr(config, field.name) for field in fields(cls)}
        if ab_dtype is not None:
            values["ab_dtype"] = ab_dtype
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, object]:
        """Return parameters suitable for development-time source construction."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def machine_key(self) -> tuple[object, ...]:
        """Return every non-dtype parameter that changes generated machine code."""
        return (
            self.H,
            self.D,
            self.c_m,
            self.TILE_I,
            self.TILE_S,
            self.TILE_J,
            self.KO_TILE,
            self.num_threads,
            self.atom_v,
            self.atom_p,
            self.num_stages,
            self.num_stages_w,
            self.swizzle,
            self.cluster,
        )

    def config_key(self) -> tuple[object, ...]:
        """Match the historical source ``PWAConfig.config_key()`` contract."""
        return (self.ab_dtype, *self.machine_key())


@dataclass(frozen=True, slots=True)
class PWAConfigSelection:
    """One immutable tuning selection."""

    params: PWAConfigParams
    n_anchor: int | None = None
    s_anchor: int | None = None
    kernel_abi: str = "sm80"
    kernel_variant: str | None = None

    def machine_key(self) -> tuple[object, ...]:
        """Return the selected machine-code identity."""
        return self.params.machine_key()

    @property
    def transposed_v(self) -> bool:
        """Whether the selected kernel loads v as ``[B, S, H*D, N]`` with j contiguous."""
        return (self.kernel_abi, self.kernel_variant) in _TRANSPOSED_V_KERNELS

    @property
    def writes_gated_value(self) -> bool:
        """Whether the selected kernel stops at the gated value and leaves the Wo projection to the caller."""
        return self.kernel_abi == "sm100" and self.kernel_variant is None and self.params.TILE_S == _SM100_SPLIT_TILE_S


def loads_transposed_value(sm_version: int, D: int, c_m: int) -> bool:
    """Whether the kernel ``sm_version`` runs for ``(D, c_m)`` loads v transposed. One file's tiles share a kernel."""
    bundle = load_kernel_configs(_PWA_CONFIGS_DIR, config_file_name(sm_version, D, c_m))
    return bundle is not None and (bundle.kernel_abi, bundle.kernel_variant) in _TRANSPOSED_V_KERNELS


def _parse_key(key: str) -> tuple[int, int, str]:
    """Parse ``"N=512|S=512|dt=bf16"`` into its selection axes."""
    parts = dict(component.split("=", 1) for component in key.split("|"))
    return int(parts["N"]), int(parts["S"]), parts["dt"]


def default_params(dtype_str: str, H: int, D: int, c_m: int) -> PWAConfigParams:
    """Build the untuned parameters for one (H, D, c_m) tuple."""
    overrides = _DEFAULT_PARAMS_BY_DIMS.get((D, c_m), {})
    return PWAConfigParams(ab_dtype=dtype_str, H=H, D=D, c_m=c_m, **overrides)


def _default_selection(
    dtype_str: str,
    H: int,
    D: int,
    c_m: int,
    kernel_abi: str = "sm80",
    kernel_variant: str | None = None,
) -> PWAConfigSelection:
    """Build the historical untuned ``PWAConfig`` defaults without importing it."""
    return PWAConfigSelection(
        params=default_params(dtype_str, H, D, c_m), kernel_abi=kernel_abi, kernel_variant=kernel_variant
    )


_SelectionBucket = tuple[PWAConfigSelection, tuple[PWAConfigSelection, ...]]
# Selections by loaded bundle and call axes. An entry keeps its bundle alive, so no other bundle can take the id
# in its key; reloading a file, or another configs directory, yields a new bundle and a fresh selection.
_SELECTIONS: dict[tuple[object, ...], tuple[object, _SelectionBucket]] = {}
_MAX_SELECTIONS = 4096


def _select_pwa_config_selection_bucket(
    sm_version: int,
    I: int,
    J: int,
    S: int,
    dtype_str: str,
    H: int = _KERNEL_H,
    D: int = _KERNEL_D,
    c_m: int = _KERNEL_CM,
) -> _SelectionBucket:
    """Return the nearest selection and unique configs in its nearest-N bucket."""
    bundle = load_kernel_configs(_PWA_CONFIGS_DIR, config_file_name(sm_version, D, c_m))
    key = (id(bundle), sm_version, I, J, S, dtype_str, H, D, c_m)
    cached = _SELECTIONS.get(key)
    if cached is not None:
        return cached[1]
    selection = _select_in_bundle(bundle, I, J, S, dtype_str, H, D, c_m)
    if len(_SELECTIONS) >= _MAX_SELECTIONS:
        _SELECTIONS.clear()
    _SELECTIONS[key] = (bundle, selection)
    return selection


def _select_in_bundle(
    bundle: KernelConfigBundle | None, I: int, J: int, S: int, dtype_str: str, H: int, D: int, c_m: int
) -> _SelectionBucket:
    # Tuning files list configs per dtype. A dtype a file does not list keeps
    # the original PWAConfig defaults instead of borrowing another dtype's tiles.
    if bundle is not None:
        side = math.sqrt(float(I) * float(J))
        candidates = [
            (n_anchor, s_anchor, key)
            for key in bundle.configs
            for n_anchor, s_anchor, config_dtype in (_parse_key(key),)
            if config_dtype == dtype_str
        ]
        if candidates:
            n_bucket = min({n_anchor for n_anchor, _, _ in candidates}, key=lambda anchor: (abs(anchor - side), anchor))
            bucket_candidates = [candidate for candidate in candidates if candidate[0] == n_bucket]
            _, _, selected_key = min(
                bucket_candidates,
                key=lambda candidate: (abs(candidate[1] - S), candidate[1]),
            )

            def make_selection(key: str) -> PWAConfigSelection:
                n_anchor, s_anchor, _ = _parse_key(key)
                raw_params = bundle.configs[key]["params"]
                return PWAConfigSelection(
                    params=PWAConfigParams.from_dict(raw_params),
                    n_anchor=n_anchor,
                    s_anchor=s_anchor,
                    kernel_abi=bundle.kernel_abi,
                    kernel_variant=bundle.kernel_variant,
                )

            selected = make_selection(selected_key)
            variants: dict[tuple[object, ...], PWAConfigSelection] = {}
            for _, _, key in sorted(bucket_candidates, key=lambda candidate: candidate[1]):
                selection = make_selection(key)
                variants.setdefault(selection.params.config_key(), selection)
            return selected, tuple(variants.values())

    kernel = ("sm80", None) if bundle is None else (bundle.kernel_abi, bundle.kernel_variant)
    default = _default_selection(dtype_str, H, D, c_m, *kernel)
    return default, (default,)


def _select_pwa_config_bucket(
    sm_version: int,
    I: int,
    J: int,
    S: int,
    dtype_str: str,
    H: int = _KERNEL_H,
    D: int = _KERNEL_D,
    c_m: int = _KERNEL_CM,
) -> tuple[PWAConfigParams, tuple[PWAConfigParams, ...]]:
    """Return immutable params using the historical public helper shape."""
    selected, variants = _select_pwa_config_selection_bucket(
        sm_version,
        I,
        J,
        S,
        dtype_str,
        H=H,
        D=D,
        c_m=c_m,
    )
    return selected.params, tuple(variant.params for variant in variants)


def select_pwa_config(
    sm_version: int,
    I: int,
    J: int,
    S: int,
    dtype_str: str,
    H: int = _KERNEL_H,
    D: int = _KERNEL_D,
    c_m: int = _KERNEL_CM,
) -> PWAConfigParams:
    """Select immutable params from the nearest-S entry in the nearest-N bucket."""
    return _select_pwa_config_bucket(sm_version, I, J, S, dtype_str, H=H, D=D, c_m=c_m)[0]
