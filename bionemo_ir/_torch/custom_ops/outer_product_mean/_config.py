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
"""Tuned kernel configuration for fused outer-product-mean."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from bionemo_ir._torch.utils.kernel import get_config_file_name, load_kernel_configs


@dataclass
class KernelConfig:
    """Machine-code configuration shared by source and CUBIN selection."""

    ab_dtype: str = "bf16"
    TILE_I: int = 4
    TILE_J: int = 8
    TILE_C: int = 32
    TILE_S: int = 32
    KO_TILE: int = 64
    num_stages: int = 3
    num_stages_w: int = 2
    reg_prefetch: bool = True
    vec_epilogue: bool = True
    swizzle: bool = True
    swizzle_ab: bool = True
    raster_factor: int = 8
    C: int = 32
    D: int = 32
    C_z: int = 128
    num_threads: int = 256
    atom_layout_s: tuple[int, int, int] = (2, 4, 1)
    atom_layout_o: tuple[int, int, int] = (2, 4, 1)

    def to_dict(self) -> dict:
        result = dict(self.__dict__)
        result["atom_layout_s"] = list(self.atom_layout_s)
        result["atom_layout_o"] = list(self.atom_layout_o)
        return result

    @staticmethod
    def from_dict(raw: dict) -> KernelConfig:
        config = KernelConfig()
        for key, value in raw.items():
            setattr(config, key, tuple(value) if key.startswith("atom_layout") else value)
        return config

    def config_key(self) -> tuple:
        return (
            self.ab_dtype,
            self.TILE_I,
            self.TILE_J,
            self.TILE_C,
            self.TILE_S,
            self.KO_TILE,
            self.num_stages,
            self.num_stages_w,
            self.reg_prefetch,
            self.vec_epilogue,
            self.swizzle,
            self.swizzle_ab,
            self.raster_factor,
            self.C,
            self.D,
            self.C_z,
            self.num_threads,
            tuple(self.atom_layout_s),
            tuple(self.atom_layout_o),
        )


@dataclass(frozen=True)
class Sm90KernelConfig:
    """Tile of the SM90 kernel (TMA + wgmma, warp specialized); shape-independent."""

    ab_dtype: str = "bf16"
    C: int = 32
    D: int = 32
    C_z: int = 128
    TI: int = 8  # i indices per CTA; the S-GEMM M is TI * C / csplit
    TJ: int = 6  # j indices per CTA; the S-GEMM N is TJ * D
    TILE_S: int = 32  # S-GEMM K tile
    stages: int = 4  # a/b ring depth
    KO: int = 64  # O-GEMM K tile (one W_o chunk)
    w_stages: int = 4  # W_o ring depth
    raster: int = 8  # L2 rasterization group
    rswap: int = 0  # rasterize in j strips instead of i strips
    o_lag: int = 1  # O-GEMM wgmma groups kept in flight across W_o chunks
    csplit: int = 1  # S-GEMM passes over the C axis
    zsep: int = 0  # 1: sZ gets its own SMEM instead of aliasing the a/b ring
    ptile: int = 1  # (i, j) tiles per CTA
    zstm: int = 0  # relayout the S-GEMM accumulator to SMEM with stmatrix

    @property
    def cz(self) -> int:
        return self.C_z

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(raw: dict) -> Sm90KernelConfig:
        return Sm90KernelConfig(**raw)

    def config_key(self) -> tuple:
        return ("sm90", *self.__dict__.values())


_KERNEL_C = 32
_KERNEL_D = 32
_KERNEL_CZ = 128
# Output widths the fused op serves: Boltz/OpenFold (128) and Protenix-v2 (256).
_SUPPORTED_CZ = (128, 256)

_OPM_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


# The default tile exceeds SM86/89 shared memory. The W_o slab (KO_TILE x C_z) shares
# the aliased S-GEMM ring, so the wider output halves KO_TILE.
_SMEM_CONSTRAINED_SMS = (86, 89)
_SMEM_CONSTRAINED_TILE_BY_CZ = {
    128: {"TILE_S": 16, "num_stages": 2, "KO_TILE": 32},
    256: {"TILE_S": 16, "num_stages": 2, "KO_TILE": 16},
}


def default_config(
    dtype_str: str, C: int = _KERNEL_C, D: int = _KERNEL_D, C_z: int = _KERNEL_CZ, sm: int | None = None
) -> KernelConfig:
    """Return the shipped fallback tile."""
    overrides = _SMEM_CONSTRAINED_TILE_BY_CZ.get(C_z, {}) if sm in _SMEM_CONSTRAINED_SMS else {}
    return KernelConfig(ab_dtype=dtype_str, C=C, D=D, C_z=C_z, **overrides)


# SM90 fallback tile per output width.
_SM90_DEFAULT_TILES = {
    128: {
        "TI": 8,
        "TJ": 6,
        "csplit": 1,
        "zsep": 0,
        "ptile": 1,
        "stages": 4,
        "KO": 64,
        "w_stages": 5,
        "o_lag": 1,
        "raster": 16,
        "zstm": 1,
    },
    256: {
        "TI": 8,
        "TJ": 8,
        "csplit": 2,
        "zsep": 0,
        "ptile": 1,
        "stages": 5,
        "KO": 64,
        "w_stages": 3,
        "o_lag": 0,
        "raster": 8,
        "zstm": 1,
    },
}


def sm90_default_config(dtype_str: str, C_z: int = _KERNEL_CZ) -> Sm90KernelConfig:
    """Return the SM90 fallback tile for ``C_z``."""
    return Sm90KernelConfig(ab_dtype=dtype_str, C_z=C_z, **_SM90_DEFAULT_TILES[C_z])


def config_identity(config: KernelConfig | Sm90KernelConfig) -> str:
    """Return the CUBIN selection identity."""
    return "|".join(
        ",".join(str(item) for item in value) if isinstance(value, tuple) else str(value)
        for value in config.config_key()
    )


def _parse_key(key: str):
    """``"N=128|S=256|nb=1|bias=1|dt=bf16[|cz=256]"`` -> (N, S, norm_before, has_bias, dt)."""
    p = dict(kv.split("=", 1) for kv in key.split("|"))
    return int(p["N"]), int(p["S"]), p["nb"] == "1", p["bias"] == "1", p["dt"]


def _select_opm_config_bucket(
    sm_version: int,
    I: int,
    J: int,
    S: int,
    norm_before: bool,
    has_bias: bool,
    dtype_str: str,
    C: int = _KERNEL_C,
    D: int = _KERNEL_D,
    C_z: int = _KERNEL_CZ,
) -> tuple[KernelConfig | Sm90KernelConfig, tuple[KernelConfig | Sm90KernelConfig, ...]]:
    """Return the nearest-S config and unique configs in its nearest-N bucket."""
    bundle = load_kernel_configs(_OPM_CONFIGS_DIR, get_config_file_name(sm_version))
    sm90 = bundle is not None and bundle.kernel_abi == "sm90"
    if sm90 and dtype_str != "bf16":
        # The SM90 kernel is bf16-only; other dtypes keep the SM80 kernel.
        default = default_config(dtype_str, C=C, D=D, C_z=C_z, sm=sm_version)
        return default, (default,)
    config_cls = Sm90KernelConfig if sm90 else KernelConfig
    if bundle is not None:
        side = math.sqrt(float(I) * float(J))
        candidates = []
        for key in bundle.configs:
            kN, kS, knb, kbias, kdt = _parse_key(key)
            raw = bundle.configs[key]
            if kdt != dtype_str or (raw.get("C", _KERNEL_C), raw.get("D", _KERNEL_D), raw.get("C_z", _KERNEL_CZ)) != (
                C,
                D,
                C_z,
            ):
                continue
            # SM90 tiles serve every flag combination; SM80 tiles are tuned per flag.
            if not sm90 and (knb != norm_before or kbias != has_bias):
                continue
            candidates.append((kN, kS, key))
        if candidates:
            n_bucket = min({kN for kN, _, _ in candidates}, key=lambda kN: (abs(kN - side), kN))
            _, _, best_key = min(
                (candidate for candidate in candidates if candidate[0] == n_bucket),
                key=lambda candidate: (abs(candidate[1] - S), candidate[1]),
            )
            selected = config_cls.from_dict(bundle.configs[best_key])
            variants = {}
            for _, _, key in sorted(
                (candidate for candidate in candidates if candidate[0] == n_bucket), key=lambda candidate: candidate[1]
            ):
                config = config_cls.from_dict(bundle.configs[key])
                variants.setdefault(config.config_key(), config)
            return selected, tuple(variants.values())
    if sm90:
        default = sm90_default_config(dtype_str, C_z=C_z)
    else:
        default = default_config(dtype_str, C=C, D=D, C_z=C_z, sm=sm_version)
    return default, (default,)


def select_opm_config(
    sm_version: int,
    I: int,
    J: int,
    S: int,
    norm_before: bool,
    has_bias: bool,
    dtype_str: str,
    C: int = _KERNEL_C,
    D: int = _KERNEL_D,
    C_z: int = _KERNEL_CZ,
) -> KernelConfig | Sm90KernelConfig:
    """Select the nearest-S config from the nearest tuned N bucket."""
    return _select_opm_config_bucket(sm_version, I, J, S, norm_before, has_bias, dtype_str, C=C, D=D, C_z=C_z)[0]
