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


@dataclass(frozen=True)
class Sm100KernelConfig:
    """Tile of the SM100/SM103 kernel (2-SM tcgen05, persistent); shape-independent."""

    ab_dtype: str = "bf16"
    C: int = 32
    D: int = 32
    C_z: int = 128
    cc: int = 4  # c values per chunk; a CTA owns 128 / cc rows i, the CTA pair cc columns j
    tile_s: int = 64  # S per a/b ring stage
    ab_stages: int = 3  # a/b ring depth
    w_stages: int = 4  # W_o ring depth, 64 columns per stage
    z_stages: int = 2  # relayout buffers, one chunk (or half of one) each
    raster: int = 0  # j tiles per raster band; 0 walks whole rows of j tiles
    epi_cols: int = 64  # output channels per TMA-store slice
    b_res: int = 0  # 0 streams b per chunk; n > 0 keeps a tile's b resident, serving S <= n * tile_s
    pairs: int = 1  # CTA pairs per cluster; 2 multicasts the shared a and W across two j tiles
    z_split: int = 1  # parts of a chunk per relayout buffer; 2 halves the buffers

    @property
    def cz(self) -> int:
        return self.C_z

    def to_dict(self) -> dict:
        # z_split postdates the shipped tables and CUBINs: whole-chunk buffers keep the identity they were
        # built under.
        return {k: v for k, v in self.__dict__.items() if k != "z_split" or v != 1}

    @staticmethod
    def from_dict(raw: dict) -> Sm100KernelConfig:
        return Sm100KernelConfig(**raw)

    def config_key(self) -> tuple:
        return ("sm100", *self.to_dict().values())


_KERNEL_C = 32
_KERNEL_D = 32
_KERNEL_CZ = 128
# Output widths the fused op serves: Boltz/OpenFold (128) and Protenix-v2 (256).
_SUPPORTED_CZ = (128, 256)

# The SM100 kernel's family. SM103 runs the same machine code; its tiles (configs/sm103.json) were swept on a B300
# (SM clock locked at 1500 MHz) at every pair N of the 17 benchmark samples and the four tuned depths. SM100's tiles
# stay within 1.3% of the best from N = 768 up, but run up to 13% slower from N = 336 to 656, where one 2-SM pair
# per cluster (C_z = 128) or rastered j tiles (S = 597, up to N = 640) win. The N = 640 and N = 656 buckets hold
# those; N = 768 and up keep SM100's tiles.
_SM100_FAMILY = (100, 103)

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


# SM100/SM103 fallback tile per output width. It streams b, so it serves every S.
_SM100_DEFAULT_TILES = {
    128: {"cc": 4, "tile_s": 64, "ab_stages": 4, "w_stages": 4, "z_stages": 2, "epi_cols": 64},
    256: {"cc": 4, "tile_s": 64, "ab_stages": 3, "w_stages": 4, "z_stages": 2, "epi_cols": 32},
}


def sm100_default_config(dtype_str: str, C_z: int = _KERNEL_CZ) -> Sm100KernelConfig:
    """Return the SM100/SM103 fallback tile for ``C_z``."""
    return Sm100KernelConfig(ab_dtype=dtype_str, C_z=C_z, **_SM100_DEFAULT_TILES[C_z])


def sm100_holds(config: Sm100KernelConfig, S: int) -> bool:
    """Whether ``config`` serves depth ``S``: it streams ``b``, or its resident ``b`` holds ``S``."""
    return not config.b_res or -(-S // config.tile_s) <= config.b_res


def sm100_fit(config: Sm100KernelConfig, S: int) -> Sm100KernelConfig:
    """``config``, or the streaming fallback tile when its resident ``b`` cannot hold ``S``."""
    return config if sm100_holds(config, S) else sm100_default_config(config.ab_dtype, config.C_z)


def sm100_a_chunks(sm_version: int, C_z: int) -> tuple[int, ...]:
    """The chunk width ``cc`` of every tile the SM100 selector can pick on ``sm_version`` for ``C_z``: the table's
    and the default it falls back to."""
    bundle = load_kernel_configs(_OPM_CONFIGS_DIR, get_config_file_name(sm_version))
    raws = bundle.configs.values() if bundle is not None and bundle.kernel_abi == "sm100" else ()
    chunks = {Sm100KernelConfig.from_dict(raw).cc for raw in raws if raw.get("C_z", _KERNEL_CZ) == C_z}
    return tuple(sorted(chunks | {sm100_default_config("bf16", C_z).cc}))


type OpmConfig = KernelConfig | Sm90KernelConfig | Sm100KernelConfig


def config_identity(config: OpmConfig) -> str:
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
) -> tuple[OpmConfig, tuple[OpmConfig, ...]]:
    """Return the nearest-S config and unique configs in its nearest-N bucket."""
    bundle = load_kernel_configs(_OPM_CONFIGS_DIR, get_config_file_name(sm_version))
    sm90 = bundle is not None and bundle.kernel_abi == "sm90"
    sm100 = bundle.kernel_abi == "sm100" if bundle is not None else sm_version in _SM100_FAMILY
    if sm90 and dtype_str != "bf16":
        # The SM90 kernel is bf16-only; other dtypes keep the SM80 kernel.
        default = default_config(dtype_str, C=C, D=D, C_z=C_z, sm=sm_version)
        return default, (default,)
    config_cls = Sm100KernelConfig if sm100 else Sm90KernelConfig if sm90 else KernelConfig

    def load(raw: dict) -> OpmConfig:
        # SM100 tiles are tuned in bf16 and serve fp16 unchanged.
        return config_cls.from_dict(dict(raw, ab_dtype=dtype_str) if sm100 else raw)

    def fit(config: OpmConfig) -> OpmConfig:
        return sm100_fit(config, S) if sm100 else config

    if bundle is not None:
        side = math.sqrt(float(I) * float(J))
        candidates = []
        for key in bundle.configs:
            kN, kS, knb, kbias, kdt = _parse_key(key)
            raw = bundle.configs[key]
            if (kdt != dtype_str and not sm100) or (
                raw.get("C", _KERNEL_C),
                raw.get("D", _KERNEL_D),
                raw.get("C_z", _KERNEL_CZ),
            ) != (C, D, C_z):
                continue
            # SM90 and SM100 tiles serve every flag combination; SM80 tiles are tuned per flag.
            if not (sm90 or sm100) and (knb != norm_before or kbias != has_bias):
                continue
            candidates.append((kN, kS, key))
        if candidates:
            n_bucket = min({kN for kN, _, _ in candidates}, key=lambda kN: (abs(kN - side), kN))
            in_bucket = [candidate for candidate in candidates if candidate[0] == n_bucket]
            # SM100: the nearest S among the tiles whose resident b holds S; fit() streams when none does.
            holding = [c for c in in_bucket if not sm100 or sm100_holds(load(bundle.configs[c[2]]), S)]
            _, _, best_key = min(holding or in_bucket, key=lambda candidate: (abs(candidate[1] - S), candidate[1]))
            selected = fit(load(bundle.configs[best_key]))
            variants = {}
            for _, _, key in sorted(in_bucket, key=lambda candidate: candidate[1]):
                config = fit(load(bundle.configs[key]))
                variants.setdefault(config.config_key(), config)
            return selected, tuple(variants.values())
    if sm100:
        default = sm100_default_config(dtype_str, C_z=C_z)
    elif sm90:
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
) -> OpmConfig:
    """Select the nearest-S config from the nearest tuned N bucket."""
    return _select_opm_config_bucket(sm_version, I, J, S, norm_before, has_bias, dtype_str, C=C, D=D, C_z=C_z)[0]
