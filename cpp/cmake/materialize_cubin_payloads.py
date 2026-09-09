#!/usr/bin/env python3
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
"""Verify public CUBIN packs and materialize deterministic native build inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import re
import sys
import tarfile
import tempfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn, cast

INDEX_FORMAT = "bioir-cubin-pack-v1"
REGISTRY_VERSION = 2
MATERIALIZATION_FORMAT = "bioir-cubin-materialization-v1"
ASSEMBLY_FORMAT = "bioir-cubin-assembly-v1"
FINGERPRINT_FORMAT = "bioir-cubin-materialization-fingerprint-v1"

_FAMILIES = frozenset(
    {
        "adaln_layernorm_sigmoid",
        "dual_gemm_x0_x1",
        "dual_gemm_x_x",
        "gated_sigmoid",
        "outer_product_mean",
        "pair_weighted_averaging",
        "pairwise_attention",
        "triangle_attention",
    }
)
_DTYPES = {
    "adaln_layernorm_sigmoid": frozenset({"fp16", "bf16", "fp32"}),
    **{family: frozenset({"fp16", "bf16"}) for family in _FAMILIES if family != "adaln_layernorm_sigmoid"},
}
_HEX20_RE = re.compile(r"[0-9a-f]{20}")
_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_FAMILY_RE = re.compile(r"[a-z][a-z0-9_]{0,127}")
_ARCH_RE = re.compile(r"sm_(?P<sm>[0-9]{2,3})(?P<suffix>[af]?)")
_ABI_RE = re.compile(r"[a-z][a-z0-9_]{0,127}")
_SYMBOL_RE = re.compile(r"k[0-9a-f_]{1,511}")
_LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"

_MAX_INDEX_BYTES = 64 * 1024 * 1024
_MAX_PACK_BYTES = 8 * 1024 * 1024 * 1024
_MAX_IMAGE_BYTES = 256 * 1024 * 1024
_MAX_FAMILY_IMAGES = 65_536
_MAX_TAR_BYTES = 16 * 1024 * 1024 * 1024
_IO_CHUNK = 1024 * 1024
_UINT32_MAX = (1 << 32) - 1
_INT32_MAX = (1 << 31) - 1
_XZ_MAGIC = b"\xfd7zXZ\x00"
_XZ_FOOTER_MAGIC = b"YZ"
_XZ_STREAM_FLAGS = b"\x00\x04"
_XZ_LZMA2_DICT_PROPERTY_8_MIB = b"\x16"

_COMMON_METADATA = frozenset({"dynamic_smem_bytes", "non_portable_cluster_size_allowed"})
# One declaration per family, replacing what used to be five parallel switch
# statements. Each field appears exactly once and drives, in this order:
#   * the accepted runtime-metadata key set,
#   * the value validation,
#   * the generated C++ ``CubinImage`` member,
#   * that member's initializer in the generated registry.
# Field order IS the C++ struct layout and the initializer order, so a launcher
# reading `image.<name>` cannot drift from what the index declares.
_POSITIVE = "positive"  # 1..UINT32_MAX
_COUNT = "count"  # 0..UINT32_MAX
_INDEX = "index"  # 0..INT32_MAX
_BOOL = "bool"
_TEXT = "text"
_TRIPLE = "triple"  # exactly three integers, each >= 1
_DTYPE_CODE = "dtype_code"  # derived from the variant, not from metadata
_NO_DEFAULT = object()


@dataclass(frozen=True)
class _Field:
    """One runtime-metadata value and the C++ member it materializes into.

    ``default`` supplies the value for an axis that predates the tracked
    artifact corpus. The generated C++ member is always present.
    """

    name: str
    kind: str
    declaration: str
    suffix: str = ""
    default: object = _NO_DEFAULT

    @property
    def from_metadata(self) -> bool:
        return self.kind != _DTYPE_CODE


@dataclass(frozen=True)
class _AliasSpec:
    """Per-variant ``RuntimeAlias`` table appended to the image record."""

    keys: tuple[str, str]
    declarations: tuple[str, str]


@dataclass(frozen=True)
class _Sm90Spec:
    """Native-SM90 launch record; families differ only in name and operands."""

    enabled_field: str
    operands: tuple[str, ...]


@dataclass(frozen=True)
class _FamilySpec:
    """Everything family-specific about one registry."""

    fields: tuple[_Field, ...]
    # Runtime uniqueness key, in order. "@name" reads the per-alias value, which
    # also makes the key expand to one entry per alias.
    runtime_key: tuple[str, ...]
    dtype_key: str = "is_bfloat16"
    alias: _AliasSpec | None = None
    sm90: _Sm90Spec | None = None

    @property
    def metadata_keys(self) -> frozenset[str]:
        keys = {field.name for field in self.fields if field.from_metadata}
        keys.add(self.dtype_key)
        if self.alias is not None:
            keys.add("runtime_aliases")
        if self.sm90 is not None:
            keys.add("sm90_launch")
        return frozenset(keys)


_ATTENTION_SPEC = _FamilySpec(
    fields=(
        _Field("head_dim", _POSITIVE, "std::int32_t head_dim;"),
        _Field("bucket", _INDEX, "std::int32_t bucket;"),
        _Field("is_bfloat16", _BOOL, "bool is_bfloat16;"),
        _Field("packed_output", _BOOL, "bool packed_output;"),
    ),
    runtime_key=("head_dim", "bucket", "is_bfloat16", "packed_output"),
    sm90=_Sm90Spec("enabled", ("q", "k", "v", "bias", "output")),
)

_FAMILY_SPECS: dict[str, _FamilySpec] = {
    "adaln_layernorm_sigmoid": _FamilySpec(
        fields=(
            _Field("dtype", _DTYPE_CODE, "std::uint8_t dtype;"),
            _Field("feature_dim", _POSITIVE, "std::int32_t feature_dim;"),
            _Field("threads_per_row", _POSITIVE, "std::int32_t threads_per_row;"),
            _Field("num_threads", _POSITIVE, "std::int32_t num_threads;"),
            _Field("cluster_n", _POSITIVE, "std::int32_t cluster_n;"),
            # Every image published before the norm axis existed is LayerNorm.
            _Field("is_rms_norm", _BOOL, "bool is_rms_norm;", default=False),
        ),
        runtime_key=("dtype_name", "feature_dim", "threads_per_row", "num_threads", "is_rms_norm"),
        dtype_key="dtype_name",
    ),
    "gated_sigmoid": _FamilySpec(
        fields=(
            _Field("is_bfloat16", _BOOL, "bool is_bfloat16;"),
            _Field("has_bias", _BOOL, "bool has_bias;"),
            _Field("m_block_size", _POSITIVE, "std::int32_t m_block_size;"),
            _Field("n_block_size", _POSITIVE, "std::int32_t n_block_size;"),
            _Field("k_block_size", _POSITIVE, "std::int32_t k_block_size;"),
            _Field("num_stages", _POSITIVE, "std::int32_t num_stages;"),
            _Field("raster_factor", _COUNT, "std::int32_t raster_factor;"),
            _Field("num_threads", _POSITIVE, "std::int32_t num_threads;"),
            _Field("atom_layout_mnk", _TRIPLE, "std::int32_t atom_layout_mnk[3];"),
        ),
        runtime_key=(
            "is_bfloat16",
            "has_bias",
            "m_block_size",
            "n_block_size",
            "k_block_size",
            "num_stages",
            "raster_factor",
            "atom_layout_mnk",
        ),
    ),
    "outer_product_mean": _FamilySpec(
        fields=(
            _Field("is_bfloat16", _BOOL, "bool is_bfloat16;"),
            _Field("has_bias", _BOOL, "bool has_bias;"),
            _Field("norm_before", _BOOL, "bool norm_before;"),
            _Field("config_identity", _TEXT, "char const* config_identity;"),
            _Field("tile_i", _POSITIVE, "std::int32_t tile_i;"),
            _Field("tile_j", _POSITIVE, "std::int32_t tile_j;"),
            _Field("raster_factor", _COUNT, "std::int32_t raster_factor;"),
            _Field("num_threads", _POSITIVE, "std::int32_t num_threads;"),
        ),
        runtime_key=("is_bfloat16", "has_bias", "norm_before", "config_identity"),
    ),
    "pair_weighted_averaging": _FamilySpec(
        fields=(
            _Field("is_bfloat16", _BOOL, "bool is_bfloat16;"),
            _Field("H", _POSITIVE, "std::int32_t H;"),
            _Field("D", _POSITIVE, "std::int32_t D;"),
            _Field("c_m", _POSITIVE, "std::int32_t c_m;"),
            _Field("tile_i", _POSITIVE, "std::uint32_t tile_i;", suffix="U"),
            _Field("tile_s", _POSITIVE, "std::uint32_t tile_s;", suffix="U"),
            _Field("tile_j", _POSITIVE, "std::uint32_t tile_j;", suffix="U"),
            _Field("num_threads", _POSITIVE, "std::uint32_t num_threads;", suffix="U"),
        ),
        runtime_key=("is_bfloat16", "D", "c_m", "@n_anchor", "@s_anchor"),
        alias=_AliasSpec(
            ("n_anchor", "s_anchor"),
            ("std::int32_t n_anchor;", "std::int32_t s_anchor;"),
        ),
    ),
    "pairwise_attention": _ATTENTION_SPEC,
    "triangle_attention": _ATTENTION_SPEC,
    "dual_gemm_x_x": _FamilySpec(
        fields=(
            _Field("K", _POSITIVE, "std::int32_t K;"),
            _Field("is_bfloat16", _BOOL, "bool is_bfloat16;"),
            _Field("transpose_out", _BOOL, "bool transpose_out;"),
            _Field("has_bias", _BOOL, "bool has_bias;"),
            _Field("has_mask", _BOOL, "bool has_mask;"),
            _Field("tile_m", _POSITIVE, "std::uint32_t tile_m;", suffix="U"),
            _Field("tile_n", _POSITIVE, "std::uint32_t tile_n;", suffix="U"),
            _Field("tile_k", _POSITIVE, "std::uint32_t tile_k;", suffix="U"),
            _Field("num_threads", _POSITIVE, "std::uint32_t num_threads;", suffix="U"),
            _Field("raster_factor", _COUNT, "std::uint32_t raster_factor;", suffix="U"),
        ),
        runtime_key=("K", "@N", "@bucket", "is_bfloat16", "transpose_out", "has_bias", "has_mask"),
        alias=_AliasSpec(("N", "bucket"), ("std::int32_t N;", "std::int32_t bucket;")),
        sm90=_Sm90Spec("enabled", ("x0", "x1", "w0", "w1", "output")),
    ),
    "dual_gemm_x0_x1": _FamilySpec(
        fields=(
            _Field("K", _POSITIVE, "std::int32_t K;"),
            _Field("N", _POSITIVE, "std::int32_t N;"),
            _Field("bucket", _INDEX, "std::int32_t bucket;"),
            _Field("is_bfloat16", _BOOL, "bool is_bfloat16;"),
            _Field("has_bias", _BOOL, "bool has_bias;"),
            _Field("tile_m", _POSITIVE, "std::uint32_t tile_m;", suffix="U"),
            _Field("tile_n", _POSITIVE, "std::uint32_t tile_n;", suffix="U"),
            _Field("num_threads", _POSITIVE, "std::uint32_t num_threads;", suffix="U"),
            _Field("raster_factor", _COUNT, "std::uint32_t raster_factor;", suffix="U"),
        ),
        runtime_key=("K", "N", "bucket", "is_bfloat16", "has_bias"),
        sm90=_Sm90Spec("is_native", ("x0", "x1", "w0", "w1", "output")),
    ),
}

# Public tuning aliases are index records, not launch metadata: each family
# declares its own key set, and two carry a "everything else" sentinel form.
_PUBLIC_ALIAS_FIELDS: dict[str, tuple[tuple[str, int | None], ...]] = {
    "adaln_layernorm_sigmoid": (("feature_dim", 1), ("num_threads", 1)),
    "dual_gemm_x0_x1": (("K", 1), ("N", 1), ("has_bias", None)),
    "dual_gemm_x_x": (("N", 1), ("bucket", 0)),
    "gated_sigmoid": (("K", 1), ("N", 1), ("m_bucket", 0)),
    "outer_product_mean": (("N", 1), ("S", 1)),
    "pair_weighted_averaging": (("n_anchor", 1), ("s_anchor", 1)),
    "pairwise_attention": (("head_dim", 1), ("packed", None)),
    "triangle_attention": (("head_dim", 1), ("packed", None)),
}
_PUBLIC_ALIAS_SENTINELS = {"gated_sigmoid": "fallback_tile", "outer_product_mean": "default_config"}
_DTYPE_CODES = {"fp16": 0, "bf16": 1, "fp32": 2}

_TMA_DATA_TYPE = {
    "float16": "CU_TENSOR_MAP_DATA_TYPE_FLOAT16",
    "bfloat16": "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16",
}
_TMA_INTERLEAVE = {"none": "CU_TENSOR_MAP_INTERLEAVE_NONE"}
_TMA_SWIZZLE = {
    "none": "CU_TENSOR_MAP_SWIZZLE_NONE",
    "32b": "CU_TENSOR_MAP_SWIZZLE_32B",
    "64b": "CU_TENSOR_MAP_SWIZZLE_64B",
    "128b": "CU_TENSOR_MAP_SWIZZLE_128B",
}
_TMA_L2_PROMOTION = {"128b": "CU_TENSOR_MAP_L2_PROMOTION_L2_128B"}
_TMA_OOB_FILL = {"none": "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE"}
_CLUSTER_POLICY = {"default": "CU_CLUSTER_SCHEDULING_POLICY_DEFAULT"}


class MaterializationError(RuntimeError):
    """A public index, pack, image, or generated-output validation failed."""


@dataclass(frozen=True)
class MaterializationResult:
    """Location of one complete content-addressed materialization."""

    fingerprint: str
    output_dir: Path


@dataclass(frozen=True)
class VerificationResult:
    """Summary of source-independent index and pack verification."""

    family_count: int
    pack_count: int
    variant_count: int
    image_count: int
    total_image_bytes: int


@dataclass(frozen=True)
class PackRecord:
    target_arch: str
    file: str
    sha256: str
    size: int
    image_count: int


@dataclass(frozen=True)
class ImageRecord:
    sha256: str
    size: int
    pack: str


@dataclass(frozen=True)
class VariantRecord:
    variant_id: str
    target_sm: int
    target_arch: str
    kernel_sm: int
    launch_abi: str
    supported_sms: tuple[int, ...]
    dtype: str
    label: str
    identity_spec: dict[str, object]
    aliases: tuple[dict[str, object], ...]
    runtime_metadata: dict[str, object]
    kernel_symbol: str
    image: ImageRecord


@dataclass(frozen=True)
class FamilyIndex:
    family: str
    path: Path
    raw_bytes: bytes
    compile_fingerprint: str
    packs: tuple[PackRecord, ...]
    variants: tuple[VariantRecord, ...]


def _fail(message: str) -> NoReturn:
    raise MaterializationError(message)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            _fail(f"JSON object contains duplicate key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> NoReturn:
    _fail(f"JSON contains non-finite number {value}")


def _as_object(value: object, where: str) -> dict[str, object]:
    if type(value) is not dict:
        _fail(f"{where} must be an object")
    return cast(dict[str, object], value)


def _as_list(value: object, where: str) -> list[object]:
    if type(value) is not list:
        _fail(f"{where} must be an array")
    return cast(list[object], value)


def _exact_keys(value: Mapping[str, object], expected: frozenset[str] | set[str], where: str) -> None:
    actual = set(value)
    if actual != set(expected):
        _fail(
            f"{where} has invalid keys: missing={sorted(set(expected) - actual)}, extra={sorted(actual - set(expected))}"
        )


def _string(value: object, where: str, *, maximum: int = 4096) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        _fail(f"{where} must be a non-empty string of at most {maximum} characters")
    result = cast(str, value)
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in result):
        _fail(f"{where} contains a control character")
    return result


def _integer(value: object, where: str, *, minimum: int = 0, maximum: int = _INT32_MAX) -> int:
    if type(value) is not int or not minimum <= cast(int, value) <= maximum:
        _fail(f"{where} must be an integer in [{minimum}, {maximum}]")
    return cast(int, value)


def _boolean(value: object, where: str) -> bool:
    if type(value) is not bool:
        _fail(f"{where} must be a boolean")
    return cast(bool, value)


def _integer_array(
    value: object,
    where: str,
    *,
    length: int | None = None,
    minimum: int = 0,
    maximum: int = _UINT32_MAX,
) -> tuple[int, ...]:
    raw = _as_list(value, where)
    if length is not None and len(raw) != length:
        _fail(f"{where} must contain exactly {length} integers")
    return tuple(
        _integer(item, f"{where}[{index}]", minimum=minimum, maximum=maximum) for index, item in enumerate(raw)
    )


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def _read_index(path: Path) -> tuple[dict[str, object], bytes]:
    if path.is_symlink():
        _fail(f"CUBIN index must not be a symlink: {path}")
    try:
        stat = path.stat()
    except OSError as error:
        raise MaterializationError(f"cannot stat CUBIN index {path}: {error}") from error
    if not path.is_file() or stat.st_size > _MAX_INDEX_BYTES:
        _fail(f"CUBIN index is not a regular file or exceeds {_MAX_INDEX_BYTES} bytes: {path}")
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_strict_object, parse_constant=_reject_json_constant)
    except MaterializationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializationError(f"cannot read valid JSON from {path}: {error}") from error
    if not raw.endswith(b"\n"):
        _fail(f"CUBIN index must end with a newline: {path}")
    return _as_object(value, str(path)), raw


def _validate_tma_descriptor(value: object, where: str, dtype: str, rank: int) -> dict[str, object]:
    descriptor = _as_object(value, where)
    _exact_keys(
        descriptor,
        {
            "data_type",
            "rank",
            "global_dim_order",
            "box_dims",
            "element_strides",
            "interleave",
            "swizzle",
            "l2_promotion",
            "oob_fill",
        },
        where,
    )
    if _integer(descriptor["rank"], f"{where}.rank", minimum=1, maximum=5) != rank:
        _fail(f"{where}.rank must be {rank}")
    expected_dtype = "bfloat16" if dtype == "bf16" else "float16"
    if _string(descriptor["data_type"], f"{where}.data_type") != expected_dtype:
        _fail(f"{where}.data_type does not match dtype {dtype}")
    order = _integer_array(descriptor["global_dim_order"], f"{where}.global_dim_order", length=rank)
    if sorted(order) != list(range(rank)):
        _fail(f"{where}.global_dim_order must be a permutation of [0, {rank})")
    _integer_array(descriptor["box_dims"], f"{where}.box_dims", length=rank, minimum=1)
    _integer_array(descriptor["element_strides"], f"{where}.element_strides", length=rank, minimum=1)
    for key, values in (
        ("data_type", _TMA_DATA_TYPE),
        ("interleave", _TMA_INTERLEAVE),
        ("swizzle", _TMA_SWIZZLE),
        ("l2_promotion", _TMA_L2_PROMOTION),
        ("oob_fill", _TMA_OOB_FILL),
    ):
        if _string(descriptor[key], f"{where}.{key}") not in values:
            _fail(f"{where}.{key} has an unsupported value")
    return descriptor


def _validate_sm90_launch(family: str, value: object, where: str, dtype: str, kernel_sm: int) -> None:
    if kernel_sm != 90:
        if value is not None:
            _fail(f"{where} must be null for a non-SM90 kernel ABI")
        return
    if value is None:
        _fail(f"{where} is required for an SM90 kernel ABI")
    launch = _as_object(value, where)
    is_dual = family in {"dual_gemm_x_x", "dual_gemm_x0_x1"}
    expected_keys = {"block_dims", "cluster_dims", "cluster_scheduling_policy", "tma_descriptors"}
    if is_dual:
        expected_keys.add("epi_tile")
    _exact_keys(launch, expected_keys, where)
    block = _integer_array(launch["block_dims"], f"{where}.block_dims", length=3, minimum=1)
    if block[0] * block[1] * block[2] > 1024:
        _fail(f"{where}.block_dims exceeds 1024 threads")
    _integer_array(launch["cluster_dims"], f"{where}.cluster_dims", length=3, minimum=1)
    if _string(launch["cluster_scheduling_policy"], f"{where}.cluster_scheduling_policy") not in _CLUSTER_POLICY:
        _fail(f"{where}.cluster_scheduling_policy is unsupported")
    if is_dual:
        _integer_array(launch["epi_tile"], f"{where}.epi_tile", length=2, minimum=1)
    descriptor_names = ("x0", "x1", "w0", "w1", "output") if is_dual else ("q", "k", "v", "bias", "output")
    descriptors = _as_object(launch["tma_descriptors"], f"{where}.tma_descriptors")
    _exact_keys(descriptors, set(descriptor_names), f"{where}.tma_descriptors")
    rank = 2 if is_dual else 4
    for name in descriptor_names:
        _validate_tma_descriptor(descriptors[name], f"{where}.tma_descriptors.{name}", dtype, rank)


def _validate_runtime_aliases(value: object, where: str, keys: tuple[str, str]) -> tuple[dict[str, object], ...]:
    aliases: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for index, raw in enumerate(_as_list(value, where)):
        alias = _as_object(raw, f"{where}[{index}]")
        _exact_keys(alias, set(keys), f"{where}[{index}]")
        normalized = tuple(_integer(alias[key], f"{where}[{index}].{key}", minimum=0) for key in keys)
        if normalized in seen:
            _fail(f"{where} contains duplicate alias {normalized}")
        seen.add(normalized)
        aliases.append(alias)
    if not aliases:
        _fail(f"{where} must not be empty")
    if [tuple(alias[key] for key in keys) for alias in aliases] != sorted(
        tuple(alias[key] for key in keys) for alias in aliases
    ):
        _fail(f"{where} must be sorted")
    return tuple(aliases)


def _validate_public_aliases(family: str, aliases: tuple[dict[str, object], ...], where: str) -> None:
    if not aliases:
        _fail(f"{where} must not be empty")
    encoded = [_json_bytes(alias) for alias in aliases]
    if encoded != sorted(set(encoded)):
        _fail(f"{where} must be unique and sorted by canonical JSON")
    fields = _PUBLIC_ALIAS_FIELDS[family]
    sentinel = _PUBLIC_ALIAS_SENTINELS.get(family)
    for index, alias in enumerate(aliases):
        alias_where = f"{where}[{index}]"
        if sentinel is not None and set(alias) == {sentinel}:
            if alias[sentinel] is not True:
                _fail(f"{alias_where}.{sentinel} must be true")
            continue
        _exact_keys(alias, {key for key, _ in fields}, alias_where)
        for key, minimum in fields:
            if minimum is None:
                _boolean(alias[key], f"{alias_where}.{key}")
            else:
                _integer(alias[key], f"{alias_where}.{key}", minimum=minimum)


def _validate_field(field: _Field, metadata: Mapping[str, object], where: str) -> None:
    at = f"{where}.{field.name}"
    value = metadata[field.name]
    if field.kind == _POSITIVE:
        _integer(value, at, minimum=1, maximum=_UINT32_MAX)
    elif field.kind == _COUNT:
        _integer(value, at, maximum=_UINT32_MAX)
    elif field.kind == _INDEX:
        _integer(value, at)
    elif field.kind == _BOOL:
        _boolean(value, at)
    elif field.kind == _TEXT:
        _string(value, at, maximum=1024)
    elif field.kind == _TRIPLE:
        _integer_array(value, at, length=3, minimum=1)
    else:
        _fail(f"{at} has no validation rule for kind {field.kind!r}")


def _apply_metadata_defaults(spec: _FamilySpec, metadata: dict[str, object]) -> None:
    """Fill fields whose artifact corpus predates them."""
    for field in spec.fields:
        if field.from_metadata and field.name not in metadata and field.default is not _NO_DEFAULT:
            metadata[field.name] = field.default


def _validate_metadata(family: str, variant: VariantRecord, where: str) -> None:
    spec = _FAMILY_SPECS[family]
    metadata = variant.runtime_metadata
    _apply_metadata_defaults(spec, metadata)
    _exact_keys(metadata, _COMMON_METADATA | spec.metadata_keys, where)
    _integer(metadata["dynamic_smem_bytes"], f"{where}.dynamic_smem_bytes", maximum=_UINT32_MAX)
    non_portable = _boolean(metadata["non_portable_cluster_size_allowed"], f"{where}.non_portable_cluster_size_allowed")
    if non_portable and variant.kernel_sm != 90:
        _fail(f"{where}.non_portable_cluster_size_allowed requires an SM90 kernel ABI")

    # Every family states its dtype twice: once as the variant's own dtype and
    # once inside the launch metadata the kernel was compiled with.
    if spec.dtype_key == "dtype_name":
        if _string(metadata["dtype_name"], f"{where}.dtype_name") != variant.dtype:
            _fail(f"{where}.dtype_name disagrees with variant dtype")
    elif _boolean(metadata["is_bfloat16"], f"{where}.is_bfloat16") != (variant.dtype == "bf16"):
        _fail(f"{where}.is_bfloat16 disagrees with variant dtype")

    for field in spec.fields:
        if field.from_metadata:
            _validate_field(field, metadata, where)
    if spec.alias is not None:
        _validate_runtime_aliases(metadata["runtime_aliases"], f"{where}.runtime_aliases", spec.alias.keys)
    if spec.sm90 is not None:
        _validate_sm90_launch(family, metadata["sm90_launch"], f"{where}.sm90_launch", variant.dtype, variant.kernel_sm)
    if "num_threads" in metadata and cast(int, metadata["num_threads"]) > 1024:
        _fail(f"{where}.num_threads exceeds 1024")


def _runtime_keys(family: str, variant: VariantRecord) -> tuple[tuple[object, ...], ...]:
    """Return every (sm, axes...) tuple this variant claims at run time.

    A key naming an alias axis expands to one key per alias, which is how the
    aliasing families cover the anchors a single image serves.
    """
    spec = _FAMILY_SPECS[family]
    metadata = variant.runtime_metadata
    aliases = cast(
        "list[dict[str, object]]",
        metadata["runtime_aliases"] if spec.alias is not None else [{}],
    )

    def value(token: str, alias: Mapping[str, object]) -> object:
        raw = alias[token[1:]] if token.startswith("@") else metadata[token]
        return tuple(cast("list[object]", raw)) if isinstance(raw, list) else raw

    return tuple(
        (sm, *(value(token, alias) for token in spec.runtime_key)) for sm in variant.supported_sms for alias in aliases
    )


def _parse_variant(family: str, registry_version: int, value: object, where: str) -> VariantRecord:
    raw = _as_object(value, where)
    _exact_keys(
        raw,
        {
            "variant_id",
            "target_sm",
            "target_arch",
            "kernel_sm",
            "launch_abi",
            "supported_sms",
            "dtype",
            "label",
            "identity_spec",
            "aliases",
            "runtime_metadata",
            "kernel_symbol",
            "image",
        },
        where,
    )
    variant_id = _string(raw["variant_id"], f"{where}.variant_id", maximum=20)
    if _HEX20_RE.fullmatch(variant_id) is None:
        _fail(f"{where}.variant_id must be 20 lowercase hexadecimal characters")
    target_sm = _integer(raw["target_sm"], f"{where}.target_sm", minimum=10, maximum=999)
    target_arch = _string(raw["target_arch"], f"{where}.target_arch", maximum=16)
    arch_match = _ARCH_RE.fullmatch(target_arch)
    if arch_match is None or int(arch_match.group("sm")) != target_sm:
        _fail(f"{where}.target_arch does not encode target_sm {target_sm}")
    kernel_sm = _integer(raw["kernel_sm"], f"{where}.kernel_sm", minimum=10, maximum=999)
    launch_abi = _string(raw["launch_abi"], f"{where}.launch_abi", maximum=128)
    if _ABI_RE.fullmatch(launch_abi) is None or not launch_abi.startswith(family):
        _fail(f"{where}.launch_abi is not a canonical {family} ABI")
    supported_sms = _integer_array(raw["supported_sms"], f"{where}.supported_sms", minimum=10, maximum=999)
    if not supported_sms or tuple(sorted(set(supported_sms))) != supported_sms or target_sm not in supported_sms:
        _fail(f"{where}.supported_sms must be sorted, unique, non-empty, and include target_sm")
    dtype = _string(raw["dtype"], f"{where}.dtype", maximum=8)
    if dtype not in _DTYPES[family]:
        _fail(f"{where}.dtype {dtype!r} is unsupported for {family}")
    label = _string(raw["label"], f"{where}.label", maximum=256)
    expected_label = f"{family}.{variant_id}"
    if label != expected_label:
        _fail(f"{where}.label must equal canonical public label {expected_label!r}")
    identity_spec = _as_object(raw["identity_spec"], f"{where}.identity_spec")
    aliases = tuple(
        _as_object(alias, f"{where}.aliases[{index}]")
        for index, alias in enumerate(_as_list(raw["aliases"], f"{where}.aliases"))
    )
    _validate_public_aliases(family, aliases, f"{where}.aliases")
    runtime_metadata = _as_object(raw["runtime_metadata"], f"{where}.runtime_metadata")
    kernel_symbol = _string(raw["kernel_symbol"], f"{where}.kernel_symbol", maximum=512)
    if _SYMBOL_RE.fullmatch(kernel_symbol) is None:
        _fail(f"{where}.kernel_symbol is not a neutralized symbol")
    image_raw = _as_object(raw["image"], f"{where}.image")
    _exact_keys(image_raw, {"sha256", "size", "pack"}, f"{where}.image")
    image_hash = _string(image_raw["sha256"], f"{where}.image.sha256", maximum=64)
    if _HEX64_RE.fullmatch(image_hash) is None:
        _fail(f"{where}.image.sha256 must be a full lowercase SHA-256")
    image = ImageRecord(
        sha256=image_hash,
        size=_integer(image_raw["size"], f"{where}.image.size", minimum=4, maximum=_MAX_IMAGE_BYTES),
        pack=_string(image_raw["pack"], f"{where}.image.pack", maximum=16),
    )
    if image.pack != target_arch:
        _fail(f"{where}.image.pack must equal target_arch")
    canonical = {
        "registry_version": registry_version,
        "family": family,
        "target_sm": target_sm,
        "dtype": dtype,
        "spec": identity_spec,
    }
    expected_id = hashlib.sha256(_json_bytes(canonical)).hexdigest()[:20]
    if variant_id != expected_id:
        _fail(f"{where}.variant_id does not match identity_spec (expected {expected_id})")
    variant = VariantRecord(
        variant_id=variant_id,
        target_sm=target_sm,
        target_arch=target_arch,
        kernel_sm=kernel_sm,
        launch_abi=launch_abi,
        supported_sms=supported_sms,
        dtype=dtype,
        label=label,
        identity_spec=identity_spec,
        aliases=aliases,
        runtime_metadata=runtime_metadata,
        kernel_symbol=kernel_symbol,
        image=image,
    )
    _validate_metadata(family, variant, f"{where}.runtime_metadata")
    return variant


def _parse_family(expected_family: str, index_path: Path) -> FamilyIndex:
    if expected_family not in _FAMILIES or _FAMILY_RE.fullmatch(expected_family) is None:
        _fail(f"unsupported CUBIN family {expected_family!r}")
    if index_path.is_symlink():
        _fail(f"CUBIN index must not be a symlink: {index_path}")
    path = index_path.resolve()
    root, raw_bytes = _read_index(path)
    _exact_keys(
        root,
        {"format", "family", "registry_version", "compile_fingerprint", "toolchain", "packs", "variants"},
        str(path),
    )
    if root["format"] != INDEX_FORMAT:
        _fail(f"{path}: unsupported format {root['format']!r}; expected {INDEX_FORMAT!r}")
    family = _string(root["family"], f"{path}.family", maximum=128)
    if family != expected_family:
        _fail(f"{path}: family is {family!r}, expected {expected_family!r}")
    registry_version = _integer(root["registry_version"], f"{path}.registry_version", minimum=0)
    if registry_version != REGISTRY_VERSION:
        _fail(f"{path}: unsupported registry_version {registry_version}; expected {REGISTRY_VERSION}")
    compile_fingerprint = _string(root["compile_fingerprint"], f"{path}.compile_fingerprint", maximum=64)
    if _HEX64_RE.fullmatch(compile_fingerprint) is None:
        _fail(f"{path}.compile_fingerprint must be a full lowercase SHA-256")
    toolchain = _as_object(root["toolchain"], f"{path}.toolchain")
    _exact_keys(toolchain, {"nvidia-cutlass-dsl", "compile_environment"}, f"{path}.toolchain")
    _string(toolchain["nvidia-cutlass-dsl"], f"{path}.toolchain.nvidia-cutlass-dsl", maximum=64)
    environment = _string(toolchain["compile_environment"], f"{path}.toolchain.compile_environment", maximum=71)
    if not environment.startswith("sha256:") or _HEX64_RE.fullmatch(environment[7:]) is None:
        _fail(f"{path}.toolchain.compile_environment must be sha256:<full digest>")

    packs_raw = _as_object(root["packs"], f"{path}.packs")
    if not packs_raw or list(packs_raw) != sorted(packs_raw):
        _fail(f"{path}.packs must be non-empty and sorted by target architecture")
    packs: list[PackRecord] = []
    for target_arch, pack_value in packs_raw.items():
        pack_where = f"{path}.packs.{target_arch}"
        arch_match = _ARCH_RE.fullmatch(target_arch)
        if arch_match is None:
            _fail(f"{pack_where} has an invalid target architecture key")
        pack = _as_object(pack_value, pack_where)
        _exact_keys(pack, {"file", "sha256", "size", "image_count"}, pack_where)
        digest = _string(pack["sha256"], f"{pack_where}.sha256", maximum=64)
        if _HEX64_RE.fullmatch(digest) is None:
            _fail(f"{pack_where}.sha256 must be a full lowercase SHA-256")
        expected_file = f"packs/{family}_{target_arch.replace('_', '')}_{digest}.tar.xz"
        file = _string(pack["file"], f"{pack_where}.file", maximum=512)
        if file != expected_file or PurePosixPath(file).parts != ("packs", expected_file.removeprefix("packs/")):
            _fail(f"{pack_where}.file must be {expected_file!r}")
        packs.append(
            PackRecord(
                target_arch=target_arch,
                file=file,
                sha256=digest,
                size=_integer(pack["size"], f"{pack_where}.size", minimum=1, maximum=_MAX_PACK_BYTES),
                image_count=_integer(
                    pack["image_count"], f"{pack_where}.image_count", minimum=1, maximum=_MAX_FAMILY_IMAGES
                ),
            )
        )

    variants_raw = _as_list(root["variants"], f"{path}.variants")
    if not variants_raw or len(variants_raw) > _MAX_FAMILY_IMAGES:
        _fail(f"{path}.variants must contain between 1 and {_MAX_FAMILY_IMAGES} entries")
    variants = tuple(
        _parse_variant(family, registry_version, value, f"{path}.variants[{index}]")
        for index, value in enumerate(variants_raw)
    )
    variant_ids = tuple(variant.variant_id for variant in variants)
    if tuple(sorted(set(variant_ids))) != variant_ids:
        _fail(f"{path}.variants must be sorted by unique variant_id")

    pack_by_arch = {pack.target_arch: pack for pack in packs}
    referenced: dict[str, set[str]] = {target_arch: set() for target_arch in pack_by_arch}
    runtime_keys: set[tuple[object, ...]] = set()
    for variant in variants:
        if variant.target_arch not in pack_by_arch:
            _fail(f"{path}: variant {variant.variant_id} references undeclared pack {variant.target_arch}")
        referenced[variant.target_arch].add(variant.image.sha256)
        for key in _runtime_keys(family, variant):
            if key in runtime_keys:
                _fail(f"{path}: multiple variants resolve to runtime key {key}")
            runtime_keys.add(key)
    for pack in packs:
        if pack.image_count != len(referenced[pack.target_arch]):
            _fail(
                f"{path}: pack {pack.target_arch} records {pack.image_count} images but variants reference "
                f"{len(referenced[pack.target_arch])}"
            )
    return FamilyIndex(family, path, raw_bytes, compile_fingerprint, tuple(packs), variants)


def _load_families(family_indexes: Sequence[tuple[str, Path]]) -> tuple[FamilyIndex, ...]:
    if not family_indexes:
        _fail("at least one family index is required")
    normalized: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for family, path in family_indexes:
        if not isinstance(family, str) or not isinstance(path, Path):
            raise TypeError("family_indexes entries must be (str, pathlib.Path)")
        if family in seen:
            _fail(f"duplicate family index argument {family!r}")
        seen.add(family)
        normalized.append((family, path))
    return tuple(_parse_family(family, path) for family, path in sorted(normalized))


def _global_images(families: Sequence[FamilyIndex]) -> dict[str, int]:
    images: dict[str, int] = {}
    for family in families:
        for variant in family.variants:
            previous = images.setdefault(variant.image.sha256, variant.image.size)
            if previous != variant.image.size:
                _fail(
                    f"image {variant.image.sha256} has inconsistent indexed sizes {previous} and {variant.image.size}"
                )
    return images


def _expected_pack_images(family: FamilyIndex, target_arch: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for variant in family.variants:
        if variant.image.pack != target_arch:
            continue
        previous = result.setdefault(variant.image.sha256, variant.image.size)
        if previous != variant.image.size:
            _fail(f"{family.path}: image {variant.image.sha256} has inconsistent sizes within one pack")
    return result


def _pack_path(family: FamilyIndex, pack: PackRecord) -> Path:
    packs_dir = family.path.parent / "packs"
    if packs_dir.is_symlink():
        _fail(f"pack directory must not be a symlink: {packs_dir}")
    path = family.path.parent / PurePosixPath(pack.file)
    if path.parent != packs_dir or path.name != PurePosixPath(pack.file).name:
        _fail(f"pack path is not a canonical direct child of packs/: {pack.file}")
    return path


def _verify_exact_pack_set(family: FamilyIndex) -> None:
    """Require the family pack directory to contain only indexed packs."""
    packs_dir = family.path.parent / "packs"
    if packs_dir.is_symlink() or not packs_dir.is_dir():
        _fail(f"pack directory must be a regular directory: {packs_dir}")
    expected = {PurePosixPath(pack.file).name for pack in family.packs}
    try:
        entries = list(packs_dir.iterdir())
    except OSError as error:
        raise MaterializationError(f"cannot inspect CUBIN pack directory {packs_dir}: {error}") from error
    actual = {entry.name for entry in entries}
    if actual != expected:
        _fail(
            f"{packs_dir} does not exactly match its family index: "
            f"missing={sorted(expected - actual)}, unreferenced={sorted(actual - expected)}"
        )
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            _fail(f"indexed CUBIN pack must be a regular file: {entry}")


def _verify_exact_pack_sets(families: Sequence[FamilyIndex]) -> None:
    for family in families:
        _verify_exact_pack_set(family)


def _xz_vli(data: bytes, offset: int, limit: int, where: str) -> tuple[int, int]:
    """Read one canonical XZ variable-length integer."""
    value = 0
    for byte_index in range(9):
        if offset >= limit:
            _fail(f"{where} contains a truncated XZ variable-length integer")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << (7 * byte_index)
        if byte & 0x80 == 0:
            if byte_index and byte == 0:
                _fail(f"{where} contains a non-canonical XZ variable-length integer")
            return value, offset
    _fail(f"{where} contains an oversized XZ variable-length integer")


def _verify_xz_profile(path: Path) -> None:
    """Validate the decoder-visible, deterministic v1 XZ/LZMA2 profile."""
    try:
        with path.open("rb") as stream:
            stream_header = stream.read(12)
            if len(stream_header) != 12 or stream_header[:6] != _XZ_MAGIC:
                _fail(f"{path} has a missing or invalid XZ stream header")
            if stream_header[6:8] != _XZ_STREAM_FLAGS:
                _fail(f"{path} does not use the required single-stream XZ CRC64 profile")
            if int.from_bytes(stream_header[8:12], "little") != zlib.crc32(stream_header[6:8]):
                _fail(f"{path} has an invalid XZ stream-header CRC")

            encoded_header_size = stream.read(1)
            if not encoded_header_size or encoded_header_size == b"\0":
                _fail(f"{path} has no XZ data block")
            block_header_size = (encoded_header_size[0] + 1) * 4
            block_header = encoded_header_size + stream.read(block_header_size - 1)
            if len(block_header) != block_header_size:
                _fail(f"{path} has a truncated XZ block header")
            if int.from_bytes(block_header[-4:], "little") != zlib.crc32(block_header[:-4]):
                _fail(f"{path} has an invalid XZ block-header CRC")

            block_flags = block_header[1]
            if block_flags & 0x3C or block_flags & 0x03:
                _fail(f"{path} must use exactly one XZ block filter with no reserved flags")
            cursor = 2
            body_end = len(block_header) - 4
            if block_flags & 0x40:
                _, cursor = _xz_vli(block_header, cursor, body_end, str(path))
            if block_flags & 0x80:
                _, cursor = _xz_vli(block_header, cursor, body_end, str(path))
            filter_id, cursor = _xz_vli(block_header, cursor, body_end, str(path))
            properties_size, cursor = _xz_vli(block_header, cursor, body_end, str(path))
            properties_end = cursor + properties_size
            if properties_end > body_end:
                _fail(f"{path} has truncated XZ filter properties")
            properties = block_header[cursor:properties_end]
            if filter_id != lzma.FILTER_LZMA2 or properties != _XZ_LZMA2_DICT_PROPERTY_8_MIB:
                _fail(f"{path} does not use the required LZMA2 8 MiB dictionary profile")
            if any(block_header[properties_end:body_end]):
                _fail(f"{path} has nonzero XZ block-header padding")

            stream.seek(-12, os.SEEK_END)
            footer = stream.read(12)
            if len(footer) != 12 or footer[10:] != _XZ_FOOTER_MAGIC:
                _fail(f"{path} has a missing or invalid XZ stream footer")
            if footer[8:10] != _XZ_STREAM_FLAGS:
                _fail(f"{path} has inconsistent XZ stream flags")
            if int.from_bytes(footer[:4], "little") != zlib.crc32(footer[4:10]):
                _fail(f"{path} has an invalid XZ stream-footer CRC")
            index_size = (int.from_bytes(footer[4:8], "little") + 1) * 4
            index_offset = stream.tell() - 12 - index_size
            if index_offset < 12 + block_header_size:
                _fail(f"{path} has an invalid XZ index location")
            stream.seek(index_offset)
            index = stream.read(index_size)
            if len(index) != index_size or index[0:1] != b"\0":
                _fail(f"{path} has a missing or invalid XZ index")
            if int.from_bytes(index[-4:], "little") != zlib.crc32(index[:-4]):
                _fail(f"{path} has an invalid XZ index CRC")
            index_end = len(index) - 4
            record_count, cursor = _xz_vli(index, 1, index_end, str(path))
            if record_count != 1:
                _fail(f"{path} must contain exactly one XZ block")
            unpadded_size, cursor = _xz_vli(index, cursor, index_end, str(path))
            uncompressed_size, cursor = _xz_vli(index, cursor, index_end, str(path))
            if unpadded_size == 0 or uncompressed_size == 0 or any(index[cursor:index_end]):
                _fail(f"{path} has an invalid or non-canonical XZ index record")
    except MaterializationError:
        raise
    except OSError as error:
        raise MaterializationError(f"cannot validate XZ profile in {path}: {error}") from error


def _verify_compressed_pack(path: Path, pack: PackRecord, maximum_tar_bytes: int) -> None:
    if path.is_symlink():
        _fail(f"CUBIN pack must not be a symlink: {path}")
    try:
        file_stat = path.stat()
    except OSError as error:
        raise MaterializationError(f"cannot stat CUBIN pack {path}: {error}") from error
    if not path.is_file():
        _fail(f"CUBIN pack is not a regular file: {path}")
    try:
        with path.open("rb") as stream:
            prefix = stream.read(len(_LFS_PREFIX))
            if prefix == _LFS_PREFIX:
                _fail(f"{path} is a Git LFS pointer; run `git lfs pull --include={path}` and retry")
            if file_stat.st_size != pack.size:
                _fail(f"{path} compressed size is {file_stat.st_size}, expected {pack.size}")
            stream.seek(0)
            digest = hashlib.sha256()
            compressed_size = 0
            decompressed_size = 0
            decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
            while chunk := stream.read(_IO_CHUNK):
                if decompressor.eof:
                    _fail(f"{path} contains trailing data or more than one XZ stream")
                digest.update(chunk)
                compressed_size += len(chunk)
                pending = chunk
                while True:
                    output = decompressor.decompress(pending, max_length=_IO_CHUNK)
                    pending = b""
                    decompressed_size += len(output)
                    if decompressed_size > min(maximum_tar_bytes, _MAX_TAR_BYTES):
                        _fail(f"{path} expands beyond its permitted tar size")
                    if decompressor.eof:
                        if decompressor.unused_data:
                            _fail(f"{path} contains trailing data or more than one XZ stream")
                        break
                    if decompressor.needs_input:
                        break
            if not decompressor.eof:
                _fail(f"{path} contains a truncated XZ stream")
            if decompressor.check != lzma.CHECK_CRC64:
                _fail(f"{path} does not use the required XZ CRC64 integrity check")
    except MaterializationError:
        raise
    except (OSError, EOFError, lzma.LZMAError) as error:
        raise MaterializationError(f"cannot verify XZ pack {path}: {error}") from error
    _verify_xz_profile(path)
    if compressed_size != pack.size:
        _fail(f"{path} compressed size is {compressed_size}, expected {pack.size}")
    if digest.hexdigest() != pack.sha256:
        _fail(f"{path} SHA-256 is {digest.hexdigest()}, expected {pack.sha256}")


def _valid_materialized_object(path: Path, expected_hash: str, expected_size: int) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        if path.stat().st_size != expected_size:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(_IO_CHUNK):
                digest.update(chunk)
        return digest.hexdigest() == expected_hash
    except OSError:
        return False


def _read_exact(stream: lzma.LZMAFile, size: int, where: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(min(_IO_CHUNK, remaining))
        if not chunk:
            _fail(f"{where} is truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _discard_exact(stream: lzma.LZMAFile, size: int, where: str) -> None:
    remaining = size
    while remaining:
        chunk = stream.read(min(_IO_CHUNK, remaining))
        if not chunk:
            _fail(f"{where} is truncated")
        remaining -= len(chunk)


def _verify_ustar_layout(path: Path, expected_names: Sequence[str], expected: Mapping[str, int]) -> None:
    """Reject GNU/PAX headers and any nonzero bytes after the USTAR terminator."""
    try:
        with lzma.open(path, "rb", format=lzma.FORMAT_XZ) as stream:
            for member_index, expected_name in enumerate(expected_names):
                where = f"{path}:header[{member_index}]"
                header = _read_exact(stream, tarfile.BLOCKSIZE, where)
                encoded_name = expected_name.encode()
                if len(encoded_name) > 100 or header[:100] != encoded_name.ljust(100, b"\0"):
                    _fail(f"{where} does not contain the expected canonical USTAR name {expected_name!r}")
                if header[156:157] != tarfile.REGTYPE:
                    _fail(f"{where} is not an explicit regular-file USTAR header")
                if header[257:263] != b"ustar\0" or header[263:265] != b"00" or any(header[345:500]):
                    _fail(f"{where} is not canonical POSIX USTAR")
                image_hash = PurePosixPath(expected_name).name.removesuffix(".cubin")
                size = expected[image_hash]
                padded_size = ((size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE
                _discard_exact(stream, size, f"{path}:{expected_name}")
                padding = _read_exact(stream, padded_size - size, f"{path}:{expected_name}:padding")
                if any(padding):
                    _fail(f"{path}:{expected_name} has nonzero USTAR member padding")
            trailing_size = 0
            trailing_nonzero = False
            while chunk := stream.read(_IO_CHUNK):
                trailing_size += len(chunk)
                trailing_nonzero = trailing_nonzero or any(chunk)
    except MaterializationError:
        raise
    except (OSError, EOFError, lzma.LZMAError) as error:
        raise MaterializationError(f"cannot validate raw USTAR layout in {path}: {error}") from error
    if trailing_size < 2 * tarfile.BLOCKSIZE or trailing_size % tarfile.BLOCKSIZE != 0 or trailing_nonzero:
        _fail(f"{path} has a missing USTAR terminator or nonzero/truncated bytes after its final member")


def _verify_pack_members(
    path: Path,
    expected: Mapping[str, int],
    object_dir: Path | None,
    installed: set[str],
) -> None:
    expected_names = [f"objects/{digest}.cubin" for digest in sorted(expected)]
    _verify_ustar_layout(path, expected_names, expected)
    observed: list[str] = []
    try:
        with tarfile.open(path, mode="r|xz") as archive:
            for member_index, member in enumerate(archive):
                if member_index >= _MAX_FAMILY_IMAGES:
                    _fail(f"{path} contains too many archive members")
                where = f"{path}:{member.name}"
                if member.type != tarfile.REGTYPE or not member.isfile():
                    _fail(f"{where} is not a canonical regular USTAR member")
                if member.pax_headers or member.sparse is not None:
                    _fail(f"{where} contains PAX or sparse metadata")
                if (
                    member.mode != 0o444
                    or member.uid != 0
                    or member.gid != 0
                    or member.mtime != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.linkname != ""
                ):
                    _fail(f"{where} has non-canonical ownership, timestamp, mode, or link metadata")
                pure_name = PurePosixPath(member.name)
                if len(pure_name.parts) != 2 or pure_name.parts[0] != "objects":
                    _fail(f"{where} is not a canonical objects/<sha256>.cubin path")
                filename = pure_name.parts[1]
                if not filename.endswith(".cubin") or _HEX64_RE.fullmatch(filename[:-6]) is None:
                    _fail(f"{where} is not content-addressed by a full SHA-256")
                image_hash = filename[:-6]
                if image_hash not in expected:
                    _fail(f"{where} is not referenced by the family index")
                if member.size != expected[image_hash]:
                    _fail(f"{where} size is {member.size}, expected {expected[image_hash]}")
                observed.append(member.name)
                extracted = archive.extractfile(member)
                if extracted is None:
                    _fail(f"cannot stream {where}")
                destination = object_dir / filename if object_dir is not None else None
                temporary: Path | None = None
                output = None
                if destination is not None and image_hash not in installed:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", dir=destination.parent)
                    temporary = Path(temporary_name)
                    output = os.fdopen(descriptor, "wb")
                digest = hashlib.sha256()
                prefix = b""
                size = 0
                try:
                    while chunk := extracted.read(_IO_CHUNK):
                        if len(prefix) < 4:
                            prefix = (prefix + chunk)[:4]
                        digest.update(chunk)
                        size += len(chunk)
                        if output is not None:
                            output.write(chunk)
                    if output is not None:
                        output.flush()
                        os.fsync(output.fileno())
                finally:
                    extracted.close()
                    if output is not None:
                        output.close()
                if prefix != b"\x7fELF":
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                    _fail(f"{where} is not an ELF CUBIN")
                if size != expected[image_hash] or digest.hexdigest() != image_hash:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                    _fail(f"{where} content does not match its indexed size and SHA-256")
                if destination is not None and temporary is not None:
                    if not _valid_materialized_object(destination, image_hash, size):
                        temporary.chmod(0o444)
                        os.replace(temporary, destination)
                    else:
                        temporary.unlink(missing_ok=True)
                    installed.add(image_hash)
    except MaterializationError:
        raise
    except (OSError, EOFError, lzma.LZMAError, tarfile.TarError) as error:
        raise MaterializationError(f"cannot verify tar pack {path}: {error}") from error
    if observed != expected_names:
        _fail(
            f"{path} members are missing, duplicated, extra, or not sorted: expected={expected_names}, got={observed}"
        )


def _verify_archives(families: Sequence[FamilyIndex], object_dir: Path | None = None) -> VerificationResult:
    global_images = _global_images(families)
    installed: set[str] = set()
    pack_count = 0
    for family in families:
        for pack in family.packs:
            expected = _expected_pack_images(family, pack.target_arch)
            tar_limit = min(
                _MAX_TAR_BYTES,
                sum(512 + ((size + 511) // 512) * 512 for size in expected.values()) + 1024 * 1024,
            )
            path = _pack_path(family, pack)
            _verify_compressed_pack(path, pack, tar_limit)
            _verify_pack_members(path, expected, object_dir, installed)
            pack_count += 1
    return VerificationResult(
        family_count=len(families),
        pack_count=pack_count,
        variant_count=sum(len(family.variants) for family in families),
        image_count=len(global_images),
        total_image_bytes=sum(global_images.values()),
    )


def verify_packs(family_indexes: Sequence[tuple[str, Path]]) -> VerificationResult:
    """Validate public family indexes, compressed packs, and every packed image.

    Args:
        family_indexes: Explicit ``(expected_family, index_path)`` pairs.

    Returns:
        Aggregate verification counts.

    Raises:
        MaterializationError: If any index, pack, or image is invalid.
    """
    families = _load_families(family_indexes)
    _verify_exact_pack_sets(families)
    return _verify_archives(families)


def prune_unreferenced_packs(family_indexes: Sequence[tuple[str, Path]]) -> tuple[Path, ...]:
    """Delete pack files no family index names, and report which went.

    A pack filename embeds the hash of its own contents, so rebuilding a family
    writes NEW files BESIDE the ones it supersedes rather than over them. Nothing
    then removes the old generation, and two of them in one directory is exactly
    what ``_verify_exact_pack_set`` refuses.

    The index is the authority and the filename is a content hash, so a pack no
    index references is superseded by construction. Removing those is what keeps
    the exact-set check meaningful; relaxing it to a subset test would instead let
    a stale pack ship inside a wheel, which is the hazard the check exists for.

    Args:
        family_indexes: Explicit ``(expected_family, index_path)`` pairs.

    Returns:
        The pruned paths, in deletion order. Empty when nothing was superseded,
        which is every call that follows no rebuild.

    Raises:
        MaterializationError: If an index is invalid, a pack directory is not a
            regular directory, or an unreferenced entry is not a regular file.
    """
    families = _load_families(family_indexes)
    pruned: list[Path] = []
    for family in families:
        packs_dir = family.path.parent / "packs"
        if packs_dir.is_symlink() or not packs_dir.is_dir():
            _fail(f"pack directory must be a regular directory: {packs_dir}")
        expected = {PurePosixPath(pack.file).name for pack in family.packs}
        try:
            entries = sorted(packs_dir.iterdir())
        except OSError as error:
            raise MaterializationError(f"cannot inspect CUBIN pack directory {packs_dir}: {error}") from error
        for entry in entries:
            if entry.name in expected:
                continue
            # Never a directory or a link: this deletes files, and an unexpected
            # kind of entry is a corpus to look at rather than one to clean up.
            if entry.is_symlink() or not entry.is_file():
                _fail(f"refusing to prune a non-regular pack entry: {entry}")
            entry.unlink()
            pruned.append(entry)
    return tuple(pruned)


def _fingerprint(families: Sequence[FamilyIndex]) -> str:
    digest = hashlib.sha256()

    def add(label: str, value: bytes) -> None:
        encoded_label = label.encode()
        digest.update(len(encoded_label).to_bytes(8, "big"))
        digest.update(encoded_label)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    add("fingerprint-format", FINGERPRINT_FORMAT.encode())
    add("assembly-format", ASSEMBLY_FORMAT.encode())
    add("host-object-format", b"elf-gnu")
    try:
        add("materializer", Path(__file__).resolve().read_bytes())
    except OSError as error:
        raise MaterializationError(f"cannot fingerprint public materializer {__file__}: {error}") from error
    for family in families:
        add(f"index:{family.family}", family.raw_bytes)
        add(
            f"packs:{family.family}",
            _json_bytes(
                [{"target_arch": pack.target_arch, "sha256": pack.sha256, "size": pack.size} for pack in family.packs]
            ),
        )
    return digest.hexdigest()


def _cpp_string(value: object) -> str:
    return json.dumps(cast(str, value), ensure_ascii=True)


def _cpp_bool(value: object) -> str:
    return "true" if cast(bool, value) else "false"


def _cpp_uint_array(value: object, size: int, fill: int = 0) -> str:
    values = list(cast(list[int], value))
    values.extend([fill] * (size - len(values)))
    return "{" + ", ".join(f"{item}U" for item in values) + "}"


def _render_tma_descriptor(value: object, indent: str) -> list[str]:
    metadata = cast(dict[str, object], value)
    rank = cast(int, metadata["rank"])
    return [
        f"{indent}{{",
        f"{indent}  {_TMA_DATA_TYPE[cast(str, metadata['data_type'])]},",
        f"{indent}  {rank}U,",
        f"{indent}  {_cpp_uint_array(metadata['global_dim_order'], 5)},",
        f"{indent}  {_cpp_uint_array(metadata['box_dims'], 5, 1)},",
        f"{indent}  {_cpp_uint_array(metadata['element_strides'], 5, 1)},",
        f"{indent}  {_TMA_INTERLEAVE[cast(str, metadata['interleave'])]},",
        f"{indent}  {_TMA_SWIZZLE[cast(str, metadata['swizzle'])]},",
        f"{indent}  {_TMA_L2_PROMOTION[cast(str, metadata['l2_promotion'])]},",
        f"{indent}  {_TMA_OOB_FILL[cast(str, metadata['oob_fill'])]},",
        f"{indent}}},",
    ]


def _render_sm90(value: object, operands: tuple[str, ...], indent: str) -> list[str]:
    if value is None:
        return [f"{indent}{{}},"]
    metadata = cast(dict[str, object], value)
    descriptors = cast(dict[str, object], metadata["tma_descriptors"])
    lines = [
        f"{indent}{{",
        f"{indent}  true,",
        f"{indent}  {_cpp_uint_array(metadata['block_dims'], 3)},",
        f"{indent}  {_cpp_uint_array(metadata['cluster_dims'], 3)},",
        f"{indent}  {_CLUSTER_POLICY[cast(str, metadata['cluster_scheduling_policy'])]},",
    ]
    for operand in operands:
        lines.extend(_render_tma_descriptor(descriptors[operand], indent + "  "))
    lines.append(f"{indent}}},")
    return lines


def _sm90_declarations(spec: _FamilySpec) -> list[str]:
    assert spec.sm90 is not None
    return [
        "struct SM90LaunchInfo",
        "{",
        f"  bool {spec.sm90.enabled_field};",
        "  std::uint32_t block_dims[3];",
        "  std::uint32_t cluster_dims[3];",
        "  CUclusterSchedulingPolicy cluster_scheduling_policy;",
        *(f"  TmaDescriptorInfo {operand};" for operand in spec.sm90.operands),
        "};",
        "",
    ]


def _family_type_declarations(family: str) -> list[str]:
    """Declare this family's registry record, in field order."""
    spec = _FAMILY_SPECS[family]
    lines: list[str] = []
    if spec.alias is not None:
        lines.extend(["struct RuntimeAlias", "{", *(f"  {item}" for item in spec.alias.declarations), "};", ""])
    if spec.sm90 is not None:
        lines.extend(_sm90_declarations(spec))
    lines.extend(
        [
            "struct CubinImage",
            "{",
            "  EmbeddedCubinImage cubin;",
            *(f"  {field.declaration}" for field in spec.fields),
        ]
    )
    if spec.sm90 is not None:
        lines.append("  SM90LaunchInfo sm90;")
    if spec.alias is not None:
        lines.extend(["  RuntimeAlias const* aliases;", "  std::size_t alias_count;"])
    lines.append("};")
    return lines


def _registry_header(family: str) -> str:
    guard = f"BIOIR_GENERATED_{family.upper()}_REGISTRY_H_"
    lines = [
        "/* SPDX-License-Identifier: Apache-2.0",
        " * Generated by cpp/cmake/materialize_cubin_payloads.py. Do not edit.",
        " */",
        "",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        '#include "cubin_runtime.h"',
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        f"namespace bioir::cutedsl::{family}::embedded",
        "{",
        "",
        *_family_type_declarations(family),
        "",
        "struct RegistryView",
        "{",
        "  CubinImage const* images;",
        "  std::size_t count;",
        "};",
        "",
        "RegistryView registry() noexcept;",
        "",
        f"}} // namespace bioir::cutedsl::{family}::embedded",
        "",
        f"#endif /* {guard} */",
        "",
    ]
    return "\n".join(lines)


def _alias_table_name(variant: VariantRecord) -> str:
    return f"kRuntimeAliases_{variant.variant_id}"


def _alias_lines(family: str, variant: VariantRecord) -> list[str]:
    spec = _FAMILY_SPECS[family]
    if spec.alias is None:
        return []
    aliases = cast("list[dict[str, object]]", variant.runtime_metadata["runtime_aliases"])
    first, second = spec.alias.keys
    name = _alias_table_name(variant)
    return [
        f"constexpr RuntimeAlias {name}[] = {{",
        *(f"  {{{alias[first]}, {alias[second]}}}," for alias in aliases),
        "};",
        "",
    ]


def _field_value(field: _Field, variant: VariantRecord) -> str:
    if field.kind == _DTYPE_CODE:
        return str(_DTYPE_CODES[variant.dtype])
    value = variant.runtime_metadata[field.name]
    if field.kind == _BOOL:
        return _cpp_bool(value)
    if field.kind == _TEXT:
        return _cpp_string(value)
    if field.kind == _TRIPLE:
        return "{" + ", ".join(str(item) for item in cast("list[object]", value)) + "}"
    return f"{value}{field.suffix}"


def _family_initializer_lines(family: str, variant: VariantRecord, indent: str) -> list[str]:
    """Initialize one ``CubinImage``, matching _family_type_declarations."""
    spec = _FAMILY_SPECS[family]
    lines = [f"{indent}{_field_value(field, variant)}," for field in spec.fields]
    if spec.sm90 is not None:
        lines.extend(_render_sm90(variant.runtime_metadata["sm90_launch"], spec.sm90.operands, indent))
    if spec.alias is not None:
        name = _alias_table_name(variant)
        lines.extend([f"{indent}{name},", f"{indent}sizeof({name}) / sizeof({name}[0]),"])
    return lines


def _registry_source(family: FamilyIndex) -> str:
    image_hashes = sorted({variant.image.sha256 for variant in family.variants})
    lines = [
        "/* SPDX-License-Identifier: Apache-2.0",
        " * Generated by cpp/cmake/materialize_cubin_payloads.py. Do not edit.",
        " */",
        "",
        f'#include "{family.family}_registry.h"',
        "",
        'extern "C"',
        "{",
        *(f"extern unsigned char bioir_cubin_{digest}_start[];" for digest in image_hashes),
        "}",
        "",
        f"namespace bioir::cutedsl::{family.family}::embedded",
        "{",
        "namespace",
        "{",
        "",
    ]
    for variant in family.variants:
        supported = ", ".join(str(sm) for sm in variant.supported_sms)
        lines.append(f"constexpr std::int32_t kSupportedSms_{variant.variant_id}[] = {{{supported}}};")
    lines.append("")
    for variant in family.variants:
        lines.extend(_alias_lines(family.family, variant))
    lines.extend(["CubinImage const kImages[] = {"])
    for variant in family.variants:
        supported_name = f"kSupportedSms_{variant.variant_id}"
        metadata = variant.runtime_metadata
        lines.extend(
            [
                "  {",
                "    {",
                f"      {variant.target_sm},",
                f"      {variant.kernel_sm},",
                f"      {metadata['dynamic_smem_bytes']}U,",
                f"      {_cpp_bool(metadata['non_portable_cluster_size_allowed'])},",
                f"      {_cpp_string(variant.launch_abi)},",
                f"      {supported_name},",
                f"      sizeof({supported_name}) / sizeof({supported_name}[0]),",
                f"      {_cpp_string(variant.variant_id)},",
                f"      {_cpp_string(variant.kernel_symbol)},",
                f"      bioir_cubin_{variant.image.sha256}_start,",
                f"      static_cast<std::size_t>({variant.image.size}),",
                "    },",
            ]
        )
        lines.extend(_family_initializer_lines(family.family, variant, "    "))
        lines.append("  },")
    lines.extend(
        [
            "};",
            "",
            "} // namespace",
            "",
            "RegistryView registry() noexcept",
            "{",
            "  return {kImages, sizeof(kImages) / sizeof(kImages[0])};",
            "}",
            "",
            f"}} // namespace bioir::cutedsl::{family.family}::embedded",
            "",
        ]
    )
    return "\n".join(lines)


def _gas_path(path: Path) -> str:
    value = str(path.resolve())
    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise MaterializationError(f"assembler input path must be ASCII: {value!r}") from error
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        _fail(f"assembler input path contains a control character: {value!r}")
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _assembly_shard(shard: str, hashes: Sequence[str], object_dir: Path) -> str:
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "# Generated by cpp/cmake/materialize_cubin_payloads.py. Do not edit.",
        "",
    ]
    for digest in hashes:
        symbol = f"bioir_cubin_{digest}"
        lines.extend(
            [
                '.section .rodata.bioir_cubins,"a",%progbits',
                ".balign 16",
                f".global {symbol}_start",
                f".hidden {symbol}_start",
                f".type {symbol}_start,%object",
                f".global {symbol}_end",
                f".hidden {symbol}_end",
                f"{symbol}_start:",
                f'.incbin "{_gas_path(object_dir / f"{digest}.cubin")}"',
                f"{symbol}_end:",
                f".size {symbol}_start,{symbol}_end-{symbol}_start",
                "",
            ]
        )
    lines.extend(['.section .note.GNU-stack,"",%progbits', ""])
    return "\n".join(lines)


def _atomic_write_if_changed(path: Path, content: bytes, mode: int = 0o644) -> None:
    try:
        if path.is_file() and not path.is_symlink() and path.read_bytes() == content:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _stamp_value(families: Sequence[FamilyIndex], fingerprint: str, images: Mapping[str, int]) -> dict[str, object]:
    files = [f"cubin_payloads_{digit}.S" for digit in "0123456789abcdef"]
    for family in families:
        files.extend([f"{family.family}_registry.h", f"{family.family}_registry.cpp"])
    files.extend(f"objects/{digest}.cubin" for digest in sorted(images))
    return {
        "format": MATERIALIZATION_FORMAT,
        "fingerprint": fingerprint,
        "families": [family.family for family in families],
        "files": sorted(files),
        "objects": [{"sha256": digest, "size": images[digest]} for digest in sorted(images)],
    }


def _has_exact_materialized_objects(object_dir: Path, images: Mapping[str, int]) -> bool:
    if object_dir.is_symlink() or not object_dir.is_dir():
        return False
    expected_names = {f"{digest}.cubin" for digest in images}
    try:
        entries = list(object_dir.iterdir())
    except OSError:
        return False
    if {entry.name for entry in entries} != expected_names:
        return False
    return all(
        _valid_materialized_object(object_dir / f"{digest}.cubin", digest, size) for digest, size in images.items()
    )


def _prepare_object_dir(output_dir: Path, images: Mapping[str, int]) -> Path:
    """Create the reserved object directory and remove safe stale entries."""
    object_dir = output_dir / "objects"
    if object_dir.is_symlink():
        _fail(f"materialized CUBIN object directory must not be a symlink: {object_dir}")
    object_dir.mkdir(exist_ok=True)
    if object_dir.is_symlink() or not object_dir.is_dir():
        _fail(f"materialized CUBIN object path must be a directory: {object_dir}")

    expected_names = {f"{digest}.cubin" for digest in images}
    for entry in object_dir.iterdir():
        if entry.name in expected_names and not entry.is_symlink() and entry.is_file():
            continue
        if entry.is_symlink() or entry.is_file():
            entry.unlink()
            continue
        _fail(f"unexpected non-file entry in materialized CUBIN object directory: {entry}")
    return object_dir


def _matches_generated_file(path: Path, expected: bytes) -> bool:
    """Return whether a regular generated file has the exact expected bytes."""
    try:
        return (
            not path.is_symlink()
            and path.is_file()
            and path.stat().st_size == len(expected)
            and path.read_bytes() == expected
        )
    except OSError:
        return False


def _is_complete(
    output_dir: Path,
    stamp: Mapping[str, object],
    images: Mapping[str, int],
    families: Sequence[FamilyIndex],
) -> bool:
    stamp_path = output_dir / "materialization.json"
    if output_dir.is_symlink() or stamp_path.is_symlink() or not stamp_path.is_file():
        return False
    try:
        actual = json.loads(
            stamp_path.read_bytes(), object_pairs_hook=_strict_object, parse_constant=_reject_json_constant
        )
    except (OSError, json.JSONDecodeError, MaterializationError):
        return False
    if actual != stamp:
        return False
    object_dir = output_dir / "objects"
    sorted_hashes = sorted(images)
    try:
        for digit in "0123456789abcdef":
            shard = output_dir / f"cubin_payloads_{digit}.S"
            shard_hashes = [digest for digest in sorted_hashes if digest[0] == digit]
            if not _matches_generated_file(shard, _assembly_shard(digit, shard_hashes, object_dir).encode()):
                return False
        for family in families:
            header = output_dir / f"{family.family}_registry.h"
            source = output_dir / f"{family.family}_registry.cpp"
            if not _matches_generated_file(header, _registry_header(family.family).encode()):
                return False
            if not _matches_generated_file(source, _registry_source(family).encode()):
                return False
    except OSError:
        return False
    return _has_exact_materialized_objects(object_dir, images)


def materialize(
    family_indexes: Sequence[tuple[str, Path]],
    output_root: Path,
) -> MaterializationResult:
    """Verify packs and generate assembly plus typed family registries.

    Args:
        family_indexes: Explicit ``(expected_family, index_path)`` pairs.
        output_root: Build-tree root under which the fingerprint directory is created.

    Returns:
        The materialization fingerprint and completed output directory.

    Raises:
        MaterializationError: If validation or materialization fails.
    """
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be pathlib.Path")
    families = _load_families(family_indexes)
    _verify_exact_pack_sets(families)
    images = _global_images(families)
    fingerprint = _fingerprint(families)
    output_root = output_root.resolve()
    output_dir = output_root / fingerprint
    stamp = _stamp_value(families, fingerprint, images)
    if _is_complete(output_dir, stamp, images, families):
        return MaterializationResult(fingerprint, output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.is_symlink():
            _fail(f"materialization directory must not be a symlink: {output_dir}")
        object_dir = _prepare_object_dir(output_dir, images)
        _verify_archives(families, object_dir)
        if not _has_exact_materialized_objects(object_dir, images):
            _fail(f"materialized CUBIN object set is incomplete or contains unindexed entries: {object_dir}")
        for digit in "0123456789abcdef":
            shard_hashes = [digest for digest in sorted(images) if digest[0] == digit]
            _atomic_write_if_changed(
                output_dir / f"cubin_payloads_{digit}.S",
                _assembly_shard(digit, shard_hashes, object_dir).encode(),
            )
        for family in families:
            _atomic_write_if_changed(
                output_dir / f"{family.family}_registry.h", _registry_header(family.family).encode()
            )
            _atomic_write_if_changed(output_dir / f"{family.family}_registry.cpp", _registry_source(family).encode())
        _atomic_write_if_changed(
            output_dir / "materialization.json",
            (json.dumps(stamp, indent=2, sort_keys=True) + "\n").encode(),
        )
    except MaterializationError:
        raise
    except OSError as error:
        raise MaterializationError(f"cannot materialize CUBIN build inputs in {output_dir}: {error}") from error
    return MaterializationResult(fingerprint, output_dir)


def _parse_index_argument(value: str) -> tuple[str, Path]:
    family, separator, raw_path = value.partition("=")
    if not separator or not family or not raw_path:
        raise argparse.ArgumentTypeError("expected FAMILY=PATH")
    return family, Path(raw_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True, help="build-tree output root")
    parser.add_argument(
        "--index",
        type=_parse_index_argument,
        action="append",
        required=True,
        metavar="FAMILY=PATH",
        help="expected family and its public index (repeatable)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the source-independent materializer CLI."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        result = materialize(arguments.index, arguments.output_root)
    except MaterializationError as error:
        parser.exit(2, f"error: {error}\n")
    print(result.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
