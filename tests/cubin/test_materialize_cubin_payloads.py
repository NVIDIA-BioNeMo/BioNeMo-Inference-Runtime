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
"""Source-independent and adversarial tests for public CUBIN materialization."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import lzma
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MATERIALIZER_PATH = REPO_ROOT / "cpp" / "cmake" / "materialize_cubin_payloads.py"


def _load_materializer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bioir_test_materializer", MATERIALIZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


materializer = _load_materializer()

FAMILIES = (
    "adaln_layernorm_sigmoid",
    "dual_gemm_x0_x1",
    "dual_gemm_x_x",
    "gated_sigmoid",
    "outer_product_mean",
    "pair_weighted_averaging",
    "pairwise_attention",
    "triangle_attention",
)


def _stable_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _aliases(family: str) -> list[dict[str, object]]:
    return {
        "adaln_layernorm_sigmoid": [{"feature_dim": 128, "num_threads": 128}],
        "dual_gemm_x0_x1": [{"K": 64, "N": 32, "has_bias": False}],
        "dual_gemm_x_x": [{"N": 32, "bucket": 64}],
        "gated_sigmoid": [{"K": 64, "N": 32, "m_bucket": 0}],
        "outer_product_mean": [{"default_config": True}],
        "pair_weighted_averaging": [{"n_anchor": 64, "s_anchor": 128}],
        "pairwise_attention": [{"head_dim": 64, "packed": False}],
        "triangle_attention": [{"head_dim": 64, "packed": False}],
    }[family]


def _tma_descriptor(dtype: str, rank: int) -> dict[str, object]:
    return {
        "data_type": "bfloat16" if dtype == "bf16" else "float16",
        "rank": rank,
        "global_dim_order": list(reversed(range(rank))),
        "box_dims": [16] * rank,
        "element_strides": [1] * rank,
        "interleave": "none",
        "swizzle": "128b",
        "l2_promotion": "128b",
        "oob_fill": "none",
    }


def _sm90_launch(family: str, dtype: str) -> dict[str, object]:
    dual = family in {"dual_gemm_x0_x1", "dual_gemm_x_x"}
    names = ("x0", "x1", "w0", "w1", "output") if dual else ("q", "k", "v", "bias", "output")
    result: dict[str, object] = {
        "block_dims": [128, 1, 1],
        "cluster_dims": [1, 1, 1],
        "cluster_scheduling_policy": "default",
        "tma_descriptors": {name: _tma_descriptor(dtype, 2 if dual else 4) for name in names},
    }
    if dual:
        result["epi_tile"] = [32, 32]
    return result


def _metadata(family: str, dtype: str, kernel_sm: int) -> dict[str, object]:
    common: dict[str, object] = {
        "dynamic_smem_bytes": 4096,
        "non_portable_cluster_size_allowed": kernel_sm == 90 and family != "adaln_layernorm_sigmoid",
    }
    concrete: dict[str, object]
    if family == "adaln_layernorm_sigmoid":
        concrete = {
            "dtype_name": dtype,
            "feature_dim": 128,
            "threads_per_row": 32,
            "num_threads": 128,
            "cluster_n": 1,
        }
    elif family == "gated_sigmoid":
        concrete = {
            "is_bfloat16": dtype == "bf16",
            "has_bias": False,
            "m_block_size": 64,
            "n_block_size": 64,
            "k_block_size": 32,
            "num_stages": 2,
            "raster_factor": 1,
            "num_threads": 128,
            "atom_layout_mnk": [2, 2, 1],
        }
    elif family == "outer_product_mean":
        concrete = {
            "is_bfloat16": dtype == "bf16",
            "has_bias": False,
            "norm_before": True,
            "config_identity": "tile-64x64",
            "tile_i": 64,
            "tile_j": 64,
            "raster_factor": 1,
            "num_threads": 128,
        }
    elif family == "pair_weighted_averaging":
        concrete = {
            "is_bfloat16": dtype == "bf16",
            "tile_i": 64,
            "tile_s": 32,
            "tile_j": 64,
            "num_threads": 128,
            "H": 4,
            "D": 32,
            "c_m": 16,
            "runtime_aliases": [{"n_anchor": 64, "s_anchor": 128}],
        }
    elif family in {"pairwise_attention", "triangle_attention"}:
        concrete = {
            "head_dim": 64,
            "bucket": 128,
            "is_bfloat16": dtype == "bf16",
            "packed_output": False,
            "sm90_launch": _sm90_launch(family, dtype) if kernel_sm == 90 else None,
        }
    elif family == "dual_gemm_x_x":
        concrete = {
            "K": 64,
            "is_bfloat16": dtype == "bf16",
            "transpose_out": False,
            "has_bias": False,
            "has_mask": False,
            "tile_m": 64,
            "tile_n": 64,
            "tile_k": 32,
            "num_threads": 128,
            "raster_factor": 1,
            "runtime_aliases": [{"N": 32, "bucket": 64}],
            "sm90_launch": _sm90_launch(family, dtype) if kernel_sm == 90 else None,
        }
    else:
        concrete = {
            "K": 64,
            "N": 32,
            "bucket": 64,
            "is_bfloat16": dtype == "bf16",
            "has_bias": False,
            "tile_m": 64,
            "tile_n": 64,
            "num_threads": 128,
            "raster_factor": 1,
            "sm90_launch": _sm90_launch(family, dtype) if kernel_sm == 90 else None,
        }
    return common | concrete


def _tar_bytes(
    members: list[tuple[str, bytes, bytes]],
    *,
    tar_format: int = tarfile.USTAR_FORMAT,
    trailing: bytes = b"",
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tar_format) as archive:
        for name, payload, member_type in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o444
            info.uid = 0
            info.gid = 0
            info.mtime = 0
            info.uname = ""
            info.gname = ""
            info.type = member_type
            if member_type == tarfile.SYMTYPE:
                info.linkname = "target"
                info.size = 0
                archive.addfile(info)
            else:
                archive.addfile(info, io.BytesIO(payload))
    return output.getvalue() + trailing


def _xz_bytes(tar_bytes: bytes) -> bytes:
    return lzma.compress(
        tar_bytes,
        format=lzma.FORMAT_XZ,
        check=lzma.CHECK_CRC64,
        filters=[{"id": lzma.FILTER_LZMA2, "preset": 6, "dict_size": 8 * 1024 * 1024}],
    )


def _write_case(
    root: Path,
    family: str,
    *,
    image: bytes | None = None,
    indexed_image: bytes | None = None,
    kernel_sm: int = 80,
    archive_name: str | None = None,
    member_type: bytes = tarfile.REGTYPE,
    duplicate_member: bool = False,
    tar_format: int = tarfile.USTAR_FORMAT,
    trailing_tar: bytes = b"",
    nonzero_padding: bool = False,
) -> Path:
    image = image if image is not None else b"\x7fELF" + family.encode()
    indexed_image = indexed_image if indexed_image is not None else image
    image_hash = hashlib.sha256(indexed_image).hexdigest()
    target_sm = 90 if kernel_sm == 90 else 80
    target_arch = "sm_90a" if kernel_sm == 90 else "sm_80"
    dtype = "fp32" if family == "adaln_layernorm_sigmoid" else "fp16"
    identity_spec = {"family_case": family, "kernel_sm": kernel_sm}
    canonical = {
        "registry_version": 2,
        "family": family,
        "target_sm": target_sm,
        "dtype": dtype,
        "spec": identity_spec,
    }
    variant_id = hashlib.sha256(_stable_json(canonical)).hexdigest()[:20]
    member_name = archive_name or f"objects/{image_hash}.cubin"
    members = [(member_name, image, member_type)]
    if duplicate_member:
        members.append((member_name, image, member_type))
    raw_tar = _tar_bytes(members, tar_format=tar_format, trailing=trailing_tar)
    if nonzero_padding:
        raw_tar = raw_tar[: 512 + len(image)] + b"x" + raw_tar[513 + len(image) :]
    compressed = _xz_bytes(raw_tar)
    pack_hash = hashlib.sha256(compressed).hexdigest()
    cubins = root / f"cutedsl_{family}" / "cubins"
    packs = cubins / "packs"
    packs.mkdir(parents=True)
    pack_name = f"{family}_{target_arch.replace('_', '')}_{pack_hash}.tar.xz"
    (packs / pack_name).write_bytes(compressed)
    index = {
        "format": "bioir-cubin-pack-v1",
        "family": family,
        "registry_version": 2,
        "compile_fingerprint": "1" * 64,
        "toolchain": {
            "nvidia-cutlass-dsl": "4.5.2",
            "compile_environment": f"sha256:{'2' * 64}",
        },
        "packs": {
            target_arch: {
                "file": f"packs/{pack_name}",
                "sha256": pack_hash,
                "size": len(compressed),
                "image_count": 1,
            }
        },
        "variants": [
            {
                "variant_id": variant_id,
                "target_sm": target_sm,
                "target_arch": target_arch,
                "kernel_sm": kernel_sm,
                "launch_abi": f"{family}_sm{kernel_sm}",
                "supported_sms": [target_sm],
                "dtype": dtype,
                "label": f"{family}.{variant_id}",
                "identity_spec": identity_spec,
                "aliases": _aliases(family),
                "runtime_metadata": _metadata(family, dtype, kernel_sm),
                "kernel_symbol": f"k{hashlib.sha256(family.encode()).hexdigest()}",
                "image": {"sha256": image_hash, "size": len(indexed_image), "pack": target_arch},
            }
        ],
    }
    index_path = cubins / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return index_path


def _pack_for_index(index_path: Path) -> Path:
    index = json.loads(index_path.read_text())
    pack = next(iter(index["packs"].values()))
    return index_path.parent / pack["file"]


@pytest.mark.parametrize("family", FAMILIES)
def test_materializes_every_family_registry_shape(tmp_path: Path, family: str) -> None:
    index = _write_case(tmp_path / "source", family)
    result = materializer.materialize([(family, index)], tmp_path / "build")

    assert result.output_dir == tmp_path / "build" / result.fingerprint
    assert len(result.fingerprint) == 64
    assert (result.output_dir / "materialization.json").is_file()
    assert len(list(result.output_dir.glob("cubin_payloads_?.S"))) == 16
    header = (result.output_dir / f"{family}_registry.h").read_text()
    source = (result.output_dir / f"{family}_registry.cpp").read_text()
    assert "struct RegistryView" in header
    assert "RegistryView registry() noexcept;" in header
    assert "RegistryView registry() noexcept" in source
    assert "CubinImage const kImages[]" in source
    public_label = json.loads(index.read_text())["variants"][0]["label"]
    assert public_label not in header
    assert public_label not in source
    objects = list((result.output_dir / "objects").glob("*.cubin"))
    assert len(objects) == 1 and objects[0].read_bytes().startswith(b"\x7fELF")


def test_sm90_registry_renders_tma_metadata(tmp_path: Path) -> None:
    family = "triangle_attention"
    index = _write_case(tmp_path / "source", family, kernel_sm=90)
    result = materializer.materialize([(family, index)], tmp_path / "build")
    source = (result.output_dir / f"{family}_registry.cpp").read_text()

    assert "CU_TENSOR_MAP_DATA_TYPE_FLOAT16" in source
    assert "CU_TENSOR_MAP_SWIZZLE_128B" in source
    assert "CU_CLUSTER_SCHEDULING_POLICY_DEFAULT" in source


def test_adaln_sm90_does_not_imply_nonportable_cluster_size(tmp_path: Path) -> None:
    family = "adaln_layernorm_sigmoid"
    index = _write_case(tmp_path / "source", family, kernel_sm=90)
    result = materializer.materialize([(family, index)], tmp_path / "build")
    source = (result.output_dir / f"{family}_registry.cpp").read_text()

    assert "      90," in source
    assert "      false," in source


def test_cross_family_images_have_one_object_and_definition(tmp_path: Path) -> None:
    image = b"\x7fELFshared-image"
    source = tmp_path / "source"
    first = _write_case(source, "triangle_attention", image=image)
    second = _write_case(source, "pairwise_attention", image=image)
    result = materializer.materialize(
        [("pairwise_attention", second), ("triangle_attention", first)],
        tmp_path / "build",
    )
    digest = hashlib.sha256(image).hexdigest()

    assert [path.name for path in (result.output_dir / "objects").iterdir()] == [f"{digest}.cubin"]
    assembly = "".join(path.read_text() for path in sorted(result.output_dir.glob("cubin_payloads_?.S")))
    assert assembly.count(f"{digest}_start:") == 1
    assert assembly.count(f'.incbin "{result.output_dir / "objects" / f"{digest}.cubin"}"') == 1


def test_verify_packs_is_write_free_and_reports_global_counts(tmp_path: Path) -> None:
    image = b"\x7fELFshared-image"
    first = _write_case(tmp_path / "source", "triangle_attention", image=image)
    second = _write_case(tmp_path / "source", "pairwise_attention", image=image)
    summary = materializer.verify_packs([("triangle_attention", first), ("pairwise_attention", second)])

    assert summary.family_count == 2
    assert summary.pack_count == 2
    assert summary.variant_count == 2
    assert summary.image_count == 1
    assert summary.total_image_bytes == len(image)
    assert not list(tmp_path.rglob("materialization.json"))


def test_materialization_is_stable_and_reuses_completion_stamp(tmp_path: Path) -> None:
    family = "gated_sigmoid"
    index = _write_case(tmp_path / "source", family)
    first = materializer.materialize([(family, index)], tmp_path / "build")
    stamp = first.output_dir / "materialization.json"
    original_stat = stamp.stat()
    second = materializer.materialize([(family, index)], tmp_path / "build")

    assert second == first
    assert stamp.stat().st_mtime_ns == original_stat.st_mtime_ns


def test_materialization_repairs_shards_copied_to_another_output_root(tmp_path: Path) -> None:
    family = "triangle_attention"
    index = _write_case(tmp_path / "source", family)
    first_root = tmp_path / "build-one"
    first = materializer.materialize([(family, index)], first_root)
    second_output = tmp_path / "build-two" / first.fingerprint
    second_output.parent.mkdir()
    shutil.copytree(first.output_dir, second_output)
    shutil.rmtree(first_root)

    second = materializer.materialize([(family, index)], second_output.parent)
    assembly = "".join(path.read_text() for path in sorted(second.output_dir.glob("cubin_payloads_?.S")))

    assert second.output_dir == second_output
    assert str(first.output_dir / "objects") not in assembly
    assert str(second.output_dir / "objects") in assembly


def test_materialization_repairs_modified_registry_source(tmp_path: Path) -> None:
    family = "pairwise_attention"
    index = _write_case(tmp_path / "source", family)
    first = materializer.materialize([(family, index)], tmp_path / "build")
    registry = first.output_dir / f"{family}_registry.cpp"
    expected = registry.read_bytes()
    registry.write_text("tampered\n")

    second = materializer.materialize([(family, index)], tmp_path / "build")

    assert second == first
    assert registry.read_bytes() == expected


def test_completion_reuse_prunes_unindexed_object(tmp_path: Path) -> None:
    family = "gated_sigmoid"
    index = _write_case(tmp_path / "source", family)
    first = materializer.materialize([(family, index)], tmp_path / "build")
    object_dir = first.output_dir / "objects"
    expected = {path.name for path in object_dir.iterdir()}
    extra_hash = hashlib.sha256(b"unindexed").hexdigest()
    extra = object_dir / f"{extra_hash}.cubin"
    extra.write_bytes(b"\x7fELFunindexed")

    second = materializer.materialize([(family, index)], tmp_path / "build")

    assert second == first
    assert {path.name for path in object_dir.iterdir()} == expected
    assert not extra.exists()


def test_completion_reuse_repairs_same_size_corrupted_object(tmp_path: Path) -> None:
    family = "gated_sigmoid"
    index = _write_case(tmp_path / "source", family)
    first = materializer.materialize([(family, index)], tmp_path / "build")
    object_path = next((first.output_dir / "objects").iterdir())
    expected = object_path.read_bytes()
    object_path.chmod(0o644)
    object_path.write_bytes(b"x" * len(expected))

    second = materializer.materialize([(family, index)], tmp_path / "build")

    assert second == first
    assert object_path.read_bytes() == expected


def test_rejects_symlinked_materialized_object_directory(tmp_path: Path) -> None:
    family = "gated_sigmoid"
    index = _write_case(tmp_path / "source", family)
    first = materializer.materialize([(family, index)], tmp_path / "build")
    object_dir = first.output_dir / "objects"
    shutil.rmtree(object_dir)
    outside = tmp_path / "outside"
    outside.mkdir()
    object_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(materializer.MaterializationError, match="object directory must not be a symlink"):
        materializer.materialize([(family, index)], tmp_path / "build")
    assert not list(outside.iterdir())


def test_all_shards_use_portable_elf_syntax_and_empty_shard_assembles(tmp_path: Path) -> None:
    family = "outer_product_mean"
    index = _write_case(tmp_path / "source", family)
    result = materializer.materialize([(family, index)], tmp_path / "build")
    shards = sorted(result.output_dir.glob("cubin_payloads_?.S"))

    assert all("%progbits" in shard.read_text() for shard in shards)
    populated = next(shard for shard in shards if ".incbin" in shard.read_text())
    assert ".type bioir_cubin_" in populated.read_text() and ",%object" in populated.read_text()
    assembler = shutil.which("as")
    if assembler is None:
        pytest.skip("GNU assembler is unavailable")
    empty = next(shard for shard in shards if ".incbin" not in shard.read_text())
    subprocess.run([assembler, "-o", tmp_path / "empty.o", empty], check=True, capture_output=True, text=True)
    subprocess.run([assembler, "-o", tmp_path / "payload.o", populated], check=True, capture_output=True, text=True)


def test_all_generated_registry_sources_compile_as_cpp17(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    cuda_include = Path("/usr/local/cuda/include")
    if compiler is None or not (cuda_include / "cuda.h").is_file():
        pytest.skip("a C++ compiler and CUDA headers are required")
    indexes = [
        (family, _write_case(tmp_path / "source", family, kernel_sm=90 if family == "triangle_attention" else 80))
        for family in FAMILIES
    ]
    result = materializer.materialize(indexes, tmp_path / "build")

    for family in FAMILIES:
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                f"-I{result.output_dir}",
                f"-I{REPO_ROOT / 'cpp' / 'kernels'}",
                f"-I{cuda_include}",
                "-c",
                result.output_dir / f"{family}_registry.cpp",
                "-o",
                tmp_path / f"{family}.o",
            ],
            check=True,
            capture_output=True,
            text=True,
        )


@pytest.mark.parametrize(
    ("case", "mutate", "match"),
    [
        (
            "duplicate-json-key",
            lambda index: index.write_text(index.read_text().replace('"format":', '"format": "bad", "format":', 1)),
            "duplicate key",
        ),
        (
            "bad-alias-shape",
            lambda index: _mutate_index(index, lambda value: value["variants"][0]["aliases"][0].update(extra=1)),
            "invalid keys",
        ),
        (
            "duplicate-runtime-key",
            lambda index: _add_duplicate_runtime_key(index),
            "runtime key",
        ),
        (
            "corrupt-compressed-pack",
            lambda index: _pack_for_index(index).write_bytes(_pack_for_index(index).read_bytes()[:-1] + b"x"),
            "XZ pack|SHA-256",
        ),
        (
            "lfs-pointer",
            lambda index: _pack_for_index(index).write_text(
                "version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 123\n"
            ),
            "Git LFS pointer",
        ),
    ],
)
def test_rejects_index_and_pack_corruption(
    tmp_path: Path,
    case: str,
    mutate: Callable[[Path], object],
    match: str,
) -> None:
    family = "triangle_attention"
    index = _write_case(tmp_path / case, family)
    mutate(index)

    with pytest.raises(materializer.MaterializationError, match=match):
        materializer.materialize([(family, index)], tmp_path / "build")
    assert not list((tmp_path / "build").rglob("materialization.json"))


def _mutate_index(index: Path, mutation: Callable[[dict[str, object]], object]) -> None:
    value = json.loads(index.read_text())
    mutation(value)
    index.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _add_duplicate_runtime_key(index: Path) -> None:
    def mutate(value: dict[str, object]) -> None:
        variants = value["variants"]
        assert isinstance(variants, list)
        duplicate = json.loads(json.dumps(variants[0]))
        duplicate["identity_spec"]["duplicate"] = True
        canonical = {
            "registry_version": 2,
            "family": value["family"],
            "target_sm": duplicate["target_sm"],
            "dtype": duplicate["dtype"],
            "spec": duplicate["identity_spec"],
        }
        duplicate["variant_id"] = hashlib.sha256(_stable_json(canonical)).hexdigest()[:20]
        duplicate["label"] = f"{value['family']}.{duplicate['variant_id']}"
        variants.append(duplicate)
        variants.sort(key=lambda variant: variant["variant_id"])

    _mutate_index(index, mutate)


def test_rejects_private_style_public_label(tmp_path: Path) -> None:
    family = "dual_gemm_x_x"
    index = _write_case(tmp_path / "source", family, kernel_sm=90)
    _mutate_index(
        index,
        lambda value: value["variants"][0].update(label="K128.DualGemmSm90Pingpong.fp16.t0.bias0.mask0"),
    )

    with pytest.raises(materializer.MaterializationError, match="canonical public label"):
        materializer.verify_packs([(family, index)])


def test_rejects_symlinked_index(tmp_path: Path) -> None:
    family = "triangle_attention"
    index = _write_case(tmp_path / "source", family)
    symlink = tmp_path / "index.json"
    symlink.symlink_to(index)

    with pytest.raises(materializer.MaterializationError, match="CUBIN index must not be a symlink"):
        materializer.materialize([(family, symlink)], tmp_path / "build")


@pytest.mark.parametrize(
    ("case", "kwargs", "match"),
    [
        ("traversal", {"archive_name": "../escape.cubin"}, "canonical USTAR name|objects/<sha256>"),
        ("symlink", {"member_type": tarfile.SYMTYPE}, "regular-file USTAR"),
        ("duplicate", {"duplicate_member": True}, "after its final member|missing, duplicated"),
        ("gnu", {"tar_format": tarfile.GNU_FORMAT}, "POSIX USTAR"),
        ("member-padding", {"nonzero_padding": True}, "nonzero USTAR member padding"),
        ("tar-trailing", {"trailing_tar": b"evil"}, "after its final member"),
        ("non-elf", {"image": b"NOPE-invalid-image"}, "not an ELF"),
        (
            "wrong-image-hash",
            {"image": b"\x7fELF-wrong-content", "indexed_image": b"\x7fELF-right-content"},
            "content does not match",
        ),
    ],
)
def test_rejects_adversarial_archive(
    tmp_path: Path,
    case: str,
    kwargs: dict[str, object],
    match: str,
) -> None:
    family = "triangle_attention"
    index = _write_case(tmp_path / case, family, **kwargs)

    with pytest.raises(materializer.MaterializationError, match=match):
        materializer.verify_packs([(family, index)])


def test_rejects_concatenated_xz_stream(tmp_path: Path) -> None:
    family = "triangle_attention"
    index = _write_case(tmp_path / "source", family)
    pack_path = _pack_for_index(index)
    concatenated = pack_path.read_bytes() + _xz_bytes(b"extra")
    pack_hash = hashlib.sha256(concatenated).hexdigest()
    old_name = pack_path.name
    target_arch = "sm_80"
    new_name = f"{family}_sm80_{pack_hash}.tar.xz"
    pack_path.rename(pack_path.with_name(new_name))
    pack_path = pack_path.with_name(new_name)
    pack_path.write_bytes(concatenated)

    def mutate(value: dict[str, object]) -> None:
        pack = value["packs"][target_arch]
        pack["file"] = f"packs/{new_name}"
        pack["sha256"] = pack_hash
        pack["size"] = len(concatenated)

    _mutate_index(index, mutate)
    assert old_name != new_name
    with pytest.raises(materializer.MaterializationError, match="more than one XZ stream|trailing data"):
        materializer.verify_packs([(family, index)])


def test_rejects_noncanonical_xz_dictionary_profile(tmp_path: Path) -> None:
    family = "triangle_attention"
    index = _write_case(tmp_path / "source", family)
    pack_path = _pack_for_index(index)
    compressed = lzma.compress(
        lzma.decompress(pack_path.read_bytes()),
        format=lzma.FORMAT_XZ,
        check=lzma.CHECK_CRC64,
        filters=[{"id": lzma.FILTER_LZMA2, "preset": 6, "dict_size": 1024 * 1024}],
    )
    pack_hash = hashlib.sha256(compressed).hexdigest()
    new_name = f"{family}_sm80_{pack_hash}.tar.xz"
    pack_path = pack_path.rename(pack_path.with_name(new_name))
    pack_path.write_bytes(compressed)

    def mutate(value: dict[str, object]) -> None:
        pack = value["packs"]["sm_80"]
        pack["file"] = f"packs/{new_name}"
        pack["sha256"] = pack_hash
        pack["size"] = len(compressed)

    _mutate_index(index, mutate)
    with pytest.raises(materializer.MaterializationError, match="LZMA2 8 MiB dictionary"):
        materializer.verify_packs([(family, index)])


@pytest.mark.parametrize("operation", ("verify", "materialize", "materialize-reuse"))
def test_rejects_unreferenced_sibling_pack(tmp_path: Path, operation: str) -> None:
    family = "triangle_attention"
    index = _write_case(tmp_path / "source", family)
    if operation == "materialize-reuse":
        materializer.materialize([(family, index)], tmp_path / "build")
    extra = index.parent / "packs" / f"{family}_sm89_{'0' * 64}.tar.xz"
    extra.write_bytes(b"unreferenced")

    with pytest.raises(materializer.MaterializationError, match="does not exactly match.*unreferenced"):
        if operation == "verify":
            materializer.verify_packs([(family, index)])
        else:
            materializer.materialize([(family, index)], tmp_path / "build")


def test_rejects_indexed_pack_size_before_decompression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    family = "triangle_attention"
    index = _write_case(tmp_path / "source", family)

    def record_wrong_size(value: dict[str, object]) -> None:
        pack = next(iter(value["packs"].values()))
        pack["size"] += 1

    _mutate_index(index, record_wrong_size)

    def unexpected_decompression(*_args: object, **_kwargs: object) -> object:
        pytest.fail("pack size mismatch reached the XZ decompressor")

    monkeypatch.setattr(materializer.lzma, "LZMADecompressor", unexpected_decompression)
    with pytest.raises(materializer.MaterializationError, match="compressed size is .* expected"):
        materializer.verify_packs([(family, index)])


def test_cli_prints_only_materialized_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    family = "adaln_layernorm_sigmoid"
    index = _write_case(tmp_path / "source", family)
    status = materializer.main(["--output-root", str(tmp_path / "build"), "--index", f"{family}={index}"])
    captured = capsys.readouterr()

    assert status == 0
    assert Path(captured.out.strip()).is_dir()
    assert captured.err == ""
