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

"""Release ABI and dynamic-loader regressions."""

import hashlib
import importlib.util
import struct
from pathlib import Path
from zipfile import ZipFile

import pytest

if not all(importlib.util.find_spec(name) for name in ("cpp", "cpp.tools", "cpp.tools.check_release_wheel")):
    pytest.skip("Release wheel tools are unavailable in this checkout.", allow_module_level=True)

_CHECK = importlib.import_module("cpp.tools.check_release_wheel")

_PLATFORM = "manylinux_2_34_x86_64"
_DRIVER = "0x0000000000000001 (NEEDED) Shared library: [libcuda.so.1]\n"


def _write_wheel(
    directory: Path,
    *,
    abi: str = "abi3",
    metadata_abi: str = "abi3",
    suffix: str = ".abi3.so",
) -> Path:
    path = directory / f"bionemo_ir-0.2.0-cp312-{abi}-{_PLATFORM}.whl"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "bionemo_ir-0.2.0.dist-info/WHEEL",
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp312-{metadata_abi}-{_PLATFORM}\n",
        )
        archive.writestr(f"bionemo_ir/libs/_cutedsl_kernels{suffix}", b"extension")
    return path


def test_stable_wheel(tmp_path: Path) -> None:
    assert _CHECK.check_wheel(_write_wheel(tmp_path), _PLATFORM) == b"extension"


@pytest.mark.parametrize(
    "options",
    [
        {"abi": "cp312"},
        {"metadata_abi": "cp312"},
        {"suffix": ".cpython-312-x86_64-linux-gnu.so"},
    ],
)
def test_reject_abi_drift(tmp_path: Path, options: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        _CHECK.check_wheel(_write_wheel(tmp_path, **options), _PLATFORM)


@pytest.mark.parametrize("tag", ["RUNPATH", "RPATH"])
@pytest.mark.parametrize(
    "path",
    [
        "/usr/local/cuda/lib64",
        "",
        "$ORIGIN:",
        ":$ORIGIN",
        "lib",
        "$ORIGIN/../../../lib",
        "$ORIGIN/../../../../usr/local/cuda/lib64",
        "$ORIGIN/nested/../../../../tmp",
        "$ORIGIN/../../../bionemo_ir/libs",
    ],
)
def test_reject_loader_paths(tag: str, path: str) -> None:
    with pytest.raises(ValueError, match="nonportable loader path"):
        _CHECK.check_dynamic(_DRIVER + f"0x1 ({tag}) Library path: [{path}]")


@pytest.mark.parametrize("library", ["libcudart.so.13", "libnvrtc.so.13", "/opt/libcuda.so.1"])
def test_reject_toolkit_link(library: str) -> None:
    with pytest.raises(ValueError, match="non-system libraries"):
        _CHECK.check_dynamic(_DRIVER + f"0x1 (NEEDED) Shared library: [{library}]")


@pytest.mark.parametrize("tag", ["RUNPATH", "RPATH"])
@pytest.mark.parametrize("path", ["$ORIGIN", "$ORIGIN/../lib", "$ORIGIN/../..", "$ORIGIN/nested/../../lib"])
def test_portable_loader_paths(tag: str, path: str) -> None:
    _CHECK.check_dynamic(_DRIVER + f"0x1 ({tag}) Library path: [{path}]")


def test_external_driver() -> None:
    with pytest.raises(ValueError, match="external CUDA driver"):
        _CHECK.check_dynamic("0x1 (NEEDED) Shared library: [libc.so.6]")
    _CHECK.check_dynamic(_DRIVER)
    _CHECK.check_dynamic(_DRIVER + "0x1 (RUNPATH) Library path: [$ORIGIN:$ORIGIN/../lib]")


def test_cubin_identity(tmp_path: Path) -> None:
    cubin = bytearray(120)
    cubin[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<H", cubin, 18, 190)
    struct.pack_into("<Q", cubin, 32, 64)
    struct.pack_into("<HH", cubin, 54, 56, 1)
    digest = hashlib.sha256(cubin).hexdigest()
    (tmp_path / f"{digest}.cubin").write_bytes(cubin)
    _CHECK.check_cubins(b"prefix" + cubin + b"suffix", tmp_path)
    with pytest.raises(ValueError, match="1 missing"):
        _CHECK.check_cubins(b"no images", tmp_path)
    with pytest.raises(ValueError, match="1 duplicated"):
        _CHECK.check_cubins(bytes(cubin) * 2, tmp_path)
