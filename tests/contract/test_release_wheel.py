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
import shutil
import struct
import sysconfig
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


def test_parse_limited_api_declarations() -> None:
    preprocessed = """
    __BIOIR_ABI3_SYM__ PyObject * PyModuleDef_Init(PyModuleDef *);
    __BIOIR_ABI3_SYM__ PyTypeObject PyLong_Type;
    __BIOIR_ABI3_SYM__ PyObject * __attribute__((deprecated)) PyExc_ValueError;
    __BIOIR_ABI3_SYM__ int Py_AtExit(void (*func)(void));
    __BIOIR_ABI3_SYM__ int (*PyOS_InputHook)(void);
    __BIOIR_ABI3_SYM__ PyObject _Py_NoneStruct;
    __BIOIR_ABI3_SYM__ const char * _Py_Array[1];
    """
    assert _CHECK.parse_limited_api_declarations(preprocessed) == frozenset(
        {
            "PyModuleDef_Init",
            "PyLong_Type",
            "PyExc_ValueError",
            "Py_AtExit",
            "PyOS_InputHook",
            "_Py_NoneStruct",
            "_Py_Array",
        }
    )


def test_check_abi3_accepts_and_rejects_symbols() -> None:
    allowed = frozenset({"PyModuleDef_Init", "PyObject_GetTypeData", "_Py_NoneStruct"})
    valid_dynsyms = (
        "     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND \n"
        "     1: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND cuModuleLoadData\n"
        "     2: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND PyModuleDef_Init\n"
        "     3: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT [VARIANT_PCS] UND PyObject_GetTypeData\n"
        "     4: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND _Py_NoneStruct@PYTHON_3.12\n"
        "     5: 0000000000001234    16 FUNC    GLOBAL DEFAULT   12 PyInit__cutedsl_kernels\n"
    )
    _CHECK.check_abi3(valid_dynsyms, allowed)

    with pytest.raises(ValueError, match="no Python C API symbols"):
        _CHECK.check_abi3("     1: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND cuInit\n", allowed)

    invalid_dynsyms = valid_dynsyms + "     6: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND PyRun_SimpleString\n"
    with pytest.raises(ValueError, match="PyRun_SimpleString"):
        _CHECK.check_abi3(invalid_dynsyms, allowed)


def test_load_limited_api_symbols_from_headers() -> None:
    include = Path(sysconfig.get_path("include"))
    if shutil.which("g++") is None or not (include / "Python.h").is_file():
        pytest.skip("g++ or Python.h unavailable in this environment.")
    symbols = _CHECK.load_limited_api_symbols()
    assert {"PyModuleDef_Init", "PyObject_GetTypeData", "Py_AtExit", "PyOS_InputHook", "_Py_NoneStruct"} <= symbols
    assert not (
        {"PyRun_SimpleString", "PyEval_SetTrace", "PyCode_GetCode", "PyUnstable_Object_IsUniquelyReferenced"} & symbols
    )
