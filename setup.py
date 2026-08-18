# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Build-time packaging shim. pyproject.toml remains the metadata source of truth.

``build_ext`` delegates the ``bionemo_ir.libs`` nanobind extension, which
embeds architecture-specific CuTeDSL CUBINs, to CMake; the local version segment
(for example ``+cu131``) records the CUDA toolkit that built the wheel, and
``BIOIR_VERSION_LOCAL`` appends more segments to it so a wheel built from an
arbitrary commit is traceable (for example ``+cu131.g1a2b3c4``). Every checkout
builds the extension, and ``BIOIR_BUILD_CUTEDSL_KERNELS=0`` opts out from the
environment or an untracked ``build.env``.
"""

import os
import re
import subprocess
import sys
from importlib import util as importlib_util
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from types import ModuleType

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT_DIR = Path(__file__).parent.resolve()
_BUILD_ENV_FILE = ROOT_DIR / "build.env"
_KERNELS_DIR = ROOT_DIR / "cpp" / "kernels"
_CUBIN_MATERIALIZER = ROOT_DIR / "cpp" / "cmake" / "materialize_cubin_payloads.py"
_PRIVATE_CUBIN_PREPARER = ROOT_DIR / "cpp" / "tools" / "prepare_cubins.py"
_BUILD_CUTEDSL_KERNELS_ENV = "BIOIR_BUILD_CUTEDSL_KERNELS"
_ALLOW_STALE_CUBIN_BUILD_ENV = "BIOIR_ALLOW_STALE_CUBIN_BUILD"
_KERNEL_LIBRARY_STEM = "_cutedsl_kernels"
_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_ENV_VALUES = frozenset({"0", "false", "no", "off"})


def _read_build_env(path: Path) -> dict[str, str]:
    """Read a small shell-compatible KEY=VALUE build environment file."""
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, separator, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or not key:
            raise RuntimeError(f"{path}:{line_number}: expected a KEY=VALUE assignment")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _env_flag(name: str, *, env_file: Path, default: bool) -> bool:
    """Resolve a boolean flag with process environment taking precedence."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        raw_value = _read_build_env(env_file).get(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    raise RuntimeError(f"{name} must be one of {sorted(_TRUE_ENV_VALUES | _FALSE_ENV_VALUES)}, got {raw_value!r}")


def _remove_stale_kernel_libraries(directories: set[Path]) -> None:
    """Remove stale unified CuTeDSL extensions before rebuilding."""
    for directory in directories:
        for suffix in EXTENSION_SUFFIXES:
            path = directory / f"{_KERNEL_LIBRARY_STEM}{suffix}"
            if path.is_file() or path.is_symlink():
                path.unlink()


def _cubin_family_indexes() -> tuple[tuple[str, Path], ...]:
    """Return every CMake CUBIN family's committed public artifact index."""
    family_indexes: list[tuple[str, Path]] = []
    missing: list[Path] = []
    for launcher in sorted(_KERNELS_DIR.glob("cutedsl_*/launcher.cpp")):
        family = launcher.parent.name.removeprefix("cutedsl_")
        index = launcher.parent / "cubins" / "index.json"
        if index.is_file():
            family_indexes.append((family, index))
        else:
            missing.append(index)
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise RuntimeError(
            "Missing committed CUBIN family indexes:\n  "
            f"{formatted}\nRestore the complete public artifact corpus before building."
        )
    if not family_indexes:
        raise RuntimeError(f"No CuTeDSL kernel families found below {_KERNELS_DIR}")
    return tuple(family_indexes)


def _load_cubin_materializer() -> ModuleType:
    """Load the source-independent materializer without making ``cpp`` a package."""
    if not _CUBIN_MATERIALIZER.is_file():
        raise RuntimeError(f"Missing public CUBIN materializer: {_CUBIN_MATERIALIZER}")
    spec = importlib_util.spec_from_file_location("_bioir_cubin_materializer", _CUBIN_MATERIALIZER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load public CUBIN materializer: {_CUBIN_MATERIALIZER}")
    module = importlib_util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _materialize_cutedsl_kernel_payloads(output_root: Path) -> Path:
    """Verify committed packs and create CMake inputs below ``output_root``."""
    # Private checkouts carry the builders and declared source inputs, so reject
    # a stale family before doing public, source-independent pack validation.
    # Source distributions deliberately omit cpp/tools and skip this gate. The
    # one exception is an ordinary internal MR build: post-merge automation owns
    # fingerprint-only drift, so CI may exercise the still-current public packs
    # before the protected refresh job updates them.
    # Release, schedule, public-export, rolling-refresh, and normal local builds
    # never set this narrowly scoped process variable and remain strict.
    allow_stale = os.environ.get(_ALLOW_STALE_CUBIN_BUILD_ENV)
    if allow_stale not in {None, "1"}:
        raise RuntimeError(f"{_ALLOW_STALE_CUBIN_BUILD_ENV} must be exactly '1' when set")
    if _PRIVATE_CUBIN_PREPARER.is_file() and allow_stale != "1":
        subprocess.run(
            [sys.executable, str(_PRIVATE_CUBIN_PREPARER), "--check-freshness"],
            cwd=ROOT_DIR,
            check=True,
        )
    module = _load_cubin_materializer()
    result = module.materialize(_cubin_family_indexes(), output_root.resolve())
    output_dir = Path(result.output_dir).resolve()
    if not output_dir.is_relative_to(output_root.resolve()):
        raise RuntimeError(f"CUBIN materializer returned a path outside its output root: {output_dir}")
    return output_dir


def _drop_opted_out_kernel_libraries() -> None:
    """Remove an extension left behind by an earlier build, now opted out of.

    ``build_extension`` never runs for a disabled extension, so this is the only
    place a previously built copy gets cleaned up.
    """
    package_dirs = {ROOT_DIR / "bionemo_ir"}
    build_root = ROOT_DIR / "build"
    if build_root.is_dir():
        package_dirs.update(build_root.glob("lib*/bionemo_ir"))
    _remove_stale_kernel_libraries(package_dirs | {path / "libs" for path in package_dirs})


_BUILD_CUTEDSL_KERNELS = _env_flag(
    _BUILD_CUTEDSL_KERNELS_ENV,
    env_file=_BUILD_ENV_FILE,
    default=True,
)


def _base_version() -> str:
    """Read ``__version__`` from version.py WITHOUT importing the package.

    ``bionemo_ir/__init__.py`` imports torch/CUDA extensions, which may be
    unavailable (or slow) during metadata resolution — so parse the trivial
    assignment directly, mirroring setuptools' own static ``attr:`` extraction.
    """
    version_file = ROOT_DIR / "bionemo_ir" / "version.py"
    for line in version_file.read_text().splitlines():
        if line.startswith("__version__"):
            namespace: dict = {}
            exec(line, namespace)  # trivial first-party literal assignment
            return namespace["__version__"]
    raise RuntimeError(f"__version__ not found in {version_file}")


def _cuda_local_version() -> str:
    """CUDA tag (e.g. ``cu131``) for the PEP 440 local version segment.

    Resolution order: the ``CUDA_TAG`` env var is an explicit override and wins
    outright; otherwise detect from nvcc, then torch. If none yields a tag, fail
    loudly rather than guessing a default that could mislabel the wheel.
    """
    override = os.environ.get("CUDA_TAG")
    if override:
        return override

    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True)
    except (OSError, subprocess.SubprocessError):
        pass
    else:
        match = re.search(r"release (\d+)\.(\d+)", out)
        if match:
            return f"cu{match.group(1)}{match.group(2)}"

    try:
        import torch
    except ImportError:
        pass
    else:
        if torch.version.cuda:
            return "cu" + torch.version.cuda.replace(".", "")

    raise RuntimeError(
        "Cannot determine the CUDA version for the wheel's local version segment: "
        "nvcc is unavailable and torch is not installed (or is CPU-only). Set "
        "CUDA_TAG (e.g. CUDA_TAG=cu131) to override."
    )


def _extra_local_version() -> str:
    """Build-identity segments appended to the local version, from the environment.

    A wheel built off a branch has to name the commit it came from — its base
    version cannot, because ``version.py`` only changes at a release. CI sets
    ``BIOIR_VERSION_LOCAL`` (for example ``g1a2b3c4`` or ``g1a2b3c4 mybranch``)
    and this normalizes it to `PEP 440 local version segments
    <https://packaging.python.org/en/latest/specifications/version-specifiers/#local-version-identifiers>`_
    — lowercase alphanumerics separated by dots, which is all the grammar
    allows. Unset (a release build) leaves the version at ``<base>+<cuda tag>``.

    Returns:
        The segments to append, each already dot-prefixed, or ``""``.
    """
    raw = os.environ.get("BIOIR_VERSION_LOCAL", "").strip()
    if not raw:
        return ""

    normalized = re.sub(r"[^A-Za-z0-9]+", ".", raw).strip(".").lower()
    if not normalized:
        raise ValueError(f"BIOIR_VERSION_LOCAL has no alphanumeric content: {raw!r}")
    return f".{normalized}"


class CMakeExtension(Extension):
    """A setuptools extension whose implementation is built by CMake."""

    def __init__(self, name: str, source_dir: Path):
        super().__init__(name, sources=[])
        self.source_dir = source_dir.resolve()


class CMakeBuild(build_ext):
    """Configure and build the unified embedded-CUBIN nanobind module."""

    def run(self) -> None:
        # Only a real build cleans up, so reading metadata (``egg_info``,
        # ``--version``) cannot delete an extension the tree still needs.
        if not _BUILD_CUTEDSL_KERNELS:
            _drop_opted_out_kernel_libraries()
        super().run()

    def build_extension(self, extension: Extension) -> None:
        if not isinstance(extension, CMakeExtension):
            super().build_extension(extension)
            return

        extension_path = Path(self.get_ext_fullpath(extension.name)).resolve()
        extension_dir = extension_path.parent
        source_package_dir = ROOT_DIR / "bionemo_ir"
        build_package_dir = Path(self.build_lib).resolve() / "bionemo_ir"
        package_dirs = {
            source_package_dir,
            build_package_dir,
            extension_dir.parent,
        }
        _remove_stale_kernel_libraries(package_dirs | {path / "libs" for path in package_dirs})
        extension_dir.mkdir(parents=True, exist_ok=True)

        configuration = "Debug" if self.debug else "Release"
        build_dir = (Path(self.build_temp) / extension.name.replace(".", "_")).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        materialized_dir = _materialize_cutedsl_kernel_payloads(Path(self.build_temp) / "bioir_cubins")

        configure_command = [
            "cmake",
            "-S",
            str(extension.source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_BUILD_TYPE={configuration}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extension_dir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{configuration.upper()}={extension_dir}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DBIOIR_CUBIN_MATERIALIZED_DIR={materialized_dir}",
        ]
        subprocess.run(configure_command, cwd=ROOT_DIR, check=True)

        build_command = [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            configuration,
            "--target",
            "_cutedsl_kernels",
        ]
        if self.parallel and "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_command.extend(["--parallel", str(self.parallel)])
        subprocess.run(build_command, cwd=ROOT_DIR, check=True)

        if not extension_path.is_file():
            raise RuntimeError(f"CMake did not produce the expected extension: {extension_path}")


setup(
    version=f"{_base_version()}+{_cuda_local_version()}{_extra_local_version()}",
    ext_modules=(
        [CMakeExtension("bionemo_ir.libs._cutedsl_kernels", ROOT_DIR / "cpp")] if _BUILD_CUTEDSL_KERNELS else []
    ),
    cmdclass={"build_ext": CMakeBuild},
)
