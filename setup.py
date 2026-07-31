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

This project has no compiled ``Extension`` of its own, but it is not portable:
its dependencies are CUDA-toolkit-specific wheels (``cuda-python>=13``,
``cuequivariance-ops-cu13``, ...) and its runtime kernels are JIT-compiled for the
installed CUDA. So the wheel must advertise the CUDA target it was built for
rather than presenting as pure Python. Two things setuptools cannot infer on its
own are supplied here:

1. A PEP 440 local version segment (e.g. ``0.4.0+cu131``) that records the CUDA
   toolkit the wheel was built against — the same scheme PyTorch ships
   (``2.7.0+cu128``). See https://peps.python.org/pep-0440/#local-version-identifiers.
2. A platform wheel tag. With no ``Extension``, setuptools treats the project as
   pure Python and tags the wheel ``py3-none-any``; ``_BinaryDistribution`` flips
   that so the wheel carries interpreter + platform tags (e.g.
   ``cp312-none-linux_x86_64``).

Keeping both inside the produced wheel's METADATA/WHEEL (rather than renaming the
file afterward) is what lets ``pip install`` accept it — a filename whose version
or tags disagree with the archive's own metadata is rejected by pip >= 24.2.
"""
import os
import re
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools.dist import Distribution

ROOT_DIR = Path(__file__).parent.resolve()


def _base_version() -> str:
    """Read ``__version__`` from version.py WITHOUT importing the package.

    ``tensorrt_bionemo/__init__.py`` imports torch/CUDA extensions, which may be
    unavailable (or slow) during metadata resolution — so parse the trivial
    assignment directly, mirroring setuptools' own static ``attr:`` extraction.
    """
    version_file = ROOT_DIR / "tensorrt_bionemo" / "version.py"
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
        "CUDA_TAG (e.g. CUDA_TAG=cu131) to override.")


class _BinaryDistribution(Distribution):
    """Mark the distribution impure so a platform (not ``py3-none-any``) wheel is built.

    The project has no ``ext_modules``, so setuptools would otherwise tag the wheel
    pure — but it targets a specific CUDA toolkit (see the module docstring).
    Returning True here does not trigger any compilation — it only flips the wheel's
    purity/tags.
    """

    def has_ext_modules(self) -> bool:  # noqa: D102 - see class docstring
        return True


class bdist_wheel(_bdist_wheel):
    """Emit an ABI-agnostic platform tag (e.g. ``cp312-none-linux_x86_64``).

    ``_BinaryDistribution`` already makes the wheel platform-specific; this drops
    the ABI tag to ``none`` because the package contains no CPython C-extension
    linked against a specific interpreter ABI — only pure-Python modules that are
    CUDA-toolkit-specific at runtime.
    """

    def get_tag(self):
        python, _abi, platform = super().get_tag()
        return python, "none", platform


setup(
    version=f"{_base_version()}+{_cuda_local_version()}",
    distclass=_BinaryDistribution,
    cmdclass={"bdist_wheel": bdist_wheel},
)
