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

"""Friendly errors for data-pipeline packages that only the ``all`` extra installs.

The base install (``pip install bionemo-ir``) is the model side only. Code that
needs a data-pipeline package wraps that import in :func:`pipeline_extra`. Keep
this file free of third-party imports so that it works in a base install.
"""

import importlib.util
from collections.abc import Iterator
from contextlib import contextmanager

INSTALL_HINT = 'pip install "bionemo-ir[all]"'

# Top-level import names that the code imports from the ``all`` extra in
# pyproject.toml. pyarrow (through ray[data]) and ihm (through modelcif) are
# installed by a package the extra names. tests/test_extras.py keeps this in sync.
# scipy is here because the pipeline imports it, but the hint never fires for it
# in practice: cuequivariance, in the base install, depends on scipy.
PIPELINE_MODULES = frozenset(
    {"Bio", "biotite", "gemmi", "ihm", "kalign", "modelcif", "pyarrow", "ray", "rdkit", "scipy"}
)


def _absent(package: str) -> bool:
    """Whether the top-level ``package`` cannot be found at all.

    ``find_spec`` raises ``ValueError`` for a module in ``sys.modules`` whose
    ``__spec__`` is None, such as a test stub. Count that as present, so the
    original error is the one the caller sees.
    """
    try:
        return importlib.util.find_spec(package) is None
    except (ImportError, ValueError):
        return False


@contextmanager
def pipeline_extra() -> Iterator[None]:
    """Turn a missing pipeline-only package into an error that names the extra.

    Only a package that is absent gets the hint. If the package is installed and
    one of its submodules is missing, installing the extra would not help, so that
    error passes through unchanged, as does any other import error. The new error
    is still a ``ModuleNotFoundError``, so existing handlers keep working.
    """
    try:
        yield
    except ModuleNotFoundError as error:
        package = (error.name or "").partition(".")[0]
        if package not in PIPELINE_MODULES or not _absent(package):
            raise
        raise ModuleNotFoundError(
            f"The '{package}' package is not installed. It is part of the BioIR data pipeline. "
            f"Install it with: {INSTALL_HINT}",
            name=error.name,
        ) from error
