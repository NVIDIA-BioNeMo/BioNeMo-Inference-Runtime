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
"""Global pytest configuration for the BioIR test suite.

Disable TF32 for fp32 math before any CUDA context / cuBLAS handle is created.
The numerical tests compare against IEEE fp32 references at tight tolerances
(atol ~1e-3, even ~1e-5), which TF32 cannot meet.

Two layers, set at conftest import time (pytest imports this before collecting or
running any test, i.e. before the first GPU op creates a cuBLAS handle):

1. ``NVIDIA_TF32_OVERRIDE=0`` -- the global CUDA kill switch, read by cuBLAS /
   cuDNN at handle creation. This is the reliable, torch-version-independent
   switch at cuBLAS/cuDNN handle creation time. Must be set
   *before* ``import torch`` / first CUDA use.
2. The torch backend flags -- explicit belt-and-suspenders for torch's own ops.

Individual tests also set these env vars inside their bodies, but that lands
after CUDA is initialized and is only honored on some torch versions (torch 2.10
/ nv25.12 honors it, torch 2.8 / nv25.08 does not), so the same test can behave
differently across containers.
"""

import os
from pathlib import Path

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import pytest  # noqa: E402

try:
    import torch  # noqa: E402  (imported after the env vars above on purpose)
except ImportError:  # the GPU-free contract suite has no torch to configure
    torch = None

from tests import require_public_cutedsl_library  # noqa: E402

if torch is not None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        # torch >= 2.9 precision API; "ieee" == full fp32 (no TF32).
        torch.backends.cuda.matmul.fp32_precision = "ieee"


# The two suites that assert on build inputs rather than running a kernel:
# tests/contract/ reads text, tests/cubin/ drives cpp/cmake/ by file path. Neither
# imports torch or bionemo_ir, so both must stay collectable with no extension
# built. Every other suite needs the launcher and has to be told when it is absent.
_TESTS_DIR = Path(__file__).resolve().parent
_LAUNCHER_FREE_DIRS = (_TESTS_DIR / "contract", _TESTS_DIR / "cubin")


def _is_launcher_free_only(config: pytest.Config) -> bool:
    """Whether every collection target lies inside a launcher-free suite."""
    targets = [Path(arg.partition("::")[0]).resolve() for arg in config.args]
    return bool(targets) and all(
        any(target == directory or directory in target.parents for directory in _LAUNCHER_FREE_DIRS)
        for target in targets
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Require the prebuilt CuTeDSL launcher before collecting public tests."""
    # Keyed on the suite, not on whether torch imported: a missing torch is why
    # the launcher check fails, so skipping the check on it would swallow the
    # one diagnostic a source-free checkout has.
    if _is_launcher_free_only(session.config):
        return
    require_public_cutedsl_library()
