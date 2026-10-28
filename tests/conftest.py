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
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.common.test_utils.openfold3.batched_input_tools import CaptureInputs

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


# Build and CI tests need no launcher.
_TESTS_DIR = Path(__file__).resolve().parent
_LAUNCHER_FREE_DIRS = (
    _TESTS_DIR / "contract",
    _TESTS_DIR / "cubin",
    _TESTS_DIR.parent / ".gitlab" / "ci" / "scripts",
)


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


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # Keep Ray on one pytest worker.
    for item in items:
        if item.path.is_relative_to(_TESTS_DIR / "pipeline"):
            item.add_marker(pytest.mark.xdist_group("pipeline"))
        if "of3_capture_inputs" in item.fixturenames:
            item.add_marker(pytest.mark.xdist_group("of3-capture"))
        if item.path == _TESTS_DIR / "_torch" / "test_attn_epilogue.py":
            parameters = getattr(getattr(item, "callspec", None), "params", {})
            head_dim = parameters.get("head_dim", "default")
            bias = parameters.get("bias", False)
            # Gate, residual and fixed bias vary by test function.
            item.add_marker(pytest.mark.xdist_group(f"attn-epilogue-{item.originalname}-d{head_dim}-bias{bias}"))


@pytest.fixture(scope="session", autouse=True)
def _verify_repeated_captures(request: pytest.FixtureRequest) -> Iterator[None]:
    """Verify every repeated-region capture against eager, as serving no longer does.

    Session scope so session fixtures that build models get the same policy as tests.
    """
    if torch is None or _is_launcher_free_only(request.config):
        yield
        return
    from bionemo_ir._torch.graph_optimization import graph_policy

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(graph_policy, "_VERIFY_REPEATED_CAPTURES", True)
        yield


@pytest.fixture(scope="session")
def of3_capture_inputs() -> "CaptureInputs":
    from tests.common.test_utils.openfold3.batched_input_tools import cached_capture_inputs

    return cached_capture_inputs()


@pytest.fixture(scope="session")
def structural_model() -> Callable[..., object]:
    """Construct structures without compiling unused kernels."""
    from bionemo_ir._torch.custom_ops.attn_epilogue.cutedsl import AttnEpilogueCuTe
    from bionemo_ir.dsl_kernels.triton.fused_ln_proj_moveaxis_pad import FusedLNProjMoveaxisPad
    from bionemo_ir.dsl_kernels.triton.fused_swiglu import FusedSwiGLU
    from bionemo_ir.dsl_kernels.triton.ln_pair_bias import LNPairBias
    from bionemo_ir.dsl_kernels.triton.moveaxis_pad import MoveaxisPad
    from bionemo_ir.dsl_kernels.triton_cache import CachedKernel, TritonKernelCache

    def skip_compile(
        cache: TritonKernelCache,
        kernel: object,
        dtypes: Sequence[object],
        make_dummy_args: object,
        **kwargs: object,
    ) -> dict[object, CachedKernel]:
        return dict.fromkeys(dtypes, CachedKernel(None, enable_driver=False))

    def build[T](model_class: type[T], **kwargs: object) -> T:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("CUTEDSL_FORCE_CUBIN", "1")
            patch.setattr(AttnEpilogueCuTe, "_compiled_cache", {})
            patch.setattr(TritonKernelCache, "compile_for_dtypes", skip_compile)
            for backend in (FusedLNProjMoveaxisPad, FusedSwiGLU, LNPairBias, MoveaxisPad):
                patch.setattr(backend, "_global_cache", {})
            return model_class(**kwargs)

    return build
