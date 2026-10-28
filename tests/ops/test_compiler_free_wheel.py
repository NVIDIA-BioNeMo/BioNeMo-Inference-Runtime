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

"""Exercise an installed wheel without compiler dependencies.

Set ``BIOIR_TEST_COMPILER_FREE_WHEEL=1`` in a clean wheel environment.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid._config import SUPPORTED_SMS
from tests._torch import SM_VERSION

_WHEEL_PROBE = """
import importlib.util
import os
import sys
from importlib.metadata import distribution
from pathlib import Path

assert "CUTEDSL_FORCE_CUBIN" not in os.environ
assert "BIOIR_TEST_CUTEDSL_MODES" not in os.environ
for name in ("cutlass", "tvm_ffi"):
    assert importlib.util.find_spec(name) is None, name
    assert name not in sys.modules, name

import torch
import bionemo_ir
from bionemo_ir._torch.attention_backend.pairwise_attention import get_nearest_bucket as pairwise_bucket
from bionemo_ir._torch.attention_backend.triangle_attention import get_nearest_bucket as triangle_bucket
from bionemo_ir._torch.custom_ops.adaln_layernorm_sigmoid import (
    AdaLNLayerNormSigmoidCuTe,
    get_adaln_layernorm_sigmoid_op,
)
from bionemo_ir._torch.utils.kernel import CuTeDSLKernelLibraryExecutable

installed = distribution("bionemo-ir").locate_file("bionemo_ir/__init__.py")
assert Path(bionemo_ir.__file__).resolve() == Path(installed).resolve()

assert pairwise_bucket(90, 64, 0) == 0
assert triangle_bucket(90, 64, 0) == 0

torch.manual_seed(41)
for dtype, tolerance in (
    (torch.float16, 0.002),
    (torch.bfloat16, 0.05),
    (torch.float32, 0.00002),
):
    operation = get_adaln_layernorm_sigmoid_op(dtype, 128)
    assert isinstance(operation, AdaLNLayerNormSigmoidCuTe)
    x = torch.randn(33, 128, device="cuda", dtype=dtype)
    scale = torch.randn_like(x)
    bias = torch.randn_like(x)
    expected = (
        torch.sigmoid(scale.float())
        * torch.nn.functional.layer_norm(x.float(), (128,), eps=1e-5)
        + bias.float()
    ).to(dtype)
    actual = operation(x, scale, bias)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert operation._compiled_cache
    assert all(
        isinstance(executable, CuTeDSLKernelLibraryExecutable)
        for executable in operation._compiled_cache.values()
    )

assert not any(name.partition(".")[0] in ("cutlass", "tvm_ffi") for name in sys.modules)
print("Compiler-free wheel runtime passed")
"""


@pytest.mark.skipif(
    os.getenv("BIOIR_TEST_COMPILER_FREE_WHEEL") != "1",
    reason="set BIOIR_TEST_COMPILER_FREE_WHEEL=1 for clean-wheel checks",
)
@pytest.mark.skipif(SM_VERSION not in SUPPORTED_SMS, reason="requires a supported CUDA GPU")
def test_installed_wheel_without_compiler(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("CUTEDSL_FORCE_CUBIN", None)
    environment.pop("BIOIR_TEST_CUTEDSL_MODES", None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", _WHEEL_PROBE],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
