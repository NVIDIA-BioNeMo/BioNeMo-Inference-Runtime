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
"""Global pytest configuration for the TRT-BioNemo test suite.

Disable TF32 for fp32 math before any CUDA context / cuBLAS handle is created.
The numerical tests compare against IEEE fp32 references at tight tolerances
(atol ~1e-3, even ~1e-5), which TF32 cannot meet.

Two layers, set at conftest import time (pytest imports this before collecting or
running any test, i.e. before the first GPU op creates a cuBLAS handle):

1. ``NVIDIA_TF32_OVERRIDE=0`` -- the global CUDA kill switch, read by cuBLAS /
   cuDNN at handle creation. This is the reliable, torch-version-independent
   switch and, crucially, also disables TF32 inside the TensorRT engines built by
   the ``_trt`` tests, which torch's backend flags do not cover. Must be set
   *before* ``import torch`` / first CUDA use.
2. The torch backend flags -- explicit belt-and-suspenders for torch's own ops.

Individual tests also set these env vars inside their bodies, but that lands
after CUDA is initialized and is only honored on some torch versions (torch 2.10
/ nv25.12 honors it, torch 2.8 / nv25.08 does not -- causing CI to fail while
local passes).
"""

import os

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import torch  # noqa: E402  (imported after the env vars above on purpose)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
    # torch >= 2.9 precision API; "ieee" == full fp32 (no TF32).
    torch.backends.cuda.matmul.fp32_precision = "ieee"
