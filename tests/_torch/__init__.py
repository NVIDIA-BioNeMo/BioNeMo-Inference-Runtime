# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch

_CUTEDSL_SUPPORTED_SM = (80, 86, 89, 90)

SM_VERSION: int = (torch.cuda.get_device_capability()[0] * 10 +
                   torch.cuda.get_device_capability()[1]
                   if torch.cuda.is_available() else 0)

SKIP_CUTEDSL_REASON = (
    f"CuTeDSL kernel requires SM{'/'.join(str(s) for s in _CUTEDSL_SUPPORTED_SM)} "
    f"(current SM{SM_VERSION})")

skip_cutedsl = pytest.mark.skipif(
    SM_VERSION not in _CUTEDSL_SUPPORTED_SM,
    reason=SKIP_CUTEDSL_REASON,
)


def skip_if_no_cutedsl():
    """Call inside a test body to skip when CuTeDSL is unsupported."""
    if SM_VERSION not in _CUTEDSL_SUPPORTED_SM:
        pytest.skip(SKIP_CUTEDSL_REASON)


def skip_if_cutedsl(backend_name: str):
    """Skip if *backend_name* is ``"CuTeDSL"`` and the GPU doesn't support it."""
    if backend_name == "CuTeDSL" and SM_VERSION not in _CUTEDSL_SUPPORTED_SM:
        pytest.skip(SKIP_CUTEDSL_REASON)
