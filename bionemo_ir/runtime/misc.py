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

import torch

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart


def query_sm_count(device: int | None = None, default: int = 132) -> int:
    """Query the multiprocessor (SM) count of a CUDA device.

    Falls back to ``default`` if the query fails for any reason (e.g. no CUDA
    device available or a driver error).

    Args:
        device: CUDA device ordinal. Defaults to the current device.
        default: Value returned if the SM count cannot be queried.
    """
    try:
        if device is None:
            device = torch.cuda.current_device()
        err, sm_count = cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrMultiProcessorCount, device)
        if err != cudart.cudaError_t.cudaSuccess:
            return default
        return sm_count
    except Exception:
        return default
