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
import ctypes
import importlib.util
from pathlib import Path


def _load_cuequivariance_lib():
    CUEQUIV_PKG = importlib.util.find_spec("cuequivariance_ops")
    CUEQUIV_PKG_LIB = None
    handle = None
    if CUEQUIV_PKG is not None:
        CUEQUIV_PKG_LIB = Path(
            CUEQUIV_PKG.origin).parent.absolute() / "lib" / "libcue_ops.so"
        if CUEQUIV_PKG_LIB.exists():
            handle = ctypes.CDLL(CUEQUIV_PKG_LIB.as_posix())

    if handle is None:
        raise ImportError('CuEquivariance Ops library is unavailable')


__all__ = [
    "_load_cuequivariance_lib",
]
