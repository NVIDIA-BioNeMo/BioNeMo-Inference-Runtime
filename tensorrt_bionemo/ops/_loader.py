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
from pathlib import Path
from typing import Any

_LIBS_DIR = Path(__file__).parent.parent / "libs"


def load_extension() -> Any:
    """Load the compiled extension."""
    # Find .so file
    so_files = list(_LIBS_DIR.glob("lib_C*.so"))

    if not so_files:
        raise ImportError(f"Extension not found in {_LIBS_DIR}.\n"
                          "Build with: pip install -e .")

    so_path = so_files[0]

    if so_path.exists():
        handle = ctypes.CDLL(so_path.as_posix())
    else:
        raise ImportError(f"Failed to load {so_path}")

    return handle
