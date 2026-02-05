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

import sys
import importlib.util
from pathlib import Path
from typing import Any


def load_extension() -> Any:
    """Load the compiled extension module from libs/."""
    # libs/ is at tensorrt_bionemo/libs/, one level up from ops/
    package_dir = Path(__file__).parent.parent
    libs_dir = package_dir / "libs"
    
    if not libs_dir.exists():
        raise ImportError(
            f"libs directory not found: {libs_dir}\n"
            "Please build the extension: RECOMPILE_CPP=1 pip install -e ."
        )
    
    # Find the .so file
    so_files = list(libs_dir.glob("trt_bnm_ops*.so"))
    
    if not so_files:
        # Try alternative patterns
        so_files = list(libs_dir.glob("*.so"))
    
    if not so_files:
        raise ImportError(
            f"Extension .so not found in {libs_dir}\n"
            "Please build the extension: RECOMPILE_CPP=1 pip install -e ."
        )
    
    so_path = so_files[0]
    
    # Extract module name from filename
    # e.g., "trt_bnm_ops.cpython-312-x86_64-linux-gnu.so" → "trt_bnm_ops"
    module_name = so_path.name.split('.')[0]
    
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {so_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return module