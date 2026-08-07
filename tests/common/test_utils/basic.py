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
from pathlib import Path
from types import ModuleType
from typing import Any

from torch import nn


def setattr_safe(module: nn.Module, attr_name: str, attr_value: Any):
    if hasattr(module, attr_name):
        setattr(module, attr_name, attr_value)
    else:
        raise AttributeError(f"module {module} does not have attribute {attr_name}")


def path_for_package_in_repo(package: ModuleType) -> Path:
    return Path(package.__file__).parent
