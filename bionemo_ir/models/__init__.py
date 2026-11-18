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

"""Model classes, loaded on first access.

Each attribute imports only its own model package, so a process that uses one
model does not pay for the others.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .boltz1 import Boltz1
    from .boltz2 import Boltz2, Boltz2Affinity
    from .openfold2 import OpenFold2
    from .openfold3 import OpenFold3
    from .rfproteina import RFProteina

_EXPORTS = {
    "Boltz1": ".boltz1",
    "Boltz2": ".boltz2",
    "Boltz2Affinity": ".boltz2",
    "OpenFold2": ".openfold2",
    "OpenFold3": ".openfold3",
    "RFProteina": ".rfproteina",
}

__all__ = [
    "Boltz1",
    "Boltz2",
    "Boltz2Affinity",
    "OpenFold2",
    "OpenFold3",
    "RFProteina",
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
