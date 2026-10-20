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
"""Custom structure-oracle contracts, providers, and index resolvers."""

from bionemo_ir.data.schemas import (
    AtomIdentity,
    EncodedStructure,
    OracleRequest,
    OracleResponse,
    OracleStatus,
    ResidueIdentity,
    ResolvedStructure,
)

from .providers import CallableOracleProvider, OracleCallable, OracleProvider
from .resolvers import (
    AtomTypeResolver,
    CallableAtomTypeResolver,
    CallableResidueTypeResolver,
    MappingAtomTypeResolver,
    MappingResidueTypeResolver,
    OracleResolutionError,
    ResidueTypeResolver,
    resolve_structure,
)

__all__ = [
    "AtomIdentity",
    "AtomTypeResolver",
    "CallableAtomTypeResolver",
    "CallableOracleProvider",
    "CallableResidueTypeResolver",
    "EncodedStructure",
    "MappingAtomTypeResolver",
    "MappingResidueTypeResolver",
    "OracleCallable",
    "OracleProvider",
    "OracleRequest",
    "OracleResolutionError",
    "OracleResponse",
    "OracleStatus",
    "ResidueIdentity",
    "ResidueTypeResolver",
    "ResolvedStructure",
    "resolve_structure",
]
