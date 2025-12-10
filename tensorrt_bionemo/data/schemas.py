# Copyright (c) 2024 Chai Discovery, Inc.
# Copyright (c) 2025 NVIDIA Corporation.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.
from pydantic import BaseModel

from .residue_constants import ResType


class EntityType(Enum):
    PROTEIN = 0
    RNA = 1
    DNA = 2
    LIGAND = 3
    POLYMER_HYBRID = 4
    WATER = 5
    UNKNOWN = 6
    MANUAL_GLYCAN = 7  # NOTE glycan parsing


class Sequence(BaseModel):
    residues: list[ResType]
    description: Optional[str] = None


class InputChain(BaseModel):
    entity_type: EntityType
    sequence: Sequence
    entity_name: Optional[str] = None
