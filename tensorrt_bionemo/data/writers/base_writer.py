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

from abc import ABC, abstractmethod

from tensorrt_bionemo.data.schemas.basic import (AtomType, AtomTypes,
                                                 FoldingOutput, ResType,
                                                 ResTypes)
from tensorrt_bionemo.logger import logger


class BaseWriter(ABC):
    """Writes a multi-chain protein structure to string and to file.

    Attributes:
        output_path (str): Path to output file created by write() method.
        res_type_mapping (dict[int, ResType]): Universe of existing residue types in the batch,
            represented as a map from integer to ResType
        atom_type_mapping (dict[int, AtomType]): Universe of existing atom types in the batch,
            represented as a map from integer to AtomType
        res_types (list[ResType]): Universe of existing residue types in the batch,
            represented as a 0-based list of ResType.  Assumes, res_type_mapping is 0-based.
        atom_types (list[AtomType]):  Universe of existing atom types in the batch,
            represented as a 0-based list of AtomType.  Assumes, atom_type_mapping is 0-based.
    """

    def __init__(
        self,
        res_type_mapping: dict[int, ResType],
        atom_type_mapping: dict[int, AtomType],
        output_path: str = "output.cif",
    ):
        """Initializes the BaseWriter with the given configuration..

        Args:
            res_type_mapping: Assigned to instance attribute.
            atom_type_mapping: Assigned to instance attribute.
            output_path: Assigned to instance attribute.
        Raises:
            None
        """
        logger.debug("BaseWriter.__init__() begin")

        if atom_type_mapping is None or res_type_mapping is None:
            raise ValueError("atom_type_mapping and res_type_mapping must be provided to dump PDB file")

        # process __init__ args
        self.output_path = output_path

        # Store the mappings as instance attributes
        self.res_type_mapping = res_type_mapping
        self.atom_type_mapping = atom_type_mapping

        # Build lists from mappings
        self.res_types = [
            y for _, y in sorted(self.res_type_mapping.items(),
                                 key=lambda pair: pair[0])
        ]
        self.atom_types = [
            y for _, y in sorted(self.atom_type_mapping.items(),
                                 key=lambda pair: pair[0])
        ]
        logger.debug("BaseWriter.__init__() end")

    def set_output_path(self, output_path: str):
        self.output_path = output_path

    @abstractmethod
    def write(self, folding_output: FoldingOutput) -> str:
        """Write the result of the network forward pass to file in the local
        environment.

        Args:
            folding_output: The result of the forward pass of a structure
                prediction network, e.g. OpenFold, or Boltz2
        """
