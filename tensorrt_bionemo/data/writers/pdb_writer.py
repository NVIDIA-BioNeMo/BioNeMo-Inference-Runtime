# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import string

import numpy as np

from tensorrt_bionemo.data.schemas.basic import FoldingOutput
from tensorrt_bionemo.data.writers.base_writer import BaseWriter

PICO_TO_ANGSTROM = 0.01

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)


class PDBWriter(BaseWriter):
    """Writes a multi-chain protein structure to PDB format.

    Inherits from BaseWriter which provides default mappings for residue types
    and atom types if not specified.
    """

    def get_pdb_headers(self) -> list[str]:
        pdb_headers = []
        parents = ["N/A"]
        pdb_headers.append(f"PARENT {' '.join(parents)}")

        return pdb_headers

    def _chain_end(self, atom_index: int, end_resname: str, chain_name: str,
                   residue_index: int) -> str:
        chain_end = 'TER'
        return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
                f'{chain_name:>1}{residue_index:>4}')

    def write(self, folding_output: FoldingOutput):

        pdb_lines = []

        # get output protein complex representation
        #   - for numpy arrays below, 0th axis is sequence position
        residue_types: np.ndarray[np.int64] = folding_output["residue_types"]
        atom_positions: np.ndarray = folding_output[
            "atom_positions"]  # shape: (n_res, n_atoms, 3)
        atom_mask: np.ndarray[np.float32] = folding_output["atom_mask"]
        chain_indices: np.ndarray[np.int64] = folding_output["chain_indices"]
        residue_indices: np.ndarray[np.int64] = folding_output[
            "residue_indices"]  # 1-based
        b_factors: np.ndarray[np.float32] = folding_output["b_factors"]

        # Construct a mapping from chain integer indices to chain ID strings.
        chain_ids = {}
        for i in np.unique(chain_indices):  # np.unique gives sorted output.
            if i >= PDB_MAX_CHAINS:
                raise ValueError(
                    f"The PDB format supports at most {PDB_MAX_CHAINS} chains."
                )
            chain_ids[i] = PDB_CHAIN_IDS[i]

        headers = self.get_pdb_headers()
        if (len(headers) > 0):
            pdb_lines.extend(headers)

        pdb_lines.append("MODEL     1")
        n = residue_types.shape[0]
        atom_index = 1
        last_chain_index = chain_indices[0]
        prev_chain_index = 0
        chain_tags = string.ascii_uppercase

        # Add all atom sites.
        for i in range(residue_types.shape[0]):
            # Close the previous chain if in a multichain PDB.
            if last_chain_index != chain_indices[i]:
                pdb_lines.append(
                    self._chain_end(
                        atom_index,
                        self.res_type_mapping[residue_types[i -
                                                            1]].canonical_name,
                        chain_ids[chain_indices[i - 1]],
                        residue_indices[i - 1]))
                last_chain_index = chain_indices[i]
                atom_index += 1  # Atom index increases at the TER symbol.

            res_name_3 = self.res_type_mapping[residue_types[i]].canonical_name

            for atom_type, pos, mask, b_factor in zip(self.atom_types,
                                                      atom_positions[i],
                                                      atom_mask[i],
                                                      b_factors[i]):
                atom_name = atom_type.name
                if mask < 0.5:
                    continue

                record_type = "ATOM"
                name = atom_name if len(atom_name) == 4 else f" {atom_name}"
                alt_loc = ""
                insertion_code = ""
                occupancy = 1.00
                # Protein supports only C, N, O, S, this works.
                element = atom_name[0]
                charge = ""

                chain_tag = "A"
                if chain_indices is not None:
                    chain_tag = chain_tags[chain_indices[i]]

                # PDB is a columnar format, every space matters here!
                atom_line = (
                    f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                    #TODO: check this refactor, chose main branch version
                    #f"{res_name_3:>3} {chain_ids[chain_indices[i]]:>1}"
                    f"{res_name_3:>3} {chain_tag:>1}"
                    f"{residue_indices[i]:>4}{insertion_code:>1}   "
                    f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                    f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                    f"{element:>2}{charge:>2}")

                pdb_lines.append(atom_line)
                atom_index += 1
            should_terminate = (i == n - 1)
            if chain_indices is not None:
                if (i != n - 1 and chain_indices[i + 1] != prev_chain_index):
                    should_terminate = True
                    prev_chain_index = chain_indices[i + 1]

            if should_terminate:
                # Close the chain.
                res_type = self.res_type_mapping[residue_types[i]]
                chain_end = "TER"
                chain_termination_line = (
                    f"{chain_end:<6}{atom_index:>5}      "
                    f"{res_type.canonical_name:>3} "
                    f"{chain_tag:>1}{residue_indices[i]:>4}")
                pdb_lines.append(chain_termination_line)
                atom_index += 1

                if (i != n - 1):
                    # "prev" is a misnomer here. This happens at the beginning of
                    # each new chain.
                    pdb_lines.extend(self.get_pdb_headers())

        pdb_lines.append("ENDMDL")
        pdb_lines.append("END")

        # Pad all lines to 80 characters
        pdb_lines = [line.ljust(80) for line in pdb_lines]
        buffer = '\n'.join(pdb_lines) + '\n'  # Add terminating newline.

        if self.output_path is not None:
            with open(self.output_path, "w") as f:
                f.write(buffer)
        return buffer
