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

import io
import string

import modelcif
import numpy as np
from modelcif import dumper, model, qa_metric  # noqa: F401

from tensorrt_bionemo.data.schemas.basic import FoldingOutput
from tensorrt_bionemo.data.writers.base_writer import BaseWriter


class CIFWriter(BaseWriter):
    """Writes a multi-chain protein structure to string and to file in cif format."""

    def set_output_path(self, output_path: str):
        self.output_path = output_path

    def write(self,
              folding_output: FoldingOutput,
              system_title: str | None = "TensorRT BioNeMo Prediction") -> str:
        """Write the result of the network forward pass to file in the local
        environment.

        Args:
            folding_output: The result of the forward pass of a structure
                prediction network, e.g. OpenFold, or Boltz2
            system_title: Written to the output cif-format file.

        Notes:
            from Protein dataclass

            # Amino-acid type for each residue represented as an integer between 0 and
            # 20, where 20 is 'X'.
            aatype: np.ndarray  # [num_res]


            (1) Copy-paste from openfold numpy-representation of a protein chain
            https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/openfold/np/protein.py

            (2) Commented these lines from to_modelcif(..)

            # restypes = residue_constants.restypes + ["X"]
            # atom_types = residue_constants.atom_types

            # atom_mask = prot.atom_mask
            # aatype = prot.aatype
            # atom_positions = prot.atom_positions
            # residue_index = prot.residue_index.astype(np.int32)
            # b_factors = prot.b_factors
            # chain_index = prot.chain_index

            (3) insert statements to get protein structure representation from folding_output
            residue_types: np.ndarray = folding_output["residue_types"]
            atom_positions: np.ndarray = folding_output["atom_positions"]
            atom_mask: np.ndarray = folding_output["atom_mask"]

            chain_indices: np.ndarray = folding_output["chain_indices"]
            residue_indices: np.ndarray = folding_output["residue_indices"]
            b_factors: np.ndarray = folding_output["b_factors"]

            (4) find-replace
            chain_index --> chain_indices
            residue_index --> residue_indices
            atom_types --> self.atom_types
            aatype --> residue_types

            (5) set restypes to single-char name from self.res_types
            (6) set atom_types to the name field of self.atom_types

        """
        # get output protein complex representation
        #   - for numpy arrays below, 0th axis is sequence position
        residue_types: np.ndarray[np.int64] = folding_output["residue_types"]
        atom_positions: np.ndarray = folding_output[
            "atom_positions"]  # shape: (n_res, n_atoms, 3)
        atom_mask: np.ndarray[np.float32] = folding_output["atom_mask"]

        chain_indices: np.ndarray[
            np.int64] | None = folding_output["chain_indices"]
        residue_indices: np.ndarray[np.int64] = folding_output[
            "residue_indices"]  # 1-based
        b_factors: np.ndarray[np.float32] = folding_output["b_factors"]

        # put residue universe in needed format
        #   - below seqs is map from integer to list of strings
        restypes: list[str] = [x.name for x in self.res_types]
        atom_types: list[str] = [x.name for x in self.atom_types]

        # sanity checks on folding_output
        if chain_indices is not None and not isinstance(
                chain_indices, np.ndarray):
            raise TypeError(
                "In the CIFWriter, received folding_output chain_indices is neither None or an array."
            )
        elif isinstance(chain_indices,
                        np.ndarray) and (min(chain_indices) < 0
                                         or max(chain_indices) > 25):
            raise ValueError(" ".join([
                "In the CIFWriter, received folding_output chain_indices has at least",
                "one value less than 0 or greater than 25."
            ]))

        # start openfold logic
        n = residue_types.shape[0]  # number of sequence positions in input
        if chain_indices is None:
            chain_indices = [0 for i in range(n)]

        system = modelcif.System(title=system_title)

        # Finding chains and creating entities.
        #
        # `residue_indices` is PDB-style numbering (per FoldingOutput's
        # docstring: "not necessarily continuous or 0-indexed"), so values
        # may be negative, non-contiguous, or duplicated across chains.
        # ihm.Entity however expects 1-indexed positions in 1..len(entity).
        # Build a (chain_idx, residue_idx) -> local 1-indexed seq_id map in
        # residue-encounter order so later calls to .residue(...) and atom
        # seq_id always match the entity's 1..N range.
        seqs = {}
        local_seq_id: dict[tuple[int, int], int] = {}
        chain_pos_counter: dict[int, int] = {}
        seq = []
        last_chain_idx = None
        for i in range(n):
            c = int(chain_indices[i])
            r = int(residue_indices[i])
            if last_chain_idx is not None and last_chain_idx != c:
                seqs[last_chain_idx] = seq
                seq = []
            seq.append(restypes[residue_types[i]])
            if (c, r) not in local_seq_id:
                chain_pos_counter[c] = chain_pos_counter.get(c, 0) + 1
                local_seq_id[(c, r)] = chain_pos_counter[c]
            last_chain_idx = c
        # finally add the last chain
        seqs[last_chain_idx] = seq

        # Remap internal canonical_name codes to CIF/IHM standard codes:
        # 'X' → 'UNK' (ihm uses 'UNK' for unknown protein residues).
        # 'RX' and 'DX' are the canonical_name values of the RNA/DNA unknown
        # entries in the ResTypes enum (see tensorrt_bionemo/data/schemas/
        # basic.py); map them to the CIF standard codes 'N' and 'DN'.
        _IHM_REMAP = {'X': 'UNK', 'RX': 'N', 'DX': 'DN'}

        # now reduce sequences to unique ones (note this won't work if different asyms have different unmodelled regions)
        unique_seqs = {}
        for chain_idx, seq_list in seqs.items():
            seq = tuple(seq_list)
            if seq in unique_seqs:
                unique_seqs[seq].append(chain_idx)
            else:
                unique_seqs[seq] = [chain_idx]

        # adding 1 entity per unique sequence
        entities_map = {}
        for key, value in unique_seqs.items():
            ihm_seq = [_IHM_REMAP.get(r, r) for r in key]
            model_e = modelcif.Entity(ihm_seq, description='Model subunit')
            for chain_idx in value:
                entities_map[chain_idx] = model_e

        chain_tags = string.ascii_uppercase
        asym_unit_map = {}
        for chain_idx in set(chain_indices):
            # Define the model assembly
            chain_id = chain_tags[chain_idx]
            asym = modelcif.AsymUnit(entities_map[chain_idx],
                                     details='Model subunit %s' % chain_id,
                                     id=chain_id)
            asym_unit_map[chain_idx] = asym
        modeled_assembly = modelcif.Assembly(asym_unit_map.values(),
                                             name='Modeled assembly')

        class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
            name = "pLDDT"
            software = None
            description = "Predicted lddt"

        class _GlobalPLDDT(modelcif.qa_metric.Global,
                           modelcif.qa_metric.PLDDT):
            name = "pLDDT"
            software = None
            description = "Global pLDDT, mean of per-residue pLDDTs"

        class _MyModel(modelcif.model.AbInitioModel):

            def get_atoms(self):
                # Add all atom sites.
                for i in range(n):
                    for atom_name, pos, mask, b_factor in zip(
                            atom_types, atom_positions[i], atom_mask[i],
                            b_factors[i]):
                        if mask < 0.5:
                            continue
                        element = atom_name[
                            0]  # Protein supports only C, N, O, S, this works.
                        yield modelcif.model.Atom(
                            asym_unit=asym_unit_map[chain_indices[i]],
                            type_symbol=element,
                            seq_id=local_seq_id[(int(chain_indices[i]),
                                                 int(residue_indices[i]))],
                            atom_id=atom_name,
                            x=pos[0],
                            y=pos[1],
                            z=pos[2],
                            het=False,
                            biso=b_factor,
                            occupancy=1.00)

            def add_scores(self):
                # local scores
                plddt_per_residue = {}
                for i in range(n):
                    for mask, b_factor in zip(atom_mask[i], b_factors[i]):
                        if mask < 0.5:
                            continue
                        # add 1 per residue, not 1 per atom
                        if chain_indices[i] not in plddt_per_residue:
                            # first time a chain index is seen: add the key and start the residue dict
                            plddt_per_residue[chain_indices[i]] = {
                                residue_indices[i]: b_factor
                            }
                        if residue_indices[i] not in plddt_per_residue[
                                chain_indices[i]]:
                            plddt_per_residue[chain_indices[i]][
                                residue_indices[i]] = b_factor
                plddts = []
                for chain_idx in plddt_per_residue:
                    for residue_idx in plddt_per_residue[chain_idx]:
                        plddt = plddt_per_residue[chain_idx][residue_idx]
                        plddts.append(plddt)
                        self.qa_metrics.append(
                            _LocalPLDDT(
                                asym_unit_map[chain_idx].residue(
                                    local_seq_id[(int(chain_idx),
                                                  int(residue_idx))]),
                                plddt))
                # global score
                self.qa_metrics.append(
                    (_GlobalPLDDT(np.mean(np.array(plddts)))))

        # Add the model and modeling protocol to the file and write them out:
        model_ = _MyModel(assembly=modeled_assembly, name='Best scoring model')
        model_.add_scores()

        model_group = modelcif.model.ModelGroup([model_], name='All models')
        system.model_groups.append(model_group)

        fh = io.StringIO()
        modelcif.dumper.write(fh, [system])
        buffer: str = fh.getvalue()

        if self.output_path is not None:
            with open(self.output_path, "w") as f:
                f.write(buffer)
        return buffer
