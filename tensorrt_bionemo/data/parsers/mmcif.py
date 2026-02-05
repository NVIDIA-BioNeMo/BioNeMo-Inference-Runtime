# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Optional, TextIO

from Bio import PDB
from tensorrt_llm_lite import logger

from tensorrt_bionemo.data.schemas.basic import ResTypes, StructureMetadata

MmcifRaw = dict[str, Any]
MmcifChain = list[tuple[str, int]]


@dataclass
class AtomSite:
    residue_name: str
    author_chain_id: str
    mmcif_chain_id: str
    author_seq_num: str
    mmcif_seq_num: int
    insertion_code: str
    hetatm_atom: str
    model_num: int


@dataclass
class StructuralResidue:
    name: str
    is_missing: bool
    hetflag: Optional[str] = None
    chain_id: Optional[str] = None
    author_index: Optional[int] = None
    insertion_code: Optional[str] = None


class MmcifParsed:

    def __init__(self,
                 parsed_info: MmcifRaw,
                 structure: Optional[PDB.Structure.Structure] = None,
                 file_id: Optional[str] = None):
        self.parsed_info = parsed_info
        self.file_id = file_id
        self.structure: Optional[PDB.Structure.Structure] = structure
        self.structure_metadata: StructureMetadata = None
        self.polymers: dict[str, MmcifChain] = {}
        self.protein_chains: dict[str, MmcifChain] = {}
        self.entity_to_chains: dict[str, list[str]] = {}
        self.seq_to_structure_mappings: dict[str,
                                             dict[int,
                                                  StructuralResidue]] = {}
        self.author_chain_to_sequence: dict[str, str] = {}
        self._extract()

    def extract(self) -> 'MmcifParsed':
        self._get_structure_metadata()
        self._get_polymers()
        self._get_entity_to_chains()
        self._get_protein_chains()
        self._map_mmcif_sequence_to_structure()
        return self

    def _get_structure_metadata(self) -> None:
        method = ",".join(self.parsed_info["_exptl.method"])
        release_date = None
        if "_pdbx_audit_revision_history.revision_date" in self.parsed_info:
            release_date = min(
                self.parsed_info["_pdbx_audit_revision_history.revision_date"])
        else:
            logger.warning("Could not determine release_date: %s",
                           self.parsed_info["_entry.id"])
        resolution = 0.0
        for res_key in (
                "_refine.ls_d_res_high",
                "_em_3d_reconstruction.resolution",
                "_reflns.d_resolution_high",
        ):
            if res_key in self.parsed_info:
                try:
                    raw_resolution = self.parsed_info[res_key][0]
                    resolution = float(raw_resolution)
                    break
                except ValueError:
                    logger.debug("Invalid resolution format: %s",
                                 self.parsed_info[res_key])
        self.structure_metadata = StructureMetadata(
            resolution=resolution,
            release_date=release_date,
            method=method,
            file_id=self.file_id,
        )

    def _get_polymers(self) -> None:
        entity_ids = self.parsed_info["_entity_poly_seq.entity_id"]
        unique_entity_ids = set(entity_ids)
        mon_ids = self.parsed_info["_entity_poly_seq.mon_id"]
        nums = self.parsed_info["_entity_poly_seq.num"]

        for unique_entity_id in unique_entity_ids:
            self.polymers[unique_entity_id] = []

        for entity_id, mon_id, num in zip(entity_ids, mon_ids, nums):
            self.polymers[entity_id].append(
                (ResTypes.from_string(mon_id, construct=True), int(num)))

    def _get_entity_to_chains(self) -> None:
        struct_asym_ids = self.parsed_info["_struct_asym.id"]
        struct_asym_entity_ids = self.parsed_info["_struct_asym.entity_id"]
        unique_entity_ids = set(struct_asym_entity_ids)
        for entity_id in unique_entity_ids:
            self.entity_to_chains[entity_id] = []
        for entity_id, struct_asym_id in zip(struct_asym_entity_ids,
                                             struct_asym_ids):
            self.entity_to_chains[entity_id].append(struct_asym_id)

    def _get_atom_site_list(self) -> list[AtomSite]:
        """Returns list of atom sites; contains data not present in the structure."""
        return [
            AtomSite(*site) for site in zip(  # pylint:disable=g-complex-comprehension
                self.parsed_info["_atom_site.label_comp_id"],
                self.parsed_info["_atom_site.auth_asym_id"],
                self.parsed_info["_atom_site.label_asym_id"],
                self.parsed_info["_atom_site.auth_seq_id"],
                self.parsed_info["_atom_site.label_seq_id"],
                self.parsed_info["_atom_site.pdbx_PDB_ins_code"],
                self.parsed_info["_atom_site.group_PDB"],
                self.parsed_info["_atom_site.pdbx_PDB_model_num"],
            )
        ]

    def _get_protein_chains(self) -> None:
        """ Get protein chains from polymers and entity_to_chains. """
        for entity_id, seq_info in self.polymers.items():
            chain_ids = self.entity_to_chains[entity_id]

            # Reject polymers without any peptide-like components, such as DNA/RNA.
            if any([ResTypes.is_peptide(res_name)
                    for res_name, _ in seq_info]):
                for chain_id in chain_ids:
                    self.protein_chains[chain_id] = seq_info

    def _map_mmcif_sequence_to_structure(self):
        """
        Maps residues in the mmCIF polymer sequence (_entity_poly_seq) to actual residues observed in the 3D structure (_atom_site),
        for protein chains only, using model 1, while respecting Biopython residue conventions.
        """
        atom_sites = self._get_atom_site_list()
        mmcif_to_author_chain_id = {}

        is_set = lambda data: data not in (".", "?")

        seq_start_num = {
            chain_id: min([x[1] for x in seq])
            for chain_id, seq in self.protein_chains.items()
        }

        seq_to_structure_mappings = {}

        for atom in atom_sites:
            if atom.model_num != "1":
                # We only process the first model at the moment.
                continue
            mmcif_to_author_chain_id[
                atom.mmcif_chain_id] = atom.author_chain_id

            if atom.mmcif_chain_id in self.protein_chains:
                hetflag = " "
                if atom.hetatm_atom == "HETATM":
                    # Water atoms are assigned a special hetflag of W in Biopython. We
                    # need to do the same, so that this hetflag can be used to fetch
                    # a residue from the Biopython structure by id.
                    if atom.residue_name in ("HOH", "WAT"):
                        hetflag = "W"
                    else:
                        hetflag = "H_" + atom.residue_name
                    if not is_set(atom.insertion_code):
                        insertion_code = " "

                    residue_idx = (int(atom.mmcif_seq_num) -
                                   seq_start_num[atom.mmcif_chain_id])
                    current_chain_mappings = seq_to_structure_mappings.get(
                        atom.author_chain_id, {})
                    if not residue_idx in current_chain_mappings:
                        current_chain_mappings[
                            residue_idx] = StructuralResidue(
                                chain_id=atom.author_chain_id,
                                author_index=int(atom.author_seq_num),
                                insertion_code=insertion_code,
                                name=atom.residue_name,
                                is_missing=False,
                                hetflag=hetflag)
                    if not atom.author_chain_id in seq_to_structure_mappings:
                        seq_to_structure_mappings[
                            atom.author_chain_id] = current_chain_mappings

        # Add missing residue information to seq_to_structure_mappings.
        for chain_id, seq_info in self.protein_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            current_mapping = seq_to_structure_mappings[author_chain]
            for idx, (res_name, _) in enumerate(seq_info):
                if idx not in current_mapping:
                    current_mapping[idx] = StructuralResidue(name=res_name,
                                                             is_missing=True,
                                                             hetflag=" ")

        author_chain_to_sequence = {}
        for chain_id, seq_info in self.protein_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            residues = []
            for res_name, _ in seq_info:
                # This is same PDBData.protein_letters_3to1_extended
                # res_name can be 1 or 3 characters long
                residue = ResTypes.from_string(res_name, return_unknown=True)
                residues.append(residue.name)
            author_chain_to_sequence[author_chain] = "".join(residues)

        self.seq_to_structure_mappings = seq_to_structure_mappings
        self.author_chain_to_sequence = author_chain_to_sequence


def parse_mmcif_content(content: StringIO | TextIO,
                        file_id: Optional[str] = None) -> MmcifParsed:
    parser = PDB.MMCIFParser(QUIET=True)
    handle = StringIO(content)
    full_structure = parser.get_structure("", handle)
    first_model_structure = next(full_structure.get_models())
    parsed_info = parser._mmcif_dict

    # Ensure all values are lists, even if singletons.
    for key, value in parsed_info.items():
        if not isinstance(value, list):
            parsed_info[key] = [value]

    ret = MmcifParsed(parsed_info=parsed_info,
                      structure=first_model_structure).extract()
    return ret


def read_mmcif(file_path: str | Path,
               file_id: Optional[str] = None) -> MmcifParsed:
    if file_id is None:
        file_id = Path(file_path).stem
    with open(file_path, "r") as file:
        return parse_mmcif_content(file.read(), file_id=file_id)
