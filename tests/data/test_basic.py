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

import numpy as np
import pytest

from bionemo_ir.data.parsers import read_a3m
from bionemo_ir.data.parsers.fasta import read_fasta
from bionemo_ir.data.schemas.basic import (
    AtomTypes,
    FoldingOutput,
    InputRequest,
    MSARecord,
    Polymer,
    PolymerType,
    ResTypes,
    Template,
)

SAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "data" / "samples" / "monomers"
FASTA_FILES = ["T1031.fasta", "T1033.fasta", "T1047s1.fasta", "T1094.fasta"]


class TestPolymer:
    def test_create_molecule_from_fasta(self):
        fasta_path = SAMPLES_DIR / "T1031.fasta"
        parsed = read_fasta(fasta_path)
        molecule = parsed["sequences"][0]
        assert molecule["sequence"] is not None
        assert len(molecule["sequence"]) == 95
        assert "T1031" in parsed["descriptions"][0]

    @pytest.mark.parametrize("fasta_file", FASTA_FILES)
    def test_all_fasta_files_readable(self, fasta_file):
        fasta_path = SAMPLES_DIR / fasta_file
        parsed = read_fasta(fasta_path)
        assert "sequences" in parsed
        assert len(parsed["sequences"]) >= 1
        molecule = parsed["sequences"][0]
        assert molecule["sequence"] is not None

    def test_molecule_from_fasta_has_correct_fields(self):
        fasta_path = SAMPLES_DIR / "T1033.fasta"
        parsed = read_fasta(fasta_path)
        molecule = parsed["sequences"][0]
        assert molecule["chain_id"] == "A"
        assert molecule["polymer_type"] == PolymerType.PROTEIN.value
        assert len(molecule["sequence"]) == 100

    def test_manual_molecule_creation(self):
        molecule = Polymer(polymer_type=PolymerType.PROTEIN, chain_id="B", sequence="ACDEFGHIKLMNPQRSTVWY")
        assert molecule["chain_id"] == "B"
        assert molecule["polymer_type"] == PolymerType.PROTEIN.value
        assert molecule["sequence"] == "ACDEFGHIKLMNPQRSTVWY"

    def test_molecule_validation_requires_sequence_for_protein(self):
        with pytest.raises(ValueError):
            Polymer(polymer_type=PolymerType.PROTEIN, chain_id="A")

    def test_ccd_ligand_requires_sequence(self):
        with pytest.raises(ValueError):
            Polymer(polymer_type=PolymerType.CCD_LIGAND, chain_id="L")

    def test_smiles_ligand_requires_sequence(self):
        with pytest.raises(ValueError):
            Polymer(polymer_type=PolymerType.SMILES_LIGAND, chain_id="L")

    def test_ccd_ligand_create(self):
        ligand = Polymer(polymer_type=PolymerType.CCD_LIGAND, chain_id="L", sequence="ATP")
        assert ligand["polymer_type"] == PolymerType.CCD_LIGAND.value
        assert ligand["sequence"] == "ATP"

    def test_ccd_ligand_create_multi_component(self):
        ligand = Polymer(polymer_type=PolymerType.CCD_LIGAND, chain_id="L", sequence="ATP_FAD")
        assert ligand["sequence"] == "ATP_FAD"
        assert ligand["sequence"].split("_") == ["ATP", "FAD"]

    @pytest.mark.parametrize(
        "bad_sequence",
        [
            "atp",
            "ATP_",
            "_ATP",
            "ATP__FAD",
            "ATP-FAD",
            "ATP FAD",
            "TOOLONG",
            "",
        ],
    )
    def test_ccd_ligand_rejects_invalid_sequence(self, bad_sequence):
        with pytest.raises(ValueError):
            Polymer(polymer_type=PolymerType.CCD_LIGAND, chain_id="L", sequence=bad_sequence)

    def test_smiles_ligand_create(self):
        ligand = Polymer(polymer_type=PolymerType.SMILES_LIGAND, chain_id="L", sequence="CCO")
        assert ligand["polymer_type"] == PolymerType.SMILES_LIGAND.value
        assert ligand["sequence"] == "CCO"

    def test_ligand_types_reject_templates(self):
        tmpl = Template(content="data_x\n", format="cif")
        with pytest.raises(ValueError):
            Polymer(polymer_type=PolymerType.CCD_LIGAND, chain_id="L", sequence="ATP", templates=[tmpl])
        with pytest.raises(ValueError):
            Polymer(polymer_type=PolymerType.SMILES_LIGAND, chain_id="L", sequence="CCO", templates=[tmpl])


class TestResTypes:
    def test_parse_residues_from_sample(self):
        fasta_path = SAMPLES_DIR / "T1031.fasta"
        parsed = read_fasta(fasta_path)
        sequence = parsed["sequences"][0]["sequence"]
        for residue in sequence:
            res_type = ResTypes.from_string(residue, return_unknown=True)
            assert res_type is not None
            assert res_type.name in "ARNDCQEGHILKMFPSTWYVX"

    def test_basic_20_residue_types(self):
        basic_20 = ResTypes.basic_20_residue_types()
        assert len(basic_20) == 20
        expected = set("ARNDCQEGHILKMFPSTWYV")
        actual = {r.name for r in basic_20}
        assert expected == actual

    def test_is_peptide(self):
        assert ResTypes.is_peptide(ResTypes.A) is True
        assert ResTypes.is_peptide(ResTypes.RA) is False
        assert ResTypes.is_peptide(ResTypes.DA) is False

    def test_is_nucleotide(self):
        assert ResTypes.is_nucleotide(ResTypes.RA) is True
        assert ResTypes.is_nucleotide(ResTypes.DA) is True
        assert ResTypes.is_nucleotide(ResTypes.A) is False

    def test_gap_residue(self):
        gap = ResTypes.GAP
        assert gap.is_gap() is True
        assert ResTypes.A.is_gap() is False


class TestAtomTypes:
    def test_basic_atom_types(self):
        assert AtomTypes.N.name == "N"
        assert AtomTypes.CA.name == "CA"
        assert AtomTypes.C.name == "C"
        assert AtomTypes.O.name == "O"

    def test_from_string(self):
        atom = AtomTypes.from_string("CA")
        assert atom == AtomTypes.CA
        assert AtomTypes.from_string("INVALID") is None

    def test_num_types(self):
        assert AtomTypes.num_types() == 95


class TestPolymerType:
    def test_polymer_types(self):
        assert PolymerType.PROTEIN.value == "protein"
        assert PolymerType.RNA.value == "rna"
        assert PolymerType.DNA.value == "dna"
        assert PolymerType.CCD_LIGAND.value == "ccd_ligand"
        assert PolymerType.SMILES_LIGAND.value == "smiles_ligand"

    def test_fasta_default_polymer_type(self):
        fasta_path = SAMPLES_DIR / "T1094.fasta"
        parsed = read_fasta(fasta_path)
        molecule = parsed["sequences"][0]
        assert molecule["polymer_type"] == PolymerType.PROTEIN.value

    def test_openfold3_polymer_type_mapping_covers_all_types(self):
        """Both ligand variants must map to MOL_TYPE_LIGAND in the OF3
        token-feature pipeline; otherwise a CCD or SMILES ligand would be
        silently routed to protein/RNA/DNA."""
        from bionemo_ir.pipeline.models.openfold3.const import (
            MOL_TYPE_DNA,
            MOL_TYPE_LIGAND,
            MOL_TYPE_PROTEIN,
            MOL_TYPE_RNA,
            POLYMER_TYPE_TO_MOL_TYPE,
        )

        assert POLYMER_TYPE_TO_MOL_TYPE[PolymerType.PROTEIN.value] == MOL_TYPE_PROTEIN
        assert POLYMER_TYPE_TO_MOL_TYPE[PolymerType.RNA.value] == MOL_TYPE_RNA
        assert POLYMER_TYPE_TO_MOL_TYPE[PolymerType.DNA.value] == MOL_TYPE_DNA
        assert POLYMER_TYPE_TO_MOL_TYPE[PolymerType.CCD_LIGAND.value] == MOL_TYPE_LIGAND
        assert POLYMER_TYPE_TO_MOL_TYPE[PolymerType.SMILES_LIGAND.value] == MOL_TYPE_LIGAND


class TestInputRequest:
    def test_create_input_request_with_polymers(self):
        polymer = Polymer(polymer_type=PolymerType.PROTEIN, chain_id="A", sequence="ACDE")
        request = InputRequest(input_id="test", polymers=[polymer])
        assert len(request["polymers"]) == 1
        assert request["polymers"][0]["sequence"] == "ACDE"


class TestFoldingOutput:
    def test_create_folding_output(self):
        num_res = 95
        num_atom_type = 37
        atom_positions = np.random.randn(num_res, num_atom_type, 3).astype(np.float32)
        residue_types = np.random.randint(0, 21, size=(num_res,), dtype=np.int32)
        atom_mask = np.ones((num_res, num_atom_type), dtype=np.float32)
        residue_indices = np.arange(num_res, dtype=np.int32)
        b_factors = np.random.rand(num_res, num_atom_type).astype(np.float32) * 50
        chain_indices = np.zeros(num_res, dtype=np.int32)

        output = FoldingOutput(
            atom_positions=atom_positions,
            residue_types=residue_types,
            atom_mask=atom_mask,
            residue_indices=residue_indices,
            b_factors=b_factors,
            chain_indices=chain_indices,
        )

        assert output["atom_positions"].shape == (num_res, num_atom_type, 3)
        assert output["residue_types"].shape == (num_res,)
        assert output["atom_mask"].shape == (num_res, num_atom_type)
        assert output["residue_indices"].shape == (num_res,)
        assert output["b_factors"].shape == (num_res, num_atom_type)
        assert output["chain_indices"].shape == (num_res,)

    def test_create_folding_output_with_confidence_metrics(self):
        num_res = 95
        num_atom_type = 37
        atom_positions = np.random.randn(num_res, num_atom_type, 3).astype(np.float32)
        residue_types = np.random.randint(0, 21, size=(num_res,), dtype=np.int32)
        atom_mask = np.ones((num_res, num_atom_type), dtype=np.float32)
        residue_indices = np.arange(num_res, dtype=np.int32)
        plddt = np.random.rand(num_res).astype(np.float32) * 100
        pae = np.random.rand(num_res, num_res).astype(np.float32) * 31.75

        output = FoldingOutput(
            atom_positions=atom_positions,
            residue_types=residue_types,
            atom_mask=atom_mask,
            residue_indices=residue_indices,
            plddt=plddt,
            ptm=0.85,
            iptm=0.72,
            pae=pae,
            max_pae=31.75,
        )

        assert output["plddt"].shape == (num_res,)
        assert output["ptm"] == pytest.approx(0.85)
        assert output["iptm"] == pytest.approx(0.72)
        assert output["pae"].shape == (num_res, num_res)
        assert output["max_pae"] == pytest.approx(31.75)

    def test_folding_output_from_sample_sequence(self):
        fasta_path = SAMPLES_DIR / "T1033.fasta"
        parsed = read_fasta(fasta_path)
        sequence = parsed["sequences"][0]["sequence"]
        num_res = len(sequence)
        num_atom_type = AtomTypes.num_types()

        residue_types = np.array(
            [
                ResTypes.basic_20_residue_types().index(ResTypes.from_string(r, return_unknown=True))
                if ResTypes.from_string(r) in ResTypes.basic_20_residue_types()
                else 20
                for r in sequence
            ],
            dtype=np.int32,
        )

        output = FoldingOutput(
            atom_positions=np.zeros((num_res, num_atom_type, 3), dtype=np.float32),
            residue_types=residue_types,
            atom_mask=np.zeros((num_res, num_atom_type), dtype=np.float32),
            residue_indices=np.arange(num_res, dtype=np.int32),
        )

        assert output["residue_types"].shape == (100,)
        assert output["b_factors"] is None
        assert output["chain_indices"] is None
        assert output["plddt"] is None
        assert output["ptm"] is None
        assert output["iptm"] is None
        assert output["pae"] is None
        assert output["max_pae"] is None

    def test_get_scores_with_all_metrics(self):
        num_res = 10
        num_atom_type = 37
        plddt = np.array([85.0, 90.1, 72.3, 95.0, 60.5, 88.2, 91.0, 77.4, 83.6, 69.8], dtype=np.float32)
        pae = np.random.rand(num_res, num_res).astype(np.float32) * 20.0

        output = FoldingOutput(
            atom_positions=np.zeros((num_res, num_atom_type, 3), dtype=np.float32),
            residue_types=np.zeros(num_res, dtype=np.int32),
            atom_mask=np.zeros((num_res, num_atom_type), dtype=np.float32),
            residue_indices=np.arange(num_res, dtype=np.int32),
            plddt=plddt,
            ptm=0.92,
            iptm=0.88,
            pae=pae,
            max_pae=31.75,
        )

        scores = output.get_scores()
        assert isinstance(scores, dict)
        assert isinstance(scores["plddt"], list)
        assert len(scores["plddt"]) == num_res
        assert scores["ptm"] == pytest.approx(0.92)
        assert scores["iptm"] == pytest.approx(0.88)
        assert isinstance(scores["pae"], list)
        assert len(scores["pae"]) == num_res
        assert len(scores["pae"][0]) == num_res
        assert scores["max_pae"] == pytest.approx(31.75)

    def test_get_scores_with_no_metrics(self):
        num_res = 10
        num_atom_type = 37

        output = FoldingOutput(
            atom_positions=np.zeros((num_res, num_atom_type, 3), dtype=np.float32),
            residue_types=np.zeros(num_res, dtype=np.int32),
            atom_mask=np.zeros((num_res, num_atom_type), dtype=np.float32),
            residue_indices=np.arange(num_res, dtype=np.int32),
        )

        scores = output.get_scores()
        assert scores["plddt"] is None
        assert scores["ptm"] is None
        assert scores["iptm"] is None
        assert scores["pae"] is None
        assert scores["max_pae"] is None

    def test_get_scores_with_nan_values(self):
        num_res = 10
        num_atom_type = 37

        output = FoldingOutput(
            atom_positions=np.zeros((num_res, num_atom_type, 3), dtype=np.float32),
            residue_types=np.zeros(num_res, dtype=np.int32),
            atom_mask=np.zeros((num_res, num_atom_type), dtype=np.float32),
            residue_indices=np.arange(num_res, dtype=np.int32),
            ptm=float("nan"),
            iptm=float("nan"),
            max_pae=float("nan"),
        )

        scores = output.get_scores()
        assert scores["ptm"] is None
        assert scores["iptm"] is None
        assert scores["max_pae"] is None


class TestMSARecord:
    def test_create_with_content(self):
        content = ">seq1\nACDEFGHIKLMNPQRSTVWY\n>seq2\nACDEFGHIKLMNPQRSTVWY"
        record = MSARecord(content=content, format="a3m")
        assert record["content"] == content
        assert record["path"] is None
        assert record["format"] == "a3m"
        assert record.is_file() is False
        assert record.get_content() == content

    def test_create_with_path(self):
        a3m_file = str(SAMPLES_DIR / "msas" / "T1031.a3m")
        record = MSARecord(path=a3m_file, format="a3m")
        assert record["content"] is None
        assert record["path"] == a3m_file
        assert record.is_file() is True
        content = record.get_content()
        assert len(content) > 0
        assert content.startswith(">")

    @pytest.mark.parametrize("target", ["T1031", "T1033", "T1047s1", "T1094"])
    def test_alignment_file_record_from_samples(self, target):
        a3m_file = str(SAMPLES_DIR / "msas" / f"{target}.a3m")
        record = MSARecord(path=a3m_file)
        content = record.get_content()
        assert content.startswith(">")
        assert len(content) > 0


class TestTemplate:
    def test_create_with_content(self):
        content = "data_sample\n_atom_site.id 1\n"
        template = Template(content=content, format="cif")
        assert template["content"] == content
        assert template["path"] is None
        assert template["format"] == "cif"
        assert template.is_file() is False
        assert template.get_content() == content

    def test_create_with_path(self):
        fasta_file = str(SAMPLES_DIR / "T1031.fasta")
        template = Template(path=fasta_file, format="fasta")
        assert template["content"] is None
        assert template["path"] == fasta_file
        assert template.is_file() is True
        content = template.get_content()
        assert "T1031" in content


class TestPolymerWithMSA:
    def test_molecule_with_msa_content(self):
        msa_content = ">seq1\nACDEFGHIKL\n>seq2\nACDEFGHIKL"
        msa_record = MSARecord(content=msa_content)
        molecule = Polymer(polymer_type=PolymerType.PROTEIN, chain_id="A", sequence="ACDEFGHIKL", msas=[msa_record])
        assert molecule["msas"][0]["content"] == msa_content

    def test_molecule_with_msa_path(self):
        a3m_file = str(SAMPLES_DIR / "msas" / "T1031.a3m")
        msa_record = MSARecord(path=a3m_file)
        fasta_path = SAMPLES_DIR / "T1031.fasta"
        parsed = read_fasta(fasta_path)
        sequence = parsed["sequences"][0]["sequence"]

        molecule = Polymer(polymer_type=PolymerType.PROTEIN, chain_id="A", sequence=sequence, msas=[msa_record])
        assert molecule["msas"][0].is_file() is True
        content = molecule["msas"][0].get_content()
        assert content.startswith(">")
        assert len(content) > 0

    def test_molecule_with_multiple_msas(self):
        msa1 = MSARecord(content=">seq1\nACDE")
        msa2 = MSARecord(content=">seq2\nFGHI")
        molecule = Polymer(polymer_type=PolymerType.PROTEIN, chain_id="A", sequence="ACDEFGHI", msas=[msa1, msa2])
        assert len(molecule["msas"]) == 2


class TestA3MIntegration:
    def test_read_a3m_from_samples(self):
        a3m_path = SAMPLES_DIR / "msas" / "T1031.a3m"
        parsed = read_a3m(a3m_path)
        assert "sequences" in parsed
        assert "raw" in parsed
        assert "descriptions" in parsed
        assert len(parsed["sequences"]) > 0

    @pytest.mark.parametrize("target", ["T1031", "T1033", "T1047s1", "T1094"])
    def test_fasta_a3m_sequence_consistency(self, target):
        fasta_path = SAMPLES_DIR / f"{target}.fasta"
        a3m_path = SAMPLES_DIR / "msas" / f"{target}.a3m"

        fasta_parsed = read_fasta(fasta_path)
        a3m_parsed = read_a3m(a3m_path)

        fasta_seq = fasta_parsed["sequences"][0]["sequence"]
        a3m_first_seq = a3m_parsed["sequences"][0]

        assert fasta_seq == a3m_first_seq
