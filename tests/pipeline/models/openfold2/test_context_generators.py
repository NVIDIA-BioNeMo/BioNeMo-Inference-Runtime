# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_bionemo.data.parsers import InputParsed, MSAParsed
from tensorrt_bionemo.data.schemas.basic import PolymerParsed
from tensorrt_bionemo.pipeline.models.openfold2.context import (
    MSAContextGenerator,
    PrimaryContextGenerator,
    TemplateContextGenerator,
)


@pytest.fixture
def sample_sequence():
    return "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"


@pytest.fixture
def sample_input_parsed(sample_sequence):
    polymer = PolymerParsed(
        polymer_type="protein",
        chain_id="A",
        sequence=sample_sequence,
        msas=None,
        templates=None,
    )
    return InputParsed(input_id="test_input", polymers=[polymer])


@pytest.fixture
def sample_input_parsed_with_msa(sample_sequence):
    msa = MSAParsed(
        sequences=[sample_sequence, sample_sequence.replace("M", "L")],
        raw=[sample_sequence, sample_sequence.replace("M", "L")],
        descriptions=["seq1", "seq2"],
    )
    polymer = PolymerParsed(
        polymer_type="protein",
        chain_id="A",
        sequence=sample_sequence,
        msas=[msa],
        templates=None,
    )
    return InputParsed(input_id="test_input", polymers=[polymer])


class TestPrimaryContextGenerator:

    def test_generates_aatype(self, sample_input_parsed, sample_sequence):
        generator = PrimaryContextGenerator()
        result = generator(sample_input_parsed)

        assert "aatype" in result
        assert result["aatype"].shape[0] == len(sample_sequence)

    def test_generates_residue_index(self, sample_input_parsed, sample_sequence):
        generator = PrimaryContextGenerator()
        result = generator(sample_input_parsed)

        assert "residue_index" in result
        assert result["residue_index"].shape[0] == len(sample_sequence)
        assert result["residue_index"][0] == 0
        assert result["residue_index"][-1] == len(sample_sequence) - 1

    def test_generates_seq_length(self, sample_input_parsed, sample_sequence):
        generator = PrimaryContextGenerator()
        result = generator(sample_input_parsed)

        assert "seq_length" in result
        assert result["seq_length"].shape[0] == len(sample_sequence)
        assert torch.all(result["seq_length"] == len(sample_sequence))

    def test_generates_between_segment_residues(self, sample_input_parsed, sample_sequence):
        generator = PrimaryContextGenerator()
        result = generator(sample_input_parsed)

        assert "between_segment_residues" in result
        assert result["between_segment_residues"].shape[0] == len(sample_sequence)
        assert torch.all(result["between_segment_residues"] == 0)

    def test_aatype_dtype(self, sample_input_parsed):
        generator = PrimaryContextGenerator()
        result = generator(sample_input_parsed)

        assert result["aatype"].dtype == torch.int64

    def test_residue_index_dtype(self, sample_input_parsed):
        generator = PrimaryContextGenerator()
        result = generator(sample_input_parsed)

        assert result["residue_index"].dtype == torch.int32


class TestMSAContextGenerator:

    def test_generates_msa_from_provided_msas(self, sample_input_parsed_with_msa):
        generator = MSAContextGenerator()
        result = generator(sample_input_parsed_with_msa)

        assert "msa" in result
        assert "deletion_matrix" in result
        assert "num_alignments" in result

    def test_msa_shape(self, sample_input_parsed_with_msa, sample_sequence):
        generator = MSAContextGenerator()
        result = generator(sample_input_parsed_with_msa)

        assert result["msa"].shape[1] == len(sample_sequence)

    def test_creates_dummy_msa_when_none_provided(self, sample_input_parsed, sample_sequence):
        generator = MSAContextGenerator()
        result = generator(sample_input_parsed)

        assert "msa" in result
        assert result["msa"].shape[0] == 1
        assert result["msa"].shape[1] == len(sample_sequence)

    def test_deletion_matrix_shape_matches_msa(self, sample_input_parsed_with_msa):
        generator = MSAContextGenerator()
        result = generator(sample_input_parsed_with_msa)

        assert result["deletion_matrix"].shape == result["msa"].shape

    def test_num_alignments_values(self, sample_input_parsed_with_msa, sample_sequence):
        generator = MSAContextGenerator()
        result = generator(sample_input_parsed_with_msa)

        num_seqs = result["msa"].shape[0]
        assert result["num_alignments"].shape[0] == len(sample_sequence)
        assert torch.all(result["num_alignments"] == num_seqs)

    def test_msa_dtype(self, sample_input_parsed_with_msa):
        generator = MSAContextGenerator()
        result = generator(sample_input_parsed_with_msa)

        assert result["msa"].dtype == torch.int32
        assert result["deletion_matrix"].dtype == torch.float32


class TestTemplateContextGenerator:

    def test_generates_empty_templates_when_none_provided(self, sample_input_parsed, sample_sequence):
        generator = TemplateContextGenerator()
        result = generator(sample_input_parsed)

        assert "template_aatype" in result
        assert "template_all_atom_mask" in result
        assert "template_all_atom_positions" in result
        assert "template_sum_probs" in result

    def test_empty_template_shapes(self, sample_input_parsed, sample_sequence):
        generator = TemplateContextGenerator()
        result = generator(sample_input_parsed)

        n_res = len(sample_sequence)
        assert result["template_aatype"].shape[0] == 0
        assert result["template_aatype"].shape[1] == n_res
        assert result["template_all_atom_mask"].shape[0] == 0
        assert result["template_all_atom_mask"].shape[1] == n_res
        assert result["template_all_atom_positions"].shape[0] == 0
        assert result["template_all_atom_positions"].shape[1] == n_res

    def test_empty_template_dtypes(self, sample_input_parsed):
        generator = TemplateContextGenerator()
        result = generator(sample_input_parsed)

        assert result["template_aatype"].dtype == torch.float32
        assert result["template_all_atom_mask"].dtype == torch.float32
        assert result["template_all_atom_positions"].dtype == torch.float32
        assert result["template_sum_probs"].dtype == torch.float32


class TestContextGeneratorIntegration:

    def test_all_generators_produce_compatible_outputs(self, sample_input_parsed_with_msa, sample_sequence):
        primary_gen = PrimaryContextGenerator()
        msa_gen = MSAContextGenerator()
        template_gen = TemplateContextGenerator()

        primary_result = primary_gen(sample_input_parsed_with_msa)
        msa_result = msa_gen(sample_input_parsed_with_msa)
        template_result = template_gen(sample_input_parsed_with_msa)

        n_res = len(sample_sequence)
        assert primary_result["aatype"].shape[0] == n_res
        assert msa_result["msa"].shape[1] == n_res
        assert template_result["template_aatype"].shape[1] == n_res

    def test_merged_context_has_all_keys(self, sample_input_parsed_with_msa):
        primary_gen = PrimaryContextGenerator()
        msa_gen = MSAContextGenerator()
        template_gen = TemplateContextGenerator()

        merged = {}
        merged.update(primary_gen(sample_input_parsed_with_msa))
        merged.update(msa_gen(sample_input_parsed_with_msa))
        merged.update(template_gen(sample_input_parsed_with_msa))

        expected_keys = [
            "aatype", "residue_index", "seq_length", "between_segment_residues",
            "msa", "deletion_matrix", "num_alignments",
            "template_aatype", "template_all_atom_mask", "template_all_atom_positions", "template_sum_probs",
        ]
        for key in expected_keys:
            assert key in merged, f"Missing key: {key}"
