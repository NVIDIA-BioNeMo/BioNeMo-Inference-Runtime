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

"""Native feature contracts that do not require checkpoint or CCD downloads."""

from pathlib import Path

import numpy as np
import pytest
import torch

from bionemo_ir.data.utils import get_all_atom_types, get_all_residue_types
from bionemo_ir.data.writers.cif_writer import CIFWriter
from bionemo_ir.data.writers.pdb_writer import PDBWriter
from bionemo_ir.pipeline.models.protenix.msa import alignment_rows
from bionemo_ir.pipeline.models.protenix.postprocessor import PostProcessor
from tests.pipeline.models.protenix.golden_io import build_conformer_dir, load_tensors


def test_msa_insertions_and_duplicate_rows() -> None:
    rows, deletion, _ = alignment_rows(
        {
            "polymer_type": "protein",
            "sequence": "ACDEF",
            "msas": [{"raw": ["ACDEF", "AaCDEF", "ACD-F"], "descriptions": ["q", "a", "b"]}],
        },
        False,
        False,
    )
    assert rows.shape == (2, 5)
    assert deletion.sum() == 0  # Insertions do not make a duplicate aligned row unique.
    _, deletion, _ = alignment_rows(
        {
            "polymer_type": "protein",
            "sequence": "ACDEF",
            "msas": [{"raw": ["ACDEF", "ACddD-F"]}],
        },
        False,
        False,
    )
    assert deletion[1, 2] == 2


@pytest.mark.parametrize("writer_class,extension", [(CIFWriter, "cif"), (PDBWriter, "pdb")])
def test_ranking_and_ion_atom_identity(tmp_path: Path, writer_class: type, extension: str) -> None:
    from bionemo_ir.pipeline.utils.atom import encode_atom_name_chars_one_hot

    # Calcium (Z=20) shares the atom name CA with an alpha carbon.
    batch = {
        "atom_mask": torch.ones(1, 2),
        "token_mask": torch.ones(1, 2),
        "atom_to_token_idx": torch.tensor([[0, 1]]),
        "ref_atom_name_chars": encode_atom_name_chars_one_hot(["CA", "CA"])[None],
        "ref_element": torch.nn.functional.one_hot(torch.tensor([[5, 19]]), 128),
        "restype": torch.nn.functional.one_hot(torch.tensor([[0, 20]]), 32),
        "residue_index": torch.tensor([[1, 1]]),
        "asym_id": torch.tensor([[0, 1]]),
        "token_resnames": ["ALA", "CA"],
        "token_mol_types": [0, 3],
    }
    output = {
        "coordinate": torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]),
        "summary_confidence": [{"ranking_score": score, "ptm": score, "iptm": score} for score in (0.2, 0.8)],
        "full_data": [{"atom_plddt": torch.tensor([0.9, 0.8]), "token_pair_pae": torch.ones(2, 2)} for _ in range(2)],
    }
    result = PostProcessor()(batch, output)
    assert result["atom_mask"].sum() == 2
    slots = np.argwhere(result["atom_mask"] > 0)
    assert [result["atom_elements"][r, c] for r, c in slots] == ["C", "Ca"]
    np.testing.assert_array_equal(result["atom_positions"][tuple(slots[0])], [7, 8, 9])
    np.testing.assert_allclose([result["b_factors"][r, c] for r, c in slots], [90, 80])
    np.testing.assert_allclose(result["plddt"], [90, 80])
    writer = writer_class(
        output_path=str(tmp_path / f"result.{extension}"),
        res_type_mapping=dict(enumerate(get_all_residue_types("protenix-v2"))),
        atom_type_mapping=dict(enumerate(get_all_atom_types("protenix-v2"))),
    )
    rendered = writer.write(result)
    assert "HETATM" in rendered


def test_real_template_geometry_matches_upstream() -> None:
    from bionemo_ir.pipeline.models.protenix.templates import template_geometry

    reference = load_tensors(Path(__file__).parent / "data/2lwu_template_golden.json")
    actual = template_geometry(
        reference["template_aatype"].numpy(),
        reference["template_atom_positions"].numpy(),
        reference["template_atom_mask"].numpy(),
    )
    for name, value in actual.items():
        torch.testing.assert_close(value, reference[name].to(value.dtype), atol=1e-5, rtol=1e-5)


_DATA = Path(__file__).parent / "data"
_UPSTREAM_KINDS = {"proteinChain": "protein", "rnaSequence": "rna", "dnaSequence": "dna"}


def _golden_polymers(sample: dict) -> list[dict]:
    """Translate the upstream sample JSON into parsed polymers with sequential chain IDs."""
    polymers, chain = [], 0
    for entry in sample["sequences"]:
        ((kind, body),) = entry.items()
        if kind in _UPSTREAM_KINDS:
            polymer_type, sequence = _UPSTREAM_KINDS[kind], body["sequence"]
        else:
            polymer_type, sequence = "ccd_ligand", body.get("ligand", body.get("ion")).removeprefix("CCD_")
        ids = [chr(ord("A") + chain + i) for i in range(body["count"])]
        chain += body["count"]
        polymers.append({"polymer_type": polymer_type, "sequence": sequence, "chain_id": ids})
    return polymers


def _pairwise_by_group(positions: torch.Tensor, groups: torch.Tensor) -> list[torch.Tensor]:
    """Rigid-motion invariant view of reference conformers."""
    return [torch.cdist(positions[groups == g], positions[groups == g]) for g in groups.unique()]


def test_features_match_upstream_release(tmp_path: Path) -> None:
    import json

    from bionemo_ir.models.protenix import Protenix
    from bionemo_ir.pipeline.models.protenix.feature_context import ProtenixContextGenerator
    from bionemo_ir.pipeline.models.protenix.feature_factory import FeatureFactory
    from bionemo_ir.pipeline.utils._rng import _feature_rng

    provenance = json.loads((_DATA / "multimol_features.json").read_text())
    reference = load_tensors(_DATA / "multimol_features_golden.json")
    factory = FeatureFactory()
    config = Protenix.get_pretrained_config("protenix-v2")
    mol_dir = build_conformer_dir(_DATA / "multimol_conformers_golden.json", tmp_path / "mols")
    metadata = {"ccd_mols_path": str(mol_dir)}
    context = factory.pre_init({"random_seed": 0})
    with _feature_rng(context["_rng"]):
        context["_row"] = ProtenixContextGenerator(config, metadata)(
            {"polymers": _golden_polymers(provenance["sample"])}
        )
        features = {}
        for spec in factory.feature_generator_specs:
            features.update(spec.functor(config=config, metadata=metadata)(features, context))

    tolerance = provenance["tolerance"]
    # The masks are all ones and the release does not emit them.
    assert set(features) - {"atom_mask", "token_mask"} == set(reference)
    for name in sorted(set(reference) - {"ref_pos"}):
        expected = reference[name].to_dense() if reference[name].is_sparse else reference[name]
        torch.testing.assert_close(features[name].double(), expected.double(), msg=name, **tolerance)
    groups = reference["ref_space_uid"]
    for actual, expected in zip(
        _pairwise_by_group(features["ref_pos"].double(), groups),
        _pairwise_by_group(reference["ref_pos"].double(), groups),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_conformer_split_writes_only_loadable_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import pickle

    import bionemo_ir
    from bionemo_ir.pipeline.models.protenix.structure import conformer_dir

    monkeypatch.setattr(bionemo_ir, "CACHE_DIR", tmp_path / "cache")
    source = tmp_path / "mols.pkl"
    source.write_bytes(pickle.dumps({"ATP": "atp", "../ESC": "escape", "TOOLONG": "long"}))
    target = conformer_dir(source)
    assert "not the v2.0.0 release" in caplog.text
    assert sorted(path.name for path in target.iterdir()) == ["ATP.pkl"]
    assert conformer_dir(source) == target
    assert not list((tmp_path / "cache").rglob("ESC.pkl"))


class _CodeGlobal:
    def __reduce__(self) -> tuple:
        return print, ("unpickled",)


def test_conformer_split_refuses_code_globals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pickle

    import bionemo_ir
    from bionemo_ir.pipeline.models.protenix.structure import conformer_dir

    monkeypatch.setattr(bionemo_ir, "CACHE_DIR", tmp_path / "cache")
    source = tmp_path / "mols.pkl"
    source.write_bytes(pickle.dumps({"ATP": _CodeGlobal()}))
    with pytest.raises(pickle.UnpicklingError, match="builtins.print"):
        conformer_dir(source)
    assert not list((tmp_path / "cache").rglob("*.pkl"))


def test_component_directory_refuses_code_globals(tmp_path: Path) -> None:
    import pickle

    from bionemo_ir.data.utils import load_component_mol
    from bionemo_ir.pipeline.models.protenix.structure import conformer_dir

    (tmp_path / "ATP.pkl").write_bytes(pickle.dumps(_CodeGlobal()))
    with pytest.raises(pickle.UnpicklingError, match="builtins.print"):
        load_component_mol(conformer_dir(tmp_path), "ATP")


def test_context_defers_metadata_loading_until_worker_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    from bionemo_ir.pipeline.models.protenix import feature_context

    calls = []
    monkeypatch.setattr(feature_context, "ComponentStore", lambda metadata: calls.append(metadata) or object())
    metadata = {"ccd_mols_path": "worker-local.pkl"}
    generator = feature_context.ProtenixContextGenerator(metadata=metadata)
    assert not calls
    generator.prepare()
    generator.prepare()
    assert calls == [metadata]


@pytest.mark.parametrize("row", ["ACDE", "ACDEFG"])
def test_msa_rejects_inconsistent_aligned_width(row: str) -> None:
    with pytest.raises(ValueError, match="aligned row length"):
        alignment_rows(
            {"polymer_type": "protein", "sequence": "ACDEF", "msas": [{"raw": ["ACDEF", row]}]}, False, False
        )


def test_templates_reject_pdb() -> None:
    from bionemo_ir.pipeline.models.protenix.templates import _template_hits

    with pytest.raises(ValueError, match="mmCIF"):
        _template_hits("ACDEF", [{"content": "ATOM", "format": "pdb"}], 4)


def test_msa_rejects_mismatched_descriptions() -> None:
    with pytest.raises(ValueError, match="descriptions"):
        alignment_rows(
            {
                "polymer_type": "protein",
                "sequence": "ACDEF",
                "paired_msas": [{"raw": ["ACDEF", "ACD-F"], "descriptions": ["query"]}],
            },
            True,
            False,
        )


@pytest.mark.parametrize("field", ["max_msa_rows", "max_paired_per_species", "max_templates"])
@pytest.mark.parametrize("value", [0, -1])
def test_data_config_rejects_nonpositive_limits(field: str, value: int) -> None:
    from pydantic import ValidationError

    from bionemo_ir.models.protenix.config import ProtenixConfig

    with pytest.raises(ValidationError, match=field):
        ProtenixConfig(**{field: value})


@pytest.mark.parametrize("kind,sequences", [("rna", ["ACGUU"]), ("protein", ["ACDEF", "ACDEG"])])
def test_msa_query_survives_one_row_budget_and_disabled_rna(kind: str, sequences: list[str]) -> None:
    from types import SimpleNamespace

    from bionemo_ir.models.protenix.config import ProtenixConfig
    from bionemo_ir.pipeline.models.protenix.const import MSA_PROTEIN_SEQ_TO_ID, MSA_RNA_SEQ_TO_ID
    from bionemo_ir.pipeline.models.protenix.msa import MsaFeatureGenerator

    chains = [
        {
            "polymer_type": kind,
            "sequence": sequence,
            "paired_msas": [{"raw": [sequence, "A----"]}],
        }
        for sequence in sequences
    ]
    atoms = SimpleNamespace(asym=np.repeat(np.arange(len(chains)), 5), res_id=np.tile(np.arange(1, 6), len(chains)))
    structure = {"atoms": atoms, "tokens": [{"center": i} for i in range(len(atoms.asym))], "chains": chains}
    config = ProtenixConfig(max_msa_rows=1, use_rna_msa=False)
    features = MsaFeatureGenerator(config=config)({}, {"_row": {"structure": structure}})
    mapping = MSA_RNA_SEQ_TO_ID if kind == "rna" else MSA_PROTEIN_SEQ_TO_ID
    expected = torch.tensor([[mapping[c] for sequence in sequences for c in sequence]])
    torch.testing.assert_close(features["msa"], expected)
    if kind == "rna":
        torch.testing.assert_close(features["profile"], torch.nn.functional.one_hot(expected[0], 32).float())
    assert chains[0]["paired_msas"]  # The caller's request remains reusable.


@pytest.mark.parametrize(("modified_as_unknown", "sequence"), [(False, "MR"), (True, "XR")])
def test_template_sequence_spells_modified_residues(modified_as_unknown: bool, sequence: str) -> None:
    from bionemo_ir.pipeline.models.openfold3.template_logic import TemplateParsing, extract_template_chains

    content = (Path(__file__).parents[1] / "openfold2/data/minimal_template.cif").read_text()
    chains = extract_template_chains(content, parsing=TemplateParsing(modified_as_unknown=modified_as_unknown))
    assert chains["L1"].canonical_seq == sequence


def test_template_altlocs_follow_biopython_per_atom_choice() -> None:
    import biotite.structure as struc

    from bionemo_ir.pipeline.models.openfold3.template_logic import _first_highest_occupancy

    # CA ties (first listed wins); OG prefers the higher-occupancy second location.
    atoms = struc.array(
        [struc.Atom(np.zeros(3), chain_id="A", res_id=1, atom_name=n) for n in ("CA", "CA", "OG", "OG")]
    )
    atoms.coord[:, 0] = [1, 2, 3, 4]
    atoms.set_annotation("occupancy", np.array([0.5, 0.5, 0.3, 0.7]))
    chosen = _first_highest_occupancy(atoms)
    assert chosen.atom_name.tolist() == ["CA", "OG"]
    assert chosen.coord[:, 0].tolist() == [1, 4]


@pytest.mark.parametrize("kind,query", [("protein", "ACDEF"), ("rna", "ACGUU")])
def test_msa_preserves_insertions_and_species(kind: str, query: str) -> None:
    from bionemo_ir.pipeline.models.protenix.const import MSA_PROTEIN_SEQ_TO_ID, MSA_RNA_SEQ_TO_ID

    mapping = MSA_PROTEIN_SEQ_TO_ID if kind == "protein" else MSA_RNA_SEQ_TO_ID
    raw = ["aa" + query + "zzz", query[:2] + "bbb" + query[2:]]
    rows, deletion, species = alignment_rows(
        {
            "polymer_type": kind,
            "sequence": query,
            "paired_msas": [{"raw": raw, "descriptions": ["sp|P12345|PROT_HUMAN", "UniRef100_TEST_MOUSE"]}],
        },
        True,
        False,
    )
    np.testing.assert_array_equal(rows, [[mapping[c] for c in query]] * 2)
    np.testing.assert_array_equal(deletion, [[2, 0, 0, 0, 0], [0, 0, 3, 0, 0]])
    assert species == ["HUMAN", "MOUSE"]


def test_template_reader_retains_requested_atoms() -> None:
    import io

    import biotite.structure as struc
    from biotite.structure.io import pdbx

    from bionemo_ir.pipeline.models.openfold3.template_logic import TemplateParsing, extract_template_chains

    atoms = struc.AtomArray(6)
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = 1
    atoms.res_name[:] = "MET"
    atoms.atom_name = np.array(["N", "CA", "C", "CB", "SD", "H"])
    atoms.element = np.array(["N", "C", "C", "C", "S", "H"])
    atoms.coord = np.arange(18, dtype=np.float32).reshape(6, 3)
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, atoms)
    stream = io.StringIO()
    cif.write(stream)
    default = extract_template_chains(stream.getvalue())["A"]
    full = extract_template_chains(stream.getvalue(), parsing=TemplateParsing(atom_names=None))["A"]
    assert set(default.coords_by_pos[1]) == {"N", "CA", "C", "CB"}
    assert set(full.coords_by_pos[1]) == {"N", "CA", "C", "CB", "SD"}
    np.testing.assert_array_equal(full.coords_by_pos[1]["SD"], atoms.coord[4])
    assert default.canonical_seq == full.canonical_seq == "M"


def test_collator_keeps_token_identity() -> None:
    from types import SimpleNamespace

    from bionemo_ir.pipeline.models.protenix.feature_collators import ProtenixFeatureCollator

    features = {"atom_to_token_idx": torch.tensor([0, 0, 1, 2])}
    # A CCD ligand named like a residue (TYR) is tokenized as protein but written as a non-polymer.
    atoms = SimpleNamespace(
        res_name=np.array(["ALA", "ALA", "CA", "TYR"]),
        mol_type=np.array([0, 0, 3, 0]),
        hetero=np.array([False, False, True, True]),
    )
    structure = {
        "atoms": atoms,
        "tokens": [{"center": 1}, {"center": 2}, {"center": 3}],
        "chains": [{"chain_id": "A"}, {"chain_id": "B"}, {"chain_id": "C"}],
    }
    batch = ProtenixFeatureCollator()(features, {"_row": {"structure": structure}})
    assert batch["chain_ids"] == ["A", "B", "C"]
    assert batch["atom_to_token_idx"].shape == (1, 4)
    assert batch["token_resnames"] == ["ALA", "CA", "TYR"]
    assert batch["token_mol_types"] == [0, 3, 3]
