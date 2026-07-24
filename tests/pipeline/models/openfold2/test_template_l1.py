# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen direct-CIF feature parity against commit-pinned OpenFold output."""

from __future__ import annotations

import ast
import hashlib
import importlib.abc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

OPENFOLD_COMMIT = "be2ec1841f16c966c65ae0e7599ebbadc725757d"
INPUT_SHA256 = "c412665889d225f21c39e31609e44e1582bd51b9feac2e592bad996347ee4606"
ARTIFACT_SHA256 = "0f1416894d6a6228f41868a7d57e2b3fdb0bb25d0be8caf1fa5a50d39d941638"
QUERY_SEQUENCE = (
    "GMEGPLNLAHQQSRRADRLLAAGKYEEAISCHKKAAAYLSEAMKLTQSEQAHLSLELQRDSH"
    "MKQLLLIQERWKRAQREERLKA")
ATOL = 1e-4
RTOL = 1e-4

DATA_DIR = Path(__file__).with_name("data")
REPO_ROOT = Path(__file__).resolve().parents[4]
CIF_PATH = (REPO_ROOT / "examples" / "data" / "samples" / "monomers" /
            "templates" / "4zey.cif")
ARTIFACT_PATH = DATA_DIR / "template_l1_golden.npz"
PROVENANCE_PATH = DATA_DIR / "template_l1_provenance.json"
GENERATOR_PATH = Path(__file__).with_name("generate_template_l1.py")

EXACT_FEATURES = {
    "template_aatype",
    "template_all_atom_mask",
    "template_domain_names",
    "template_sequence",
    "template_pseudo_beta_mask",
    "template_torsion_angles_mask",
}
APPROX_FEATURES = {
    "template_all_atom_positions",
    "template_sum_probs",
    "template_pseudo_beta",
    "template_torsion_angles_sin_cos",
    "template_alt_torsion_angles_sin_cos",
}
EXPECTED_FEATURES = EXACT_FEATURES | APPROX_FEATURES
EXPECTED_SHAPES = {
    "template_aatype": (1, 84, 22),
    "template_all_atom_mask": (1, 84, 37),
    "template_all_atom_positions": (1, 84, 37, 3),
    "template_domain_names": (1, ),
    "template_pseudo_beta": (1, 84, 3),
    "template_pseudo_beta_mask": (1, 84),
    "template_sequence": (1, ),
    "template_sum_probs": (1, 1),
    "template_torsion_angles_sin_cos": (1, 84, 7, 2),
    "template_alt_torsion_angles_sin_cos": (1, 84, 7, 2),
    "template_torsion_angles_mask": (1, 84, 7),
}
EXPECTED_ARCHIVE_DTYPES = {
    "template_aatype": "int64",
    "template_all_atom_mask": "float32",
    "template_all_atom_positions": "float32",
    "template_domain_names": "|S6",
    "template_pseudo_beta": "float32",
    "template_pseudo_beta_mask": "float32",
    "template_sequence": "|S84",
    "template_sum_probs": "float32",
    "template_torsion_angles_sin_cos": "float32",
    "template_alt_torsion_angles_sin_cos": "float32",
    "template_torsion_angles_mask": "float32",
}
EXPECTED_SOURCE_DTYPES = {
    **EXPECTED_ARCHIVE_DTYPES,
    "template_domain_names": "object",
    "template_sequence": "object",
}
CRITICAL_SOURCE_HASHES = {
    "openfold/data/data_transforms.py":
    "000cc73fcd68603173af05dcbbcbd2663c97c467759d073783d20f38febf8311",
    "openfold/data/mmcif_parsing.py":
    "3728888533a46c689bd55b7002936a7b1b09f4c61fcc2274e7376c2b822ded14",
    "openfold/data/templates.py":
    "6c1f1534548543df6a26a3060ae25b9947d75cdf839e464f3415bb123f43f95e",
    "openfold/np/residue_constants.py":
    "19c94d0c104cf45ab303efb5b21fed36e746402163b742da0726bd2a8484a52b",
}


class _BlockOpenFoldImports(importlib.abc.MetaPathFinder):
    """Normal CI must exercise the frozen artifact without OSS installed."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None):
        del path, target
        if fullname == "openfold" or fullname.startswith("openfold."):
            raise AssertionError(
                f"Parity test attempted a runtime OpenFold import: {fullname}")
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value,
                         sort_keys=True,
                         separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_real_file(path: Path) -> None:
    assert path.is_file(), f"Required frozen oracle file is missing: {path}"
    prefix = path.read_bytes()[:128]
    assert not prefix.startswith(b"version https://git-lfs.github.com/spec"), (
        f"Frozen oracle is an unresolved Git LFS pointer: {path}")


def _load_provenance() -> dict[str, Any]:
    _require_real_file(PROVENANCE_PATH)
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert isinstance(provenance, dict)
    return provenance


def _load_reference() -> dict[str, np.ndarray]:
    _require_real_file(ARTIFACT_PATH)
    with np.load(ARTIFACT_PATH, allow_pickle=False) as archive:
        assert set(archive.files) == EXPECTED_FEATURES
        return {name: archive[name].copy() for name in archive.files}


def _encode_string_array(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    assert value.dtype.hasobject
    encoded = []
    for item in value.reshape(-1):
        if isinstance(item, (bytes, np.bytes_)):
            encoded.append(bytes(item))
        elif isinstance(item, str):
            encoded.append(item.encode("utf-8"))
        else:
            raise TypeError(f"Unexpected template string value: {type(item)}")
    width = max((len(item) for item in encoded), default=1)
    return np.asarray(encoded, dtype=f"S{width}").reshape(value.shape)


def _assert_generator_has_no_trt_imports() -> None:
    tree = ast.parse(GENERATOR_PATH.read_text(encoding="utf-8"))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
    assert not [
        name for name in imported_modules
        if name == "tensorrt_bionemo" or name.startswith("tensorrt_bionemo.")
    ]


def _derive_features(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from tensorrt_bionemo.pipeline.models.openfold2 import common
    from tensorrt_bionemo.pipeline.models.openfold2 import const as rc

    hhblits_aatype = torch.from_numpy(np.asarray(
        base["template_aatype"])).argmax(dim=-1)
    new_order = torch.tensor(rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE,
                             dtype=torch.int64).expand(hhblits_aatype.shape[0],
                                                       -1)
    aatype = torch.gather(new_order, 1, hhblits_aatype)
    positions = torch.from_numpy(
        np.asarray(base["template_all_atom_positions"]))
    mask = torch.from_numpy(np.asarray(base["template_all_atom_mask"]))
    pseudo_beta, pseudo_beta_mask = common.pseudo_beta_fn(
        aatype, positions, mask)
    torsions = common.atom37_to_torsion_angles(
        {
            "template_aatype": aatype,
            "template_all_atom_positions": positions,
            "template_all_atom_mask": mask,
        },
        prefix="template_",
    )
    return {
        "template_pseudo_beta":
        pseudo_beta.numpy(),
        "template_pseudo_beta_mask":
        pseudo_beta_mask.numpy(),
        "template_torsion_angles_sin_cos":
        torsions["template_torsion_angles_sin_cos"].numpy(),
        "template_alt_torsion_angles_sin_cos":
        torsions["template_alt_torsion_angles_sin_cos"].numpy(),
        "template_torsion_angles_mask":
        torsions["template_torsion_angles_mask"].numpy(),
    }


def test_template_l1_oracle_provenance_is_complete_and_pickle_free():
    for path in (CIF_PATH, ARTIFACT_PATH, PROVENANCE_PATH, GENERATOR_PATH):
        _require_real_file(path)

    provenance = _load_provenance()
    _assert_generator_has_no_trt_imports()
    assert provenance["schema_version"] == 1
    assert provenance["sample_id"] == "4zey_A_self_template"
    assert provenance["source"] == "oss"
    assert provenance["reference_kind"] == "oss_featurization_primitives"
    assert provenance["oss_git_commit"] == OPENFOLD_COMMIT
    assert provenance["artifact_format"] == "numpy_npz_no_object_arrays"
    assert provenance["checkpoint_identifier"] == "not_applicable_feature_only"
    assert provenance["checkpoint_sha256"] is None
    assert provenance["input_sha256"] == INPUT_SHA256 == _sha256(CIF_PATH)
    assert provenance["artifact_sha256"] == ARTIFACT_SHA256 == _sha256(
        ARTIFACT_PATH)
    assert provenance["sha256"] == provenance["artifact_sha256"]
    assert provenance["generator_sha256"] == _sha256(GENERATOR_PATH)
    assert OPENFOLD_COMMIT in provenance["oss_source_url"]
    assert OPENFOLD_COMMIT in provenance["input_source_url"]
    assert provenance["resolved_config_hash"] == _canonical_json_hash(
        provenance["resolved_config"])
    assert "generate_template_l1.py" in provenance["generation_command"]
    assert "--oss-root" in provenance["generation_command"]
    assert provenance["artifact_path"] == (
        "tests/pipeline/models/openfold2/data/template_l1_golden.npz")
    assert provenance["input_path"] == (
        "examples/data/samples/monomers/templates/4zey.cif")
    assert provenance["generator_path"] == (
        "tests/pipeline/models/openfold2/generate_template_l1.py")
    assert provenance["input_source_path"] == "tests/test_data/mmcifs/4zey.cif"
    assert provenance["oss_source_path"] == "openfold/data/templates.py"
    assert provenance["oss_source_files"] == CRITICAL_SOURCE_HASHES
    assert provenance["oss_parser"] == "openfold.data.mmcif_parsing.parse"
    assert provenance["oss_input_format"] == "mmCIF"
    assert provenance["oracle_mode"] == "openfold_identity_extract"
    assert provenance["warnings"] == []
    assert provenance["string_serialization"] == {
        "template_domain_names": "object bytes -> fixed-width bytes",
        "template_sequence": "object bytes -> fixed-width bytes",
    }

    reference = _load_reference()
    inventory = provenance["feature_inventory"]
    source_inventory = provenance["source_feature_inventory"]
    assert set(inventory) == EXPECTED_FEATURES == set(reference)
    assert set(source_inventory) == EXPECTED_FEATURES
    for name, value in reference.items():
        assert value.shape == EXPECTED_SHAPES[name], name
        assert str(value.dtype) == EXPECTED_ARCHIVE_DTYPES[name], name
        assert inventory[name]["shape"] == list(EXPECTED_SHAPES[name]), name
        assert inventory[name]["dtype"] == EXPECTED_ARCHIVE_DTYPES[name], name
        assert source_inventory[name]["shape"] == list(
            EXPECTED_SHAPES[name]), name
        assert source_inventory[name]["dtype"] == EXPECTED_SOURCE_DTYPES[
            name], name
        assert not value.dtype.hasobject, name


def test_template_l1_matches_pinned_openfold_features_without_runtime_oss():
    provenance = _load_provenance()
    reference = _load_reference()
    assert set(provenance["feature_inventory"]) == EXPECTED_FEATURES
    loaded_openfold = sorted(
        name for name in sys.modules
        if name == "openfold" or name.startswith("openfold."))
    assert not loaded_openfold, (
        f"OpenFold was already loaded before the offline parity test: "
        f"{loaded_openfold}")

    guard = _BlockOpenFoldImports()
    sys.meta_path.insert(0, guard)
    try:
        from tensorrt_bionemo.pipeline.models.openfold2.template_logic import \
            build_template_feats

        actual = build_template_feats(
            QUERY_SEQUENCE,
            [{
                "content": CIF_PATH.read_text(encoding="utf-8"),
                "format": "cif",
                "chain_id": "A",
            }],
            max_templates=1,
        )
        actual.update(_derive_features(actual))
    finally:
        sys.meta_path.remove(guard)
    loaded_openfold = sorted(
        name for name in sys.modules
        if name == "openfold" or name.startswith("openfold."))
    assert not loaded_openfold, (
        f"Production code loaded OpenFold at runtime: {loaded_openfold}")

    assert set(actual) == EXPECTED_FEATURES
    assert float(np.asarray(reference["template_all_atom_mask"]).sum()) > 5.0
    assert float(np.asarray(
        reference["template_pseudo_beta_mask"]).sum()) > 0.0

    for name in sorted(EXPECTED_FEATURES):
        expected = reference[name]
        raw_observed = np.asarray(actual[name])
        observed = (_encode_string_array(raw_observed)
                    if expected.dtype.kind == "S" else raw_observed)
        assert observed.shape == expected.shape, name
        assert observed.dtype == expected.dtype, name
        if observed.dtype.kind not in {"S", "U"}:
            assert np.isfinite(observed).all(), name
        if name in EXACT_FEATURES:
            np.testing.assert_array_equal(observed, expected, err_msg=name)
        else:
            np.testing.assert_allclose(observed,
                                       expected,
                                       atol=ATOL,
                                       rtol=RTOL,
                                       err_msg=name)
