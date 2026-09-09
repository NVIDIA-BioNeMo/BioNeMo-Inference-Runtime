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

"""Regenerate the frozen OpenFold2 direct-CIF template oracle.

This script is intentionally outside the pytest path. It imports featurization
primitives from a commit-pinned OpenFold checkout and never imports the
BioIR OpenFold2 pipeline. Normal CI only consumes the generated NPZ and
does not need OpenFold installed.

Example:
    python tests/pipeline/models/openfold2/generate_template_l1.py \
        --oss-root /path/to/aqlaboratory/openfold
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np

OPENFOLD_COMMIT = "be2ec1841f16c966c65ae0e7599ebbadc725757d"
QUERY_SEQUENCE = "GMEGPLNLAHQQSRRADRLLAAGKYEEAISCHKKAAAYLSEAMKLTQSEQAHLSLELQRDSHMKQLLLIQERWKRAQREERLKA"
CHAIN_ID = "A"
SAMPLE_ID = "4zey_A_self_template"
INPUT_RELATIVE_PATH = Path("tests/test_data/mmcifs/4zey.cif")
INPUT_SHA256 = "c412665889d225f21c39e31609e44e1582bd51b9feac2e592bad996347ee4606"
FORBIDDEN_IMPORT_PREFIX = "bionemo_ir.pipeline.models.openfold2"

SCRIPT_PATH = Path(__file__).resolve()
DATA_DIR = SCRIPT_PATH.with_name("data")
REPO_ROOT = SCRIPT_PATH.parents[4]
CIF_PATH = REPO_ROOT / "examples" / "data" / "samples" / "monomers" / "templates" / "4zey.cif"
ARTIFACT_PATH = DATA_DIR / "template_l1_golden.npz"
PROVENANCE_PATH = DATA_DIR / "template_l1_provenance.json"

CRITICAL_SOURCE_HASHES = {
    "openfold/data/data_transforms.py": "000cc73fcd68603173af05dcbbcbd2663c97c467759d073783d20f38febf8311",
    "openfold/data/mmcif_parsing.py": "3728888533a46c689bd55b7002936a7b1b09f4c61fcc2274e7376c2b822ded14",
    "openfold/data/templates.py": "6c1f1534548543df6a26a3060ae25b9947d75cdf839e464f3415bb123f43f95e",
    "openfold/np/residue_constants.py": "19c94d0c104cf45ab303efb5b21fed36e746402163b742da0726bd2a8484a52b",
}

DERIVED_FEATURES = (
    "template_pseudo_beta",
    "template_pseudo_beta_mask",
    "template_torsion_angles_sin_cos",
    "template_alt_torsion_angles_sin_cos",
    "template_torsion_angles_mask",
)

EXPECTED_SHAPES = {
    "template_aatype": (1, 84, 22),
    "template_all_atom_mask": (1, 84, 37),
    "template_all_atom_positions": (1, 84, 37, 3),
    "template_domain_names": (1,),
    "template_pseudo_beta": (1, 84, 3),
    "template_pseudo_beta_mask": (1, 84),
    "template_sequence": (1,),
    "template_sum_probs": (1, 1),
    "template_torsion_angles_sin_cos": (1, 84, 7, 2),
    "template_alt_torsion_angles_sin_cos": (1, 84, 7, 2),
    "template_torsion_angles_mask": (1, 84, 7),
}
EXPECTED_SOURCE_DTYPES = {
    "template_aatype": "int64",
    "template_all_atom_mask": "float32",
    "template_all_atom_positions": "float32",
    "template_domain_names": "object",
    "template_pseudo_beta": "float32",
    "template_pseudo_beta_mask": "float32",
    "template_sequence": "object",
    "template_sum_probs": "float32",
    "template_torsion_angles_sin_cos": "float32",
    "template_alt_torsion_angles_sin_cos": "float32",
    "template_torsion_angles_mask": "float32",
}
EXPECTED_ARCHIVE_DTYPES = {
    **EXPECTED_SOURCE_DTYPES,
    "template_domain_names": "|S6",
    "template_sequence": "|S84",
}


class _BlockBioIROpenFold2Imports:
    """Make accidental self-reference fail before Python resolves a module."""

    @staticmethod
    def find_spec(fullname: str, path: Any = None, target: Any = None) -> None:
        del path, target
        if fullname == FORBIDDEN_IMPORT_PREFIX or fullname.startswith(FORBIDDEN_IMPORT_PREFIX + "."):
            raise RuntimeError(f"Reference generator attempted a forbidden import: {fullname}")
        return None


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_ast(path: Path) -> str:
    """Hash the generator's parsed AST, not its bytes.

    The pin's job is to detect logic edits to this oracle generator, not
    formatting churn. Hashing ``ast.dump(ast.parse(...))`` stays stable across
    reformats and comment changes (e.g. a ``ruff`` version bump) while still
    catching any real change to statements or expressions.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return _sha256_bytes(ast.dump(tree).encode("utf-8"))


def _canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return _sha256_bytes(payload)


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _assert_no_self_reference() -> None:
    loaded = sorted(
        name
        for name in sys.modules
        if name == FORBIDDEN_IMPORT_PREFIX or name.startswith(FORBIDDEN_IMPORT_PREFIX + ".")
    )
    if loaded:
        raise RuntimeError(f"BioIR OpenFold2 modules are loaded by the oracle: {loaded}")


def _verify_checkout(oss_root: Path) -> dict[str, str]:
    process = subprocess.run(
        ["git", "-c", f"safe.directory={oss_root}", "-C", str(oss_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Cannot resolve OpenFold checkout {oss_root}: {process.stderr.strip()}")
    actual_commit = process.stdout.strip()
    if actual_commit != OPENFOLD_COMMIT:
        raise RuntimeError(f"OpenFold commit mismatch: expected={OPENFOLD_COMMIT}, actual={actual_commit}")

    verified = {}
    for relative_path, expected_hash in CRITICAL_SOURCE_HASHES.items():
        source_path = oss_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Pinned OpenFold source is missing: {source_path}")
        actual_hash = _sha256_file(source_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"OpenFold source hash mismatch for {relative_path}: expected={expected_hash}, actual={actual_hash}"
            )
        verified[relative_path] = actual_hash
    return verified


def _load_openfold_modules(oss_root: Path) -> tuple[Any, Any, Any]:
    if any(name == "openfold" or name.startswith("openfold.") for name in sys.modules):
        raise RuntimeError("OpenFold was imported before pin verification")

    package_dir = oss_root / "openfold"
    package = types.ModuleType("openfold")
    package.__file__ = str(package_dir / "__init__.py")
    package.__package__ = "openfold"
    package.__path__ = [str(package_dir)]
    package.__spec__ = importlib.machinery.ModuleSpec("openfold", loader=None, is_package=True)
    sys.modules["openfold"] = package

    from openfold.data import (
        data_transforms,  # pylint: disable=import-outside-toplevel
        mmcif_parsing,  # pylint: disable=import-outside-toplevel
        templates,  # pylint: disable=import-outside-toplevel
    )

    package_prefix = str(package_dir.resolve()) + os.sep
    for name, module in sorted(sys.modules.items()):
        if not name.startswith("openfold."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        if not str(Path(module_file).resolve()).startswith(package_prefix):
            raise RuntimeError(f"OpenFold import escaped the pinned checkout: {name}")
    _assert_no_self_reference()
    return (data_transforms, mmcif_parsing, templates)


def _generate_features(template_path: Path, modules: tuple[Any, Any, Any]) -> tuple[dict[str, np.ndarray], list[str]]:
    data_transforms, mmcif_parsing, templates = modules
    parse_result = mmcif_parsing.parse(file_id="4zey", mmcif_string=template_path.read_text(encoding="utf-8"))
    if parse_result.mmcif_object is None:
        raise RuntimeError(f"OpenFold failed to parse 4zey: {parse_result.errors}")
    mmcif_object = parse_result.mmcif_object
    template_sequence = mmcif_object.chain_to_seqres.get(CHAIN_ID)
    if template_sequence != QUERY_SEQUENCE:
        raise RuntimeError(
            "4zey chain A is no longer the declared identity template: "
            f"expected={len(QUERY_SEQUENCE)} residues, "
            f"actual={len(template_sequence or '')}"
        )

    current, warning = templates._extract_template_features(
        mmcif_object=mmcif_object,
        pdb_id="4zey",
        mapping={index: index for index in range(len(QUERY_SEQUENCE))},
        template_sequence=template_sequence,
        query_sequence=QUERY_SEQUENCE,
        template_chain_id=CHAIN_ID,
        kalign_binary_path="",
        _zero_center_positions=True,
    )
    current["template_sum_probs"] = [1.0]
    features = {
        name: np.stack([current[name]], axis=0).astype(dtype) for name, dtype in templates.TEMPLATE_FEATURES.items()
    }

    transform_input = {
        "template_aatype": data_transforms.torch.from_numpy(features["template_aatype"].copy()),
        "template_all_atom_mask": data_transforms.torch.from_numpy(features["template_all_atom_mask"].copy()),
        "template_all_atom_positions": data_transforms.torch.from_numpy(features["template_all_atom_positions"].copy()),
    }
    transform_input = data_transforms.fix_templates_aatype(transform_input)
    transform_input = data_transforms.make_pseudo_beta("template_")(transform_input)
    transform_input = data_transforms.atom37_to_torsion_angles("template_")(transform_input)
    for name in DERIVED_FEATURES:
        features[name] = transform_input[name].detach().cpu().numpy()

    warnings = [] if warning is None else [str(warning)]
    return features, warnings


def _pickle_free_array(value: np.ndarray) -> np.ndarray:
    """Encode object-string arrays as fixed-width bytes for allow_pickle=False."""
    value = np.asarray(value)
    if not value.dtype.hasobject:
        return value
    encoded = []
    for item in value.reshape(-1):
        if isinstance(item, (bytes, np.bytes_)):
            encoded.append(bytes(item))
        elif isinstance(item, str):
            encoded.append(item.encode("utf-8"))
        else:
            raise TypeError(f"Oracle contains a non-string object value: {type(item)}")
    width = max((len(item) for item in encoded), default=1)
    return np.asarray(encoded, dtype=f"S{width}").reshape(value.shape)


def _inventory(features: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    result = {}
    for name, value in sorted(features.items()):
        entry: dict[str, Any] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if not value.dtype.hasobject and value.dtype.kind not in {"S", "U"}:
            if not np.isfinite(value).all():
                raise RuntimeError(f"Non-finite values in oracle feature {name}")
            entry["min"] = float(value.min())
            entry["max"] = float(value.max())
        result[name] = entry
    return result


def _validate_contract(features: dict[str, np.ndarray], expected_dtypes: dict[str, str]) -> None:
    if set(features) != set(EXPECTED_SHAPES):
        raise RuntimeError(
            "Oracle feature set drifted: "
            f"missing={sorted(set(EXPECTED_SHAPES) - set(features))}, "
            f"extra={sorted(set(features) - set(EXPECTED_SHAPES))}"
        )
    for name, value in features.items():
        if value.shape != EXPECTED_SHAPES[name]:
            raise RuntimeError(f"Oracle shape drift for {name}: expected {EXPECTED_SHAPES[name]}, actual {value.shape}")
        if str(value.dtype) != expected_dtypes[name]:
            raise RuntimeError(f"Oracle dtype drift for {name}: expected {expected_dtypes[name]}, actual {value.dtype}")


def _write_npz(path: Path, features: dict[str, np.ndarray]) -> None:
    object_arrays = [name for name, value in features.items() if value.dtype.hasobject]
    if object_arrays:
        raise RuntimeError(f"Refusing to pickle object arrays in NPZ: {object_arrays}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary_path = Path(handle.name)
        np.savez_compressed(handle, **{name: features[name] for name in sorted(features)})
    temporary_path.replace(path)
    path.chmod(0o644)


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(payload)
    temporary_path.replace(path)
    path.chmod(0o644)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oss-root", type=Path, required=True, help="Pinned aqlaboratory/openfold checkout")
    args = parser.parse_args()
    oss_root = args.oss_root.expanduser().resolve()

    _assert_no_self_reference()
    sys.meta_path.insert(0, _BlockBioIROpenFold2Imports())
    source_hashes = _verify_checkout(oss_root)
    source_cif = oss_root / INPUT_RELATIVE_PATH
    if not source_cif.is_file():
        raise FileNotFoundError(f"Pinned 4zey input is missing: {source_cif}")
    source_payload = source_cif.read_bytes()
    if _sha256_bytes(source_payload) != INPUT_SHA256:
        raise RuntimeError("Pinned 4zey input hash does not match the fixture pin")

    modules = _load_openfold_modules(oss_root)
    raw_features, warnings = _generate_features(source_cif, modules)
    _validate_contract(raw_features, EXPECTED_SOURCE_DTYPES)
    safe_features = {name: _pickle_free_array(value) for name, value in raw_features.items()}
    _validate_contract(safe_features, EXPECTED_ARCHIVE_DTYPES)
    if float(safe_features["template_all_atom_mask"].sum()) < 5.0:
        raise RuntimeError("Generated template oracle is effectively empty")

    _write_bytes(CIF_PATH, source_payload)
    _write_npz(ARTIFACT_PATH, safe_features)
    artifact_hash = _sha256_file(ARTIFACT_PATH)
    resolved_config = {
        "sample_id": SAMPLE_ID,
        "query_sequence": QUERY_SEQUENCE,
        "chain_id": CHAIN_ID,
        "oracle_mode": "openfold_identity_extract",
        "zero_center_positions": True,
        "template_sum_probs": 1.0,
        "oss_git_commit": OPENFOLD_COMMIT,
        "input_sha256": INPUT_SHA256,
        "serialization": "numpy_npz_no_object_arrays",
        "critical_oss_source_hashes": source_hashes,
    }
    command = shlex.join(
        [
            sys.executable,
            _repo_relative(SCRIPT_PATH),
            "--oss-root",
            str(oss_root),
        ]
    )
    provenance = {
        "schema_version": 1,
        "sample_id": SAMPLE_ID,
        "source": "oss",
        "reference_kind": "oss_featurization_primitives",
        "artifact_path": _repo_relative(ARTIFACT_PATH),
        "artifact_format": "numpy_npz_no_object_arrays",
        "artifact_sha256": artifact_hash,
        "sha256": artifact_hash,
        "input_path": _repo_relative(CIF_PATH),
        "input_sha256": INPUT_SHA256,
        "input_source_path": INPUT_RELATIVE_PATH.as_posix(),
        "input_source_url": (
            f"https://github.com/aqlaboratory/openfold/blob/{OPENFOLD_COMMIT}/{INPUT_RELATIVE_PATH.as_posix()}"
        ),
        "generation_command": command,
        "generator_path": _repo_relative(SCRIPT_PATH),
        # ast.dump output is stable only within a Python minor (new AST fields
        # appear across minors), so the pin is comparable only on the minor that
        # produced it. Record that minor; the test gates the check on it.
        "generator_ast_python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "generator_ast_sha256": _sha256_ast(SCRIPT_PATH),
        "oss_source_path": "openfold/data/templates.py",
        "oss_source_url": (
            f"https://github.com/aqlaboratory/openfold/blob/{OPENFOLD_COMMIT}/openfold/data/templates.py"
        ),
        "oss_source_files": source_hashes,
        "oss_git_commit": OPENFOLD_COMMIT,
        "oss_parser": "openfold.data.mmcif_parsing.parse",
        "oss_input_format": "mmCIF",
        "oracle_mode": "openfold_identity_extract",
        "checkpoint_identifier": "not_applicable_feature_only",
        "checkpoint_sha256": None,
        "resolved_config": resolved_config,
        "resolved_config_hash": _canonical_json_hash(resolved_config),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "biopython": importlib.metadata.version("biopython"),
            "openfold_import_mode": "pinned_source_data_namespace",
        },
        "source_feature_inventory": _inventory(raw_features),
        "feature_inventory": _inventory(safe_features),
        "string_serialization": {
            "template_domain_names": "object bytes -> fixed-width bytes",
            "template_sequence": "object bytes -> fixed-width bytes",
        },
        "warnings": warnings,
    }
    _write_bytes(PROVENANCE_PATH, (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    _assert_no_self_reference()
    print(f"Wrote {ARTIFACT_PATH} ({artifact_hash})")
    print(f"Wrote {CIF_PATH} ({INPUT_SHA256})")
    print(f"Wrote {PROVENANCE_PATH}")


if __name__ == "__main__":
    main()
