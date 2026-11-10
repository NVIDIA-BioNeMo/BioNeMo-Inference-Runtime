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

"""Controls for the eval fixtures and verifier; these do not run live agents."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parents[2] / "skills" / "bench-perf-oss"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


grader = _load("bench_perf_oss_eval_grader", SKILL / "evals" / "grader.py")
contract = _load("bench_perf_oss_eval_contract", SKILL / "harness" / "check_results.py")
summary = _load("bench_perf_oss_eval_summary", SKILL / "harness" / "summarize_speedup.py")
ENTRIES = json.loads((SKILL / "evals" / "evals.json").read_text())["evals"]


def _write(root: Path, relative: str, data: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _entry(number: int) -> dict:
    return ENTRIES[number - 1]


def _grade(number: int, root: Path) -> dict:
    return grader.assess(_entry(number), root, SKILL / "evals" / "files")


def test_aggregation_fixture_is_valid_and_has_independent_numeric_oracle(tmp_path: Path) -> None:
    """Use the real helper on the fixture, including both bin boundaries."""
    source = SKILL / "evals" / "files" / "aggregation" / "results"
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    inputs = {p.name: json.loads(p.read_text()) for p in source.glob("*.json")}
    for name, data in inputs.items():
        assert contract.check_file(data) == []
        _write(result_dir, name, data)
    assert contract.check_pair(inputs) == []
    data = summary.summarize(result_dir)
    assert data["vs_eager"]["overall"]["geomean"] == pytest.approx(32**0.25)
    assert data["vs_eager"]["overall"]["median"] == 3.0
    assert data["vs_eager"]["medium"]["ids"] == ["medium-low", "medium-high"]
    assert _grade(6, tmp_path)["overall"] == 1.0


@pytest.mark.parametrize("fault", ["arithmetic_mean", "wrong_bin", "include_oom", "nan", "bool_count"])
def test_aggregation_grader_rejects_wrong_statistics(tmp_path: Path, fault: str) -> None:
    source = SKILL / "evals" / "files" / "aggregation" / "results"
    for path in source.glob("*.json"):
        _write(tmp_path / "results", path.name, json.loads(path.read_text()))
    data = summary.summarize(tmp_path / "results")
    if fault == "arithmetic_mean":
        data["vs_eager"]["overall"]["geomean"] = 3.625
    elif fault == "wrong_bin":
        data["vs_eager"]["short"]["ids"].append("medium-low")
    elif fault == "include_oom":
        data["vs_eager"]["overall"]["ids"].append("oom-bioir")
    elif fault == "nan":
        data["vs_eager"]["overall"]["geomean"] = float("nan")
    else:
        data["vs_eager"]["short"]["n"] = True
    _write(tmp_path, "results/speedup.json", data)
    assert _grade(6, tmp_path)["overall"] == 0.0


def test_audit_fixture_contains_the_intended_faults_and_a_clean_control() -> None:
    source = SKILL / "evals" / "files" / "result-audit" / "results"
    inputs = {p.name: json.loads(p.read_text()) for p in source.glob("*.json")}
    assert contract.check_file(inputs["bioir_forward.json"]) == []
    assert contract.check_file(inputs["oss_eager.json"]) == []
    problems = contract.check_file(inputs["oss_compile.json"])
    for sample_id in ("recaptured", "monomer-zero", "no-polls"):
        assert any(sample_id in problem for problem in problems)
    assert not any("samples[clean]" in problem for problem in problems)
    assert any("template-mismatch" in problem for problem in contract.check_pair(inputs))


ANSWERS = {
    2: {"status": "blocked", "sample_status": {"clean": "ready", "template-gap": "blocked", "paired-gap": "blocked"}},
    3: {
        "model_source": "protenix-v2",
        "feature_path": "processor",
        "bioir": {"recycling_steps": 5, "num_sampling_steps": 200, "diffusion_samples": 5},
        "oss": {"model.N_cycle": 6, "sample_diffusion.N_step": 200, "sample_diffusion.N_sample": 5},
        "seed": 101,
        "oss_compile": False,
        "cuda_wheel": "cuequivariance-ops-torch-cu13",
        "ready_to_run": False,
        "weights": {"strict_load_required": True, "download_from_hub": False},
    },
    4: {
        "bioir_nonquery_deletion_sum": 42,
        "oss_nonquery_deletion_sum": 0,
        "modify_bioir_deletions": False,
        "can_attribute_gap_to_engine": False,
    },
    5: {
        "publish": False,
        "clean_sample_ok": True,
        "issues_by_sample": {
            "recaptured": ["measure_compile_delta"],
            "monomer-zero": ["dockq"],
            "template-mismatch": ["n_templates_attached"],
            "no-polls": ["gpu_telemetry.n_polls"],
        },
    },
    7: {
        "status": "blocked",
        "required_oss_tag": "0.4.3",
        "allow_non_strict_load": False,
        "allow_hub_requests_in_timed_loop": False,
    },
    8: {"bioir_mae": 2.0, "oss_mae": 1.0},
    9: {"total_items": 250, "wall_time_s": 10.0, "items_per_s": 25.0},
    10: {
        "model_source": "alphafold2_multimer_1",
        "oss_preset": "model_1_multimer_v3",
        "recycling_steps": 3,
        "bioir_model": {"max_recycling_iters": 2, "recycle_early_stop_tolerance": -1},
        "oss_model": {"max_recycling_iters": 2, "recycle_early_stop_tolerance": -1},
        "in_scope_ids": ["protein-pair"],
        "out_of_scope_ids": ["protein-ligand", "rna"],
    },
}


@pytest.mark.parametrize("number", ANSWERS)
def test_domain_checks_accept_reference_artifacts_and_reject_empty_outputs(tmp_path: Path, number: int) -> None:
    artifact = "ref_data/bench_config.json" if number in {3, 10} else "answer.json"
    _write(tmp_path, artifact, ANSWERS[number])
    assert _grade(number, tmp_path)["overall"] == 1.0
    _write(tmp_path, artifact, {})
    assert _grade(number, tmp_path)["overall"] == 0.0


@pytest.mark.parametrize(
    ("number", "key", "wrong"),
    [
        (3, "ready_to_run", True),
        (4, "modify_bioir_deletions", True),
        (5, "publish", True),
        (7, "allow_non_strict_load", True),
    ],
)
def test_grader_rejects_unsafe_decisions_even_when_other_checks_pass(
    tmp_path: Path, number: int, key: str, wrong: bool
) -> None:
    answer = copy.deepcopy(ANSWERS[number])
    answer[key] = wrong
    artifact = "ref_data/bench_config.json" if number == 3 else "answer.json"
    _write(tmp_path, artifact, answer)
    assert _grade(number, tmp_path)["overall"] == 0.0


@pytest.mark.parametrize("number", [3, 10])
def test_recycle_rules_cannot_be_transferred_between_model_families(tmp_path: Path, number: int) -> None:
    answer = copy.deepcopy(ANSWERS[number])
    if number == 3:
        answer["oss"]["model.N_cycle"] = 5
    else:
        answer["bioir_model"]["max_recycling_iters"] = 3
        answer["oss_model"]["max_recycling_iters"] = 3
    _write(tmp_path, "ref_data/bench_config.json", answer)
    assert _grade(number, tmp_path)["overall"] == 0.0


def test_manifest_checks_require_all_copies_and_distinct_staging_links(tmp_path: Path) -> None:
    source_root = SKILL / "evals" / "files"
    dataset = source_root / "manifest" / "dataset"
    spec = json.loads((dataset / "spec_full.json").read_text())
    samples = {}
    index = {"samples": {}}
    for item in spec["items"]:
        chains = {}
        residues = 0
        for polymer in item["polymers"]:
            ids = polymer["chain_id"]
            ids = [ids] if isinstance(ids, str) else ids
            if polymer["polymer_type"] in {"protein", "rna", "dna"}:
                residues += len(polymer["sequence"]) * len(ids)
            for chain_id in ids:
                chain = {}
                for key, source_key, stem, suffix in (
                    ("unpaired_msas", "msas", "unpaired", "a3m"),
                    ("paired_msas", "paired_msas", "paired", "a3m"),
                    ("templates", "templates", "template", "cif"),
                ):
                    paths = [dataset / record["path"] for record in polymer[source_key]]
                    # Harbor stages these under its fixed input root. The
                    # local grader reads links against the local corpus.
                    chain[key] = ["/workspace/input/" + str(p.relative_to(source_root)) for p in paths]
                    for i, path in enumerate(paths):
                        staged = tmp_path / "oss_data" / item["id"] / chain_id / f"{stem}_{i}.{suffix}"
                        staged.parent.mkdir(parents=True, exist_ok=True)
                        staged.symlink_to(path)
                chains[chain_id] = chain
        samples[item["id"]] = {"residues": residues, "in_scope": True, "chains": chains}
        index["samples"][item["id"]] = {"chain_ids": list(chains)}
    _write(tmp_path, "ref_data/sample_manifest.json", {"samples": samples})
    _write(tmp_path, "oss_data/index.json", index)
    script = tmp_path / "bench" / "stage_oss_data.py"
    script.parent.mkdir()
    script.write_text("# reference-control marker; no agent code is executed by these tests\n")
    assert _grade(1, tmp_path)["overall"] == 1.0
    samples["fixture-complex"]["chains"]["B"]["unpaired_msas"].pop()
    _write(tmp_path, "ref_data/sample_manifest.json", {"samples": samples})
    assert _grade(1, tmp_path)["overall"] == 0.0


def test_changed_input_cannot_be_accepted_as_valid_evidence(tmp_path: Path) -> None:
    entry = copy.deepcopy(_entry(8))
    input_root = tmp_path / "input"
    _write(input_root, "affinity/predictions.json", {"rows": []})
    _write(tmp_path, "answer.json", ANSWERS[8])
    assert grader.assess(entry, tmp_path, input_root)["overall"] == 0.0


def test_output_cannot_alias_verifier_or_input_data(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(ANSWERS[8]))
    output = tmp_path / "output"
    output.mkdir()
    (output / "answer.json").symlink_to(outside)
    assert _grade(8, output)["overall"] == 0.0
