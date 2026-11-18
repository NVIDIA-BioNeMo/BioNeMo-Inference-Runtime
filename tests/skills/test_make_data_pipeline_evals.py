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

"""Controls for the onboarded skill eval fixtures. These do not run live agents."""

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ("make-data-pipeline",)


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _skill(name: str) -> Path:
    return ROOT / "skills" / name


def _entries(name: str) -> list[dict]:
    return json.loads((_skill(name) / "evals" / "evals.json").read_text())["evals"]


def _grader(name: str) -> ModuleType:
    return _load(f"{name.replace('-', '_')}_eval_grader", _skill(name) / "evals" / "grader.py")


def _artifact(entry: dict) -> dict:
    """Build the decision object implied by the verifier checks."""
    document: dict = {}
    for check in entry["domain_checks"]:
        if check["artifact"] != "answer.json":
            raise AssertionError(f"{entry['id']} checks an unexpected artifact")
        node = document
        keys = check.get("path", [])
        for key in keys[:-1]:
            child = node.setdefault(key, {})
            if not isinstance(child, dict):
                raise AssertionError(f"{entry['id']} mixes a leaf with children")
            node = child
        node[keys[-1]] = copy.deepcopy(check["expected"])
    return document


def _grade(name: str, entry: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    (output / "answer.json").write_text(json.dumps(_artifact(entry)))
    return _grader(name).assess(entry, output, _skill(name) / "evals" / "files")


@pytest.mark.parametrize("name", SKILLS)
def test_eval_contract_and_fixture_hashes(name: str) -> None:
    """Every case names its fixture, checks a decision, and keeps one negative."""
    payload = json.loads((_skill(name) / "evals" / "evals.json").read_text())
    assert payload["skill_name"] == name
    negatives = 0
    for entry in payload["evals"]:
        assert entry["id"].startswith(name)
        assert entry["assertions"]
        assert entry["domain_checks"]
        for relative in entry["files"]:
            assert (_skill(name) / relative).is_dir()
        if "expected_skill" in entry:
            assert entry["expected_skill"] is None
            negatives += 1
    assert negatives >= 1


@pytest.mark.parametrize("name", SKILLS)
def test_reference_decisions_pass_and_one_fault_fails(name: str, tmp_path: Path) -> None:
    """A document that satisfies every check scores 1, and a flipped leaf scores 0."""
    for entry in _entries(name):
        reward = _grade(name, entry, tmp_path / entry["id"])
        assert reward["overall"] == 1.0, reward["details"]
        faulted = _artifact(entry)
        leaf = faulted
        while isinstance(next(iter(leaf.values())), dict):
            leaf = next(iter(leaf.values()))
        key, value = next(iter(leaf.items()))
        if isinstance(value, bool):
            leaf[key] = not value
        elif isinstance(value, str):
            leaf[key] = value + "-fault"
        elif isinstance(value, list):
            leaf[key] = [*value, "fault"]
        else:
            raise AssertionError(f"{entry['id']} has no boolean, string, or list leaf")
        root = tmp_path / f"{entry['id']}-fault"
        root.mkdir()
        (root / "answer.json").write_text(json.dumps(faulted))
        assert _grader(name).assess(entry, root, _skill(name) / "evals" / "files")["overall"] == 0.0


@pytest.mark.parametrize("name", SKILLS)
def test_evaluator_seals_paired_plan_with_declared_grader_metrics(name: str, tmp_path: Path) -> None:
    """Exercise CI's pre-execution contract when its evaluator is available."""
    native = pytest.importorskip("skillevaluator.tier3.harbor.native_contract")
    from skillevaluator.tier3.evals_config import load_evals_config
    from skillevaluator.tier3.harbor import ENV_MODE_ASTRA_SANDBOX
    from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks
    from skillevaluator.tier3.output_provenance import GENERATED_OUTPUT_MARKER

    skill = _skill(name)
    config, _ = load_evals_config(skill)
    mode = config["grading"]["mode"]
    roots = {arm: tmp_path / "staged" / arm for arm in ("with_skill", "baseline")}
    tasks = {}
    for arm, root in roots.items():
        tasks[arm] = generate_harbor_tasks(
            skill, root, with_skill=arm == "with_skill", grading_mode=mode, env_mode=ENV_MODE_ASTRA_SANDBOX
        )
        for generated in ("dataset.toml", "metric.py", GENERATED_OUTPUT_MARKER):
            (root / generated).unlink(missing_ok=True)
    agents = [
        {"agent": agent, "result_agent": agent, "occurrence": 1, "model": model, "model_source": source}
        for agent, model, source in (
            ("claude-code", "aws/anthropic/bedrock-claude-opus-5-5", "ANTHROPIC_MODEL"),
            ("codex", "openai/openai/gpt-6.1-sol", "CODEX_MODEL"),
        )
    ]
    plan_arguments = {
        "run_dir": tmp_path,
        "run_id": f"{name}-cpu-contract",
        "skill_path": skill,
        "task_source": "evals_json",
        "evals_file": skill / "evals" / "evals.json",
        "native_harbor_dir": skill / "evals" / "harbor",
        "evals_config": config,
        "grading_mode": mode,
        "agent_entries": agents,
        "task_paths": tasks["with_skill"],
        "with_skill_root": roots["with_skill"],
        "baseline_root": roots["baseline"],
        "skip_baseline": False,
        "n_attempts": 1,
        "stop_on_pass": False,
        "pass_threshold": 0.5,
        "agent_validity_policy": "any-valid",
        "min_valid_agents": 1,
        "required_agents": [],
        "contract_requests": ["agent-coverage/1", "tier3-result/3"],
    }
    undeclared = copy.deepcopy(config)
    undeclared["grading"].pop("custom_metrics")
    with pytest.raises(native.ContractError, match="requires declared metrics and a schema digest"):
        native.seal_plan(**{**plan_arguments, "evals_config": undeclared})
    sealed = native.seal_plan(**plan_arguments)
    reward_contract = sealed.plan["reward_contract"]
    metrics = {item["name"] for item in reward_contract["custom_metrics"]}
    assert metrics == set(_grader(name).assess({}, tmp_path, tmp_path)["custom_metrics"])
    assert reward_contract["custom_grader_schema_digest"].startswith("sha256:")
    assert len(tasks["with_skill"]) == len(tasks["baseline"]) == len(_entries(name))


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_oversized_numeric_output_receives_a_failed_score(tmp_path: Path, value: int) -> None:
    """A huge JSON integer is a failed close check, and the grader still returns."""
    entry = {
        "domain_checks": [
            {"label": "score", "artifact": "answer.json", "op": "close", "path": ["score"], "expected": 1.0},
            {"label": "label", "artifact": "answer.json", "op": "equals", "path": ["label"], "expected": "ok"},
        ]
    }
    output = tmp_path / "out"
    output.mkdir()
    (output / "answer.json").write_text(json.dumps({"score": value, "label": "ok"}))
    reward = _grader("make-data-pipeline").assess(entry, output, tmp_path)
    assert reward["overall"] == 0.0
    assert reward["details"]["score"]["passed"] is False
    assert reward["details"]["label"]["passed"] is True


@pytest.mark.parametrize("operation", ["close", "symlink_to"])
@pytest.mark.parametrize("location", ["artifact", "parent"])
def test_cyclic_output_paths_receive_a_failed_score(tmp_path: Path, operation: str, location: str) -> None:
    """A cyclic artifact or parent symlink fails the check instead of aborting the grader."""
    output = tmp_path / "output"
    output.mkdir()
    (output / "cycle").symlink_to("cycle")
    relative = "cycle" if location == "artifact" else "cycle/answer.json"
    check = {
        "label": "cyclic artifact",
        "artifact": relative,
        "path": ["value"],
        "op": operation,
        "expected": "request.json" if operation == "symlink_to" else 1.0,
    }
    reward = _grader("make-data-pipeline").assess({"domain_checks": [check]}, output, tmp_path)
    assert reward["overall"] == 0.0
    assert reward["details"]["cyclic artifact"]["passed"] is False


def test_cyclic_input_checksum_receives_a_failed_score(tmp_path: Path) -> None:
    """A cyclic input symlink fails the checksum and leaves the other check scored."""
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "fixture.json").symlink_to("fixture.json")
    output = tmp_path / "output"
    output.mkdir()
    (output / "answer.json").write_text(json.dumps({"label": "ok"}))
    entry = {
        "domain_checks": [
            {"label": "label", "artifact": "answer.json", "op": "equals", "path": ["label"], "expected": "ok"}
        ],
        "input_sha256": {"fixture.json": "0" * 64},
    }
    reward = _grader("make-data-pipeline").assess(entry, output, inputs)
    assert reward["overall"] == 0.0
    assert reward["details"]["label"]["passed"] is True
    assert reward["details"]["unchanged:fixture.json"]["passed"] is False
