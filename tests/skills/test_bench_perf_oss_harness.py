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
"""CPU checks for the shared helpers the bench-perf-oss skill ships."""

import copy
import importlib.util
import json
import stat
import subprocess
import threading
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[2] / ".agents" / "skills" / "bench-perf-oss" / "harness"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bench_perf_oss_{name}", HARNESS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timing = _load("timing")
scoring = _load("scoring")
telemetry = _load("gpu_telemetry")
speedup = _load("summarize_speedup")
check = _load("check_results")


def _row(residues, forward_s, **extra):
    row = {
        "residues": residues,
        "status": "ok",
        "forward_s": [forward_s],
        "peak_alloc_gb": 1.0,
        "peak_reserved_gb": 1.5,
        "gpu_telemetry": {"n_polls": 3, "n_busy": 3, "throttle": []},
        "unpaired_msas": [],
        "paired_msas": [],
        "templates": [],
        "template_status": "none_declared",
        "n_templates_supplied": 0,
        "n_templates_attached": 0,
        "lddt": 0.8,
        "dockq": None,
        "dockq_status": "single_chain",
        "dockq_scope": "protein_only",
    }
    row.update(extra)
    return row


def _result(backend, samples):
    data = {
        "backend": backend,
        "model_source": "boltz-2",
        "gpu": {"device_name": "test"},
        "config": {"recycling_steps": 3},
        "warmup": 1,
        "repeats": 1,
        "samples": samples,
        "gpu_telemetry": {
            "n_ok": sum(row.get("status") == "ok" for row in samples.values()),
            **dict.fromkeys(check.TELEMETRY_FIELDS),
        },
    }
    if backend == "oss_compile":
        data["compile_stats"] = {"measurement_stable": True, "compile_dynamic": None}
        for row in samples.values():
            row.setdefault("warmup_compile_delta", 1)
            row.setdefault("measure_compile_delta", 0)
    return data


def _write_results(results, bioir, eager, compiled=None):
    results.mkdir(parents=True, exist_ok=True)
    (results / "bioir_forward.json").write_text(json.dumps(_result("bioir_build_processor", bioir)))
    (results / "oss_eager.json").write_text(json.dumps(_result("oss_eager", eager)))
    if compiled is not None:
        (results / "oss_compile.json").write_text(json.dumps(_result("oss_compile", compiled)))


def test_residues_sum_every_polymer_chain():
    item = {
        "polymers": [
            {"polymer_type": "protein", "chain_id": "A", "sequence": "M" * 867},
            {"polymer_type": "protein", "chain_id": "B", "sequence": "M" * 867},
            {"polymer_type": "rna", "chain_id": ["C", "D"], "sequence": "ACGU"},
            {"polymer_type": "ligand", "chain_id": "E", "sequence": "ATP"},
        ]
    }
    assert timing.residues_all_chains(item) == 867 + 867 + 2 * 4


def test_speedup_bins_and_empty_bin(tmp_path):
    results = tmp_path / "results"
    bioir = {"a": _row(100, 1.0), "b": _row(300, 2.0), "c": _row(800, 4.0)}
    eager = {"a": _row(100, 2.0), "b": _row(300, 8.0), "c": _row(800, 4.0)}
    _write_results(results, bioir, eager)

    payload = speedup.summarize(results)

    block = payload["vs_eager"]
    assert block["overall"]["n"] == 3
    assert block["overall"]["geomean"] == pytest.approx((2.0 * 4.0 * 1.0) ** (1 / 3))
    assert block["overall"]["median"] == pytest.approx(2.0)
    assert block["short"]["ids"] == ["a", "b"]
    assert block["medium"]["ids"] == ["c"]
    assert block["long"] == {"n": 0, "geomean": None, "median": None, "ids": [], "rule": "residues > 1024"}
    assert "vs_compile" not in payload
    assert json.loads((results / "speedup.json").read_text()) == payload


def test_speedup_skips_rows_not_ok_on_both_sides(tmp_path):
    results = tmp_path / "results"
    bioir = {"a": _row(100, 1.0), "b": _row(300, 2.0)}
    eager = {"a": _row(100, 3.0), "b": {"residues": 300, "status": "oom"}}
    _write_results(results, bioir, eager, compiled={"a": _row(100, 1.5)})

    payload = speedup.summarize(results)

    assert payload["vs_eager"]["overall"]["ids"] == ["a"]
    assert payload["vs_compile"]["overall"]["geomean"] == pytest.approx(1.5)


def test_valid_results_have_no_problems():
    samples = {"a": _row(100, 1.0), "b": _row(300, 2.0, dockq=0.6, dockq_status="ok")}
    files = {
        "bioir": _result("bioir_build_processor", copy.deepcopy(samples)),
        "eager": _result("oss_eager", copy.deepcopy(samples)),
        "compile": _result("oss_compile", copy.deepcopy(samples)),
    }
    assert [p for data in files.values() for p in check.check_file(data)] == []
    assert check.check_pair(files) == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"dockq": 0.0}, "dockq must be null"),
        ({"forward_s": [1.0, 1.1]}, "forward_s must be one positive time"),
        ({"residues": 0}, "residues must be a positive int"),
        ({"status": "error"}, "status 'error' not in"),
        ({"template_status": "dropped"}, "template_status 'dropped' not in"),
        ({"n_templates_attached": 2, "n_templates_supplied": 1}, "n_templates_attached <= n_templates_supplied"),
        ({"gpu_telemetry": None}, "gpu_telemetry missing"),
        ({"gpu_telemetry": {}}, "gpu_telemetry.n_polls must be a positive int"),
        ({"gpu_telemetry": {"n_polls": 2, "n_busy": 3}}, "gpu_telemetry.n_busy must be in [0, n_polls]"),
        ({"lddt": 81.0}, "lddt must be null or in [0, 1]"),
    ],
)
def test_bad_row_is_reported(change, message):
    row = _row(100, 1.0)
    row.update(change)
    data = _result("bioir_build_processor", {"a": row})
    assert any(message in p for p in check.check_file(data))


def test_file_telemetry_must_cover_every_ok_row():
    data = _result("bioir_build_processor", {"a": _row(100, 1.0), "b": _row(300, 2.0)})
    data["gpu_telemetry"]["n_ok"] = 1
    assert any("gpu_telemetry.n_ok must be 2" in p for p in check.check_file(data))


@pytest.mark.parametrize("key", ["n_templates_supplied", "n_templates_attached"])
def test_boolean_counts_are_rejected(key):
    data = _result("bioir_build_processor", {"a": _row(100, 1.0, **{key: True})})
    assert any("n_templates_attached <= n_templates_supplied" in p for p in check.check_file(data))


def test_boolean_compile_delta_is_rejected():
    data = _result("oss_compile", {"a": _row(100, 1.0, measure_compile_delta=False)})
    assert any("measure_compile_delta missing" in p for p in check.check_file(data))


def test_malformed_samples_are_reported_not_raised():
    good = _result("bioir_build_processor", {"a": _row(100, 1.0)})
    listed = _result("oss_eager", {"a": _row(100, 1.0)})
    listed["samples"] = [listed["samples"]["a"]]
    bad_row = _result("oss_eager", {"a": _row(100, 1.0)})
    bad_row["samples"]["a"] = "oom"

    assert any("samples must be a non-empty object" in p for p in check.check_file(listed))
    assert any("samples[a] must be an object" in p for p in check.check_file(bad_row))
    assert check.check_pair({"bioir": good, "eager": listed}) == []
    assert check.check_pair({"bioir": good, "eager": bad_row}) == []


def test_compile_recapture_on_measured_forward_is_reported():
    data = _result("oss_compile", {"a": _row(100, 1.0, measure_compile_delta=1)})
    assert any("measured forward recompiled" in p for p in check.check_file(data))


def test_sides_must_agree_on_samples_and_templates():
    bioir = _result("bioir_build_processor", {"a": _row(100, 1.0, n_templates_supplied=4, n_templates_attached=4)})
    eager = _result(
        "oss_eager",
        {"a": _row(100, 1.0, n_templates_supplied=4, n_templates_attached=3), "b": _row(300, 1.0)},
    )
    problems = check.check_pair({"bioir": bioir, "eager": eager})
    assert any("cover different samples" in p for p in problems)
    assert any("n_templates_attached differs" in p for p in problems)


@pytest.mark.parametrize(
    ("types", "n_chains", "has_gt", "expected"),
    [
        (["protein"], 2, False, "no_gt"),
        (["protein"], 1, True, "single_chain"),
        (["protein", "dna"], 3, True, "unsupported_chain_types"),
        (["rna"], 1, True, "unsupported_chain_types"),
        (["protein", "rna"], 2, False, "unsupported_chain_types"),
        (["protein", "ligand"], 2, True, None),
    ],
)
def test_dockq_skip_status(types, n_chains, has_gt, expected):
    assert scoring.dockq_skip_status(types, n_chains, has_gt) == expected


@pytest.mark.parametrize(("value", "band"), [(0.1, "incorrect"), (0.23, "acceptable"), (0.5, "medium"), (0.8, "high")])
def test_dockq_band(value, band):
    assert scoring.dockq_band(value) == band


def _fake_tool(path, body):
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_score_dockq_reads_global_dockq(tmp_path):
    report = {"GlobalDockQ": 0.62, "best_mapping_str": "AB:AB", "best_result": {"AB": {"DockQ": 0.62}}}
    tool = _fake_tool(tmp_path / "DockQ", f"echo '{json.dumps(report)}' > \"$4\"\n")

    row = scoring.score_dockq(
        tool, "model.cif", "native.cif", tmp_path / "out.json", ["--short"], polymer_types=["protein"], n_chains=2
    )

    assert row["dockq"] == pytest.approx(0.62)
    assert row["dockq_status"] == "ok"
    assert row["dockq_scope"] == "protein_only"
    assert row["dockq_mapping"] == "AB:AB"


def test_score_dockq_without_native_interface_is_not_zero(tmp_path):
    tool = _fake_tool(tmp_path / "DockQ", f"echo '{scoring.NO_NATIVE_INTERFACES}' >&2\nexit 1\n")

    row = scoring.score_dockq(
        tool, "model.cif", "native.cif", tmp_path / "out.json", [], polymer_types=["protein"], n_chains=2
    )

    assert row["dockq"] is None
    assert row["dockq_status"] == "no_native_interfaces"


def test_report_without_lddt_is_a_failed_score(tmp_path):
    ost = _fake_tool(tmp_path / "ost", 'echo \'{"lddt": null}\' > "$7"\n')

    row = scoring.score_lddt(ost, "m.cif", "r.cif", tmp_path / "ost.json")

    assert row["lddt"] is None
    assert "no lddt value" in row["lddt_stderr"]


def test_unreadable_reports_are_failed_scores(tmp_path):
    ost = _fake_tool(tmp_path / "ost", 'echo "not json" > "$7"\n')
    dockq = _fake_tool(tmp_path / "DockQ", 'echo \'{"best_mapping_str": "AB:AB"}\' > "$4"\n')

    lddt = scoring.score_lddt(ost, "m.cif", "r.cif", tmp_path / "ost.json")
    row = scoring.score_dockq(
        dockq, "m.cif", "r.cif", tmp_path / "dockq.json", [], polymer_types=["protein"], n_chains=2
    )

    assert lddt["lddt"] is None
    assert "invalid lDDT report" in lddt["lddt_stderr"]
    assert (row["dockq"], row["dockq_status"]) == (None, "failed")
    assert "invalid DockQ report" in row["dockq_stderr"]


def test_score_lddt_failure_keeps_stderr(tmp_path):
    ok = _fake_tool(tmp_path / "ost_ok", 'echo \'{"lddt": 0.81}\' > "$7"\n')
    bad = _fake_tool(tmp_path / "ost_bad", "echo boom >&2\nexit 2\n")

    assert scoring.score_lddt(ok, "m.cif", "r.cif", tmp_path / "ok.json")["lddt"] == pytest.approx(0.81)
    failed = scoring.score_lddt(bad, "m.cif", "r.cif", tmp_path / "bad.json")
    assert failed["lddt"] is None
    assert "boom" in failed["lddt_stderr"]


def test_nvidia_smi_timeout_is_passed_and_survived(monkeypatch):
    seen = {}
    ready = threading.Event()

    def hung(*_args, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        ready.set()
        raise subprocess.TimeoutExpired("nvidia-smi", kwargs.get("timeout"))

    monkeypatch.setattr(telemetry.subprocess, "check_output", hung)
    sampler = telemetry.GpuSampler(interval_s=0.01)
    sampler.start()
    assert ready.wait(1.0)
    summary = sampler.stop()

    assert seen["timeout"] == telemetry.SMI_TIMEOUT_S
    assert summary["n_polls"] == 0


def test_sampler_that_cannot_stop_fails_loudly(monkeypatch):
    release = threading.Event()
    ready = threading.Event()

    def blocked():
        ready.set()
        release.wait(5.0)
        return {}

    monkeypatch.setattr(telemetry, "query_gpu_snapshot", blocked)
    sampler = telemetry.GpuSampler(interval_s=0.01, stop_timeout_s=0.05)
    sampler.start()
    try:
        assert ready.wait(1.0)
        with pytest.raises(RuntimeError, match="did not stop"):
            sampler.stop()
        with pytest.raises(RuntimeError, match="already running"):
            sampler.start()
    finally:
        release.set()


def test_telemetry_summaries_use_busy_polls_only():
    polls = [
        {"gpu_idle": "Active", "power_draw_w": 60.0, "clocks_sm_mhz": 300.0},
        {"gpu_idle": "Not Active", "power_draw_w": 500.0, "clocks_sm_mhz": 1800.0, "sw_power_cap": "Active"},
        {"gpu_idle": "Not Active", "power_draw_w": 600.0, "clocks_sm_mhz": 1700.0},
    ]
    summary = telemetry.summarize_polls(polls)
    assert summary["n_polls"] == 3
    assert summary["n_busy"] == 2
    assert summary["power_draw_w"] == {"min": 500.0, "max": 600.0, "mean": 550.0}
    assert summary["throttle"] == ["sw_power_cap"]

    aggregate = telemetry.aggregate_file_telemetry(
        {"a": {"status": "ok", "gpu_telemetry": summary}, "b": {"status": "oom", "gpu_telemetry": summary}}
    )
    assert aggregate["n_ok"] == 1
    assert aggregate["clocks_sm_mhz"] == {"min": 1700.0, "max": 1800.0, "mean": 1750.0}


def test_plot_writes_both_charts(tmp_path):
    pytest.importorskip("matplotlib")
    plot = _load("plot_latency")
    results = tmp_path / "results"
    _write_results(results, {"a": _row(100, 1.0)}, {"a": _row(100, 2.0)})
    config = tmp_path / "bench_config.json"
    config.write_text(json.dumps({"model_source": "boltz-2", "gpu_name": "test"}))

    written = plot.plot(results, config)

    assert [p.name for p in written] == ["latency_vs_residues.png", "speedup_vs_residues.png"]
    assert all(p.stat().st_size > 0 for p in written)
