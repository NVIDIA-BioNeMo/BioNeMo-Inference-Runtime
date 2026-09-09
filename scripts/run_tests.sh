#!/usr/bin/env bash
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

# Run the test suite the way CI does: stage model weights, then run pytest in
# two phases that share the on-disk kernel cache, so phase 1 warms the cache
# that phase 2 reuses.
#
#   phase 1: the xdist-safe bulk (worker count scaled to VRAM);
#   phase 2: the Ray/model_forwards trees, serial (they OOM / deadlock the
#            raylet under xdist).
#
# Both phases emit a JUnit XML under REPORT_DIR. Coverage is off by default —
# instrumentation costs wall-clock and nothing local reads the report. COVERAGE=1
# turns it on: both phases then share one data file, and the combined branch
# coverage is rendered as a terminal summary and HTML under REPORT_DIR. CI sets
# it, which is what feeds the published number.
#
# Weight download and staging live in scripts/fetch_weights.sh; this script
# invokes it and sources the env file it writes. Weights it cannot supply are
# skipped, and the tests that need them skip themselves — so this runs to
# completion with no credentials and no downloads.
#
# See docs/dev.md for the full development workflow.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Repo root the test phase cd's into. The CI test job sets PROJECT_WORKDIR
# explicitly; otherwise derive it from this script's location
# (<repo>/scripts/ -> one level up).
PROJECT_WORKDIR="${PROJECT_WORKDIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"

print_help() {
  cat <<EOF
run_tests.sh — stage model weights, then run the test suite.

Usage:
  run_tests.sh                  Stage weights, then run tests.
  run_tests.sh --download       Stage weights and exit (no tests).
  run_tests.sh --no-weights     Skip staging entirely; run whatever resolves.
  run_tests.sh -h | --help      Show this help.

Any option not listed here is forwarded to scripts/fetch_weights.sh, so
  run_tests.sh --source public
  run_tests.sh --download --model boltz
work as expected. See fetch_weights.sh --help for the full set.

Environment:
  TEST_PHASE      1 | 2 | all (default)
  PHASE2_TARGETS  paths phase 2 runs
  PHASE2_ADDOPTS  extra pytest args for phase 2
  XDIST_WORKERS   phase 1 worker count (default: scaled to VRAM)
  COVERAGE        1 to enable coverage instrumentation (default: 0; CI sets 1)
  REPORT_DIR      where JUnit/coverage/metrics land
EOF
}

DOWNLOAD_ONLY=0
STAGE_WEIGHTS=1
declare -a FETCH_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --download) DOWNLOAD_ONLY=1 ;;
    --no-weights) STAGE_WEIGHTS=0 ;;
    -h | --help)
      print_help
      exit 0
      ;;
    *) FETCH_ARGS+=("$1") ;;
  esac
  shift
done

# Contradictory: --download means "stage, then stop", --no-weights means "stage
# nothing". Accepting both would report success over an empty cache.
if ((DOWNLOAD_ONLY)) && ((!STAGE_WEIGHTS)); then
  echo "--download and --no-weights contradict each other" >&2
  exit 2
fi

# Per-phase timing so the CI log shows where run_tests.sh spends its time.
_t0=$SECONDS
_lap=$_t0
lap() {
  local now=$SECONDS
  echo ">>> [timing] $1: $((now - _lap))s (elapsed $((now - _t0))s)"
  _lap=$now
}

WEIGHTS_ENV_FILE="${WEIGHTS_ENV_FILE:-${TMPDIR:-/tmp}/bioir-weights.$$.env}"
export WEIGHTS_ENV_FILE
if ((STAGE_WEIGHTS)); then
  bash "${SCRIPT_DIR}/fetch_weights.sh" ${FETCH_ARGS[@]+"${FETCH_ARGS[@]}"}
  lap "stage weights"
else
  : >"${WEIGHTS_ENV_FILE}"
  echo "skipping weight staging (--no-weights)"
fi

if ((DOWNLOAD_ONLY)); then
  echo ">>> [timing] run_tests.sh --download total: $((SECONDS - _t0))s"
  echo "Staging complete. pytest resolves these via the cache probe" \
    "— no *_CKPT needed."
  exit 0
fi

# Weight paths for this run's own pytest children. Staging into the probe dirs
# already covers a separate pytest process.
# shellcheck source=/dev/null
[[ -s "${WEIGHTS_ENV_FILE}" ]] && source "${WEIGHTS_ENV_FILE}"

cd "${PROJECT_WORKDIR}"
# The package is installed by the caller (CI's before_script, or `pip install -e .`
# on first attach in the dev container), so we do not reinstall here.

lap "setup (checkpoints + env)"
echo ">>> [timing] run_tests.sh setup total: $((SECONDS - _t0))s"

# Test reports (JUnit) + branch coverage.
#
# Everything lands under tmp/, which is already gitignored, so no module's
# reports need a new ignore rule. REPORT_DIR must sit inside the CI checkout,
# not PROJECT_WORKDIR: this script cd's into the image's baked source copy, and
# a CI job only uploads artifacts from its own checkout. Off CI it falls back to
# the repo root. Override REPORT_DIR to put them elsewhere.
REPORT_DIR="${REPORT_DIR:-${CI_PROJECT_DIR:-${PROJECT_WORKDIR}}/tmp/reports}"
# tmp/reports/coverage-data/ is the pipeline-wide drop point every test job
# writes into; the reporting job combines whatever it finds there into the one
# published number. This suite is one component of it — see
# coverage_aggregate.sh.
COVERAGE_DATA_DIR="${COVERAGE_DATA_DIR:-${REPORT_DIR}/coverage-data}"
COVERAGE_COMPONENT="${COVERAGE_COMPONENT:-gpu}"
mkdir -p "${REPORT_DIR}" "${COVERAGE_DATA_DIR}"
# Both phases write into one data file so this component's report spans the whole
# suite; parallel=true (pyproject.toml) makes each process/worker write its own
# .coverage.<host>.<pid> sidecar that pytest-cov combines back into this file.
export COVERAGE_FILE="${COVERAGE_DATA_DIR}/.coverage.${COVERAGE_COMPONENT}"
rm -f "${COVERAGE_FILE}" "${COVERAGE_FILE}".*

# Degrade to a plain run when pytest-cov is missing (older image): a report gap
# must not take the test job down.
COVERAGE="${COVERAGE:-0}"
if ((COVERAGE)) \
  && ! { python3 -c "import pytest_cov" && command -v coverage; } >/dev/null 2>&1; then
  echo "pytest-cov / coverage not installed — running without coverage"
  COVERAGE=0
fi
# Per-shard HTML: off on CI (coverage_aggregate.sh renders the combined superset),
# on everywhere else. Set COVERAGE_HTML explicitly to override either way.
COVERAGE_HTML="${COVERAGE_HTML:-${CI:+0}}"
COVERAGE_HTML="${COVERAGE_HTML:-1}"
# --cov-report= (empty) suppresses per-phase reports; the combined data is
# rendered once, after both phases, by report_coverage below. --cov-append adds
# phase 2 to phase 1's data instead of replacing it, so the total spans the whole
# suite. Both arrays are expanded as ${a[@]+"${a[@]}"} — plain "${a[@]}" on an
# empty array is an unbound-variable error under `set -u` on bash < 4.4.
cov_opts=()
cov_append_opts=()
if ((COVERAGE)); then
  cov_opts=(--cov=bionemo_ir --cov-report=)
  cov_append_opts=("${cov_opts[@]}" --cov-append)
fi

# Render THIS component's data as a terminal summary (so the job log stands on
# its own) plus HTML for a local run. The published pipeline number and the
# Cobertura report come from coverage_aggregate.sh, which combines this data
# file with every other component's. Best-effort: report failures are reported,
# never masked as test failures.
report_coverage() {
  ((COVERAGE)) || return 0
  local rc=0
  # pytest-cov already combined each phase's sidecars; this only picks up any
  # stragglers (e.g. a worker that died after writing its data).
  coverage combine --append >/dev/null 2>&1 || true
  coverage report || rc=1
  # `if`, not `((COVERAGE_HTML)) && ... || rc=1`: under `set -e` a false `(( ))`
  # would fall through to the `||` and mark every green shard as failed.
  if ((COVERAGE_HTML)); then
    coverage html -d "${REPORT_DIR}/htmlcov" --quiet || rc=1
  fi
  if ((rc)); then
    echo "coverage report generation failed" >&2
  fi
  return "${rc}"
}

# Which GPU ran this, and how warm its kernel cache was, as an OpenMetrics
# report CI can render alongside the run. Both change what the suite exercises:
# the CuTeDSL kernel sources only count as covered when a kernel is JIT-traced,
# which needs a matching arch AND a cold cache — so the same commit measures
# several points apart on an H100 with a warm cache versus a cold A100.
write_metrics() {
  local gpu_name gpu_cap gpu_mem host kcache pct
  IFS=, read -r gpu_name gpu_cap gpu_mem < <(
    nvidia-smi --query-gpu=name,compute_cap,memory.total \
      --format=csv,noheader,nounits 2>/dev/null | head -1
  )
  host=$(hostname)
  # File count in the restored per-arch kernel cache: high = warm (nothing
  # compiles, kernel sources stay uncovered), 0/low = cold.
  kcache=$(find "${BIOIR_KERNEL_CACHE_DIR:-/nonexistent}" -type f 2>/dev/null | wc -l | tr -d ' ')
  {
    printf 'test_gpu_info{name="%s",compute_cap="%s",vram_mib="%s",host="%s"} 1\n' \
      "${gpu_name# }" "${gpu_cap# }" "${gpu_mem# }" "${host}"
    printf 'test_xdist_workers{component="%s"} %s\n' "${COVERAGE_COMPONENT}" "${XDIST_WORKERS}"
    printf 'test_kernel_cache_files{component="%s"} %s\n' "${COVERAGE_COMPONENT}" "${kcache}"
    if ((COVERAGE)) && pct=$(coverage report --format=total 2>/dev/null); then
      printf 'test_coverage_percent{component="%s"} %s\n' "${COVERAGE_COMPONENT}" "${pct}"
    fi
  } >"${REPORT_DIR}/metrics.txt"
  echo "metrics:"
  sed 's/^/  /' "${REPORT_DIR}/metrics.txt"
}

# Phase selection, so CI can shard the suite across concurrent GPU jobs; the
# shards must partition PHASE2_TARGETS between them. Unset, both phases run back
# to back. See print_help above for the variables.
TEST_PHASE="${TEST_PHASE:-all}"
PHASE2_TARGETS="${PHASE2_TARGETS:-tests/pipeline tests/_torch/model_forwards}"
PHASE2_ADDOPTS="${PHASE2_ADDOPTS:-}"
case "${TEST_PHASE}" in
  1 | 2 | all) ;;
  *)
    echo "TEST_PHASE must be 1, 2 or all (got '${TEST_PHASE}')" >&2
    exit 2
    ;;
esac

# Phase 1 (parallel): the xdist-safe bulk, warmed by the kernel cache.
#
# Worker count is sized to VRAM, not core count: the suite is GPU-memory-bound at
# roughly 8 GB/worker. Do NOT use `-n auto` — it keys off logical CPUs and would
# spawn a worker per core → OOM. With no nvidia-smi the tiers below fall back to
# the largest, which is what CI runs on; XDIST_WORKERS overrides them for a
# smaller dev box without editing this script.
auto_workers=8
gpu_mem_mib=$(
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
    2>/dev/null | head -1 | tr -dc '0-9' || true
)
if [[ -n "${gpu_mem_mib}" ]]; then
  if ((gpu_mem_mib <= 24576)); then
    auto_workers=2
  elif ((gpu_mem_mib <= 49152)); then
    auto_workers=4
  fi
fi
XDIST_WORKERS="${XDIST_WORKERS:-${auto_workers}}"
# A failing phase must not skip phase 2 or the reports — the JUnit/coverage
# artifacts are exactly what the reviewer needs when the suite is red. Keep the
# worst status and exit with it at the end.
phase1_rc=0
if [[ "${TEST_PHASE}" == "all" || "${TEST_PHASE}" == "1" ]]; then
  echo "==== phase 1: xdist bulk" \
    "(-n ${XDIST_WORKERS}, GPU VRAM ${gpu_mem_mib:-unknown} MiB) ===="
  phase1_addopts="-n ${XDIST_WORKERS} --dist worksteal"
  phase1_addopts+=" --ignore=tests/pipeline"
  phase1_addopts+=" --ignore=tests/_torch/model_forwards"
  PYTEST_ADDOPTS="${phase1_addopts}" pytest -s -ra \
    ${cov_opts[@]+"${cov_opts[@]}"} --junitxml="${REPORT_DIR}/junit-phase1.xml" \
    tests || phase1_rc=$?
  lap "phase 1 (xdist bulk)"
fi

# Phase 2 (serial): the xdist-incompatible trees, run after phase 1 so they reuse
# the on-disk kernel cache phase 1 just warmed.
#
# Set before pytest: the raylet reads this at startup, and the first test to
# touch ray.data starts it implicitly. =0 opts into Ray's future behaviour of
# not rewriting CUDA_VISIBLE_DEVICES for num_gpus=0 actors.
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
phase2_rc=0
if [[ "${TEST_PHASE}" == "all" || "${TEST_PHASE}" == "2" ]]; then
  echo "==== phase 2: serial (${PHASE2_TARGETS}${PHASE2_ADDOPTS:+ ${PHASE2_ADDOPTS}}) ===="
  # PHASE2_TARGETS is unquoted on purpose — it carries several paths. PHASE2_ADDOPTS
  # goes through PYTEST_ADDOPTS, which pytest shlex-parses, so a quoted -k survives.
  # shellcheck disable=SC2086
  PYTEST_ADDOPTS="${PHASE2_ADDOPTS}" pytest -s -ra \
    ${cov_append_opts[@]+"${cov_append_opts[@]}"} \
    --junitxml="${REPORT_DIR}/junit-phase2.xml" \
    ${PHASE2_TARGETS} || phase2_rc=$?
  lap "phase 2 (serial)"
fi

# Only banner a report that is actually coming: with COVERAGE=0 report_coverage
# returns immediately, and an unconditional heading over nothing reads as though
# instrumentation ran and measured zero.
if ((COVERAGE)); then
  echo "==== coverage report ===="
fi
report_rc=0
report_coverage || report_rc=$?
write_metrics || echo "metrics report generation failed (non-fatal)" >&2
lap "coverage report"
echo "reports written to ${REPORT_DIR}"

# Test failures win over a report failure — the first tells you what broke.
if ((phase1_rc)); then
  exit "${phase1_rc}"
elif ((phase2_rc)); then
  exit "${phase2_rc}"
fi
exit "${report_rc}"
