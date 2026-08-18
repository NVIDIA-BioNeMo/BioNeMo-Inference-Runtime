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

#
# Combine every test job's coverage data into the ONE number the pipeline
# publishes (OBS-P1-001 measures the repository, not any single suite).
#
# The repo has several independent test suites that cannot share a job: the GPU
# battery needs a CUDA image and a GPU runner, while the uv sub-projects (sync/,
# bench/) are pure-CPU and must not queue behind an image build. Each therefore
# measures its own slice, and this script merges the slices.
#
# ---------------------------------------------------------------------------
# Onboarding a new component — nothing here or in run_tests.sh needs editing:
#
#   1. Run pytest with branch coverage FROM THE REPO ROOT, so the recorded paths
#      are repo-relative and merge cleanly with everyone else's. For a uv
#      sub-project use `--project`, not `--directory`: `--directory` cd's into
#      the sub-project, and the resulting paths (`foo/bar.py` instead of
#      `bench/foo/bar.py`) collide with the root project's.
#          uv run --project bench pytest --cov=bioir_bench bench/tests
#      If a component genuinely must run from its own directory, map its tree
#      back with a [tool.coverage.paths] entry in the root pyproject.toml
#      instead: https://coverage.readthedocs.io/en/latest/config.html#paths
#   2. Point COVERAGE_FILE at the shared drop point, naming the file after the
#      component. tmp/ is already gitignored, so no new ignore rule is needed:
#          tmp/reports/coverage-data/.coverage.<component>
#   3. Upload `tmp/reports/` as an artifact (`when: always`), from a job in a stage
#      BEFORE `report`. That is the whole wiring: coverage:report declares no
#      needs/dependencies, so it inherits the artifacts of every earlier-stage
#      job and picks the new data file up automatically.
#
# Components inherit the root [tool.coverage.run] config (branch coverage,
# relative_files), so their data is merge-compatible by default.
# ---------------------------------------------------------------------------
#
# Reads : REPORT_DIR (default <checkout>/tmp/reports), COVERAGE_DATA_DIR.
# Writes: the combined terminal summary (the CI `coverage:` regex reads its
#         TOTAL line), Cobertura XML for the MR diff, HTML, and a metrics
#         report carrying the total and the component list.
set -euo pipefail

REPORT_DIR="${REPORT_DIR:-${CI_PROJECT_DIR:-$PWD}/tmp/reports}"
COVERAGE_DATA_DIR="${COVERAGE_DATA_DIR:-${REPORT_DIR}/coverage-data}"
mkdir -p "${REPORT_DIR}"

shopt -s nullglob
data_files=("${COVERAGE_DATA_DIR}"/.coverage.*)
if ((${#data_files[@]} == 0)); then
  echo "no coverage data in ${COVERAGE_DATA_DIR} — did a test job fail before" \
    "writing one, or is its artifact missing from needs:?" >&2
  exit 1
fi

# Component name = the data file's suffix (.coverage.<component>), which is the
# contract above. Reported so the log names what the total actually covers.
components=()
for f in "${data_files[@]}"; do
  name="${f##*/.coverage.}"
  components+=("${name%%.*}")
done
echo "combining ${#data_files[@]} coverage data file(s): ${components[*]}"

export COVERAGE_FILE="${REPORT_DIR}/.coverage"
rm -f "${COVERAGE_FILE}"
# --keep: leave the per-component files in place so the artifact stays
# inspectable and a re-run of this job is idempotent.
coverage combine --keep "${data_files[@]}"

echo "==== combined coverage ===="
coverage report
coverage xml -o "${REPORT_DIR}/coverage.xml"
coverage html -d "${REPORT_DIR}/htmlcov" --quiet

# Total + which components fed it, for the CI summary. The per-job GPU metrics
# (see run_tests.sh) land in the same widget.
{
  if total=$(coverage report --format=total 2>/dev/null); then
    printf 'coverage_total_percent %s\n' "${total}"
  fi
  printf 'coverage_components %s\n' "${#components[@]}"
} >"${REPORT_DIR}/metrics.txt"

echo "combined reports written to ${REPORT_DIR}"
