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
# Merge every test job's coverage slice into one repository total. The suites
# cannot share a job -- the GPU battery needs a CUDA image and a GPU runner,
# while pure-CPU suites must not queue behind that image build -- so each
# measures its own slice.
#
# ---------------------------------------------------------------------------
# Onboarding a component -- nothing here or in run_tests.sh needs editing:
#
#   1. Run pytest with branch coverage FROM THE REPO ROOT, so the recorded paths
#      are repo-relative and merge cleanly with everyone else's. For a uv
#      sub-project that means `--project`, not `--directory`: `--directory` cd's
#      in, and the resulting paths (`foo/bar.py` instead of `bench/foo/bar.py`)
#      collide with the root project's.
#          uv run --project bench pytest --cov=bioir_bench bench/tests
#      A component that must run from its own directory maps its tree back with
#      a [tool.coverage.paths] entry in the root pyproject.toml instead:
#      https://coverage.readthedocs.io/en/latest/config.html#paths
#   2. Point COVERAGE_FILE at the shared drop point, named after the component
#      (run_tests.sh does this via COVERAGE_COMPONENT). tmp/ is gitignored, so
#      no new ignore rule is needed:
#          tmp/reports/coverage-data/.coverage.<component>
#   3. Get that file to this job. On GitLab, uploading tmp/reports/ from any
#      earlier stage is enough: the reporting job declares no needs and inherits
#      every earlier job's artifacts. GitHub Actions has no such inheritance, so
#      each component needs its own upload-artifact and a matching
#      download-artifact `pattern`.
#
# Components inherit the root [tool.coverage.run] config (branch coverage,
# relative_files), so their data is merge-compatible by default.
# ---------------------------------------------------------------------------
#
# Reads : REPORT_DIR (default <checkout>/tmp/reports), COVERAGE_DATA_DIR.
# Writes: terminal summary (GitLab's `coverage:` regex scrapes its TOTAL line),
#         Cobertura XML for the MR diff, HTML, and metrics.txt carrying the
#         total and the component count.
set -euo pipefail

# Must sit in the CI checkout -- a job only uploads artifacts from its own
# workspace. Both CIs start the job there, so $PWD is a fallback, not a guess.
REPORT_DIR="${REPORT_DIR:-${CI_PROJECT_DIR:-${GITHUB_WORKSPACE:-$PWD}}/tmp/reports}"
COVERAGE_DATA_DIR="${COVERAGE_DATA_DIR:-${REPORT_DIR}/coverage-data}"
mkdir -p "${REPORT_DIR}"

shopt -s nullglob
data_files=("${COVERAGE_DATA_DIR}"/.coverage.*)
if ((${#data_files[@]} == 0)); then
  echo "no coverage data in ${COVERAGE_DATA_DIR} — did a test job fail before" \
    "writing one, or did its artifact not reach this job?" >&2
  exit 1
fi

# Component name = the data file's suffix (.coverage.<component>), which is the
# contract above. Named in the log so it says what the total actually covers.
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

total=""
if ! total=$(coverage report --format=total 2>/dev/null); then
  total=""
fi

# Total + how many components fed it, for the CI summary. The per-job GPU
# metrics (see run_tests.sh) land in the same widget.
{
  if [[ -n ${total} ]]; then
    printf 'coverage_total_percent %s\n' "${total}"
  fi
  printf 'coverage_components %s\n' "${#components[@]}"
} >"${REPORT_DIR}/metrics.txt"

# Nothing on GitHub scrapes the log for a percentage. Unset elsewhere, so this
# is skipped off GitHub.
if [[ -n ${GITHUB_STEP_SUMMARY:-} && -n ${total} ]]; then
  {
    printf '### Coverage: %s%%\n\n' "${total}"
    printf '%s component(s): %s\n' "${#components[@]}" "${components[*]}"
  } >>"${GITHUB_STEP_SUMMARY}"
fi

echo "combined reports written to ${REPORT_DIR}"
