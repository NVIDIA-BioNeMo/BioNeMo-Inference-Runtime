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

# Check a pull-request title against the Conventional Commits rule prek's
# commit-msg hook applies to commit messages. PRs are squash-merged with
# `squash_merge_commit_title = PR_TITLE`, so this title is the subject of the
# commit that lands on main -- the same gate, moved to where it will be read.
# The repo setting is load-bearing: the `COMMIT_OR_PR_TITLE` default would use
# the branch's own subject whenever a PR carries exactly one commit, and that
# subject is linted only by the commit-msg hook, which is opt-in.
#
# The check is prek itself, so commitizen's version and configuration stay
# single-sourced in prek.toml and pyproject.toml. AGENTS.md allows an optional
# tracker reference in front of the summary; commitizen anchors its pattern at
# the start of the message and would reject one, so strip it first.
#
# Usage: check-pr-title.sh "<title>"
set -euo pipefail

if (($# != 1)); then
  echo "usage: check-pr-title.sh <title>" >&2
  exit 2
fi
title="$1"

# `[PROJ-123]` (JIRA key) or `[5123456]` (NVBug ID), and the space behind it.
summary="${title}"
if [[ "${summary}" =~ ^\[([A-Za-z][A-Za-z0-9]*-)?[0-9]+\][[:space:]]*(.*)$ ]]; then
  summary="${BASH_REMATCH[2]}"
fi

# commitizen reads the message from a file, the way the commit-msg hook feeds it
# .git/COMMIT_EDITMSG.
message_file="$(mktemp)"
trap 'rm -f "${message_file}"' EXIT
printf '%s\n' "${summary}" >"${message_file}"

echo "checking title: ${title}"
if [[ "${summary}" != "${title}" ]]; then
  echo "tracker reference stripped, checking: ${summary}"
fi

if prek run commitizen --stage commit-msg --commit-msg-filename "${message_file}"; then
  exit 0
fi

cat >&2 <<'EOF'

The pull request title must be a Conventional Commits subject, optionally
preceded by a tracker reference:

  feat: add a triangle-attention fallback for sm80
  fix(pipeline): stop dropping the last MSA row
  [PROJ-123] docs: describe the CUBIN pack layout

Types: build, bump, chore, ci, docs, feat, fix, perf, refactor, revert, style,
test. Append `!` before the colon for a breaking change. See AGENTS.md and
docs/coding.md, and edit the title -- this check re-runs on its own.
EOF
exit 1
