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

# Enforce the Developer Certificate of Origin on every commit a pull request
# adds: each needs a Signed-off-by trailer carrying the author's own address,
# which is what `git commit -s` writes. See CONTRIBUTING.md.
#
# Usage: check-dco.sh <base-sha> <head-sha>
set -euo pipefail

if (($# != 2)); then
  echo "usage: check-dco.sh <base-sha> <head-sha>" >&2
  exit 2
fi
base="$1"
head="$2"

# The base of a pull request is the tip of the target branch, which moves on
# without the branch. Only commits after the fork point belong to this PR, so
# ask git where the two diverged rather than diffing against the tip.
if ! merge_base="$(git merge-base "${base}" "${head}")"; then
  echo "cannot find a merge base for ${base} and ${head} -- was the branch" \
    "checked out with fetch-depth: 0?" >&2
  exit 2
fi

# --no-merges: a merge from the target branch carries no contribution of its
# own, and the merge commits GitHub itself writes are never signed off.
#
# Read in a `while` over a process substitution rather than `mapfile`, which
# bash 3.2 -- what macOS still ships, and what a contributor testing this
# locally runs -- does not have.
checked=0
failed=0
while read -r sha; do
  checked=$((checked + 1))
  author_email="$(git show --no-patch --format='%ae' "${sha}")"
  subject="$(git show --no-patch --format='%s' "${sha}")"
  # Trailers only. A "Signed-off-by:" written into the middle of a commit body
  # is prose, and git does not treat it as a sign-off either.
  signoffs="$(git show --no-patch --format='%(trailers:key=Signed-off-by,valueonly)' "${sha}")"
  if grep -qiF "<${author_email}>" <<<"${signoffs}"; then
    echo "ok    ${sha:0:12} ${subject}"
    continue
  fi
  failed=1
  echo "FAIL  ${sha:0:12} ${subject}"
  if [[ -z "${signoffs//[[:space:]]/}" ]]; then
    echo "        no Signed-off-by trailer"
  else
    echo "        signed off by ${signoffs//$'\n'/, }, but authored by <${author_email}>"
  fi
done < <(git rev-list --no-merges "${merge_base}..${head}")

if ((checked == 0)); then
  echo "no commits to check between ${merge_base:0:12} and ${head:0:12}"
  exit 0
fi
((failed)) || exit 0

cat >&2 <<EOF

Every commit must carry a Signed-off-by trailer matching its author address,
certifying the Developer Certificate of Origin (https://developercertificate.org/).

Set the identity git records, then sign off the whole branch:

  git config user.name "Your Name"
  git config user.email "your@email.com"
  git rebase --signoff ${merge_base:0:12}
  git push --force-with-lease

New commits pick it up from \`git commit -s\`. See CONTRIBUTING.md.
EOF
exit 1
