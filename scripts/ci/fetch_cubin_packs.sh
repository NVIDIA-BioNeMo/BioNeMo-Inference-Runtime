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

# Pull the LFS-tracked CUBIN packs and nothing else. `git lfs pull` with no
# filter fetches every object in the repo, measured at 88.2 MB against the
# 2.5 MB of packs the build and the contract suite read; the rest is sample MSAs
# and structures nothing here opens.
#
# .gitattributes is the source of truth for the pattern -- it is what marks
# these paths as LFS, and tests/contract/test_build_contract.py asserts the
# line. A copy that drifts from it fails quietly: the pull matches nothing, the
# packs stay pointers, and the error surfaces later as "is a Git LFS pointer".
#
# Callers check out with the LFS smudge skipped and keep the credentials the
# checkout leaves in the local git config; the pull needs them.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

packs='cpp/kernels/cutedsl_*/cubins/packs/*.tar.xz'
git lfs pull --include="${packs}" --exclude=''
echo "materialized $(git lfs ls-files -I "${packs}" | wc -l) packs"
