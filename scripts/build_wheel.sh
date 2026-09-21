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

# Build BioIR's wheel in an isolated PEP 517 environment constrained and hashed
# from uv.lock. Release CI separately proves that the public sdist builds with
# pypa/build.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Git is absent from Docker build contexts, whose caller supplies this value.
# Host and GitHub builds otherwise use the source commit timestamp directly.
if [ -z "${SOURCE_DATE_EPOCH:-}" ]; then
  SOURCE_DATE_EPOCH="$(git show -s --format=%ct HEAD)"
  export SOURCE_DATE_EPOCH
fi

BUILD_REQUIREMENTS="$(mktemp)"
trap 'rm -f "${BUILD_REQUIREMENTS}"' EXIT
uv export --quiet --locked --only-group build --no-emit-project \
  --output-file "${BUILD_REQUIREMENTS}"
uv build --wheel --no-sources --build-constraints "${BUILD_REQUIREMENTS}" \
  --require-hashes "$@"
