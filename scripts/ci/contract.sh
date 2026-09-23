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

# The slice of the suite that needs no GPU, no CUDA driver and no built
# extension: the public build contract and the CUBIN corpus. Every CI runs this
# script, so the command lives here and its dependencies in the `test` group of
# pyproject.toml.
#
# Nothing here imports torch or the package itself, so there is no install
# step and no build. tests/pytest.ini supplies the options. The committed packs
# are load-bearing: test_committed_corpus_materializes reads them and fails on
# a pointer, which is it doing its job.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
# shellcheck source=/dev/null
source scripts/ci/uv_env.sh

scripts/ci/fetch_cubin_packs.sh
uv run --locked --only-group test pytest tests/contract tests/cubin
