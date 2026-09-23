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

# The style gate: the hooks prek.toml pins, from the locked `lint` group. With
# no arguments every file is checked, the `prek run --all-files` every PR must
# pass; with a base and a head revision, only the files that range touches.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
# shellcheck source=/dev/null
source scripts/ci/uv_env.sh

case $# in
  0) range=(--all-files) ;;
  2) range=(--from-ref "$1" --to-ref "$2") ;;
  *)
    echo "usage: $0 [<base-rev> <head-rev>]" >&2
    exit 2
    ;;
esac

uv run --locked --only-group lint prek run "${range[@]}" --show-diff-on-failure --color=always
