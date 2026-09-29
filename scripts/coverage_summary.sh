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

coverage_summary() {
  local report rc=0
  export BIOIR_COVERAGE_TOTAL=""
  if report=$(coverage report --format=text); then
    # Single-file reports omit the TOTAL row.
    BIOIR_COVERAGE_TOTAL=$(awk '
      { for (i = 1; i <= NF; i++)
          if ($i ~ /^[0-9]+([.][0-9]+)?%$/) total = $i }
      END { sub(/%$/, "", total); print total }
    ' <<<"${report}")
  else
    rc=$?
  fi
  printf '%s\n' "${report}"
  return "${rc}"
}
