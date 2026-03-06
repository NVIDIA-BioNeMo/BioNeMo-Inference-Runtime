# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

set(CUTLASS_INCLUDE_DIRS
    ${TRT_BIONEMO_THIRDPARTY_DIR}/cutlass/include
    CACHE PATH "CUTLASS Header Library")
set(CUTLASS_TOOLS_UTIL_INCLUDE_DIRS
    ${TRT_BIONEMO_THIRDPARTY_DIR}/cutlass/tools/util/include
    CACHE PATH "CUTLASS Tools Util Header Library")
set(CUTLASS_TOOLS_LIBRARY_INCLUDE_DIRS
    ${TRT_BIONEMO_THIRDPARTY_DIR}/cutlass/tools/library/include
    CACHE PATH "CUTLASS Tools Library Header Library")
message(STATUS "CUTLASS_INCLUDE_DIRS: ${CUTLASS_INCLUDE_DIRS}")
message(
  STATUS "CUTLASS_TOOLS_UTIL_INCLUDE_DIRS: ${CUTLASS_TOOLS_UTIL_INCLUDE_DIRS}")
message(
  STATUS
    "CUTLASS_TOOLS_LIBRARY_INCLUDE_DIRS: ${CUTLASS_TOOLS_LIBRARY_INCLUDE_DIRS}")
include_directories(
  SYSTEM ${CUTLASS_INCLUDE_DIRS} ${CUTLASS_TOOLS_UTIL_INCLUDE_DIRS}
  ${CUTLASS_TOOLS_LIBRARY_INCLUDE_DIRS})
