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

find_package(Python3 REQUIRED)
set(CUE_OPS_VERSION 0.7.0)
execute_process(
  COMMAND
    ${Python3_EXECUTABLE} -c
    "import os; import cuequivariance_ops;print(os.path.dirname(cuequivariance_ops.__file__),end='');"
  RESULT_VARIABLE FOUND_STATUS
  OUTPUT_VARIABLE CUE_OPS_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if("${CUE_OPS_PATH}" MATCHES "not found")
  message(
    STATUS
      "FATAL_ERROR: ${FOUND_STATUS}, Cannot find cuequivariance_ops package.")
else()
  set(CUE_OPS_PATH ${CUE_OPS_PATH})
  message(STATUS "Found cuequivariance_ops package: ${CUE_OPS_PATH}")
endif()

find_package(CuEquiv REQUIRED)
