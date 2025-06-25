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

find_package(Python REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import cuequivariance_ops; print(cuequivariance_ops.root_dir())"
  RESULT_VARIABLE STATUS
  OUTPUT_VARIABLE CUE_OPS_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(STATUS AND NOT STATUS EQUAL 0)
  message(
    STATUS
      "WARNING: ${STATUS}, Cannot find cuequivariance_ops package. Installing it from 3rdparty/cuequiv-ops"
  )
  execute_process(COMMAND "${Python_EXECUTABLE}" -m pip install
                          scikit-build-core nanobind pynvml)
  execute_process(WORKING_DIRECTORY ${TRT_BIONEMO_THIRDPARTY_DIR}/cuequiv-ops
                  COMMAND bash build.sh cue-ops)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import cuequivariance_ops; print(cuequivariance_ops.root_dir())"
    RESULT_VARIABLE STATUS
    OUTPUT_VARIABLE CUE_OPS_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(STATUS AND NOT STATUS EQUAL 0)
    message(
      STATUS
        "ERROR: ${STATUS}, Cannot find cuequivariance_ops package. Please check the build.sh script in 3rdparty/cuequiv-ops"
    )
  else()
    message(STATUS "Found cuequivariance_ops package: ${CUE_OPS_PATH}")
  endif()
else()
  message(STATUS "Found cuequivariance_ops package: ${CUE_OPS_PATH}")
endif()

find_package(CuEquiv REQUIRED)
