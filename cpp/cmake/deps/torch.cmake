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
find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)
message(STATUS "Found Python executable at ${Python3_EXECUTABLE}")
message(STATUS "Found Python libraries at ${Python3_LIBRARY_DIRS}")
execute_process(
  COMMAND
    ${Python3_EXECUTABLE} -c
    "import os; import torch;print(os.path.dirname(torch.__file__),end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  ERROR_VARIABLE stderr_output
  OUTPUT_VARIABLE TORCH_DIR)
message(STATUS ${TORCH_DIR})

if(NOT _PYTHON_SUCCESS EQUAL 0)
  message(FATAL_ERROR "Torch config Error.STDERR: ${stderr_output}")
endif()
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)

message(
  STATUS
    "Removing Torch generated placeholder CUDA architecture flags: -gencode arch=compute_75,code=sm_75."
)
string(REPLACE "-gencode arch=compute_75,code=sm_75 " "" CMAKE_CUDA_FLAGS_NEW
               "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS_NEW}")
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
