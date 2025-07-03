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

# Get cuequiv version Read the file content into a variable
file(READ "${CUE_OPS_PATH}/VERSION"
     VERSION_FILE_CONTENT)

# Remove any whitespace or newlines that might be present
string(STRIP "${VERSION_FILE_CONTENT}" VERSION_STRING)

# Use regex to extract version components
if(VERSION_STRING MATCHES "([0-9]+)\\.([0-9]+)\\.([0-9]+)")
  set(VERSION_MAJOR ${CMAKE_MATCH_1})
  set(VERSION_MINOR ${CMAKE_MATCH_2})
  set(VERSION_PATCH ${CMAKE_MATCH_3})
  message(STATUS "Version: ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
else()
  message(FATAL_ERROR "Failed to parse version string: ${VERSION_STRING}")
endif()

set(CUE_OPS_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

find_library(
  CUE_OPS_LIB
  NAMES libcue_ops.so REQUIRED
  PATHS ${CUE_OPS_PATH}/lib)
mark_as_advanced(CUE_OPS_LIB)

add_library(CuEquiv::ops SHARED IMPORTED GLOBAL)
set_target_properties(
  CuEquiv::ops
  PROPERTIES IMPORTED_LOCATION "${CUE_OPS_LIB}" INTERFACE_INCLUDE_DIRECTORIES
                                                "${CUE_OPS_PATH}/equivariance")

# Generate CuEquiv_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  CuEquiv
  FOUND_VAR CuEquiv_FOUND
  VERSION_VAR CUE_OPS_VERSION
  REQUIRED_VARS CUE_OPS_LIB # no need for libs/targets, since
                # find_library is REQUIRED
)
