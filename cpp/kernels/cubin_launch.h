/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Common CUDA Driver API state for loading and launching a CUBIN kernel.
 */

#ifndef TENSORRT_BIONEMO_CPP_KERNELS_CUBIN_LAUNCH_H_
#define TENSORRT_BIONEMO_CPP_KERNELS_CUBIN_LAUNCH_H_

#include <cuda.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C"
{
#endif

  typedef struct
  {
    CUmodule module;
    CUfunction function;
  } cubin_kernel_t;

  typedef struct
  {
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    uint32_t block_x;
    uint32_t block_y;
    uint32_t block_z;
    /* All zero selects a normal launch. Otherwise every cluster dimension
     * must be nonzero and divide the corresponding grid dimension.
     */
    uint32_t cluster_x;
    uint32_t cluster_y;
    uint32_t cluster_z;
    CUclusterSchedulingPolicy cluster_scheduling_policy;
    uint32_t dynamic_smem_bytes;
    CUstream stream;
  } cubin_launch_config_t;

  static inline CUresult
  launch_cubin_kernel(cubin_kernel_t kernel, cubin_launch_config_t const* config, void** kernel_params, void** extra)
  {
    if (kernel.function == NULL || config == NULL)
      return CUDA_ERROR_INVALID_VALUE;

    int const uses_clusters = config->cluster_x != 0 || config->cluster_y != 0 || config->cluster_z != 0;
    if (!uses_clusters)
    {
      return cuLaunchKernel(
        kernel.function,
        config->grid_x,
        config->grid_y,
        config->grid_z,
        config->block_x,
        config->block_y,
        config->block_z,
        config->dynamic_smem_bytes,
        config->stream,
        kernel_params,
        extra);
    }

    if (
      config->cluster_x == 0 || config->cluster_y == 0 || config->cluster_z == 0
      || config->grid_x % config->cluster_x != 0 || config->grid_y % config->cluster_y != 0
      || config->grid_z % config->cluster_z != 0)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    CUlaunchAttribute attributes[2];
    memset(attributes, 0, sizeof(attributes));
    attributes[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attributes[0].value.clusterDim.x = config->cluster_x;
    attributes[0].value.clusterDim.y = config->cluster_y;
    attributes[0].value.clusterDim.z = config->cluster_z;
    attributes[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
    attributes[1].value.clusterSchedulingPolicyPreference = config->cluster_scheduling_policy;

    CUlaunchConfig driver_config = {0};
    driver_config.gridDimX = config->grid_x;
    driver_config.gridDimY = config->grid_y;
    driver_config.gridDimZ = config->grid_z;
    driver_config.blockDimX = config->block_x;
    driver_config.blockDimY = config->block_y;
    driver_config.blockDimZ = config->block_z;
    driver_config.sharedMemBytes = config->dynamic_smem_bytes;
    driver_config.hStream = config->stream;
    driver_config.attrs = attributes;
    driver_config.numAttrs = 2;
    return cuLaunchKernelEx(&driver_config, kernel.function, kernel_params, extra);
  }

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TENSORRT_BIONEMO_CPP_KERNELS_CUBIN_LAUNCH_H_ */
