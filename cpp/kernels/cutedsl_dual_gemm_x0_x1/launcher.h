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

/* Dual-GEMM x0_x1 CUBIN ABI and launcher interface. */

#ifndef BIOIR_CPP_KERNELS_CUTEDSL_DUAL_GEMM_X0_X1_LAUNCHER_H_
#define BIOIR_CPP_KERNELS_CUTEDSL_DUAL_GEMM_X0_X1_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_launch_utils.h"
#include "cutedsl_tensor_abi.h"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace bioir::cutedsl::dual_gemm_x0_x1::embedded
{
struct CubinImage;
}

/* Ampere device ABI from EIATTR_KPARAM_INFO. With bias:
 *
 *   ord 0  0x00  0x18  x0            cute_tensor_s2_d1_t
 *   ord 1  0x18  0x18  x1            cute_tensor_s2_d1_t
 *   ord 2  0x30  0x18  w0            cute_tensor_s2_d1_t
 *   ord 3  0x48  0x18  w1            cute_tensor_s2_d1_t
 *   ord 4  0x60  0x10  bias0         cute_tensor_s1_d0_t
 *   ord 5  0x70  0x10  bias1         cute_tensor_s1_d0_t
 *   ord 6  0x80  0x18  out           cute_tensor_s2_d1_t
 *   ord 7  0x98  0x04  raster_factor std::int32_t
 *
 * Without bias, out moves to 0x60 and raster_factor to 0x78.
 */
namespace bioir::cutedsl::dual_gemm_x0_x1::abi
{

inline constexpr std::size_t kSM80BiasParameterCount = 8;
inline constexpr std::size_t kSM80NoBiasParameterCount = 6;
inline constexpr std::size_t kSM80MaxParameterCount = kSM80BiasParameterCount;

struct SM80Params
{
  cute_tensor_s2_d1_t x0;
  cute_tensor_s2_d1_t x1;
  cute_tensor_s2_d1_t w0;
  cute_tensor_s2_d1_t w1;
  cute_tensor_s1_d0_t bias0;
  cute_tensor_s1_d0_t bias1;
  cute_tensor_s2_d1_t out;
  std::int32_t raster_factor;
};

/* Returns the number of packed parameters. */
inline std::size_t
pack_sm80_kernel_params(SM80Params* params, bool has_bias, void* kernel_params[kSM80MaxParameterCount])
{
  std::size_t count = 0;
  kernel_params[count++] = &params->x0;
  kernel_params[count++] = &params->x1;
  kernel_params[count++] = &params->w0;
  kernel_params[count++] = &params->w1;
  if (has_bias)
  {
    kernel_params[count++] = &params->bias0;
    kernel_params[count++] = &params->bias1;
  }
  kernel_params[count++] = &params->out;
  kernel_params[count++] = &params->raster_factor;
  return count;
}

/* Ampere grid remap:
 *
 *   grid = ceil_div((M, N), (tile_m, tile_n))
 *   launch = (grid_m * R, ceil_div(grid_n, R), 1)
 */
inline cubin_launch_config_t sm80_launch_config(
  std::uint32_t grid_m,
  std::uint32_t grid_n,
  std::uint32_t raster_factor,
  std::uint32_t num_threads,
  std::uint32_t smem_bytes,
  CUstream stream)
{
  cubin_launch_config_t config = {0};
  config.grid_x = grid_m * raster_factor;
  config.grid_y = (grid_n + raster_factor - 1U) / raster_factor;
  config.grid_z = 1;
  config.block_x = num_threads;
  config.block_y = 1;
  config.block_z = 1;
  config.dynamic_smem_bytes = smem_bytes;
  config.stream = stream;
  return config;
}

/* Hopper device ABI from EIATTR_KPARAM_INFO. With bias:
 *
 *   ord  0  0x000  0x80  x0_tma        CUtensorMap
 *   ord  1  0x080  0x08  x0_coord      CoordTensorS2
 *   ord  2  0x0c0  0x80  x1_tma        CUtensorMap
 *   ord  3  0x140  0x08  x1_coord      CoordTensorS2
 *   ord  4  0x180  0x80  w0_tma        CUtensorMap
 *   ord  5  0x200  0x08  w0_coord      CoordTensorS2
 *   ord  6  0x240  0x80  w1_tma        CUtensorMap
 *   ord  7  0x2c0  0x08  w1_coord      CoordTensorS2
 *   ord  8  0x300  0x80  output_tma    CUtensorMap
 *   ord  9  0x380  0x08  output_coord  CoordTensorS2
 *   ord 10  0x388  0x10  bias0         cute_tensor_s1_d0_t
 *   ord 11  0x398  0x10  bias1         cute_tensor_s1_d0_t
 *   ord 12  0x3a8  0x04  i_dim         std::int32_t
 *   ord 13  0x3ac  0x01  tiled_mma     std::uint8_t
 *
 * x0_x1 coordinates use two dynamic extents (CoordTensorS2), unlike x_x.
 *
 * Without bias, i_dim moves to 0x388 and tiled_mma to 0x38c.
 */
struct SM90Params
{
  CUtensorMap x0_tma;
  CoordTensorS2 x0_coord;
  CUtensorMap x1_tma;
  CoordTensorS2 x1_coord;
  CUtensorMap w0_tma;
  CoordTensorS2 w0_coord;
  CUtensorMap w1_tma;
  CoordTensorS2 w1_coord;
  CUtensorMap output_tma;
  CoordTensorS2 output_coord;
  cute_tensor_s1_d0_t bias0;
  cute_tensor_s1_d0_t bias1;
  std::int32_t i_dim;
  std::uint8_t tiled_mma;
};

constexpr std::size_t sm90_parameter_count(bool has_bias)
{
  return 12U + (has_bias ? 2U : 0U);
}

inline constexpr std::size_t kSM90MaxParameterCount = sm90_parameter_count(true);

inline std::size_t
pack_sm90_kernel_params(SM90Params* params, bool has_bias, void* kernel_params[kSM90MaxParameterCount])
{
  std::size_t count = 0;
  kernel_params[count++] = &params->x0_tma;
  kernel_params[count++] = &params->x0_coord;
  kernel_params[count++] = &params->x1_tma;
  kernel_params[count++] = &params->x1_coord;
  kernel_params[count++] = &params->w0_tma;
  kernel_params[count++] = &params->w0_coord;
  kernel_params[count++] = &params->w1_tma;
  kernel_params[count++] = &params->w1_coord;
  kernel_params[count++] = &params->output_tma;
  kernel_params[count++] = &params->output_coord;
  if (has_bias)
  {
    kernel_params[count++] = &params->bias0;
    kernel_params[count++] = &params->bias1;
  }
  kernel_params[count++] = &params->i_dim;
  kernel_params[count++] = &params->tiled_mma;
  return count;
}

static_assert(sizeof(SM80Params) >= 156, "unexpected dual-GEMM x0_x1 SM80 parameter storage");
static_assert(offsetof(SM80Params, x0) == 0x00, "unexpected dual-GEMM x0_x1 SM80 x0 offset");
static_assert(offsetof(SM80Params, x1) == 0x18, "unexpected dual-GEMM x0_x1 SM80 x1 offset");
static_assert(offsetof(SM80Params, w0) == 0x30, "unexpected dual-GEMM x0_x1 SM80 w0 offset");
static_assert(offsetof(SM80Params, w1) == 0x48, "unexpected dual-GEMM x0_x1 SM80 w1 offset");
static_assert(offsetof(SM80Params, bias0) == 0x60, "unexpected dual-GEMM x0_x1 SM80 bias0 offset");
static_assert(offsetof(SM80Params, bias1) == 0x70, "unexpected dual-GEMM x0_x1 SM80 bias1 offset");
static_assert(offsetof(SM80Params, out) == 0x80, "unexpected dual-GEMM x0_x1 SM80 out offset");
static_assert(offsetof(SM80Params, raster_factor) == 0x98, "unexpected dual-GEMM x0_x1 SM80 raster offset");

static_assert(std::is_standard_layout_v<SM80Params>);
static_assert(std::is_standard_layout_v<SM90Params>);
static_assert(sizeof(CUtensorMap) == 128);
static_assert(alignof(CUtensorMap) >= 64);
static_assert(sizeof(CoordTensorS2) == 8, "dual-GEMM x0_x1 SM90 inputs carry two dynamic extents");
static_assert(alignof(CoordTensorS2) == alignof(std::int32_t));
static_assert(offsetof(CoordTensorS2, dynamic_shapes) == 0);

/* The driver uses parameter pointers, so host struct padding is irrelevant;
 * only each parameter width must match the CUBIN.
 */
static_assert(offsetof(SM90Params, x0_tma) == 0);
static_assert(sizeof(SM90Params::x0_tma) == 0x80, "unexpected dual-GEMM x0_x1 SM90 tensor-map width");
static_assert(sizeof(SM90Params::x0_coord) == 0x08, "unexpected dual-GEMM x0_x1 SM90 coordinate width");
static_assert(sizeof(SM90Params::output_coord) == 0x08, "unexpected dual-GEMM x0_x1 SM90 output coordinate width");
static_assert(sizeof(SM90Params::bias0) == 0x10, "unexpected dual-GEMM x0_x1 SM90 bias width");
static_assert(sizeof(SM90Params::i_dim) == 0x04, "unexpected dual-GEMM x0_x1 SM90 i_dim width");
static_assert(sizeof(SM90Params::tiled_mma) == 0x01, "unexpected dual-GEMM x0_x1 SM90 MMA token width");
static_assert(sm90_parameter_count(true) == kSM90MaxParameterCount);
static_assert(sm90_parameter_count(false) == 12U);

} // namespace bioir::cutedsl::dual_gemm_x0_x1::abi

namespace bioir::cutedsl::dual_gemm_x0_x1
{

enum class DType : std::uint8_t
{
  kFloat16,
  kBFloat16,
};

/* Launch fields generated with the corresponding CUBIN. */
struct KernelSpec
{
  std::int32_t target_sm;
  /* Device ABI generation, distinct from target_sm. */
  std::int32_t kernel_sm;
  std::int32_t K;
  std::int32_t N;
  std::int32_t bucket;
  bool has_bias;

  std::uint32_t tile_m;
  std::uint32_t tile_n;
  std::uint32_t num_threads;
  std::uint32_t raster_factor;
};

struct KernelConfig
{
  KernelSpec spec;
  DType dtype;
  bool has_bias;
  EmbeddedCubinImage cubin;
  embedded::CubinImage const* embedded_image;
};

inline std::uint32_t dynamic_smem_bytes(KernelConfig const& config)
{
  if (config.cubin.dynamic_smem_bytes == 0)
    throw std::invalid_argument("dual-GEMM x0_x1 CUBIN has no dynamic shared-memory metadata");
  return config.cubin.dynamic_smem_bytes;
}

struct LaunchParams
{
  Tensor2View x0;
  Tensor2View x1;
  Tensor2View w0;
  Tensor2View w1;
  Tensor1View bias0;
  Tensor1View bias1;
  Tensor2View out;
  std::uint64_t stream{};
};

std::vector<KernelSpec> kernel_specs();

KernelConfig
make_kernel_config(std::int32_t target_sm, std::int32_t K, std::int32_t N, std::int32_t S, DType dtype, bool has_bias);

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace bioir::cutedsl::dual_gemm_x0_x1

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_DUAL_GEMM_X0_X1_LAUNCHER_H_ */
