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

/* Conditioned transition MLP CUBIN configuration, device ABI, and launcher interface. */

#ifndef BIOIR_CPP_KERNELS_CUTEDSL_CONDITIONED_TRANSITION_MLP_LAUNCHER_H_
#define BIOIR_CPP_KERNELS_CUTEDSL_CONDITIONED_TRANSITION_MLP_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_launch_utils.h"
#include "cutedsl_tensor_abi.h"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace bioir::cutedsl::conditioned_transition_mlp::embedded
{
struct CubinImage;
}

/* conditioned_transition_mlp_sm100_v1, from EIATTR_KPARAM_INFO; the 2-way and 3-way SwiGLU share it, and only W1's
 * tensor map differs:
 *
 *   ord  0  0x000  0x03  gemm1         tiled MMA record
 *   ord  1  0x003  0x03  gemm2         tiled MMA record
 *   ord  2  0x040  0x80  x_tma         CUtensorMap
 *   ord  3  0x0c0  0x04  x_coord       CoordTensorS1
 *   ord  4  0x100  0x80  w1_tma        CUtensorMap
 *   ord  5  0x180  0x80  w2_tma        CUtensorMap
 *   ord  6  0x200  0x18  residual      cute_tensor_s1_d1_t, x itself
 *   ord  7  0x218  0x18  output        cute_tensor_s1_d1_t
 *   ord  8  0x240  0x80  output_tma    CUtensorMap
 *   ord  9  0x2c0  0x04  output_coord  CoordTensorS1
 *   ord 10  0x2c8  0x18  cond          cute_tensor_s1_d1_t
 *
 * W1 and W2 have static shapes, so their coordinate tensors lower to no parameter; layouts occupy none either. A
 * sample-major image's x and output maps are rank 3, [atoms, C, samples], and their coordinates hold the atom count
 * instead of the row count; the widths are the same. Each TMA slot carries the non-executable atom
 * finalize_sm100_tma_atom writes: a load for x, W1 and W2 and a store for the output.
 */
namespace bioir::cutedsl::conditioned_transition_mlp::abi
{

using SM100TiledMmaDescriptor = bioir::cutedsl::SM100TiledMmaToken;

struct SM100Params
{
  SM100TiledMmaDescriptor gemm1;
  SM100TiledMmaDescriptor gemm2;
  CUtensorMap x_tma;
  CoordTensorS1 x_coord;
  CUtensorMap w1_tma;
  CUtensorMap w2_tma;
  cute_tensor_s1_d1_t residual;
  cute_tensor_s1_d1_t output;
  CUtensorMap output_tma;
  CoordTensorS1 output_coord;
  cute_tensor_s1_d1_t cond;
};

inline constexpr std::size_t kSM100ParameterCount = 11U;

/* params and kernel_params must stay alive until the launch call returns. */
inline void pack_sm100_kernel_params(SM100Params* params, void* kernel_params[kSM100ParameterCount])
{
  kernel_params[0] = &params->gemm1;
  kernel_params[1] = &params->gemm2;
  kernel_params[2] = &params->x_tma;
  kernel_params[3] = &params->x_coord;
  kernel_params[4] = &params->w1_tma;
  kernel_params[5] = &params->w2_tma;
  kernel_params[6] = &params->residual;
  kernel_params[7] = &params->output;
  kernel_params[8] = &params->output_tma;
  kernel_params[9] = &params->output_coord;
  kernel_params[10] = &params->cond;
}

/* The driver takes one pointer per parameter, so host struct padding is irrelevant; only each parameter's width must
 * match the CUBIN.
 */
static_assert(std::is_standard_layout_v<SM100Params>);
static_assert(sizeof(SM100TiledMmaDescriptor) == 64 && alignof(SM100TiledMmaDescriptor) == 1);
static_assert(sizeof(SM100Params::x_tma) == 128 && sizeof(SM100Params::output_tma) == 128);
static_assert(sizeof(SM100Params::x_coord) == 4 && sizeof(SM100Params::output_coord) == 4);
static_assert(sizeof(SM100Params::residual) == 24 && sizeof(SM100Params::output) == 24);
static_assert(sizeof(SM100Params::cond) == 24);

} // namespace bioir::cutedsl::conditioned_transition_mlp::abi

namespace bioir::cutedsl::conditioned_transition_mlp
{

enum class DType : std::uint8_t
{
  kBFloat16,
};

/* Launch fields generated with the corresponding CUBIN. is_three_way, cond_stage and samples tell a shape's images
 * apart: the 3-way SwiGLU rather than the 2-way one, whether the image stages each atom's conditioning in shared
 * memory, and the samples of a sample-major image's tiles (0 for row tiles). A CTA pair takes tile_m rows of a
 * row-tiled call, or tile_atoms atoms of each sample of a sample-major one.
 */
struct KernelSpec
{
  std::int32_t target_sm;
  std::int32_t kernel_sm;
  std::int32_t width;
  std::int32_t hidden;
  bool is_three_way;
  bool cond_stage;
  std::uint32_t samples;
  std::uint32_t tile_m;
  std::uint32_t tile_atoms;
  std::uint32_t num_threads;
};

struct KernelConfig
{
  KernelSpec spec;
  DType dtype;
  EmbeddedCubinImage cubin;
  embedded::CubinImage const* embedded_image;
};

inline std::uint32_t dynamic_smem_bytes(KernelConfig const& config)
{
  if (config.cubin.dynamic_smem_bytes == 0)
    throw std::invalid_argument("conditioned transition MLP CUBIN has no dynamic shared-memory metadata");
  return config.cubin.dynamic_smem_bytes;
}

/* x and output are [rows, width] and cond [atoms, 3 * width]: row r of x takes cond row r % atoms. w1 is
 * [2 * hidden, width], value rows then gate rows, or with the 3-way SwiGLU [3 * hidden, width] with second-value
 * rows last; w2 is [width, hidden]. A sample-major image takes rows == samples * atoms with contiguous x and output.
 */
struct LaunchParams
{
  Tensor2View x;
  Tensor2View w1;
  Tensor2View w2;
  Tensor2View cond;
  Tensor2View output;
  std::uint64_t stream{};
};

std::vector<KernelSpec> kernel_specs();

KernelConfig make_kernel_config(
  std::int32_t target_sm,
  DType dtype,
  std::int32_t width,
  std::int32_t hidden,
  bool is_three_way,
  bool cond_stage,
  std::int32_t samples);

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace bioir::cutedsl::conditioned_transition_mlp

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_CONDITIONED_TRANSITION_MLP_LAUNCHER_H_ */
