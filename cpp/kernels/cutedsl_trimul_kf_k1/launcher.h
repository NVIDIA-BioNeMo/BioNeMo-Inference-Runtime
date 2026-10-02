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

/* SM90 TriMul KF K1 CUBIN configuration, device ABI, and launcher interface. */

#ifndef BIOIR_CPP_KERNELS_CUTEDSL_TRIMUL_KF_K1_LAUNCHER_H_
#define BIOIR_CPP_KERNELS_CUTEDSL_TRIMUL_KF_K1_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_launch_utils.h"
#include "cutedsl_tensor_abi.h"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace bioir::cutedsl::trimul_kf_k1::embedded
{
struct CubinImage;
}

/* Hopper device ABI trimul_kf_k1_sm90_v1, read from EIATTR_KPARAM_INFO for variant K1_0:
 *
 *   ord  0  0x030  0x80  x_tma       CUtensorMap
 *   ord  1  0x0b0  0x04  x_coord     CoordTensorS1   rows
 *   ord  2  0x0f0  0x80  w_proj_tma  CUtensorMap
 *   ord  3  0x170  0x80  w_gate_tma  CUtensorMap
 *   ord  4  0x1f0  0x80  a_tma       CUtensorMap
 *   ord  5  0x270  0x08  a_coord     CoordTensorS2   (N * N, B)
 *   ord  6  0x2b0  0x80  b_tma       CUtensorMap
 *   ord  7  0x330  0x08  b_coord     CoordTensorS2   (N * N, B)
 *   ord  8  0x338  0x10  seqlen      cute_tensor_s1_d0_t
 *   ord  9  0x348  0x08  vec         cute_tensor_s0_d0_t
 *   ord 10  0x350  0x01  tiled_mma   std::uint8_t
 *   ord 11  0x354  0x04  n           std::int32_t
 *   ord 12  0x358  0x04  num_tiles   std::int32_t
 *   ord 13  0x35c  0x04  eps         float
 *   ord 14  0x360  0x04  rows        std::int32_t
 *
 * The ping-pong variants K1_1 and K1_2 read one interleaved [4D, C] weight fold through w_proj_tma,
 * so w_gate_tma is absent, and a second MMA token, mma_out, follows tiled_mma. K1_2 also passes its
 * row-statistics view, stats (extent = rows), right after vec. The weights have static shapes, so
 * their coordinate tensors lower to no parameter; the shared-memory layouts are compile-time objects
 * and occupy no slot either.
 *
 * Each 0x80 TMA slot carries CuTe DSL 4.5.2's 0x40-byte non-executable CopyAtom payload followed
 * by zero padding: plain loads of x and the weights, and stores of a and b.
 */
namespace bioir::cutedsl::trimul_kf_k1::abi
{

struct SM90Params
{
  CUtensorMap x_tma;
  CoordTensorS1 x_coord;
  CUtensorMap w_proj_tma;
  CUtensorMap w_gate_tma;
  CUtensorMap a_tma;
  CoordTensorS2 a_coord;
  CUtensorMap b_tma;
  CoordTensorS2 b_coord;
  cute_tensor_s1_d0_t seqlen;
  cute_tensor_s0_d0_t vec;
  cute_tensor_s1_d0_t stats;
  std::uint8_t tiled_mma;
  std::uint8_t mma_out;
  std::int32_t n;
  std::int32_t num_tiles;
  float eps;
  std::int32_t rows;
};

constexpr std::size_t sm90_parameter_count(std::int32_t kernel_variant)
{
  return kernel_variant == 2 ? 16U : 15U;
}

inline constexpr std::size_t kSM90MaxParameterCount = 16;

/* params and kernel_params must stay alive until the launch call returns. */
inline std::size_t
pack_sm90_kernel_params(SM90Params* params, std::int32_t kernel_variant, void* kernel_params[kSM90MaxParameterCount])
{
  bool const pingpong = kernel_variant != 0;
  std::size_t count = 0;
  kernel_params[count++] = &params->x_tma;
  kernel_params[count++] = &params->x_coord;
  kernel_params[count++] = &params->w_proj_tma;
  if (!pingpong)
    kernel_params[count++] = &params->w_gate_tma;
  kernel_params[count++] = &params->a_tma;
  kernel_params[count++] = &params->a_coord;
  kernel_params[count++] = &params->b_tma;
  kernel_params[count++] = &params->b_coord;
  kernel_params[count++] = &params->seqlen;
  kernel_params[count++] = &params->vec;
  if (kernel_variant == 2)
    kernel_params[count++] = &params->stats;
  kernel_params[count++] = &params->tiled_mma;
  if (pingpong)
    kernel_params[count++] = &params->mma_out;
  kernel_params[count++] = &params->n;
  kernel_params[count++] = &params->num_tiles;
  kernel_params[count++] = &params->eps;
  kernel_params[count++] = &params->rows;
  return count;
}

/* The driver takes one pointer per parameter, so host struct padding is irrelevant; only each
 * parameter's width must match the CUBIN.
 */
static_assert(std::is_standard_layout_v<SM90Params>);
static_assert(sizeof(CUtensorMap) == 0x80 && alignof(CUtensorMap) >= 64);
static_assert(sizeof(SM90Params::x_coord) == 0x04, "unexpected trimul KF K1 row-coordinate width");
static_assert(sizeof(SM90Params::a_coord) == 0x08, "unexpected trimul KF K1 a/b coordinate width");
static_assert(sizeof(SM90Params::seqlen) == 0x10, "unexpected trimul KF K1 seqlen width");
static_assert(sizeof(SM90Params::vec) == 0x08, "unexpected trimul KF K1 vec width");
static_assert(sizeof(SM90Params::stats) == 0x10, "unexpected trimul KF K1 stats width");
static_assert(sizeof(SM90Params::tiled_mma) == 0x01, "unexpected trimul KF K1 MMA token width");
static_assert(sizeof(SM90Params::eps) == 0x04, "unexpected trimul KF K1 eps width");
static_assert(sm90_parameter_count(0) == 15U && sm90_parameter_count(1) == 15U && sm90_parameter_count(2) == 16U);

} // namespace bioir::cutedsl::trimul_kf_k1::abi

namespace bioir::cutedsl::trimul_kf_k1
{

enum class DType : std::uint8_t
{
  kBFloat16,
};

/* Launch fields generated with the corresponding CUBIN, which runs the configs' K1_<kernel_variant>.
 * Variants 1 and 2 read the interleaved weight fold; variant 2 also writes the row statistics.
 */
struct KernelSpec
{
  std::int32_t target_sm;
  std::int32_t kernel_sm;
  std::int32_t C;
  std::int32_t D;
  std::int32_t kernel_variant;
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
    throw std::invalid_argument("trimul KF K1 CUBIN has no dynamic shared-memory metadata");
  return config.cubin.dynamic_smem_bytes;
}

/* K1 operands, all flat and contiguous: x is [rows * C] with rows = nb * n * n; seqlen holds one
 * int32 prefix length per (b, i) row; w_in is the [2D * C] projection fold, or the [4D * C]
 * interleaved fold; w_gate_in the [2D * C] gate fold, read by K1_0 alone; vec_in the fp32 [8D]
 * folded bias terms, which carry the LayerNorm shift and any projection biases; a and b the bf16
 * [nb * D * n * n] channel-major outputs; stats the fp32 row statistics, written by K1_2 alone and
 * sized for rows rounded up to whole 128-row tiles.
 */
struct LaunchParams
{
  Tensor1View x;
  Tensor1View seqlen;
  Tensor1View w_in;
  Tensor1View w_gate_in;
  Tensor1View vec_in;
  Tensor1View a;
  Tensor1View b;
  Tensor1View stats;
  std::int32_t rows{};
  std::int32_t n{};
  std::int32_t nb{};
  float eps{};
  std::uint64_t stream{};
};

std::vector<KernelSpec> kernel_specs();

KernelConfig
make_kernel_config(std::int32_t target_sm, DType dtype, std::int32_t C, std::int32_t D, std::int32_t kernel_variant);

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace bioir::cutedsl::trimul_kf_k1

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_TRIMUL_KF_K1_LAUNCHER_H_ */
