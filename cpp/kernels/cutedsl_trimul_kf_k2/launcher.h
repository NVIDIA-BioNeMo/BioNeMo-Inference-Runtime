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

/* SM90 TriMul KF K2 CUBIN configuration, device ABI, and launcher interface. */

#ifndef BIOIR_CPP_KERNELS_CUTEDSL_TRIMUL_KF_K2_LAUNCHER_H_
#define BIOIR_CPP_KERNELS_CUTEDSL_TRIMUL_KF_K2_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_launch_utils.h"
#include "cutedsl_tensor_abi.h"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace bioir::cutedsl::trimul_kf_k2::embedded
{
struct CubinImage;
}

/* Hopper device ABI trimul_kf_k2_sm90_v1, read from EIATTR_KPARAM_INFO for variant K2_0:
 *
 *   ord  0  0x030  0x80  a_tma          CUtensorMap
 *   ord  1  0x0b0  0x0c  a_coord        CoordTensorS3   (n, n, l)
 *   ord  2  0x0f0  0x80  b_tma          CUtensorMap
 *   ord  3  0x170  0x0c  b_coord        CoordTensorS3   (n, n, l)
 *   ord  4  0x1b0  0x80  prod_tma       CUtensorMap
 *   ord  5  0x230  0x0c  prod_coord     CoordTensorS3   (n, n, l)
 *   ord  6  0x23c  0x01  tiled_mma      std::uint8_t
 *   ord  7  0x23d  0x01  tiled_mma_out  std::uint8_t
 *   ord  8  0x240  0x04  m_clusters     std::int32_t
 *   ord  9  0x244  0x04  n_tiles        std::int32_t
 *   ord 10  0x248  0x04  work_units     std::int32_t
 *
 * K2_1 has a single MMA token and ends with its row-tile count and total tile count
 * (m_tiles, num_tiles) instead of the cluster schedule. Each 0x80 TMA slot carries CuTe DSL
 * 4.5.2's 0x40-byte non-executable CopyAtom payload followed by zero padding: loads of a and b,
 * one of them multicast across a 2-CTA cluster when the image clusters, and the store of prod.
 */
namespace bioir::cutedsl::trimul_kf_k2::abi
{

struct SM90Params
{
  CUtensorMap a_tma;
  CoordTensorS3 a_coord;
  CUtensorMap b_tma;
  CoordTensorS3 b_coord;
  CUtensorMap prod_tma;
  CoordTensorS3 prod_coord;
  std::uint8_t tiled_mma;
  std::uint8_t tiled_mma_out;
  /* K2_0: m_clusters, n_tiles, work_units. K2_1: m_tiles, num_tiles, and the third unused. */
  std::int32_t schedule[3];
};

constexpr std::size_t sm90_parameter_count(std::int32_t kernel_variant)
{
  return kernel_variant == 0 ? 11U : 9U;
}

inline constexpr std::size_t kSM90MaxParameterCount = 11;

/* params and kernel_params must stay alive until the launch call returns. */
inline std::size_t
pack_sm90_kernel_params(SM90Params* params, std::int32_t kernel_variant, void* kernel_params[kSM90MaxParameterCount])
{
  std::size_t count = 0;
  kernel_params[count++] = &params->a_tma;
  kernel_params[count++] = &params->a_coord;
  kernel_params[count++] = &params->b_tma;
  kernel_params[count++] = &params->b_coord;
  kernel_params[count++] = &params->prod_tma;
  kernel_params[count++] = &params->prod_coord;
  kernel_params[count++] = &params->tiled_mma;
  if (kernel_variant == 0)
    kernel_params[count++] = &params->tiled_mma_out;
  std::size_t const schedule_count = kernel_variant == 0 ? 3U : 2U;
  for (std::size_t index = 0; index < schedule_count; ++index)
    kernel_params[count++] = &params->schedule[index];
  return count;
}

/* The driver takes one pointer per parameter, so host struct padding is irrelevant; only each
 * parameter's width must match the CUBIN.
 */
static_assert(std::is_standard_layout_v<SM90Params>);
static_assert(sizeof(CUtensorMap) == 0x80 && alignof(CUtensorMap) >= 64);
static_assert(sizeof(SM90Params::a_coord) == 0x0c, "unexpected trimul KF K2 coordinate width");
static_assert(sizeof(SM90Params::tiled_mma) == 0x01, "unexpected trimul KF K2 MMA token width");
static_assert(sizeof(SM90Params::schedule[0]) == 0x04, "unexpected trimul KF K2 schedule width");
static_assert(sm90_parameter_count(0) == 11U && sm90_parameter_count(1) == 9U);

} // namespace bioir::cutedsl::trimul_kf_k2::abi

namespace bioir::cutedsl::trimul_kf_k2
{

enum class DType : std::uint8_t
{
  kBFloat16,
};

/* Launch fields generated with the corresponding CUBIN, which runs the configs' K2_<kernel_variant>
 * on 128 x tile_n output tiles. K2_0 clusters cluster_m CTAs along M, overlaps each tile's stores
 * with the next tile's first defer_kmin k-blocks, and splits its epilogue per warpgroup under
 * split_epi; K2_1 clusters along N at tile_n 192.
 */
struct KernelSpec
{
  std::int32_t target_sm;
  std::int32_t kernel_sm;
  bool outgoing;
  std::int32_t kernel_variant;
  std::uint32_t tile_m;
  std::uint32_t tile_n;
  std::uint32_t cluster_m;
  std::uint32_t defer_kmin;
  bool split_epi;
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
    throw std::invalid_argument("trimul KF K2 CUBIN has no dynamic shared-memory metadata");
  return config.cubin.dynamic_smem_bytes;
}

/* a, b and prod are flat, channel-major bf16 buffers holding l = B * D matrices of n x n:
 * prod[l, i, j] = sum_k a[l, i, k] b[l, j, k] outgoing, a[l, k, i] b[l, k, j] incoming. The rows of
 * a and b sit ab_pitch elements apart and their matrices ab_plane apart; prod is dense, [l * n * n].
 */
struct LaunchParams
{
  FlatTensorView a;
  FlatTensorView b;
  FlatTensorView prod;
  std::int32_t n{};
  std::int32_t l{};
  std::int64_t ab_pitch{};
  std::int64_t ab_plane{};
  std::uint64_t stream{};
};

std::vector<KernelSpec> kernel_specs();

KernelConfig make_kernel_config(
  std::int32_t target_sm,
  DType dtype,
  bool outgoing,
  std::int32_t kernel_variant,
  std::uint32_t tile_n,
  std::uint32_t cluster_m,
  std::uint32_t defer_kmin,
  bool split_epi);

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace bioir::cutedsl::trimul_kf_k2

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_TRIMUL_KF_K2_LAUNCHER_H_ */
