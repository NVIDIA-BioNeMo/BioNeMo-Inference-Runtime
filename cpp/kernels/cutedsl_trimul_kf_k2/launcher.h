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
 *   ord 11  0x250  0x08  counter        std::uint64_t   (dynamic images only)
 *
 * K2_2 lowers the same bank. A dynamic image of K2_0 or K2_2 claims its work units from a global
 * int32 counter and takes the counter's address last. K2_1 has a single MMA token and ends with
 * its row-tile count and total tile count (m_tiles, num_tiles) instead of the cluster schedule.
 * Each 0x80 TMA slot
 * carries CuTe DSL 4.5.2's 0x40-byte non-executable CopyAtom payload followed by zero padding:
 * loads of a and b, one of them multicast across a 2-CTA cluster when the image clusters, and
 * the store of prod.
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
  /* K2_0 and K2_2: m_clusters, column clusters, work_units. K2_1: m_tiles, num_tiles, and the
   * third unused.
   */
  std::int32_t schedule[3];
  /* Dynamic images: global address of the int32 work counter. */
  std::uint64_t counter;
};

constexpr std::size_t sm90_parameter_count(std::int32_t kernel_variant, bool dynamic = false)
{
  return kernel_variant == 1 ? 9U : dynamic ? 12U : 11U;
}

inline constexpr std::size_t kSM90MaxParameterCount = 12;

/* params and kernel_params must stay alive until the launch call returns. */
inline std::size_t pack_sm90_kernel_params(
  SM90Params* params, std::int32_t kernel_variant, bool dynamic, void* kernel_params[kSM90MaxParameterCount])
{
  std::size_t count = 0;
  kernel_params[count++] = &params->a_tma;
  kernel_params[count++] = &params->a_coord;
  kernel_params[count++] = &params->b_tma;
  kernel_params[count++] = &params->b_coord;
  kernel_params[count++] = &params->prod_tma;
  kernel_params[count++] = &params->prod_coord;
  kernel_params[count++] = &params->tiled_mma;
  if (kernel_variant != 1)
    kernel_params[count++] = &params->tiled_mma_out;
  std::size_t const schedule_count = kernel_variant == 1 ? 2U : 3U;
  for (std::size_t index = 0; index < schedule_count; ++index)
    kernel_params[count++] = &params->schedule[index];
  if (dynamic)
    kernel_params[count++] = &params->counter;
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
static_assert(sizeof(SM90Params::counter) == 0x08, "unexpected trimul KF K2 counter width");
static_assert(sm90_parameter_count(0) == 11U && sm90_parameter_count(1) == 9U && sm90_parameter_count(2) == 11U);
static_assert(sm90_parameter_count(0, true) == 12U && sm90_parameter_count(2, true) == 12U);

} // namespace bioir::cutedsl::trimul_kf_k2::abi

namespace bioir::cutedsl::trimul_kf_k2
{

enum class DType : std::uint8_t
{
  kBFloat16,
};

/* Launch fields generated with the corresponding CUBIN, which runs the configs' K2_<kernel_variant>
 * on 128 x tile_n output tiles. K2_0 clusters cluster_m CTAs along M and K2_2 along N; both overlap
 * each tile's stores with the next tile's first defer_kmin k-blocks, split their epilogue per
 * warpgroup under split_epi, and under dynamic claim their work units from a counter instead of a
 * static round-robin. K2_1 clusters along N at tile_n 192.
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
  bool dynamic;
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
 * a and b sit ab_pitch elements apart and their matrices ab_plane apart; the rows of prod sit
 * prod_pitch apart and its matrices prod_plane apart (n and n * n for a dense product). The kernel
 * writes the first n columns of every prod row. A dynamic image claims its work from counter, int32
 * words on the launch's device: zero before the launch and zero again after it, so launches sharing one
 * must be stream-ordered. Static images ignore it.
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
  std::int64_t prod_pitch{};
  std::int64_t prod_plane{};
  FlatTensorView counter;
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
  bool split_epi,
  bool dynamic = false);

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace bioir::cutedsl::trimul_kf_k2

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_TRIMUL_KF_K2_LAUNCHER_H_ */
