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

/* SM90 TriMul KF K3 CUBIN configuration, device ABI, and launcher interface. */

#ifndef BIOIR_CPP_KERNELS_CUTEDSL_TRIMUL_KF_K3_LAUNCHER_H_
#define BIOIR_CPP_KERNELS_CUTEDSL_TRIMUL_KF_K3_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_launch_utils.h"
#include "cutedsl_tensor_abi.h"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace bioir::cutedsl::trimul_kf_k3::embedded
{
struct CubinImage;
}

/* Hopper device ABI trimul_kf_k3_sm90_v2, read from EIATTR_KPARAM_INFO for variant K3_1 with
 * the fused residual (K3_3 lays out the same bank):
 *
 *   ord  0  0x030  0x80  prod_tma    CUtensorMap
 *   ord  1  0x0b0  0x08  prod_coord  CoordTensorS2   (N * N, B)
 *   ord  2  0x0f0  0x80  x_tma       CUtensorMap
 *   ord  3  0x170  0x04  x_coord     CoordTensorS1   rows
 *   ord  4  0x1b0  0x80  w_out_tma   CUtensorMap
 *   ord  5  0x230  0x80  w_gate_tma  CUtensorMap
 *   ord  6  0x2b0  0x80  output_tma  CUtensorMap
 *   ord  7  0x330  0x04  output_coord CoordTensorS1  rows
 *   ord  8  0x338  0x08  vec         cute_tensor_s0_d0_t
 *   ord  9  0x340  0x10  stats       cute_tensor_s1_d0_t  (extent = rows)
 *   ord 10  0x350  0x10  seqlen      cute_tensor_s1_d0_t
 *   ord 11  0x360  0x01  mma_p       std::uint8_t
 *   ord 12  0x361  0x01  mma_x       std::uint8_t
 *   ord 13  0x364  0x04  n           std::int32_t
 *   ord 14  0x368  0x04  num_tiles   std::int32_t
 *   ord 15  0x36c  0x04  eps         float
 *   ord 16  0x370  0x04  rows        std::int32_t
 *
 * K3_0 and K3_2 re-reduce the row statistics, so stats is absent; without the fused residual, seqlen
 * is absent; the remaining ordinals shift down. The weights have static shapes, so their coordinate
 * tensors lower to no parameter. Each 0x80 TMA slot carries CuTe DSL 4.5.2's 0x40-byte
 * non-executable CopyAtom payload followed by zero padding: plain loads of prod, x and the weights,
 * and one store of the output.
 */
namespace bioir::cutedsl::trimul_kf_k3::abi
{

struct SM90Params
{
  CUtensorMap prod_tma;
  CoordTensorS2 prod_coord;
  CUtensorMap x_tma;
  CoordTensorS1 x_coord;
  CUtensorMap w_out_tma;
  CUtensorMap w_gate_tma;
  CUtensorMap output_tma;
  CoordTensorS1 output_coord;
  cute_tensor_s0_d0_t vec;
  cute_tensor_s1_d0_t stats;
  cute_tensor_s1_d0_t seqlen;
  std::uint8_t mma_p;
  std::uint8_t mma_x;
  std::int32_t n;
  std::int32_t num_tiles;
  float eps;
  std::int32_t rows;
};

constexpr std::size_t sm90_parameter_count(bool reads_stats, bool residual)
{
  return 15U + (reads_stats ? 1U : 0U) + (residual ? 1U : 0U);
}

inline constexpr std::size_t kSM90MaxParameterCount = sm90_parameter_count(true, true);

/* params and kernel_params must stay alive until the launch call returns. */
inline std::size_t pack_sm90_kernel_params(
  SM90Params* params, bool reads_stats, bool residual, void* kernel_params[kSM90MaxParameterCount])
{
  std::size_t count = 0;
  kernel_params[count++] = &params->prod_tma;
  kernel_params[count++] = &params->prod_coord;
  kernel_params[count++] = &params->x_tma;
  kernel_params[count++] = &params->x_coord;
  kernel_params[count++] = &params->w_out_tma;
  kernel_params[count++] = &params->w_gate_tma;
  kernel_params[count++] = &params->output_tma;
  kernel_params[count++] = &params->output_coord;
  kernel_params[count++] = &params->vec;
  if (reads_stats)
    kernel_params[count++] = &params->stats;
  if (residual)
    kernel_params[count++] = &params->seqlen;
  kernel_params[count++] = &params->mma_p;
  kernel_params[count++] = &params->mma_x;
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
static_assert(sizeof(SM90Params::prod_coord) == 0x08, "unexpected trimul KF K3 product-coordinate width");
static_assert(sizeof(SM90Params::x_coord) == 0x04, "unexpected trimul KF K3 row-coordinate width");
static_assert(sizeof(SM90Params::vec) == 0x08, "unexpected trimul KF K3 vec width");
static_assert(sizeof(SM90Params::stats) == 0x10, "unexpected trimul KF K3 stats width");
static_assert(sizeof(SM90Params::seqlen) == 0x10, "unexpected trimul KF K3 seqlen width");
static_assert(sizeof(SM90Params::mma_p) == 0x01, "unexpected trimul KF K3 MMA token width");
static_assert(sizeof(SM90Params::eps) == 0x04, "unexpected trimul KF K3 eps width");
static_assert(sm90_parameter_count(false, false) == 15U && sm90_parameter_count(true, true) == 17U);

} // namespace bioir::cutedsl::trimul_kf_k3::abi

namespace bioir::cutedsl::trimul_kf_k3
{

enum class DType : std::uint8_t
{
  kBFloat16,
};

/* Launch fields generated with the corresponding CUBIN, which runs the configs' K3_<kernel_variant>.
 * Variants 1 and 3 read K1's row statistics; residual fuses (x + update) * mask into the output.
 * Each tile_m-row tile spans tile_ctas CTAs, one per output-column group.
 */
struct KernelSpec
{
  std::int32_t target_sm;
  std::int32_t kernel_sm;
  std::int32_t C;
  std::int32_t D;
  std::int32_t kernel_variant;
  bool residual;
  std::uint32_t num_threads;
  std::uint32_t tile_m;
  std::uint32_t tile_ctas;
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
    throw std::invalid_argument("trimul KF K3 CUBIN has no dynamic shared-memory metadata");
  return config.cubin.dynamic_smem_bytes;
}

/* K3 operands, all flat and contiguous: prod is the bf16 [nb * D * n * n] channel-major product;
 * x and output are [rows * C] with rows = nb * n * n; w_out is the [C * D] output-projection fold,
 * w_gate_out the [C * C] gate fold and vec_out the fp32 [4C] folded bias terms, which carry the
 * LayerNorm shifts and any output biases; stats K1's row statistics, read by K3_1 and K3_3; seqlen
 * one int32 prefix length per (b, i) output row, read with the fused residual alone. output must
 * not alias x: a CTA streams every column of its x rows again for each output column group.
 */
struct LaunchParams
{
  FlatTensorView prod;
  FlatTensorView x;
  Tensor1View w_out;
  Tensor1View w_gate_out;
  Tensor1View vec_out;
  FlatTensorView stats;
  Tensor1View seqlen;
  FlatTensorView output;
  std::int32_t rows{};
  std::int32_t n{};
  std::int32_t nb{};
  float eps{};
  std::uint64_t stream{};
};

std::vector<KernelSpec> kernel_specs();

KernelConfig make_kernel_config(
  std::int32_t target_sm, DType dtype, std::int32_t C, std::int32_t D, std::int32_t kernel_variant, bool residual);

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace bioir::cutedsl::trimul_kf_k3

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_TRIMUL_KF_K3_LAUNCHER_H_ */
