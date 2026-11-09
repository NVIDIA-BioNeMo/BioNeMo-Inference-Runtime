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

/* Outer-product-mean CUBIN configuration, device ABI, and launcher interface. */

#ifndef BIOIR_CPP_KERNELS_CUTEDSL_OUTER_PRODUCT_MEAN_LAUNCHER_H_
#define BIOIR_CPP_KERNELS_CUTEDSL_OUTER_PRODUCT_MEAN_LAUNCHER_H_

#include "cubin_runtime.h"
#include "cutedsl_launch_utils.h"
#include "cutedsl_tensor_abi.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace bioir::cutedsl::outer_product_mean::embedded
{
struct CubinImage;
}

/* Direct CUDA Driver launch ABI for the Ampere-style outer-product-mean.
 *
 * This is the device-kernel ABI, not the high-level CuTeDSL __call__ ABI, and
 * was read back from the compiled CUBINs' EIATTR_KPARAM_INFO: six parameters
 * over 176 bytes.
 *
 * `weight` and `bias` are BARE POINTERS rather than descriptors because the
 * kernel compiles C, D and C_z as constants (32 / 32 / 128), leaving W_o's
 * [C_z, C*D] and the bias's [C_z] fully static. Nothing about them is dynamic
 * except the address.
 *
 * `has_bias=false` REMOVES the bias slot rather than passing null, so the bank
 * is 168 bytes over five parameters instead of 176 over six, and `output`
 * shifts down one ordinal. Verified against both compiled CUBINs; the bias flag
 * is therefore an ABI axis. `norm_before` is not -- it only moves where the
 * division happens inside the kernel.
 *
 * The layout/tiled-copy/tiled-mma arguments in the @cute.kernel signature are
 * compile-time objects and are traced away; they occupy no parameter slot.
 */
namespace bioir::cutedsl::outer_product_mean::abi
{

inline constexpr std::size_t kSM80ParameterCountWithBias = 6;
inline constexpr std::size_t kSM80ParameterCountNoBias = 5;
inline constexpr std::size_t kSM80MaxParameterCount = kSM80ParameterCountWithBias;

struct SM80Params
{
  cute_tensor_s3_d2_t a;
  cute_tensor_s3_d2_t b;
  cute_tensor_s3_d2_t num_mask;
  cute_tensor_s0_d0_t weight;
  cute_tensor_s0_d0_t bias;
  cute_tensor_s3_d2_t output;
};

/* Both params and kernel_params must remain alive until the CUDA launch call
 * returns.
 */
inline std::size_t
pack_sm80_kernel_params(SM80Params* params, void* kernel_params[kSM80MaxParameterCount], bool has_bias)
{
  std::size_t index = 0;
  kernel_params[index++] = &params->a;
  kernel_params[index++] = &params->b;
  kernel_params[index++] = &params->num_mask;
  kernel_params[index++] = &params->weight;
  if (has_bias)
    kernel_params[index++] = &params->bias;
  kernel_params[index++] = &params->output;
  return index;
}

static_assert(sizeof(cute_tensor_s3_d2_t) == 40, "outer-product-mean s3_d2 operand size changed");
static_assert(sizeof(cute_tensor_s0_d0_t) == 8, "outer-product-mean bare-pointer operand size changed");
static_assert(offsetof(SM80Params, a) == 0, "unexpected outer-product-mean a offset");
static_assert(offsetof(SM80Params, b) == 40, "unexpected outer-product-mean b offset");
static_assert(offsetof(SM80Params, num_mask) == 80, "unexpected outer-product-mean num_mask offset");
static_assert(offsetof(SM80Params, weight) == 120, "unexpected outer-product-mean weight offset");
static_assert(offsetof(SM80Params, bias) == 128, "unexpected outer-product-mean bias offset");
static_assert(offsetof(SM80Params, output) == 136, "unexpected outer-product-mean output offset");
static_assert(sizeof(SM80Params) == 176, "outer-product-mean SM80 parameter bank size changed");
/* Without the bias slot the lowered bank is 168 bytes over five parameters.
 * The backing struct is unchanged; only which pointers are handed to the
 * driver, and in what order, differ.
 */
static_assert(
  offsetof(SM80Params, output) - sizeof(cute_tensor_s0_d0_t) == 128,
  "outer-product-mean no-bias output ordinal changed");

/* Hopper device ABI from EIATTR_KPARAM_INFO (launch ABI outer_product_mean_sm90):
 *
 *   ord  0  0x030  0x80  a_tma       non-executable TMA load atom
 *   ord  1  0x0b0  0x0c  a_coord     CoordTensorS3 {I, S, B}
 *   ord  2  0x0f0  0x80  a2_tma      second C half (the same map at csplit=1)
 *   ord  3  0x170  0x0c  a2_coord
 *   ord  4  0x1b0  0x80  b_tma
 *   ord  5  0x230  0x0c  b_coord     CoordTensorS3 {J*D, S, B}
 *   ord  6  0x270  0x80  w_tma       static extents: no coordinate tensor
 *   ord  7  0x2f0  0x28  num_mask    cute_tensor_s3_d2_t
 *   ord  8  0x318  0x08  bias        pointer; the slot is absent without a bias
 *   ord  9  0x320  0x28  output      cute_tensor_s3_d2_t
 *   ord 10  0x348  0x04  i_tiles, then j_tiles, k_tiles, I and J (int32 each)
 *   ord 15  0x35c  0x01  S-GEMM tiled-MMA accumulate flag
 *   ord 16  0x35d  0x01  O-GEMM tiled-MMA accumulate flag
 *
 * The bank is the same for every tile, C split and C_z.
 */
struct SM90Params
{
  CUtensorMap a_tma;
  CoordTensorS3 a_coord;
  CUtensorMap a2_tma;
  CoordTensorS3 a2_coord;
  CUtensorMap b_tma;
  CoordTensorS3 b_coord;
  CUtensorMap w_tma;
  cute_tensor_s3_d2_t num_mask;
  cute_tensor_s0_d0_t bias;
  cute_tensor_s3_d2_t output;
  std::int32_t scalars[5];
  std::uint8_t mma_s;
  std::uint8_t mma_o;
};

inline constexpr std::size_t kSM90MaxParameterCount = 17;

inline constexpr std::size_t sm90_parameter_count(bool has_bias)
{
  return has_bias ? kSM90MaxParameterCount : kSM90MaxParameterCount - 1;
}

/* Returns the number of packed parameters. */
inline std::size_t
pack_sm90_kernel_params(SM90Params* params, void* kernel_params[kSM90MaxParameterCount], bool has_bias)
{
  std::size_t index = 0;
  kernel_params[index++] = &params->a_tma;
  kernel_params[index++] = &params->a_coord;
  kernel_params[index++] = &params->a2_tma;
  kernel_params[index++] = &params->a2_coord;
  kernel_params[index++] = &params->b_tma;
  kernel_params[index++] = &params->b_coord;
  kernel_params[index++] = &params->w_tma;
  kernel_params[index++] = &params->num_mask;
  if (has_bias)
    kernel_params[index++] = &params->bias;
  kernel_params[index++] = &params->output;
  for (std::int32_t& scalar : params->scalars)
    kernel_params[index++] = &scalar;
  kernel_params[index++] = &params->mma_s;
  kernel_params[index++] = &params->mma_o;
  return index;
}

static_assert(sizeof(CUtensorMap) == 0x80, "outer-product-mean TMA atom slot size changed");
static_assert(sizeof(CoordTensorS3) == 0x0c, "outer-product-mean coordinate tensor size changed");

/* Blackwell device ABI from EIATTR_KPARAM_INFO (launch ABI outer_product_mean_sm100):
 *
 *   ord  0  0x000  0x03  mma_s       S-GEMM tiled-MMA flags, three i1 the host passes clear
 *   ord  1  0x003  0x03  mma_o       O-GEMM tiled-MMA flags
 *   ord  2  0x040  0x80  a_tma       2-SM load atom over the relaid a
 *   ord  3  0x0c0  0x0c  a_coord     CoordTensorS3 {rows, S, B}
 *   ord  4  0x100  0x80  b_tma       2-SM load atom
 *   ord  5  0x180  0x0c  b_coord     CoordTensorS3 {J*D, S, B}
 *   ord  6  0x1c0  0x80  w_tma       static extents: no coordinate tensor
 *   ord  7  0x240  0x80  o_tma       store atom
 *   ord  8  0x2c0  0x0c  o_coord     CoordTensorS3 {I, J, B}
 *   ord  9  0x2d0  0x28  num_mask    cute_tensor_s3_d2_t
 *   ord 10  0x2f8  0x08  bias        pointer; the slot is absent without a bias
 *   ord 11  0x300  0x28  output      cute_tensor_s3_d2_t, at 0x2f8 without a bias
 *
 * The bank is the same for every tile, cluster shape and C_z. The 2-SM a and
 * the W atoms multicast across the cluster's CTA pairs when it holds two.
 */
struct SM100Params
{
  std::uint8_t mma_s[3];
  std::uint8_t mma_o[3];
  CUtensorMap a_tma;
  CoordTensorS3 a_coord;
  CUtensorMap b_tma;
  CoordTensorS3 b_coord;
  CUtensorMap w_tma;
  CUtensorMap o_tma;
  CoordTensorS3 o_coord;
  cute_tensor_s3_d2_t num_mask;
  cute_tensor_s0_d0_t bias;
  cute_tensor_s3_d2_t output;
};

inline constexpr std::size_t kSM100MaxParameterCount = 12;

inline constexpr std::size_t sm100_parameter_count(bool has_bias)
{
  return has_bias ? kSM100MaxParameterCount : kSM100MaxParameterCount - 1;
}

/* Returns the number of packed parameters. */
inline std::size_t
pack_sm100_kernel_params(SM100Params* params, void* kernel_params[kSM100MaxParameterCount], bool has_bias)
{
  std::size_t index = 0;
  kernel_params[index++] = &params->mma_s;
  kernel_params[index++] = &params->mma_o;
  kernel_params[index++] = &params->a_tma;
  kernel_params[index++] = &params->a_coord;
  kernel_params[index++] = &params->b_tma;
  kernel_params[index++] = &params->b_coord;
  kernel_params[index++] = &params->w_tma;
  kernel_params[index++] = &params->o_tma;
  kernel_params[index++] = &params->o_coord;
  kernel_params[index++] = &params->num_mask;
  if (has_bias)
    kernel_params[index++] = &params->bias;
  kernel_params[index++] = &params->output;
  return index;
}

static_assert(std::is_standard_layout_v<SM100Params>);
static_assert(sizeof(SM100Params::mma_s) == 0x03, "unexpected outer-product-mean SM100 tiled-MMA width");
static_assert(sizeof(SM100Params::mma_o) == 0x03, "unexpected outer-product-mean SM100 tiled-MMA width");
static_assert(sizeof(SM100Params::a_coord) == 0x0c, "unexpected outer-product-mean SM100 coordinate width");
static_assert(sizeof(SM100Params::num_mask) == 0x28, "unexpected outer-product-mean SM100 num_mask width");
static_assert(sizeof(SM100Params::bias) == 0x08, "unexpected outer-product-mean SM100 bias width");
static_assert(sizeof(SM100Params::output) == 0x28, "unexpected outer-product-mean SM100 output width");
static_assert(sm100_parameter_count(false) == 11U);

} // namespace bioir::cutedsl::outer_product_mean::abi

namespace bioir::cutedsl::outer_product_mean
{

enum class DType : std::uint8_t
{
  kFloat16,
  kBFloat16,
};

/* Static problem dimensions the kernel bakes in. The launcher checks the
 * caller's operands against these and the image's C_z; a mismatch means the
 * interface picked the wrong payload, not a recoverable shape.
 */
inline constexpr std::int32_t kChannelsC = 32;
inline constexpr std::int32_t kChannelsD = 32;

/* One compiled tile configuration.
 *
 * As with gated sigmoid there is no hand-written spec table: B/S/I/J are
 * symbolic, so the config IS the runtime key. make_kernel_config() looks the
 * caller's config identity up directly in the generated registry, which the
 * builder emits from the same JSON the Python interface reads.
 */
struct KernelSpec
{
  std::int32_t target_sm;
  std::int32_t tile_i;
  std::int32_t tile_j;
  std::int32_t raster_factor;
  std::int32_t num_threads;
  std::int32_t c_z;
  /* SM90 kernel only: S-GEMM K tile, (i, j) tiles per CTA and C passes. */
  std::int32_t tile_s;
  std::int32_t ptile;
  std::int32_t csplit;
  bool is_sm90;
  /* The SM100 kernel's CTA owns tile_i = 128 / cc rows i; its CTA pair owns tile_j = cc columns j. */
  bool is_sm100;
};

struct KernelConfig
{
  KernelSpec spec;
  DType dtype;
  bool has_bias;
  bool norm_before;
  EmbeddedCubinImage cubin;
  embedded::CubinImage const* embedded_image;
};

inline std::uint32_t dynamic_smem_bytes(KernelConfig const& config)
{
  if (config.cubin.dynamic_smem_bytes == 0)
    throw std::invalid_argument("Outer-product-mean CUBIN has no dynamic shared-memory metadata");
  return config.cubin.dynamic_smem_bytes;
}

struct LaunchParams
{
  /* a is [B, S, I, C] and b is [B, S, J, D]; the descriptors keep the leading
   * three extents and two strides because C and D are static. The SM100 kernel
   * takes a relaid instead, as [B, C / cc, ceil(I / tile_i) * 128, S] with S
   * contiguous: each row holds one channel of one token.
   */
  Tensor4View a;
  Tensor4View b;
  Tensor3View num_mask;
  /* Rank-2 [C_z, C*D] and rank-1 [C_z] views. Only the address reaches the
   * kernel, but the extents let the launcher reject a wrong-shaped operand and
   * the device ordinal lets it reject a cross-device one.
   */
  Tensor2View weight;
  Tensor1View bias;
  Tensor4View output;
  std::uint64_t stream{};
};

KernelConfig
make_kernel_config(std::int32_t target_sm, DType dtype, bool has_bias, bool norm_before, char const* config_identity);

void launch(KernelConfig const& config, LaunchParams const& params);

} // namespace bioir::cutedsl::outer_product_mean

#endif /* BIOIR_CPP_KERNELS_CUTEDSL_OUTER_PRODUCT_MEAN_LAUNCHER_H_ */
