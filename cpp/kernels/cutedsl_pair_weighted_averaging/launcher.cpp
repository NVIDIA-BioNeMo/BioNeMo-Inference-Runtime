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

#include "launcher.h"

#include "pair_weighted_averaging_registry.h"

#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace bioir::cutedsl::pair_weighted_averaging
{
namespace
{

constexpr char kSM80LaunchAbi[] = "pair_weighted_averaging_sm80";
constexpr char kSM90LaunchAbi[] = "pair_weighted_averaging_sm90";
constexpr char kSM100LaunchAbi[] = "pair_weighted_averaging_sm100";
constexpr std::int32_t kSM80KernelSM = 80;
constexpr std::int32_t kSM90KernelSM = 90;
constexpr std::int32_t kSM100KernelSM = 100;
/* SM103 runs the SM100 kernels, built for its own architecture. */
constexpr std::int32_t kSM103TargetSM = 103;
/* The two Hopper kernels share kSM90LaunchAbi; each image records which one it holds. */
constexpr std::int32_t kSM90D32Variant = 0;
constexpr std::int32_t kSM90D8Variant = 1;
/* The two tcgen05 kernels share kSM100LaunchAbi the same way. */
constexpr std::int32_t kSM100D32Variant = 0;
constexpr std::int32_t kSM100D8Variant = 1;
constexpr std::int32_t kH = 8;
constexpr std::int32_t kSM90D8HeadDim = 8;
constexpr std::int32_t kSM100D32HeadDim = 32;
constexpr std::int32_t kSM100D8HeadDim = 8;
constexpr std::int32_t kJAlignment = 8;
/* A tcgen05 CTA owns 128 tokens; a 256-token tile (the D=32 pair tile, a D=8 pair) runs on a CTA pair. */
constexpr std::uint32_t kSM100CtaTokens = 128;
/* The D=32 kernel's tiles: the direct and pair tiles take four MSA rows and the split tile eight, all over 192
 * threads; the TMA epilogue takes one, two or four over 192 or 320.
 */
constexpr std::uint32_t kSM100DirectTileS = 4;
constexpr std::uint32_t kSM100SplitTileS = 8;
constexpr std::uint32_t kSM100DirectThreads = 192;
constexpr std::uint32_t kSM100WideThreads = 320;
constexpr std::uint32_t kSM100D8Threads = 512;
/* The portable cluster limit; the D=8 configs stay within it. */
constexpr std::uint32_t kSM100MaxClusterCtas = 8;

/* A Hopper D=8 cluster is i token tiles by s MSA-row tiles: w multicasts across the MSA-row tiles, v across the
 * token tiles, and the grid lists clusters token-major, then MSA rows, then batch.
 */
struct SM90Cluster
{
  std::uint32_t i;
  std::uint32_t s;
};

/* The shape of a Hopper D=8 cluster from its size. The kernel builds no two shapes of one size, and the committed
 * SM90 images predate cluster_i and cluster_s (which read 1x1 there), so the size names the shape. Returns {0, 0}
 * for a size the kernel never builds.
 */
SM90Cluster sm90_d8_cluster(std::uint32_t size)
{
  switch (size)
  {
  case 1: return {1, 1};
  case 4: return {1, 4};
  case 8: return {2, 4};
  default: return {0, 0};
  }
}

/* A tcgen05 D=8 cluster is `i` token tiles by `s` MSA-row tiles, each tile on one CTA or, at 256 tokens, on a CTA pair.
 * w multicasts across the MSA-row tiles and v across the token tiles: each CTA of a pair loads v for half the tile's
 * MSA rows, and the `i` CTAs that share them split those rows evenly. The kernel strides the cluster tiles over a
 * persistent grid.
 */
struct SM100D8Cluster
{
  std::uint32_t pair;
  std::uint32_t i;
  std::uint32_t s;
  std::uint32_t ctas;
};

/* Returns zero CTAs for a tile or cluster the kernel does not build, or one the launch record disagrees with. */
SM100D8Cluster sm100_d8_cluster(embedded::CubinImage const& image)
{
  if (image.tile_i != kSM100CtaTokens && image.tile_i != 2 * kSM100CtaTokens)
    return {};
  std::uint32_t const pair = image.tile_i / kSM100CtaTokens;
  std::uint64_t const ctas = std::uint64_t{pair} * image.cluster_i * image.cluster_s;
  if (
    image.cluster_i == 0 || image.cluster_s == 0 || image.tile_s % pair != 0
    || (image.tile_s / pair) % image.cluster_i != 0 || ctas > kSM100MaxClusterCtas)
  {
    return {};
  }
  if (image.sm100.cluster_dims[0] != ctas || image.sm100.cluster_dims[1] != 1 || image.sm100.cluster_dims[2] != 1)
    return {};
  return {pair, image.cluster_i, image.cluster_s, static_cast<std::uint32_t>(ctas)};
}

struct EmbeddedSelection
{
  embedded::CubinImage const* image;
  embedded::RuntimeAlias const* alias;
};

bool dtype_is_bfloat16(DType dtype)
{
  switch (dtype)
  {
  case DType::kFloat16: return false;
  case DType::kBFloat16: return true;
  }
  throw std::invalid_argument("pair_weighted_averaging config has an unsupported dtype");
}

bool equal_c_strings(char const* lhs, char const* rhs)
{
  return lhs != nullptr && rhs != nullptr && std::strcmp(lhs, rhs) == 0;
}

double anchor_distance(std::int32_t anchor, double runtime_extent)
{
  return std::abs(static_cast<double>(anchor) - runtime_extent);
}

/* The rank of each tensor map a tcgen05 kernel takes, 0 for a map it does not build. The materializer's _Sm100Spec
 * declares the same operands and the builder's _SM100_OPERANDS the atoms behind them; the views they map are listed
 * above head_split_source().
 */
struct SM100MapRanks
{
  std::uint32_t w;
  std::uint32_t v;
  std::uint32_t wo;
  std::uint32_t g;
  std::uint32_t output;
};

constexpr SM100MapRanks kSM100D8Maps{4, 5, 2, 4, 0};

bool has_map_ranks(embedded::SM100LaunchInfo const& launch, SM100MapRanks const& ranks)
{
  return launch.w.rank == ranks.w && launch.v.rank == ranks.v && launch.wo.rank == ranks.wo && launch.g.rank == ranks.g
    && launch.output.rank == ranks.output;
}

/* One D=32 tile: the device kernel it runs, the tensor maps it takes, and the CTAs one token tile occupies. */
struct SM100D32Tile
{
  abi::SM100D32Kernel kernel;
  SM100MapRanks maps;
  std::uint32_t ctas;
};

/* The direct and pair tiles load g and store the output from registers; the split tile writes the gated value and
 * leaves Wo to the caller.
 */
constexpr SM100D32Tile kSM100D32TmaEpilogueTile{abi::SM100D32Kernel::kTmaEpilogue, {4, 5, 2, 5, 4}, 1};
constexpr SM100D32Tile kSM100D32DirectTile{abi::SM100D32Kernel::kRegisterEpilogue, {4, 5, 2, 0, 0}, 1};
constexpr SM100D32Tile kSM100D32PairTile{abi::SM100D32Kernel::kRegisterEpilogue, {4, 5, 2, 0, 0}, 2};
constexpr SM100D32Tile kSM100D32SplitTile{abi::SM100D32Kernel::kGatedValue, {4, 5, 0, 0, 0}, 1};

/* Tells the D=32 tiles apart as the kernel does; nullptr for a tile it does not build. */
SM100D32Tile const* sm100_d32_tile(embedded::CubinImage const& image)
{
  bool const narrow = image.num_threads == kSM100DirectThreads;
  if (narrow && image.tile_s == kSM100SplitTileS && image.tile_i == kSM100CtaTokens)
    return &kSM100D32SplitTile;
  if (narrow && image.tile_s == kSM100DirectTileS)
  {
    if (image.tile_i == 2 * kSM100CtaTokens)
      return &kSM100D32PairTile;
    if (image.tile_i == kSM100CtaTokens)
      return &kSM100D32DirectTile;
  }
  bool const tma_rows = image.tile_s == 1 || image.tile_s == 2 || image.tile_s == kSM100DirectTileS;
  if (image.tile_i == kSM100CtaTokens && tma_rows && (narrow || image.num_threads == kSM100WideThreads))
    return &kSM100D32TmaEpilogueTile;
  return nullptr;
}

bool is_sm100_d32(embedded::CubinImage const& image)
{
  return image.cubin.kernel_sm == kSM100KernelSM && image.kernel_variant == kSM100D32Variant;
}

/* The tcgen05 D=8 kernel, the one kernel that loads v transposed. */
bool is_sm100_d8(embedded::CubinImage const& image)
{
  return image.cubin.kernel_sm == kSM100KernelSM && image.kernel_variant == kSM100D8Variant;
}

/* N of v: [B, S, N, H*D], or [B, S, H*D, N] for the tcgen05 D=8 kernel, which loads v transposed. */
std::int32_t value_tokens(embedded::CubinImage const& image, Tensor4View const& v)
{
  return is_sm100_d8(image) ? v.shape[3] : v.shape[2];
}

/* Columns of the output an image writes: c_m, or H*D for the gated value of the D=32 split tile. */
std::int32_t output_features(embedded::CubinImage const& image)
{
  if (is_sm100_d32(image) && sm100_d32_tile(image) == &kSM100D32SplitTile)
    return image.H * image.D;
  return image.c_m;
}

/* Each tcgen05 kernel's block, cluster and tensor maps. */
bool has_valid_sm100_launch(embedded::CubinImage const& image)
{
  embedded::SM100LaunchInfo const& launch = image.sm100;
  bool const target = image.cubin.target_sm == kSM100KernelSM || image.cubin.target_sm == kSM103TargetSM;
  if (
    !equal_c_strings(image.cubin.launch_abi, kSM100LaunchAbi) || !target || !launch.enabled || image.sm90.enabled
    || launch.block_dims[0] != image.num_threads || launch.block_dims[1] != 1 || launch.block_dims[2] != 1)
  {
    return false;
  }
  if (image.kernel_variant == kSM100D32Variant)
  {
    SM100D32Tile const* tile = sm100_d32_tile(image);
    return tile != nullptr && image.D == kSM100D32HeadDim && image.cluster_i == 1 && image.cluster_s == 1
      && launch.cluster_dims[0] == tile->ctas && launch.cluster_dims[1] == 1 && launch.cluster_dims[2] == 1
      && has_map_ranks(launch, tile->maps);
  }
  if (image.kernel_variant == kSM100D8Variant)
  {
    return image.D == kSM100D8HeadDim && image.num_threads == kSM100D8Threads && sm100_d8_cluster(image).ctas != 0
      && has_map_ranks(launch, kSM100D8Maps);
  }
  return false;
}

bool has_valid_launch_abi(embedded::CubinImage const& image)
{
  /* The registry flags exactly the SM100 images, whose clusters stay within the portable 8 CTAs all the same. */
  if (image.cubin.non_portable_cluster_size_allowed != (image.cubin.kernel_sm == kSM100KernelSM))
    return false;
  if (image.cubin.kernel_sm == kSM80KernelSM)
  {
    return equal_c_strings(image.cubin.launch_abi, kSM80LaunchAbi) && !image.sm90.enabled && !image.sm100.enabled
      && image.kernel_variant == 0 && image.cluster_i == 1 && image.cluster_s == 1;
  }
  if (image.cubin.kernel_sm == kSM90KernelSM)
  {
    /* Only the D=8 kernel stores its output through a tensor map, and only it launches clusters. */
    bool const d8 = image.kernel_variant == kSM90D8Variant;
    if (!equal_c_strings(image.cubin.launch_abi, kSM90LaunchAbi) || (!d8 && image.kernel_variant != kSM90D32Variant))
      return false;
    std::uint32_t const cluster_size = image.sm90.cluster_dims[0];
    bool const valid_cluster = d8 ? sm90_d8_cluster(cluster_size).i != 0 : cluster_size == 1;
    return image.cubin.target_sm == kSM90KernelSM && d8 == (image.D == kSM90D8HeadDim)
      && d8 == (image.sm90.output.rank != 0) && image.sm90.enabled && !image.sm100.enabled
      && image.sm90.block_dims[0] == image.num_threads && image.sm90.block_dims[1] == 1 && image.sm90.block_dims[2] == 1
      && valid_cluster && image.sm90.cluster_dims[1] == 1 && image.sm90.cluster_dims[2] == 1;
  }
  if (image.cubin.kernel_sm == kSM100KernelSM)
    return has_valid_sm100_launch(image);
  return false;
}

bool image_matches_axes(
  embedded::CubinImage const& image, std::int32_t target_sm, std::int32_t D, std::int32_t c_m, bool is_bfloat16)
{
  return image.cubin.target_sm == target_sm && cubin_supports_sm(image.cubin, target_sm) && image.D == D
    && image.c_m == c_m && image.is_bfloat16 == is_bfloat16;
}

EmbeddedSelection find_embedded_cubin(
  std::int32_t target_sm, std::int32_t I, std::int32_t J, std::int32_t S, std::int32_t D, std::int32_t c_m, DType dtype)
{
  if (target_sm <= 0 || I <= 0 || J <= 0 || S <= 0 || D <= 0 || c_m <= 0)
    throw std::invalid_argument("pair_weighted_averaging target SM, I, J, S, D, and c_m must be positive");

  bool const is_bfloat16 = dtype_is_bfloat16(dtype);
  double const side = std::sqrt(static_cast<double>(I) * static_cast<double>(J));
  embedded::RegistryView const registry = embedded::registry();
  bool found_n_anchor = false;
  std::int32_t nearest_n_anchor = 0;
  double nearest_n_distance = 0;

  for (std::size_t image_index = 0; image_index < registry.count; ++image_index)
  {
    embedded::CubinImage const& image = registry.images[image_index];
    if (!image_matches_axes(image, target_sm, D, c_m, is_bfloat16))
      continue;
    if (image.aliases == nullptr || image.alias_count == 0)
      throw std::invalid_argument("pair_weighted_averaging CUBIN has no runtime aliases");

    for (std::size_t alias_index = 0; alias_index < image.alias_count; ++alias_index)
    {
      embedded::RuntimeAlias const& alias = image.aliases[alias_index];
      if (alias.n_anchor <= 0 || alias.s_anchor <= 0)
        throw std::invalid_argument("pair_weighted_averaging CUBIN has a non-positive runtime anchor");

      double const distance = anchor_distance(alias.n_anchor, side);
      if (
        !found_n_anchor || distance < nearest_n_distance
        || (distance == nearest_n_distance && alias.n_anchor < nearest_n_anchor))
      {
        found_n_anchor = true;
        nearest_n_anchor = alias.n_anchor;
        nearest_n_distance = distance;
      }
    }
  }

  if (!found_n_anchor)
  {
    throw std::invalid_argument(
      "No embedded pair_weighted_averaging CUBIN for SM" + std::to_string(target_sm) + ", I=" + std::to_string(I)
      + ", J=" + std::to_string(J) + ", S=" + std::to_string(S) + ", D=" + std::to_string(D)
      + ", c_m=" + std::to_string(c_m));
  }

  EmbeddedSelection nearest{};
  double nearest_s_distance = 0;
  for (std::size_t image_index = 0; image_index < registry.count; ++image_index)
  {
    embedded::CubinImage const& image = registry.images[image_index];
    if (!image_matches_axes(image, target_sm, D, c_m, is_bfloat16))
      continue;

    for (std::size_t alias_index = 0; alias_index < image.alias_count; ++alias_index)
    {
      embedded::RuntimeAlias const& alias = image.aliases[alias_index];
      if (alias.n_anchor != nearest_n_anchor)
        continue;

      double const distance = anchor_distance(alias.s_anchor, S);
      if (
        nearest.image == nullptr || distance < nearest_s_distance
        || (distance == nearest_s_distance && alias.s_anchor < nearest.alias->s_anchor))
      {
        nearest = EmbeddedSelection{&image, &alias};
        nearest_s_distance = distance;
      }
    }
  }

  if (nearest.image != nullptr)
    return nearest;
  throw std::logic_error("pair_weighted_averaging nearest N bucket has no S anchor");
}

bool same_cubin_image(EmbeddedCubinImage const& lhs, EmbeddedCubinImage const& rhs)
{
  return lhs.target_sm == rhs.target_sm && lhs.kernel_sm == rhs.kernel_sm
    && lhs.dynamic_smem_bytes == rhs.dynamic_smem_bytes
    && lhs.non_portable_cluster_size_allowed == rhs.non_portable_cluster_size_allowed
    && equal_c_strings(lhs.launch_abi, rhs.launch_abi) && lhs.supported_sms == rhs.supported_sms
    && lhs.supported_sm_count == rhs.supported_sm_count && equal_c_strings(lhs.variant_id, rhs.variant_id)
    && equal_c_strings(lhs.kernel_symbol, rhs.kernel_symbol) && lhs.data == rhs.data && lhs.size == rhs.size;
}

void validate_config(KernelConfig const& config)
{
  if (config.embedded_image == nullptr)
    throw std::invalid_argument("pair_weighted_averaging config has no generated CUBIN image");
  embedded::CubinImage const& image = *config.embedded_image;

  if (config.cubin.data == nullptr || config.cubin.size == 0)
    throw std::invalid_argument("embedded CUBIN image must not be empty");
  if (config.cubin.variant_id == nullptr || config.cubin.variant_id[0] == '\0')
    throw std::invalid_argument("embedded CUBIN variant_id must not be empty");
  if (config.cubin.kernel_symbol == nullptr || config.cubin.kernel_symbol[0] == '\0')
    throw std::invalid_argument("kernel_symbol must not be empty");
  if (!same_cubin_image(config.cubin, image.cubin))
    throw std::invalid_argument("pair_weighted_averaging config and generated CUBIN image disagree");

  if (
    config.target_sm <= 0 || config.target_sm != config.cubin.target_sm
    || !cubin_supports_sm(config.cubin, config.target_sm))
  {
    throw std::invalid_argument(
      "embedded CUBIN does not support configured device SM" + std::to_string(config.target_sm));
  }
  if (image.is_bfloat16 != dtype_is_bfloat16(config.dtype))
  {
    throw std::invalid_argument("pair_weighted_averaging config axes disagree with the generated CUBIN image");
  }
  if (image.H != kH || image.D != config.D || image.c_m != config.c_m || config.D <= 0 || config.c_m <= 0)
    throw std::invalid_argument("pair_weighted_averaging CUBIN has incompatible fixed dimensions");
  if (
    image.tile_i == 0 || image.tile_s == 0 || image.tile_j == 0 || image.num_threads == 0 || image.num_threads > 1024
    || config.cubin.dynamic_smem_bytes == 0)
  {
    throw std::invalid_argument("pair_weighted_averaging CUBIN has invalid launch resources");
  }
  if (!has_valid_launch_abi(image))
  {
    throw std::invalid_argument("pair_weighted_averaging CUBIN has incompatible launch metadata");
  }
  if (config.n_anchor <= 0 || config.s_anchor <= 0 || image.aliases == nullptr || image.alias_count == 0)
    throw std::invalid_argument("pair_weighted_averaging config has invalid runtime anchors");

  bool alias_found = false;
  for (std::size_t index = 0; index < image.alias_count; ++index)
  {
    embedded::RuntimeAlias const& alias = image.aliases[index];
    if (alias.n_anchor == config.n_anchor && alias.s_anchor == config.s_anchor)
    {
      alias_found = true;
      break;
    }
  }
  if (!alias_found)
    throw std::invalid_argument("pair_weighted_averaging config does not name a generated runtime alias");
}

std::int64_t checked_stride_product(std::int64_t stride, std::int32_t extent, char const* name)
{
  if (stride <= 0 || extent <= 0 || stride > std::numeric_limits<std::int64_t>::max() / extent)
    throw std::overflow_error(std::string(name) + " contiguous stride overflows int64");
  return stride * extent;
}

void validate_contiguous_tensor4(Tensor4View const& view, char const* name)
{
  std::int64_t const stride_2 = view.shape[3];
  std::int64_t const stride_1 = checked_stride_product(stride_2, view.shape[2], name);
  std::int64_t const stride_0 = checked_stride_product(stride_1, view.shape[1], name);
  if (view.strides[0] != stride_0 || view.strides[1] != stride_1 || view.strides[2] != stride_2)
    throw std::invalid_argument(std::string(name) + " must be contiguous");
}

void validate_launch(KernelConfig const& config, LaunchParams const& params)
{
  validate_config(config);
  validate_tensor(params.w, "w", 16);
  validate_tensor(params.v, "v", 16);
  validate_tensor(params.g, "g", 16);
  validate_tensor(params.weight, "weight", 16);
  validate_tensor(params.output, "output", 16);

  embedded::CubinImage const& image = *config.embedded_image;
  bool const transposed = is_sm100_d8(image);
  std::int32_t const B = params.w.shape[0];
  std::int32_t const I = params.w.shape[2];
  std::int32_t const Jp = params.w.shape[3];
  std::int32_t const S = params.v.shape[1];
  std::int32_t const N = value_tokens(image, params.v);
  std::int32_t const hidden = image.H * image.D;

  if (params.w.shape[1] != kH)
    throw std::invalid_argument("w shape must be [B, 8, I, Jp]");
  if (
    params.v.shape[0] != B || params.v.shape[transposed ? 2 : 3] != hidden || params.g.shape[0] != B
    || params.g.shape[1] != S || params.g.shape[2] != I || params.g.shape[3] != hidden)
  {
    std::string const h = std::to_string(hidden);
    throw std::invalid_argument(
      (transposed ? "v/g shapes must be [B, S, " + h + ", N]" : "v/g shapes must be [B, S, N, " + h + "]")
      + " and [B, S, I, " + h + "]");
  }
  if (params.weight.shape[0] != image.c_m || params.weight.shape[1] != hidden)
  {
    throw std::invalid_argument(
      "weight shape must be [" + std::to_string(image.c_m) + ", " + std::to_string(hidden) + "]");
  }
  std::int32_t const out_features = output_features(image);
  if (
    params.output.shape[0] != B || params.output.shape[1] != S || params.output.shape[2] != I
    || params.output.shape[3] != out_features)
  {
    throw std::invalid_argument("output shape must be [B, S, I, " + std::to_string(out_features) + "]");
  }

  if (
    Jp < N || Jp % kJAlignment != 0
    || (config.cubin.kernel_sm == kSM80KernelSM
        && ceil_div(static_cast<std::uint64_t>(Jp), image.tile_j)
             != ceil_div(static_cast<std::uint64_t>(N), image.tile_j)))
  {
    throw std::invalid_argument(
      "w.shape[3] must cover v.shape[2], be 8-aligned, and use the same number of TILE_J blocks");
  }

  validate_contiguous_tensor4(params.w, "w");
  validate_contiguous_tensor4(params.output, "output");
  if (params.weight.strides[0] != hidden)
    throw std::invalid_argument("weight must be contiguous row-major");
  /* The tcgen05 kernels read rows of g in 16-byte pieces, some outside any tensor map. */
  if (config.cubin.kernel_sm == kSM100KernelSM)
  {
    for (std::int64_t const stride : params.g.strides)
    {
      if (stride % kJAlignment != 0)
        throw std::invalid_argument("g strides must be multiples of 8 elements for the tcgen05 kernels");
    }
  }
  /* v^T's tensor map walks rows of N contiguous j values, strides[2] (the pitch) apart. TMA needs the pitch and the
   * outer strides in whole 16-byte units, so rows of N % 8 != 0 values are padded; transposed_value_buffer()
   * allocates such a buffer.
   */
  if (transposed)
  {
    std::int64_t const pitch = params.v.strides[2];
    if (
      pitch < N || pitch % kJAlignment != 0 || params.v.strides[1] % kJAlignment != 0
      || params.v.strides[0] % kJAlignment != 0)
    {
      throw std::invalid_argument(
        "v^T [B, S, H*D, N] needs a j pitch (stride 2) of at least N and strides in multiples of 8 elements; got pitch "
        + std::to_string(pitch) + " for N=" + std::to_string(N));
    }
  }
}

void validate_operand_devices(LaunchParams const& params, std::int32_t device)
{
  auto const check = [device](std::int32_t operand_device, char const* name)
  {
    if (operand_device == kUnknownDevice)
    {
      throw std::invalid_argument(
        std::string(name) + " has no CUDA device: it is a host tensor, or its view was built without one");
    }
    if (operand_device != device)
    {
      throw std::invalid_argument(
        std::string(name) + " is on CUDA device " + std::to_string(operand_device) + " but the launch targets device "
        + std::to_string(device));
    }
  };

  check(params.w.device, "w");
  check(params.v.device, "v");
  check(params.g.device, "g");
  check(params.weight.device, "weight");
  check(params.output.device, "output");
}

cute_tensor_s3_d3_t make_w_descriptor(Tensor4View const& view)
{
  cute_tensor_s3_d3_t descriptor{};
  descriptor.data = static_cast<CUdeviceptr>(view.data);
  descriptor.dynamic_shapes[0] = view.shape[0];
  descriptor.dynamic_shapes[1] = view.shape[2];
  descriptor.dynamic_shapes[2] = view.shape[3];
  for (std::size_t index = 0; index < view.strides.size(); ++index)
    descriptor.dynamic_strides[index] = view.strides[index];
  return descriptor;
}

cute_tensor_s0_d0_t make_weight_descriptor(Tensor2View const& view)
{
  cute_tensor_s0_d0_t descriptor{};
  descriptor.data = static_cast<CUdeviceptr>(view.data);
  return descriptor;
}

TmaTensorSource make_w_tma_source(Tensor4View const& w, std::int32_t N)
{
  return TmaTensorSource{
    w.data,
    {
      static_cast<std::uint64_t>(w.shape[0]),
      static_cast<std::uint64_t>(w.shape[1]),
      static_cast<std::uint64_t>(w.shape[2]),
      static_cast<std::uint64_t>(N),
    },
    {
      static_cast<std::uint64_t>(w.strides[0]),
      static_cast<std::uint64_t>(w.strides[1]),
      static_cast<std::uint64_t>(w.strides[2]),
      1,
    },
  };
}

cute_tensor_s3_d3_t make_sm90_output_descriptor(Tensor4View const& view)
{
  cute_tensor_s3_d3_t descriptor{};
  descriptor.data = static_cast<CUdeviceptr>(view.data);
  descriptor.dynamic_shapes[0] = view.shape[2];
  descriptor.dynamic_shapes[1] = view.shape[1];
  descriptor.dynamic_shapes[2] = view.shape[0];
  descriptor.dynamic_strides[0] = view.strides[2];
  descriptor.dynamic_strides[1] = view.strides[1];
  descriptor.dynamic_strides[2] = view.strides[0];
  return descriptor;
}

/* Clusters of `ci` token tiles by `cs` MSA-row tiles that cover the tokens, MSA rows and batch of a D=8 launch. A
 * cluster that overhangs the tokens or MSA rows counts whole.
 */
std::uint64_t
d8_cluster_tiles(embedded::CubinImage const& image, LaunchParams const& params, std::uint32_t ci, std::uint32_t cs)
{
  std::uint64_t const i_clusters = ceil_div(ceil_div(static_cast<std::uint64_t>(params.w.shape[2]), image.tile_i), ci);
  std::uint64_t const s_clusters = ceil_div(ceil_div(static_cast<std::uint64_t>(params.v.shape[1]), image.tile_s), cs);
  return static_cast<std::uint64_t>(checked_mul(
    checked_mul(
      static_cast<std::int64_t>(i_clusters),
      static_cast<std::int64_t>(s_clusters),
      "pair_weighted_averaging D=8 cluster tiles"),
    params.w.shape[0],
    "pair_weighted_averaging D=8 cluster tiles"));
}

cubin_launch_config_t
make_launch_config(embedded::CubinImage const& image, LaunchParams const& params, std::uint32_t smem_bytes)
{
  cubin_launch_config_t launch_config{};
  launch_config.grid_x = checked_u32(
    ceil_div(static_cast<std::uint64_t>(params.w.shape[2]), image.tile_i), "pair_weighted_averaging grid.x");
  launch_config.grid_y = checked_u32(static_cast<std::uint64_t>(params.w.shape[0]), "pair_weighted_averaging grid.y");
  launch_config.grid_z = checked_u32(
    ceil_div(static_cast<std::uint64_t>(params.v.shape[1]), image.tile_s), "pair_weighted_averaging grid.z");
  launch_config.block_x = image.num_threads;
  launch_config.block_y = 1;
  launch_config.block_z = 1;
  launch_config.dynamic_smem_bytes = smem_bytes;
  launch_config.stream = reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream));
  return launch_config;
}

void launch_sm80(
  cubin_kernel_t loaded, KernelConfig const& config, LaunchParams const& params, std::uint32_t smem_bytes)
{
  abi::SM80Params device_params{};
  device_params.w = make_w_descriptor(params.w);
  device_params.v = make_tensor4_s3_d3_descriptor(params.v);
  device_params.g = make_tensor4_s3_d3_descriptor(params.g);
  device_params.weight = make_weight_descriptor(params.weight);
  device_params.output = make_tensor4_s3_d3_descriptor(params.output);

  void* kernel_params[abi::kParameterCount]{};
  abi::pack_sm80_kernel_params(&device_params, kernel_params);
  static_assert(sizeof(kernel_params) / sizeof(kernel_params[0]) == 5);

  cubin_launch_config_t const launch_config = make_launch_config(*config.embedded_image, params, smem_bytes);
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr),
    "launch_cubin_kernel(pair_weighted_averaging_sm80)");
}

/* Every map of the Hopper and the tcgen05 kernels is a CuTe DSL non-executable atom: the encoded map, tagged at byte
 * 8 and, once its tensor spans 2^16 elements, at byte 10 (finalize_sm90_tma_atom). The driver sets the byte-10 bit
 * itself for a tensor that holds 2^16 elements, so the span rule matters only for a sparse strided view, such as a
 * small g that is half of a wider buffer. Captured against the CuTe DSL 4.8.0 host launcher for the pair-weighted
 * averaging tcgen05 kernels as well.
 */
void encode_tma_atom(
  CUtensorMap& atom,
  TmaDescriptorInfo const& info,
  CUtensorMapDataType dtype,
  TmaTensorSource const& source,
  char const* name)
{
  encode_tma_descriptor(atom, info, dtype, source, name);
  finalize_sm90_tma_atom(atom, sm90_tma_operation_tag(source));
}

/* Both Hopper kernels load w, v, g, and Wo through the same tensor maps and coordinate tensors. A multicast load
 * fetches only this CTA's share of the recorded tile.
 */
template <typename DeviceParams>
void encode_sm90_loads(
  DeviceParams& device_params,
  embedded::SM90LaunchInfo const& metadata,
  CUtensorMapDataType dtype,
  LaunchParams const& params,
  std::uint32_t w_multicast = 1,
  std::uint32_t v_multicast = 1)
{
  std::int32_t const B = params.w.shape[0];
  std::int32_t const I = params.w.shape[2];
  std::int32_t const S = params.v.shape[1];
  std::int32_t const N = params.v.shape[2];

  encode_tma_atom(
    device_params.w_tma,
    multicast_tma_descriptor(metadata.w, w_multicast, "w"),
    dtype,
    make_w_tma_source(params.w, N),
    "w");
  encode_tma_atom(
    device_params.v_tma,
    multicast_tma_descriptor(metadata.v, v_multicast, "v"),
    dtype,
    make_tma_bias_source(params.v),
    "v");
  encode_tma_atom(device_params.g_tma, metadata.g, dtype, make_tma_bias_source(params.g), "g");
  encode_tma_atom(device_params.wo_tma, metadata.wo, dtype, make_tma_tensor2_source(params.weight, false), "weight");
  device_params.w_coord = CoordTensorS3{{I, N, B}};
  device_params.v_coord = CoordTensorS3{{N, S, B}};
  device_params.g_coord = CoordTensorS3{{I, S, B}};
}

/* The D=32 kernel stores through an output descriptor on a plain grid. The D=8 kernel stores through a tensor map
 * and launches a 1D grid of whole clusters, multicasting w across the MSA-row tiles and v across the token tiles.
 */
void launch_sm90(
  cubin_kernel_t loaded, KernelConfig const& config, LaunchParams const& params, std::uint32_t smem_bytes)
{
  embedded::CubinImage const& image = *config.embedded_image;
  embedded::SM90LaunchInfo const& metadata = image.sm90;
  if (!metadata.enabled)
    throw std::invalid_argument("pair_weighted_averaging CUBIN is missing its Hopper launch metadata");
  CUtensorMapDataType const dtype = tma_data_type(dtype_is_bfloat16(config.dtype));
  cubin_launch_config_t launch_config = make_launch_config(image, params, smem_bytes);
  auto const launch_with = [&](void** kernel_params)
  {
    check_cuda_driver(
      launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr),
      "launch_cubin_kernel(pair_weighted_averaging_sm90)");
  };

  if (image.kernel_variant == kSM90D32Variant)
  {
    abi::SM90Params device_params{};
    encode_sm90_loads(device_params, metadata, dtype, params);
    device_params.output = make_sm90_output_descriptor(params.output);
    void* kernel_params[abi::kSM90ParameterCount]{};
    abi::pack_sm90_kernel_params(&device_params, kernel_params);
    static_assert(sizeof(kernel_params) / sizeof(kernel_params[0]) == 11);
    launch_with(kernel_params);
    return;
  }

  if (metadata.output.rank == 0)
    throw std::invalid_argument("pair_weighted_averaging D=8 CUBIN is missing its Hopper output tensor map");
  SM90Cluster const cluster = sm90_d8_cluster(metadata.cluster_dims[0]);
  if (cluster.i == 0)
    throw std::invalid_argument("pair_weighted_averaging D=8 CUBIN records an unsupported cluster size");
  abi::SM90D8Params device_params{};
  encode_sm90_loads(device_params, metadata, dtype, params, cluster.s, cluster.i);
  encode_tma_atom(device_params.output_tma, metadata.output, dtype, make_tma_bias_source(params.output), "output");
  device_params.output_coord = device_params.g_coord;
  void* kernel_params[abi::kSM90D8ParameterCount]{};
  abi::pack_sm90_d8_kernel_params(&device_params, kernel_params);
  static_assert(sizeof(kernel_params) / sizeof(kernel_params[0]) == 11);

  /* The kernel maps the grid to tiles itself, and a cluster that overhangs the tokens or MSA rows still launches all
   * of its CTAs.
   */
  std::int64_t const clusters = static_cast<std::int64_t>(d8_cluster_tiles(image, params, cluster.i, cluster.s));
  launch_config.grid_x = checked_u32(
    static_cast<std::uint64_t>(checked_mul(clusters, metadata.cluster_dims[0], "pair_weighted_averaging D=8 grid.x")),
    "pair_weighted_averaging D=8 grid.x");
  launch_config.grid_y = 1;
  launch_config.grid_z = 1;
  /* A one-CTA cluster launches as a plain grid; the cluster attribute alone costs small grids up to 4%. */
  if (metadata.cluster_dims[0] > 1)
  {
    launch_config.cluster_x = metadata.cluster_dims[0];
    launch_config.cluster_y = 1;
    launch_config.cluster_z = 1;
  }
  launch_with(kernel_params);
}

/* The launch operands as the tcgen05 tensor maps view them, outermost first; each recipe names the view dimension
 * behind every TMA dimension (the builder's _SM100_OPERANDS):
 *
 *   w       [B, H, I, J]     make_w_tma_source; J is Jp for D=32, while the D=8 kernel stops at N and lets its loads
 *                            zero-fill the pad
 *   v       [B, S, N, H, D]  head_split_source for the D=32 kernel, one head's d values per box; the D=8 kernel's v^T
 *           [B, S, H, D, N]  as transposed_value_source, one head's d rows of j values per box
 *   wo      [c_m, H*D]       make_tma_tensor2_source
 *   g       [B, S, I, H, D]  head_split_source for the D=32 TMA epilogue; [B, S, I, H*D], make_tma_bias_source, for D=8
 *   output  [B, S, I, c_m]   make_tma_bias_source
 */
TmaTensorSource head_split_source(Tensor4View const& view, std::int32_t head_dim)
{
  return TmaTensorSource{
    view.data,
    {
      static_cast<std::uint64_t>(view.shape[0]),
      static_cast<std::uint64_t>(view.shape[1]),
      static_cast<std::uint64_t>(view.shape[2]),
      static_cast<std::uint64_t>(view.shape[3] / head_dim),
      static_cast<std::uint64_t>(head_dim),
    },
    {
      static_cast<std::uint64_t>(view.strides[0]),
      static_cast<std::uint64_t>(view.strides[1]),
      static_cast<std::uint64_t>(view.strides[2]),
      static_cast<std::uint64_t>(head_dim),
      1,
    },
  };
}

/* v^T [B, S, H*D, N] split into heads: rows of N contiguous j values, strides[2] elements apart, D rows a head. */
TmaTensorSource transposed_value_source(Tensor4View const& view, std::int32_t head_dim)
{
  std::uint64_t const pitch = static_cast<std::uint64_t>(view.strides[2]);
  return TmaTensorSource{
    view.data,
    {
      static_cast<std::uint64_t>(view.shape[0]),
      static_cast<std::uint64_t>(view.shape[1]),
      static_cast<std::uint64_t>(view.shape[2] / head_dim),
      static_cast<std::uint64_t>(head_dim),
      static_cast<std::uint64_t>(view.shape[3]),
    },
    {
      static_cast<std::uint64_t>(view.strides[0]),
      static_cast<std::uint64_t>(view.strides[1]),
      static_cast<std::uint64_t>(head_dim) * pitch,
      pitch,
      1,
    },
  };
}

/* The D=32 kernel launches one CTA per (token tile, batch, MSA-row tile), or a CTA pair per tile for the pair tile,
 * and reorders the tiles on the device. Its TMA epilogue loads g and stores the output through tensor maps; the
 * direct and pair tiles read both as plain tensors, and the split tile writes the gated value without Wo.
 */
void launch_sm100_d32(
  cubin_kernel_t loaded, KernelConfig const& config, LaunchParams const& params, std::uint32_t smem_bytes)
{
  embedded::CubinImage const& image = *config.embedded_image;
  embedded::SM100LaunchInfo const& metadata = image.sm100;
  SM100D32Tile const* tile = sm100_d32_tile(image);
  if (tile == nullptr)
    throw std::invalid_argument("pair_weighted_averaging D=32 CUBIN records an unsupported tile");
  CUtensorMapDataType const dtype = tma_data_type(dtype_is_bfloat16(config.dtype));
  std::int32_t const B = params.w.shape[0];
  std::int32_t const I = params.w.shape[2];
  std::int32_t const Jp = params.w.shape[3];
  std::int32_t const S = params.v.shape[1];
  std::int32_t const N = value_tokens(image, params.v);

  abi::SM100D32Params device_params{};
  encode_tma_atom(device_params.w_tma, metadata.w, dtype, make_w_tma_source(params.w, Jp), "w");
  encode_tma_atom(device_params.v_tma, metadata.v, dtype, head_split_source(params.v, image.D), "v");
  device_params.w_coord = CoordTensorS3{{I, Jp, B}};
  device_params.v_coord = CoordTensorS3{{N, S, B}};
  if (tile->kernel != abi::SM100D32Kernel::kGatedValue)
    encode_tma_atom(device_params.wo_tma, metadata.wo, dtype, make_tma_tensor2_source(params.weight, false), "weight");
  if (tile->kernel == abi::SM100D32Kernel::kTmaEpilogue)
  {
    encode_tma_atom(device_params.g_tma, metadata.g, dtype, head_split_source(params.g, image.D), "g");
    encode_tma_atom(device_params.output_tma, metadata.output, dtype, make_tma_bias_source(params.output), "output");
    device_params.g_coord = CoordTensorS3{{I, S, B}};
    device_params.output_coord = device_params.g_coord;
  }
  else
  {
    device_params.g = make_tensor4_s3_d3_descriptor(params.g);
    device_params.output = make_tensor4_s3_d3_descriptor(params.output);
  }
  device_params.S = S;

  void* kernel_params[abi::kSM100D32ParameterCount]{};
  abi::pack_sm100_d32_kernel_params(&device_params, tile->kernel, kernel_params);

  cubin_launch_config_t launch_config = make_launch_config(image, params, smem_bytes);
  if (tile->ctas > 1)
  {
    launch_config.grid_x = checked_u32(
      static_cast<std::uint64_t>(launch_config.grid_x) * tile->ctas, "pair_weighted_averaging D=32 grid.x");
    launch_config.cluster_x = tile->ctas;
    launch_config.cluster_y = 1;
    launch_config.cluster_z = 1;
    launch_config.cluster_scheduling_policy = metadata.cluster_scheduling_policy;
  }
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr),
    "launch_cubin_kernel(pair_weighted_averaging_sm100)");
}

/* Clusters of `cluster_ctas` CTAs the device keeps resident at once, which caps the persistent D=8 grid as the source
 * launch caps it: one per SM, or one per two SMs for a pair. GPCs do not divide evenly into larger clusters (B200
 * holds 33 of four CTAs and 15 of eight), so past two the driver answers for this kernel, which holds an SM alone. The
 * answer is cached per loaded function.
 */
std::uint64_t resident_d8_clusters(
  cubin_kernel_t loaded, CUcontext context, std::uint32_t cluster_ctas, std::uint32_t block_x, std::uint32_t smem_bytes)
{
  if (cluster_ctas <= 2)
  {
    std::int32_t const multiprocessor_count = cuda_multiprocessor_count_for_context(context);
    if (multiprocessor_count <= 0)
      throw std::invalid_argument("current CUDA device has no active multiprocessors");
    return std::max<std::uint64_t>(1, static_cast<std::uint64_t>(multiprocessor_count) / cluster_ctas);
  }

  static std::mutex mutex;
  static std::unordered_map<CUfunction, std::uint64_t> cache;
  std::lock_guard<std::mutex> const lock(mutex);
  auto const cached = cache.find(loaded.function);
  if (cached != cache.end())
    return cached->second;

  CUlaunchAttribute attribute{};
  attribute.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attribute.value.clusterDim.x = cluster_ctas;
  attribute.value.clusterDim.y = 1;
  attribute.value.clusterDim.z = 1;
  CUlaunchConfig query{};
  query.gridDimX = cluster_ctas;
  query.gridDimY = 1;
  query.gridDimZ = 1;
  query.blockDimX = block_x;
  query.blockDimY = 1;
  query.blockDimZ = 1;
  query.sharedMemBytes = smem_bytes;
  query.attrs = &attribute;
  query.numAttrs = 1;
  int clusters = 0;
  check_cuda_driver(
    cuOccupancyMaxActiveClusters(&clusters, loaded.function, &query),
    "cuOccupancyMaxActiveClusters(pair_weighted_averaging)");
  if (clusters <= 0)
    throw std::runtime_error("pair_weighted_averaging D=8 kernel cannot keep a cluster resident");
  return cache.emplace(loaded.function, static_cast<std::uint64_t>(clusters)).first->second;
}

/* The D=8 kernel is persistent: min(cluster tiles, resident clusters) whole clusters on grid.x. It reads its cluster
 * count from the grid and walks the cluster tiles in its own order, and each cluster tile's result does not depend on
 * which cluster computes it. A cluster multicasts w across its MSA-row tiles and v^T across its token tiles, so each
 * CTA's maps fetch only its share.
 */
void launch_sm100_d8(
  cubin_kernel_t loaded,
  KernelConfig const& config,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  embedded::CubinImage const& image = *config.embedded_image;
  embedded::SM100LaunchInfo const& metadata = image.sm100;
  SM100D8Cluster const cluster = sm100_d8_cluster(image);
  if (cluster.ctas == 0)
    throw std::invalid_argument("pair_weighted_averaging D=8 CUBIN records an unsupported tile or cluster");
  CUtensorMapDataType const dtype = tma_data_type(dtype_is_bfloat16(config.dtype));
  std::int32_t const B = params.w.shape[0];
  std::int32_t const I = params.w.shape[2];
  std::int32_t const S = params.v.shape[1];
  std::int32_t const N = value_tokens(image, params.v);

  abi::SM100D8Params device_params{};
  encode_tma_atom(
    device_params.w_tma,
    multicast_tma_descriptor(metadata.w, cluster.s, "w"),
    dtype,
    make_w_tma_source(params.w, N),
    "w");
  encode_tma_atom(
    device_params.v_tma,
    multicast_tma_descriptor(metadata.v, cluster.i, "v"),
    dtype,
    transposed_value_source(params.v, image.D),
    "v");
  encode_tma_atom(device_params.wo_tma, metadata.wo, dtype, make_tma_tensor2_source(params.weight, false), "weight");
  encode_tma_atom(device_params.g_tma, metadata.g, dtype, make_tma_bias_source(params.g), "g");
  device_params.w_coord = CoordTensorS3{{I, N, B}};
  device_params.v_coord = CoordTensorS3{{N, S, B}};
  device_params.g_coord = CoordTensorS3{{I, S, B}};
  device_params.g = make_tensor4_s3_d3_descriptor(params.g);
  device_params.output = make_tensor4_s3_d3_descriptor(params.output);

  void* kernel_params[abi::kSM100D8ParameterCount]{};
  abi::pack_sm100_d8_kernel_params(&device_params, kernel_params);

  std::uint64_t const clusters = std::min(
    d8_cluster_tiles(image, params, cluster.i, cluster.s),
    resident_d8_clusters(loaded, context, cluster.ctas, image.num_threads, smem_bytes));

  cubin_launch_config_t launch_config = make_launch_config(image, params, smem_bytes);
  launch_config.grid_x = checked_u32(clusters * cluster.ctas, "pair_weighted_averaging D=8 grid.x");
  launch_config.grid_y = 1;
  launch_config.grid_z = 1;
  if (cluster.ctas > 1)
  {
    launch_config.cluster_x = cluster.ctas;
    launch_config.cluster_y = 1;
    launch_config.cluster_z = 1;
    launch_config.cluster_scheduling_policy = metadata.cluster_scheduling_policy;
  }
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr),
    "launch_cubin_kernel(pair_weighted_averaging_sm100)");
}

void launch_sm100(
  cubin_kernel_t loaded,
  KernelConfig const& config,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  embedded::CubinImage const& image = *config.embedded_image;
  if (!image.sm100.enabled)
    throw std::invalid_argument("pair_weighted_averaging CUBIN is missing its Blackwell launch metadata");
  if (image.kernel_variant == kSM100D8Variant)
    launch_sm100_d8(loaded, config, params, context, smem_bytes);
  else
    launch_sm100_d32(loaded, config, params, smem_bytes);
}

bool is_preload_compatible(embedded::CubinImage const& image, std::int32_t device_sm)
{
  return image.cubin.target_sm == device_sm && cubin_supports_sm(image.cubin, device_sm) && has_valid_launch_abi(image)
    && image.cubin.dynamic_smem_bytes > 0 && image.tile_i > 0 && image.tile_s > 0 && image.tile_j > 0
    && image.num_threads > 0 && image.num_threads <= 1024;
}

std::size_t preload_kernels(CUcontext context, std::int32_t device_sm)
{
  return preload_registry_kernels(context, device_sm, embedded::registry(), is_preload_compatible);
}

CubinPreloadRegistration const kPreloader{
  "pair_weighted_averaging",
  &preload_kernels,
};

} // namespace

KernelConfig make_kernel_config(
  std::int32_t target_sm, std::int32_t I, std::int32_t J, std::int32_t S, std::int32_t D, std::int32_t c_m, DType dtype)
{
  EmbeddedSelection const selection = find_embedded_cubin(target_sm, I, J, S, D, c_m, dtype);
  KernelConfig const config{
    target_sm,
    D,
    c_m,
    dtype,
    selection.alias->n_anchor,
    selection.alias->s_anchor,
    selection.image->cubin,
    selection.image,
  };
  validate_config(config);
  return config;
}

std::uint32_t dynamic_smem_bytes(KernelConfig const& config)
{
  validate_config(config);
  return config.cubin.dynamic_smem_bytes;
}

void launch(KernelConfig const& config, LaunchParams const& params)
{
  validate_launch(config, params);
  CUcontext const context = current_cuda_context();
  std::int32_t const device_sm = cuda_sm_for_context(context);
  if (device_sm != config.target_sm || !cubin_supports_sm(config.cubin, device_sm))
  {
    throw std::invalid_argument(
      "pair_weighted_averaging config selects SM" + std::to_string(config.target_sm) + " but current device is SM"
      + std::to_string(device_sm));
  }
  validate_operand_devices(params, cuda_device_for_context(context));

  cubin_kernel_t const loaded = load_embedded_kernel(context, config.cubin);
  if (config.cubin.kernel_sm == kSM100KernelSM)
    launch_sm100(loaded, config, params, context, config.cubin.dynamic_smem_bytes);
  else if (config.cubin.kernel_sm == kSM90KernelSM)
    launch_sm90(loaded, config, params, config.cubin.dynamic_smem_bytes);
  else
    launch_sm80(loaded, config, params, config.cubin.dynamic_smem_bytes);
}

} // namespace bioir::cutedsl::pair_weighted_averaging
