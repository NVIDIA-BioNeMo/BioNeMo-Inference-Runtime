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

#include "outer_product_mean_registry.h"

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace bioir::cutedsl::outer_product_mean
{
namespace
{

constexpr char kSM80LaunchAbi[] = "outer_product_mean_sm80";
constexpr char kSM90LaunchAbi[] = "outer_product_mean_sm90";
constexpr char kSM100LaunchAbi[] = "outer_product_mean_sm100";
/* Rows of a the SM100 CTA owns: tile_i tokens of cc channels each. */
constexpr std::int32_t kSM100CtaRows = 128;
constexpr std::uint32_t kSM100PairCtas = 2;

bool has_launch_abi(EmbeddedCubinImage const& cubin, std::int32_t kernel_sm, char const* launch_abi)
{
  return cubin.kernel_sm == kernel_sm && cubin.launch_abi != nullptr && std::strcmp(cubin.launch_abi, launch_abi) == 0;
}

void validate_launch(KernelConfig const& config, LaunchParams const& params)
{
  if (config.cubin.data == nullptr || config.cubin.size == 0)
    throw std::invalid_argument("embedded CUBIN image must not be empty");
  if (config.cubin.variant_id == nullptr || config.cubin.variant_id[0] == '\0')
    throw std::invalid_argument("embedded CUBIN variant_id must not be empty");
  if (config.cubin.kernel_symbol == nullptr || config.cubin.kernel_symbol[0] == '\0')
    throw std::invalid_argument("kernel_symbol must not be empty");
  bool const compatible = config.spec.is_sm100 ? has_launch_abi(config.cubin, 100, kSM100LaunchAbi)
    : config.spec.is_sm90                      ? has_launch_abi(config.cubin, 90, kSM90LaunchAbi)
                                               : has_launch_abi(config.cubin, 80, kSM80LaunchAbi);
  if (!compatible)
    throw std::invalid_argument("outer-product-mean CUBIN has an incompatible launch ABI");
  if (!cubin_supports_sm(config.cubin, config.spec.target_sm))
    throw std::invalid_argument("outer-product-mean CUBIN does not support its configured target SM");
  if (config.spec.num_threads <= 0 || config.spec.tile_i <= 0 || config.spec.tile_j <= 0)
    throw std::invalid_argument("outer-product-mean spec has a non-positive tile or thread count");
  // The SM100 kernel bakes its raster band in; zero walks whole rows of j tiles.
  if (config.spec.is_sm100 ? config.spec.raster_factor < 0 : config.spec.raster_factor <= 0)
    throw std::invalid_argument("outer-product-mean spec has an invalid rasterization factor");

  validate_tensor(params.a, "a", 16);
  validate_tensor(params.b, "b", 16);
  validate_tensor(params.num_mask, "num_mask", 4);
  validate_tensor(params.weight, "weight", 16);
  if (config.has_bias)
    validate_tensor(params.bias, "bias", 16);
  validate_tensor(params.output, "output", 16);
  bool const relaid_a = config.spec.is_sm100;
  if (!relaid_a)
    validate_static_tail(params.a, kChannelsC, "a");
  validate_static_tail(params.b, kChannelsD, "b");
  validate_static_tail(params.output, config.spec.c_z, "output");

  /* C, D and C_z stay in host metadata for validation, then the launcher
   * projects only the dynamic leading extents into the device descriptors.
   */
  std::int32_t const batch = params.a.shape[0];
  std::int32_t const sequence = relaid_a ? params.a.shape[3] : params.a.shape[1];
  std::int32_t const rows = relaid_a ? params.num_mask.shape[1] : params.a.shape[2];
  std::int32_t const columns = params.b.shape[2];

  if (params.b.shape[0] != batch || params.b.shape[1] != sequence)
    throw std::invalid_argument("a and b must share their batch and sequence extents");
  if (params.num_mask.shape[0] != batch || params.num_mask.shape[1] != rows || params.num_mask.shape[2] != columns)
    throw std::invalid_argument("num_mask must be [B, I, J]");
  if (params.output.shape[0] != batch || params.output.shape[1] != rows || params.output.shape[2] != columns)
    throw std::invalid_argument("output must be [B, I, J, C_z]");
  if (params.weight.shape[0] != config.spec.c_z || params.weight.shape[1] != kChannelsC * kChannelsD)
  {
    throw std::invalid_argument(
      "weight must be [" + std::to_string(config.spec.c_z) + ", " + std::to_string(kChannelsC * kChannelsD)
      + "]: the kernel compiles C, D and C_z as constants");
  }
  if (config.has_bias && params.bias.shape[0] != config.spec.c_z)
    throw std::invalid_argument("bias must be [C_z]");
  if (config.spec.is_sm90)
  {
    /* TMA reads a and b as dense (i, c) and (j, d) rows of each s. */
    if (params.a.strides[2] != kChannelsC || params.b.strides[2] != kChannelsD)
      throw std::invalid_argument("the SM90 kernel needs a and b dense over their last two dimensions");
    if (config.spec.tile_s <= 0 || config.spec.ptile <= 0 || (config.spec.csplit != 1 && config.spec.csplit != 2))
      throw std::invalid_argument("outer-product-mean SM90 spec has an invalid tile");
  }
  if (relaid_a)
  {
    std::int32_t const tile_i = config.spec.tile_i;
    std::int32_t const channels_per_chunk = config.spec.tile_j;
    if (tile_i * channels_per_chunk != kSM100CtaRows || kChannelsC % channels_per_chunk != 0)
      throw std::invalid_argument("outer-product-mean SM100 spec has an invalid tile");
    std::uint64_t const relaid_rows = ceil_div(static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(tile_i))
      * static_cast<std::uint64_t>(kSM100CtaRows);
    if (
      params.a.shape[1] != kChannelsC / channels_per_chunk
      || static_cast<std::uint64_t>(params.a.shape[2]) != relaid_rows)
      throw std::invalid_argument("the SM100 kernel needs a relaid as [B, C / cc, ceil(I / tile_i) * 128, S]");
    /* TMA reads b as dense (j, d) rows of each s and stores dense (j, z) output rows. */
    if (params.b.strides[2] != kChannelsD || params.output.strides[2] != config.spec.c_z)
      throw std::invalid_argument("the SM100 kernel needs b and output dense over their last two dimensions");
  }
}

void validate_operand_devices(KernelConfig const& config, LaunchParams const& params, std::int32_t device)
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

  check(params.a.device, "a");
  check(params.b.device, "b");
  check(params.num_mask.device, "num_mask");
  check(params.weight.device, "weight");
  if (config.has_bias)
    check(params.bias.device, "bias");
  check(params.output.device, "output");
}

/* Reproduces the CuTeDSL host launcher's grid arithmetic exactly:
 *
 *   grid = (ceil_div(J, TILE_J) * f, ceil_div(ceil_div(I, TILE_I), f), B)
 *
 * where f is the L2 rasterization factor, decoded back inside the kernel.
 */
cubin_launch_config_t
make_launch_config(KernelSpec const& spec, LaunchParams const& params, std::uint32_t smem_bytes, CUstream stream)
{
  std::uint64_t const rows = static_cast<std::uint64_t>(params.a.shape[2]);
  std::uint64_t const columns = static_cast<std::uint64_t>(params.b.shape[2]);
  std::uint64_t const raster = static_cast<std::uint64_t>(spec.raster_factor);
  std::uint64_t const column_tiles = ceil_div(columns, static_cast<std::uint64_t>(spec.tile_j));
  std::uint64_t const row_tiles = ceil_div(rows, static_cast<std::uint64_t>(spec.tile_i));

  cubin_launch_config_t config = {0};
  config.grid_x = checked_u32(column_tiles * raster, "outer-product-mean grid.x");
  config.grid_y = checked_u32(ceil_div(row_tiles, raster), "outer-product-mean grid.y");
  config.grid_z = checked_u32(static_cast<std::uint64_t>(params.a.shape[0]), "outer-product-mean grid.z");
  config.block_x = static_cast<std::uint32_t>(spec.num_threads);
  config.block_y = 1;
  config.block_z = 1;
  config.dynamic_smem_bytes = smem_bytes;
  config.stream = stream;
  /* Cluster dimensions stay zero: this kernel launches through the plain
   * cuLaunchKernel path on every target SM, Blackwell included.
   */
  return config;
}

void launch_sm80(
  cubin_kernel_t loaded,
  KernelConfig const& config,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  validate_operand_devices(config, params, cuda_device_for_context(context));

  abi::SM80Params device_params{};
  device_params.a = make_tensor3_descriptor(params.a);
  device_params.b = make_tensor3_descriptor(params.b);
  device_params.num_mask = make_tensor3_descriptor(params.num_mask);
  device_params.weight.data = static_cast<CUdeviceptr>(params.weight.data);
  if (config.has_bias)
    device_params.bias.data = static_cast<CUdeviceptr>(params.bias.data);
  device_params.output = make_tensor3_descriptor(params.output);

  void* kernel_params[abi::kSM80MaxParameterCount];
  std::size_t const count = abi::pack_sm80_kernel_params(&device_params, kernel_params, config.has_bias);
  std::size_t const expected = config.has_bias ? abi::kSM80ParameterCountWithBias : abi::kSM80ParameterCountNoBias;
  if (count != expected)
    throw std::invalid_argument("outer-product-mean packed an unexpected number of kernel parameters");

  cubin_launch_config_t const launch_config = make_launch_config(
    config.spec, params, smem_bytes, reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream)));

  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr),
    "launch_cubin_kernel(outer_product_mean_sm80)");
}

/* TMA source views, outermost dimension first, matching the recipes the builder records. */
TmaTensorSource a_source(Tensor4View const& a, std::int32_t csplit, std::int32_t c_half)
{
  if (csplit == 1) // contiguous (i, c) rows fold into one dimension
  {
    return TmaTensorSource{
      a.data,
      {static_cast<std::uint64_t>(a.shape[0]),
       static_cast<std::uint64_t>(a.shape[1]),
       static_cast<std::uint64_t>(a.shape[2]) * kChannelsC,
       0},
      {static_cast<std::uint64_t>(a.strides[0]), static_cast<std::uint64_t>(a.strides[1]), 1, 0},
    };
  }
  std::int32_t const c_pass = kChannelsC / csplit;
  return TmaTensorSource{
    a.data + static_cast<std::uint64_t>(c_half) * static_cast<std::uint64_t>(c_pass) * 2,
    {static_cast<std::uint64_t>(a.shape[0]),
     static_cast<std::uint64_t>(a.shape[1]),
     static_cast<std::uint64_t>(a.shape[2]),
     static_cast<std::uint64_t>(c_pass)},
    {static_cast<std::uint64_t>(a.strides[0]),
     static_cast<std::uint64_t>(a.strides[1]),
     static_cast<std::uint64_t>(a.strides[2]),
     1},
  };
}

TmaTensorSource b_source(Tensor4View const& b)
{
  return TmaTensorSource{
    b.data,
    {static_cast<std::uint64_t>(b.shape[0]),
     static_cast<std::uint64_t>(b.shape[1]),
     static_cast<std::uint64_t>(b.shape[2]) * kChannelsD,
     0},
    {static_cast<std::uint64_t>(b.strides[0]), static_cast<std::uint64_t>(b.strides[1]), 1, 0},
  };
}

std::int32_t checked_i32(std::uint64_t value, char const* name)
{
  if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::overflow_error(std::string(name) + " does not fit int32");
  return static_cast<std::int32_t>(value);
}

/* Reproduces the CuTeDSL host launcher's arithmetic: one CTA per `ptile` (i, j)
 * tiles, else a (i tiles, j tiles) grid, with the batch on grid.z.
 */
void launch_sm90(
  cubin_kernel_t loaded,
  KernelConfig const& config,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  validate_operand_devices(config, params, cuda_device_for_context(context));
  embedded::CubinImage const& image = *config.embedded_image;
  embedded::SM90LaunchInfo const& metadata = image.sm90;
  if (!metadata.is_sm90)
    throw std::invalid_argument("outer-product-mean CUBIN is missing its Hopper launch metadata");
  KernelSpec const& spec = config.spec;
  CUtensorMapDataType const dtype = tma_data_type(true);

  abi::SM90Params device_params{};
  auto const encode
    = [dtype](CUtensorMap& atom, TmaDescriptorInfo const& info, TmaTensorSource const& source, char const* name)
  {
    encode_tma_descriptor(atom, info, dtype, source, name);
    finalize_sm90_tma_atom(atom, sm90_tma_operation_tag(source));
  };
  encode(device_params.a_tma, metadata.a, a_source(params.a, spec.csplit, 0), "a");
  encode(device_params.a2_tma, metadata.a2, a_source(params.a, spec.csplit, spec.csplit > 1 ? 1 : 0), "a2");
  encode(device_params.b_tma, metadata.b, b_source(params.b), "b");
  encode(device_params.w_tma, metadata.w, make_tma_tensor2_source(params.weight, false), "w");

  std::int32_t const batch = params.a.shape[0];
  std::int32_t const sequence = params.a.shape[1];
  std::int32_t const rows = params.a.shape[2];
  std::int32_t const columns = params.b.shape[2];
  device_params.a_coord = CoordTensorS3{{rows, sequence, batch}};
  device_params.a2_coord = device_params.a_coord;
  device_params.b_coord
    = CoordTensorS3{{checked_i32(static_cast<std::uint64_t>(columns) * kChannelsD, "J * D"), sequence, batch}};
  device_params.num_mask = make_tensor3_descriptor(params.num_mask);
  if (config.has_bias)
    device_params.bias.data = static_cast<CUdeviceptr>(params.bias.data);
  device_params.output = make_tensor3_descriptor(params.output);

  std::uint64_t const i_tiles = ceil_div(static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(spec.tile_i));
  std::uint64_t const j_tiles = ceil_div(static_cast<std::uint64_t>(columns), static_cast<std::uint64_t>(spec.tile_j));
  std::uint64_t const k_tiles = ceil_div(static_cast<std::uint64_t>(sequence), static_cast<std::uint64_t>(spec.tile_s));
  std::int32_t const scalars[5] = {
    checked_i32(i_tiles, "i tiles"),
    checked_i32(j_tiles, "j tiles"),
    checked_i32(k_tiles, "k tiles"),
    rows,
    columns,
  };
  std::copy(std::begin(scalars), std::end(scalars), std::begin(device_params.scalars));

  void* kernel_params[abi::kSM90MaxParameterCount]{};
  if (
    abi::pack_sm90_kernel_params(&device_params, kernel_params, config.has_bias)
    != abi::sm90_parameter_count(config.has_bias))
    throw std::logic_error("outer-product-mean SM90 parameter packer produced the wrong ABI count");

  cubin_launch_config_t launch_config{};
  if (spec.ptile > 1)
  {
    launch_config.grid_x
      = checked_u32(ceil_div(i_tiles * j_tiles, static_cast<std::uint64_t>(spec.ptile)), "outer-product-mean grid.x");
    launch_config.grid_y = 1;
  }
  else
  {
    launch_config.grid_x = checked_u32(i_tiles, "outer-product-mean grid.x");
    launch_config.grid_y = checked_u32(j_tiles, "outer-product-mean grid.y");
  }
  launch_config.grid_z = checked_u32(static_cast<std::uint64_t>(batch), "outer-product-mean grid.z");
  launch_config.block_x = metadata.block_dims[0];
  launch_config.block_y = metadata.block_dims[1];
  launch_config.block_z = metadata.block_dims[2];
  launch_config.dynamic_smem_bytes = smem_bytes;
  launch_config.stream = reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream));
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr),
    "launch_cubin_kernel(outer_product_mean_sm90)");
}

/* Clusters the device keeps resident, which caps the persistent grid. The CuTe
 * host launcher allots one per two SMs to a lone CTA pair; a larger cluster
 * takes CuTe's occupancy query for that many one-CTA-per-SM blocks, asked here
 * of this kernel, which also holds an SM alone. The query is cached per loaded
 * function, as CuTe caches it per process.
 */
std::uint64_t max_resident_clusters(
  cubin_kernel_t loaded, CUcontext context, embedded::SM100LaunchInfo const& metadata, std::uint32_t smem_bytes)
{
  std::uint32_t const cluster_ctas = metadata.cluster_dims[0] * metadata.cluster_dims[1] * metadata.cluster_dims[2];
  if (cluster_ctas == kSM100PairCtas)
  {
    std::int32_t const multiprocessor_count = cuda_multiprocessor_count_for_context(context);
    if (multiprocessor_count <= 0)
      throw std::invalid_argument("current CUDA device has no active multiprocessors");
    return static_cast<std::uint64_t>(multiprocessor_count) / kSM100PairCtas;
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
  query.blockDimX = metadata.block_dims[0];
  query.blockDimY = metadata.block_dims[1];
  query.blockDimZ = metadata.block_dims[2];
  query.sharedMemBytes = smem_bytes;
  query.attrs = &attribute;
  query.numAttrs = 1;
  int clusters = 0;
  check_cuda_driver(
    cuOccupancyMaxActiveClusters(&clusters, loaded.function, &query),
    "cuOccupancyMaxActiveClusters(outer_product_mean)");
  if (clusters <= 0)
    throw std::runtime_error("outer-product-mean SM100 kernel cannot keep a cluster resident");
  return cache.emplace(loaded.function, static_cast<std::uint64_t>(clusters)).first->second;
}

/* Reproduces the CuTe host launcher's persistent grid: CTA pairs on grid.x,
 * the cluster's pairs on grid.y, and min(tiles, resident clusters) clusters on
 * grid.z, where a tile is one batch's 2 * tile_i rows by tile_j * pairs columns.
 */
void launch_sm100(
  cubin_kernel_t loaded,
  KernelConfig const& config,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  validate_operand_devices(config, params, cuda_device_for_context(context));
  embedded::CubinImage const& image = *config.embedded_image;
  embedded::SM100LaunchInfo const& metadata = image.sm100;
  if (!metadata.is_sm100)
    throw std::invalid_argument("outer-product-mean CUBIN is missing its Blackwell launch metadata");
  if (metadata.cluster_dims[0] != kSM100PairCtas || metadata.cluster_dims[2] != 1 || metadata.cluster_dims[1] == 0)
    throw std::invalid_argument("outer-product-mean SM100 CUBIN has an invalid cluster shape");
  KernelSpec const& spec = config.spec;
  std::uint32_t const pairs = metadata.cluster_dims[1];
  CUtensorMapDataType const dtype = tma_data_type(config.dtype == DType::kBFloat16);

  abi::SM100Params device_params{};
  auto const encode = [dtype](
                        CUtensorMap& atom,
                        TmaDescriptorInfo const& info,
                        std::uint32_t multicast,
                        TmaTensorSource const& source,
                        char const* name)
  {
    encode_tma_descriptor(atom, multicast_tma_descriptor(info, multicast, name), dtype, source, name);
    finalize_sm100_tma_atom(atom);
  };
  encode(device_params.a_tma, metadata.a, pairs, make_tma_bias_source(params.a), "a");
  encode(device_params.b_tma, metadata.b, 1, b_source(params.b), "b");
  encode(device_params.w_tma, metadata.w, pairs, make_tma_tensor2_source(params.weight, false), "w");
  encode(device_params.o_tma, metadata.output, 1, make_tma_bias_source(params.output), "output");

  std::int32_t const batch = params.a.shape[0];
  std::int32_t const sequence = params.a.shape[3];
  std::int32_t const rows = params.output.shape[1];
  std::int32_t const columns = params.output.shape[2];
  device_params.a_coord = CoordTensorS3{{params.a.shape[2], sequence, batch}};
  device_params.b_coord
    = CoordTensorS3{{checked_i32(static_cast<std::uint64_t>(columns) * kChannelsD, "J * D"), sequence, batch}};
  device_params.o_coord = CoordTensorS3{{rows, columns, batch}};
  device_params.num_mask = make_tensor3_descriptor(params.num_mask);
  if (config.has_bias)
    device_params.bias.data = static_cast<CUdeviceptr>(params.bias.data);
  device_params.output = make_tensor3_descriptor(params.output);

  void* kernel_params[abi::kSM100MaxParameterCount]{};
  if (
    abi::pack_sm100_kernel_params(&device_params, kernel_params, config.has_bias)
    != abi::sm100_parameter_count(config.has_bias))
    throw std::logic_error("outer-product-mean SM100 parameter packer produced the wrong ABI count");

  std::uint64_t const i_tiles = ceil_div(
    static_cast<std::uint64_t>(rows), std::uint64_t{kSM100PairCtas} * static_cast<std::uint64_t>(spec.tile_i));
  std::uint64_t const j_tiles
    = ceil_div(static_cast<std::uint64_t>(columns), static_cast<std::uint64_t>(spec.tile_j) * pairs);
  std::uint64_t const tiles = static_cast<std::uint64_t>(batch) * i_tiles * j_tiles;

  cubin_launch_config_t launch_config{};
  launch_config.grid_x = kSM100PairCtas;
  launch_config.grid_y = pairs;
  // Even waves, as the kernel's own launch: as many clusters as an even split of the tiles over the persistent
  // waves needs, so a ragged last wave does not hold every SM (and the power-capped clock) for the waves before it.
  std::uint64_t const resident
    = std::max<std::uint64_t>(std::min(tiles, max_resident_clusters(loaded, context, metadata, smem_bytes)), 1);
  std::uint64_t const waves = std::max<std::uint64_t>(ceil_div(tiles, resident), 1);
  launch_config.grid_z = checked_u32(ceil_div(tiles, waves), "outer-product-mean grid.z");
  launch_config.block_x = metadata.block_dims[0];
  launch_config.block_y = metadata.block_dims[1];
  launch_config.block_z = metadata.block_dims[2];
  launch_config.cluster_x = metadata.cluster_dims[0];
  launch_config.cluster_y = metadata.cluster_dims[1];
  launch_config.cluster_z = metadata.cluster_dims[2];
  launch_config.cluster_scheduling_policy = metadata.cluster_scheduling_policy;
  launch_config.dynamic_smem_bytes = smem_bytes;
  launch_config.stream = reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream));
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr),
    "launch_cubin_kernel(outer_product_mean_sm100)");
}

embedded::CubinImage const&
find_embedded_cubin(std::int32_t target_sm, DType dtype, bool has_bias, bool norm_before, char const* config_identity)
{
  if (config_identity == nullptr || config_identity[0] == '\0')
    throw std::invalid_argument("outer-product-mean config identity must not be empty");

  bool const is_bfloat16 = dtype == DType::kBFloat16;
  embedded::RegistryView const registry = embedded::registry();
  for (std::size_t index = 0; index < registry.count; ++index)
  {
    embedded::CubinImage const& image = registry.images[index];
    if (
      cubin_supports_sm(image.cubin, target_sm) && image.is_bfloat16 == is_bfloat16 && image.has_bias == has_bias
      && image.norm_before == norm_before && image.config_identity != nullptr
      && std::strcmp(image.config_identity, config_identity) == 0)
    {
      return image;
    }
  }

  // A distinct type: a library without the image is not a malformed request.
  throw std::out_of_range(
    "No embedded outer-product-mean CUBIN for SM" + std::to_string(target_sm) + ", bf16=" + std::to_string(is_bfloat16)
    + ", bias=" + std::to_string(has_bias) + ", norm_before=" + std::to_string(norm_before)
    + ", config=" + std::string(config_identity));
}

std::size_t preload_kernels(CUcontext context, std::int32_t device_sm)
{
  return preload_registry_kernels(context, device_sm, embedded::registry());
}

CubinPreloadRegistration const kPreloader{
  "outer_product_mean",
  &preload_kernels,
};

} // namespace

KernelConfig
make_kernel_config(std::int32_t target_sm, DType dtype, bool has_bias, bool norm_before, char const* config_identity)
{
  embedded::CubinImage const& image = find_embedded_cubin(target_sm, dtype, has_bias, norm_before, config_identity);
  KernelSpec const spec{
    target_sm,
    image.tile_i,
    image.tile_j,
    image.raster_factor,
    image.num_threads,
    image.c_z,
    image.tile_s,
    image.ptile,
    image.csplit,
    has_launch_abi(image.cubin, 90, kSM90LaunchAbi),
    has_launch_abi(image.cubin, 100, kSM100LaunchAbi),
  };
  return KernelConfig{
    spec,
    dtype,
    has_bias,
    norm_before,
    image.cubin,
    &image,
  };
}

void launch(KernelConfig const& config, LaunchParams const& params)
{
  validate_launch(config, params);
  CUcontext const context = current_cuda_context();
  std::int32_t const device_sm = cuda_sm_for_context(context);
  if (device_sm != config.spec.target_sm || !cubin_supports_sm(config.cubin, device_sm))
  {
    throw std::invalid_argument(
      "outer-product-mean config selects SM" + std::to_string(config.spec.target_sm) + " but current device is SM"
      + std::to_string(device_sm));
  }

  cubin_kernel_t const loaded = load_embedded_kernel(context, config.cubin);
  if (config.spec.is_sm100)
    launch_sm100(loaded, config, params, context, dynamic_smem_bytes(config));
  else if (config.spec.is_sm90)
    launch_sm90(loaded, config, params, context, dynamic_smem_bytes(config));
  else
    launch_sm80(loaded, config, params, context, dynamic_smem_bytes(config));
}

} // namespace bioir::cutedsl::outer_product_mean
