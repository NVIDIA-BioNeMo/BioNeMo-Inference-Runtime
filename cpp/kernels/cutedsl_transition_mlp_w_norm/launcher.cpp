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

#include "transition_mlp_w_norm_registry.h"

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace bioir::cutedsl::transition_mlp_w_norm
{
namespace
{

constexpr char kSM90LaunchAbi[] = "transition_mlp_w_norm_sm90_v3";
constexpr std::int32_t kSM90KernelSM = 90;
/* The kernel's per-item tile table holds this many pair maps. */
constexpr std::int64_t kMaxBatch = 64;
/* Rows are int32 in the kernel's tile arithmetic, with a tile of headroom: rows < 2^31 - 256. */
constexpr std::int64_t kMaxRows = (std::int64_t{1} << 31) - 257;
/* x, output, mask and the image are bf16; vec is fp32 and n_valid int32. */
constexpr std::uint64_t kBf16Bytes = 2;
constexpr std::uint64_t kWordBytes = 4;
/* A warpgroup's threads, and the largest split: the portable cluster size. */
constexpr std::uint32_t kWarpgroupThreads = 128;
constexpr std::int32_t kMaxSplit = 8;

/* The warpgroups per CTA. */
std::int32_t num_wg(KernelSpec const& spec)
{
  return static_cast<std::int32_t>(spec.num_threads / kWarpgroupThreads);
}

/* The bit of (tile_wgs, split) in KernelSpec::layouts, or -1 for a pair no kernel takes. */
std::int32_t layout_bit(std::int32_t tile_wgs, std::int32_t split)
{
  std::int32_t const index = tile_wgs == 1 ? 0 : tile_wgs == 2 ? 1 : tile_wgs == 4 ? 2 : -1;
  if (index < 0 || split < 1 || split > kMaxSplit)
    return -1;
  return index * kMaxSplit + split - 1;
}

/* Bare device pointers require explicit cross-device validation. */
void check_device(std::int32_t operand_device, std::int32_t device, char const* name)
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
}

/* A flat operand holding exactly ``elements`` values. */
void validate_flat(
  FlatTensorView const& view, std::int64_t elements, std::uint64_t alignment, std::int32_t device, char const* name)
{
  validate_flat_tensor(view, name, alignment);
  check_device(view.device, device, name);
  if (view.extent != elements)
  {
    throw std::invalid_argument(
      std::string(name) + " must hold " + std::to_string(elements) + " elements, got " + std::to_string(view.extent));
  }
}

/* The W1 row blocks of an activation: the ReLU input, or the SwiGLU value and gate. */
std::int64_t w1_blocks(Activation activation)
{
  return activation == Activation::kSiLUGate ? 2 : 1;
}

/* One operand's bytes, for the aliasing checks. */
struct Span
{
  std::uint64_t begin;
  std::uint64_t bytes;
  char const* name;
};

bool overlaps(Span const& lhs, Span const& rhs)
{
  return lhs.begin < rhs.begin + rhs.bytes && rhs.begin < lhs.begin + lhs.bytes;
}

void validate_image(KernelConfig const& config)
{
  if (config.cubin.data == nullptr || config.cubin.size == 0)
    throw std::invalid_argument("embedded CUBIN image must not be empty");
  if (config.cubin.variant_id == nullptr || config.cubin.variant_id[0] == '\0')
    throw std::invalid_argument("embedded CUBIN variant_id must not be empty");
  if (config.cubin.kernel_symbol == nullptr || config.cubin.kernel_symbol[0] == '\0')
    throw std::invalid_argument("kernel_symbol must not be empty");
  if (
    config.cubin.kernel_sm != kSM90KernelSM || config.cubin.launch_abi == nullptr
    || std::strcmp(config.cubin.launch_abi, kSM90LaunchAbi) != 0)
    throw std::invalid_argument("fused transition CUBIN has an incompatible launch ABI");
  if (config.cubin.non_portable_cluster_size_allowed)
    throw std::invalid_argument("fused transition CUBIN must launch with portable cluster sizes");
  if (config.embedded_image == nullptr)
    throw std::invalid_argument("fused transition config has no generated CUBIN image");
  if (!cubin_supports_sm(config.cubin, config.spec.target_sm))
    throw std::invalid_argument(
      "embedded CUBIN does not support configured device SM" + std::to_string(config.spec.target_sm));
  KernelSpec const& spec = config.spec;
  if (
    spec.width <= 0 || spec.hidden <= 0 || spec.num_threads == 0 || spec.num_threads > 1024
    || spec.num_threads % kWarpgroupThreads != 0 || spec.tile_m == 0 || spec.tile_m % num_wg(spec) != 0
    || spec.ctas_per_sm == 0 || !supports_layout(spec, 1, 0))
    throw std::invalid_argument("fused transition CUBIN has invalid launch geometry");
}

void launch_sm90(
  cubin_kernel_t loaded,
  KernelConfig const& config,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  std::int32_t const device = cuda_device_for_context(context);
  KernelSpec const& spec = config.spec;
  std::int64_t const width = spec.width;
  std::int64_t const hidden = spec.hidden;
  std::int64_t const blocks = w1_blocks(spec.activation);
  std::int64_t const rows = params.rows;
  std::int32_t const split = params.split;
  std::int32_t const tile_wgs = params.tile_wgs == 0 ? num_wg(spec) : params.tile_wgs;
  if (!supports_layout(spec, split, tile_wgs))
  {
    throw std::invalid_argument(
      "fused transition image takes no layout split=" + std::to_string(split)
      + ", tile_wgs=" + std::to_string(tile_wgs));
  }
  if (rows <= 0 || rows > kMaxRows)
  {
    throw std::invalid_argument(
      "fused transition needs between 1 and " + std::to_string(kMaxRows) + " rows a launch; got "
      + std::to_string(rows));
  }
  bool const pair_map = params.n_valid.data != 0;
  if (pair_map)
  {
    std::int64_t const n = params.n;
    std::int64_t const n_batch = params.n_batch;
    if (n <= 0 || n_batch <= 0 || n_batch > kMaxBatch)
    {
      throw std::invalid_argument(
        "fused transition with n_valid needs positive n and between 1 and " + std::to_string(kMaxBatch) + " pair maps");
    }
    if (rows != checked_mul(n_batch, checked_mul(n, n, "fused transition plane"), "fused transition rows"))
      throw std::invalid_argument("fused transition with n_valid needs rows == n_batch * n * n");
    validate_tensor(params.n_valid, "n_valid", kWordBytes);
    check_device(params.n_valid.device, device, "n_valid");
    if (params.n_valid.shape[0] < n_batch)
      throw std::invalid_argument("n_valid needs one length per pair map");
  }

  std::int64_t const operand_elements = checked_mul(rows, width, "fused transition x");
  validate_flat(params.x, operand_elements, 16, device, "x");
  validate_flat(params.output, operand_elements, 16, device, "output");
  bool const has_mask = params.mask.data != 0;
  if (has_mask)
    validate_flat(params.mask, rows, kBf16Bytes, device, "mask");
  std::int64_t const image_elements
    = checked_mul(checked_mul(blocks + 1, width, "fused transition image"), hidden, "fused transition image");
  validate_flat(params.image, image_elements, 16, device, "image");
  validate_tensor(params.vec, "vec", 16);
  check_device(params.vec.device, device, "vec");
  if (params.vec.shape[0] != blocks * hidden + width)
    throw std::invalid_argument("vec must hold one fp32 bias per W1 row and per output channel");

  /* Each row is read before the thread that writes it stores it, so output may be x itself. A partial overlap would
   * let one tile write rows another has yet to read, and every CTA reads the mask, the lengths and the weights.
   */
  std::uint64_t const operand_bytes = static_cast<std::uint64_t>(operand_elements) * kBf16Bytes;
  Span const output{params.output.data, operand_bytes, "output"};
  Span const x{params.x.data, operand_bytes, "x"};
  if (output.begin != x.begin && overlaps(output, x))
    throw std::invalid_argument("fused transition output must be x itself or not overlap it");
  Span const read_only[] = {
    {params.mask.data, has_mask ? static_cast<std::uint64_t>(rows) * kBf16Bytes : 0, "mask"},
    {params.image.data, static_cast<std::uint64_t>(image_elements) * kBf16Bytes, "image"},
    {params.vec.data, static_cast<std::uint64_t>(params.vec.shape[0]) * kWordBytes, "vec"},
    {params.n_valid.data, pair_map ? static_cast<std::uint64_t>(params.n_valid.shape[0]) * kWordBytes : 0, "n_valid"},
  };
  for (Span const& operand : read_only)
  {
    if (overlaps(output, operand))
      throw std::invalid_argument(std::string("fused transition output must not overlap ") + operand.name);
  }

  abi::SM90Params device_params{};
  device_params.x = {static_cast<CUdeviceptr>(params.x.data), params.x.extent};
  device_params.mask = params.mask.data;
  device_params.image = {static_cast<CUdeviceptr>(params.image.data), params.image.extent};
  device_params.vec.data = static_cast<CUdeviceptr>(params.vec.data);
  device_params.output = {static_cast<CUdeviceptr>(params.output.data), params.output.extent};
  device_params.n_valid = params.n_valid.data;
  /* A row tile of tile_wgs of the CTA's warpgroups. A pair item's tiles never span two items, so each item's n * n
     rows (its valid rows at most) round up on their own, as the kernel's tile table counts them. */
  std::uint64_t const tile_rows = static_cast<std::uint64_t>(spec.tile_m) / num_wg(spec) * tile_wgs;
  std::uint64_t num_tiles = ceil_div(static_cast<std::uint64_t>(rows), tile_rows);
  if (pair_map)
  {
    std::uint64_t const plane = static_cast<std::uint64_t>(params.n) * static_cast<std::uint64_t>(params.n);
    num_tiles = static_cast<std::uint64_t>(params.n_batch) * ceil_div(plane, tile_rows);
  }
  device_params.rows = static_cast<std::int32_t>(rows);
  device_params.num_tiles = static_cast<std::int32_t>(checked_u32(num_tiles, "fused transition tile count"));
  /* Flat rows ignore the pair map's shape; pass zeros rather than whatever the caller left there. */
  device_params.n = pair_map ? params.n : 0;
  device_params.n_batch = pair_map ? params.n_batch : 0;
  device_params.eps = params.eps;
  device_params.residual = params.residual ? 1 : 0;
  device_params.split = split;
  device_params.tile_wgs = tile_wgs;
  void* kernel_params[abi::kSM90ParameterCount]{};
  abi::pack_sm90_kernel_params(&device_params, kernel_params);

  std::int32_t const multiprocessor_count = cuda_multiprocessor_count_for_context(context);
  if (multiprocessor_count <= 0)
    throw std::invalid_argument("current CUDA device has no active multiprocessors");
  cubin_launch_config_t launch_config{};
  if (split == 1)
  {
    /* A persistent grid: at most ctas_per_sm CTAs per SM, each striding over the tiles. */
    std::uint64_t const resident = static_cast<std::uint64_t>(multiprocessor_count) * spec.ctas_per_sm;
    launch_config.grid_x = checked_u32(std::min<std::uint64_t>(num_tiles, resident), "fused transition grid.x");
  }
  else
  {
    /* One cluster of split CTAs per row tile, each CTA one tile's share. */
    std::int64_t const grid = checked_mul(static_cast<std::int64_t>(num_tiles), split, "fused transition grid");
    launch_config.grid_x = checked_u32(static_cast<std::uint64_t>(grid), "fused transition grid.x");
    launch_config.cluster_x = static_cast<std::uint32_t>(split);
    launch_config.cluster_y = 1;
    launch_config.cluster_z = 1;
  }
  launch_config.grid_y = 1;
  launch_config.grid_z = 1;
  launch_config.block_x = spec.num_threads;
  launch_config.block_y = 1;
  launch_config.block_z = 1;
  launch_config.dynamic_smem_bytes = smem_bytes;
  launch_config.stream = reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream));
  /* The kernel reads x from its first instruction on and never waits on its predecessor: no programmatic launch. */
  launch_config.programmatic_stream_serialization = 0;
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr),
    "launch_cubin_kernel(transition_mlp_w_norm_sm90)");
}

KernelSpec make_kernel_spec(embedded::CubinImage const& image)
{
  return KernelSpec{
    image.cubin.target_sm,
    image.cubin.kernel_sm,
    image.width,
    image.hidden,
    image.is_silu_gate ? Activation::kSiLUGate : Activation::kReLU,
    image.num_threads,
    image.tile_m,
    image.ctas_per_sm,
    image.layouts,
    image.is_persistent,
  };
}

/* Images of an earlier launch ABI stay in the registry of a corpus built before it; they never match. */
embedded::CubinImage const& find_embedded_cubin(
  std::int32_t target_sm, DType dtype, std::int32_t width, std::int32_t hidden, Activation activation, bool persistent)
{
  bool const is_bfloat16 = dtype == DType::kBFloat16;
  bool const is_silu_gate = activation == Activation::kSiLUGate;
  embedded::RegistryView const registry = embedded::registry();
  for (std::size_t index = 0; index < registry.count; ++index)
  {
    embedded::CubinImage const& image = registry.images[index];
    if (
      cubin_supports_sm(image.cubin, target_sm) && image.cubin.launch_abi != nullptr
      && std::strcmp(image.cubin.launch_abi, kSM90LaunchAbi) == 0 && image.is_bfloat16 == is_bfloat16
      && image.width == width && image.hidden == hidden && image.is_silu_gate == is_silu_gate
      && image.is_persistent == persistent)
      return image;
  }
  throw std::invalid_argument(
    "No embedded fused transition CUBIN for SM" + std::to_string(target_sm) + ", width=" + std::to_string(width)
    + ", hidden=" + std::to_string(hidden) + ", activation=" + (is_silu_gate ? "silu_gate" : "relu")
    + (persistent ? ", persistent" : ""));
}

std::size_t preload_kernels(CUcontext context, std::int32_t device_sm)
{
  return preload_registry_kernels(context, device_sm, embedded::registry());
}

CubinPreloadRegistration const kPreloader{
  "transition_mlp_w_norm",
  &preload_kernels,
};

} // namespace

std::uint32_t dynamic_smem_bytes(KernelConfig const& config)
{
  if (config.cubin.dynamic_smem_bytes == 0)
    throw std::invalid_argument("fused transition CUBIN has no dynamic shared-memory metadata");
  return config.cubin.dynamic_smem_bytes;
}

std::vector<KernelSpec> kernel_specs()
{
  return map_registry(embedded::registry(), make_kernel_spec);
}

bool supports_layout(KernelSpec const& spec, std::int32_t split, std::int32_t tile_wgs)
{
  std::int32_t const wgs = tile_wgs == 0 ? num_wg(spec) : tile_wgs;
  std::int32_t const bit = layout_bit(wgs, split);
  return bit >= 0 && wgs <= num_wg(spec) && ((spec.layouts >> bit) & 1U) != 0;
}

KernelConfig make_kernel_config(
  std::int32_t target_sm, DType dtype, std::int32_t width, std::int32_t hidden, Activation activation, bool persistent)
{
  embedded::CubinImage const& image = find_embedded_cubin(target_sm, dtype, width, hidden, activation, persistent);
  return KernelConfig{make_kernel_spec(image), dtype, image.cubin, &image};
}

void launch(KernelConfig const& config, LaunchParams const& params)
{
  validate_image(config);
  CUcontext const context = current_cuda_context();
  std::int32_t const device_sm = cuda_sm_for_context(context);
  if (device_sm != config.spec.target_sm || !cubin_supports_sm(config.cubin, device_sm))
  {
    throw std::invalid_argument(
      "fused transition config selects SM" + std::to_string(config.spec.target_sm) + " but current device is SM"
      + std::to_string(device_sm));
  }
  cubin_kernel_t const loaded = load_embedded_kernel(context, config.cubin);
  launch_sm90(loaded, config, params, context, dynamic_smem_bytes(config));
}

} // namespace bioir::cutedsl::transition_mlp_w_norm
