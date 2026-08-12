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

#include "cubins/embedded_cubins.h"

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtbnm::cutedsl::dual_gemm_x_x
{
namespace
{

constexpr char kSM80LaunchAbi[] = "dual_gemm_x_x_sm80";
constexpr char kSM90LaunchAbi[] = "dual_gemm_x_x_sm90";
constexpr std::int64_t kElementsPer16Bytes = 8;

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
  throw std::invalid_argument("dual_gemm_x_x config has an unsupported dtype");
}

bool equal_c_strings(char const* lhs, char const* rhs)
{
  return lhs != nullptr && rhs != nullptr && std::strcmp(lhs, rhs) == 0;
}

std::uint64_t checked_multiply(std::uint64_t lhs, std::uint64_t rhs, char const* name)
{
  if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
    throw std::overflow_error(std::string(name) + " overflow");
  return lhs * rhs;
}

cute_tensor_s1_d0_t make_static_weight_descriptor(Tensor2View const& view)
{
  cute_tensor_s1_d0_t descriptor{};
  descriptor.data = static_cast<CUdeviceptr>(view.data);
  descriptor.dynamic_shapes[0] = view.shape[0];
  return descriptor;
}

void validate_tma_metadata(
  TmaDescriptorInfo const& info, CUtensorMapDataType expected_dtype, bool is_column_major, char const* name)
{
  if (info.rank != 2)
    throw std::invalid_argument(std::string("native SM90 ") + name + " metadata must describe a rank-2 TMA map");
  if (info.data_type != expected_dtype)
    throw std::invalid_argument(std::string("native SM90 ") + name + " TMA metadata has the wrong dtype");

  std::uint32_t const expected_inner = is_column_major ? 0U : 1U;
  std::uint32_t const expected_outer = is_column_major ? 1U : 0U;
  if (info.global_dim_order[0] != expected_inner || info.global_dim_order[1] != expected_outer)
  {
    throw std::invalid_argument(
      std::string("native SM90 ") + name + " TMA metadata has an incompatible dimension order");
  }
  for (std::size_t index = 0; index < 2; ++index)
  {
    if (info.box_dims[index] == 0 || info.element_strides[index] != 1)
      throw std::invalid_argument(std::string("native SM90 ") + name + " TMA metadata has invalid traversal geometry");
  }
}

void validate_sm90_metadata(KernelConfig const& config, embedded::CubinImage const& image)
{
  embedded::SM90LaunchInfo const& sm90 = image.sm90;
  if (!sm90.enabled)
    throw std::invalid_argument("native SM90 dual_gemm_x_x CUBIN has no host launch metadata");
  for (std::uint32_t dimension : sm90.block_dims)
  {
    if (dimension == 0)
      throw std::invalid_argument("native SM90 dual_gemm_x_x CUBIN has an invalid block dimension");
  }
  for (std::uint32_t dimension : sm90.cluster_dims)
  {
    if (dimension == 0)
      throw std::invalid_argument("native SM90 dual_gemm_x_x CUBIN has an invalid cluster dimension");
  }
  if (sm90.block_dims[0] != image.num_threads || sm90.block_dims[1] != 1 || sm90.block_dims[2] != 1)
    throw std::invalid_argument("native SM90 block metadata disagrees with the generated launch geometry");
  if (sm90.cluster_dims[1] != 1 || sm90.cluster_dims[2] != 1)
  {
    throw std::invalid_argument(
      "native SM90 dual_gemm_x_x flattened persistent grid requires cluster dimensions [x, 1, 1]");
  }

  CUtensorMapDataType const expected_dtype = tma_data_type(config.dtype == DType::kBFloat16);
  validate_tma_metadata(sm90.x0, expected_dtype, false, "x0");
  validate_tma_metadata(sm90.x1, expected_dtype, false, "x1");
  validate_tma_metadata(sm90.w0, expected_dtype, false, "w0");
  validate_tma_metadata(sm90.w1, expected_dtype, false, "w1");
  validate_tma_metadata(sm90.output, expected_dtype, config.transpose_out, "output");
}

void validate_config(KernelConfig const& config)
{
  if (config.embedded_image == nullptr)
    throw std::invalid_argument("dual_gemm_x_x config has no generated CUBIN image");
  embedded::CubinImage const& image = *config.embedded_image;

  if (config.cubin.data == nullptr || config.cubin.size == 0)
    throw std::invalid_argument("embedded CUBIN image must not be empty");
  if (config.cubin.variant_id == nullptr || config.cubin.variant_id[0] == '\0')
    throw std::invalid_argument("embedded CUBIN variant_id must not be empty");
  if (config.cubin.kernel_symbol == nullptr || config.cubin.kernel_symbol[0] == '\0')
    throw std::invalid_argument("kernel_symbol must not be empty");
  if (config.cubin.launch_abi == nullptr || config.cubin.launch_abi[0] == '\0')
    throw std::invalid_argument("launch_abi must not be empty");
  if (
    image.cubin.target_sm != config.cubin.target_sm || image.cubin.kernel_sm != config.cubin.kernel_sm
    || image.cubin.dynamic_smem_bytes != config.cubin.dynamic_smem_bytes
    || image.cubin.non_portable_cluster_size_allowed != config.cubin.non_portable_cluster_size_allowed
    || image.cubin.supported_sms != config.cubin.supported_sms
    || image.cubin.supported_sm_count != config.cubin.supported_sm_count || image.cubin.data != config.cubin.data
    || image.cubin.size != config.cubin.size || !equal_c_strings(image.cubin.variant_id, config.cubin.variant_id)
    || !equal_c_strings(image.cubin.kernel_symbol, config.cubin.kernel_symbol)
    || !equal_c_strings(image.cubin.launch_abi, config.cubin.launch_abi))
  {
    throw std::invalid_argument("dual_gemm_x_x config and generated CUBIN image disagree");
  }
  if (config.target_sm != config.cubin.target_sm || !cubin_supports_sm(config.cubin, config.target_sm))
  {
    throw std::invalid_argument(
      "embedded CUBIN does not support configured device SM" + std::to_string(config.target_sm));
  }
  if (
    config.K != image.K || image.is_bfloat16 != dtype_is_bfloat16(config.dtype)
    || config.transpose_out != image.transpose_out || config.has_bias != image.has_bias
    || config.has_mask != image.has_mask)
  {
    throw std::invalid_argument("dual_gemm_x_x config axes disagree with the generated CUBIN image");
  }

  bool alias_found = false;
  if (image.aliases != nullptr)
  {
    for (std::size_t index = 0; index < image.alias_count; ++index)
    {
      embedded::RuntimeAlias const& alias = image.aliases[index];
      if (alias.N == config.N && alias.bucket == config.bucket)
      {
        alias_found = true;
        break;
      }
    }
  }
  if (!alias_found)
    throw std::invalid_argument("dual_gemm_x_x config does not name a generated runtime alias");

  if (image.tile_m == 0 || image.tile_n == 0 || image.tile_k == 0 || image.num_threads == 0 || image.num_threads > 1024)
  {
    throw std::invalid_argument("dual_gemm_x_x CUBIN has invalid launch geometry");
  }
  if (config.cubin.dynamic_smem_bytes == 0)
    throw std::invalid_argument("dual_gemm_x_x CUBIN has no dynamic shared-memory metadata");

  bool const is_sm80 = config.cubin.kernel_sm == 80;
  bool const is_sm90 = config.cubin.kernel_sm == 90;
  if (!is_sm80 && !is_sm90)
    throw std::invalid_argument("dual_gemm_x_x CUBIN has no registered kernel-SM launcher");
  if ((config.target_sm == 90) != is_sm90)
    throw std::invalid_argument("dual_gemm_x_x target SM and kernel SM are inconsistent");

  char const* expected_abi = is_sm80 ? kSM80LaunchAbi : kSM90LaunchAbi;
  if (!equal_c_strings(config.cubin.launch_abi, expected_abi))
    throw std::invalid_argument("dual_gemm_x_x CUBIN has an incompatible launch ABI");
  if (config.cubin.non_portable_cluster_size_allowed != is_sm90)
    throw std::invalid_argument("dual_gemm_x_x CUBIN has inconsistent cluster function metadata");

  if (is_sm80)
  {
    if (image.sm90.enabled)
      throw std::invalid_argument("SM80 dual_gemm_x_x CUBIN unexpectedly carries native SM90 launch metadata");
    if (
      image.raster_factor == 0
      || image.raster_factor > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
      throw std::invalid_argument("SM80 dual_gemm_x_x CUBIN requires a positive raster factor");
  }
  else
  {
    validate_sm90_metadata(config, image);
  }
}

void validate_launch(KernelConfig const& config, LaunchParams const& params)
{
  validate_config(config);
  if (params.i_dim <= 0)
    throw std::invalid_argument("i_dim must be positive");

  validate_tensor(params.x, "x", 16);
  validate_tensor(params.w0, "w0", 16);
  validate_tensor(params.w1, "w1", 16);
  validate_tensor(params.output, "output", 16);

  std::int32_t const M = params.x.shape[0];
  if (params.x.shape[1] != config.K)
    throw std::invalid_argument("x shape must be [M, K] for the configured K");
  if (params.output.shape[0] != M || params.output.shape[1] != config.N)
    throw std::invalid_argument("output shape must be [M, N] for the configured N");
  if (params.w0.shape[0] != config.N || params.w0.shape[1] != config.K || params.w1.shape != params.w0.shape)
  {
    throw std::invalid_argument("w0 and w1 shapes must both be [N, K]");
  }
  if (params.w0.strides[0] != config.K || params.w1.strides[0] != config.K)
    throw std::invalid_argument("w0 and w1 must use the static contiguous row stride K");
  if (params.x.strides[0] < config.K || params.x.strides[0] % kElementsPer16Bytes != 0)
    throw std::invalid_argument("x row stride must cover K and be 16-byte aligned");

  std::int64_t const output_contiguous_extent = config.transpose_out ? M : config.N;
  if (params.output.strides[0] < output_contiguous_extent || params.output.strides[0] % kElementsPer16Bytes != 0)
  {
    throw std::invalid_argument(
      config.transpose_out ? "column-major output stride must cover M and be 16-byte aligned"
                           : "row-major output stride must cover N and be 16-byte aligned");
  }
  if (M % params.i_dim != 0)
    throw std::invalid_argument("x.shape[0] must be divisible by i_dim");

  if (config.has_bias)
  {
    validate_tensor(params.bias0, "bias0", 16);
    validate_tensor(params.bias1, "bias1", 16);
    if (params.bias0.shape[0] != config.N || params.bias1.shape[0] != config.N)
      throw std::invalid_argument("bias0 and bias1 shapes must both be [N]");
  }
  if (config.has_mask)
  {
    validate_tensor(params.actual_seqlen, "actual_seqlen", 4);
    if (params.actual_seqlen.shape[0] != M / params.i_dim)
      throw std::invalid_argument("actual_seqlen shape must be [M / i_dim]");
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

  check(params.x.device, "x");
  check(params.w0.device, "w0");
  check(params.w1.device, "w1");
  check(params.output.device, "output");
  if (config.has_bias)
  {
    check(params.bias0.device, "bias0");
    check(params.bias1.device, "bias1");
  }
  if (config.has_mask)
    check(params.actual_seqlen.device, "actual_seqlen");
}

cubin_launch_config_t
make_sm80_launch_config(embedded::CubinImage const& image, LaunchParams const& params, std::uint32_t smem_bytes)
{
  std::uint64_t const grid_m
    = ceil_div(static_cast<std::uint64_t>(params.x.shape[0]), static_cast<std::uint64_t>(image.tile_m));
  std::uint64_t const grid_n
    = ceil_div(static_cast<std::uint64_t>(params.output.shape[1]), static_cast<std::uint64_t>(image.tile_n));
  std::uint64_t const grid_x
    = checked_multiply(grid_m, static_cast<std::uint64_t>(image.raster_factor), "dual_gemm_x_x SM80 grid.x");
  std::uint64_t const grid_y = ceil_div(grid_n, static_cast<std::uint64_t>(image.raster_factor));

  cubin_launch_config_t launch_config{};
  launch_config.grid_x = checked_u32(grid_x, "dual_gemm_x_x SM80 grid.x");
  launch_config.grid_y = checked_u32(grid_y, "dual_gemm_x_x SM80 grid.y");
  launch_config.grid_z = 1;
  launch_config.block_x = image.num_threads;
  launch_config.block_y = 1;
  launch_config.block_z = 1;
  launch_config.dynamic_smem_bytes = smem_bytes;
  launch_config.stream = reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream));
  return launch_config;
}

cubin_launch_config_t make_sm90_launch_config(
  embedded::CubinImage const& image, LaunchParams const& params, CUcontext context, std::uint32_t smem_bytes)
{
  embedded::SM90LaunchInfo const& metadata = image.sm90;
  std::uint64_t const gm
    = ceil_div(static_cast<std::uint64_t>(params.x.shape[0]), static_cast<std::uint64_t>(image.tile_m));
  std::uint64_t const gn
    = ceil_div(static_cast<std::uint64_t>(params.output.shape[1]), static_cast<std::uint64_t>(image.tile_n));
  std::uint64_t const logical_gm = ceil_div(gm, static_cast<std::uint64_t>(metadata.cluster_dims[0]));
  std::uint64_t const logical_gn = ceil_div(gn, static_cast<std::uint64_t>(metadata.cluster_dims[1]));
  std::uint64_t const cluster_xy = checked_multiply(
    static_cast<std::uint64_t>(metadata.cluster_dims[0]),
    static_cast<std::uint64_t>(metadata.cluster_dims[1]),
    "dual_gemm_x_x SM90 cluster size");
  std::uint64_t const cluster_size = checked_multiply(
    cluster_xy, static_cast<std::uint64_t>(metadata.cluster_dims[2]), "dual_gemm_x_x SM90 cluster size");

  std::int32_t const multiprocessor_count = cuda_multiprocessor_count_for_context(context);
  if (multiprocessor_count <= 0)
    throw std::invalid_argument("current CUDA device has no active multiprocessors");
  std::uint64_t const max_clusters = static_cast<std::uint64_t>(multiprocessor_count) / cluster_size;
  if (max_clusters == 0)
    throw std::invalid_argument("native SM90 cluster size exceeds the current device's multiprocessor count");

  std::uint64_t persistent_clusters = 0;
  if (image.raster_factor > 0)
  {
    std::uint64_t const raster_factor = image.raster_factor;
    std::uint64_t const padded_gm
      = checked_multiply(ceil_div(logical_gm, raster_factor), raster_factor, "dual_gemm_x_x SM90 rasterized M tiles");
    std::uint64_t const total_iterations
      = checked_multiply(padded_gm, logical_gn, "dual_gemm_x_x SM90 total iterations");
    std::uint64_t const rounded_cap = max_clusters / raster_factor * raster_factor;
    std::uint64_t const cap = std::max(rounded_cap, raster_factor);
    persistent_clusters = std::min(total_iterations, cap);
  }
  else
  {
    std::uint64_t const logical_total
      = checked_multiply(logical_gm, logical_gn, "dual_gemm_x_x SM90 logical tile count");
    persistent_clusters = std::min(logical_total, max_clusters);
  }
  std::uint64_t const grid_x = checked_multiply(persistent_clusters, cluster_size, "dual_gemm_x_x SM90 grid.x");

  cubin_launch_config_t launch_config{};
  launch_config.grid_x = checked_u32(grid_x, "dual_gemm_x_x SM90 grid.x");
  launch_config.grid_y = 1;
  launch_config.grid_z = 1;
  launch_config.block_x = metadata.block_dims[0];
  launch_config.block_y = metadata.block_dims[1];
  launch_config.block_z = metadata.block_dims[2];
  launch_config.cluster_x = metadata.cluster_dims[0];
  launch_config.cluster_y = metadata.cluster_dims[1];
  launch_config.cluster_z = metadata.cluster_dims[2];
  launch_config.cluster_scheduling_policy = metadata.cluster_scheduling_policy;
  launch_config.dynamic_smem_bytes = smem_bytes;
  launch_config.stream = reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream));
  return launch_config;
}

void launch_sm80(
  cubin_kernel_t loaded, KernelConfig const& config, LaunchParams const& params, std::uint32_t smem_bytes)
{
  embedded::CubinImage const& image = *config.embedded_image;
  abi::SM80Params device_params{};
  device_params.x = make_tensor2_s1_d1_descriptor(params.x);
  device_params.w0 = make_static_weight_descriptor(params.w0);
  device_params.w1 = make_static_weight_descriptor(params.w1);
  if (config.has_bias)
  {
    device_params.bias0 = make_tensor1_descriptor(params.bias0);
    device_params.bias1 = make_tensor1_descriptor(params.bias1);
  }
  if (config.has_mask)
    device_params.actual_seqlen = make_tensor1_descriptor(params.actual_seqlen);
  device_params.output = make_tensor2_s2_d1_descriptor(params.output);
  device_params.i_dim = params.i_dim;
  device_params.raster_factor = static_cast<std::int32_t>(image.raster_factor);

  void* kernel_params[abi::kSM80MaxParameterCount]{};
  std::size_t const parameter_count
    = abi::pack_sm80_kernel_params(&device_params, config.has_bias, config.has_mask, kernel_params);
  if (parameter_count != abi::sm80_parameter_count(config.has_bias, config.has_mask))
    throw std::logic_error("dual_gemm_x_x SM80 parameter packer produced the wrong ABI count");

  cubin_launch_config_t const launch_config = make_sm80_launch_config(image, params, smem_bytes);
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(dual_gemm_x_x_sm80)");
}

void launch_sm90(
  cubin_kernel_t loaded,
  KernelConfig const& config,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  embedded::CubinImage const& image = *config.embedded_image;
  embedded::SM90LaunchInfo const& metadata = image.sm90;
  CUtensorMapDataType const expected_dtype = tma_data_type(config.dtype == DType::kBFloat16);

  abi::SM90Params device_params{};
  TmaTensorSource const x_source = make_tma_tensor2_source(params.x, false);
  TmaTensorSource const w0_source = make_tma_tensor2_source(params.w0, false);
  TmaTensorSource const w1_source = make_tma_tensor2_source(params.w1, false);
  TmaTensorSource const output_source = make_tma_tensor2_source(params.output, config.transpose_out);
  encode_tma_descriptor(device_params.x0_tma, metadata.x0, expected_dtype, x_source, "x0");
  encode_tma_descriptor(device_params.x1_tma, metadata.x1, expected_dtype, x_source, "x1");
  encode_tma_descriptor(device_params.w0_tma, metadata.w0, expected_dtype, w0_source, "w0");
  encode_tma_descriptor(device_params.w1_tma, metadata.w1, expected_dtype, w1_source, "w1");
  encode_tma_descriptor(device_params.output_tma, metadata.output, expected_dtype, output_source, "output");

  device_params.x0_coord = make_tensor2_s1_coord(params.x);
  device_params.x1_coord = device_params.x0_coord;
  device_params.w0_coord = make_tensor2_s1_coord(params.w0);
  device_params.w1_coord = make_tensor2_s1_coord(params.w1);
  device_params.output_coord = make_tensor2_s2_coord(params.output);
  if (config.has_bias)
  {
    device_params.bias0 = make_tensor1_descriptor(params.bias0);
    device_params.bias1 = make_tensor1_descriptor(params.bias1);
  }
  if (config.has_mask)
    device_params.actual_seqlen = make_tensor1_descriptor(params.actual_seqlen);
  device_params.i_dim = params.i_dim;
  device_params.tiled_mma = 0;

  void* kernel_params[abi::kSM90MaxParameterCount]{};
  std::size_t const parameter_count
    = abi::pack_sm90_kernel_params(&device_params, config.has_bias, config.has_mask, kernel_params);
  if (parameter_count != abi::sm90_parameter_count(config.has_bias, config.has_mask))
    throw std::logic_error("dual_gemm_x_x SM90 parameter packer produced the wrong ABI count");

  cubin_launch_config_t const launch_config = make_sm90_launch_config(image, params, context, smem_bytes);
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(dual_gemm_x_x_sm90)");
}

EmbeddedSelection find_embedded_cubin(
  std::int32_t target_sm,
  std::int32_t K,
  std::int32_t N,
  std::int32_t S,
  DType dtype,
  bool transpose_out,
  bool has_bias,
  bool has_mask)
{
  if (target_sm <= 0 || K <= 0 || N <= 0)
    throw std::invalid_argument("dual_gemm_x_x target SM, K, and N must be positive");
  if (S < 0)
    throw std::invalid_argument("dual_gemm_x_x S must be non-negative");

  bool const is_bfloat16 = dtype_is_bfloat16(dtype);
  EmbeddedSelection nearest{};
  std::uint64_t nearest_distance = 0;
  for (std::size_t image_index = 0; image_index < embedded::kCubinCount; ++image_index)
  {
    embedded::CubinImage const& image = embedded::kCubins[image_index];
    if (
      !cubin_supports_sm(image.cubin, target_sm) || image.K != K || image.is_bfloat16 != is_bfloat16
      || image.transpose_out != transpose_out || image.has_bias != has_bias || image.has_mask != has_mask)
    {
      continue;
    }

    if (image.aliases == nullptr)
      continue;
    for (std::size_t alias_index = 0; alias_index < image.alias_count; ++alias_index)
    {
      embedded::RuntimeAlias const& alias = image.aliases[alias_index];
      if (alias.N != N)
        continue;

      std::int64_t const delta = static_cast<std::int64_t>(alias.bucket) - static_cast<std::int64_t>(S);
      std::uint64_t const distance = static_cast<std::uint64_t>(delta < 0 ? -delta : delta);
      if (
        nearest.image == nullptr || distance < nearest_distance
        || (distance == nearest_distance && alias.bucket < nearest.alias->bucket))
      {
        nearest = EmbeddedSelection{&image, &alias};
        nearest_distance = distance;
      }
    }
  }
  if (nearest.image != nullptr)
    return nearest;

  throw std::invalid_argument(
    "No embedded dual_gemm_x_x CUBIN for SM" + std::to_string(target_sm) + ", K=" + std::to_string(K)
    + ", N=" + std::to_string(N) + ", S=" + std::to_string(S));
}

std::size_t preload_kernels(CUcontext context, std::int32_t device_sm)
{
  std::size_t loaded = 0;
  for (std::size_t index = 0; index < embedded::kCubinCount; ++index)
  {
    EmbeddedCubinImage const& image = embedded::kCubins[index].cubin;
    if (!cubin_supports_sm(image, device_sm))
      continue;
    (void) load_embedded_kernel(context, image, false);
    ++loaded;
  }
  return loaded;
}

CubinPreloadRegistration const kPreloader{
  "dual_gemm_x_x",
  &preload_kernels,
};

} // namespace

KernelConfig make_kernel_config(
  std::int32_t target_sm,
  std::int32_t K,
  std::int32_t N,
  std::int32_t S,
  DType dtype,
  bool transpose_out,
  bool has_bias,
  bool has_mask)
{
  EmbeddedSelection const selection = find_embedded_cubin(target_sm, K, N, S, dtype, transpose_out, has_bias, has_mask);
  return KernelConfig{
    target_sm,
    selection.image->K,
    selection.alias->N,
    selection.alias->bucket,
    dtype,
    transpose_out,
    has_bias,
    has_mask,
    selection.image->cubin,
    selection.image,
  };
}

std::uint32_t dynamic_smem_bytes(KernelConfig const& config)
{
  if (config.cubin.dynamic_smem_bytes == 0)
    throw std::invalid_argument("dual_gemm_x_x CUBIN has no dynamic shared-memory metadata");
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
      "dual_gemm_x_x config selects SM" + std::to_string(config.target_sm) + " but current device is SM"
      + std::to_string(device_sm));
  }
  validate_operand_devices(config, params, cuda_device_for_context(context));

  cubin_kernel_t const loaded = load_embedded_kernel(context, config.cubin);
  std::uint32_t const smem_bytes = dynamic_smem_bytes(config);
  if (config.cubin.kernel_sm == 80)
  {
    launch_sm80(loaded, config, params, smem_bytes);
    return;
  }
  if (config.cubin.kernel_sm == 90)
  {
    launch_sm90(loaded, config, params, context, smem_bytes);
    return;
  }
  throw std::invalid_argument("dual_gemm_x_x config has no registered SM launcher");
}

} // namespace trtbnm::cutedsl::dual_gemm_x_x
