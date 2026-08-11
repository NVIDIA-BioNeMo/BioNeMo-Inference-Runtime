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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtbnm::cutedsl::pair_weighted_averaging
{
namespace
{

constexpr char kLaunchAbi[] = "pair_weighted_averaging_sm80_v1";
constexpr std::int32_t kKernelSM = 80;
constexpr std::int32_t kH = 8;
constexpr std::int32_t kJAlignment = 8;

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
  bool found_n_anchor = false;
  std::int32_t nearest_n_anchor = 0;
  double nearest_n_distance = 0;

  for (std::size_t image_index = 0; image_index < embedded::kCubinCount; ++image_index)
  {
    embedded::CubinImage const& image = embedded::kCubins[image_index];
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
  for (std::size_t image_index = 0; image_index < embedded::kCubinCount; ++image_index)
  {
    embedded::CubinImage const& image = embedded::kCubins[image_index];
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
  if (
    config.cubin.kernel_sm != kKernelSM || !equal_c_strings(config.cubin.launch_abi, kLaunchAbi)
    || config.cubin.non_portable_cluster_size_allowed)
  {
    throw std::invalid_argument("pair_weighted_averaging CUBIN has incompatible SM80 launch metadata");
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

  std::int32_t const B = params.w.shape[0];
  std::int32_t const I = params.w.shape[2];
  std::int32_t const Jp = params.w.shape[3];
  std::int32_t const S = params.v.shape[1];
  std::int32_t const N = params.v.shape[2];

  embedded::CubinImage const& image = *config.embedded_image;
  std::int32_t const hidden = image.H * image.D;

  if (params.w.shape[1] != kH)
    throw std::invalid_argument("w shape must be [B, 8, I, Jp]");
  if (
    params.v.shape[0] != B || params.v.shape[3] != hidden || params.g.shape[0] != B || params.g.shape[1] != S
    || params.g.shape[2] != I || params.g.shape[3] != hidden)
  {
    throw std::invalid_argument(
      "v/g shapes must be [B, S, N, " + std::to_string(hidden) + "] and [B, S, I, " + std::to_string(hidden) + "]");
  }
  if (params.weight.shape[0] != image.c_m || params.weight.shape[1] != hidden)
  {
    throw std::invalid_argument(
      "weight shape must be [" + std::to_string(image.c_m) + ", " + std::to_string(hidden) + "]");
  }
  if (
    params.output.shape[0] != B || params.output.shape[1] != S || params.output.shape[2] != I
    || params.output.shape[3] != image.c_m)
  {
    throw std::invalid_argument("output shape must be [B, S, I, " + std::to_string(image.c_m) + "]");
  }

  if (
    Jp < N || Jp % kJAlignment != 0
    || ceil_div(static_cast<std::uint64_t>(Jp), image.tile_j) != ceil_div(static_cast<std::uint64_t>(N), image.tile_j))
  {
    throw std::invalid_argument(
      "w.shape[3] must cover v.shape[2], be 8-aligned, and use the same number of TILE_J blocks");
  }

  validate_contiguous_tensor4(params.w, "w");
  validate_contiguous_tensor4(params.output, "output");
  if (params.weight.strides[0] != hidden)
    throw std::invalid_argument("weight must be contiguous row-major");
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

bool is_preload_compatible(embedded::CubinImage const& image, std::int32_t device_sm)
{
  return image.cubin.target_sm == device_sm && cubin_supports_sm(image.cubin, device_sm)
    && image.cubin.kernel_sm == kKernelSM && equal_c_strings(image.cubin.launch_abi, kLaunchAbi)
    && !image.cubin.non_portable_cluster_size_allowed && image.cubin.dynamic_smem_bytes > 0 && image.tile_i > 0
    && image.tile_s > 0 && image.tile_j > 0 && image.num_threads > 0 && image.num_threads <= 1024;
}

std::size_t preload_kernels(CUcontext context, std::int32_t device_sm)
{
  std::size_t loaded = 0;
  for (std::size_t index = 0; index < embedded::kCubinCount; ++index)
  {
    embedded::CubinImage const& image = embedded::kCubins[index];
    if (!is_preload_compatible(image, device_sm))
      continue;
    (void) load_embedded_kernel(context, image.cubin, false);
    ++loaded;
  }
  return loaded;
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
  launch_sm80(loaded, config, params, config.cubin.dynamic_smem_bytes);
}

} // namespace trtbnm::cutedsl::pair_weighted_averaging
