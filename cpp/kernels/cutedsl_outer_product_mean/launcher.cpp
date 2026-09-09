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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace bioir::cutedsl::outer_product_mean
{
namespace
{

constexpr char kSM80LaunchAbi[] = "outer_product_mean_sm80";

void validate_launch(KernelConfig const& config, LaunchParams const& params)
{
  if (config.cubin.data == nullptr || config.cubin.size == 0)
    throw std::invalid_argument("embedded CUBIN image must not be empty");
  if (config.cubin.variant_id == nullptr || config.cubin.variant_id[0] == '\0')
    throw std::invalid_argument("embedded CUBIN variant_id must not be empty");
  if (config.cubin.kernel_symbol == nullptr || config.cubin.kernel_symbol[0] == '\0')
    throw std::invalid_argument("kernel_symbol must not be empty");
  if (
    config.cubin.kernel_sm != 80 || config.cubin.launch_abi == nullptr
    || std::strcmp(config.cubin.launch_abi, kSM80LaunchAbi) != 0)
  {
    throw std::invalid_argument("outer-product-mean CUBIN has an incompatible launch ABI");
  }
  if (!cubin_supports_sm(config.cubin, config.spec.target_sm))
    throw std::invalid_argument("outer-product-mean CUBIN does not support its configured target SM");
  if (config.spec.num_threads <= 0 || config.spec.tile_i <= 0 || config.spec.tile_j <= 0)
    throw std::invalid_argument("outer-product-mean spec has a non-positive tile or thread count");
  if (config.spec.raster_factor <= 0)
    throw std::invalid_argument("outer-product-mean spec has a non-positive rasterization factor");

  validate_tensor(params.a, "a", 16);
  validate_tensor(params.b, "b", 16);
  validate_tensor(params.num_mask, "num_mask", 4);
  validate_tensor(params.weight, "weight", 16);
  if (config.has_bias)
    validate_tensor(params.bias, "bias", 16);
  validate_tensor(params.output, "output", 16);

  /* a is [B, S, I, C] and b is [B, S, J, D]; the views keep (B, S, I) and
   * (B, S, J). C, D and C_z never reach the descriptors because the kernel
   * baked them in, so check them against the compiled constants here.
   */
  std::int32_t const batch = params.a.shape[0];
  std::int32_t const sequence = params.a.shape[1];
  std::int32_t const rows = params.a.shape[2];
  std::int32_t const columns = params.b.shape[2];

  if (params.b.shape[0] != batch || params.b.shape[1] != sequence)
    throw std::invalid_argument("a and b must share their batch and sequence extents");
  if (params.num_mask.shape[0] != batch || params.num_mask.shape[1] != rows || params.num_mask.shape[2] != columns)
    throw std::invalid_argument("num_mask must be [B, I, J]");
  if (params.output.shape[0] != batch || params.output.shape[1] != rows || params.output.shape[2] != columns)
    throw std::invalid_argument("output must be [B, I, J, C_z]");
  if (params.weight.shape[0] != kChannelsCz || params.weight.shape[1] != kChannelsC * kChannelsD)
  {
    throw std::invalid_argument(
      "weight must be [" + std::to_string(kChannelsCz) + ", " + std::to_string(kChannelsC * kChannelsD)
      + "]: the kernel compiles C, D and C_z as constants");
  }
  if (config.has_bias && params.bias.shape[0] != kChannelsCz)
    throw std::invalid_argument("bias must be [C_z]");
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

  throw std::invalid_argument(
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
  launch_sm80(loaded, config, params, context, dynamic_smem_bytes(config));
}

} // namespace bioir::cutedsl::outer_product_mean
