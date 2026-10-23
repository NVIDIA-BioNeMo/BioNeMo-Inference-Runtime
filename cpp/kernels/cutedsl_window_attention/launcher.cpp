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

#include "window_attention_registry.h"

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace bioir::cutedsl::window_attention
{
namespace
{

constexpr char kSM80LaunchAbi[] = "window_attention_sm80_v1";
constexpr char kSM90LaunchAbi[] = "window_attention_sm90_v1";
constexpr std::int32_t kSM80KernelSM = 80;
constexpr std::int32_t kSM90KernelSM = 90;
// Both images were compiled for sample, row, and bias strides divisible by 8 elements (16 bytes).
constexpr std::int64_t kStrideAlignment = 8;
// Samples and windows sit on the grid's y and z axes.
constexpr std::int32_t kMaxGridYZ = 65535;

bool dtype_is_bfloat16(DType dtype)
{
  switch (dtype)
  {
  case DType::kFloat16: return false;
  case DType::kBFloat16: return true;
  }
  throw std::invalid_argument("window_attention config has an unsupported dtype");
}

bool equal_c_strings(char const* lhs, char const* rhs)
{
  return lhs != nullptr && rhs != nullptr && std::strcmp(lhs, rhs) == 0;
}

bool is_sm90_abi(EmbeddedCubinImage const& cubin)
{
  return cubin.kernel_sm == kSM90KernelSM && equal_c_strings(cubin.launch_abi, kSM90LaunchAbi);
}

bool is_sm80_abi(EmbeddedCubinImage const& cubin)
{
  return cubin.kernel_sm == kSM80KernelSM && equal_c_strings(cubin.launch_abi, kSM80LaunchAbi);
}

bool has_launch_metadata(embedded::CubinImage const& image)
{
  // The WGMMA image runs only on the SM90 it was built for.
  bool const abi_matches = is_sm80_abi(image.cubin) || (is_sm90_abi(image.cubin) && image.cubin.target_sm == 90);
  return abi_matches && !image.cubin.non_portable_cluster_size_allowed && image.cubin.dynamic_smem_bytes > 0
    && image.head_dim > 0 && image.n_queries > 0 && image.n_keys > 0 && image.num_threads > 0
    && image.num_threads <= 1024;
}

embedded::CubinImage const* find_embedded_cubin(
  std::int32_t target_sm, std::int32_t head_dim, std::int32_t n_queries, std::int32_t n_keys, DType dtype)
{
  if (target_sm <= 0 || head_dim <= 0 || n_queries <= 0 || n_keys <= 0)
    throw std::invalid_argument("window_attention target SM, head_dim, n_queries, and n_keys must be positive");

  bool const is_bfloat16 = dtype_is_bfloat16(dtype);
  embedded::RegistryView const registry = embedded::registry();
  for (std::size_t index = 0; index < registry.count; ++index)
  {
    embedded::CubinImage const& image = registry.images[index];
    if (
      image.cubin.target_sm == target_sm && cubin_supports_sm(image.cubin, target_sm) && image.head_dim == head_dim
      && image.n_queries == n_queries && image.n_keys == n_keys && image.is_bfloat16 == is_bfloat16)
    {
      return &image;
    }
  }
  throw std::invalid_argument(
    "No embedded window_attention CUBIN for SM" + std::to_string(target_sm) + ", head_dim=" + std::to_string(head_dim)
    + ", n_queries=" + std::to_string(n_queries) + ", n_keys=" + std::to_string(n_keys)
    + (is_bfloat16 ? ", bf16" : ", fp16"));
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
    throw std::invalid_argument("window_attention config has no generated CUBIN image");
  embedded::CubinImage const& image = *config.embedded_image;

  if (config.cubin.data == nullptr || config.cubin.size == 0)
    throw std::invalid_argument("embedded CUBIN image must not be empty");
  if (config.cubin.variant_id == nullptr || config.cubin.variant_id[0] == '\0')
    throw std::invalid_argument("embedded CUBIN variant_id must not be empty");
  if (config.cubin.kernel_symbol == nullptr || config.cubin.kernel_symbol[0] == '\0')
    throw std::invalid_argument("kernel_symbol must not be empty");
  if (!same_cubin_image(config.cubin, image.cubin))
    throw std::invalid_argument("window_attention config and generated CUBIN image disagree");

  if (
    config.target_sm <= 0 || config.target_sm != config.cubin.target_sm
    || !cubin_supports_sm(config.cubin, config.target_sm))
  {
    throw std::invalid_argument(
      "embedded CUBIN does not support configured device SM" + std::to_string(config.target_sm));
  }
  if (
    image.is_bfloat16 != dtype_is_bfloat16(config.dtype) || image.head_dim != config.head_dim
    || image.n_queries != config.n_queries || image.n_keys != config.n_keys)
  {
    throw std::invalid_argument("window_attention config axes disagree with the generated CUBIN image");
  }
  if (!has_launch_metadata(image))
    throw std::invalid_argument("window_attention CUBIN has incompatible launch metadata");
}

void validate_rows(Tensor4View const& view, Tensor4View const& q, std::int32_t head_dim, char const* name)
{
  validate_tensor(view, name, 16);
  if (view.shape != q.shape)
    throw std::invalid_argument(std::string(name) + " must share q's shape [S, N, H, D]");
  if (view.shape[3] != head_dim)
    throw std::invalid_argument(std::string(name) + " head dim must be " + std::to_string(head_dim));
  if (view.strides[2] != head_dim)
    throw std::invalid_argument(std::string(name) + " heads must be adjacent: the head stride must be D");
  if (view.strides[0] % kStrideAlignment != 0 || view.strides[1] % kStrideAlignment != 0)
    throw std::invalid_argument(std::string(name) + " sample and row strides must be multiples of 8 elements");
}

void validate_launch(KernelConfig const& config, LaunchParams const& params)
{
  validate_config(config);
  embedded::CubinImage const& image = *config.embedded_image;
  validate_rows(params.q, params.q, image.head_dim, "q");
  validate_rows(params.k, params.q, image.head_dim, "k");
  validate_rows(params.v, params.q, image.head_dim, "v");
  validate_rows(params.output, params.q, image.head_dim, "output");
  validate_tensor(params.bias, "bias", 16);

  std::int32_t const S = params.q.shape[0];
  std::int32_t const N = params.q.shape[1];
  std::int32_t const H = params.q.shape[2];
  std::int32_t const B = params.bias.shape[0];
  std::int32_t const T = params.bias.shape[1];
  if (params.bias.shape[2] != H || params.bias.shape[3] != image.n_queries || params.bias.shape[4] != image.n_keys)
  {
    throw std::invalid_argument(
      "bias shape must be [B, T, H, " + std::to_string(image.n_queries) + ", " + std::to_string(image.n_keys) + "]");
  }
  if (N % image.n_queries != 0 || N / image.n_queries != T)
  {
    throw std::invalid_argument(
      "q must hold T windows of " + std::to_string(image.n_queries) + " rows, one per bias window");
  }
  if (S % B != 0)
    throw std::invalid_argument("bias batch items must divide the samples evenly");
  for (std::int64_t stride : params.bias.strides)
  {
    if (stride % kStrideAlignment != 0)
      throw std::invalid_argument("bias outer strides must be multiples of 8 elements");
  }
  if (S > kMaxGridYZ || T > kMaxGridYZ)
    throw std::invalid_argument("window_attention launches at most 65535 samples and 65535 windows");
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

  check(params.q.device, "q");
  check(params.k.device, "k");
  check(params.v.device, "v");
  check(params.bias.device, "bias");
  check(params.output.device, "output");
}

cute_tensor_s3_d4_t make_bias_descriptor(Tensor5View const& view)
{
  cute_tensor_s3_d4_t descriptor{};
  descriptor.data = static_cast<CUdeviceptr>(view.data);
  for (std::size_t index = 0; index < 3; ++index)
    descriptor.dynamic_shapes[index] = view.shape[index];
  for (std::size_t index = 0; index < view.strides.size(); ++index)
    descriptor.dynamic_strides[index] = view.strides[index];
  return descriptor;
}

abi::FastDivmod make_fast_divmod(std::int32_t divisor)
{
  if (divisor <= 0)
    throw std::invalid_argument("window_attention samples-per-item divisor must be positive");

  std::uint32_t const value = static_cast<std::uint32_t>(divisor);
  std::uint32_t log2 = 0;
  std::uint64_t power_of_two = 1;
  while (power_of_two < value)
  {
    power_of_two <<= 1;
    ++log2;
  }

  std::uint64_t const numerator = (power_of_two - value) << 32;
  std::uint64_t const multiplier = numerator / value + 1;
  if (multiplier > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("window_attention samples-per-item multiplier overflow");

  std::uint8_t const shift_right_1 = static_cast<std::uint8_t>(std::min(log2, 1U));
  std::uint8_t const shift_right_2 = static_cast<std::uint8_t>(log2 - shift_right_1);
  return abi::FastDivmod{
    divisor,
    static_cast<std::uint32_t>(multiplier),
    shift_right_1,
    shift_right_2,
  };
}

cubin_launch_config_t make_launch_config(embedded::CubinImage const& image, LaunchParams const& params)
{
  cubin_launch_config_t launch_config{};
  launch_config.grid_x = checked_u32(static_cast<std::uint64_t>(params.q.shape[2]), "window_attention grid.x");
  launch_config.grid_y = checked_u32(static_cast<std::uint64_t>(params.q.shape[0]), "window_attention grid.y");
  launch_config.grid_z = checked_u32(static_cast<std::uint64_t>(params.bias.shape[1]), "window_attention grid.z");
  launch_config.block_x = image.num_threads;
  launch_config.block_y = 1;
  launch_config.block_z = 1;
  launch_config.dynamic_smem_bytes = image.cubin.dynamic_smem_bytes;
  launch_config.stream = reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream));
  return launch_config;
}

void launch_sm80(cubin_kernel_t loaded, KernelConfig const& config, LaunchParams const& params)
{
  abi::SM80Params device_params{};
  device_params.q = make_tensor3_descriptor(params.q);
  device_params.k = make_tensor3_descriptor(params.k);
  device_params.v = make_tensor3_descriptor(params.v);
  device_params.bias = make_bias_descriptor(params.bias);
  device_params.output = make_tensor3_descriptor(params.output);
  device_params.softmax_scale_log2 = params.softmax_scale_log2;
  device_params.softmax_scale = params.softmax_scale;

  void* kernel_params[abi::kSM80ParameterCount]{};
  abi::pack_sm80_kernel_params(&device_params, kernel_params);

  cubin_launch_config_t const launch_config = make_launch_config(*config.embedded_image, params);
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(window_attention_sm80)");
}

void launch_sm90(cubin_kernel_t loaded, KernelConfig const& config, LaunchParams const& params)
{
  abi::SM90Params device_params{};
  device_params.q = static_cast<CUdeviceptr>(params.q.data);
  device_params.k = static_cast<CUdeviceptr>(params.k.data);
  device_params.q_sample_stride = params.q.strides[0];
  device_params.q_row_stride = params.q.strides[1];
  device_params.k_sample_stride = params.k.strides[0];
  device_params.k_row_stride = params.k.strides[1];
  device_params.rows = params.q.shape[1];
  device_params.samples_per_item = make_fast_divmod(params.q.shape[0] / params.bias.shape[0]);
  device_params.bias = static_cast<CUdeviceptr>(params.bias.data);
  device_params.bias_batch_stride = params.bias.strides[0];
  device_params.bias_window_stride = params.bias.strides[1];
  device_params.bias_head_stride = params.bias.strides[2];
  device_params.bias_query_stride = params.bias.strides[3];
  device_params.v = static_cast<CUdeviceptr>(params.v.data);
  device_params.v_sample_stride = params.v.strides[0];
  device_params.v_row_stride = params.v.strides[1];
  device_params.output = static_cast<CUdeviceptr>(params.output.data);
  device_params.output_sample_stride = params.output.strides[0];
  device_params.output_row_stride = params.output.strides[1];
  device_params.softmax_scale = params.softmax_scale;

  void* kernel_params[abi::kSM90ParameterCount]{};
  abi::pack_sm90_kernel_params(&device_params, kernel_params);

  cubin_launch_config_t const launch_config = make_launch_config(*config.embedded_image, params);
  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(window_attention_sm90)");
}

bool is_preload_compatible(embedded::CubinImage const& image, std::int32_t device_sm)
{
  return image.cubin.target_sm == device_sm && cubin_supports_sm(image.cubin, device_sm) && has_launch_metadata(image);
}

std::size_t preload_kernels(CUcontext context, std::int32_t device_sm)
{
  return preload_registry_kernels(context, device_sm, embedded::registry(), is_preload_compatible);
}

CubinPreloadRegistration const kPreloader{
  "window_attention",
  &preload_kernels,
};

} // namespace

KernelConfig make_kernel_config(
  std::int32_t target_sm, std::int32_t head_dim, std::int32_t n_queries, std::int32_t n_keys, DType dtype)
{
  embedded::CubinImage const* image = find_embedded_cubin(target_sm, head_dim, n_queries, n_keys, dtype);
  KernelConfig const config{target_sm, head_dim, n_queries, n_keys, dtype, image->cubin, image};
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
      "window_attention config selects SM" + std::to_string(config.target_sm) + " but current device is SM"
      + std::to_string(device_sm));
  }
  validate_operand_devices(params, cuda_device_for_context(context));

  cubin_kernel_t const loaded = load_embedded_kernel(context, config.cubin);
  if (is_sm90_abi(config.cubin))
    launch_sm90(loaded, config, params);
  else
    launch_sm80(loaded, config, params);
}

} // namespace bioir::cutedsl::window_attention
