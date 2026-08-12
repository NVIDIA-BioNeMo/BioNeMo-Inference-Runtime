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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace trtbnm::cutedsl::adaln_layernorm_sigmoid
{
namespace
{

constexpr char kSM80LaunchAbi[] = "adaln_layernorm_sigmoid_sm80";
constexpr char kSM90LaunchAbi[] = "adaln_layernorm_sigmoid_sm90";

void validate_launch(KernelConfig const& config, LaunchParams const& params)
{
  if (config.cubin.data == nullptr || config.cubin.size == 0)
    throw std::invalid_argument("embedded CUBIN image must not be empty");
  if (config.cubin.variant_id == nullptr || config.cubin.variant_id[0] == '\0')
    throw std::invalid_argument("embedded CUBIN variant_id must not be empty");
  if (config.cubin.kernel_symbol == nullptr || config.cubin.kernel_symbol[0] == '\0')
    throw std::invalid_argument("kernel_symbol must not be empty");

  /* The kernel lowers a cluster reduction above sm_90 from the same source, so
   * the ABI generation tracks the target rather than the class.
   */
  char const* const expected_abi = config.cubin.kernel_sm >= 90 ? kSM90LaunchAbi : kSM80LaunchAbi;
  if (config.cubin.launch_abi == nullptr || std::strcmp(config.cubin.launch_abi, expected_abi) != 0)
    throw std::invalid_argument("AdaLN CUBIN has an incompatible launch ABI");
  if (!cubin_supports_sm(config.cubin, config.spec.target_sm))
    throw std::invalid_argument("AdaLN CUBIN does not support its configured target SM");

  /* Every shipped N resolves to a single-CTA cluster, so this launcher stays on
   * the plain cuLaunchKernel path. A cluster-needing payload launched without
   * cluster dimensions would deadlock in cluster.sync rather than fail, so
   * reject it rather than let it default to zero.
   */
  if (config.spec.cluster_n != 1)
    throw std::invalid_argument("AdaLN CUBIN needs a cluster launch this launcher does not implement");
  if (config.spec.num_threads <= 0 || config.spec.threads_per_row <= 0)
    throw std::invalid_argument("AdaLN spec has a non-positive thread geometry");
  if (config.spec.feature_dim <= 0)
    throw std::invalid_argument("AdaLN spec has a non-positive feature dimension");

  validate_tensor(params.x, "x", 4);
  validate_tensor(params.s_scale, "s_scale", 4);
  validate_tensor(params.s_bias, "s_bias", 4);
  validate_tensor(params.output, "output", 4);

  /* N is compiled in, so a caller whose feature dimension differs has been
   * matched to the wrong payload; that is a selection bug, not a shape error.
   */
  std::int32_t const feature_dim = config.spec.feature_dim;
  for (auto const& [view, name] : {
         std::pair<Tensor2View const&, char const*>{params.x, "x"},
         std::pair<Tensor2View const&, char const*>{params.s_scale, "s_scale"},
         std::pair<Tensor2View const&, char const*>{params.s_bias, "s_bias"},
         std::pair<Tensor2View const&, char const*>{params.output, "output"},
       })
  {
    if (view.shape[1] != feature_dim)
    {
      throw std::invalid_argument(
        std::string(name) + " has feature dimension " + std::to_string(view.shape[1])
        + " but the CUBIN compiled N=" + std::to_string(feature_dim));
    }
  }

  if (params.output.shape[0] != params.x.shape[0])
    throw std::invalid_argument("output and x must have the same row count");
  if (params.s_bias.shape[0] != params.s_scale.shape[0])
    throw std::invalid_argument("s_scale and s_bias must have the same row count");
  if (params.mult <= 0)
    throw std::invalid_argument("mult must be positive");
  if (params.inner <= 0)
    throw std::invalid_argument("inner must be positive");
  /* x carries `mult` times as many rows as the gate operands. */
  if (static_cast<std::int64_t>(params.s_scale.shape[0]) * params.mult != params.x.shape[0])
    throw std::invalid_argument("x row count must equal s_scale row count times mult");
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

  check(params.x.device, "x");
  check(params.s_scale.device, "s_scale");
  check(params.s_bias.device, "s_bias");
  check(params.output.device, "output");
}

/* Reproduces the CuTeDSL host launcher's grid exactly:
 *
 *   grid = (ceil_div(M, tiler_mn[0]), cluster_n, 1)
 *
 * with tiler_mn[0] == num_threads / threads_per_row.
 */
cubin_launch_config_t
make_launch_config(KernelSpec const& spec, LaunchParams const& params, std::uint32_t smem_bytes, CUstream stream)
{
  std::uint64_t const rows = static_cast<std::uint64_t>(params.x.shape[0]);
  std::uint64_t const tile_rows = static_cast<std::uint64_t>(rows_per_block(spec));

  cubin_launch_config_t config = {0};
  config.grid_x = checked_u32(ceil_div(rows, tile_rows), "AdaLN grid.x");
  config.grid_y = checked_u32(static_cast<std::uint64_t>(spec.cluster_n), "AdaLN grid.y");
  config.grid_z = 1;
  config.block_x = static_cast<std::uint32_t>(spec.num_threads);
  config.block_y = 1;
  config.block_z = 1;
  config.dynamic_smem_bytes = smem_bytes;
  config.stream = stream;
  /* Cluster dimensions stay zero: validate_launch has already rejected any
   * payload whose cluster_n is not 1.
   */
  return config;
}

embedded::CubinImage const& find_embedded_cubin(
  std::int32_t target_sm, DType dtype, std::int32_t feature_dim, std::int32_t threads_per_row, std::int32_t num_threads)
{
  for (std::size_t index = 0; index < embedded::kCubinCount; ++index)
  {
    embedded::CubinImage const& image = embedded::kCubins[index];
    if (
      cubin_supports_sm(image.cubin, target_sm) && image.dtype == static_cast<std::uint8_t>(dtype)
      && image.feature_dim == feature_dim && image.threads_per_row == threads_per_row
      && image.num_threads == num_threads)
    {
      return image;
    }
  }

  throw std::invalid_argument(
    "No embedded AdaLN CUBIN for SM" + std::to_string(target_sm) + ", dtype=" + std::to_string(static_cast<int>(dtype))
    + ", N=" + std::to_string(feature_dim) + ", threads_per_row=" + std::to_string(threads_per_row)
    + ", num_threads=" + std::to_string(num_threads)
    + ". N is compiled into the kernel, so an unshipped feature dimension has no payload.");
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
  "adaln_layernorm_sigmoid",
  &preload_kernels,
};

} // namespace

KernelConfig make_kernel_config(
  std::int32_t target_sm, DType dtype, std::int32_t feature_dim, std::int32_t threads_per_row, std::int32_t num_threads)
{
  embedded::CubinImage const& image = find_embedded_cubin(target_sm, dtype, feature_dim, threads_per_row, num_threads);
  KernelSpec const spec{
    target_sm,
    image.feature_dim,
    image.threads_per_row,
    image.num_threads,
    image.cluster_n,
  };
  return KernelConfig{
    spec,
    dtype,
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
      "AdaLN config selects SM" + std::to_string(config.spec.target_sm) + " but current device is SM"
      + std::to_string(device_sm));
  }
  validate_operand_devices(params, cuda_device_for_context(context));

  cubin_kernel_t const loaded = load_embedded_kernel(context, config.cubin);
  std::uint32_t const smem_bytes = dynamic_smem_bytes(config);

  abi::Params device_params{};
  device_params.x = make_tensor2_s1_d1_descriptor(params.x);
  device_params.s_scale = make_tensor2_s1_d1_descriptor(params.s_scale);
  device_params.s_bias = make_tensor2_s1_d1_descriptor(params.s_bias);
  device_params.output = make_tensor2_s1_d1_descriptor(params.output);
  device_params.eps = params.eps;
  device_params.mult = params.mult;
  device_params.inner = params.inner;

  void* kernel_params[abi::kParameterCount];
  abi::pack_kernel_params(&device_params, kernel_params);

  cubin_launch_config_t const launch_config = make_launch_config(
    config.spec, params, smem_bytes, reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream)));

  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr),
    "launch_cubin_kernel(adaln_layernorm_sigmoid)");
}

} // namespace trtbnm::cutedsl::adaln_layernorm_sigmoid
