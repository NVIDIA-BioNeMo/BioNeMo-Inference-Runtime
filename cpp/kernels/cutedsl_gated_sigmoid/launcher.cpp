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

namespace trtbnm::cutedsl::gated_sigmoid
{
namespace
{

constexpr char kSM80LaunchAbi[] = "gated_sigmoid_sm80";

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
    throw std::invalid_argument("gated-sigmoid CUBIN has an incompatible launch ABI");
  }
  if (!cubin_supports_sm(config.cubin, config.spec.target_sm))
    throw std::invalid_argument("gated-sigmoid CUBIN does not support its configured target SM");
  if (config.spec.num_threads <= 0)
    throw std::invalid_argument("gated-sigmoid spec has a non-positive thread count");
  if (config.spec.m_block_size <= 0 || config.spec.n_block_size <= 0)
    throw std::invalid_argument("gated-sigmoid spec has a non-positive tile");
  if (config.spec.raster_factor <= 0)
    throw std::invalid_argument("gated-sigmoid spec has a non-positive rasterization factor");

  validate_tensor(params.s, "s", 16);
  validate_tensor(params.weight, "weight", 16);
  validate_tensor(params.mha_out, "mha_out", 16);
  validate_tensor(params.output, "output", 16);
  if (config.has_bias)
    validate_tensor(params.bias, "bias", 16);

  std::int32_t const K = params.s.shape[1];
  std::int32_t const N = params.weight.shape[0];
  if (params.weight.shape[1] != K)
    throw std::invalid_argument("weight inner extent must match s's inner extent (K)");
  if (params.mha_out.shape[1] != N || params.output.shape[1] != N)
    throw std::invalid_argument("mha_out and output inner extents must equal weight.shape[0] (N)");
  if (params.output.shape[0] != params.mha_out.shape[0])
    throw std::invalid_argument("output and mha_out must have the same row count");
  if (config.has_bias && params.bias.shape[0] != N)
    throw std::invalid_argument("bias extent must equal N");

  if (params.mult <= 0)
    throw std::invalid_argument("mult must be positive");
  if (params.inner <= 0)
    throw std::invalid_argument("inner must be positive");
  if (params.s.shape[0] % params.inner != 0)
    throw std::invalid_argument("s row count must be divisible by inner");
  /* The epilogue fans each gate tile out to `mult` output tiles, so the output
   * must have exactly that many times as many rows as the gate.
   */
  if (static_cast<std::int64_t>(params.s.shape[0]) * params.mult != params.mha_out.shape[0])
    throw std::invalid_argument("mha_out row count must equal s row count times mult");
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

  check(params.s.device, "s");
  check(params.weight.device, "weight");
  check(params.mha_out.device, "mha_out");
  check(params.output.device, "output");
  if (config.has_bias)
    check(params.bias.device, "bias");
}

/* Reproduces the CuTeDSL host launcher's grid arithmetic exactly:
 *
 *   inner_tiles = ceil_div(inner, bM)
 *   batches     = s.rows / inner
 *   grid        = (batches * inner_tiles * raster, ceil_div(ceil_div(N, bN), raster), 1)
 */
cubin_launch_config_t
make_launch_config(KernelSpec const& spec, LaunchParams const& params, std::uint32_t smem_bytes, CUstream stream)
{
  std::uint64_t const inner_tiles
    = ceil_div(static_cast<std::uint64_t>(params.inner), static_cast<std::uint64_t>(spec.m_block_size));
  std::uint64_t const batches
    = static_cast<std::uint64_t>(params.s.shape[0]) / static_cast<std::uint64_t>(params.inner);
  std::uint64_t const grid_m = batches * inner_tiles;
  std::uint64_t const grid_n
    = ceil_div(static_cast<std::uint64_t>(params.output.shape[1]), static_cast<std::uint64_t>(spec.n_block_size));
  std::uint64_t const raster = static_cast<std::uint64_t>(spec.raster_factor);

  cubin_launch_config_t config = {0};
  config.grid_x = checked_u32(grid_m * raster, "gated-sigmoid grid.x");
  config.grid_y = checked_u32(ceil_div(grid_n, raster), "gated-sigmoid grid.y");
  config.grid_z = 1;
  config.block_x = static_cast<std::uint32_t>(spec.num_threads);
  config.block_y = 1;
  config.block_z = 1;
  config.dynamic_smem_bytes = smem_bytes;
  config.stream = stream;
  /* Cluster dimensions stay zero: this kernel launches through the plain
   * cuLaunchKernel path on every target SM, including sm_90a.
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
  device_params.s = make_tensor2_s2_d1_descriptor(params.s);
  device_params.weight = make_tensor2_s2_d1_descriptor(params.weight);
  if (config.has_bias)
    device_params.bias = make_tensor1_descriptor(params.bias);
  device_params.mha_out = make_tensor2_s2_d1_descriptor(params.mha_out);
  device_params.output = make_tensor2_s2_d1_descriptor(params.output);
  device_params.rasterization_factor = config.spec.raster_factor;
  device_params.mult = params.mult;
  device_params.inner = params.inner;

  void* kernel_params[abi::kSM80MaxParameterCount];
  std::size_t const count = abi::pack_sm80_kernel_params(&device_params, kernel_params, config.has_bias);
  std::size_t const expected = config.has_bias ? abi::kSM80ParameterCountWithBias : abi::kSM80ParameterCountNoBias;
  if (count != expected)
    throw std::invalid_argument("gated-sigmoid packed an unexpected number of kernel parameters");

  cubin_launch_config_t const launch_config = make_launch_config(
    config.spec, params, smem_bytes, reinterpret_cast<CUstream>(static_cast<std::uintptr_t>(params.stream)));

  check_cuda_driver(
    launch_cubin_kernel(loaded, &launch_config, kernel_params, nullptr), "launch_cubin_kernel(gated_sigmoid_sm80)");
}

embedded::CubinImage const& find_embedded_cubin(
  std::int32_t target_sm,
  DType dtype,
  bool has_bias,
  std::int32_t m_block_size,
  std::int32_t n_block_size,
  std::int32_t k_block_size,
  std::int32_t num_stages,
  std::int32_t raster_factor,
  std::int32_t atom_layout_m,
  std::int32_t atom_layout_n,
  std::int32_t atom_layout_k)
{
  bool const is_bfloat16 = dtype == DType::kBFloat16;
  for (std::size_t index = 0; index < embedded::kCubinCount; ++index)
  {
    embedded::CubinImage const& image = embedded::kCubins[index];
    if (
      cubin_supports_sm(image.cubin, target_sm) && image.is_bfloat16 == is_bfloat16 && image.has_bias == has_bias
      && image.m_block_size == m_block_size && image.n_block_size == n_block_size && image.k_block_size == k_block_size
      && image.num_stages == num_stages && image.raster_factor == raster_factor
      && image.atom_layout_mnk[0] == atom_layout_m && image.atom_layout_mnk[1] == atom_layout_n
      && image.atom_layout_mnk[2] == atom_layout_k)
    {
      return image;
    }
  }

  throw std::invalid_argument(
    "No embedded gated-sigmoid CUBIN for SM" + std::to_string(target_sm) + ", bf16=" + std::to_string(is_bfloat16)
    + ", bias=" + std::to_string(has_bias) + ", tile=" + std::to_string(m_block_size) + "x"
    + std::to_string(n_block_size) + "x" + std::to_string(k_block_size) + ", stages=" + std::to_string(num_stages)
    + ", raster=" + std::to_string(raster_factor) + ", atom_layout=" + std::to_string(atom_layout_m) + "x"
    + std::to_string(atom_layout_n) + "x" + std::to_string(atom_layout_k));
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
  "gated_sigmoid",
  &preload_kernels,
};

} // namespace

KernelConfig make_kernel_config(
  std::int32_t target_sm,
  DType dtype,
  bool has_bias,
  std::int32_t m_block_size,
  std::int32_t n_block_size,
  std::int32_t k_block_size,
  std::int32_t num_stages,
  std::int32_t raster_factor,
  std::int32_t atom_layout_m,
  std::int32_t atom_layout_n,
  std::int32_t atom_layout_k)
{
  embedded::CubinImage const& image = find_embedded_cubin(
    target_sm,
    dtype,
    has_bias,
    m_block_size,
    n_block_size,
    k_block_size,
    num_stages,
    raster_factor,
    atom_layout_m,
    atom_layout_n,
    atom_layout_k);
  KernelSpec const spec{
    target_sm,
    image.m_block_size,
    image.n_block_size,
    image.k_block_size,
    image.num_stages,
    image.raster_factor,
    {image.atom_layout_mnk[0], image.atom_layout_mnk[1], image.atom_layout_mnk[2]},
    image.num_threads,
  };
  return KernelConfig{
    spec,
    dtype,
    has_bias,
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
      "gated-sigmoid config selects SM" + std::to_string(config.spec.target_sm) + " but current device is SM"
      + std::to_string(device_sm));
  }

  cubin_kernel_t const loaded = load_embedded_kernel(context, config.cubin);
  launch_sm80(loaded, config, params, context, dynamic_smem_bytes(config));
}

} // namespace trtbnm::cutedsl::gated_sigmoid
