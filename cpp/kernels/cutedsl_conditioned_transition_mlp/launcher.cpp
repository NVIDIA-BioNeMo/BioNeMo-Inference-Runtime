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

#include "conditioned_transition_mlp_registry.h"

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace bioir::cutedsl::conditioned_transition_mlp
{
namespace
{

constexpr char kSM100LaunchAbi[] = "conditioned_transition_mlp_sm100_v1";

/* Bare device pointers require explicit cross-device validation. */
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
  check(params.w1.device, "w1");
  check(params.w2.device, "w2");
  check(params.cond.device, "cond");
  check(params.output.device, "output");
}

/* x and the output are [rows, width] with 16-byte rows; a sample-major image reads them as whole contiguous
 * [atoms, width] blocks, one per sample.
 */
void validate_row_operand(
  Tensor2View const& view, std::int32_t rows, std::int32_t width, bool contiguous, char const* name)
{
  validate_tensor(view, name, 16);
  if (view.shape[0] != rows || view.shape[1] != width)
    throw std::invalid_argument(std::string(name) + " must be [rows, " + std::to_string(width) + "]");
  if (view.strides[0] < width || view.strides[0] % 8 != 0)
    throw std::invalid_argument(std::string(name) + " row stride must cover the width and keep 16-byte rows");
  if (contiguous && view.strides[0] != width)
    throw std::invalid_argument(std::string(name) + " must be contiguous for a sample-major image");
}

void validate_launch(KernelConfig const& config, LaunchParams const& params)
{
  if (config.cubin.data == nullptr || config.cubin.size == 0)
    throw std::invalid_argument("embedded CUBIN image must not be empty");
  if (config.cubin.variant_id == nullptr || config.cubin.variant_id[0] == '\0')
    throw std::invalid_argument("embedded CUBIN variant_id must not be empty");
  if (config.cubin.kernel_symbol == nullptr || config.cubin.kernel_symbol[0] == '\0')
    throw std::invalid_argument("kernel_symbol must not be empty");
  KernelSpec const& spec = config.spec;
  /* SM100 and SM103 both run the SM100 kernel's images. */
  if (
    config.cubin.kernel_sm != 100 || (spec.target_sm != 100 && spec.target_sm != 103)
    || config.cubin.launch_abi == nullptr || std::strcmp(config.cubin.launch_abi, kSM100LaunchAbi) != 0)
  {
    throw std::invalid_argument("conditioned transition MLP CUBIN has an incompatible launch ABI");
  }
  if (config.embedded_image == nullptr || !config.embedded_image->sm100.enabled)
    throw std::invalid_argument("conditioned transition MLP CUBIN has no Blackwell launch metadata");
  if (!cubin_supports_sm(config.cubin, spec.target_sm))
    throw std::invalid_argument(
      "embedded CUBIN does not support configured device SM" + std::to_string(spec.target_sm));
  if (
    spec.width <= 0 || spec.hidden <= 0 || spec.tile_m == 0 || spec.num_threads == 0
    || (spec.samples != 0) != (spec.tile_atoms != 0))
    throw std::invalid_argument("conditioned transition MLP CUBIN has invalid launch geometry");

  std::int32_t const rows = params.x.shape[0];
  std::int32_t const atoms = params.cond.shape[0];
  if (rows <= 0 || atoms <= 0 || rows % atoms != 0)
    throw std::invalid_argument("x needs a positive whole number of rows per cond row");
  if (spec.samples != 0 && std::int64_t{atoms} * spec.samples != rows)
    throw std::invalid_argument("a sample-major image needs rows == samples * atoms");
  bool const sample_major = spec.samples != 0;
  validate_row_operand(params.x, rows, spec.width, sample_major, "x");
  validate_row_operand(params.output, rows, spec.width, sample_major, "output");
  validate_tensor(params.cond, "cond", 16);
  if (
    params.cond.shape[1] != 3 * spec.width || params.cond.strides[0] < 3 * spec.width
    || params.cond.strides[0] % 8 != 0)
    throw std::invalid_argument("cond must be [atoms, " + std::to_string(3 * spec.width) + "] with 16-byte rows");
  /* The weights were compiled with static shapes and row strides. */
  std::int32_t const w1_rows = (spec.is_three_way ? 3 : 2) * spec.hidden;
  validate_tensor(params.w1, "w1", 16);
  if (params.w1.shape[0] != w1_rows || params.w1.shape[1] != spec.width || params.w1.strides[0] != spec.width)
  {
    throw std::invalid_argument(
      "w1 must be contiguous [" + std::to_string(w1_rows) + ", " + std::to_string(spec.width) + "]");
  }
  validate_tensor(params.w2, "w2", 16);
  if (params.w2.shape[0] != spec.width || params.w2.shape[1] != spec.hidden || params.w2.strides[0] != spec.hidden)
  {
    throw std::invalid_argument(
      "w2 must be contiguous [" + std::to_string(spec.width) + ", " + std::to_string(spec.hidden) + "]");
  }
}

/* A sample-major image reads x and writes the output as [atoms, width, samples]: sample s of atom a is row
 * s * atoms + a.
 */
TmaTensorSource sample_major_source(Tensor2View const& view, std::int32_t atoms, std::uint32_t samples)
{
  auto const width = static_cast<std::uint64_t>(view.shape[1]);
  auto const atom_count = static_cast<std::uint64_t>(atoms);
  return TmaTensorSource{view.data, {atom_count, width, samples, 0}, {width, 1, atom_count * width, 0}};
}

void launch_sm100(
  cubin_kernel_t loaded,
  KernelConfig const& config,
  LaunchParams const& params,
  CUcontext context,
  std::uint32_t smem_bytes)
{
  validate_operand_devices(params, cuda_device_for_context(context));
  auto const& metadata = config.embedded_image->sm100;
  if (metadata.cluster_dims[0] != 2 || metadata.cluster_dims[1] != 1 || metadata.cluster_dims[2] != 1)
    throw std::invalid_argument("conditioned transition MLP SM100 requires a two-CTA cluster");

  KernelSpec const& spec = config.spec;
  std::int32_t const atoms = params.cond.shape[0];
  abi::SM100Params device_params{};
  auto const width = static_cast<std::uint64_t>(spec.width);
  auto const hidden = static_cast<std::uint64_t>(spec.hidden);
  /* W1's logical rows group each 64-row hidden chunk's value and gate blocks, one per CTA of the pair; for the 3-way
   * SwiGLU, each 32-row half of a chunk's value, gate and second-value blocks, one half per CTA. */
  std::uint64_t const block_rows = spec.is_three_way ? 32 : 64;
  std::uint64_t const blocks = spec.is_three_way ? 3 : 2;
  TmaTensorSource const w1_source{
    params.w1.data, {block_rows, blocks, hidden / block_rows, width}, {width, hidden * width, block_rows * width, 1}};
  bool const sample_major = spec.samples != 0;
  TmaTensorSource const x_source
    = sample_major ? sample_major_source(params.x, atoms, spec.samples) : make_tma_tensor2_source(params.x, false);
  TmaTensorSource const output_source = sample_major ? sample_major_source(params.output, atoms, spec.samples)
                                                     : make_tma_tensor2_source(params.output, false);
  auto const dtype = tma_data_type(true);
  encode_tma_descriptor(device_params.x_tma, metadata.x, dtype, x_source, "x");
  encode_tma_descriptor(device_params.w1_tma, metadata.w1, dtype, w1_source, "w1");
  encode_tma_descriptor(device_params.w2_tma, metadata.w2, dtype, make_tma_tensor2_source(params.w2, false), "w2");
  encode_tma_descriptor(device_params.output_tma, metadata.output, dtype, output_source, "output");
  finalize_sm100_tma_atom(device_params.x_tma);
  finalize_sm100_tma_atom(device_params.w1_tma);
  finalize_sm100_tma_atom(device_params.w2_tma);
  finalize_sm100_tma_atom(device_params.output_tma);
  /* A rank-3 map's only dynamic extent is the atom count. */
  std::int32_t const tiled_extent = sample_major ? atoms : params.x.shape[0];
  device_params.x_coord = CoordTensorS1{{tiled_extent}};
  device_params.output_coord = CoordTensorS1{{tiled_extent}};
  device_params.residual = make_tensor2_s1_d1_descriptor(params.x);
  device_params.output = make_tensor2_s1_d1_descriptor(params.output);
  device_params.cond = make_tensor2_s1_d1_descriptor(params.cond);
  void* kernel_params[abi::kSM100ParameterCount]{};
  abi::pack_sm100_kernel_params(&device_params, kernel_params);

  /* Persistent CTA pairs stride over the tiles, each pair owning at least one. */
  auto const pair_tiles = sample_major ? ceil_div(static_cast<std::uint64_t>(atoms), spec.tile_atoms)
                                       : ceil_div(static_cast<std::uint64_t>(params.x.shape[0]), spec.tile_m);
  auto const sm_count = cuda_multiprocessor_count_for_context(context);
  auto const clusters = std::min(pair_tiles, static_cast<std::uint64_t>(std::max(1, sm_count / 2)));
  /* A staged image holds one tile's conditioning per pair; a second tile would trap the kernel. */
  if (spec.cond_stage && pair_tiles > clusters)
    throw std::invalid_argument("a staged conditioned transition image takes at most one tile per CTA pair");
  cubin_launch_config_t launch_config{};
  launch_config.grid_x = checked_u32(2 * clusters, "conditioned transition MLP SM100 grid.x");
  launch_config.grid_y = launch_config.grid_z = 1;
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
    "launch_cubin_kernel(conditioned_transition_mlp_sm100)");
}

KernelSpec make_kernel_spec(embedded::CubinImage const& image)
{
  return KernelSpec{
    image.cubin.target_sm,
    image.cubin.kernel_sm,
    image.width,
    image.hidden,
    image.is_three_way,
    image.cond_stage,
    image.samples,
    image.tile_m,
    image.tile_atoms,
    image.num_threads,
  };
}

embedded::CubinImage const& find_embedded_cubin(
  std::int32_t target_sm,
  DType dtype,
  std::int32_t width,
  std::int32_t hidden,
  bool is_three_way,
  bool cond_stage,
  std::int32_t samples)
{
  bool const is_bfloat16 = dtype == DType::kBFloat16;
  embedded::RegistryView const registry = embedded::registry();
  for (std::size_t index = 0; index < registry.count; ++index)
  {
    embedded::CubinImage const& image = registry.images[index];
    if (
      cubin_supports_sm(image.cubin, target_sm) && image.is_bfloat16 == is_bfloat16 && image.width == width
      && image.is_three_way == is_three_way && image.hidden == hidden && image.cond_stage == cond_stage
      && static_cast<std::int64_t>(image.samples) == samples)
      return image;
  }
  // A distinct type: a library without the image is not a malformed request.
  throw std::out_of_range(
    "No embedded conditioned transition MLP CUBIN for SM" + std::to_string(target_sm) + ", width="
    + std::to_string(width) + ", hidden=" + std::to_string(hidden) + ", three_way=" + (is_three_way ? "true" : "false")
    + ", cond_stage=" + (cond_stage ? "true" : "false") + ", samples=" + std::to_string(samples));
}

std::size_t preload_kernels(CUcontext context, std::int32_t device_sm)
{
  return preload_registry_kernels(context, device_sm, embedded::registry());
}

CubinPreloadRegistration const kPreloader{
  "conditioned_transition_mlp",
  &preload_kernels,
};

} // namespace

std::vector<KernelSpec> kernel_specs()
{
  return map_registry(embedded::registry(), make_kernel_spec);
}

KernelConfig make_kernel_config(
  std::int32_t target_sm,
  DType dtype,
  std::int32_t width,
  std::int32_t hidden,
  bool is_three_way,
  bool cond_stage,
  std::int32_t samples)
{
  embedded::CubinImage const& image
    = find_embedded_cubin(target_sm, dtype, width, hidden, is_three_way, cond_stage, samples);
  return KernelConfig{make_kernel_spec(image), dtype, image.cubin, &image};
}

void launch(KernelConfig const& config, LaunchParams const& params)
{
  validate_launch(config, params);
  CUcontext const context = current_cuda_context();
  std::int32_t const device_sm = cuda_sm_for_context(context);
  if (device_sm != config.spec.target_sm || !cubin_supports_sm(config.cubin, device_sm))
  {
    throw std::invalid_argument(
      "conditioned transition MLP config selects SM" + std::to_string(config.spec.target_sm)
      + " but current device is SM" + std::to_string(device_sm));
  }
  cubin_kernel_t const loaded = load_embedded_kernel(context, config.cubin);
  launch_sm100(loaded, config, params, context, dynamic_smem_bytes(config));
}

} // namespace bioir::cutedsl::conditioned_transition_mlp
