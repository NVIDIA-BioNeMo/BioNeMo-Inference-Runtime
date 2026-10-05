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

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace bioir::claude_kit_triangle_attention_sm90_d32
{
namespace
{

// Tile sizes of the attention kernel (Traits in kernel.cuh) that shape the workspace and the work-tile count.
constexpr std::int32_t kBlockM = 128;
constexpr std::int32_t kBlockN = 128;
constexpr std::int32_t kRows = 3;
constexpr std::int32_t kKeyColumns = 4;
constexpr std::size_t kSlotElems = 4096;
constexpr std::size_t kCounterBytes = 16; // the work-tile counter, padded to keep the staged lengths aligned

void check_cuda(cudaError_t status, char const* operation)
{
  if (status != cudaSuccess)
  {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

std::int32_t device_attribute(cudaDeviceAttr attribute, std::int32_t device)
{
  std::int32_t value = 0;
  check_cuda(cudaDeviceGetAttribute(&value, attribute, device), "cudaDeviceGetAttribute");
  return value;
}

std::size_t checked_product(std::initializer_list<std::size_t> factors, char const* name)
{
  std::size_t product = 1;
  for (std::size_t factor : factors)
  {
    if (factor != 0 && product > std::numeric_limits<std::size_t>::max() / factor)
    {
      throw std::overflow_error(std::string(name) + " size overflow");
    }
    product *= factor;
  }
  return product;
}

class AsyncAllocation
{
  public:
  AsyncAllocation(std::size_t bytes, cudaStream_t stream)
    : stream_(stream)
  {
    check_cuda(cudaMallocAsync(&data_, bytes, stream_), "cudaMallocAsync");
  }

  AsyncAllocation(AsyncAllocation const&) = delete;
  AsyncAllocation& operator=(AsyncAllocation const&) = delete;

  ~AsyncAllocation()
  {
    if (data_ != nullptr)
    {
      (void) cudaFreeAsync(data_, stream_);
    }
  }

  char* get() const
  {
    return static_cast<char*>(data_);
  }

  private:
  void* data_{};
  cudaStream_t stream_{};
};

template <std::size_t N>
void validate_pointer(cutedsl::TensorView<N> const& tensor, char const* name, std::uint64_t alignment)
{
  if (tensor.data == 0)
  {
    throw std::invalid_argument(std::string(name) + " pointer must not be null");
  }
  if (tensor.data % alignment != 0)
  {
    throw std::invalid_argument(std::string(name) + " pointer must be " + std::to_string(alignment) + "-byte aligned");
  }
}

template <std::size_t N>
void validate_device(cutedsl::TensorView<N> const& tensor, char const* name, std::int32_t current_device)
{
  if (tensor.device == cutedsl::kUnknownDevice)
  {
    throw std::invalid_argument(std::string(name) + " must be a CUDA tensor with a known device");
  }
  if (tensor.device != current_device)
  {
    throw std::invalid_argument(
      std::string(name) + " is on CUDA device " + std::to_string(tensor.device) + " but the launch targets device "
      + std::to_string(current_device));
  }
}

void validate_positive_strides(std::array<std::int64_t, 3> const& strides, char const* name)
{
  for (std::int64_t stride : strides)
  {
    if (stride <= 0)
    {
      throw std::invalid_argument(std::string(name) + " strides must be positive");
    }
  }
}

// TMA needs 16-byte aligned strides.
void validate_qkv_stride(std::array<std::int64_t, 3> const& strides, char const* name)
{
  validate_positive_strides(strides, name);
  for (std::int64_t stride : strides)
  {
    if (stride % 8 != 0)
    {
      throw std::invalid_argument(std::string(name) + " outer strides must be multiples of 8 bf16 elements");
    }
  }
}

void validate_launch(LaunchParams const& params, std::int32_t current_device)
{
  if (params.i_dim <= 0)
  {
    throw std::invalid_argument("i_dim must be positive");
  }
  if (!std::isfinite(params.softmax_scale) || params.softmax_scale <= 0.0F)
  {
    throw std::invalid_argument("softmax_scale must be finite and positive");
  }

  validate_pointer(params.q, "q", 16);
  validate_pointer(params.k, "k", 16);
  validate_pointer(params.v, "v", 16);
  validate_pointer(params.bias, "bias", 2);
  validate_pointer(params.actual_s_kv, "actual_s_kv", 4);
  validate_pointer(params.output, "output", 16);
  bool const wants_lse = params.lse.data != 0;
  if (wants_lse)
  {
    validate_pointer(params.lse, "lse", 4);
  }

  validate_device(params.q, "q", current_device);
  validate_device(params.k, "k", current_device);
  validate_device(params.v, "v", current_device);
  validate_device(params.bias, "bias", current_device);
  validate_device(params.actual_s_kv, "actual_s_kv", current_device);
  validate_device(params.output, "output", current_device);
  if (wants_lse)
  {
    validate_device(params.lse, "lse", current_device);
  }

  if (params.q.shape[0] <= 0 || params.q.shape[1] <= 0 || params.q.shape[2] <= 0)
  {
    throw std::invalid_argument("q dimensions must be positive");
  }
  if (params.q.shape[3] != 32)
  {
    throw std::invalid_argument("q head dimension must be 32");
  }
  if (params.q.shape[0] % params.i_dim != 0)
  {
    throw std::invalid_argument("q.shape[0] must be divisible by i_dim");
  }
  if (params.k.shape != params.q.shape || params.v.shape != params.q.shape)
  {
    throw std::invalid_argument("q, k, and v shapes must match");
  }
  if (params.output.shape != params.q.shape)
  {
    throw std::invalid_argument("output shape must match q");
  }

  std::int32_t const batch = params.q.shape[0] / params.i_dim;
  std::int32_t const seqlen = params.q.shape[1];
  std::int32_t const num_heads = params.q.shape[2];
  if (
    params.bias.shape[0] != batch || params.bias.shape[1] != num_heads || params.bias.shape[2] != seqlen
    || params.bias.shape[3] < seqlen)
  {
    throw std::invalid_argument("bias shape must be [B, H, S, S_padded]");
  }
  if (params.actual_s_kv.shape[0] != params.q.shape[0])
  {
    throw std::invalid_argument("actual_s_kv shape must be [B*I]");
  }
  if (
    wants_lse
    && (params.lse.shape[0] != params.q.shape[0] || params.lse.shape[1] != seqlen || params.lse.shape[2] != num_heads))
  {
    throw std::invalid_argument("lse shape must be [B*I, S, H]");
  }

  validate_qkv_stride(params.q.strides, "q");
  validate_qkv_stride(params.k.strides, "k");
  validate_qkv_stride(params.v.strides, "v");
  validate_qkv_stride(params.output.strides, "output");
  validate_positive_strides(params.bias.strides, "bias");
  if (wants_lse && (params.lse.strides[0] <= 0 || params.lse.strides[1] <= 0))
  {
    throw std::invalid_argument("lse strides must be positive");
  }
  // stage_bias launches a (key columns, query tiles, batch * heads) grid.
  std::int64_t const query_tiles = (std::int64_t(seqlen) + kBlockM - 1) / kBlockM;
  if (
    query_tiles > device_attribute(cudaDevAttrMaxGridDimY, current_device)
    || std::int64_t(batch) * num_heads > device_attribute(cudaDevAttrMaxGridDimZ, current_device))
  {
    throw std::invalid_argument("the bias staging grid exceeds the device's y or z grid limit");
  }
  std::int64_t const work_tiles
    = std::int64_t((seqlen + kBlockM - 1) / kBlockM) * ((params.i_dim + kRows - 1) / kRows) * batch * num_heads;
  if (work_tiles > std::numeric_limits<std::int32_t>::max())
  {
    throw std::invalid_argument("the work-tile count exceeds the int32 range");
  }
}

// Heads of one group run back to back and share DRAM fetches of the packed in_proj rows; the group is the largest
// divisor of the head count whose staged bias stays L2-resident.
std::int32_t head_group(std::int32_t num_heads, std::int32_t seqlen)
{
  constexpr std::int64_t kResidentBiasBytes = 26LL << 20;
  std::int64_t const padded = (std::int64_t(seqlen) + kBlockN - 1) / kBlockN * kBlockN;
  std::int64_t const head_bytes = padded * padded * std::int64_t(sizeof(float));
  std::int32_t group = num_heads;
  while (group > 1 && (num_heads % group != 0 || group * head_bytes > kResidentBiasBytes))
  {
    --group;
  }
  return group;
}

} // namespace

void launch(LaunchParams const& params)
{
  std::int32_t device = -1;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  std::int32_t const major = device_attribute(cudaDevAttrComputeCapabilityMajor, device);
  std::int32_t const minor = device_attribute(cudaDevAttrComputeCapabilityMinor, device);
  if (major != 9 || minor != 0)
  {
    throw std::invalid_argument(
      "claude_kit_triangle_attention_sm90_D32 requires an SM90 device; current capability is " + std::to_string(major)
      + "." + std::to_string(minor));
  }
  validate_launch(params, device);

  std::int32_t const batch = params.q.shape[0] / params.i_dim;
  std::int32_t const seqlen = params.q.shape[1];
  std::int32_t const num_heads = params.q.shape[2];
  std::size_t const query_tiles = (seqlen + kBlockM - 1) / kBlockM;
  std::size_t const key_columns = kKeyColumns * ((seqlen + kBlockN - 1) / kBlockN);

  // The workspace: the staged bias (a multiple of 16 KiB), the work-tile counter, then the staged length of each batch.
  std::size_t const bias_bytes = checked_product(
    {std::size_t(batch), std::size_t(num_heads), query_tiles, key_columns, kSlotElems, sizeof(float)}, "staged bias");
  std::size_t const length_bytes = std::size_t(batch) * sizeof(std::int32_t);
  if (bias_bytes > std::numeric_limits<std::size_t>::max() - kCounterBytes - length_bytes)
  {
    throw std::overflow_error("workspace size overflow");
  }
  cudaStream_t const stream = reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(params.stream));
  AsyncAllocation workspace(bias_bytes + kCounterBytes + length_bytes, stream);
  auto* const staged_bias = reinterpret_cast<float*>(workspace.get());
  auto* const tile_counter = reinterpret_cast<std::int32_t*>(workspace.get() + bias_bytes);
  auto* const staged_length = reinterpret_cast<std::int32_t*>(workspace.get() + bias_bytes + kCounterBytes);
  auto const* const actual_s_kv
    = reinterpret_cast<std::int32_t const*>(static_cast<std::uintptr_t>(params.actual_s_kv.data));

  BiasStageParams bias_params;
  bias_params.bias = reinterpret_cast<void const*>(static_cast<std::uintptr_t>(params.bias.data));
  bias_params.actual_s_kv = actual_s_kv;
  bias_params.staged_bias = staged_bias;
  bias_params.staged_length = staged_length;
  bias_params.tile_counter = tile_counter;
  bias_params.stride_batch = params.bias.strides[0];
  bias_params.stride_head = params.bias.strides[1];
  bias_params.stride_query = params.bias.strides[2];
  bias_params.batch = batch;
  bias_params.i_dim = params.i_dim;
  bias_params.num_heads = num_heads;
  bias_params.seqlen = seqlen;
  bias_params.inverse_softmax_scale = 1.0F / params.softmax_scale;
  bias_params.stream = params.stream;
  stage_bias(bias_params);

  KernelLaunchParams kernel_params;
  kernel_params.q = reinterpret_cast<void const*>(static_cast<std::uintptr_t>(params.q.data));
  kernel_params.k = reinterpret_cast<void const*>(static_cast<std::uintptr_t>(params.k.data));
  kernel_params.v = reinterpret_cast<void const*>(static_cast<std::uintptr_t>(params.v.data));
  kernel_params.staged_bias = staged_bias;
  kernel_params.staged_length = staged_length;
  kernel_params.actual_s_kv = actual_s_kv;
  kernel_params.output = reinterpret_cast<void*>(static_cast<std::uintptr_t>(params.output.data));
  kernel_params.lse = reinterpret_cast<float*>(static_cast<std::uintptr_t>(params.lse.data));
  kernel_params.tile_counter = tile_counter;
  kernel_params.q_stride_bi = params.q.strides[0];
  kernel_params.q_stride_s = params.q.strides[1];
  kernel_params.q_stride_h = params.q.strides[2];
  kernel_params.k_stride_bi = params.k.strides[0];
  kernel_params.k_stride_s = params.k.strides[1];
  kernel_params.k_stride_h = params.k.strides[2];
  kernel_params.v_stride_bi = params.v.strides[0];
  kernel_params.v_stride_s = params.v.strides[1];
  kernel_params.v_stride_h = params.v.strides[2];
  kernel_params.output_stride_bi = params.output.strides[0];
  kernel_params.output_stride_s = params.output.strides[1];
  kernel_params.output_stride_h = params.output.strides[2];
  kernel_params.lse_stride_bi = params.lse.strides[0];
  kernel_params.lse_stride_s = params.lse.strides[1];
  kernel_params.batch = batch;
  kernel_params.i_dim = params.i_dim;
  kernel_params.num_heads = num_heads;
  kernel_params.seqlen = seqlen;
  kernel_params.head_group = head_group(num_heads, seqlen);
  kernel_params.sm_count = device_attribute(cudaDevAttrMultiProcessorCount, device);
  kernel_params.softmax_scale = params.softmax_scale;
  kernel_params.stream = params.stream;
  launch_attention(kernel_params);
}

} // namespace bioir::claude_kit_triangle_attention_sm90_d32
