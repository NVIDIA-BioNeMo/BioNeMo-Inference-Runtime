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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Anthropic, PBC
 * Modified by NVIDIA Corporation and affiliates.
 *
 * Adapted from the Uplifting Biomolecular Modeling M1 bias staging. NOTICE.md lists the changes.
 */

#include "launcher.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace bioir::claude_kit_triangle_attention_sm90_d32
{
namespace
{

void check_cuda(cudaError_t status, char const* operation)
{
  if (status != cudaSuccess)
  {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

// The staged length of each batch: its longest row, actual_s_kv clamped to [0, seqlen]. Keys at or past it stage as
// -inf, so rows of that length need no mask code. One block per batch.
__global__ void stage_length_kernel(
  std::int32_t const* __restrict__ actual_s_kv,
  std::int32_t i_dim,
  std::int32_t seqlen,
  std::int32_t* __restrict__ staged_length)
{
  std::int32_t longest = 0;
  for (std::int32_t row = threadIdx.x; row < i_dim; row += blockDim.x)
  {
    longest = max(longest, min(max(actual_s_kv[std::int64_t(blockIdx.x) * i_dim + row], 0), seqlen));
  }
  for (std::int32_t offset = 16; offset > 0; offset /= 2)
  {
    longest = max(longest, __shfl_xor_sync(0xffffffffu, longest, offset));
  }
  __shared__ std::int32_t warp_longest[32];
  std::int32_t const warp = threadIdx.x / 32;
  std::int32_t const lane = threadIdx.x % 32;
  if (lane == 0)
  {
    warp_longest[warp] = longest;
  }
  __syncthreads();
  if (warp == 0)
  {
    longest = lane < std::int32_t(blockDim.x / 32) ? warp_longest[lane] : 0;
    for (std::int32_t offset = 16; offset > 0; offset /= 2)
    {
      longest = max(longest, __shfl_xor_sync(0xffffffffu, longest, offset));
    }
    if (lane == 0)
    {
      staged_length[blockIdx.x] = longest;
    }
  }
}

// Stage bias / scale in the attention kernel's fragment order: block (column, query tile, b * H + h) writes one slot,
// value index ((half * 4 + u) * 128 + thread) * 4 + element. Block (0, 0, 0) also resets the work-tile counter.
__global__ void stage_bias_kernel(
  __nv_bfloat16 const* __restrict__ source,
  std::int64_t stride_batch,
  std::int64_t stride_head,
  std::int64_t stride_query,
  std::int32_t num_heads,
  std::int32_t seqlen,
  std::int32_t const* __restrict__ staged_length,
  float inverse_softmax_scale,
  float* __restrict__ staged,
  std::int32_t* __restrict__ tile_counter)
{
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
  {
    *tile_counter = 0;
  }
  std::int32_t const key_tile = blockIdx.x >> 2;
  std::int32_t const column = blockIdx.x & 3;
  std::int32_t const query_tile = blockIdx.y;
  std::int32_t const batch_head = blockIdx.z;
  std::int32_t const batch = batch_head / num_heads;
  std::int32_t const head = batch_head % num_heads;
  std::int32_t const key_columns = gridDim.x;
  std::int32_t const query_tiles = gridDim.y;
  float* output = staged + ((std::int64_t(batch_head) * query_tiles + query_tile) * key_columns + blockIdx.x) * 4096;
  __nv_bfloat16 const* base = source + std::int64_t(batch) * stride_batch + std::int64_t(head) * stride_head;
  std::int32_t const key_end = staged_length[batch];

  for (std::int32_t index = threadIdx.x; index < 4096; index += blockDim.x)
  {
    std::int32_t const element = index & 3;
    std::int32_t const thread = (index >> 2) & 127;
    std::int32_t const half_and_u = index >> 9;
    std::int32_t const u = half_and_u & 3;
    std::int32_t const half = half_and_u >> 2;
    std::int32_t const query_in_tile = 64 * half + 16 * (thread >> 5) + ((thread & 31) >> 2) + 8 * (element >> 1);
    std::int32_t const key_in_tile = 32 * column + 8 * u + 2 * (thread & 3) + (element & 1);
    std::int32_t const query = query_tile * 128 + query_in_tile;
    std::int32_t const key = key_tile * 128 + key_in_tile;

    // Dead keys stage -inf; queries past the sequence end are never stored and stage zeros.
    bool const key_live = key < key_end;
    float value = key_live ? 0.0F : -INFINITY;
    if (key_live && query < seqlen)
    {
      value = __bfloat162float(base[std::int64_t(query) * stride_query + key]) * inverse_softmax_scale;
    }
    output[index] = value;
  }
}

} // namespace

void stage_bias(BiasStageParams const& params)
{
  std::int32_t const query_tiles = (params.seqlen + 127) / 128;
  std::int32_t const key_columns = 4 * ((params.seqlen + 127) / 128);
  cudaStream_t const stream = reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(params.stream));
  stage_length_kernel<<<params.batch, 256, 0, stream>>>(
    params.actual_s_kv, params.i_dim, params.seqlen, params.staged_length);
  check_cuda(cudaGetLastError(), "stage_length_kernel launch");
  dim3 const grid(key_columns, query_tiles, params.batch * params.num_heads);
  stage_bias_kernel<<<grid, 256, 0, stream>>>(
    static_cast<__nv_bfloat16 const*>(params.bias),
    params.stride_batch,
    params.stride_head,
    params.stride_query,
    params.num_heads,
    params.seqlen,
    params.staged_length,
    params.inverse_softmax_scale,
    params.staged_bias,
    params.tile_counter);
  check_cuda(cudaGetLastError(), "stage_bias_kernel launch");
}

} // namespace bioir::claude_kit_triangle_attention_sm90_d32
