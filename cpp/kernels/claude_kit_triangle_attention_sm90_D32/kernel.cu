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
 * Adapted from the Uplifting Biomolecular Modeling M1 launcher (launch_m1.cuh). NOTICE.md lists the changes.
 */

#include "kernel.cuh"
#include "launcher.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace bioir::claude_kit_triangle_attention_sm90_d32
{
namespace
{

using Traits = detail::Traits;

void check_cuda(cudaError_t status, char const* operation)
{
  if (status != cudaSuccess)
  {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

// Raise the kernel's dynamic shared-memory limit, once per device.
void configure_kernel(std::size_t smem_bytes)
{
  static std::mutex mutex;
  static std::vector<bool> configured;
  std::int32_t device = -1;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  if (device < 0)
  {
    throw std::runtime_error("cudaGetDevice returned a negative device ordinal");
  }
  std::lock_guard<std::mutex> const lock(mutex);
  if (static_cast<std::size_t>(device) >= configured.size())
  {
    configured.resize(static_cast<std::size_t>(device) + 1, false);
  }
  if (!configured[device])
  {
    check_cuda(
      cudaFuncSetAttribute(
        reinterpret_cast<void const*>(detail::triangle_attention_kernel<Traits>),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)),
      "cudaFuncSetAttribute(triangle_attention_kernel)");
    configured[device] = true;
  }
}

} // namespace

void launch_attention(KernelLaunchParams const& args)
{
  using namespace cute;
  using Element = Traits::Element;

  // Q, K, V and O as (S, D, H, I, B) views of [B * I, S, H, D] tensors with unit D stride.
  Traits::ShapeQK const shape_qk = make_shape(args.seqlen, Traits::kHeadDim, args.num_heads, args.i_dim, args.batch);
  auto const view = [&](auto* data, std::int64_t stride_bi, std::int64_t stride_s, std::int64_t stride_h)
  {
    Traits::StrideQK const stride{stride_s, _1{}, stride_h, stride_bi, stride_bi * args.i_dim};
    return make_tensor(make_gmem_ptr(data), shape_qk, stride);
  };
  Tensor const q = view(static_cast<Element const*>(args.q), args.q_stride_bi, args.q_stride_s, args.q_stride_h);
  Tensor const k = view(static_cast<Element const*>(args.k), args.k_stride_bi, args.k_stride_s, args.k_stride_h);
  Tensor const v = view(static_cast<Element const*>(args.v), args.v_stride_bi, args.v_stride_s, args.v_stride_h);
  Tensor const o
    = view(static_cast<Element*>(args.output), args.output_stride_bi, args.output_stride_s, args.output_stride_h);

  std::int32_t const n_qtiles = (args.seqlen + Traits::kBlockM - 1) / Traits::kBlockM;
  std::int32_t const n_key_cols = Traits::kBiasSlots * ((args.seqlen + Traits::kBlockN - 1) / Traits::kBlockN);
  std::int32_t const n_row_groups = (args.i_dim + Traits::kRows - 1) / Traits::kRows;
  std::int64_t const n_work = std::int64_t(n_qtiles) * n_row_groups * args.batch * args.num_heads;
  // The staged bias as (256, 8, half slot, query tile, b * H + h); see stage_bias.
  Traits::ShapeBias const shape_bias = make_shape(256, 8, 2 * n_key_cols, n_qtiles, args.batch * args.num_heads);
  Traits::StrideBias const stride_bias{
    _1{},
    _256{},
    std::int64_t(Traits::kBiasHalfElems),
    std::int64_t(Traits::kBiasSlotElems) * n_key_cols,
    std::int64_t(Traits::kBiasSlotElems) * n_key_cols * n_qtiles,
  };
  Tensor const staged_bias = make_tensor(make_gmem_ptr(args.staged_bias), shape_bias, stride_bias);

  Traits::Params params{};
  params.tma_q = make_tma_copy(
    SM90_TMA_LOAD{},
    q,
    take<0, 2>(Traits::SmemLayoutQ{}),
    make_shape(Int<Traits::kBlockM>{}, Int<Traits::kHeadDim>{}),
    _1{});
  params.tma_k = make_tma_copy(
    SM90_TMA_LOAD{},
    k,
    take<0, 2>(Traits::SmemLayoutK{}),
    make_shape(Int<Traits::kBlockN>{}, Int<Traits::kHeadDim>{}),
    _1{});
  params.tma_v = make_tma_copy(
    SM90_TMA_LOAD{},
    v,
    take<0, 2>(Traits::SmemLayoutV{}),
    make_shape(Int<Traits::kBlockN>{}, Int<Traits::kHeadDim>{}),
    _1{});
  params.tma_bias
    = make_tma_copy(SM90_TMA_LOAD{}, staged_bias, Traits::SmemLayoutBiasHalf{}, make_shape(_256{}, _8{}), _1{});
  params.tma_o = make_tma_copy(
    SM90_TMA_STORE{},
    o,
    take<0, 2>(Traits::SmemLayoutQ{}),
    make_shape(Int<Traits::kBlockM>{}, Int<Traits::kHeadDim>{}),
    _1{});
  params.shape_qk = shape_qk;
  params.shape_bias = shape_bias;
  params.seqlen = args.seqlen;
  params.i_dim = args.i_dim;
  params.num_heads = args.num_heads;
  params.n_qtiles = n_qtiles;
  params.n_row_groups = n_row_groups;
  params.head_group = args.head_group;
  params.n_work = static_cast<std::int32_t>(n_work);
  params.tile_counter = args.tile_counter;
  params.softmax_scale = args.softmax_scale;
  params.staged_length = args.staged_length;
  params.actual_s_kv = args.actual_s_kv;
  params.lse = args.lse;
  params.lse_stride_bi = args.lse_stride_bi;
  params.lse_stride_s = args.lse_stride_s;

  std::size_t const smem_bytes = attention_dynamic_smem_bytes();
  configure_kernel(smem_bytes);
  // Persistent: at most one CTA per SM (one fits), each taking work tiles until none is left.
  dim3 const grid(static_cast<unsigned int>(std::min<std::int64_t>(n_work, std::max(args.sm_count, 1))));
  dim3 const block(Traits::kNumThreads);
  cudaStream_t const stream = reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(args.stream));
  detail::triangle_attention_kernel<Traits><<<grid, block, smem_bytes, stream>>>(params);
  check_cuda(cudaGetLastError(), "triangle_attention_kernel launch");
}

std::size_t attention_dynamic_smem_bytes()
{
  return sizeof(Traits::SharedStorage);
}

} // namespace bioir::claude_kit_triangle_attention_sm90_d32
