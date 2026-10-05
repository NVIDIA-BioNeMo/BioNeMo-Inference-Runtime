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

#ifndef BIOIR_CPP_KERNELS_CLAUDE_KIT_TRIANGLE_ATTENTION_SM90_D32_LAUNCHER_H_
#define BIOIR_CPP_KERNELS_CLAUDE_KIT_TRIANGLE_ATTENTION_SM90_D32_LAUNCHER_H_

#include "cutedsl_tensor_abi.h"

#include <cstddef>
#include <cstdint>

namespace bioir::claude_kit_triangle_attention_sm90_d32
{

// One triangle-attention call. q, k, v and output are [B * I, S, H, 32] bf16 views with unit innermost stride (the
// packed in_proj views qualify), bias is [B, H, S, S_padded] bf16, actual_s_kv is [B * I] int32 (live leading keys of
// each row), and lse is an optional [B * I, S, H] fp32 view (null data: not written).
struct LaunchParams
{
  cutedsl::Tensor4View q;
  cutedsl::Tensor4View k;
  cutedsl::Tensor4View v;
  cutedsl::Tensor4View bias;
  cutedsl::Tensor1View actual_s_kv;
  cutedsl::Tensor4View output;
  cutedsl::Tensor3View lse;
  float softmax_scale{};
  std::int32_t i_dim{};
  std::uint64_t stream{};
};

// The attention kernel's arguments after stage_bias; strides in elements (bi: B * I rows, s: sequence, h: heads).
struct KernelLaunchParams
{
  void const* q{};
  void const* k{};
  void const* v{};
  float const* staged_bias{};
  std::int32_t const* staged_length{};
  std::int32_t const* actual_s_kv{};
  void* output{};
  float* lse{};
  std::int32_t* tile_counter{}; // zeroed by stage_bias

  std::int64_t q_stride_bi{};
  std::int64_t q_stride_s{};
  std::int64_t q_stride_h{};
  std::int64_t k_stride_bi{};
  std::int64_t k_stride_s{};
  std::int64_t k_stride_h{};
  std::int64_t v_stride_bi{};
  std::int64_t v_stride_s{};
  std::int64_t v_stride_h{};
  std::int64_t output_stride_bi{};
  std::int64_t output_stride_s{};
  std::int64_t output_stride_h{};
  std::int64_t lse_stride_bi{};
  std::int64_t lse_stride_s{};

  std::int32_t batch{};
  std::int32_t i_dim{};
  std::int32_t num_heads{};
  std::int32_t seqlen{};
  std::int32_t head_group{1}; // heads whose work tiles run back to back
  std::int32_t sm_count{1};
  float softmax_scale{};
  std::uint64_t stream{};
};

// Bias staging: each batch's staged length (keys past it stage as -inf) and bias / scale in fragment order.
struct BiasStageParams
{
  void const* bias{};
  std::int32_t const* actual_s_kv{}; // [B * I]
  float* staged_bias{};
  std::int32_t* staged_length{}; // [B]
  std::int32_t* tile_counter{};
  std::int64_t stride_batch{};
  std::int64_t stride_head{};
  std::int64_t stride_query{};
  std::int32_t batch{};
  std::int32_t i_dim{};
  std::int32_t num_heads{};
  std::int32_t seqlen{};
  float inverse_softmax_scale{};
  std::uint64_t stream{};
};

// Validate the call, stage the bias into a stream-ordered workspace and launch the attention kernel.
void launch(LaunchParams const& params);

void stage_bias(BiasStageParams const& params);
void launch_attention(KernelLaunchParams const& params);

std::size_t attention_dynamic_smem_bytes();

} // namespace bioir::claude_kit_triangle_attention_sm90_d32

#endif // BIOIR_CPP_KERNELS_CLAUDE_KIT_TRIANGLE_ATTENTION_SM90_D32_LAUNCHER_H_
