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

#include "policy.h"

#include <cstdint>
#include <initializer_list>
#include <limits>

namespace bioir::heuristic
{

namespace
{

// ClaudeKit's fixed per-call cost pays off only past this much work, counted in attention scores.
constexpr std::int64_t kClaudeKitMinScores = 23'000'000;

// batch * i_dim * tokens^2 * num_heads of a request with positive extents, saturated at the int64 maximum.
std::int64_t attention_scores(TriangleAttentionRequest const& request)
{
  std::int64_t scores = 1;
  for (std::int64_t const extent : {request.batch, request.i_dim, request.tokens, request.tokens, request.num_heads})
  {
    if (scores > std::numeric_limits<std::int64_t>::max() / extent)
    {
      return std::numeric_limits<std::int64_t>::max();
    }
    scores *= extent;
  }
  return scores;
}

bool cutedsl_supports(TriangleAttentionRequest const& request)
{
  bool const half = request.dtype == DType::kFloat16 || request.dtype == DType::kBFloat16;
  bool const sm_supported = request.target_sm == 80 || request.target_sm == 86 || request.target_sm == 89
    || request.target_sm == 90 || request.target_sm == 100 || request.target_sm == 103;
  bool const head_dim_supported
    = request.head_dim == 32 || request.head_dim == 64 || request.head_dim == 128 || request.head_dim == 256;
  return half && sm_supported && head_dim_supported && request.tokens > 0 && request.i_dim > 0;
}

} // namespace

TriangleAttentionImplementation select_triangle_attention(TriangleAttentionRequest const& request)
{
  bool const positive = request.batch > 0 && request.num_heads > 0 && request.tokens > 0 && request.i_dim > 0;
  if (
    request.target_sm == 90 && request.dtype == DType::kBFloat16 && request.head_dim == 32 && positive
    && attention_scores(request) >= kClaudeKitMinScores)
  {
    return TriangleAttentionImplementation::kClaudeKit;
  }
  if (cutedsl_supports(request))
  {
    return TriangleAttentionImplementation::kCuTeDSL;
  }
  return TriangleAttentionImplementation::kUnsupported;
}

} // namespace bioir::heuristic
