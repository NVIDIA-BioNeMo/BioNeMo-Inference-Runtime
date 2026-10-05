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

#ifndef BIOIR_CPP_KERNELS_HEURISTIC_POLICY_H_
#define BIOIR_CPP_KERNELS_HEURISTIC_POLICY_H_

#include <cstdint>

namespace bioir::heuristic
{

enum class DType : std::uint8_t
{
  kFloat16,
  kBFloat16,
  kFloat32,
};

enum class TriangleAttentionImplementation : std::uint8_t
{
  kClaudeKit,
  kCuTeDSL,
  kUnsupported,
};

// The static shape of a triangle-attention call: q is [batch, i_dim, tokens, num_heads * head_dim].
struct TriangleAttentionRequest
{
  std::int32_t target_sm{};
  std::int32_t head_dim{};
  std::int32_t num_heads{};
  std::int32_t batch{};
  std::int32_t tokens{};
  std::int32_t i_dim{};
  DType dtype{DType::kFloat32};
  bool has_lse{};
};

// The fastest implementation that supports the request.
TriangleAttentionImplementation select_triangle_attention(TriangleAttentionRequest const& request);

} // namespace bioir::heuristic

#endif // BIOIR_CPP_KERNELS_HEURISTIC_POLICY_H_
