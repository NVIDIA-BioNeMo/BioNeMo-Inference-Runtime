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

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace bioir::heuristic
{

void bind(nb::module_& parent)
{
  nb::module_ module = parent.def_submodule("heuristic", "BioIR kernel selection policies");

  nb::enum_<DType>(module, "DType")
    .value("FLOAT16", DType::kFloat16)
    .value("BFLOAT16", DType::kBFloat16)
    .value("FLOAT32", DType::kFloat32);

  nb::enum_<TriangleAttentionImplementation>(module, "TriangleAttentionImplementation")
    .value("CLAUDE_KIT", TriangleAttentionImplementation::kClaudeKit)
    .value("CUTEDSL", TriangleAttentionImplementation::kCuTeDSL)
    .value("UNSUPPORTED", TriangleAttentionImplementation::kUnsupported);

  module.def(
    "select_triangle_attention",
    [](
      std::int32_t target_sm,
      std::int32_t head_dim,
      std::int32_t num_heads,
      std::int32_t batch,
      std::int32_t tokens,
      std::int32_t i_dim,
      DType dtype,
      bool has_lse)
    {
      TriangleAttentionRequest request;
      request.target_sm = target_sm;
      request.head_dim = head_dim;
      request.num_heads = num_heads;
      request.batch = batch;
      request.tokens = tokens;
      request.i_dim = i_dim;
      request.dtype = dtype;
      request.has_lse = has_lse;
      return select_triangle_attention(request);
    },
    "target_sm"_a,
    "head_dim"_a,
    "num_heads"_a,
    "batch"_a,
    "tokens"_a,
    "i_dim"_a,
    "dtype"_a,
    "has_lse"_a = true,
    "The fastest triangle-attention implementation for q of shape [batch, i_dim, tokens, num_heads * head_dim].");
}

} // namespace bioir::heuristic
