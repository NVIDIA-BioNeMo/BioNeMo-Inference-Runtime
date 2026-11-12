# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from bionemo_ir._torch.modules.boltz.structure import DiffusionConditioning

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# The current CUDA device, resolved when a test allocates rather than at collection.
DEVICE = torch.device("cuda")


def _conditioning(token_z: int, depth: int, heads: int) -> DiffusionConditioning:
    module = DiffusionConditioning(
        token_s=32,
        token_z=token_z,
        atom_s=16,
        atom_z=8,
        token_transformer_depth=depth,
        token_transformer_heads=heads,
        version="v2",
        token_trans_bias_dtype=torch.bfloat16,
    ).to(DEVICE)
    with torch.no_grad():
        for parameter in module.token_trans_proj_z.parameters():
            parameter.normal_(mean=0.0, std=0.5)
    return module.eval()


def _concatenated(module: DiffusionConditioning, z: torch.Tensor) -> torch.Tensor:
    return torch.cat([layer(z) for layer in module.token_trans_proj_z], dim=-1)


@pytest.mark.parametrize("batch", [1, 2])
def test_token_trans_bias_assembly_matches_concatenation(batch: int) -> None:
    torch.manual_seed(0)
    depth, heads, tokens = 5, 3, 19
    module = _conditioning(token_z=16, depth=depth, heads=heads)
    z = torch.randn(batch, tokens, tokens, 16, device=DEVICE, dtype=torch.bfloat16)

    with torch.inference_mode():
        expected = _concatenated(module, z)
        actual = module._assemble_token_trans_bias(z)

    assert actual.shape == (batch, tokens, tokens, depth * heads)
    assert actual.dtype == expected.dtype
    assert actual.stride() == expected.stride()
    assert torch.equal(actual, expected)


def test_token_trans_bias_assembly_holds_one_layer_output() -> None:
    torch.manual_seed(1)
    depth, heads, tokens, token_z = 24, 16, 256, 128
    module = _conditioning(token_z=token_z, depth=depth, heads=heads)
    z = torch.randn(1, tokens, tokens, token_z, device=DEVICE, dtype=torch.bfloat16)
    element = z.element_size()
    layer_bytes = tokens * tokens * heads * element
    norm_bytes = z.numel() * element

    def transient_peak(build) -> tuple[torch.Tensor, int]:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.inference_mode():
            out = build()
        torch.cuda.synchronize()
        return out, torch.cuda.max_memory_allocated() - base

    expected, concat_peak = transient_peak(lambda: _concatenated(module, z))
    del expected
    actual, assembled_peak = transient_peak(lambda: module._assemble_token_trans_bias(z))

    out_bytes = actual.numel() * element
    # The output, one LayerNorm temporary and one layer output, with allocator rounding.
    assert assembled_peak <= out_bytes + norm_bytes + 2 * layer_bytes
    # Concatenation keeps every layer output next to its result.
    assert concat_peak >= 2 * out_bytes
