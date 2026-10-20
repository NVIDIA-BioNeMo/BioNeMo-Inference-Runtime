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
"""Exercise the reusable constructor and fused checkpoint layout without kernels."""

from __future__ import annotations

import ast
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
ROOT = Path(__file__).resolve().parents[2]


def _load_constructor(path: Path, name: str, namespace: dict[str, object]) -> type:
    tree = ast.parse(path.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    # Run the actual constructor, keeping unrelated GPU-only forward paths out
    # of this launcher-free contract test.
    definition.body = [
        node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), definition],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class _CpuAllocation:
    """Redirect only the linear module's explicit CUDA allocations."""

    def __getattr__(self, name: str) -> object:
        return getattr(torch, name)

    def device(self, value: str) -> torch.device:
        return torch.device("cpu" if value == "cuda" else value)


def _block_type() -> type:
    linear = runpy.run_path(str(ROOT / "bionemo_ir/_torch/layers/linear.py"))
    # Keep real Linear creation, loading and forward code. Only its allocation
    # device differs; no compiled operator or model-forward parity is asserted.
    linear["Linear"].create_weights.__globals__["torch"] = _CpuAllocation()
    namespace = {
        "torch": torch,
        "nn": nn,
        "Linear": linear["Linear"],
        "WeightMode": linear["WeightMode"],
        "WeightsLoadingConfig": linear["WeightsLoadingConfig"],
        "AttentionType": SimpleNamespace(PAIRWISE="PAIRWISE"),
        "create_attention": lambda *args, **kwargs: nn.Identity(),
        "get_attn_epilogue_op": lambda *args, **kwargs: None,
        "get_gated_sigmoid_op": lambda *args, **kwargs: None,
        "LNProjMoveaxisPad": lambda **kwargs: nn.Identity(),
        "_make_norm": lambda kind, width, **kwargs: (
            nn.RMSNorm(width, **kwargs) if kind == "rms_norm" else nn.LayerNorm(width, **kwargs)
        ),
        "AdaLN": lambda *args, **kwargs: nn.Identity(),
        "ConditionedTransitionBlock": lambda **kwargs: nn.Identity(),
    }
    _load_constructor(ROOT / "bionemo_ir/_torch/layers/attention.py", "AttentionPairBias", namespace)
    return _load_constructor(
        ROOT / "bionemo_ir/_torch/layers/transformers/blocks.py", "ConditionedTransformerBlock", namespace
    )


@pytest.mark.parametrize("norm_type", ["rms_norm", "layer_norm"])
@pytest.mark.parametrize("pair_bias", [False, True])
def test_conditioned_constructor_preserves_fused_gate_bias(norm_type: str, pair_bias: bool) -> None:
    block_type = _block_type()
    args = (0, 16, 8, 4, 2, 2, "SDPA", 1e6, True, 1e-5, torch.float32)
    block = block_type(*args, norm_type=norm_type, use_pair_bias=pair_bias)
    gate_bias = torch.linspace(-1.0, 1.0, 16)
    segments = [
        {"weight": torch.zeros(16, 16)},
        {"weight": torch.zeros(16, 16), "bias": gate_bias},
        {"weight": torch.zeros(16, 16), "bias": torch.full((16,), 2.0)},
        {"weight": torch.zeros(16, 16), "bias": torch.full((16,), 3.0)},
    ]
    block.mha.in_proj.load_weights(segments)
    restored = block_type(*args, norm_type=norm_type, use_pair_bias=pair_bias)
    restored.load_state_dict(block.state_dict(), strict=True)
    query, gate, key, value = restored.mha.in_proj(torch.zeros(1, 16)).split((16, 16, 16, 16), dim=-1)
    torch.testing.assert_close(query, torch.zeros(1, 16))
    torch.testing.assert_close(gate, gate_bias.unsqueeze(0))
    torch.testing.assert_close(key, torch.full((1, 16), 2.0))
    torch.testing.assert_close(value, torch.full((1, 16), 3.0))


def test_conditioned_constructor_gives_the_attention_the_adaln_zero_gate() -> None:
    """The checkpoint converter writes the adaLN-zero gate to ``mha.output_projection``."""
    block = _block_type()(0, 16, 8, 4, 2, 2, "SDPA", 1e6, True, 1e-5, torch.float32)
    state = block.state_dict()
    assert not [key for key in state if "adaln_zero" in key]
    torch.testing.assert_close(state["mha.output_projection.weight"], torch.zeros(16, 4))
    torch.testing.assert_close(state["mha.output_projection.bias"], torch.full((16,), -2.0))
