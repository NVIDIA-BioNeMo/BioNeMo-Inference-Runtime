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

from collections.abc import Callable
from typing import cast

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bionemo_ir._torch.graph_optimization.rewrite import rewrite_modules
from bionemo_ir._torch.layers.normalization import (
    FusedLayerNorm,
    FusedRMSNorm,
    HighPrecisionLayerNorm,
    replace_with_fused_layernorm,
    replace_with_fused_rmsnorm,
    replace_with_high_precision_layernorm,
)

_ROWS = 512


class _Block(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class _Stack(nn.Module):
    def __init__(self, depth: int = 3, dim: int = 128):
        super().__init__()
        self.blocks = nn.ModuleList(_Block(dim) for _ in range(depth))
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)


class _RMSBlock(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class _RMSStack(nn.Module):
    def __init__(self, depth: int = 3, dim: int = 128):
        super().__init__()
        self.blocks = nn.ModuleList(_RMSBlock(dim) for _ in range(depth))
        self.norm_out = nn.RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)


def test_replace_with_fused_layernorm_replaces_every_plain_layernorm() -> None:
    model = _Stack(depth=3)
    assert replace_with_fused_layernorm(model) == 4
    replaced = [m for m in model.modules() if isinstance(m, FusedLayerNorm)]
    assert len(replaced) == 4
    assert not [m for m in model.modules() if type(m) is nn.LayerNorm]


def test_rewrite_is_idempotent() -> None:
    """Exact-type matching keeps a replacement subclass from matching again."""
    model = _Stack(depth=2)
    assert replace_with_fused_layernorm(model) == 3
    assert replace_with_fused_layernorm(model) == 0


def test_rewrite_modules_replaces_a_with_b_recursively() -> None:
    model = nn.Sequential(nn.ReLU(), nn.Sequential(nn.ReLU()))

    assert rewrite_modules(model, nn.ReLU, lambda _: nn.Identity()) == 2
    assert isinstance(model[0], nn.Identity)
    assert isinstance(model[1][0], nn.Identity)


def test_rewrite_modules_preserves_source_subclasses() -> None:
    class ReLUSubclass(nn.ReLU):
        pass

    model = nn.Sequential(nn.ReLU(), ReLUSubclass())

    assert rewrite_modules(model, nn.ReLU, lambda _: nn.Identity()) == 1
    assert isinstance(model[0], nn.Identity)
    assert type(model[1]) is ReLUSubclass


def test_rewrite_modules_rejects_non_module_replacement() -> None:
    model = nn.Sequential(nn.ReLU())
    invalid_replacement = cast(Callable[[nn.ReLU], nn.Module], lambda _: "not a module")

    with pytest.raises(TypeError, match="expected nn.Module"):
        rewrite_modules(model, nn.ReLU, invalid_replacement)


def test_rewrite_modules_identity_replacement_is_a_noop() -> None:
    model = nn.Sequential(nn.ReLU(), nn.Sequential(nn.ReLU()))
    first = model[0]

    assert rewrite_modules(model, nn.ReLU, lambda module: module) == 0
    assert model[0] is first
    assert type(model[1][0]) is nn.ReLU


def test_state_dict_keys_and_values_survive_the_rewrite() -> None:
    model = _Stack(depth=2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    replace_with_fused_layernorm(model)
    after = model.state_dict()
    assert set(after) == set(before)
    for key, value in before.items():
        torch.testing.assert_close(after[key], value)


def test_rewrite_shares_parameters_rather_than_copying() -> None:
    model = _Block()
    original_weight = model.norm.weight
    replace_with_fused_layernorm(model)
    assert model.norm.weight is original_weight


def test_load_state_dict_still_works_after_rewrite() -> None:
    source = _Stack(depth=2)
    target = _Stack(depth=2)
    replace_with_fused_layernorm(target)
    target.load_state_dict(source.state_dict())
    torch.testing.assert_close(target.norm_out.weight, source.norm_out.weight)


def test_skip_types_leaves_the_whole_subtree_alone() -> None:
    model = _Stack(depth=2)
    assert replace_with_fused_layernorm(model, skip_types=(_Block,)) == 1
    assert type(model.blocks[0].norm) is nn.LayerNorm
    assert isinstance(model.norm_out, FusedLayerNorm)


def test_layernorm_subclasses_are_left_alone() -> None:
    """``HighPrecisionLayerNorm`` has its own dtype contract to preserve."""
    model = _Block()
    model.norm = HighPrecisionLayerNorm.from_layernorm(model.norm, out_dtype=torch.bfloat16)
    assert replace_with_fused_layernorm(model) == 0
    assert isinstance(model.norm, HighPrecisionLayerNorm)


def test_high_precision_layernorm_uses_the_generic_rewrite() -> None:
    model = _Stack(depth=2)

    assert replace_with_high_precision_layernorm(model, out_dtype=torch.bfloat16) == 3
    assert len([module for module in model.modules() if isinstance(module, HighPrecisionLayerNorm)]) == 3
    assert replace_with_high_precision_layernorm(model, out_dtype=torch.bfloat16) == 0


@pytest.mark.parametrize(
    ("normalized_shape", "input_shape"),
    [(128, (2, 17, 128)), ((4, 128), (2, 4, 128))],
    ids=["one_axis", "multiple_axes"],
)
@pytest.mark.parametrize(
    ("elementwise_affine", "bias"),
    [(False, False), (True, False), (True, True)],
    ids=["no_affine", "weight", "weight_bias"],
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_high_precision_layernorm_uses_triton_kernel(
    monkeypatch: pytest.MonkeyPatch,
    normalized_shape: int | tuple[int, ...],
    input_shape: tuple[int, ...],
    elementwise_affine: bool,
    bias: bool,
    out_dtype: torch.dtype,
) -> None:
    torch.manual_seed(19)
    reference = nn.LayerNorm(
        normalized_shape,
        elementwise_affine=elementwise_affine,
        bias=bias,
        dtype=torch.float32,
    ).cuda()
    if reference.weight is not None:
        with torch.no_grad():
            reference.weight.normal_()
            if reference.bias is not None:
                reference.bias.normal_()
    fused = HighPrecisionLayerNorm.from_layernorm(reference, out_dtype=out_dtype)
    x = torch.randn(*input_shape, device="cuda", dtype=torch.bfloat16)

    expected = F.layer_norm(x.float(), reference.normalized_shape, reference.weight, reference.bias, reference.eps)
    expected = expected.to(out_dtype)

    def unexpected_torch_layer_norm(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("ATen LayerNorm fallback must not run")

    monkeypatch.setattr(F, "layer_norm", unexpected_torch_layer_norm)
    with torch.inference_mode():
        actual = fused(x)

    tolerance = 2e-5 if out_dtype == torch.float32 else 2e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert actual.dtype == out_dtype


def test_multi_axis_layernorm_stays_eager() -> None:
    model = _Block()
    original = nn.LayerNorm([4, 128])
    model.norm = original
    assert replace_with_fused_layernorm(model) == 0
    assert model.norm is original
    x = torch.randn(2, 4, 128)
    torch.testing.assert_close(model.norm(x), F.layer_norm(x, (4, 128), model.norm.weight, model.norm.bias))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kernel_path_matches_torch(dtype: torch.dtype) -> None:
    torch.manual_seed(7)
    dim = 128
    reference = nn.LayerNorm(dim, dtype=dtype).cuda()
    with torch.no_grad():
        reference.weight.normal_()
        reference.bias.normal_()
    fused = FusedLayerNorm.from_layernorm(reference)
    assert isinstance(fused, FusedLayerNorm)

    x = torch.randn(_ROWS, dim, device="cuda", dtype=dtype)
    with torch.inference_mode():
        actual = fused(x)
        expected = reference(x)

    tolerance = 2e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_small_cuda_input_uses_kernel() -> None:
    """The explicit rewrite has no hidden shape-dependent dispatch."""
    torch.manual_seed(7)
    dim = 128
    reference = nn.LayerNorm(dim, dtype=torch.bfloat16).cuda()
    fused = FusedLayerNorm.from_layernorm(reference)
    assert isinstance(fused, FusedLayerNorm)

    x = torch.randn(8, dim, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        torch.testing.assert_close(fused(x), reference(x), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("elementwise_affine", "bias"),
    [(True, True), (True, False), (False, False)],
)
def test_missing_affine_parameters_are_handled(elementwise_affine: bool, bias: bool) -> None:
    """A norm without weight or bias still takes the kernel path."""
    torch.manual_seed(5)
    dim = 128
    reference = nn.LayerNorm(dim, elementwise_affine=elementwise_affine, bias=bias, dtype=torch.bfloat16).cuda()
    if elementwise_affine:
        with torch.no_grad():
            reference.weight.normal_()
            if bias:
                reference.bias.normal_()
    fused = FusedLayerNorm.from_layernorm(reference)
    assert isinstance(fused, FusedLayerNorm)
    if reference.weight is None and reference.bias is None:
        fused = fused.to(device="cuda", dtype=torch.bfloat16)

    # The kernel accepts missing affine tensors directly, without synthetic buffers.
    assert set(fused.state_dict()) == set(reference.state_dict())
    assert "_ones" not in fused._buffers
    assert "_zeros" not in fused._buffers

    x = torch.randn(_ROWS, dim, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        torch.testing.assert_close(fused(x), reference(x), atol=2e-2, rtol=2e-2)


def test_higher_rank_input_keeps_its_shape() -> None:
    torch.manual_seed(9)
    dim = 128
    reference = nn.LayerNorm(dim, dtype=torch.bfloat16).cuda()
    fused = FusedLayerNorm.from_layernorm(reference)
    x = torch.randn(2, 8, _ROWS // 16, dim, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        actual = fused(x)
    assert actual.shape == x.shape
    with torch.inference_mode():
        torch.testing.assert_close(actual, reference(x), atol=2e-2, rtol=2e-2)


def test_replace_with_fused_rmsnorm_replaces_every_plain_rmsnorm() -> None:
    model = _RMSStack(depth=3)
    assert replace_with_fused_rmsnorm(model) == 4
    replaced = [m for m in model.modules() if isinstance(m, FusedRMSNorm)]
    assert len(replaced) == 4
    assert not [m for m in model.modules() if type(m) is nn.RMSNorm]


def test_rmsnorm_rewrite_is_idempotent() -> None:
    model = _RMSStack(depth=2)
    assert replace_with_fused_rmsnorm(model) == 3
    assert replace_with_fused_rmsnorm(model) == 0


def test_rmsnorm_rewrite_does_not_touch_layernorm() -> None:
    model = _Stack(depth=2)
    assert replace_with_fused_rmsnorm(model) == 0
    assert not [m for m in model.modules() if isinstance(m, FusedRMSNorm)]
    assert [m for m in model.modules() if type(m) is nn.LayerNorm]


def test_rmsnorm_state_dict_and_parameter_sharing() -> None:
    model = _RMSBlock()
    original_weight = model.norm.weight
    before = {k: v.clone() for k, v in model.state_dict().items()}
    replace_with_fused_rmsnorm(model)
    assert model.norm.weight is original_weight
    after = model.state_dict()
    assert set(after) == set(before)
    for key, value in before.items():
        torch.testing.assert_close(after[key], value)


def test_multi_axis_rmsnorm_stays_eager() -> None:
    model = _RMSBlock()
    original = nn.RMSNorm([4, 128])
    model.norm = original
    assert replace_with_fused_rmsnorm(model) == 0
    assert model.norm is original
    x = torch.randn(2, 4, 128)
    torch.testing.assert_close(model.norm(x), F.rms_norm(x, (4, 128), model.norm.weight))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("elementwise_affine", [True, False])
def test_rmsnorm_kernel_path_matches_torch(dtype: torch.dtype, elementwise_affine: bool) -> None:
    torch.manual_seed(7)
    dim = 128
    reference = nn.RMSNorm(dim, eps=1e-5, elementwise_affine=elementwise_affine, dtype=dtype).cuda()
    if elementwise_affine:
        with torch.no_grad():
            reference.weight.normal_()
    fused = FusedRMSNorm.from_rmsnorm(reference)
    assert isinstance(fused, FusedRMSNorm)

    x = torch.randn(_ROWS, dim, device="cuda", dtype=dtype)
    with torch.inference_mode():
        actual = fused(x)
        expected = reference(x)

    tolerance = 2e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rmsnorm_kernel_uses_dtype_default_epsilon(dtype: torch.dtype) -> None:
    torch.manual_seed(8)
    dim = 256
    reference = nn.RMSNorm(dim, dtype=dtype).cuda()
    fused = FusedRMSNorm.from_rmsnorm(reference)
    x = torch.randn(11, dim, device="cuda", dtype=dtype)

    with torch.inference_mode():
        actual = fused(x)
        expected = reference(x)

    tolerance = 2e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_rmsnorm_higher_rank_input_keeps_its_shape() -> None:
    torch.manual_seed(9)
    dim = 128
    reference = nn.RMSNorm(dim, eps=1e-5, dtype=torch.bfloat16).cuda()
    fused = FusedRMSNorm.from_rmsnorm(reference)
    x = torch.randn(2, 8, _ROWS // 16, dim, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        actual = fused(x)
    assert actual.shape == x.shape
    with torch.inference_mode():
        torch.testing.assert_close(actual, reference(x), atol=2e-2, rtol=2e-2)
