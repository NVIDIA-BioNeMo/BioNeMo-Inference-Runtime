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
"""Transition layers on the LayerNorm-fused ``transition_mlp_w_norm`` kernel.

The routing and weight-folding tests run the layers on a PyTorch stand-in for the op, so they hold on any GPU and for
any shape. The kernel tests run each layer at the shapes the models use, wherever the op builds them, shipped or
candidate, against the layer's own unfused path.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from bionemo_ir._torch.custom_ops.transition_mlp_w_norm import cutedsl as w_norm_cutedsl
from bionemo_ir._torch.custom_ops.transition_mlp_w_norm import get_transition_mlp_w_norm_op
from bionemo_ir._torch.custom_ops.transition_mlp_w_norm.cutedsl import TransitionMlpWNormCuTe
from bionemo_ir._torch.layers import transition
from bionemo_ir._torch.layers.normalization import FusedLayerNorm, replace_with_fused_layernorm
from bionemo_ir._torch.layers.transition import MSATransition, PairTransition, Transition
from bionemo_ir._torch.utils import ChunkPolicy
from bionemo_ir._torch.utils.common import recursive_calling_load_weights
from tests._torch import cutedsl_test_modes, require_cubin_family, run_cutedsl_test_mode, skip_if_not_sm90

_SOURCE = "bionemo_ir._torch.custom_ops.transition_mlp_w_norm._source"
_MODES = cutedsl_test_modes(_SOURCE)
ATOL, RTOL = 5e-2, 2e-2  # tests/ops/test_transition_mlp.py's tolerance
TransitionLayer = PairTransition | MSATransition | Transition


@dataclass(frozen=True)
class _Site:
    """A transition layer as a model builds and calls it: kind, width C, hidden width H and the rows of its input."""

    kind: str  # "pair" (PairTransition, ReLU), "msa" (MSATransition, ReLU) or "swiglu" (Transition)
    width: int
    hidden: int
    rows: tuple[int, ...]

    def __str__(self) -> str:
        return f"{self.kind}-C{self.width}-H{self.hidden}-{'x'.join(map(str, self.rows))}"


# One row per call site in the models: ReLU pair maps [B, N, N, C] (a template's carries a size-1 template axis),
# flat MSA rows [B, S, N, C], and SwiGLU pair maps, MSA rows and diffusion-conditioning pair maps.
_SITES = [
    _Site("pair", 128, 512, (2, 24, 24)),  # OpenFold2 Evoformer and extra-MSA pair transitions
    _Site("msa", 256, 1024, (1, 6, 23)),  # OpenFold2 Evoformer MSA transition
    _Site("msa", 64, 256, (1, 9, 23)),  # OpenFold2 extra-MSA MSA transition
    _Site("pair", 64, 128, (1, 1, 23, 23)),  # OpenFold2 template pair stack, one template
    _Site("pair", 384, 768, (2, 24, 24)),  # PairUpdateBlock's pair transition
    _Site("swiglu", 128, 512, (2, 24, 24)),  # Boltz and OpenFold3 Pairformer and MSA-module pair transitions
    _Site("swiglu", 256, 1024, (1, 23, 23)),  # Protenix Pairformer and MSA-module pair transitions
    _Site("swiglu", 64, 256, (1, 6, 23)),  # Boltz and OpenFold3 MSA-module MSA transitions
    _Site("swiglu", 64, 256, (2, 24, 24)),  # Boltz-2 template pair stack
    _Site("swiglu", 128, 512, (1, 6, 23)),  # Protenix MSA-module MSA transition
    _Site("swiglu", 64, 128, (2, 23, 23)),  # OpenFold3 and Protenix template pair stacks
    _Site("swiglu", 128, 256, (1, 1, 24, 24)),  # Boltz and OpenFold3 diffusion conditioning, with a sample axis
    _Site("swiglu", 256, 512, (1, 23, 23)),  # Protenix diffusion conditioning
]


def _layer(site: _Site, **kwargs: object) -> TransitionLayer:
    if site.kind == "pair":
        return PairTransition(c_z=site.width, n=site.hidden // site.width, dtype=torch.bfloat16, **kwargs)
    if site.kind == "msa":
        return MSATransition(c_m=site.width, n=site.hidden // site.width, dtype=torch.bfloat16, **kwargs)
    return Transition(site.width, site.hidden, dtype=torch.bfloat16, **kwargs)


def _released_state(layer: torch.nn.Module, seed: int) -> dict[str, torch.Tensor]:
    """A state dict at the released weights' scale: LayerNorm gamma near 1, small biases, fan-in scaled linears."""
    g = torch.Generator().manual_seed(seed)
    state = {}
    for name, value in layer.state_dict().items():
        if value.ndim == 2:
            sample = value.shape[1] ** -0.5 * torch.randn(value.shape, generator=g)
        elif name.endswith("norm.weight"):
            sample = 1 + 0.1 * torch.randn(value.shape, generator=g)
        else:
            sample = 0.1 * torch.randn(value.shape, generator=g)
        state[name] = sample.to(value.dtype)
    return state


def _loaded(site: _Site, seed: int = 0, **kwargs: object) -> TransitionLayer:
    """The site's layer on the GPU with weights loaded through ``load_state_dict``."""
    layer = _layer(site, **kwargs).cuda().eval()
    layer.load_state_dict(_released_state(layer, seed))
    return layer


def _inputs(site: _Site, mask_form: str | None, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor | None]:
    """bf16 input, and a 0/1 bf16 mask with one value per row: ``[*rows]``, ``[*rows, 1]`` or none."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    z = torch.randn(*site.rows, site.width, device="cuda", generator=g).to(torch.bfloat16)
    if mask_form is None:
        return z, None
    mask = (torch.rand(site.rows, device="cuda", generator=g) > 0.25).to(torch.bfloat16)
    return z, mask if mask_form == "rows" else mask.unsqueeze(-1)


def _call(layer: TransitionLayer, x: torch.Tensor, mask: torch.Tensor | None, residual: bool, inplace: bool):
    if isinstance(layer, Transition):
        return layer(x, mask, residual=residual, inplace=inplace)
    return layer(x, mask, residual=residual)


def _unfused(layer: TransitionLayer, x: torch.Tensor, mask: torch.Tensor | None, residual: bool) -> torch.Tensor:
    """The layer's own path without any fused kernel."""
    if isinstance(layer, PairTransition):
        return layer._forward_impl(x, mask, residual=residual)
    update = layer._forward_impl(x, mask)
    return x + update if residual else update


def _assert_masked_rows(output: torch.Tensor, x: torch.Tensor, mask: torch.Tensor | None, residual: bool) -> None:
    """A row whose mask is 0 comes out equal to ``x`` with the residual and 0 without it."""
    if mask is None:
        return
    masked = mask.reshape(x.shape[:-1]) == 0
    assert masked.any()
    expected = x[masked] if residual else torch.zeros_like(x[masked])
    assert torch.equal(output[masked], expected)


class _CountingOp:
    """Counts the calls a layer makes to its fused op."""

    def __init__(self, op) -> None:
        self.op = op
        self.calls = 0

    def accepts(self, *args, **kwargs) -> bool:
        return self.op.accepts(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        self.calls += 1
        return self.op(*args, **kwargs)


# ---- the routing and the folded weights, on a PyTorch stand-in for the op --------------------------------------------


@dataclass(frozen=True)
class _ReferencePack:
    """fp32 copies of the weights as they were at the fold."""

    weights: tuple[torch.Tensor | None, ...]
    activation: str
    device: torch.device


def _reference_pack_weights(*weights: torch.Tensor | None, activation: str = "relu") -> _ReferencePack:
    copies = tuple(None if w is None else w.detach().float().clone() for w in weights)
    return _ReferencePack(copies, activation, weights[2].device)


class _ReferenceOp:
    """The op's contract in PyTorch: any contiguous bf16 ``[..., width]`` as flat rows, and a contiguous bf16 mask
    shaped like its rows, or none; ``out`` may be ``z`` itself."""

    def __init__(self, width: int, hidden: int, activation: str) -> None:
        self.width, self.hidden, self.activation = width, hidden, activation
        self.calls = 0

    def accepts(self, z: torch.Tensor, mask: torch.Tensor | None, *, n_valid: torch.Tensor | None = None) -> bool:
        if n_valid is not None or not z.is_cuda or z.dtype != torch.bfloat16 or not z.is_contiguous():
            return False
        if z.shape[-1] != self.width:
            return False
        return mask is None or (
            mask.dtype == torch.bfloat16
            and mask.is_contiguous()
            and mask.shape == z.shape[:-1]
            and mask.device == z.device
        )

    def __call__(self, z, mask, pack, eps, *, residual=True, n_valid=None, out=None) -> torch.Tensor:
        assert self.accepts(z, mask, n_valid=n_valid)
        assert pack.activation == self.activation and pack.device == z.device
        self.calls += 1
        norm_weight, norm_bias, w1, b1, w2, b2 = pack.weights
        zf = z.float()
        projected = F.linear(F.layer_norm(zf, (self.width,), norm_weight, norm_bias, eps), w1, b1)
        if self.activation == "silu_gate":
            value, gate = projected.chunk(2, dim=-1)
            hidden = F.silu(gate) * value
        else:
            hidden = torch.relu(projected)
        update = F.linear(hidden, w2, b2)
        if mask is not None:
            update = update * mask.unsqueeze(-1).float()
        result = (zf + update if residual else update).to(z.dtype)
        return result if out is None else out.copy_(result)


def _reference_get_op(dtype: torch.dtype, width: int, hidden: int, activation: str = "relu") -> _ReferenceOp | None:
    return _ReferenceOp(width, hidden, activation) if dtype == torch.bfloat16 else None


@pytest.fixture
def reference_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build transition layers on the stand-in op, for every shape."""
    monkeypatch.setattr(transition, "get_transition_mlp_w_norm_op", _reference_get_op)
    monkeypatch.setattr(transition, "pack_weights", _reference_pack_weights)


_SMALL = [_Site("pair", 32, 128, (2, 9, 9)), _Site("msa", 32, 128, (1, 5, 9)), _Site("swiglu", 32, 64, (2, 9, 9))]


@pytest.mark.parametrize("site", _SMALL, ids=str)
@pytest.mark.parametrize("mask_kind", [None, "rows", "trailing", "bool", "float32", "broadcast"])
@pytest.mark.parametrize(("residual", "inplace"), [(False, False), (True, False), (True, True)])
def test_routes_each_call_with_one_bf16_value_per_row(reference_op, site, mask_kind, residual, inplace) -> None:
    """Every call the op accepts runs on it, with the mask turned into one bf16 value per row, whatever its form."""
    if site.kind == "msa" and mask_kind in (None, "trailing"):
        pytest.skip("MSATransition takes a [B, S, N] mask")
    if inplace and site.kind != "swiglu":
        pytest.skip("only Transition updates in place")
    layer = _loaded(site)
    z, row_mask = _inputs(site, None if mask_kind is None else "rows")
    mask = row_mask
    if mask_kind == "trailing":
        mask = row_mask.unsqueeze(-1)
    elif mask_kind in ("bool", "float32"):
        mask = row_mask.to(getattr(torch, mask_kind))
    elif mask_kind == "broadcast":  # one value per token, as an MSA's token mask broadcast over its sequences
        mask = row_mask[:, :1]
        row_mask = mask.expand_as(row_mask)
    original = z.clone()
    with torch.inference_mode():
        expected = _unfused(layer, z, mask, residual)
        output = _call(layer, z, mask, residual, inplace)
    assert layer._w_norm_op.calls == 1
    # MSATransition's unfused path promotes the update to an fp32 mask's dtype; the kernel keeps z's.
    assert output.dtype == z.dtype
    torch.testing.assert_close(output, expected, atol=ATOL, rtol=RTOL, check_dtype=False)
    assert (output.data_ptr() == z.data_ptr()) == inplace
    if not inplace:
        assert torch.equal(z, original)
    _assert_masked_rows(output, original, row_mask, residual)


@pytest.mark.parametrize("site", _SMALL, ids=str)
def test_keeps_its_path_for_calls_the_op_declines(reference_op, site) -> None:
    """A strided input and a per-channel mask take the unfused path."""
    layer = _loaded(site)
    z, mask = _inputs(site, "rows")
    strided, strided_mask = z.transpose(-2, -3), mask.transpose(-1, -2)
    with torch.inference_mode():
        output = layer(strided, strided_mask)
        torch.testing.assert_close(output, _unfused(layer, strided, strided_mask, False), atol=0, rtol=0)
        if site.kind != "msa":
            channels = torch.rand_like(z) > 0.5
            output = layer(z, channels)
            torch.testing.assert_close(output, _unfused(layer, z, channels, False), atol=0, rtol=0)
    assert layer._w_norm_op.calls == 0


@pytest.mark.parametrize("site", [s for s in _SMALL if s.kind != "msa"], ids=str)
def test_runs_a_map_the_chunk_policy_would_split_whole(reference_op, site) -> None:
    """The kernel holds no activation beyond its output, so it takes the whole map in one call."""
    layer = _loaded(site, auto_chunk_policy=ChunkPolicy(chunk_size=2, min_size=1))
    z, mask = _inputs(site, "rows")
    assert layer.auto_chunk_policy.should_chunk(z)
    with torch.inference_mode():
        output = layer(z, mask, residual=True)
        expected = _unfused(layer, z, mask, True)
    assert layer._w_norm_op.calls == 1
    torch.testing.assert_close(output, expected, atol=ATOL, rtol=RTOL)


def test_runs_only_a_normalized_transition_as_wide_as_its_input(reference_op) -> None:
    """``Transition`` folds a LayerNorm into the first linear and writes over its input: it needs both."""
    assert Transition(32, 64, dtype=torch.bfloat16)._w_norm_pack is not None
    for layer in (
        Transition(32, 64, dtype=torch.bfloat16, normalize=False),
        Transition(32, 64, out_dim=16, dtype=torch.bfloat16),
    ):
        assert layer._w_norm_op is None and layer._w_norm_pack is None
        layer.post_load_weights()
        assert layer._w_norm_pack is None


def _changed(layer: TransitionLayer, z: torch.Tensor, mask: torch.Tensor) -> None:
    """Assert the layer's fused output follows its current weights."""
    with torch.inference_mode():
        torch.testing.assert_close(layer(z, mask, residual=True), _unfused(layer, z, mask, True), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("site", _SMALL, ids=str)
def test_folds_when_built_and_again_after_each_load(reference_op, site) -> None:
    """The layer folds at construction, after ``load_state_dict`` (its own or a parent's), and in
    ``post_load_weights``, which ``recursive_calling_load_weights`` runs."""
    layer = _layer(site).cuda()
    built = layer._w_norm_pack
    assert built is not None and built.device == layer._w_norm_parts()[1].weight.device
    z, mask = _inputs(site, "rows")

    layer.load_state_dict(_released_state(layer, 1))
    assert layer._w_norm_pack is not built
    _changed(layer, z, mask)

    parent = torch.nn.Sequential(layer)
    parent.load_state_dict({f"0.{name}": value for name, value in _released_state(layer, 2).items()})
    _changed(layer, z, mask)

    with torch.no_grad():
        layer._w_norm_parts()[2].weight.neg_()
    layer.post_load_weights()
    _changed(layer, z, mask)
    assert layer._w_norm_op.calls == 3


@pytest.mark.parametrize("site", _SMALL, ids=str)
def test_refolds_after_a_load_into_one_child_or_an_in_place_edit(reference_op, site) -> None:
    """A load straight into one child or an in-place edit runs none of the layer's hooks; its next call refolds
    from the weights it now holds. Unchanged weights keep their pack."""
    layer = _loaded(site)
    z, mask = _inputs(site, "rows")
    _changed(layer, z, mask)
    pack = layer._w_norm_pack
    _changed(layer, z, mask)
    assert layer._w_norm_pack is pack

    norm, first, _ = layer._w_norm_parts()
    child = next(name for name, module in layer.named_children() if module is first)
    state = _released_state(layer, 5)
    first.load_state_dict(
        {key.removeprefix(f"{child}."): value for key, value in state.items() if key.startswith(f"{child}.")}
    )
    _changed(layer, z, mask)
    assert layer._w_norm_pack is not pack

    with torch.no_grad():
        norm.weight.mul_(1.5)
    _changed(layer, z, mask)


def test_refolds_after_recursive_calling_load_weights(reference_op) -> None:
    site = _SMALL[0]
    layer = _loaded(site)
    z, mask = _inputs(site, "rows")
    state = _released_state(layer, 3)
    weights = {
        name: [{"weight": state[f"{name}.weight"].cuda(), "bias": state[f"{name}.bias"].cuda()}]
        for name in ("layer_norm", "linear_1", "linear_2")
    }
    pack = layer._w_norm_pack
    recursive_calling_load_weights(layer, weights)
    assert layer._w_norm_pack is not pack
    _changed(layer, z, mask)


@pytest.mark.parametrize("site", _SMALL, ids=str)
def test_holds_no_pack_without_weights(reference_op, site) -> None:
    """A layer built without weights folds once they are created and loaded."""
    layer = _layer(site, skip_create_weights=True)
    assert layer._w_norm_op is not None and layer._w_norm_pack is None
    for linear in layer._w_norm_parts()[1:]:
        linear.create_weights()
    layer.cuda().load_state_dict(_released_state(layer, 4))
    assert layer._w_norm_pack is not None
    z, mask = _inputs(site, "rows")
    _changed(layer, z, mask)


def test_refolds_for_a_call_from_another_device(reference_op) -> None:
    site = _SMALL[2]
    layer = _loaded(site)
    layer._w_norm_pack = layer._fold_w_norm_weights(torch.device("cpu"))
    assert layer._w_norm_pack.device.type == "cpu"
    z, mask = _inputs(site, "rows")
    _changed(layer, z, mask)
    assert layer._w_norm_pack.device == z.device


def test_does_not_keep_a_fold_made_under_graph_capture(reference_op) -> None:
    """A fold recorded into a graph holds its values only when the graph replays, so eager calls refold."""
    site = _SMALL[2]
    layer = _loaded(site)
    z, mask = _inputs(site, "rows")
    x = z.clone()
    with torch.inference_mode():
        layer(x, mask, residual=True, inplace=True)  # warm up outside the capture
        x.copy_(z)
        layer._w_norm_pack = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            layer(x, mask, residual=True, inplace=True)
        assert layer._w_norm_pack is None
        x.copy_(z)
        graph.replay()
        torch.testing.assert_close(x, _unfused(layer, z, mask, True), atol=ATOL, rtol=RTOL)
        layer(z.clone(), mask, residual=True)
    assert layer._w_norm_pack is not None


# ---- the kernel at the models' shapes -------------------------------------------------------------------------------


@pytest.fixture
def candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the layers on candidate shapes too: built, but not yet shown to beat the path they would replace."""
    get_op = functools.partial(get_transition_mlp_w_norm_op, candidates=True)
    monkeypatch.setattr(transition, "get_transition_mlp_w_norm_op", get_op)


def _shipped(site: _Site, seed: int = 0, **kwargs: object) -> TransitionLayer:
    """The site's loaded layer on SM90, or skip where the op builds no kernel for its shape."""
    skip_if_not_sm90()
    layer = _loaded(site, seed, **kwargs)
    if layer._w_norm_op is None:
        pytest.skip(f"transition_mlp_w_norm builds no C={site.width}, H={site.hidden} kernel for {site.kind}")
    return layer


@pytest.mark.usefixtures("candidates")
@pytest.mark.parametrize("site", _SITES, ids=str)
@pytest.mark.parametrize("mask_form", [None, "rows", "trailing"])
@pytest.mark.parametrize(("residual", "inplace"), [(False, False), (True, False), (True, True)])
def test_fused_layer_matches_its_unfused_path(site, mask_form, residual, inplace) -> None:
    """The layer at each model shape takes the kernel and matches its unfused path; masked rows pass exactly."""
    if site.kind == "msa" and mask_form != "rows":
        pytest.skip("MSATransition takes a [B, S, N] mask")
    if inplace and site.kind != "swiglu":
        pytest.skip("only Transition updates in place")
    layer = _shipped(site, seed=5)
    counter = layer._w_norm_op = _CountingOp(layer._w_norm_op)
    z, mask = _inputs(site, mask_form, seed=6)
    original = z.clone()
    with torch.inference_mode():
        expected = _unfused(layer, z, mask, residual)
        output = _call(layer, z, mask, residual, inplace)
    assert counter.calls == 1
    assert (output.data_ptr() == z.data_ptr()) == inplace
    torch.testing.assert_close(output, expected, atol=ATOL, rtol=RTOL)
    _assert_masked_rows(output, original, mask, residual)


def _layernorm_then_transition_mlp(layer: TransitionLayer, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """The residual call on the path the kernel replaces: the layer's LayerNorm, then the ``transition_mlp`` op where
    it ships for the call, else the layer's own linears (SM90 ships no ``transition_mlp`` image the kernel replaces)."""
    if isinstance(layer, Transition):
        op = layer._fused_mlp_ops.get((mask is not None, True))
        weights = (layer.fused_fc2_fc1.weight, None, layer.fc3.weight, None)
    else:
        op = layer._fused_mlp_ops[True]
        weights = (layer.linear_1.weight, layer.linear_1.bias, layer.linear_2.weight, layer.linear_2.bias)
    if op is None or not op.accepts(x, mask, *weights):
        return _unfused(layer, x, mask, residual=True)
    output = op(layer._w_norm_parts()[0](x), *weights, x, mask)
    assert output is not None
    return output


@pytest.mark.usefixtures("candidates")
@pytest.mark.parametrize("site", _SITES, ids=str)
def test_fused_layer_after_replace_with_fused_layernorm(site) -> None:
    """Models swap each ``nn.LayerNorm`` for a ``FusedLayerNorm`` sharing its parameters, then load the weights: the
    kernel runs the swapped module's weights and matches it followed by the layer's MLP.

    The ReLU layers take a masked residual call and the SwiGLU ones an unmasked one, as the models call them.
    """
    skip_if_not_sm90()
    layer = _layer(site)
    replace_with_fused_layernorm(layer)
    layer = layer.cuda().eval()
    assert type(layer._w_norm_parts()[0]) is FusedLayerNorm
    layer.load_state_dict(_released_state(layer, 14))
    if layer._w_norm_op is None:
        pytest.skip(f"transition_mlp_w_norm builds no C={site.width}, H={site.hidden} kernel for {site.kind}")
    counter = layer._w_norm_op = _CountingOp(layer._w_norm_op)
    z, mask = _inputs(site, None if site.kind == "swiglu" else "rows", seed=15)
    with torch.inference_mode():
        expected = _layernorm_then_transition_mlp(layer, z, mask)
        fused = layer(z, mask, residual=True)
    assert counter.calls == 1
    torch.testing.assert_close(fused, expected, atol=ATOL, rtol=RTOL)
    _assert_masked_rows(fused, z, mask, True)


@pytest.mark.usefixtures("candidates")
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("site", [_SITES[0], _SITES[1], _SITES[3], _SITES[4], _SITES[5], _SITES[7]], ids=str)
def test_fused_layer_runs_from_source_and_cubins(mode, site, monkeypatch) -> None:
    """Each layer kind on exactly one of the source and CUBIN paths."""
    layer = _shipped(site, seed=7)
    z, mask = _inputs(site, "rows", seed=8)
    if mode == "cubin":
        require_cubin_family("transition_mlp_w_norm")
    with torch.inference_mode():
        fused = run_cutedsl_test_mode(
            mode, monkeypatch, TransitionMlpWNormCuTe, w_norm_cutedsl, lambda: layer(z, mask, residual=True)
        )
        expected = _unfused(layer, z, mask, True)
    torch.testing.assert_close(fused, expected, atol=ATOL, rtol=RTOL)
    _assert_masked_rows(fused, z, mask, True)


@pytest.mark.usefixtures("candidates")
@pytest.mark.parametrize("site", [_SITES[0], _SITES[2], _SITES[4], _SITES[5], _SITES[11]], ids=str)
def test_fused_layer_refolds_after_loads(site) -> None:
    """The kernel runs the weights ``load_state_dict`` and ``post_load_weights`` leave, not the ones it folded first."""
    layer = _shipped(site, seed=9)
    z, mask = _inputs(site, "rows", seed=10)
    with torch.inference_mode():
        before = layer(z, mask, residual=True)
    layer.load_state_dict(_released_state(layer, 11))
    with torch.inference_mode():
        loaded = layer(z, mask, residual=True)
        torch.testing.assert_close(loaded, _unfused(layer, z, mask, True), atol=ATOL, rtol=RTOL)
    assert not torch.allclose(before, loaded, atol=ATOL, rtol=RTOL)
    with torch.no_grad():
        layer._w_norm_parts()[0].weight.mul_(0.5)
    layer.post_load_weights()
    with torch.inference_mode():
        torch.testing.assert_close(layer(z, mask, residual=True), _unfused(layer, z, mask, True), atol=ATOL, rtol=RTOL)


@pytest.mark.usefixtures("candidates")
@pytest.mark.parametrize("site", [_SITES[0], _SITES[2], _SITES[4], _SITES[5], _SITES[7]], ids=str)
def test_fused_layer_replays_under_cuda_graph_capture(site) -> None:
    """A captured call reads its input and writes its output, in place for ``Transition``, on every replay."""
    layer = _shipped(site, seed=12)
    z, mask = _inputs(site, "rows", seed=13)
    x = z.clone()
    inplace = site.kind == "swiglu"
    with torch.inference_mode():
        _call(layer, x, mask, True, inplace)
        x.copy_(z)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = _call(layer, x, mask, True, inplace)
        for scale in (1.0, -2.0):
            x.copy_(scale * z)
            expected = _unfused(layer, x, mask, True)
            graph.replay()
            torch.testing.assert_close(output, expected, atol=ATOL, rtol=RTOL)
    assert (output.data_ptr() == x.data_ptr()) == inplace
