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
from torch import nn

import bionemo_ir._torch.layers.conditioning as conditioning_module
from bionemo_ir._torch.layers.conditioning import DiffusionConditioning
from bionemo_ir._torch.layers.normalization import replace_with_fused_layernorm
from bionemo_ir._torch.modules.openfold3.diffusion_module import DiffusionModule
from bionemo_ir._torch.modules.openfold3.utils.relpos import relpos_complex
from bionemo_ir._torch.utils import (
    CHUNK_REGISTRY,
    DIFFUSION_CONDITIONING_PROJECTION,
    DIFFUSION_PAIR_TRANSITION,
)


class _ScaledTransition(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return value * self.scale


def test_openfold3_diffusion_pair_transition_chunks_token_rows() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device())
    c_z, tokens = 128, 16
    module = DiffusionConditioning(
        c_s_input=32,
        c_s=64,
        c_z=c_z,
        c_fourier_emb=32,
        max_relative_idx=32,
        max_relative_chain=2,
        sigma_data=16.0,
        dtype=torch.bfloat16,
    ).to(device)

    registry_policy = CHUNK_REGISTRY.get(DIFFUSION_PAIR_TRANSITION)
    assert registry_policy is not None
    assert registry_policy.dim == 1
    assert all(layer.auto_chunk_policy is not None for layer in module.transition_z)
    assert all(layer.auto_chunk_policy.dim == 2 for layer in module.transition_z)

    pair = torch.randn(1, 1, tokens, tokens, c_z, dtype=torch.bfloat16, device=device)
    with torch.inference_mode():
        for layer in module.transition_z:
            assert layer.auto_chunk_policy is not None
            layer.auto_chunk_policy = layer.auto_chunk_policy.replace(chunk_size=7, min_size=1)
            for parameter in layer.parameters():
                parameter.normal_(mean=0.0, std=0.1)
            expected = layer._forward_impl(pair)
            actual = layer(pair)
            torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
            pair = pair + actual


def test_openfold3_diffusion_pair_residual_updates_owned_inference_storage() -> None:
    module = DiffusionConditioning(
        c_s_input=8,
        c_s=16,
        c_z=8,
        c_fourier_emb=8,
        max_relative_idx=2,
        max_relative_chain=1,
        sigma_data=16.0,
        dtype=torch.float32,
    )
    module.transition_z = torch.nn.ModuleList([_ScaledTransition(0.5), _ScaledTransition(0.25)])
    module.transition_s = torch.nn.ModuleList()
    pair = torch.randn(1, 4, 4, 8)
    expected = pair + pair * 0.5
    expected = expected + expected * 0.25
    pair_pointer = pair.data_ptr()

    with torch.inference_mode():
        actual_pair = module._apply_pair_transitions(
            pair,
            torch.ones(1, 4),
            inplace_safe=True,
        )

    assert actual_pair.data_ptr() == pair_pointer
    assert torch.equal(actual_pair, expected)


def test_openfold3_diffusion_pair_residual_avoids_capture_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    module = DiffusionConditioning(
        c_s_input=8,
        c_s=16,
        c_z=8,
        c_fourier_emb=8,
        max_relative_idx=2,
        max_relative_chain=1,
        sigma_data=16.0,
        dtype=torch.float32,
    ).to(device)
    module.transition_z = torch.nn.ModuleList([_ScaledTransition(0.5)])
    module.transition_s = torch.nn.ModuleList()
    pair = torch.randn(1, 4, 4, 8, device=device)
    original = pair.clone()
    expected = pair + pair * 0.5

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with torch.inference_mode():
        actual_pair = module._apply_pair_transitions(
            pair,
            torch.ones(1, 4, device=device),
            inplace_safe=True,
        )

    assert actual_pair.data_ptr() != pair.data_ptr()
    assert torch.equal(pair, original)
    assert torch.equal(actual_pair, expected)


def test_openfold3_diffusion_pair_projection_chunks_token_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(1)
    device = torch.device("cuda", torch.cuda.current_device())
    c_z, tokens = 8, 16
    module = DiffusionConditioning(
        c_s_input=8,
        c_s=16,
        c_z=c_z,
        c_fourier_emb=8,
        max_relative_idx=2,
        max_relative_chain=1,
        sigma_data=16.0,
        dtype=torch.float32,
    ).to(device)
    replace_with_fused_layernorm(module)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.1)

    pair = torch.randn(1, 1, tokens, tokens, c_z, device=device)
    batch = {
        name: torch.randint(0, 4, (1, 1, tokens), device=device)
        for name in ("residue_index", "asym_id", "entity_id", "token_index", "sym_id")
    }
    relpos = relpos_complex(batch, module.max_relative_idx, module.max_relative_chain)

    policy = CHUNK_REGISTRY.get(DIFFUSION_CONDITIONING_PROJECTION)
    assert policy is not None
    assert policy.dim == 2
    module.pair_projection_chunk_policy = policy.replace(enabled=False)
    with torch.inference_mode():
        expected = module._project_pair_inputs_dense(pair, relpos)
        torch.testing.assert_close(module._project_pair_inputs(pair, batch), expected, atol=0, rtol=0)

    seen_rows: list[int] = []
    encoded_rows: list[int] = []

    def record_relative_positions(
        batch: dict[str, torch.Tensor],
        max_relative_idx: int,
        max_relative_chain: int,
        *,
        row_slice: slice | None = None,
    ) -> torch.Tensor:
        result = relpos_complex(batch, max_relative_idx, max_relative_chain, row_slice=row_slice)
        encoded_rows.append(result.shape[-3])
        return result

    monkeypatch.setattr(conditioning_module, "relpos_complex", record_relative_positions)

    def record_rows(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        seen_rows.append(inputs[0].shape[-3])

    handle = module.layer_norm_z.register_forward_pre_hook(record_rows)
    module.pair_projection_chunk_policy = policy.replace(chunk_size=7, min_size=1)
    try:
        with torch.inference_mode():
            actual = module._project_pair_inputs(pair, batch)
    finally:
        handle.remove()

    assert seen_rows == [7, 7, 2]
    assert encoded_rows == [7, 7, 2]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    with torch.inference_mode():
        host_actual = module._project_pair_inputs(pair.cpu(), batch)
    assert torch.equal(host_actual, actual)


def test_openfold3_diffusion_conditioning_split_is_exact_with_host_pair() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda", torch.cuda.current_device())
    c_s_input, c_s, c_z, tokens = 8, 16, 8, 16
    module = DiffusionConditioning(
        c_s_input=c_s_input,
        c_s=c_s,
        c_z=c_z,
        c_fourier_emb=8,
        max_relative_idx=2,
        max_relative_chain=1,
        sigma_data=16.0,
        dtype=torch.float32,
    ).to(device)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.1)

    projection_policy = CHUNK_REGISTRY.get(DIFFUSION_CONDITIONING_PROJECTION)
    assert projection_policy is not None
    module.pair_projection_chunk_policy = projection_policy.replace(chunk_size=7, min_size=1)
    for layer in module.transition_z:
        assert layer.auto_chunk_policy is not None
        layer.auto_chunk_policy = layer.auto_chunk_policy.replace(chunk_size=7, min_size=1)

    batch = {
        name: torch.randint(0, 4, (1, 1, tokens), device=device)
        for name in ("residue_index", "asym_id", "entity_id", "token_index", "sym_id")
    }
    batch["token_mask"] = torch.ones(1, 1, tokens, device=device)
    si_input = torch.randn(1, 1, tokens, c_s_input, device=device)
    si_trunk = torch.randn(1, 1, tokens, c_s, device=device)
    zij_trunk = torch.randn(1, 1, tokens, tokens, c_z, device=device)
    noise_level = torch.tensor([[1.5]], device=device)

    with torch.inference_mode():
        expected_si, expected_zij = module(
            batch=batch,
            t=noise_level,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
        )
        actual_zij = module.prepare_pair(batch=batch, zij_trunk=zij_trunk.cpu())
        actual_si = module.forward_single(
            batch=batch,
            t=noise_level,
            si_input=si_input,
            si_trunk=si_trunk,
        )

    assert torch.equal(actual_si, expected_si)
    assert torch.equal(actual_zij, expected_zij)


def test_openfold3_diffusion_pair_projection_preserves_autocast_dtype() -> None:
    torch.manual_seed(13)
    device = torch.device("cuda", torch.cuda.current_device())
    tokens, c_z = 8, 8
    module = DiffusionConditioning(
        c_s_input=8,
        c_s=16,
        c_z=c_z,
        c_fourier_emb=8,
        max_relative_idx=2,
        max_relative_chain=1,
        sigma_data=16.0,
        dtype=torch.float32,
    ).to(device)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.1)

    pair = torch.randn(1, 1, tokens, tokens, c_z, device=device)
    batch = {
        name: torch.randint(0, 4, (1, 1, tokens), device=device)
        for name in ("residue_index", "asym_id", "entity_id", "token_index", "sym_id")
    }
    policy = CHUNK_REGISTRY.get(DIFFUSION_CONDITIONING_PROJECTION)
    assert policy is not None

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        module.pair_projection_chunk_policy = policy.replace(enabled=False)
        dense = module._project_pair_inputs(pair, batch)
        module.pair_projection_chunk_policy = policy.replace(chunk_size=3, min_size=1)
        chunked = module._project_pair_inputs(pair, batch)

    assert dense.dtype == chunked.dtype == torch.bfloat16
    torch.testing.assert_close(chunked, dense, atol=3e-3, rtol=3e-3)


class _ConditioningDispatchProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.full_calls = 0
        self.single_calls = 0

    def forward(self, *, si_trunk: torch.Tensor, zij_trunk: torch.Tensor, **_kwargs):
        self.full_calls += 1
        return torch.zeros_like(si_trunk), torch.zeros_like(zij_trunk)

    def forward_single(self, **_kwargs):
        self.single_calls += 1
        raise AssertionError("an unconditioned call must ignore the prepared pair")


class _AtomEncoderProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pairs: list[torch.Tensor] = []

    def forward(self, *, rl: torch.Tensor, zij_trunk: torch.Tensor, **_kwargs):
        self.pairs.append(zij_trunk.clone())
        ai = torch.zeros((*rl.shape[:-2], zij_trunk.shape[-2], zij_trunk.shape[-1]), device=rl.device)
        return ai, torch.empty(0, device=rl.device), torch.empty(0, device=rl.device), torch.empty(0, device=rl.device)


class _DiffusionTransformerProbe(nn.Module):
    dtype = torch.float32

    def forward(self, *, a: torch.Tensor, **_kwargs) -> torch.Tensor:
        return a


class _AtomDecoderProbe(nn.Module):
    def forward(self, *, ai: torch.Tensor, **_kwargs) -> torch.Tensor:
        return torch.zeros((*ai.shape[:-1], 3), device=ai.device)


def test_unconditioned_diffusion_ignores_supplied_prepared_pair() -> None:
    module = DiffusionModule.__new__(DiffusionModule)
    nn.Module.__init__(module)
    conditioning = _ConditioningDispatchProbe()
    encoder = _AtomEncoderProbe()
    module.diffusion_conditioning = conditioning
    module.atom_attn_enc = encoder
    module.layer_norm_s = nn.Identity()
    module.linear_s = nn.Identity()
    module.diffusion_transformer = _DiffusionTransformerProbe()
    module.layer_norm_a = nn.Identity()
    module.atom_attn_dec = _AtomDecoderProbe()
    module.sigma_data = 16.0
    module.sq_sigma_data = 16.0**2

    batch = {"atom_mask": torch.ones(1, 1, 2)}
    kwargs = {
        "batch": batch,
        "xl_noisy": torch.randn(1, 1, 2, 3),
        "token_mask": torch.ones(1, 1, 2),
        "atom_mask": batch["atom_mask"],
        "t": torch.ones(1, 1),
        "si_input": torch.randn(1, 1, 2, 4),
        "si_trunk": torch.randn(1, 1, 2, 4),
        "zij_trunk": torch.randn(1, 1, 2, 2, 4),
        "attn_metadata": None,
        "use_conditioning": False,
    }

    with torch.inference_mode():
        without_cache = module(**kwargs)
        with_cache = module(**kwargs, prepared_zij=torch.full_like(kwargs["zij_trunk"], 123.0))

    assert conditioning.full_calls == 2
    assert conditioning.single_calls == 0
    assert len(encoder.pairs) == 2
    assert torch.equal(encoder.pairs[0], encoder.pairs[1])
    assert not torch.any(encoder.pairs[1] == 123.0)
    assert torch.equal(with_cache, without_cache)
