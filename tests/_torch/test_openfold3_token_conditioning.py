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

"""OpenFold3 token conditioning projected for all layers at once."""

import pytest
import torch

from bionemo_ir._torch.layers.normalization import replace_with_fused_layernorm
from bionemo_ir._torch.layers.transformers.diffusion_transformer import OpenFold3DiffusionTransformer
from bionemo_ir.models.openfold3.config import OpenFold3Config


def _stack(post_layer_norm: bool = False) -> OpenFold3DiffusionTransformer:
    config = OpenFold3Config().diffusion_module_config.diffusion_transformer_config.token_transformer.model_copy(
        update={
            "num_blocks": 2,
            "dtype": "bfloat16",
            "pairwise_attention_backend": "SDPA",
            "post_layer_norm": post_layer_norm,
        }
    )
    stack = OpenFold3DiffusionTransformer(config).cuda().eval()
    with torch.no_grad():
        for name, parameter in stack.named_parameters():
            parameter.normal_(mean=1.0 if name.endswith("s_norm.weight") else 0.0, std=0.02)
    replace_with_fused_layernorm(stack)
    return stack


def _rmse_ratio(actual: torch.Tensor, expected: torch.Tensor) -> float:
    expected = expected.float()
    return ((actual.float() - expected).pow(2).mean().sqrt() / expected.pow(2).mean().sqrt().clamp_min(1e-12)).item()


@pytest.mark.parametrize("tokens", [29, 373])
@pytest.mark.parametrize("samples", [1, 5])
def test_openfold3_token_conditioning_matches_layers(tokens: int, samples: int) -> None:
    """The batched conditioning and the stack with it match the layers' own projections to bf16 rounding.

    The folded LayerNorm scale rounds with the weight instead of the activations, so the match is not exact.
    """
    torch.manual_seed(42)
    stack = _stack()
    dtype, device = torch.bfloat16, torch.device("cuda")
    s = torch.randn(1, 1, tokens, 384, dtype=dtype, device=device)
    a = torch.randn(1, samples, tokens, 768, dtype=dtype, device=device)
    z = torch.randn(1, 1, tokens, tokens, 128, dtype=dtype, device=device) * 0.1
    mask = torch.ones(1, 1, tokens, dtype=dtype, device=device)
    mask[..., -1] = 0

    def check_layers(weights: tuple[torch.Tensor, ...]) -> None:
        conditioning = stack.prepare_token_conditioning(s, weights)
        assert conditioning is not None, "the batched path did not engage"
        for layer, projected in zip(stack.layers, conditioning, strict=True):
            for adaln, scale, shift in (
                (layer.adaln, projected.attn_scale, projected.attn_shift),
                (layer.transition.adaln, projected.transition_scale, projected.transition_shift),
            ):
                assert _rmse_ratio(torch.cat((scale, shift), dim=-1), adaln.project_condition(s)) < 1e-2
            assert _rmse_ratio(projected.attn_gate, layer.pair_bias_attn.output_projection(s)) < 1e-2

    with torch.inference_mode():
        weights = stack.conditioning_weights()
        assert weights is not None
        check_layers(weights)
        biases = stack.prepare_pair_biases(z)
        actual = stack(a.clone(), s, mask=mask, prepared_pair_biases=biases, prepared_conditioning_weights=weights)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(OpenFold3DiffusionTransformer, "prepare_token_conditioning", lambda self, *args: None)
            expected = stack(a.clone(), s, mask=mask, prepared_pair_biases=biases)
        assert _rmse_ratio(actual, expected) < 1e-2

        # The rollout cache reads the parameters again for each rollout.
        stack.layers[0].adaln.fused_s_scale_s_bias.bias.add_(0.5)
        stack.layers[1].pair_bias_attn.output_projection.weight.mul_(2.0)
        check_layers(stack.conditioning_weights())


def test_openfold3_token_conditioning_with_post_norm_falls_back() -> None:
    """Layers with a post LayerNorm keep their own conditioning projections."""
    stack = _stack(post_layer_norm=True)
    with torch.inference_mode():
        assert stack.conditioning_weights() is None
        s = torch.randn(1, 29, 384, dtype=torch.bfloat16, device="cuda")
        assert stack.prepare_token_conditioning(s) is None
