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

from unittest.mock import patch

import pytest
import torch

from bionemo_ir._torch.attention_backend.pairwise_attention import _config as pairwise_config
from bionemo_ir._torch.layers.transformers.diffusion_transformer import (
    BoltzDiffusionTransformer,
    OpenFold3DiffusionTransformer,
    ProtenixDiffusionTransformer,
)
from bionemo_ir._torch.utils.kernel import get_config_file_name, load_kernel_configs
from bionemo_ir.configs import DiffusionTransformerConfig
from tests._torch import SM_VERSION

# 37 tokens pad the CuTeDSL key axis to 40; head_dim 32 has CuTeDSL configs for SM80-SM90.
TOKENS, SAMPLES, DIM, HEADS, PAIR, LAYERS = 37, 3, 128, 4, 16, 2


def _model(model_type: type, backend: str) -> torch.nn.Module:
    config = DiffusionTransformerConfig(
        num_blocks=LAYERS,
        num_heads=HEADS,
        dim=DIM,
        dim_single_cond=DIM,
        dim_pairwise=PAIR,
        bias_proj=True,
        precompute_bias=True,
        dtype="bfloat16",
        pairwise_attention_backend=backend,
        version="v1",
    )
    model = model_type(config).cuda().eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim > 1:
                param.normal_(std=0.02)
            elif name.endswith("weight"):
                param.fill_(1)
            else:
                param.zero_()
    return model


def _inputs(model_type: type) -> tuple[dict, torch.Tensor]:
    """Token-path inputs in the layout each diffusion module passes, and the step-invariant pair."""
    bf16 = {"device": "cuda", "dtype": torch.bfloat16}
    a = torch.randn(1, SAMPLES, TOKENS, DIM, **bf16)
    s = torch.randn(1, 1, TOKENS, DIM, **bf16)
    if model_type is BoltzDiffusionTransformer:
        # Boltz projects every layer's bias once per rollout, then lays it out per call.
        z = torch.randn(1, 1, TOKENS, TOKENS, LAYERS * HEADS, **bf16)
        return {"a": a, "s": s.expand_as(a), "mask": a.new_ones(1, 1, TOKENS)}, z
    z = torch.randn(1, TOKENS, TOKENS, PAIR, **bf16)
    if model_type is ProtenixDiffusionTransformer:
        # Protenix folds samples into the batch and broadcasts the pair over them.
        return {
            "a": a.reshape(SAMPLES, TOKENS, DIM),
            "s": s.expand_as(a).reshape(SAMPLES, TOKENS, DIM),
            "mask": a.new_ones(1, TOKENS),
        }, z
    return {"a": a, "s": s, "mask": a.new_ones(1, TOKENS)}, z


@pytest.mark.parametrize("backend", ["SDPA", "CuTeDSL"])
@pytest.mark.parametrize(
    "model_type", [OpenFold3DiffusionTransformer, ProtenixDiffusionTransformer, BoltzDiffusionTransformer]
)
def test_prepared_token_biases_match_per_call_projection(model_type: type, backend: str) -> None:
    if backend == "CuTeDSL" and torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("CuTeDSL attention needs SM80+")
    head_dim = DIM // HEADS
    config_file = get_config_file_name(SM_VERSION, D=head_dim)
    if backend == "CuTeDSL" and load_kernel_configs(pairwise_config._PW_CONFIGS_DIR, config_file) is None:
        pytest.skip(f"No CuTeDSL pairwise-attention config for SM{SM_VERSION}, head_dim={head_dim}")
    torch.manual_seed(0)
    model = _model(model_type, backend)
    kwargs, z = _inputs(model_type)
    # The prepared path must neither project nor lay out the pair again.
    rebuild = "_pad_bias" if model_type is BoltzDiffusionTransformer else "_precompute_all_biases"
    with torch.inference_mode():
        expected = model(**kwargs, z=z)
        biases = model.prepare_pair_biases(z)
        assert len(biases) == LAYERS
        with patch.object(model, rebuild, side_effect=AssertionError("pair bias rebuilt")):
            actual = model(**kwargs, z=z, prepared_pair_biases=biases)
        with pytest.raises(ValueError, match="count"):
            model(**kwargs, z=z, prepared_pair_biases=biases[:1])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
