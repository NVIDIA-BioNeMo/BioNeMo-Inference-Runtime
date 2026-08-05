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
"""The ``support_graph_optimization`` decorator.

Covers, for every decorated module class:

* **Default config content** — the decorator packages its tie points + per-dim
  input management into a ``graph_opt_default`` (a ``GraphOptimizationConfig``)
  whose ``input_routing_config`` ties the ``num_tokens`` dim, caps acceptance at
  1024, and carries the declared workspace/static names.
* **Signature validation** passes for every decorated class and
  rejects a tie name that is not a real ``forward`` parameter.
* **``Signature.bind`` normalization** maps positional calls to
  parameter names — one name-based spec serves both a keyword caller (OpenFold3)
  and a positional one (boltz-2).
* **Effective-range resolution** + rejecting an acceptance max
  above the largest capture bucket.
* **GPU parity smoke** — the decorator's config still drives a correct
  capture/replay through the real OpenFold3 pipeline.
"""

import pytest
import torch

import tests._torch.model_forwards.test_model_forward_with_cuda_graph as H
from tensorrt_bionemo._torch.graph_optimization.config import (
    CUDAGraphOptimizationConfig,
    GraphOptimizationConfig,
    GraphOptimizationMode,
    InputKeyMethod,
    InputRoutingConfigFactory,
    NamedDimTies,
    effective_ranges,
)
from tensorrt_bionemo._torch.graph_optimization.decorator import (
    bind_forward_args,
    support_graph_optimization,
    validate_spec_against_forward,
)
from tensorrt_bionemo._torch.layers.transformers.diffusion_transformer import (
    BoltzDiffusionTransformer,
    OpenFold3DiffusionTransformer,
    ProtenixDiffusionTransformer,
)
from tensorrt_bionemo._torch.layers.transformers.pairformer import PairformerModule
from tensorrt_bionemo._torch.modules.boltz.structure import DiffusionModule as BoltzDiffusionModule
from tensorrt_bionemo._torch.modules.openfold3.diffusion_module import DiffusionModule as OF3DiffusionModule
from tensorrt_bionemo._torch.modules.protenix.diffusion import ProtenixDiffusionModule
from tests.common.test_utils.model_forwards import _CKPT_ENV, _HF_CKPT, _model_weights_available

_SAMPLE_IDS = ("T1047s1",)

# (decorated class, expected workspace kwargs, expected static args). Every
# module ties ``num_tokens`` with a 1024 acceptance max; the transformers carry
# a ``buffers`` workspace, the pairformer additionally declares its masks static.
_MODULE_CASES = [
    pytest.param(PairformerModule, ["buffers"], ["mask", "pair_mask"], id="of3_pairformer"),
    pytest.param(OpenFold3DiffusionTransformer, ["buffers"], [], id="of3_token_transformer"),
    pytest.param(OF3DiffusionModule, [], [], id="of3_diffusion_module"),
    pytest.param(BoltzDiffusionTransformer, ["buffers"], [], id="boltz2_token_transformer"),
    pytest.param(BoltzDiffusionModule, [], [], id="boltz2_diffusion_module"),
    pytest.param(ProtenixDiffusionTransformer, ["buffers"], [], id="protenix_token_transformer"),
    pytest.param(ProtenixDiffusionModule, [], [], id="protenix_diffusion_module"),
]


@pytest.mark.parametrize("cls, workspaces, static_args", _MODULE_CASES)
def test_graph_opt_default_declares_num_tokens(cls, workspaces, static_args):
    """The decorator stores a ``graph_opt_default`` config whose routing ties the
    ``num_tokens`` dim, caps acceptance at 1024, and carries the declared
    workspace kwargs and static args."""
    cfg = cls.graph_opt_default
    assert isinstance(cfg, GraphOptimizationConfig)
    routing = cfg.input_routing_config
    assert [t.name for t in routing.named_dim_ties] == ["num_tokens"]
    assert [(a.name, a.dim_len_max) for a in routing.input_acceptance_dims] == [("num_tokens", 1024)]
    assert routing.internal_workspace_kwargs == workspaces
    assert routing.static_args == static_args


@pytest.mark.parametrize("cls, _workspaces, _static_args", _MODULE_CASES)
def test_decorated_spec_validates_against_forward(cls, _workspaces, _static_args):
    """Every declared tie/workspace/static name is a real forward
    parameter, so validation (run at decoration time) passes on a re-check."""
    validate_spec_against_forward(cls)


def test_validation_rejects_unknown_name():
    """A tie point that is not a forward parameter raises at decoration time."""
    import torch.nn as nn

    with pytest.raises(ValueError, match="not a parameter of forward"):

        @support_graph_optimization(
            named_dims=(NamedDimTies(name="num_tokens", input_dims=(("does_not_exist", (-2,)),)),),
            graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
            input_key_method=InputKeyMethod.EXACT,
        )
        class _Bad(nn.Module):  # noqa: N801
            def forward(self, s, z):  # no **kwargs to absorb it
                return s


def test_bind_forward_args_maps_positional_to_names():
    """Positional args normalize to parameter names (the boltz-2 pairformer case).

    Mirrors ``PairformerModule.forward(s, z, mask, pair_mask, ...)``: boltz-2
    passes s/z positionally, OpenFold3 by keyword; both must normalize to the
    same name-keyed dict so one name-based spec serves both.
    """

    def forward(s, z, mask, pair_mask, buffers=None, **kwargs):
        return s, z

    positional = bind_forward_args(forward, (1, 2), {"mask": 3, "pair_mask": 4})
    keyword = bind_forward_args(forward, (), {"s": 1, "z": 2, "mask": 3, "pair_mask": 4})
    assert positional == {"s": 1, "z": 2, "mask": 3, "pair_mask": 4}
    assert positional == keyword
    # Defaults for un-passed params are not fabricated.
    assert "buffers" not in positional


# ---------------------------------------------------------------------------
# Effective-range resolution + reject acceptance > largest bucket.
# ---------------------------------------------------------------------------
def _factory_with_tie() -> InputRoutingConfigFactory:
    factory = InputRoutingConfigFactory()
    factory.set_named_dim_ties([NamedDimTies(name="num_tokens", input_dims=(("s", (-2,)),))])
    return factory


def test_export_rejects_acceptance_above_largest_bucket():
    factory = _factory_with_tie()
    factory.set_input_acceptance_dim("num_tokens", 2048)  # above largest bucket
    factory.set_padded_dim("num_tokens", dim_len_min=1, dim_len_max=1024, num_intervals=8, multiple_of=128)
    with pytest.raises(ValueError, match="exceeds its largest capture bucket"):
        factory.export_config()


def test_effective_ranges_reports_live_max_and_covering_bucket():
    # Acceptance 300, buckets 1..1024 snapped to /128 -> covering bucket 384.
    factory = _factory_with_tie()
    factory.set_input_acceptance_dim("num_tokens", 300)
    factory.set_padded_dim(
        "num_tokens", dim_len_min=1, dim_len_max=1024, num_intervals=8, multiple_of=128, spacing_method="linear"
    )
    ranges = effective_ranges(factory.export_config())["num_tokens"]
    assert ranges["live_max"] == 300
    assert ranges["largest_bucket"] == 1024
    assert ranges["covering_bucket"] == 384  # smallest /128 bucket >= 300


# ---------------------------------------------------------------------------
# GPU parity smoke: the decorator-derived config still drives a correct graph.
# ---------------------------------------------------------------------------
def _decorator_pairformer_config(module_name, input_key_method, sample_ids, verify_capture=True):
    assert module_name == "structure_pairformer"
    return CUDAGraphOptimizationConfig(
        graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
        input_key_method=input_key_method,
        input_routing_config=H._routing_from_decorator(
            PairformerModule, bucket=input_key_method == InputKeyMethod.BUCKETED_SHAPES
        ),
        verify_capture=verify_capture,
        num_graphs_max_for_this_module=len(sample_ids),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="decorator GPU parity requires CUDA")
@pytest.mark.skipif(not H._SAMPLES_AVAILABLE, reason="sample data not found")
@pytest.mark.skipif(
    not _model_weights_available("openfold3", _ckpt_env=_CKPT_ENV, _hf_ckpt=_HF_CKPT),
    reason="openfold3 checkpoint unavailable",
)
def test_pairformer_decorator_gpu_parity(monkeypatch):
    monkeypatch.setattr(H, "_openfold3_graph_optimization_config", _decorator_pairformer_config)
    H._assert_cuda_graph_parity(
        "openfold3", "structure_pairformer", sample_ids=_SAMPLE_IDS, input_key_method=InputKeyMethod.EXACT
    )
