# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Upstream OpenFold/Boltz reference implementation, mirrored for parity tests.
# Kept in upstream style (star imports, forward refs), not held to these rules.
# ruff: noqa: B905, E402
import sys
from unittest.mock import MagicMock

import tests
from tests.common.test_utils.basic import path_for_package_in_repo

MOCK_MODULES = ["gemmi"]

# Inject mock modules BEFORE importing
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()
sys.modules["gemmi"].__version__ = "0.7.3"

# Now add 3rdparty to path and import
sys.path.insert(0, str(path_for_package_in_repo(tests).parent / "3rdparty/openfold-3"))

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.latent.msa_module import MSAModuleBlock as OF3OSS_MSAModuleBlock
from openfold3.core.model.latent.template_module import TemplatePairBlock as OF3OSS_TemplatePairBlock
from openfold3.core.model.layers.msa import MSAPairWeightedAveraging as OF3OSS_MSAPairWeightedAveraging
from openfold3.core.model.layers.outer_product_mean import OuterProductMean as OF3OSS_OuterProductMean
from openfold3.core.model.layers.transition import SwiGLUTransition as OF3OSS_SwiGLUTransition
from openfold3.core.model.layers.triangular_attention import TriangleAttention as OF3OSS_TriangleAttention
from openfold3.core.model.layers.triangular_multiplicative_update import (
    TriangleMultiplicativeUpdate as OF3OSS_TriangleMultiplicativeUpdate,
)

from tensorrt_bionemo.hubs import load_weights
from tests.common.test_utils.basic import setattr_safe


class RefMSAPairWeightedAveragingFromOF3OSS(OF3OSS_MSAPairWeightedAveraging):
    def __init__(
        self,
        c_in: int = 64,
        c_hidden: int = 8,
        c_z: int = 128,
        no_heads: int = 8,
        inf: float = 1e9,
        linear_init_params=lin_init.msa_pair_avg_init,
    ):
        super().__init__(
            c_in=c_in, c_hidden=c_hidden, c_z=c_z, no_heads=no_heads, inf=inf, linear_init_params=linear_init_params
        )

    @classmethod
    def load_weights(cls, model: str = "openfold3", layer_path: str = "msa_module.blocks.0", state_dict: dict = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        # infer class init params from state_dict
        c_in: int = None
        c_hidden: int = None
        c_z: int = None
        no_heads: int = None

        filter_state_dict = {}
        for layer_name, weight in state_dict.items():
            if layer_path in layer_name:
                layer_name_postfix = layer_name.replace(layer_path + ".", "")
                filter_state_dict[layer_name_postfix] = weight
                if layer_name_postfix == "linear_z.weight":
                    no_heads, c_z = weight.shape
                elif layer_name_postfix == "linear_v.weight":
                    a, c_in = weight.shape
                    c_hidden = a // no_heads

        m = cls(
            c_in=c_in,
            c_hidden=c_hidden,
            c_z=c_z,
            no_heads=no_heads,
            inf=1e9,
            linear_init_params=lin_init.msa_pair_avg_init,
        )
        m.load_state_dict(filter_state_dict)

        return m


class RefSwiGLUTransitionFromOF3OSS(OF3OSS_SwiGLUTransition):
    @classmethod
    def load_weights(cls, model: str = "openfold3", layer_path: str = "msa_module.blocks.0", state_dict: dict = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        weights_biases_path = [
            (f"{layer_path}.layer_norm.weight", f"{layer_path}.layer_norm.bias"),
            (f"{layer_path}.swiglu.linear_a.weight", None),
            (f"{layer_path}.swiglu.linear_b.weight", None),
            (f"{layer_path}.linear_out.weight", None),
        ]
        c_in = state_dict[f"{layer_path}.layer_norm.weight"].shape[0]
        n = state_dict[f"{layer_path}.swiglu.linear_a.weight"].shape[0] // c_in
        m = cls(c_in=c_in, n=n)
        layers = [m.layer_norm, m.swiglu.linear_a, m.swiglu.linear_b, m.linear_out]
        for (weights_path, bias_path), layer in zip(weights_biases_path, layers):
            if bias_path is not None:
                layer.bias.data.copy_(state_dict[bias_path])
            layer.weight.data.copy_(state_dict[weights_path])
        return m


class RefOuterProductMeanFromOF3OSS(OF3OSS_OuterProductMean):
    @classmethod
    def load_weights(cls, model: str = "openfold3", layer_path: str = "msa_module.blocks.0", state_dict: dict = None):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        c_m: int = None
        c_z: int = None
        c_hidden: int = None

        filter_state_dict = {}
        for layer_name, weight in state_dict.items():
            if layer_path in layer_name:
                layer_name_postfix = layer_name.replace(layer_path + ".", "")

                filter_state_dict[layer_name_postfix] = weight

                if layer_name_postfix == "layer_norm.weight":
                    c_m = weight.shape[-1]
                elif layer_name_postfix == "linear_1.weight":
                    c_hidden = weight.shape[0]
                elif layer_name_postfix == "linear_out.weight":
                    c_z = weight.shape[0]

        m = cls(c_m=c_m, c_z=c_z, c_hidden=c_hidden)
        m.load_state_dict(filter_state_dict)

        return m


class RefTriangleAttentionFromOF3OSS(OF3OSS_TriangleAttention):
    @classmethod
    def load_weights(
        cls,
        model: str = "openfold3",
        layer_path: str = "msa_module.blocks.0.pair_stack.tri_att_start",
        state_dict: dict = None,
        starting: bool = True,
    ):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        c_in: int = None
        c_hidden: int = None
        no_heads: int = None

        filter_state_dict = {}
        for layer_name, weight in state_dict.items():
            if layer_path in layer_name:
                layer_name_postfix = layer_name.replace(layer_path + ".", "")

                if layer_name_postfix == "layer_norm.weight":
                    c_in = weight.shape[0]
                elif layer_name_postfix == "linear_z.weight":
                    no_heads = weight.shape[0]
                elif layer_name_postfix == "mha.linear_q.weight":
                    c_hidden = weight.shape[0] // no_heads

                filter_state_dict[layer_name_postfix] = weight
        m = cls(
            c_in=c_in, c_hidden=c_hidden, no_heads=no_heads, starting=starting, linear_init_params=lin_init.tri_att_init
        )
        m.load_state_dict(filter_state_dict)

        return m


class RefTriangleMultiplicationFromOF3OSS(OF3OSS_TriangleMultiplicativeUpdate):
    @classmethod
    def load_weights(
        cls,
        model: str = "openfold3",
        layer_path: str = "msa_module.blocks.0.pair_stack.tri_att_start",
        state_dict: dict = None,
        _outgoing: bool = True,
    ):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        c_z: int = None
        c_hidden: int = None

        filter_state_dict = {}
        for layer_name, weight in state_dict.items():
            if layer_path in layer_name:
                layer_name_postfix = layer_name.replace(layer_path + ".", "")
                filter_state_dict[layer_name_postfix] = weight
                if layer_name_postfix == "linear_a_p.weight":
                    c_hidden, c_z = weight.shape

        m = cls(c_z=c_z, c_hidden=c_hidden, _outgoing=_outgoing)
        m.load_state_dict(filter_state_dict)

        return m


class RefMSAModuleBlockFromOF3OSS(OF3OSS_MSAModuleBlock):
    @classmethod
    def load_weights(
        cls, model: str = "openfold3", layer_path: str = "msa_module.blocks.0", state_dict: dict = None
    ) -> "RefMSAModuleBlockFromOF3OSS":
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        msa_att_row = RefMSAPairWeightedAveragingFromOF3OSS.load_weights(
            model=model, layer_path=f"{layer_path}.msa_att_row", state_dict=state_dict
        )

        msa_transition = RefSwiGLUTransitionFromOF3OSS.load_weights(
            model=model, layer_path=f"{layer_path}.msa_transition", state_dict=state_dict
        )

        outer_product_mean = RefOuterProductMeanFromOF3OSS.load_weights(
            model=model, layer_path=f"{layer_path}.outer_product_mean", state_dict=state_dict
        )

        tri_mul_out = RefTriangleMultiplicationFromOF3OSS.load_weights(
            model=model, layer_path=f"{layer_path}.pair_stack.tri_mul_out", state_dict=state_dict, _outgoing=True
        )
        tri_mul_in = RefTriangleMultiplicationFromOF3OSS.load_weights(
            model=model, layer_path=f"{layer_path}.pair_stack.tri_mul_in", state_dict=state_dict, _outgoing=False
        )
        tri_att_start = RefTriangleAttentionFromOF3OSS.load_weights(
            model=model, layer_path=f"{layer_path}.pair_stack.tri_att_start", state_dict=state_dict, starting=True
        )

        # for OF3 OSS, the adjustment for the 'end' version of tri att
        # is done outside of the TriangleAttention classes
        tri_att_end = RefTriangleAttentionFromOF3OSS.load_weights(
            model=model, layer_path=f"{layer_path}.pair_stack.tri_att_end", state_dict=state_dict, starting=True
        )
        pair_transition = RefSwiGLUTransitionFromOF3OSS.load_weights(
            model=model, layer_path=f"{layer_path}.pair_stack.pair_transition", state_dict=state_dict
        )
        m = cls(
            c_m=msa_att_row.c_in,
            c_z=outer_product_mean.c_z,
            c_hidden_msa_att=msa_att_row.c_hidden,
            c_hidden_opm=outer_product_mean.c_hidden,
            c_hidden_mul=tri_mul_out.c_hidden,
            c_hidden_pair_att=tri_att_start.c_hidden,
            no_heads_msa=msa_att_row.no_heads,
            no_heads_pair=tri_att_start.no_heads,
            transition_n=msa_transition.n,
            transition_type="swiglu",
            msa_dropout=0.0,
            pair_dropout=0.0,
            fuse_projection_weights=False,
            opm_first=False,
            inf=msa_att_row.inf,
            eps=1e-5 if not hasattr(msa_att_row, "eps") else msa_att_row.eps,
        )
        setattr_safe(m, "msa_att_row", msa_att_row)
        setattr_safe(m, "msa_transition", msa_transition)
        setattr_safe(m, "outer_product_mean", outer_product_mean)
        setattr_safe(m.pair_stack, "tri_mul_out", tri_mul_out)
        setattr_safe(m.pair_stack, "tri_mul_in", tri_mul_in)
        setattr_safe(m.pair_stack, "tri_att_start", tri_att_start)
        setattr_safe(m.pair_stack, "tri_att_end", tri_att_end)
        setattr_safe(m.pair_stack, "pair_transition", pair_transition)

        return m


class RefTemplatePairBlockFromOF3OSS(OF3OSS_TemplatePairBlock):
    @classmethod
    def load_weights(
        cls,
        model: str = "openfold3",
        layer_path: str = "template_embedder.template_pair_stack.blocks.0",
        state_dict: dict = None,
    ):
        if state_dict is None:
            state_dict = load_weights(model, local_files_only=False)

        tri_mul_out = RefTriangleMultiplicationFromOF3OSS.load_weights(
            model=model, layer_path=f"{layer_path}.tri_mul_out", state_dict=state_dict, _outgoing=True
        )
        tri_mul_in = RefTriangleMultiplicationFromOF3OSS.load_weights(
            model=model, layer_path=f"{layer_path}.tri_mul_in", state_dict=state_dict, _outgoing=False
        )
        tri_att_start = RefTriangleAttentionFromOF3OSS.load_weights(
            model=model, layer_path=f"{layer_path}.tri_att_start", state_dict=state_dict, starting=True
        )

        # for OF3 OSS, the adjustment for the 'end' version of tri att
        # is done outside of the TriangleAttention classes
        tri_att_end = RefTriangleAttentionFromOF3OSS.load_weights(
            model=model, layer_path=f"{layer_path}.tri_att_end", state_dict=state_dict, starting=True
        )
        pair_transition = RefSwiGLUTransitionFromOF3OSS.load_weights(
            model=model, layer_path=f"{layer_path}.pair_transition", state_dict=state_dict
        )

        # ``ckpt_per_template`` controls activation-checkpointing per
        # template inside ``TemplatePairBlock`` and is wired off in
        # OF3's all-atom model config (see
        # ``openfold3/projects/of3_all_atom/config/model_config.py``).
        # The OSS class made it a required positional argument, so we
        # must pass it explicitly.
        m = cls(
            c_t=tri_mul_out.c_z,
            c_hidden_tri_att=tri_att_start.c_hidden,
            c_hidden_tri_mul=tri_mul_out.c_hidden,
            no_heads=tri_att_start.no_heads,
            pair_transition_n=pair_transition.n,
            tri_mul_first=True,
            transition_type="swiglu",
            dropout_rate=0.0,
            fuse_projection_weights=False,
            ckpt_per_template=False,
            inf=1e9,
        )

        setattr_safe(m, "tri_mul_out", tri_mul_out)
        setattr_safe(m, "tri_mul_in", tri_mul_in)
        setattr_safe(m, "tri_att_start", tri_att_start)
        setattr_safe(m, "tri_att_end", tri_att_end)
        setattr_safe(m, "pair_transition", pair_transition)
        return m
