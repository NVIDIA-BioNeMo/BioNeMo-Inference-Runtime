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
"""OSS Protenix reference layers, loaded from the ``3rdparty/protenix`` submodule.

Mirrors ``tests/common/test_utils/openfold3/ref_layers_from_oss.py``: injects
the pinned OSS source (Protenix ``v2.0.0``) on ``sys.path``, stubs missing
optional dependencies, and exposes the OSS atom-transformer classes plus thin
``Ref*FromOSS`` wrappers with a seeded ``build`` helper for module-equivalence
tests.
"""

import os
import sys
from unittest.mock import MagicMock

import torch

import tests
from tests.common.test_utils.basic import path_for_package_in_repo

# Force the torch LayerNorm path: Protenix's default ``fast_layernorm`` triggers
# a blocking JIT CUDA-extension build on import.
os.environ.setdefault("LAYERNORM_TYPE", "torch")

# Optional deps that are not needed for the atom-transformer modules.
MOCK_MODULES = ["gemmi"]
for _mod_name in MOCK_MODULES:
    sys.modules.setdefault(_mod_name, MagicMock())
if isinstance(sys.modules.get("gemmi"), MagicMock):
    sys.modules["gemmi"].__version__ = "0.7.3"

sys.path.insert(0, str(path_for_package_in_repo(tests).parent / "3rdparty/protenix"))

from protenix.model.modules.confidence import ConfidenceHead as ProtenixOSS_ConfidenceHead  # noqa: E402
from protenix.model.modules.diffusion import DiffusionConditioning as ProtenixOSS_DiffusionConditioning  # noqa: E402
from protenix.model.modules.diffusion import DiffusionModule as ProtenixOSS_DiffusionModule  # noqa: E402
from protenix.model.modules.embedders import ConstraintEmbedder as ProtenixOSS_ConstraintEmbedder  # noqa: E402
from protenix.model.modules.embedders import (  # noqa: E402 -- import after sys.path insert for the vendored 3rdparty/protenix package
    RelativePositionEncoding as ProtenixOSS_RelativePositionEncoding,
)
from protenix.model.modules.head import DistogramHead as ProtenixOSS_DistogramHead  # noqa: E402
from protenix.model.modules.pairformer import MSAModule as ProtenixOSS_MSAModule  # noqa: E402
from protenix.model.modules.pairformer import PairformerStack as ProtenixOSS_PairformerStack  # noqa: E402
from protenix.model.modules.pairformer import TemplateEmbedder as ProtenixOSS_TemplateEmbedder  # noqa: E402
from protenix.model.modules.transformer import AtomAttentionDecoder as ProtenixOSS_AtomAttentionDecoder  # noqa: E402
from protenix.model.modules.transformer import AtomAttentionEncoder as ProtenixOSS_AtomAttentionEncoder  # noqa: E402
from protenix.model.modules.transformer import AtomTransformer as ProtenixOSS_AtomTransformer  # noqa: E402
from protenix.model.protenix import update_input_feature_dict  # noqa: E402,F401
from protenix.model.sample_confidence import compute_contact_prob as oss_compute_contact_prob  # noqa: E402,F401
from protenix.model.sample_confidence import (  # noqa: E402 -- import after sys.path insert for the vendored 3rdparty/protenix package
    compute_full_data_and_summary as oss_compute_full_data_and_summary,  # noqa: F401
)


class RefProtenixAtomTransformerFromOSS(ProtenixOSS_AtomTransformer):
    """OSS ``AtomTransformer`` (AF3 Algorithm 7) with a seeded builder."""

    @classmethod
    def build(
        cls,
        *,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        seed: int = 42,
    ) -> "RefProtenixAtomTransformerFromOSS":
        torch.manual_seed(seed)
        return cls(
            c_atom=c_atom, c_atompair=c_atompair, n_blocks=n_blocks, n_heads=n_heads, n_queries=n_queries, n_keys=n_keys
        )


class RefProtenixAtomAttentionEncoderFromOSS(ProtenixOSS_AtomAttentionEncoder):
    """OSS ``AtomAttentionEncoder`` (AF3 Algorithm 5) with a seeded builder."""

    @classmethod
    def build(
        cls,
        *,
        has_coords: bool = False,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_s: int = 384,
        c_z: int = 128,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        seed: int = 42,
    ) -> "RefProtenixAtomAttentionEncoderFromOSS":
        torch.manual_seed(seed)
        return cls(
            has_coords=has_coords,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s=c_s,
            c_z=c_z,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_queries=n_queries,
            n_keys=n_keys,
        )


class RefProtenixRelativePositionEncodingFromOSS(ProtenixOSS_RelativePositionEncoding):
    """OSS ``RelativePositionEncoding`` (AF3 Algorithm 3) with a seeded builder."""

    @classmethod
    def build(
        cls, *, r_max: int = 32, s_max: int = 2, c_z: int = 256, seed: int = 42
    ) -> "RefProtenixRelativePositionEncodingFromOSS":
        torch.manual_seed(seed)
        return cls(r_max=r_max, s_max=s_max, c_z=c_z)


class RefProtenixDistogramHeadFromOSS(ProtenixOSS_DistogramHead):
    """OSS ``DistogramHead`` (AF3 Algorithm 1, line 17) with a seeded builder."""

    @classmethod
    def build(cls, *, c_z: int = 256, no_bins: int = 64, seed: int = 42) -> "RefProtenixDistogramHeadFromOSS":
        torch.manual_seed(seed)
        return cls(c_z=c_z, no_bins=no_bins)


class RefProtenixConfidenceHeadFromOSS(ProtenixOSS_ConfidenceHead):
    """OSS ``ConfidenceHead`` (AF3 Algorithm 31) with a seeded builder."""

    @classmethod
    def build(
        cls,
        *,
        n_blocks: int = 4,
        c_s: int = 384,
        c_z: int = 256,
        c_s_inputs: int = 449,
        max_atoms_per_token: int = 24,
        hidden_scale_up: bool = True,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
        stop_gradient: bool = True,
        seed: int = 42,
    ) -> "RefProtenixConfidenceHeadFromOSS":
        torch.manual_seed(seed)
        return cls(
            n_blocks=n_blocks,
            c_s=c_s,
            c_z=c_z,
            c_s_inputs=c_s_inputs,
            max_atoms_per_token=max_atoms_per_token,
            hidden_scale_up=hidden_scale_up,
            distance_bin_start=distance_bin_start,
            distance_bin_end=distance_bin_end,
            distance_bin_step=distance_bin_step,
            stop_gradient=stop_gradient,
        )


class RefProtenixTemplateEmbedderFromOSS(ProtenixOSS_TemplateEmbedder):
    """OSS ``TemplateEmbedder`` (AF3 Algorithm 16) with a seeded builder."""

    @classmethod
    def build(
        cls,
        *,
        n_blocks: int = 2,
        c: int = 64,
        c_z: int = 256,
        num_intermediate_factor: int = 2,
        hidden_scale_up: bool = True,
        seed: int = 42,
    ) -> "RefProtenixTemplateEmbedderFromOSS":
        torch.manual_seed(seed)
        return cls(
            n_blocks=n_blocks,
            c=c,
            c_z=c_z,
            num_intermediate_factor=num_intermediate_factor,
            hidden_scale_up=hidden_scale_up,
        )


class RefProtenixConstraintEmbedderFromOSS(ProtenixOSS_ConstraintEmbedder):
    """OSS ``ConstraintEmbedder`` with a seeded builder.

    Enables the three ``LinearNoBias`` sub-embedders (pocket / contact /
    contact-atom); substructure stays disabled. OSS zero-inits the projections,
    so callers should randomize the weights for a non-trivial comparison.
    """

    @classmethod
    def build(
        cls,
        *,
        c_constraint_z: int = 32,
        pocket: bool = True,
        contact: bool = True,
        contact_atom: bool = True,
        seed: int = 42,
    ) -> "RefProtenixConstraintEmbedderFromOSS":
        torch.manual_seed(seed)
        return cls(
            pocket_embedder={"enable": pocket, "c_z_input": 1},
            contact_embedder={"enable": contact, "c_z_input": 2},
            contact_atom_embedder={"enable": contact_atom, "c_z_input": 2},
            substructure_embedder={"enable": False},
            c_constraint_z=c_constraint_z,
            initialize_method="zero",
        )


class RefProtenixPairformerStackFromOSS(ProtenixOSS_PairformerStack):
    """OSS ``PairformerStack`` (AF3 Algorithm 17) with a seeded builder."""

    @classmethod
    def build(
        cls,
        *,
        n_blocks: int = 2,
        n_heads: int = 4,
        c_z: int = 64,
        c_s: int = 64,
        hidden_scale_up: bool = True,
        seed: int = 42,
    ) -> "RefProtenixPairformerStackFromOSS":
        torch.manual_seed(seed)
        return cls(n_blocks=n_blocks, n_heads=n_heads, c_z=c_z, c_s=c_s, hidden_scale_up=hidden_scale_up)


class RefProtenixDiffusionConditioningFromOSS(ProtenixOSS_DiffusionConditioning):
    """OSS ``DiffusionConditioning`` (AF3 Algorithm 21) with a seeded builder."""

    @classmethod
    def build(
        cls,
        *,
        sigma_data: float = 16.0,
        c_z: int = 256,
        c_s: int = 384,
        c_s_inputs: int = 449,
        c_noise_embedding: int = 256,
        seed: int = 42,
    ) -> "RefProtenixDiffusionConditioningFromOSS":
        torch.manual_seed(seed)
        return cls(sigma_data=sigma_data, c_z=c_z, c_s=c_s, c_s_inputs=c_s_inputs, c_noise_embedding=c_noise_embedding)


class RefProtenixDiffusionModuleFromOSS(ProtenixOSS_DiffusionModule):
    """OSS ``DiffusionModule`` (AF3 Algorithm 20) with a seeded builder."""

    @classmethod
    def build(
        cls,
        *,
        sigma_data: float = 16.0,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 32,
        c_s: int = 32,
        c_z: int = 16,
        c_s_inputs: int = 16,
        atom_n_blocks: int = 2,
        atom_n_heads: int = 4,
        token_n_blocks: int = 2,
        token_n_heads: int = 4,
        seed: int = 42,
    ) -> "RefProtenixDiffusionModuleFromOSS":
        torch.manual_seed(seed)
        return cls(
            sigma_data=sigma_data,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            c_s=c_s,
            c_z=c_z,
            c_s_inputs=c_s_inputs,
            atom_encoder={"n_blocks": atom_n_blocks, "n_heads": atom_n_heads},
            transformer={"n_blocks": token_n_blocks, "n_heads": token_n_heads},
            atom_decoder={"n_blocks": atom_n_blocks, "n_heads": atom_n_heads},
        )


class RefProtenixAtomAttentionDecoderFromOSS(ProtenixOSS_AtomAttentionDecoder):
    """OSS ``AtomAttentionDecoder`` (AF3 Algorithm 6) with a seeded builder."""

    @classmethod
    def build(
        cls,
        *,
        n_blocks: int = 3,
        n_heads: int = 4,
        c_token: int = 768,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_queries: int = 32,
        n_keys: int = 128,
        seed: int = 42,
    ) -> "RefProtenixAtomAttentionDecoderFromOSS":
        torch.manual_seed(seed)
        return cls(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
        )


class RefProtenixMSAModuleFromOSS(ProtenixOSS_MSAModule):
    """OSS ``MSAModule`` (AF3 Algorithm 8) with a seeded builder.

    ``msa_configs`` makes the internal row subsampling a no-op (sequential,
    cutoff above the test MSA depth) — the TRT-BNM port consumes prepared MSA
    features, so the reference must not shuffle/drop them.
    """

    @classmethod
    def build(
        cls,
        *,
        n_blocks: int = 2,
        c_m: int = 128,
        c_z: int = 64,
        c_s_inputs: int = 449,
        hidden_scale_up: bool = True,
        seed: int = 42,
    ) -> "RefProtenixMSAModuleFromOSS":
        torch.manual_seed(seed)
        # strategy="topk" + lower_bound >= MSA depth forces sample_size == depth
        # and cutoff above depth keeps all rows in order (identity subsample).
        return cls(
            n_blocks=n_blocks,
            c_m=c_m,
            c_z=c_z,
            c_s_inputs=c_s_inputs,
            hidden_scale_up=hidden_scale_up,
            msa_configs={
                "enable": True,
                "strategy": "topk",
                "sample_cutoff": {"train": 999999, "test": 999999},
                "min_size": {"train": 999999, "test": 999999},
            },
        )
