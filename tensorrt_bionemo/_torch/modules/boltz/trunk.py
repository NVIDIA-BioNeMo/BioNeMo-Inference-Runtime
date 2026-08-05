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


import torch
import torch.nn as nn

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.attention_backend.utils import PrecomputedPairMasks, precompute_pair_masks
from tensorrt_bionemo._torch.auto_chunk import CHUNK_REGISTRY, PAIR_TRANSITION
from tensorrt_bionemo._torch.layers.linear import Linear
from tensorrt_bionemo._torch.layers.outer_product_mean import OuterProductMean
from tensorrt_bionemo._torch.layers.pair_averaging import PairWeightedAveraging
from tensorrt_bionemo._torch.layers.transformers.pairformer import PairformerModule, PairformerNoSeqLayer
from tensorrt_bionemo._torch.layers.transition import Transition
from tensorrt_bionemo._torch.modules.boltz.template import TemplateV2Module
from tensorrt_bionemo._torch.utils import recursive_calling_load_weights
from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.pipeline.models.boltz2.const import pocket_contact_info


class MSALayer(nn.Module):
    def __init__(
        self,
        msa_s: int,
        token_z: int,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        layer_idx: int = 0,
        eps: float = 1e-5,
        inf: float = 1e9,
        dtype: torch.dtype = None,
        skip_create_weights: bool = False,
        triangle_attn_backend: str = "VANILLA",
        trimul_high_precision: bool = False,
    ) -> None:
        super().__init__()
        self.msa_s = msa_s
        self.token_z = token_z
        self.pairwise_head_width = pairwise_head_width
        self.pairwise_num_heads = pairwise_num_heads
        self.eps = eps
        self.inf = inf
        self.dtype = dtype

        self.msa_transition = Transition(
            dim=msa_s,
            hidden=msa_s * 4,
            layer_idx=layer_idx,
            eps=eps,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            # Row-chunk the MSA-transition FFN over the sequence dim S at large S (position-wise,
            # numerically identical) -- replaces the old chunk_heads_pwa-coupled chunk_size, now that
            # PWA auto-chunks via its own registry policy.
            auto_chunk_policy=CHUNK_REGISTRY.get(PAIR_TRANSITION),
        )

        self.pair_weighted_averaging = PairWeightedAveraging(
            c_m=msa_s,
            c_z=token_z,
            c_h=32,
            num_heads=8,
            eps=eps,
            inf=inf,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

        self.pairformer_layer = PairformerNoSeqLayer(
            layer_idx=layer_idx,
            token_z=token_z,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            eps=eps,
            inf=inf,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            triangle_attn_backend=triangle_attn_backend,
            trimul_high_precision=trimul_high_precision,
        )
        self.outer_product_mean = OuterProductMean(
            c_in=msa_s, c_hidden=32, c_out=token_z, eps=eps, dtype=dtype, skip_create_weights=skip_create_weights
        )

    def forward(
        self,
        z: torch.Tensor,
        m: torch.Tensor,
        token_mask: torch.Tensor,
        msa_mask: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
        precomputed_masks: PrecomputedPairMasks | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z(Tensor): The input tensor of shape (B, N, N, token_z).
            m(Tensor): The input tensor of shape (B, S, N, msa_s).
            token_mask(Tensor): The mask tensor of shape (B, N, N).
            msa_mask(Tensor): The mask tensor of shape (B, S, N).
            precomputed_masks: Precomputed mask biases for triangle attention.
        Returns:
            Tuple[Tensor, Tensor]: The output tensor of shape (B, N, N, token_z), (B, S, N, msa_s).
        """
        # PWA and msa_transition auto-chunk internally via their registry policies at large N/S.
        m += self.pair_weighted_averaging(m, z, token_mask)
        m += self.msa_transition(m)
        z += self.outer_product_mean(m, msa_mask)

        z = self.pairformer_layer(
            z, token_mask, attn_metadatas={"triangle_attn": attn_metadata}, precomputed_masks=precomputed_masks
        )
        return z, m


class MSAModule(nn.Module):
    def __init__(self, config: BaseConfig) -> None:
        """
        Boltz MSAModule
        TODO: add support for subsampling, chunking
        """
        super().__init__()

        self.msa_s = config.msa_s
        self.token_z = config.token_z
        self.token_s = config.token_s
        self.msa_blocks = config.msa_blocks
        self.num_tokens = config.num_tokens
        self.pairwise_head_width = config.pairwise_head_width
        self.pairwise_num_heads = config.pairwise_num_heads
        self.use_paired_feature = config.use_paired_feature
        self.dtype = config.torch_dtype
        self.version = config.version
        # Propagate the config's trimul precision to the MSA-module pairformer. Without this the
        # MSALayer leaves PairformerNoSeqLayer at PairformerLayerV1's default (high_precision=True
        # -> fp32 -> the memory-heavy vanilla dual-GEMM); MSAModuleConfig sets it False.
        self.trimul_high_precision = getattr(config, "trimul_high_precision", False)

        if config.version == "v1":
            s_input_dim = self.token_s + 2 * self.num_tokens + 1 + len(pocket_contact_info)
        else:
            s_input_dim = self.token_s
        self.s_proj = Linear(
            s_input_dim, self.msa_s, bias=False, dtype=self.dtype, skip_create_weights=config.skip_create_weights
        )
        self.msa_proj = Linear(
            self.num_tokens + 2 + int(self.use_paired_feature),
            self.msa_s,
            bias=False,
            dtype=self.dtype,
            skip_create_weights=config.skip_create_weights,
        )

        self.layers = nn.ModuleList()
        for layer_idx in range(self.msa_blocks):
            self.layers.append(
                MSALayer(
                    msa_s=self.msa_s,
                    token_z=self.token_z,
                    layer_idx=layer_idx,
                    pairwise_head_width=self.pairwise_head_width,
                    pairwise_num_heads=self.pairwise_num_heads,
                    eps=config.norm_epsilon,
                    inf=config.mask_inf,
                    dtype=self.dtype,
                    skip_create_weights=config.skip_create_weights,
                    triangle_attn_backend=config.triangle_attention_backend,
                    trimul_high_precision=self.trimul_high_precision,
                )
            )

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # verify whether all the weights are loaded
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weights}")

    def forward(
        self,
        z: torch.Tensor,
        emb: torch.Tensor,
        msa: torch.Tensor,
        has_deletion: torch.Tensor,
        deletion_value: torch.Tensor,
        msa_paired: torch.Tensor,
        msa_mask: torch.Tensor,
        token_pad_mask: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z(Tensor): The input tensor of shape (B, N, N, token_z).
            emb(Tensor): The input tensor of shape (B, N, token_s).
            msa(Tensor): The input tensor of shape (B, N_msa, N).
            has_deletion(Tensor): The input tensor of shape (B, N_msa, N).
            deletion_value(Tensor): The input tensor of shape (B, N_msa, N).
            msa_paired(Tensor): The input tensor of shape (B, N_msa, N).
            msa_mask(Tensor): The input tensor of shape (B, N_msa, N).
            token_pad_mask(Tensor): The input tensor of shape (B, N).
            attn_metadata(Optional[AttentionMetadata]): The attention metadata.
        Returns:
            Tensor: The output tensor of shape (B, N, N, token_z).
        """
        if self.version == "v2":
            msa = torch.nn.functional.one_hot(msa, num_classes=self.num_tokens)
        has_deletion = has_deletion.unsqueeze(-1)
        deletion_value = deletion_value.unsqueeze(-1)
        is_paired = msa_paired.unsqueeze(-1)
        # Optimized here: token_pad_mask is already in the begin of trunk module
        # token_mask = token_pad_mask.to(self.dtype)
        # token_mask = token_mask[:, :, None] * token_mask[:, None, :]

        # Compute MSA embeddings
        if self.use_paired_feature:
            m = torch.cat([msa, has_deletion, deletion_value, is_paired], dim=-1)
        else:
            m = torch.cat([msa, has_deletion, deletion_value], dim=-1)

        # Compute input projections
        m = self.msa_proj(m.to(self.dtype))
        m = m + self.s_proj(emb).unsqueeze(1)

        first_layer = self.layers[0]
        precomputed = precompute_pair_masks(
            first_layer.pairformer_layer.triangle_attn_backend,
            token_pad_mask,
            inf=first_layer.inf,
            dtype=first_layer.dtype,
        )

        for i in range(self.msa_blocks):
            z, m = self.layers[i](z, m, token_pad_mask, msa_mask, attn_metadata, precomputed_masks=precomputed)
        return z


class Trunk(nn.Module):
    """Trunk module for Boltz1-2"""

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()

        self.msa_module = MSAModule(config.msa_module)
        self.pairformer_module = PairformerModule(config.pairformer)

        token_s = config.pairformer.token_s
        token_z = config.pairformer.token_z
        self.dtype = config.torch_dtype

        assert self.dtype == config.msa_module.torch_dtype, (
            f"Trunk dtype: {self.dtype}, msa_module dtype: {config.msa_module.torch_dtype}"
        )
        assert self.dtype == config.pairformer.torch_dtype, (
            f"Trunk dtype: {self.dtype}, pairformer dtype: {config.pairformer.torch_dtype}"
        )

        # ``TrunkConfig.use_templates_v2`` (e.g. when loading a checkpoint
        # trained with ``use_templates_v2=True``).
        self.use_templates_v2 = getattr(config, "use_templates_v2", False)
        self.template_module: TemplateV2Module | None = None
        if self.use_templates_v2:
            self.template_module = TemplateV2Module(config.template_module)

        self.s_norm = nn.LayerNorm(token_s, dtype=self.dtype, eps=config.norm_epsilon)
        self.z_norm = nn.LayerNorm(token_z, dtype=self.dtype, eps=config.norm_epsilon)
        self.skip_create_weights = config.skip_create_weights

        self.s_recycle = Linear(
            token_s, token_s, bias=False, dtype=self.dtype, skip_create_weights=self.skip_create_weights
        )
        self.z_recycle = Linear(
            token_z, token_z, bias=False, dtype=self.dtype, skip_create_weights=self.skip_create_weights
        )

    def load_weights(self, weights: dict):
        """Load weights for the Trunk module
        Args:
            dict: {
                "msa_module": dict,
                "pairformer_module": dict,
                "template_module": dict (optional, only if use_templates_v2),
                ...
            }
        """
        msa_module_weights = weights.pop("msa_module")
        pairformer_module_weights = weights.pop("pairformer_module")
        self.msa_module.load_weights(weights=msa_module_weights)
        self.pairformer_module.load_weights(weights=pairformer_module_weights)

        template_module_weights = weights.pop("template_module", None)
        if template_module_weights is not None:
            assert self.template_module is not None, (
                "Got template_module weights but the trunk was built without use_templates_v2=True"
            )
            self.template_module.load_weights(weights=template_module_weights)

        # Skip loading the weights for the submodules already loaded above
        def filter_func(name, _):
            return name.startswith(("msa_module", "pairformer_module", "template_module"))

        loaded_weight = recursive_calling_load_weights(self, weights, filter_func)
        # verify whether all the weights are loaded
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weights}")

    def forward(
        self,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        s_inputs: torch.Tensor,
        msa: torch.Tensor,
        has_deletion: torch.Tensor,
        deletion_value: torch.Tensor,
        msa_paired: torch.Tensor,
        msa_mask: torch.Tensor,
        token_pad_mask: torch.Tensor,
        recycling_steps: int = 3,
        attn_metadata: AttentionMetadata | None = None,
        template_feats: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recycling forward pass for Boltz1-2
        Args:
            s_init(Tensor): The initial sequence embeddings of shape (B, N, token_s).
            z_init(Tensor): The initial pairwise embeddings of shape (B, N, N, token_z).
            s_inputs(Tensor): The input embeddings of shape (B, N, token_s).
            msa(Tensor): The MSA embeddings of shape (B, N_msa, N).
            has_deletion(Tensor): The has deletion embeddings of shape (B, N_msa, N).
            deletion_value(Tensor): The deletion value embeddings of shape (B, N_msa, N).
            msa_paired(Tensor): The MSA paired embeddings of shape (B, N_msa, N).
            msa_mask(Tensor): The MSA mask of shape (B, N_msa, N).
            token_pad_mask(Tensor): The token pad mask of shape (B, N).
            recycling_steps(int): The number of recycling steps.
            attn_metadata(Optional[AttentionMetadata]): The attention metadata.
            template_feats(Optional[dict]): Per-template features required by
                :class:`TemplateV2Module` when ``use_templates_v2`` is enabled.
                Ignored when the template module is not built. Pass ``None``
                (e.g. when the batch's ``has_templates`` flag is False) to
                skip the template pairformer entirely.
        Returns:
            Tuple[Tensor, Tensor]: The output sequence and pairwise embeddings of shape (B, N, token_s), (B, N, N, token_z).
        """
        # Ensure the inputs are in the correct dtype
        s_init = s_init.to(self.dtype)
        z_init = z_init.to(self.dtype)
        mask = token_pad_mask.to(self.dtype)
        pair_mask = mask[:, :, None] * mask[:, None, :]
        s_inputs = s_inputs.to(self.dtype)
        msa_mask = msa_mask.to(self.dtype)

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        run_template = self.template_module is not None and template_feats is not None

        for _ in range(1 + recycling_steps):
            s = s_init + self.s_recycle(self.s_norm(s))
            z = z_init + self.z_recycle(self.z_norm(z))

            if run_template:
                z = z + self.template_module(z, template_feats, pair_mask, attn_metadata=attn_metadata).to(self.dtype)

            z = z + self.msa_module(
                z,
                s_inputs,
                msa,
                has_deletion,
                deletion_value,
                msa_paired,
                msa_mask=msa_mask,
                token_pad_mask=pair_mask,
                attn_metadata=attn_metadata,
            )

            s, z = self.pairformer_module(s, z, mask=mask, pair_mask=pair_mask, attn_metadata=attn_metadata)
        return s, z
