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
from tensorrt_bionemo._torch.attention_backend.utils import (
    PrecomputedPairMasks,
    PrecomputedSingleMasks,
    precompute_pair_masks,
    precompute_single_masks,
)
from tensorrt_bionemo._torch.auto_chunk import CHUNK_REGISTRY, PAIR_TRANSITION
from tensorrt_bionemo._torch.graph_optimization.config import (
    GraphOptimizationMode,
    InputAcceptanceDimSpec,
    InputKeyMethod,
    PaddedDimSpec,
    SpacingMethod,
)
from tensorrt_bionemo._torch.graph_optimization.decorator import NamedDimTies, support_graph_optimization
from tensorrt_bionemo._torch.layers.attention import AttentionPairBias
from tensorrt_bionemo._torch.layers.transition import Transition
from tensorrt_bionemo._torch.layers.triangle_nodes import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationNode,
    TriangleMultiplicationNodeType,
)
from tensorrt_bionemo._torch.utils import recursive_calling_load_weights
from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.runtime.buffers import PreallocatedBuffers
from tensorrt_bionemo.utils import str_dtype_to_torch


class PairformerLayerV1(nn.Module):
    def __init__(
        self,
        layer_idx: int = 0,
        token_s: int = 384,
        token_z: int = 128,
        num_heads: int = 16,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        no_update_s: bool = False,
        no_update_z: bool = False,
        dtype: torch.dtype = None,
        eps: float = 1e-5,
        inf: float = 1e9,
        triangle_attn_backend: str = "VANILLA",
        pairwise_attn_backend: str = "VANILLA",
        skip_create_weights: bool = False,
        attention_initial_norm: bool = False,
        s_path_dtype: str | torch.dtype | None = None,
        trimul_high_precision: bool = True,
        trimul_mean_normalization: bool = False,
        pair_mask_left_aligned: bool = True,
        pair_transition_factor: int = 4,
        **kwargs,
    ):
        """Pairformer layer.

        Args:
            pair_mask_left_aligned: Whether the runtime ``pair_mask`` is
                guaranteed left-aligned (``1...1 0...0``) along both
                masked axes. Gates prefix-length fast paths in triangle
                multiplication and cuEquivariance triangle attention. Set
                ``False`` for bipartite masks with interior zeros.
        """
        super().__init__()
        self.dtype = dtype
        self.no_update_s = no_update_s
        self.no_update_z = no_update_z
        self.token_s = token_s
        self.token_z = token_z
        self.triangle_attn_backend = triangle_attn_backend
        self.pairwise_attn_backend = pairwise_attn_backend
        self.pair_mask_left_aligned = pair_mask_left_aligned
        if isinstance(s_path_dtype, str):
            s_path_dtype = str_dtype_to_torch(s_path_dtype)
        if s_path_dtype is None:
            s_path_dtype = dtype
        self.s_path_dtype = s_path_dtype

        if not self.no_update_s:
            self.attention = AttentionPairBias(
                layer_idx=layer_idx,
                c_s=token_s,
                c_z=token_z,
                num_heads=num_heads,
                dtype=s_path_dtype,
                bias_proj=True,
                eps=eps,
                inf=inf,
                skip_create_weights=skip_create_weights,
                attn_backend=pairwise_attn_backend,
                initial_norm=attention_initial_norm,
            )
        self.tri_mul_out = TriangleMultiplicationNode(
            layer_idx=layer_idx,
            dim=token_z,
            eps=eps,
            multiplication_type=TriangleMultiplicationNodeType.OUTGOING,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            high_precision=trimul_high_precision,
            mean_normalization=trimul_mean_normalization,
            pair_mask_left_aligned=pair_mask_left_aligned,
        )
        self.tri_mul_in = TriangleMultiplicationNode(
            layer_idx=layer_idx,
            dim=token_z,
            eps=eps,
            multiplication_type=TriangleMultiplicationNodeType.INCOMING,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            high_precision=trimul_high_precision,
            mean_normalization=trimul_mean_normalization,
            pair_mask_left_aligned=pair_mask_left_aligned,
        )
        self.tri_attn_start = TriangleAttentionStartingNode(
            token_z,
            pairwise_head_width,
            pairwise_num_heads,
            inf=inf,
            layer_idx=layer_idx,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            attn_backend=triangle_attn_backend,
            pair_mask_left_aligned=pair_mask_left_aligned,
        )
        self.tri_attn_end = TriangleAttentionEndingNode(
            token_z,
            pairwise_head_width,
            pairwise_num_heads,
            inf=inf,
            layer_idx=layer_idx,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            attn_backend=triangle_attn_backend,
            pair_mask_left_aligned=pair_mask_left_aligned,
        )
        if not self.no_update_s:
            self.transition_s = Transition(
                token_s,
                token_s * 4,
                layer_idx=layer_idx,
                eps=eps,
                dtype=s_path_dtype,
                skip_create_weights=skip_create_weights,
            )
        self.transition_z = Transition(
            token_z,
            token_z * pair_transition_factor,
            layer_idx=layer_idx,
            eps=eps,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            # Row-chunk large pair FFNs to bound [N, N, 2*hidden] activations.
            auto_chunk_policy=CHUNK_REGISTRY.get(PAIR_TRANSITION),
        )

    def _transform_z(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        attn_metadatas: dict[str, AttentionMetadata] | None = None,
        precomputed_masks: PrecomputedPairMasks | None = None,
        buffers: PreallocatedBuffers | None = None,
    ) -> torch.Tensor:
        # Reuse CuTeDSL's int32 row lengths only for left-aligned masks.
        # Otherwise, let the wrapper derive masking from ``pair_mask``.
        tri_out_actual_seqlen = tri_in_actual_seqlen = None
        if (
            self.pair_mask_left_aligned
            and precomputed_masks is not None
            and precomputed_masks.mask_bias.dtype == torch.int32
        ):
            tri_out_actual_seqlen = precomputed_masks.mask_bias
            tri_in_actual_seqlen = precomputed_masks.mask_bias_transposed
        z = z + self.tri_mul_out(z, mask=pair_mask, actual_seqlen=tri_out_actual_seqlen)
        z = z + self.tri_mul_in(z, mask=pair_mask, actual_seqlen=tri_in_actual_seqlen)
        z = z.to(self.dtype)

        tri_attn_metadata = (attn_metadatas or {}).get("triangle_attn")
        # Same left-aligned gate as tri_mul: CuTeDSL int32 row lengths are
        # invalid for bipartite / interior-zero masks.
        if self.pair_mask_left_aligned and precomputed_masks is not None:
            mb_start = precomputed_masks.mask_bias
            mb_end = precomputed_masks.mask_bias_transposed
        else:
            mb_start = mb_end = None
            pair_mask = pair_mask.to(self.dtype)

        z = z + self.tri_attn_start(
            z, mask=pair_mask, mask_bias=mb_start, attn_metadata=tri_attn_metadata, buffers=buffers
        )
        z = z + self.tri_attn_end(z, mask=pair_mask, mask_bias=mb_end, attn_metadata=tri_attn_metadata, buffers=buffers)

        z = z + self.transition_z(z)
        return z

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        pair_mask: torch.Tensor,
        attn_metadatas: dict[str, AttentionMetadata] | None = None,
        precomputed_masks: PrecomputedPairMasks | None = None,
        precomputed_single_masks: PrecomputedSingleMasks | None = None,
        buffers: PreallocatedBuffers | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._transform_z(z, pair_mask, attn_metadatas, precomputed_masks=precomputed_masks, buffers=buffers)
        if not self.no_update_s:
            mask_bias = precomputed_single_masks.mask_bias if precomputed_single_masks else None
            s = s + self.attention(
                s,
                z,
                mask,
                attn_metadata=(attn_metadatas or {}).get("pairwise_attn"),
                mask_bias=mask_bias,
                buffers=buffers,
            )
            s = s + self.transition_s(s)
        return s, z


class PairformerNoSeqLayer(PairformerLayerV1):
    def __init__(
        self,
        *,
        layer_idx: int,
        token_z: int = 128,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        **kwargs,
    ):
        kwargs["no_update_s"] = True
        super().__init__(
            layer_idx=layer_idx,
            token_z=token_z,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            **kwargs,
        )

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        attn_metadatas: dict[str, AttentionMetadata] | None = None,
        precomputed_masks: PrecomputedPairMasks | None = None,
        precomputed_single_masks: PrecomputedSingleMasks | None = None,
        buffers: PreallocatedBuffers | None = None,
        **kwargs,
    ) -> torch.Tensor:
        _, update_z = super().forward(
            s=None,
            z=z,
            mask=None,
            pair_mask=pair_mask,
            attn_metadatas=attn_metadatas,
            precomputed_masks=precomputed_masks,
            precomputed_single_masks=precomputed_single_masks,
            buffers=buffers,
        )
        return update_z


class PairformerNoSeqModule(nn.Module):
    def __init__(
        self,
        num_blocks: int = 8,
        token_z: int = 128,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PairformerNoSeqLayer(
                    layer_idx=i,
                    token_z=token_z,
                    pairwise_head_width=pairwise_head_width,
                    pairwise_num_heads=pairwise_num_heads,
                    **kwargs,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        attn_metadatas: dict[str, AttentionMetadata] | None = None,
        buffers: PreallocatedBuffers | None = None,
        **kwargs,
    ) -> torch.Tensor:
        first_layer = self.layers[0]
        precomputed = precompute_pair_masks(
            first_layer.triangle_attn_backend,
            pair_mask,
            inf=first_layer.tri_attn_start.inf,
            dtype=first_layer.dtype,
        )
        if buffers is None and first_layer.triangle_attn_backend == "CuTeDSL":
            buffers = {}
        for layer in self.layers:
            z = layer(z, pair_mask, attn_metadatas, precomputed_masks=precomputed, buffers=buffers)
        return z


class PairformerLayerV2(PairformerLayerV1):
    def __init__(self, post_layer_norm: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.post_layer_norm = post_layer_norm
        self.pre_norm_s = nn.LayerNorm(self.token_s, dtype=self.s_path_dtype)
        self.post_norm_s = None
        if self.post_layer_norm:
            self.post_norm_s = nn.LayerNorm(self.token_s, dtype=self.s_path_dtype)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        pair_mask: torch.Tensor,
        attn_metadatas: dict[str, AttentionMetadata] | None = None,
        precomputed_masks: PrecomputedPairMasks | None = None,
        precomputed_single_masks: PrecomputedSingleMasks | None = None,
        buffers: PreallocatedBuffers | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._transform_z(z, pair_mask, attn_metadatas, precomputed_masks=precomputed_masks, buffers=buffers)
        original_s_dtype = s.dtype
        original_z_dtype = z.dtype

        # v2 use float precision on the computing of s
        z = z.to(self.s_path_dtype)
        s = s.to(self.s_path_dtype)
        s_normed = self.pre_norm_s(s)
        mask_bias = precomputed_single_masks.mask_bias if precomputed_single_masks else None
        s = s + self.attention(
            s_normed,
            z,
            mask,
            attn_metadata=(attn_metadatas or {}).get("pairwise_attn"),
            mask_bias=mask_bias,
            buffers=buffers,
        )
        s = s + self.transition_s(s)
        if self.post_layer_norm:
            s = self.post_norm_s(s)
        s = s.to(original_s_dtype)
        z = z.to(original_z_dtype)
        return s, z


@support_graph_optimization(
    # ``num_tokens`` rides s (-2), z (-2 and -3), mask (-1), and pair_mask
    # (-1 and -2) on the inputs, and both returned reps (s at -2, z at -2/-3)
    # on the outputs. These ties are fixed by ``forward``'s signature.
    named_dims=[
        NamedDimTies(
            name="num_tokens",
            input_dims=(
                ("s", (-2,)),
                ("z", (-2, -3)),
                ("mask", (-1,)),
                ("pair_mask", (-1, -2)),
            ),
            output_dims=(
                (0, (-2,)),
                (1, (-2, -3)),
            ),
        ),
    ],
    static_args=("mask", "pair_mask"),
    workspace_kwargs=("buffers",),
    graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
    verify_capture=False,
    input_key_method=InputKeyMethod.BUCKETED_SHAPES,
    input_acceptance_dim_spec=InputAcceptanceDimSpec(
        name="num_tokens",
        dim_len_max=1024,
    ),
    padded_dim_spec=PaddedDimSpec(
        name="num_tokens",
        dim_len_min=4,
        dim_len_max=1024,
        num_intervals=8,
        multiple_of=128,
        spacing_method=SpacingMethod.LINEAR,
    ),
)
class PairformerModule(nn.Module):
    def __init__(self, config: BaseConfig):
        """
        Args:
            config: tensorrt_bionemo.configs.modules.PairformerConfig
                The configuration of the pairformer module.
        """
        super().__init__()
        self.config = config
        layer_cls = PairformerLayerV1 if config.version == "v1" else PairformerLayerV2
        self.layers = nn.ModuleList()
        for i in range(config.num_blocks):
            self.layers.append(
                layer_cls(
                    layer_idx=i,
                    token_s=config.token_s,
                    token_z=config.token_z,
                    num_heads=config.num_heads,
                    pairwise_head_width=config.pairwise_head_width,
                    pairwise_num_heads=config.pairwise_num_heads,
                    no_update_s=config.no_update_s,
                    no_update_z=config.no_update_z,
                    dtype=config.torch_dtype,
                    eps=config.norm_epsilon,
                    inf=config.mask_inf,
                    skip_create_weights=config.skip_create_weights,
                    triangle_attn_backend=config.triangle_attention_backend,
                    pairwise_attn_backend=config.pairwise_attention_backend,
                    post_layer_norm=config.post_layer_norm,
                    attention_initial_norm=config.attention_initial_norm,
                    trimul_high_precision=config.trimul_high_precision,
                    trimul_mean_normalization=getattr(config, "trimul_mean_normalization", False),
                    s_path_dtype=config.s_path_dtype,
                )
            )

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # Every entry of ``weights`` must have been consumed.
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weights}")

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        pair_mask: torch.Tensor,
        attn_metadatas: dict[str, AttentionMetadata] | None = None,
        buffers: PreallocatedBuffers | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        precomputed = precompute_pair_masks(
            self.config.triangle_attention_backend,
            pair_mask,
            inf=self.config.mask_inf,
            dtype=self.config.torch_dtype,
        )
        precomputed_single = precompute_single_masks(
            self.config.pairwise_attention_backend,
            mask,
            inf=self.config.mask_inf,
        )
        _uses_cute = "CuTeDSL" in (self.config.triangle_attention_backend, self.config.pairwise_attention_backend)
        if buffers is None and _uses_cute:
            buffers = {}
        for layer in self.layers:
            s, z = layer(
                s,
                z,
                mask,
                pair_mask,
                attn_metadatas,
                precomputed_masks=precomputed,
                precomputed_single_masks=precomputed_single,
                buffers=buffers,
            )
        return s, z
