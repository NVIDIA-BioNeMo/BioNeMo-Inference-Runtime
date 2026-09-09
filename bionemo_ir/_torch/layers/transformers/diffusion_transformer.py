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
import torch.nn.functional as F

from bionemo_ir._torch.attention_backend import AttentionMetadata
from bionemo_ir._torch.attention_backend.utils import PrecomputedSingleMasks, precompute_single_masks
from bionemo_ir._torch.custom_ops.gated_sigmoid import get_gated_sigmoid_op
from bionemo_ir._torch.graph_optimization.config import (
    GraphOptimizationMode,
    InputAcceptanceDimSpec,
    InputKeyMethod,
)
from bionemo_ir._torch.graph_optimization.decorator import NamedDimTies, support_graph_optimization
from bionemo_ir._torch.layers.attention import AttentionPairBias
from bionemo_ir._torch.layers.linear import Linear
from bionemo_ir._torch.layers.normalization import AdaLN
from bionemo_ir._torch.layers.sequence_local_atom import create_gather_indices, query_to_keys_optimized, to_blocks
from bionemo_ir._torch.layers.transition import ConditionedTransitionBlock
from bionemo_ir._torch.utils import recursive_calling_load_weights
from bionemo_ir.configs import BaseConfig
from bionemo_ir.dsl_kernels.triton.fused_layer_norm_transpose import layer_norm_transpose
from bionemo_ir.runtime.buffers import PreallocatedBuffers, ensure_buffer


def build_bias_mega_weight(layers: nn.ModuleList) -> torch.Tensor:
    """Fold LayerNorm gamma into pair-bias weights and stack them.

    Returns ``[num_layers * num_heads, c_z]`` for one shared normalization.
    """
    parts: list[torch.Tensor] = []
    for layer in layers:
        proj_z = layer.pair_bias_attn.proj_z
        ln = proj_z[0] if len(proj_z) > 1 else None
        proj = proj_z[-1]
        w = proj.weight.data  # [H, D]
        if ln is not None:
            w = w * ln.weight.data.unsqueeze(0)  # [H, D] * [1, D]
        parts.append(w)
    return torch.cat(parts, dim=0).contiguous()  # [N*H, D]


def precompute_pair_biases(
    z: torch.Tensor,
    w_mega: torch.Tensor,
    num_layers: int,
    num_heads: int,
    norm_eps: float,
    bias_pad_multiple: int = -1,
) -> list[torch.Tensor]:
    """Project every layer's pair bias with one normalization and GEMM.

    Args:
        z: pair representation ``[*, I, J, c_z]`` (contiguous).
        w_mega: fused projection ``[num_layers * num_heads, c_z]`` from
            :func:`build_bias_mega_weight`.
        bias_pad_multiple: key padding multiple; ``<= 0`` disables padding.

    Returns:
        ``num_layers`` tensors, each ``[*, num_heads, I, J_pad]``.
    """
    *batch_dims, I, J, D = z.shape
    b_flat = 1
    for d in batch_dims:
        b_flat *= d
    J_pad = ((J + bias_pad_multiple - 1) // bias_pad_multiple) * bias_pad_multiple if bias_pad_multiple > 0 else J
    if J_pad != J:
        z = F.pad(z, (0, 0, 0, J_pad - J))
    z_hat = layer_norm_transpose(
        z.reshape(-1, D),
        None,
        None,
        eps=norm_eps,
        elementwise_affine=False,
        layout="nd->nd",  # codespell:ignore nd
    ).view_as(z)
    out = torch.mm(w_mega, z_hat.reshape(-1, D).t().contiguous())
    out = out.reshape(num_layers, num_heads, b_flat, I, J_pad)
    biases: list[torch.Tensor] = []
    for i in range(num_layers):
        if b_flat == 1:
            b = out[i].squeeze(1).unsqueeze(0)  # [1, H, I, J_pad] view
        else:
            b = out[i].permute(1, 0, 2, 3).contiguous()
        biases.append(b.view(*batch_dims, num_heads, I, J_pad))
    return biases


class DiffusionTransformerLayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        dim: int = 384,
        dim_single_cond: int | None = None,
        dim_pairwise: int = 128,
        bias_proj: bool = False,
        pair_norm: bool = True,
        dtype: torch.dtype = None,
        eps: float = 1e-5,
        inf: float = 1e9,
        attention_initial_norm: bool = False,
        post_layer_norm: bool = False,
        use_ada_layer_norm: bool = True,
        use_separate_layer_norm: bool = False,
        chain_kv_norm: bool = False,
        skip_create_weights: bool = False,
        initial_norm: bool = True,
        conditioned_transition_using_silu: bool = False,
        attn_output_gate: bool = True,
        attn_gate_bias: bool = False,
        transition_expansion_factor: int = 2,
        attn_backend: str = "VANILLA",
    ):
        super().__init__()

        self.initial_norm = initial_norm
        self.use_separate_layer_norm = use_separate_layer_norm
        self.attn_output_gate = attn_output_gate

        if initial_norm:
            self.adaln = AdaLN(dim, dim_single_cond, eps=eps, dtype=dtype, skip_create_weights=skip_create_weights)

        self.pair_bias_attn = AttentionPairBias(
            layer_idx=layer_idx,
            c_s=dim,
            c_z=dim_pairwise,
            num_heads=num_heads,
            initial_norm=attention_initial_norm,
            bias_proj=bias_proj,
            pair_norm=pair_norm,
            use_ada_layer_norm=use_ada_layer_norm,
            use_separate_layer_norm=use_separate_layer_norm,
            chain_kv_norm=chain_kv_norm,
            gate_bias=attn_gate_bias,
            eps=eps,
            inf=inf,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            attn_backend=attn_backend,
        )

        self.output_projection = None
        self._can_fuse_output_gate = False
        self._gated_sigmoid_op = None
        if self.attn_output_gate:
            self.output_projection = Linear(dim_single_cond, dim, dtype=dtype, skip_create_weights=skip_create_weights)
            self._can_fuse_output_gate = True
            self._gated_sigmoid_op = get_gated_sigmoid_op(
                dtype or torch.get_default_dtype(),
                N=dim,
                K=dim_single_cond,
            )
        self.transition = ConditionedTransitionBlock(
            dim_single=dim,
            dim_single_cond=dim_single_cond,
            expansion_factor=transition_expansion_factor,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            using_silu=conditioned_transition_using_silu,
        )
        self.post_lnorm = None
        if post_layer_norm:
            self.post_lnorm = nn.LayerNorm(dim, dtype=dtype, eps=eps)

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        precomputed_single_masks: PrecomputedSingleMasks | None = None,
        buffers: PreallocatedBuffers | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward for a single layer.

        Does not support multiplicity > 1, nor the atom encoder / decoder.
        """
        if self.initial_norm:
            b = self.adaln(a, s, buffers=buffers, buffer_key="dit_bsd_scratch")
        else:
            b = a

        mask_bias = precomputed_single_masks.mask_bias if precomputed_single_masks else None
        mask_bias_local = precomputed_single_masks.mask_bias_local if precomputed_single_masks else None
        b = self.pair_bias_attn(
            s=b,
            z=bias,
            single_embedding=s if self.use_separate_layer_norm else None,
            mask=mask,
            attn_metadata=attn_metadata,
            mask_bias=mask_bias,
            mask_bias_local=mask_bias_local,
            buffers=buffers,
        )
        if self.attn_output_gate:
            if self._can_fuse_output_gate:
                # The gated-sigmoid op broadcasts `s` (gate) across the
                # multiplicity dim of `b` when their leading shapes differ,
                # falling back to torch internally for unsupported patterns.
                gs_buf = ensure_buffer(buffers, "dit_bsd_scratch", b.shape, b.dtype, b.device)
                b = self._gated_sigmoid_op(
                    s,
                    self.output_projection.weight,
                    b,
                    self.output_projection.bias,
                    output=gs_buf,
                )
            else:
                b = F.sigmoid(self.output_projection(s)) * b
        a = a + b
        a = a + self.transition(a, s, buffers=buffers, buffer_key="dit_bsd_scratch")
        if self.post_lnorm is not None:
            a = self.post_lnorm(a)
        return a


# a/s (-2), z (-2 and -3), and mask (-1) carry ``num_tokens`` on the inputs; the
# single returned atom representation carries it at -2. Shared by both diffusion
# (token) transformers, which have the same forward signature and token ties.
_TOKEN_TRANSFORMER_DIMS = (
    NamedDimTies(
        name="num_tokens",
        input_dims=(
            ("a", (-2,)),
            ("s", (-2,)),
            ("z", (-2, -3)),
            ("mask", (-1,)),
        ),
        output_dims=((0, (-2,)),),
    ),
)


@support_graph_optimization(
    named_dims=_TOKEN_TRANSFORMER_DIMS,
    workspace_kwargs=("buffers",),
    graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
    input_key_method=InputKeyMethod.EXACT,
    # Default input management: accept up to 1024 tokens before falling back to
    # eager.
    input_acceptance_dim_spec=InputAcceptanceDimSpec(name="num_tokens", dim_len_max=1024),
)
class BoltzDiffusionTransformer(nn.Module):
    def __init__(self, config: BaseConfig):
        """
        Args:
            config: bionemo_ir.configs.modules.DiffusionTransformerConfig
                The configuration of the token transformer module.
        """
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.version = config.version
        self.num_blocks = config.num_blocks
        self.pairwise_attention_backend = config.pairwise_attention_backend
        self.mask_inf = config.mask_inf
        for i in range(config.num_blocks):
            self.layers.append(
                DiffusionTransformerLayer(
                    layer_idx=i,
                    num_heads=config.num_heads,
                    dim=config.dim,
                    dim_single_cond=config.dim_single_cond,
                    dim_pairwise=config.dim_pairwise,
                    post_layer_norm=config.post_layer_norm,
                    bias_proj=False,
                    dtype=config.torch_dtype,
                    eps=config.norm_epsilon,
                    inf=config.mask_inf,
                    attention_initial_norm=config.attention_initial_norm,
                    skip_create_weights=config.skip_create_weights,
                    attn_output_gate=getattr(config, "attn_output_gate", True),
                    attn_gate_bias=getattr(config, "attn_gate_bias", False),
                    transition_expansion_factor=getattr(config, "transition_expansion_factor", 2),
                    attn_backend=config.pairwise_attention_backend,
                )
            )

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # Every entry of ``weights`` must have been consumed.
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weights}")

    def _pad_bias(self, z: torch.Tensor) -> torch.Tensor:
        """Pad Sk and lay out z for efficient per-layer slicing by CuTeDSL.

        For CuTeDSL the kernel requires Sk to be aligned to 128 bits
        (multiple of 8 for bf16/fp16).  We also move the layer dim L to the
        front so that ``z[i]`` returns a contiguous ``[*, H, Sq, Sk_padded]``
        slice — avoiding a ``contiguous()`` copy inside each layer's forward.

        Input ``z`` has shape ``[*, H, Sq, Sk, L]`` after moveaxis.
        CuTeDSL returns ``[L, *, H, Sq, Sk_padded]`` (contiguous).
        Other backends return ``[*, H, Sq, Sk, L]`` (contiguous, no pad).
        """
        if self.pairwise_attention_backend != "CuTeDSL":
            return z.contiguous()
        Sk = z.shape[-2]
        align = 8
        Sk_padded = ((Sk + align - 1) // align) * align
        if Sk_padded != Sk:
            z = F.pad(z, (0, 0, 0, Sk_padded - Sk))
        # Move L (layer index, last dim) to front: [*, H, Sq, Sk_padded, L] → [L, *, H, Sq, Sk_padded]
        return z.movedim(-1, 0).contiguous()

    def forward(
        self,
        a: torch.Tensor = None,
        s: torch.Tensor = None,
        z: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        precomputed_single_masks: PrecomputedSingleMasks | None = None,
        buffers: PreallocatedBuffers | None = None,
        **kwargs,
    ) -> torch.Tensor:
        L = self.num_blocks
        # Transformer z -> [*, heads, N, N, L]
        N, M, D = z.shape[-3:]
        heads = D // L
        batch_dims = z.shape[:-3]
        z = z.view(*batch_dims, N, M, L, heads)  # [*, N, N, L, heads]
        z = torch.moveaxis(z, -1, -4)  # [*, heads, N, N, L]
        z = self._pad_bias(z)

        if precomputed_single_masks is None and mask is not None:
            query_to_keys = attn_metadata.query_to_keys if attn_metadata else None
            precomputed_single_masks = precompute_single_masks(
                self.pairwise_attention_backend, mask, inf=self.mask_inf, query_to_keys=query_to_keys
            )

        if buffers is None and self.pairwise_attention_backend == "CuTeDSL":
            buffers = {}

        for i, layer in enumerate(self.layers):
            # CuTeDSL: z is [L, *, H, Sq, Sk_padded] -> z[i] is contiguous
            # Others:  z is [*, H, Sq, Sk, L]        -> z[..., i]
            bias = z[i] if self.pairwise_attention_backend == "CuTeDSL" else z[..., i]
            a = layer(
                a, s, bias, mask, attn_metadata, precomputed_single_masks=precomputed_single_masks, buffers=buffers
            )
        return a


@support_graph_optimization(
    named_dims=_TOKEN_TRANSFORMER_DIMS,
    workspace_kwargs=("buffers",),
    graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
    input_key_method=InputKeyMethod.EXACT,
    # Default input management: accept up to 1024 tokens before falling back to
    # eager.
    input_acceptance_dim_spec=InputAcceptanceDimSpec(name="num_tokens", dim_len_max=1024),
)
class OpenFold3DiffusionTransformer(nn.Module):
    def __init__(self, config: BaseConfig):
        """
        Args:
            config: DiffusionTransformerConfig for the token transformer.
                ``config.precompute_bias`` (default True): when True and
                ``bias_proj`` is enabled, all per-layer pair biases are
                computed **before** the layer loop via a single transposed
                mega-GEMM.  Per-layer LN γ weights are fused into the
                projection weights so only one normalization (no affine)
                + one ``W_mega @ z_hat.T`` call is needed.  The GEMM output
                ``[NH, BIJ]`` has J contiguous, giving zero-copy
                ``[B, H, I, J]`` views for B=1.
        """
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.version = config.version
        self.num_blocks = config.num_blocks
        self.dtype = config.torch_dtype
        self.pairwise_attention_backend = config.pairwise_attention_backend
        self.mask_inf = config.mask_inf
        shared_pair_norm = getattr(config, "shared_pair_norm", False)

        if shared_pair_norm:
            self.layer_norm_z = nn.LayerNorm(config.dim_pairwise, bias=False, eps=config.norm_epsilon, dtype=self.dtype)
        for i in range(config.num_blocks):
            layer = DiffusionTransformerLayer(
                layer_idx=i,
                num_heads=config.num_heads,
                dim=config.dim,
                dim_single_cond=config.dim_single_cond,
                dim_pairwise=config.dim_pairwise,
                post_layer_norm=config.post_layer_norm,
                bias_proj=config.bias_proj,
                pair_norm=not shared_pair_norm,
                dtype=config.torch_dtype,
                eps=config.norm_epsilon,
                inf=config.mask_inf,
                attention_initial_norm=config.attention_initial_norm,
                skip_create_weights=config.skip_create_weights,
                conditioned_transition_using_silu=config.conditioned_transition_using_silu,
                initial_norm=True if not hasattr(config, "initial_norm") else config.initial_norm,
                use_ada_layer_norm=True if not hasattr(config, "use_ada_layer_norm") else config.use_ada_layer_norm,
                use_separate_layer_norm=False
                if not hasattr(config, "use_separate_layer_norm")
                else config.use_separate_layer_norm,
                attn_output_gate=getattr(config, "attn_output_gate", True),
                attn_gate_bias=getattr(config, "attn_gate_bias", False),
                transition_expansion_factor=getattr(config, "transition_expansion_factor", 2),
                attn_backend=config.pairwise_attention_backend,
            )

            if not shared_pair_norm and config.bias_proj:
                # Replace the default bias=True LayerNorm in proj_z[0] with
                # bias=False to match the reference model.
                dim = layer.pair_bias_attn.proj_z[0].weight.shape
                eps = layer.pair_bias_attn.proj_z[0].eps
                layer.pair_bias_attn.proj_z[0] = nn.LayerNorm(dim, bias=False, eps=eps, dtype=config.torch_dtype)
            self.layers.append(layer)

        # Mega-GEMM precomputed bias: fuse per-layer LN γ into projection
        # weights → single W_mega [N*H, D].  Forward does:
        #   z_hat = layer_norm(z)          # one normalization, no affine
        #   out   = W_mega @ z_hat.T       # [NH, BIJ], J contiguous
        #   slice → N × [B, H, I, J]      # zero-copy views (B=1)
        self._precompute_bias = (
            getattr(config, "precompute_bias", True) and getattr(config, "bias_proj", False) and not shared_pair_norm
        )
        if self._precompute_bias:
            self._num_heads = config.num_heads
            self._dim_pairwise = config.dim_pairwise
            self._norm_eps = config.norm_epsilon
            self._bias_pad_multiple = 8 if config.pairwise_attention_backend == "CuTeDSL" else -1
            self._W_mega: torch.Tensor | None = None

    def _build_mega_weight(self) -> None:
        """Fuse per-layer LN gamma into projection weights -> cached ``[N*H, D]``."""
        self._W_mega = build_bias_mega_weight(self.layers)

    def _precompute_all_biases(self, z: torch.Tensor) -> list[torch.Tensor]:
        """Project all pair biases with the cached mega weight."""
        if self._W_mega is None:
            self._build_mega_weight()
        return precompute_pair_biases(
            z, self._W_mega, len(self.layers), self._num_heads, self._norm_eps, self._bias_pad_multiple
        )

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # Every entry of ``weights`` must have been consumed.
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weights}")
        if self._precompute_bias:
            self._W_mega = None

    def forward(
        self,
        a: torch.Tensor = None,
        s: torch.Tensor = None,
        z: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
        **kwargs,
    ) -> torch.Tensor:

        if hasattr(self, "layer_norm_z"):
            z = self.layer_norm_z(z)

        precomputed_single_masks = None
        if mask is not None:
            query_to_keys = attn_metadata.query_to_keys if attn_metadata else None
            precomputed_single_masks = precompute_single_masks(
                self.pairwise_attention_backend, mask, inf=self.mask_inf, query_to_keys=query_to_keys
            )

        if buffers is None and self.pairwise_attention_backend == "CuTeDSL":
            buffers = {}

        if self._precompute_bias:
            all_biases = self._precompute_all_biases(z)

            # Temporarily disable per-layer bias projection so that
            # AttentionPairBias accepts the pre-computed [B, H, I, J_pad]
            # directly instead of re-projecting raw z.
            for layer in self.layers:
                layer.pair_bias_attn.bias_proj = False
            try:
                for i, layer in enumerate(self.layers):
                    a = layer(
                        a,
                        s,
                        all_biases[i],
                        mask,
                        attn_metadata,
                        precomputed_single_masks=precomputed_single_masks,
                        buffers=buffers,
                    )
            finally:
                for layer in self.layers:
                    layer.pair_bias_attn.bias_proj = True
        else:
            for layer in self.layers:
                a = layer(
                    a, s, z, mask, attn_metadata, precomputed_single_masks=precomputed_single_masks, buffers=buffers
                )
        return a


@support_graph_optimization(
    named_dims=_TOKEN_TRANSFORMER_DIMS,
    workspace_kwargs=("buffers",),
    graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
    input_key_method=InputKeyMethod.EXACT,
    # Default input management: accept up to 1024 tokens before falling back to
    # eager.
    input_acceptance_dim_spec=InputAcceptanceDimSpec(name="num_tokens", dim_len_max=1024),
)
class ProtenixDiffusionTransformer(nn.Module):
    """Protenix transformer for local atom and global token diffusion.

    Pass ``n_queries`` and ``n_keys`` for windowed atom attention; omit both for
    global token attention. Construction flags must match the selected path.
    The local path requires SDPA, while the global path also supports CuTeDSL.
    ``precompute_bias`` projects every layer's pair bias in one GEMM.
    """

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        dtype = config.torch_dtype
        c_a = config.dim
        # Dims-only configs default to the local atom variant.
        bias_proj = True if config.bias_proj is None else config.bias_proj
        cts = config.conditioned_transition_using_silu
        cts = True if cts is None else cts
        attn_backend = config.pairwise_attention_backend
        self.layers = nn.ModuleList()
        for i in range(config.num_blocks):
            layer = DiffusionTransformerLayer(
                layer_idx=i,
                num_heads=config.num_heads,
                dim=c_a,
                dim_single_cond=config.dim_single_cond or c_a,
                dim_pairwise=config.dim_pairwise,
                bias_proj=bias_proj,
                pair_norm=getattr(config, "pair_norm", True),
                initial_norm=getattr(config, "initial_norm", False),
                attention_initial_norm=bool(config.attention_initial_norm),
                use_ada_layer_norm=getattr(config, "use_ada_layer_norm", True),
                use_separate_layer_norm=getattr(config, "use_separate_layer_norm", True),
                chain_kv_norm=getattr(config, "chain_kv_norm", True),
                attn_output_gate=config.attn_output_gate,
                conditioned_transition_using_silu=cts,
                transition_expansion_factor=config.transition_expansion_factor,
                dtype=dtype,
                skip_create_weights=config.skip_create_weights,
                attn_backend=attn_backend,
            )
            # Protenix ``layernorm_z`` has no offset (create_offset=False).
            proj_ln = layer.pair_bias_attn.proj_z[0]
            layer.pair_bias_attn.proj_z[0] = nn.LayerNorm(
                proj_ln.normalized_shape, bias=False, eps=proj_ln.eps, dtype=dtype
            )
            self.layers.append(layer)

        self._num_heads = config.num_heads
        self._norm_eps = config.norm_epsilon
        self.pairwise_attention_backend = attn_backend
        self._bias_pad_multiple = 8 if attn_backend == "CuTeDSL" else -1
        self._precompute_bias = bool(getattr(config, "precompute_bias", True)) and bias_proj

    @staticmethod
    def build_attn_metadata(num_blocks: int, n_queries: int, n_keys: int, device: torch.device) -> AttentionMetadata:
        """Build metadata with local-window gather indices."""
        gather_indices, _ = create_gather_indices(num_blocks, n_queries, n_keys, device)

        def _query_to_keys(x: torch.Tensor) -> torch.Tensor:
            return query_to_keys_optimized(x, gather_indices, W=n_queries, H=n_keys)

        return AttentionMetadata(query_to_keys=_query_to_keys, bias_cache={})

    def _precompute_all_biases(self, z: torch.Tensor) -> list[torch.Tensor]:
        # Rebuild to observe direct weight mutations.
        w_mega = build_bias_mega_weight(self.layers)
        return precompute_pair_biases(
            z, w_mega, len(self.layers), self._num_heads, self._norm_eps, self._bias_pad_multiple
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        n_queries: int | None = None,
        n_keys: int | None = None,
        attn_metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
    ) -> torch.Tensor:
        """Apply local atom or global token diffusion attention.

        Args:
            a: ``[B, N, C]`` locally or ``[B, S, N, C]`` globally.
            s: conditioning with the same leading dimensions as ``a``.
            z: ``[B, K, Q, Kv, C_z]`` locally or ``[B, N, N, C_z]`` globally;
                global ``z`` broadcasts over ``S``.
            mask: validity mask ``[B, N]``.
            n_queries / n_keys: window sizes; both enable the local path.
            attn_metadata: optional local gather metadata.
            buffers: optional shared layer-stack buffers.

        Returns:
            Updated representation with the same shape as ``a``.
        """
        local = n_queries is not None and n_keys is not None
        if local:
            B, N, _ = a.shape
            W, K = n_queries, z.shape[1]
            if attn_metadata is None:
                attn_metadata = self.build_attn_metadata(K, n_queries, n_keys, a.device)
            a_in = to_blocks(a, K, W).unsqueeze(1)  # [B, 1, K, W, c_a]
            s_in = to_blocks(s, K, W).unsqueeze(1)  # [B, 1, K, W, c_a]
            z_in = z.unsqueeze(1)  # [B, 1, K, W, H, c_z]
            mask_in = to_blocks(mask.unsqueeze(-1).to(a.dtype), K, W).squeeze(-1).unsqueeze(1)  # [B, 1, K, W]
        else:
            # Global self-attention: no windowing, full pair bias, no gather.
            a_in, s_in, z_in, mask_in = a, s, z, mask
            attn_metadata = None

        # CuTeDSL reuses shared output and LSE buffers across layers.
        if buffers is None and self.pairwise_attention_backend == "CuTeDSL":
            buffers = {}

        if self._precompute_bias:
            all_biases = self._precompute_all_biases(z_in)
            for layer in self.layers:
                layer.pair_bias_attn.bias_proj = False
            try:
                for i, layer in enumerate(self.layers):
                    a_in = layer(a_in, s_in, all_biases[i], mask_in, attn_metadata, buffers=buffers)
            finally:
                for layer in self.layers:
                    layer.pair_bias_attn.bias_proj = True
        else:
            for layer in self.layers:
                a_in = layer(a_in, s_in, z_in, mask_in, attn_metadata, buffers=buffers)

        if local:
            return a_in.reshape(B, K * W, -1)[:, :N]
        return a_in
