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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.attention_backend.utils import (
    PrecomputedSingleMasks, precompute_single_masks)
from tensorrt_bionemo._torch.custom_ops.fused_layer_norm_no_affine import \
    fused_layer_norm_no_affine
from tensorrt_bionemo._torch.custom_ops.gated_sigmoid import \
    get_gated_sigmoid_op
from tensorrt_bionemo._torch.distributed import AllReduceParams
from tensorrt_bionemo._torch.layers.attention import AttentionPairBias
from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo._torch.layers.normalization import AdaLN
from tensorrt_bionemo._torch.layers.transition import \
    ConditionedTransitionBlock
from tensorrt_bionemo._torch.utils import recursive_calling_load_weights
from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.runtime.buffers import PreallocatedBuffers, ensure_buffer


class DiffusionTransformerLayer(nn.Module):

    def __init__(self,
                 layer_idx: int,
                 num_heads: int,
                 dim: int = 384,
                 dim_single_cond: Optional[int] = None,
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
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False,
                 initial_norm: bool = True,
                 conditioned_transition_using_silu: bool = False,
                 attn_output_gate: bool = True,
                 attn_gate_bias: bool = False,
                 transition_expansion_factor: int = 2,
                 attn_backend: str = "VANILLA"):
        super().__init__()

        self.initial_norm = initial_norm
        self.use_separate_layer_norm = use_separate_layer_norm
        self.attn_output_gate = attn_output_gate

        if initial_norm:
            self.adaln = AdaLN(dim,
                               dim_single_cond,
                               eps=eps,
                               dtype=dtype,
                               mapping=mapping,
                               skip_create_weights=skip_create_weights)

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
            gate_bias=attn_gate_bias,
            eps=eps,
            inf=inf,
            dtype=dtype,
            mapping=mapping,
            skip_create_weights=skip_create_weights,
            attn_backend=attn_backend)

        self.output_projection = None
        self._can_fuse_output_gate = False
        if self.attn_output_gate:
            self.output_projection = Linear(
                dim_single_cond,
                dim,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                skip_create_weights=skip_create_weights)
            self._can_fuse_output_gate = (mapping is None
                                          or mapping.tp_size == 1)
        self.transition = ConditionedTransitionBlock(
            dim_single=dim,
            dim_single_cond=dim_single_cond,
            expansion_factor=transition_expansion_factor,
            dtype=dtype,
            mapping=mapping,
            skip_create_weights=skip_create_weights,
            using_silu=conditioned_transition_using_silu)
        self.post_lnorm = None
        if post_layer_norm:
            self.post_lnorm = nn.LayerNorm(dim, dtype=dtype, eps=eps)

    def forward(
            self,
            a: torch.Tensor,
            s: torch.Tensor,
            bias: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            attn_metadata: Optional[AttentionMetadata] = None,
            all_reduce_params: Optional[AllReduceParams] = None,
            precomputed_single_masks: Optional[PrecomputedSingleMasks] = None,
            buffers: Optional[PreallocatedBuffers] = None,
            **kwargs) -> torch.Tensor:
        """ First version of DiffusionTransformerLayer, does not support multiplicity > 1 and atom encoder, decoder"""
        if self.initial_norm:
            b = self.adaln(a, s, buffers=buffers,
                           buffer_key="dit_bsd_scratch")
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
            all_reduce_params=all_reduce_params,
            mask_bias=mask_bias,
            mask_bias_local=mask_bias_local,
            buffers=buffers)
        if self.attn_output_gate:
            if self._can_fuse_output_gate and s.shape[:-1] == b.shape[:-1]:
                _gs_op = get_gated_sigmoid_op(b.dtype)
                gs_buf = ensure_buffer(buffers, "dit_bsd_scratch", b.shape,
                                       b.dtype, b.device)
                b = _gs_op(s,
                           self.output_projection.weight,
                           b,
                           self.output_projection.bias,
                           output=gs_buf)
            else:
                b = F.sigmoid(self.output_projection(s)) * b
        a = a + b
        a = a + self.transition(a, s,
                                all_reduce_params=all_reduce_params,
                                buffers=buffers,
                                buffer_key="dit_bsd_scratch")
        if self.post_lnorm is not None:
            a = self.post_lnorm(a)
        return a


class BoltzDiffusionTransformer(nn.Module):

    def __init__(self, config: BaseConfig):
        """
        Args:
            config: tensorrt_bionemo.models.boltz1.configs.DiffusionTransformerConfig
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
                    mapping=config.mapping,
                    skip_create_weights=config.skip_create_weights,
                    attn_output_gate=getattr(config, 'attn_output_gate', True),
                    attn_gate_bias=getattr(config, 'attn_gate_bias', False),
                    transition_expansion_factor=getattr(
                        config, 'transition_expansion_factor', 2),
                    attn_backend=config.pairwise_attention_backend,
                ))

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # verify whether all the weights are loaded
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weights}")

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
            z: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            attn_metadata: Optional[AttentionMetadata] = None,
            all_reduce_params: Optional[AllReduceParams] = None,
            precomputed_single_masks: Optional[PrecomputedSingleMasks] = None,
            buffers: Optional[PreallocatedBuffers] = None,
            **kwargs) -> torch.Tensor:
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
                self.pairwise_attention_backend,
                mask,
                inf=self.mask_inf,
                query_to_keys=query_to_keys)

        if buffers is None and self.pairwise_attention_backend == "CuTeDSL":
            buffers = {}

        for i, layer in enumerate(self.layers):
            # CuTeDSL: z is [L, *, H, Sq, Sk_padded] -> z[i] is contiguous
            # Others:  z is [*, H, Sq, Sk, L]        -> z[..., i] (legacy)
            bias = z[i] if self.pairwise_attention_backend == "CuTeDSL" else z[
                ..., i]
            a = layer(a,
                      s,
                      bias,
                      mask,
                      attn_metadata,
                      all_reduce_params,
                      precomputed_single_masks=precomputed_single_masks,
                      buffers=buffers)
        return a


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
        shared_pair_norm = getattr(config, 'shared_pair_norm', False)

        if shared_pair_norm:
            self.layer_norm_z = nn.LayerNorm(config.dim_pairwise,
                                             bias=False,
                                             eps=config.norm_epsilon,
                                             dtype=self.dtype)
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
                mapping=config.mapping,
                skip_create_weights=config.skip_create_weights,
                conditioned_transition_using_silu=config.
                conditioned_transition_using_silu,
                initial_norm=True if not hasattr(config, 'initial_norm') else
                config.initial_norm,
                use_ada_layer_norm=True
                if not hasattr(config, 'use_ada_layer_norm') else
                config.use_ada_layer_norm,
                use_separate_layer_norm=False
                if not hasattr(config, 'use_separate_layer_norm') else
                config.use_separate_layer_norm,
                attn_output_gate=getattr(config, 'attn_output_gate', True),
                attn_gate_bias=getattr(config, 'attn_gate_bias', False),
                transition_expansion_factor=getattr(
                    config, 'transition_expansion_factor', 2),
                attn_backend=config.pairwise_attention_backend)

            if not shared_pair_norm and config.bias_proj:
                # Replace the default bias=True LayerNorm in proj_z[0] with
                # bias=False to match the reference model.
                dim = layer.pair_bias_attn.proj_z[0].weight.shape
                eps = layer.pair_bias_attn.proj_z[0].eps
                layer.pair_bias_attn.proj_z[0] = nn.LayerNorm(
                    dim, bias=False, eps=eps, dtype=config.torch_dtype)
            self.layers.append(layer)

        # Mega-GEMM precomputed bias: fuse per-layer LN γ into projection
        # weights → single W_mega [N*H, D].  Forward does:
        #   z_hat = layer_norm(z)          # one normalization, no affine
        #   out   = W_mega @ z_hat.T       # [NH, BIJ], J contiguous
        #   slice → N × [B, H, I, J]      # zero-copy views (B=1)
        self._precompute_bias = (getattr(config, 'precompute_bias', True)
                                 and getattr(config, 'bias_proj', False)
                                 and not shared_pair_norm)
        if self._precompute_bias:
            self._num_heads = config.num_heads
            self._dim_pairwise = config.dim_pairwise
            self._norm_eps = config.norm_epsilon
            self._bias_pad_multiple = (8 if config.pairwise_attention_backend
                                       == "CuTeDSL" else -1)
            self._W_mega: Optional[torch.Tensor] = None

    def _build_mega_weight(self) -> None:
        """Fuse per-layer LN γ into projection weights → single ``[N*H, D]``.

        ``W_fused[i, h, d] = W_proj[i, h, d] * γ_i[d]`` so that the
        per-layer ``LayerNorm(z) @ W_proj.T`` collapses to a single
        ``layer_norm_no_affine(z) @ W_mega.T``.
        """
        parts: list[torch.Tensor] = []
        for layer in self.layers:
            proj_z = layer.pair_bias_attn.proj_z
            ln = proj_z[0] if len(proj_z) > 1 else None
            proj = proj_z[-1]
            w = proj.weight.data  # [H, D]
            if ln is not None:
                w = w * ln.weight.data.unsqueeze(0)  # [H, D] * [1, D]
            parts.append(w)
        self._W_mega = torch.cat(parts, dim=0).contiguous()  # [N*H, D]

    def _precompute_all_biases(self, z: torch.Tensor) -> list[torch.Tensor]:
        """Mega-GEMM transposed: one GEMM → zero-copy per-layer views.

        1. Pad ``z`` in the J dimension to ``J_pad`` *before* the GEMM so the
           output naturally has J_pad contiguous — avoids 24 separate
           ``F.pad`` calls that would break the zero-copy views.
           (Padding cost: ~1 extra column × B×I rows × D, negligible.)
        2. Normalize ``z`` once (no affine — γ is absorbed into W_mega).
        3. ``W_mega @ z_hat.T`` → ``[NH, B*I*J_pad]`` with J_pad contiguous.
        4. Reshape to ``[N, H, B, I, J_pad]`` and slice per layer.
           For B=1 each slice is a contiguous ``[1, H, I, J_pad]`` view.

        Args:
            z: pair representation ``[B, I, J, D]`` (contiguous, bf16).

        Returns:
            List of ``num_blocks`` tensors, each ``[B, H, I, J_pad]``.
        """
        if self._W_mega is None:
            self._build_mega_weight()

        # z may have arbitrary leading batch dims: [*, I, J, D]
        *batch_dims, I, J, D = z.shape
        B = 1
        for d in batch_dims:
            B *= d
        N = len(self.layers)
        H = self._num_heads

        pad = self._bias_pad_multiple
        J_pad = ((J + pad - 1) // pad) * pad if pad > 0 else J

        if J_pad != J:
            # (0,0) on D, (0, delta) on J — F.pad works from last dim inward,
            # so arbitrary leading batch_dims are handled automatically.
            z = F.pad(z, (0, 0, 0, J_pad - J))  # [*batch_dims, I, J_pad, D]

        z_hat = fused_layer_norm_no_affine(z, eps=self._norm_eps)

        # [NH, D] @ [D, B*I*J_pad] → [NH, B*I*J_pad] with J_pad contiguous
        out = torch.mm(self._W_mega,
                       z_hat.reshape(-1,
                                     D).t().contiguous())  # [NH, B*I*J_pad]
        out = out.reshape(N, H, B, I, J_pad)

        biases: list[torch.Tensor] = []
        for i in range(N):
            if B == 1:
                b = out[i].squeeze(1).unsqueeze(0)  # [1,H,I,J_pad] view
            else:
                b = out[i].permute(1, 0, 2, 3).contiguous()
            # Restore original batch dimensions: [*batch_dims, H, I, J_pad]
            b = b.view(*batch_dims, H, I, J_pad)
            biases.append(b)
        return biases

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # verify whether all the weights are loaded
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weights}")
        if self._precompute_bias:
            self._W_mega = None

    def forward(self,
                a: torch.Tensor = None,
                s: torch.Tensor = None,
                z: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                attn_metadata: Optional[AttentionMetadata] = None,
                all_reduce_params: Optional[AllReduceParams] = None,
                buffers: Optional[PreallocatedBuffers] = None,
                **kwargs) -> torch.Tensor:

        if hasattr(self, 'layer_norm_z'):
            z = self.layer_norm_z(z)

        precomputed_single_masks = None
        if mask is not None:
            query_to_keys = attn_metadata.query_to_keys if attn_metadata else None
            precomputed_single_masks = precompute_single_masks(
                self.pairwise_attention_backend,
                mask,
                inf=self.mask_inf,
                query_to_keys=query_to_keys)

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
                        all_reduce_params,
                        precomputed_single_masks=precomputed_single_masks,
                        buffers=buffers)
            finally:
                for layer in self.layers:
                    layer.pair_bias_attn.bias_proj = True
        else:
            for layer in self.layers:
                a = layer(a,
                          s,
                          z,
                          mask,
                          attn_metadata,
                          all_reduce_params,
                          precomputed_single_masks=precomputed_single_masks,
                          buffers=buffers)
        return a
