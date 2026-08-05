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


import torch
import torch.nn as nn

from tensorrt_bionemo._torch.custom_ops.fused_ln_proj_moveaxis_pad import LNProjMoveaxisPad
from tensorrt_bionemo._torch.custom_ops.gated_sigmoid import get_gated_sigmoid_op
from tensorrt_bionemo._torch.layers.linear import Linear, WeightMode, WeightsLoadingConfig
from tensorrt_bionemo._torch.layers.normalization import AdaLN
from tensorrt_bionemo.runtime.buffers import PreallocatedBuffers, ensure_buffer

from ..attention_backend import AttentionMetadata, AttentionType
from ..attention_backend.utils import create_attention
from ..tensor_utils import permute_final_dims


class TriangleAttention(nn.Module):
    """
    A module that implements the triangle attention mechanism with tensor parallelism in torch.
    The module implements lines 2,4, and 5-7 from Algorithm 14 from https://www.nature.com/articles/s41586-024-07487-w#Sec19.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int | None = None,
        layer_idx: int,
        bias_flags: dict[str, bool] | None = None,
        gating: bool = True,
        dtype: torch.dtype = None,
        skip_create_weights: bool = False,
        attn_backend: str = "VANILLA",
    ):
        super().__init__()
        if bias_flags is None:
            bias_flags = {
                "q": False,
                "k": False,
                "v": False,
                "g": False,
                "o": False,
            }
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = head_dim

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.qkv_proj = Linear(
            self.hidden_size,
            self.q_size + 2 * self.kv_size,
            bias=bias_flags["q"] or bias_flags["k"] or bias_flags["v"],
            dtype=dtype,
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_QKV_LINEAR),
            skip_create_weights=skip_create_weights,
        )
        self.o_proj = Linear(
            self.q_size,
            self.hidden_size,
            bias=bias_flags["o"],
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        self.g_proj = None
        if gating:
            self.g_proj = Linear(
                self.hidden_size,
                self.q_size,
                bias=bias_flags["g"],
                dtype=dtype,
                skip_create_weights=skip_create_weights,
            )
        self.attn = create_attention(
            attn_backend,
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            attention_type=AttentionType.TRIANGLE,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        attn_metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, I, J, F]
            biases: Include two biases:
                - mask_bias: [B, I, 1, 1, J]
                - triangle_bias: [B, H, J, J]
            buffers: Optional dict of shared pre-allocated output buffers
                (e.g. ``tri_attn_output`` and ``tri_attn_lse``) to avoid
                per-call allocation. The LSE buffer is consumed by the
                left-mask CuTeDSL kernels (Ampere SM80/86/89 and Hopper
                SM90).
        """
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # q: [B, I, J, H*D] → kernel output: [B*I, J, H, D]
        attn_buf = (
            ensure_buffer(
                buffers,
                "tri_attn_output",
                (q.shape[0] * q.shape[1], q.shape[2], self.num_heads, self.head_dim),
                q.dtype,
                q.device,
            )
            if buffers is not None
            else None
        )
        # LSE companion: [B*I, J, H, 1] float32. Consumed by the left-mask
        # CuTeDSL kernels (Ampere SM80/86/89 and Hopper SM90); ignored by all
        # other backends (they accept it via **kwargs without using it).
        attn_lse_buf = (
            ensure_buffer(
                buffers,
                "tri_attn_lse",
                (q.shape[0] * q.shape[1], q.shape[2], self.num_heads, 1),
                torch.float32,
                q.device,
            )
            if buffers is not None
            else None
        )
        mha_o = self.attn.forward(
            q, k, v, biases=biases, metadata=attn_metadata, output=attn_buf, output_lse=attn_lse_buf
        )
        if self.g_proj is not None:
            mha_flat = mha_o.reshape(-1, self.num_heads * self.head_dim)
            _gs_op = get_gated_sigmoid_op(mha_flat.dtype)
            attn_output = _gs_op(hidden_states, self.g_proj.weight, mha_flat, self.g_proj.bias, output=mha_flat)
            attn_output = attn_output.reshape(hidden_states.shape[:-1] + (self.num_heads * self.head_dim,))
        else:
            attn_output = mha_o.reshape(mha_o.shape[:-2] + (self.num_heads * self.head_dim,))
        if not attn_output.is_contiguous():
            attn_output = attn_output.contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class CrossTriangleAttention(nn.Module):
    """
    A module that implements the triangle attention mechanism with tensor parallelism in torch
    """

    def __init__(
        self,
        *,
        q_hidden_size: int,
        kv_hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int | None = None,
        layer_idx: int,
        bias_flags: dict[str, bool] | None = None,
        gating: bool = True,
        dtype: torch.dtype = None,
        skip_create_weights: bool = False,
        attn_backend: str = "VANILLA",
    ):
        super().__init__()
        if bias_flags is None:
            bias_flags = {
                "q": False,
                "k": False,
                "v": False,
                "g": False,
                "z": False,
                "o": False,
            }
        self.layer_idx = layer_idx
        self.q_hidden_size = q_hidden_size
        self.kv_hidden_size = kv_hidden_size

        self.num_heads = num_attention_heads
        self.head_dim = head_dim
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.q_proj = Linear(
            q_hidden_size,
            self.q_size,
            bias=bias_flags["q"],
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

        self.kv_proj = Linear(
            kv_hidden_size,
            2 * self.kv_size,
            bias=bias_flags["k"] or bias_flags["v"],
            dtype=dtype,
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_KV_LINEAR),
            skip_create_weights=skip_create_weights,
        )
        self.o_proj = Linear(
            self.q_size,
            self.q_hidden_size,
            bias=bias_flags["o"],
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        self.g_proj = None
        if gating:
            self.g_proj = Linear(
                self.q_hidden_size,
                self.q_size,
                bias=bias_flags["g"],
                dtype=dtype,
                skip_create_weights=skip_create_weights,
            )
        self.attn = create_attention(
            attn_backend,
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            attention_type=AttentionType.TRIANGLE,
        )

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """
        Currently used only by the TemplatePointwiseAttention layer.
        Args:
            q_x: [*, N_res, N_res, 1, C_z]
            kv_x: [*, N_res, N_res, N_temp, C_t]
            biases: Include bias:
                - triangle_bias: [B, 1, 1, 1, N_temp]
        """
        q = self.q_proj(q_x)
        kv = self.kv_proj(kv_x)
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        mha_o = self.attn.forward(q, k, v, biases=biases, metadata=attn_metadata)
        if self.g_proj is not None:
            mha_flat = mha_o.reshape(mha_o.shape[:-2] + (self.num_heads * self.head_dim,))
            _gs_op = get_gated_sigmoid_op(mha_flat.dtype)
            attn_output = _gs_op(q_x, self.g_proj.weight, mha_flat, self.g_proj.bias)
        else:
            attn_output = mha_o.reshape(mha_o.shape[:-2] + (self.num_heads * self.head_dim,))
        if not attn_output.is_contiguous():
            attn_output = attn_output.contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class AttentionPairBias(nn.Module):
    """
    Self-attention with pair bias and tensor-parallel projections.

    This layer is used by pairformer-style blocks across Boltz, OpenFold2, and
    OpenFold3. The core attention pattern is similar across those variants: the
    single representation provides queries/keys/values while the pair
    representation contributes an additive attention bias.

    OpenFold3 adds two notable differences compared with Boltz and OpenFold2:
    it can use separate layer norms for the query and key/value paths, and it
    can derive the key/value inputs with a `query_to_keys` mapping from
    `attn_metadata` for sequence-local atom attention. In practice,
    `use_separate_layer_norm=True` is used for OpenFold3, while Boltz and
    OpenFold2 keep it disabled. OpenFold3 may also disable pair normalization
    (`pair_norm=False`) before projecting the pair bias.

    With `chain_kv_norm=True`, local key/value inputs are gathered from the
    query-normalized representation before applying their AdaLN.
    """

    def __init__(
        self,
        layer_idx: int,
        c_s: int,
        c_z: int,
        num_heads: int,
        initial_norm: bool = True,
        bias_proj: bool = False,
        pair_norm: bool = True,
        dtype: torch.dtype = None,
        inf: float = 1e6,
        eps: float = 1e-5,
        use_initial_ada_layer_norm: bool = False,
        use_separate_layer_norm: bool = False,
        use_ada_layer_norm: bool = True,
        chain_kv_norm: bool = False,
        gate_bias: bool = False,
        skip_create_weights: bool = False,
        attn_backend: str = "VANILLA",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.initial_norm = initial_norm
        # Use separate layer norm for query and key instead of a shared one (e.g. OpenFold3).
        self.use_separate_layer_norm = use_separate_layer_norm
        self.use_ada_layer_norm = use_ada_layer_norm
        self.chain_kv_norm = chain_kv_norm
        self.inf = inf

        self.num_key_value_heads = num_heads
        # This equal to 1 for self-attention
        self.num_key_value_groups = num_heads // self.num_key_value_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        self.bias_proj = bias_proj

        self.norm_s = None
        if initial_norm:
            self.norm_s = nn.LayerNorm(c_s, dtype=dtype, eps=eps)

        if self.use_separate_layer_norm:
            if self.use_ada_layer_norm:
                self.layer_norm_a_q = AdaLN(
                    self.c_s, self.c_s, eps=eps, dtype=dtype, skip_create_weights=skip_create_weights
                )
                self.layer_norm_a_k = AdaLN(
                    self.c_s, self.c_s, dtype=dtype, eps=eps, skip_create_weights=skip_create_weights
                )
            else:
                self.layer_norm_a_q = nn.LayerNorm(c_s, dtype=dtype, eps=eps)
                self.layer_norm_a_k = nn.LayerNorm(c_s, dtype=dtype, eps=eps)

        self.proj_q = Linear(
            self.c_s,
            self.q_size,
            bias=True,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        self.proj_kv = Linear(
            self.c_s,
            2 * self.kv_size,
            bias=False,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_KV_LINEAR),
        )
        self.proj_g = Linear(
            self.c_s,
            self.q_size,
            bias=gate_bias,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        if self.bias_proj:
            linear_z = Linear(
                c_z,
                self.num_heads,
                bias=False,
                dtype=dtype,
                skip_create_weights=skip_create_weights,
            )
            if pair_norm:
                self.proj_z = nn.Sequential(
                    nn.LayerNorm(c_z, dtype=dtype, eps=eps),
                    linear_z,
                )
            else:
                self.proj_z = nn.Sequential(linear_z)
        self.proj_o = Linear(
            self.q_size,
            self.c_s,
            bias=False,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        self.attn = create_attention(
            attn_backend,
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            attention_type=AttentionType.PAIRWISE,
        )
        self.attn_backend = attn_backend
        self._bias_pad_multiple = 8 if attn_backend == "CuTeDSL" else -1
        self._ln_proj_moveaxis_pad = (
            LNProjMoveaxisPad(D=c_z, H=self.num_heads, dtype=dtype or torch.bfloat16) if self.bias_proj else None
        )

    def _prep_inputs(
        self,
        s: torch.Tensor,
        mask: torch.Tensor,
        single_embedding: torch.Tensor | None,
        attn_metadata: AttentionMetadata | None,
        mask_bias: torch.Tensor | None,
        mask_bias_local: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Normalize *s*, derive *kv_in*, and route masks through ``query_to_keys``.

        Two distinct flows depending on the model family:

        **Boltz family** (``single_embedding=None``, no AdaLN):
            1. ``s`` is LayerNorm'd (``initial_norm``) → ``kv_in = s``.
            2. If ``attn_metadata.query_to_keys`` is provided (sequence-local
               atom attention), ``kv_in`` and ``mask`` are gathered to the
               local neighbourhood.  Separate Q/K LayerNorms are applied if
               ``use_separate_layer_norm`` is set.

        **OpenFold3** (``single_embedding`` provided, ``use_ada_layer_norm=True``):
            1. ``s`` is LayerNorm'd → ``kv_in = s``.
            2. When ``query_to_keys`` is present, ``single_embedding`` is also
               gathered to produce ``single_embedding_kv``.  AdaLN then
               conditions the Q and K paths independently:
               ``s = AdaLN_q(s, single_embedding)`` and
               ``kv_in = AdaLN_k(kv_in, single_embedding_kv)``.

        Args:
            s: Token or atom embedding (see ``forward`` for shapes).
            mask: Sequence mask ``[B, (*), N]``.
            single_embedding: Single representation for AdaLN conditioning.
                ``None`` for Boltz family; ``[B, (*), N, C_s]`` for OpenFold3.
            attn_metadata: Optional metadata with ``query_to_keys`` gather
                indices (used in sequence-local atom attention) and
                ``bias_cache``.
            mask_bias: Optional precomputed additive mask bias.
            mask_bias_local: Optional precomputed mask bias already gathered
                via ``query_to_keys``.  When provided, skips re-gathering
                *mask* and uses this directly.

        Returns:
            ``(s, kv_in, mask, mask_bias)`` ready for the projection and
            bias stages.
        """
        if self.initial_norm:
            s = self.norm_s(s)
        if not s.is_contiguous():
            s = s.contiguous()
        kv_in = s

        if attn_metadata is not None:
            query_to_keys = attn_metadata.query_to_keys

            if query_to_keys is not None:
                if mask_bias_local is not None:
                    mask_bias = mask_bias_local
                else:
                    mask = query_to_keys(mask.unsqueeze(-1)).squeeze(-1)
                    mask_bias = None
                if self.use_separate_layer_norm and self.chain_kv_norm:
                    # Gather commutes with the per-row query AdaLN.
                    assert self.use_ada_layer_norm, "chain_kv_norm requires use_ada_layer_norm"
                    assert single_embedding is not None, "single_embedding is required for AdaLN"
                    s = self.layer_norm_a_q(s, single_embedding)
                    kv_in = self.layer_norm_a_k(query_to_keys(s), query_to_keys(single_embedding))
                else:
                    kv_in = query_to_keys(s)
                    if self.use_separate_layer_norm:
                        if self.use_ada_layer_norm:
                            assert single_embedding is not None, "single_embedding is required for AdaLN"
                            single_embedding_kv = query_to_keys(single_embedding)
                            s = self.layer_norm_a_q(s, single_embedding)
                            kv_in = self.layer_norm_a_k(kv_in, single_embedding_kv)
                        else:
                            s = self.layer_norm_a_q(s)
                            kv_in = self.layer_norm_a_k(kv_in)

        return s, kv_in, mask, mask_bias

    def _prep_qkv(
        self,
        s: torch.Tensor,
        kv_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project Q from *s* and packed K/V from *kv_in*.

        Returns:
            (q, k, v) — k and v are non-contiguous views of the fused KV buffer.
        """
        q = self.proj_q(s)
        kv = self.proj_kv(kv_in)
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        return q, k, v

    def _prep_mask_bias(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        mask_bias: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Build the ``[mask_bias, pair_bias]`` list consumed by the attention kernel.

        Args:
            s: Token or atom embedding.
                - Token transformer: ``[B, N, C_s]`` or ``[B, S, N, C_s]``.
                - Atom transformer: ``[B, 1, K, N_q, C_s]`` or ``[B, S, K, N_q, C_s]``.
            z: Pair representation.
                - With ``bias_proj``: ``[B, (*), N_q, N_k, C_z]`` — projected
                  by ``LNProjMoveaxisPad`` to ``[B, (*), H, N_q, N_k_padded]``.
                - Without ``bias_proj``: already ``[B, (*), H, N_q, N_k]``.
            mask: Sequence mask.
                - Token path: ``[B, N]``.
                - Atom path: ``[B, K, N_q]``.
            mask_bias: Optional precomputed additive mask bias.  When ``None``,
                computed from *mask* (see below).

        Shape logic by backend:

        **CuTeDSL** — biases are consumed directly by the CUTLASS kernel:
            - ``mask_bias``: ``mask.float()`` — same shape as *mask*.
            - ``pair_bias``: output of ``LNProjMoveaxisPad`` — same ndim as *z*.
            No extra unsqueeze is applied; the kernel handles broadcasting.

        **SDPA / VANILLA** — biases are added to the attention logits in PyTorch:
            - ``mask_bias``: ``(1 - mask) * -inf``, with trailing dims
              ``[..., 1, 1, N_k]`` for per-key masking.
            - ``pair_bias``: ``[B, (*), H, N_q, N_k_padded]`` after projection.
            Both are then unsqueezed at dim 1 until ``ndim == s.ndim + 1``
            so they broadcast over any multiplicity / sample dimensions that
            *s* carries but *z* does not.  Final shapes:
                - Token ``mult=5``: ``[B, 1, H, N, N]`` broadcasts with
                  q ``[B, mult, H, N, D]``.
                - Atom: ``[B, 1, K, H, N_q, N_k]`` broadcasts with
                  q ``[B, mult, K, H, N_q, D]``.

        Returns:
            ``[mask_bias, pair_bias]`` — list of length 2.
        """
        if mask_bias is None:
            if self.attn_backend == "CuTeDSL":
                mask_bias = mask.float()
            else:
                mask = mask[..., None, None, :]
                mask_bias = (1 - mask.float()) * -self.inf

        pair_bias = z
        if self.bias_proj:
            ln = self.proj_z[0] if len(self.proj_z) > 1 else None
            proj = self.proj_z[-1]
            pair_bias = self._ln_proj_moveaxis_pad(
                z,
                ln_weight=ln.weight if ln is not None else None,
                ln_bias=getattr(ln, "bias", None) if ln is not None else None,
                proj_weight=proj.weight,
                pad_multiple=self._bias_pad_multiple,
                proj_z=self.proj_z,
            )

        if self.attn_backend != "CuTeDSL":
            # Insert broadcast-1 dims so pair_bias broadcasts over any
            # multiplicity dimensions that s carries but z does not.
            # The extra dims (e.g. sample/multiplicity) sit at position 1
            # (right after the batch dim), so we insert there rather than
            # at a fixed negative offset.
            while pair_bias.ndim < s.ndim + 1:
                pair_bias = pair_bias.unsqueeze(1)
            while mask_bias.ndim < pair_bias.ndim:
                mask_bias = mask_bias.unsqueeze(1)

        return [mask_bias, pair_bias]

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        single_embedding: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        mask_bias: torch.Tensor | None = None,
        mask_bias_local: torch.Tensor | None = None,
        buffers: PreallocatedBuffers | None = None,
    ) -> torch.Tensor:
        """Single and TP Distributed version for AttentionPairBias.

        Args:
            s: Token or atom-level embedding.
                - Token transformer: ``[B, N, C_s]`` or ``[B, mult, N, C_s]``
                  where *mult* is the number of diffusion samples / rollouts.
                - Atom transformer: ``[B, 1, K, N_q, C_s]`` or
                  ``[B, mult, K, N_q, C_s]`` where *K* is the number of
                  local-attention blocks.
            z: Pair representation or pre-projected pair bias.
                - With ``bias_proj=True``: ``[B, (*), N_q, N_k, C_z]`` —
                  internally projected to ``[B, (*), H, N_q, N_k_padded]``.
                - With ``bias_proj=False``: ``[B, (*), H, N_q, N_k]``,
                  already projected by the caller.
            mask: Sequence-level binary mask (1 = valid, 0 = padded).
                - Token path: ``[B, N]``.
                - Atom path: ``[B, K, N_q]``.
            single_embedding: Optional single representation ``[B, (*), N, C_s]``
                used by AdaLN conditioning (OpenFold3 diffusion transformer).
                ``None`` when AdaLN is not used.
            attn_metadata: Optional attention metadata carrying
                ``query_to_keys`` gather indices for sequence-local attention
                and an optional ``bias_cache`` for reusing projected pair bias
                across layers.
                ``None`` for single-GPU execution.
            mask_bias: Optional precomputed additive mask bias.
                - SDPA / VANILLA: ``[B, (*), 1, 1, N_k]`` — additive
                  ``(1 - mask) * -inf`` bias already expanded for broadcasting.
                - CuTeDSL: ``[B, (*), N]`` — ``mask.float()`` passed directly.
                When ``None``, computed internally from *mask*.
            mask_bias_local: Optional precomputed mask bias after
                ``query_to_keys`` gathering, shape ``[B, (*), 1, 1, N_k_local]``.
                Used in sequence-local atom attention to avoid recomputing
                the gathered mask each layer.
            buffers: Optional dict of shared pre-allocated output buffers
                (e.g. ``pw_attn_output`` and ``pw_attn_lse``) to avoid
                per-call allocation. The LSE buffer is consumed by the
                left-mask CuTeDSL kernels (Ampere SM80/86/89 and Hopper
                SM90).

        Returns:
            Output tensor with the same shape as *s*.
        """
        s, kv_in, mask, mask_bias = self._prep_inputs(
            s, mask, single_embedding, attn_metadata, mask_bias, mask_bias_local
        )

        q, k, v = self._prep_qkv(s, kv_in)

        biases = self._prep_mask_bias(s, z, mask, mask_bias)

        _b_flat = 1
        for _d in q.shape[:-2]:
            _b_flat *= _d
        attn_buf = (
            ensure_buffer(
                buffers, "pw_attn_output", (_b_flat, q.shape[-2], self.num_heads, self.head_dim), q.dtype, q.device
            )
            if buffers is not None
            else None
        )
        # LSE companion: [B_flat, Sq, H, 1] float32. Consumed by the
        # left-mask CuTeDSL kernels (Ampere SM80/86/89 and Hopper SM90);
        # ignored by all other backends (they accept it via **kwargs without
        # using it).
        attn_lse_buf = (
            ensure_buffer(buffers, "pw_attn_lse", (_b_flat, q.shape[-2], self.num_heads, 1), torch.float32, q.device)
            if buffers is not None
            else None
        )
        mha_o = self.attn.forward(
            q, k, v, biases=biases, metadata=attn_metadata, output=attn_buf, output_lse=attn_lse_buf
        )
        batch_dims = mha_o.shape[:-2]
        o = mha_o.reshape(-1, self.num_heads * self.head_dim)

        _gs_op = get_gated_sigmoid_op(o.dtype)
        o = _gs_op(s, self.proj_g.weight, o, self.proj_g.bias, output=o)
        o = o.reshape(*batch_dims, self.num_heads * self.head_dim)
        o = self.proj_o(o)
        return o


class MSAAttention(nn.Module):
    def __init__(
        self,
        *,
        local_layer_idx: int,
        c_in: int,
        num_heads: int,
        c_z: int | None = None,
        triangle_attn_backend: str = "VANILLA",
        support_batch: bool = True,
        need_project_z: bool = True,
        transpose_input: bool = False,
        bias_flags: dict[str, bool] | None = None,
        eps: float = 1e-5,
        inf: float = 1e9,
        dtype: torch.dtype = None,
        skip_create_weights: bool = False,
        **kwargs,
    ):
        super().__init__()
        if bias_flags is None:
            bias_flags = {"z": False}
        assert support_batch, "support_batch is required for MSAAttention"
        self.local_layer_idx = local_layer_idx
        self.num_heads = num_heads
        self.c_in = c_in
        self.c_z = c_z
        self.inf = inf
        self.support_batch = support_batch
        self.dtype = dtype
        self.transpose_input = transpose_input
        self.triangle_attn_backend = triangle_attn_backend
        self.J_padded_multiple = 8 if triangle_attn_backend == "CuTeDSL" else -1

        self.layer_norm_m = nn.LayerNorm(c_in, dtype=dtype, eps=eps)

        self.proj_z_norm = None
        self.proj_z = None
        self._ln_proj_moveaxis_pad = None
        if need_project_z:
            self._ln_proj_moveaxis_pad = LNProjMoveaxisPad(D=c_z, H=self.num_heads, dtype=dtype or torch.bfloat16)
            self.proj_z_norm = nn.LayerNorm(c_z, dtype=dtype, eps=eps)
            self.proj_z = Linear(
                self.c_z,
                self.num_heads,
                bias=bias_flags["z"],
                dtype=dtype,
                skip_create_weights=skip_create_weights,
            )
        self.mha = TriangleAttention(
            layer_idx=local_layer_idx,
            hidden_size=self.c_in,
            head_dim=self.c_in // self.num_heads,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_heads,
            gating=True,
            bias_flags={
                "q": False,
                "k": False,
                "v": False,
                "g": True,
                "z": False,
                "o": True,
            },
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            attn_backend=self.triangle_attn_backend,
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
    ):
        """
        Args:
            m: [*, J, I, c_in]
            z: [*, I, I, c_z]
            mask: [*, J, I]
            buffers: Shared pre-allocated buffer dict.
        """
        if self.transpose_input:
            m = permute_final_dims(m, (1, 0, 2))
            mask = permute_final_dims(mask, (1, 0))
        if self.triangle_attn_backend == "CuTeDSL":
            mask_bias = (mask > 0.5).sum(dim=-1).to(torch.int32).contiguous()
        else:
            mask_bias = (mask - 1.0) * self.inf
            mask_bias = mask_bias.unsqueeze(-2).unsqueeze(-3)

        if self.proj_z_norm and self.proj_z and z is not None:
            z = self._ln_proj_moveaxis_pad(
                z,
                ln_weight=self.proj_z_norm.weight,
                ln_bias=self.proj_z_norm.bias,
                proj_weight=self.proj_z.weight,
                pad_multiple=self.J_padded_multiple,
                proj_z=nn.Sequential(self.proj_z_norm, self.proj_z),
            )
        else:
            if self.triangle_attn_backend != "VANILLA":
                z_shape = [*m.shape[: m.ndim - 3], self.num_heads, m.size(-2), m.size(-2)]
                z = torch.zeros(z_shape, dtype=self.dtype, device=m.device)
        biases = [mask_bias, z]
        m = self.layer_norm_m(m)

        output = self.mha(m, biases=biases, attn_metadata=attn_metadata, buffers=buffers)
        if self.transpose_input:
            output = permute_final_dims(output, (1, 0, 2))
        return output


class GlobalAttention(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        bias_flags: dict[str, bool] | None = None,
        eps: float = 1e-5,
        inf: float = 1e9,
        dtype: torch.dtype = None,
        skip_create_weights: bool = False,
        **kwargs,
    ):
        super().__init__()
        if bias_flags is None:
            bias_flags = {
                "q": False,
                "k": False,
                "v": False,
                "g": True,
                "o": True,
            }
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps
        self.dtype = dtype

        self.proj_q = Linear(
            c_in,
            c_hidden * self.no_heads,
            bias=bias_flags["q"],
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

        self.fused_proj_kv = Linear(
            c_in,
            c_hidden * 2,
            bias=bias_flags["k"] or bias_flags["v"],
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_KV_LINEAR),
        )
        self.proj_g = Linear(
            c_in,
            c_hidden * self.no_heads,
            bias=bias_flags["g"],
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        self.proj_o = Linear(
            c_hidden * self.no_heads,
            c_in,
            bias=bias_flags["o"],
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, m: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            m: [*, N_res, C_in]
            mask: [B, N_res, N_seq]
        """
        kv = self.fused_proj_kv(m)
        k, v = kv.split([self.c_hidden, self.c_hidden], dim=-1)

        q = torch.sum(m * mask.unsqueeze(-1), dim=-2) / (torch.sum(mask, dim=-1)[..., None] + self.eps)

        q = self.proj_q(q)  # tp by heads not by c_hidden
        q *= self.c_hidden ** (-0.5)
        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        bias = (self.inf * (mask - 1))[..., :, None, :]
        a = torch.matmul(
            q,
            k.transpose(-1, -2),  # [*, N_res, C_hidden, N_seq]
        )
        a += bias  # [*, N_res, H, N_seq]
        a = torch.nn.functional.softmax(a, dim=-1)
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a,
            v,
        )

        # [*, N_res, N_seq, C_hidden*H]
        g = self.sigmoid(self.proj_g(m))
        # [*, N_res, N_seq, H, C_hidden]
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, H, C_hidden]
        o = o.unsqueeze(-3) * g

        # [*, N_res, N_seq, H * C_hidden]
        o = o.reshape(o.shape[:-2] + (-1,))

        # [*, N_res, N_seq, C_in]
        m = self.proj_o(o)

        return m


class MSAColumnGlobalAttention(nn.Module):
    def __init__(
        self,
        *,
        local_layer_idx: int,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        attn_bias_flags: dict[str, bool] | None = None,
        eps: float = 1e-5,
        inf: float = 1e9,
        dtype: torch.dtype = None,
        skip_create_weights: bool = False,
        **kwargs,
    ):
        super().__init__()
        if attn_bias_flags is None:
            attn_bias_flags = {
                "q": False,
                "k": False,
                "v": False,
                "g": True,
                "o": True,
            }
        self.local_layer_idx = local_layer_idx
        self.layer_norm_m = nn.LayerNorm(c_in, dtype=dtype, eps=eps)

        self.global_attention = GlobalAttention(
            c_in=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
            bias_flags=attn_bias_flags,
            eps=eps,
            inf=inf,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

    def forward(self, m: torch.Tensor, mask: torch.Tensor):
        # [*, N_seq, N_res]
        m = m.transpose(-2, -3)
        mask = mask.transpose(-1, -2)
        m = self.layer_norm_m(m)
        m = self.global_attention(m=m, mask=mask)

        # [*, N_seq, N_res, C_in]
        m = m.transpose(-2, -3)

        return m
