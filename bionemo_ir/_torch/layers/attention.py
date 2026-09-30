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
import torch.nn.functional as F

from bionemo_ir._torch.custom_ops.fused_ln_proj_moveaxis_pad import LNProjMoveaxisPad
from bionemo_ir._torch.custom_ops.gated_sigmoid import get_gated_sigmoid_op
from bionemo_ir._torch.layers.linear import Linear, WeightMode, WeightsLoadingConfig
from bionemo_ir._torch.layers.normalization import AdaLN
from bionemo_ir._torch.utils import ChunkPolicy, chunk_apply, permute_final_dims
from bionemo_ir.dsl_kernels.triton.moveaxis_pad import MoveaxisPad
from bionemo_ir.runtime.buffers import PreallocatedBuffers, ensure_buffer

from ..attention_backend import AttentionMetadata, AttentionType
from ..attention_backend.utils import create_attention


def _make_norm(norm_type: str, dim: int, eps: float = 1e-5, dtype: torch.dtype = None, bias: bool = True) -> nn.Module:
    """Build a LayerNorm or RMSNorm by name (RMSNorm has weight only)."""
    if norm_type == "rms_norm":
        return nn.RMSNorm(dim, eps=eps, dtype=dtype)
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=eps, dtype=dtype, bias=bias)
    raise ValueError(f"Unsupported norm_type={norm_type!r}; expected 'layer_norm' or 'rms_norm'")


# Pad the pair-bias heads to a multiple of 32 so the fused projection width
# stays one too: cuBLAS runs that GEMM up to 2.8x slower at other widths, and
# the CuTeDSL triangle kernel needs q/k/v row strides divisible by 8 elements.
_PAIR_BIAS_ROW_ALIGN = 32


def pair_bias_rows(num_heads: int) -> int:
    """Rows ``TriangleAttention.in_proj`` reserves for pair bias: ``num_heads`` rounded up to a multiple of 32."""
    return -(-num_heads // _PAIR_BIAS_ROW_ALIGN) * _PAIR_BIAS_ROW_ALIGN


class TriangleAttention(nn.Module):
    """
    A module that implements the triangle attention mechanism with tensor parallelism in torch.
    The module implements lines 2-7 from Algorithm 14 from https://www.nature.com/articles/s41586-024-07487-w#Sec19;
    line 3, the triangle bias, only with ``bias_proj``.
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
        bias_proj: bool = False,
        transposed_bias: bool = False,
        chunk_policy: ChunkPolicy | None = None,
        dtype: torch.dtype = None,
        skip_create_weights: bool = False,
        attn_backend: str = "VANILLA",
    ):
        """
        Args:
            bias_proj: Project the triangle bias in ``in_proj`` too and use it
                when the caller passes none.
            transposed_bias: Project the triangle bias from the transposed
                ``hidden_states``.
            chunk_policy: Query-row chunking policy; ``None`` attends densely.
        """
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
        self.gating = gating
        self.bias_proj = bias_proj
        self.transposed_bias = transposed_bias
        self.chunk_policy = chunk_policy

        # One GEMM projects q, k, v, then the gate and the pair-bias heads when present.
        self._in_proj_sizes = [self.q_size, self.kv_size, self.kv_size]
        if gating:
            self._in_proj_sizes.append(self.q_size)
        if bias_proj:
            self._in_proj_sizes.append(pair_bias_rows(self.num_heads))
        self.in_proj = Linear(
            self.hidden_size,
            sum(self._in_proj_sizes),
            bias=bias_flags["q"] or bias_flags["k"] or bias_flags["v"] or (gating and bias_flags["g"]),
            dtype=dtype,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_SEGMENTS_LINEAR,
                segment_sizes=tuple(self._in_proj_sizes),
            ),
            skip_create_weights=skip_create_weights,
        )
        self.o_proj = Linear(
            self.q_size,
            self.hidden_size,
            bias=bias_flags["o"],
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        if bias_proj:
            self._bias_pad_multiple = 8 if attn_backend == "CuTeDSL" else -1
            self._moveaxis_pad = MoveaxisPad(H=self.num_heads, dtype=dtype or torch.bfloat16)
        self.attn = create_attention(
            attn_backend,
            self.layer_idx,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            attention_type=AttentionType.TRIANGLE,
        )

    def _pair_bias(self, heads: torch.Tensor) -> torch.Tensor:
        """Lay projected ``[B, I, J, H]`` bias heads out as ``[B, H, I, J_padded]``."""
        # The projection is position-wise, so transposing its heads equals
        # projecting the transposed pair. Transpose before padding: the pad
        # lands on the key axis.
        if self.transposed_bias:
            heads = heads.transpose(1, 2)
        return self._moveaxis_pad(heads, multiple=self._bias_pad_multiple)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask_bias: torch.Tensor,
        triangle_bias: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        buffers: PreallocatedBuffers | None = None,
        use_kv_lengths: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, I, J, F]
            mask_bias: Per-row mask in the backend's layout, e.g.
                [B, I, 1, 1, J] additive or CuTeDSL's [B, I] int32 lengths.
            triangle_bias: [B, H, J, J]. ``None`` projects it from
                ``hidden_states`` when the module has ``bias_proj``.
            buffers: Optional dict of shared pre-allocated output buffers
                (e.g. ``tri_attn_output`` and ``tri_attn_lse``) to avoid
                per-call allocation. The LSE buffer is consumed by the
                left-mask CuTeDSL kernels (Ampere SM80/86/89 and Hopper
                SM90).
            use_kv_lengths: Encode a left-aligned mask as per-row lengths for
                the cuEquivariance SM100f fast path.
        """
        if self.chunk_policy is None or not self.chunk_policy.should_chunk(hidden_states):
            return self._attend(
                hidden_states,
                mask_bias,
                triangle_bias=triangle_bias,
                attn_metadata=attn_metadata,
                buffers=buffers,
                use_kv_lengths=use_kv_lengths,
            )
        if triangle_bias is None and self.bias_proj:
            # A row slice cannot project the bias, which spans every row, so
            # project just the real bias heads over the whole pair first.
            start = sum(self._in_proj_sizes[:-1])
            rows = slice(start, start + self.num_heads)
            bias = None if self.in_proj.bias is None else self.in_proj.bias[rows]
            triangle_bias = self._pair_bias(F.linear(hidden_states, self.in_proj.weight[rows], bias))
        # Query rows attend independently, so row slices are exact; the
        # triangle bias passes through unsliced.
        return chunk_apply(
            self._attend,
            hidden_states,
            mask_bias,
            policy=self.chunk_policy,
            cat_dim=1,
            triangle_bias=triangle_bias,
            attn_metadata=attn_metadata,
            buffers=buffers,
            use_kv_lengths=use_kv_lengths,
        )

    def _attend(
        self,
        hidden_states: torch.Tensor,
        mask_bias: torch.Tensor,
        *,
        triangle_bias: torch.Tensor | None,
        attn_metadata: AttentionMetadata | None,
        buffers: PreallocatedBuffers | None,
        use_kv_lengths: bool,
    ) -> torch.Tensor:
        """Attend over all or a row slice of ``hidden_states``."""
        q, k, v, *gate_and_bias = self.in_proj(hidden_states).split(self._in_proj_sizes, dim=-1)
        if triangle_bias is None and self.bias_proj:
            triangle_bias = self._pair_bias(gate_and_bias[-1][..., : self.num_heads])
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
            q,
            k,
            v,
            biases=[mask_bias, triangle_bias],
            metadata=attn_metadata,
            output=attn_buf,
            output_lse=attn_lse_buf,
            use_kv_lengths=use_kv_lengths,
        )
        attn_output = mha_o.reshape(*hidden_states.shape[:-1], self.q_size)
        if self.gating:
            # Nothing reads the gate columns afterwards.
            attn_output = attn_output.mul_(gate_and_bias[0].sigmoid_())
        return self.o_proj(attn_output)


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
        self._gated_sigmoid_op = None
        if gating:
            self.g_proj = Linear(
                self.q_hidden_size,
                self.q_size,
                bias=bias_flags["g"],
                dtype=dtype,
                skip_create_weights=skip_create_weights,
            )
            self._gated_sigmoid_op = get_gated_sigmoid_op(
                dtype or torch.get_default_dtype(),
                N=self.q_size,
                K=self.q_hidden_size,
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
            attn_output = self._gated_sigmoid_op(q_x, self.g_proj.weight, mha_flat, self.g_proj.bias)
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
        norm_type: str = "layer_norm",
        use_qk_norm: bool = False,
        kv_bias: bool = False,
        out_bias: bool = False,
        pair_norm_type: str = "layer_norm",
        output_gate_dim: int | None = None,
        skip_create_weights: bool = False,
        attn_backend: str = "VANILLA",
        mask_left_aligned: bool = True,
    ):
        """Self-attention with pair bias.

        Args (beyond the base Boltz/OF variants):
            norm_type: node input norm — ``"layer_norm"`` or ``"rms_norm"``.
                Also used for the QK norm.
            use_qk_norm: apply a whole-vector norm to Q and K after
                projection.
            kv_bias: add a bias to the fused K/V projection when the fused
                QKV is fully biased.
            out_bias: add a bias to the output projection.
            pair_norm_type: pair-bias norm — ``"layer_norm"`` or
                ``"rms_norm"``.
            output_gate_dim: Width of the conditioning that gates the output,
                ``sigmoid(output_projection(single_embedding)) * update``
                (the diffusion transformer's adaLN-Zero gate); ``None``
                leaves the output ungated.
            mask_left_aligned: Whether valid keys form a prefix. Set false for
                arbitrary valid-token positions.
        """
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
        self.use_qk_norm = use_qk_norm
        self.norm_type = norm_type
        self.inf = inf
        self.mask_left_aligned = mask_left_aligned

        self.num_key_value_heads = num_heads
        # This equal to 1 for self-attention
        self.num_key_value_groups = num_heads // self.num_key_value_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        self.bias_proj = bias_proj

        self.norm_s = None
        if initial_norm:
            self.norm_s = _make_norm(norm_type, c_s, eps=eps, dtype=dtype)

        self.q_norm = None
        self.k_norm = None
        if self.use_qk_norm:
            self.q_norm = _make_norm(norm_type, self.q_size, eps=eps, dtype=dtype)
            self.k_norm = _make_norm(norm_type, self.kv_size, eps=eps, dtype=dtype)

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

        # One GEMM projects q, the gate, k and v when the keys attend over the
        # query input. Keys from another input, such as a local-window gather,
        # take separate [q; g] and [k; v] GEMMs over in_proj's row blocks.
        self._in_proj_sizes = (self.q_size, self.q_size, self.kv_size, self.kv_size)
        self.in_proj = Linear(
            self.c_s,
            sum(self._in_proj_sizes),
            bias=True,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_SEGMENTS_LINEAR,
                segment_sizes=self._in_proj_sizes,
            ),
        )
        self.kv_bias = kv_bias
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
                    _make_norm(pair_norm_type, c_z, dtype=dtype, eps=eps),
                    linear_z,
                )
            else:
                self.proj_z = nn.Sequential(linear_z)
        self.proj_o = Linear(
            self.q_size,
            self.c_s,
            bias=out_bias,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )
        self.output_projection = None
        self._output_gate_op = None
        if output_gate_dim is not None:
            self.output_projection = Linear(
                output_gate_dim, self.c_s, dtype=dtype, skip_create_weights=skip_create_weights
            )
            self._output_gate_op = get_gated_sigmoid_op(
                dtype or torch.get_default_dtype(), N=self.c_s, K=output_gate_dim
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
            LNProjMoveaxisPad(
                D=c_z,
                H=self.num_heads,
                dtype=dtype or torch.bfloat16,
                rms_norm=(pair_norm_type == "rms_norm"),
                eps=eps,
            )
            if self.bias_proj
            else None
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

        **Boltz family** (no AdaLN):
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
            single_embedding: Single representation ``[B, (*), N, C_s]`` for
                AdaLN conditioning; ignored without ``use_separate_layer_norm``
                and ``use_ada_layer_norm``.
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

    def _prep_qkvg(
        self,
        s: torch.Tensor,
        kv_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project Q and the gate from *s* and K/V from *kv_in*.

        Returns:
            (q, k, v, g) — non-contiguous views of the projection output (the
            attention kernels accept strided operands). When ``use_qk_norm``
            is set, q and k are whole-vector normed, which also makes them
            contiguous.
        """
        if kv_in is s:
            q, g, k, v = self.in_proj(s).split(self._in_proj_sizes, dim=-1)
        else:
            # Slice per call: in_proj's parameters may be created or replaced after __init__.
            rows = 2 * self.q_size
            weight, bias = self.in_proj.weight, self.in_proj.bias
            q, g = F.linear(s, weight[:rows], bias[:rows]).split(self._in_proj_sizes[:2], dim=-1)
            kv_bias = bias[rows:] if self.kv_bias else None
            k, v = F.linear(kv_in, weight[rows:], kv_bias).split(self._in_proj_sizes[2:], dim=-1)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q, k, v, g

    def project_pair_bias(self, z: torch.Tensor) -> torch.Tensor:
        """Project a pair representation into this layer's attention bias.

        Args:
            z: ``[B, (*), N_q, N_k, C_z]`` pair features.

        Returns:
            ``[B, (*), H, N_q, N_k_padded]`` bias, keys padded for the backend.
        """
        ln = self.proj_z[0] if len(self.proj_z) > 1 else None
        return self._ln_proj_moveaxis_pad(
            z,
            ln_weight=ln.weight if ln is not None else None,
            ln_bias=getattr(ln, "bias", None) if ln is not None else None,
            proj_weight=self.proj_z[-1].weight,
            pad_multiple=self._bias_pad_multiple,
            proj_z=self.proj_z,
        )

    def _prep_mask_bias(
        self,
        s: torch.Tensor,
        z: torch.Tensor | None,
        mask: torch.Tensor,
        mask_bias: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Build the ``[mask_bias, pair_bias]`` list consumed by the attention kernel.

        Args:
            s: Token or atom embedding.
                - Token transformer: ``[B, N, C_s]`` or ``[B, S, N, C_s]``.
                - Atom transformer: ``[B, 1, K, N_q, C_s]`` or ``[B, S, K, N_q, C_s]``.
            z: Optional pair representation.
                - With ``bias_proj``: ``[B, (*), N_q, N_k, C_z]`` — projected
                  by ``LNProjMoveaxisPad`` to ``[B, (*), H, N_q, N_k_padded]``.
                - Without ``bias_proj``: already ``[B, (*), H, N_q, N_k]``.
                - ``None``: mask-only attention with no pair bias.
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
            ``[mask_bias, pair_bias]``, or ``[mask_bias]`` when ``z`` is
            ``None``.
        """
        sequence_mask = mask
        if mask_bias is None:
            if self.attn_backend == "CuTeDSL":
                mask_bias = mask.float()
            else:
                mask = mask[..., None, None, :]
                mask_bias = (1 - mask.float()) * -self.inf

        if z is None:
            if self.attn_backend != "CuTeDSL":
                while mask_bias.ndim < s.ndim + 1:
                    mask_bias = mask_bias.unsqueeze(1)
            return [mask_bias]

        pair_bias = self.project_pair_bias(z) if self.bias_proj else z

        if not self.mask_left_aligned:
            valid_keys = F.pad(
                sequence_mask.bool(),
                (0, pair_bias.shape[-1] - sequence_mask.shape[-1]),
                value=False,
            )
            mask_value = (
                -min(float(self.inf), torch.finfo(pair_bias.dtype).max) if self.attn_backend == "CuTeDSL" else 0.0
            )
            pair_bias = pair_bias.masked_fill(~valid_keys[..., None, None, :], mask_value)
            if self.attn_backend == "CuTeDSL":
                mask_bias = torch.full(
                    sequence_mask.shape[:-1],
                    sequence_mask.shape[-1],
                    device=sequence_mask.device,
                    dtype=torch.int32,
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
        z: torch.Tensor | None,
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
            z: Pair representation or pre-projected pair bias. ``None`` runs
                mask-only attention.
                - With ``bias_proj=True``: ``[B, (*), N_q, N_k, C_z]`` —
                  internally projected to ``[B, (*), H, N_q, N_k_padded]``.
                - With ``bias_proj=False``: ``[B, (*), H, N_q, N_k]``,
                  already projected by the caller.
            mask: Sequence-level binary mask (1 = valid, 0 = padded).
                - Token path: ``[B, N]``.
                - Atom path: ``[B, K, N_q]``.
            single_embedding: Optional single representation ``[B, (*), N, C]``
                conditioning the AdaLN (OpenFold3 diffusion transformer) and
                the output gate. Its leading dimensions match *s*'s, except
                that one of them may be 1 and broadcast, such as the
                multiplicity. Required with an output gate.
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
        if self.output_projection is not None and single_embedding is None:
            raise ValueError("the output gate needs single_embedding")
        s, kv_in, mask, mask_bias = self._prep_inputs(
            s, mask, single_embedding, attn_metadata, mask_bias, mask_bias_local
        )

        q, k, v, g = self._prep_qkvg(s, kv_in)

        biases = self._prep_mask_bias(s, z, mask, mask_bias)

        # CuTeDSL kernels are fp16/bf16 only; restore attn_in_dtype after the kernel.
        attn_in_dtype = q.dtype
        if self.attn_backend == "CuTeDSL":
            kernel_dtype = attn_in_dtype if attn_in_dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
            q = q.to(dtype=kernel_dtype)
            k = k.to(dtype=kernel_dtype)
            v = v.to(dtype=kernel_dtype)
            if len(biases) > 1:
                biases = [biases[0], biases[1].to(dtype=kernel_dtype)]

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
        if mha_o.dtype != attn_in_dtype:
            mha_o = mha_o.to(dtype=attn_in_dtype)

        # Nothing reads the attention output or the gate columns again.
        update = self.proj_o(mha_o.reshape(g.shape).mul_(g.sigmoid_()))
        if self.output_projection is not None:
            update = self._output_gate_op(
                single_embedding,
                self.output_projection.weight,
                update,
                self.output_projection.bias,
                output=update,
            )
        return update


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
            self._ln_proj_moveaxis_pad = LNProjMoveaxisPad(
                D=c_z,
                H=self.num_heads,
                dtype=dtype or torch.bfloat16,
                eps=eps,
            )
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
        m = self.layer_norm_m(m)

        output = self.mha(m, mask_bias, triangle_bias=z, attn_metadata=attn_metadata, buffers=buffers)
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
