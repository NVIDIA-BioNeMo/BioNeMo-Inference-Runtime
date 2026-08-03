# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protenix atom attention encoder / decoder (AF3 Algorithms 5 / 6)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.layers.linear import (Linear, WeightMode,
                                                   WeightsLoadingConfig)
from tensorrt_bionemo._torch.layers.sequence_local_atom import to_blocks
from tensorrt_bionemo._torch.layers.transformers.diffusion_transformer import \
    ProtenixDiffusionTransformer
from tensorrt_bionemo.configs import BaseConfig


def _broadcast_token_to_atom(x_token: torch.Tensor,
                             atom_to_token_idx: torch.Tensor) -> torch.Tensor:
    """Gather per-token features to per-atom (OSS ``broadcast_token_to_atom``)."""
    idx = atom_to_token_idx.unsqueeze(-1).expand(*atom_to_token_idx.shape,
                                                 x_token.shape[-1])
    return torch.gather(x_token, -2, idx)


def _broadcast_token_pair_to_blocks(
        z_token: torch.Tensor, atom_to_token_idx: torch.Tensor, K: int, W: int,
        attn_metadata: AttentionMetadata) -> torch.Tensor:
    """Broadcast token-pair ``z`` to windowed atom-pair blocks ``[B, K, W, H, d]``.

    Query/key token indices use the same windowing as atom features (``to_blocks``
    + ``query_to_keys``). Out-of-range maps to token 0 (masked downstream).
    """
    B = z_token.shape[0]
    # Feed *fp32* token indices into query_to_keys: bf16 cannot represent
    # integer indices exactly above 256, so for N_token > 256 they round and
    # produce out-of-bounds gathers.
    q_blocked = to_blocks(
        atom_to_token_idx.unsqueeze(-1).to(torch.float32), K,
        W)  # [B, K, W, 1]
    q_tok = q_blocked.squeeze(-1).long()  # [B, K, W]
    k_tok = attn_metadata.query_to_keys(q_blocked).squeeze(
        -1).long()  # [B,K,H]
    batch_idx = torch.arange(B, device=z_token.device).view(B, 1, 1, 1)
    return z_token[
        batch_idx,
        q_tok.unsqueeze(-1),  # [B, K, W, 1]
        k_tok.unsqueeze(-2)]  # [B, K, 1, H]


def _expand_bs(x: torch.Tensor, B: int, S: int) -> torch.Tensor:
    """Expand ``[B, ...]`` over samples and fold to ``[B*S, ...]``."""
    return x.unsqueeze(1).expand(B, S,
                                 *x.shape[1:]).reshape(B * S, *x.shape[1:])


def _fold_bs(x: torch.Tensor, B: int, S: int) -> torch.Tensor:
    """Fold leading ``[B, S, ...]`` into ``[B*S, ...]``."""
    return x.reshape(B * S, *x.shape[2:])


class ProtenixAtomAttentionEncoder(nn.Module):
    """Protenix AF3 Algorithm 5 atom attention encoder.

    Reference-conformer embedding + shared :class:`ProtenixDiffusionTransformer`.
    With ``has_coords``, also conditions on trunk ``s`` / ``z`` and noisy
    coords ``r_l`` (``N_sample`` folded into batch). ``prepare_coords_cache`` /
    ``run_coords_cached`` split the sample-invariant base from the per-step
    coord path.
    """

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        dtype = config.torch_dtype
        skip = config.skip_create_weights
        c_atom = config.c_atom
        c_atompair = config.c_atompair
        c_token = config.c_token
        self.has_coords = config.has_coords
        self.dtype = dtype
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.n_queries = config.n_queries
        self.n_keys = config.n_keys

        self.input_feature = {
            "ref_mask": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 4 * 64
        }

        self.linear_no_bias_ref = Linear(
            3 + 1 + sum(self.input_feature.values()),
            c_atom,
            bias=False,
            dtype=dtype,
            skip_create_weights=skip,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_ALL_LINEAR_LAST_DIM))
        self.linear_no_bias_pair = Linear(
            3 + 1 + 1,
            c_atompair,
            bias=False,
            dtype=dtype,
            skip_create_weights=skip,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_ALL_LINEAR_LAST_DIM))
        self.linear_no_bias_cl = Linear(c_atom,
                                        c_atompair,
                                        bias=False,
                                        dtype=dtype,
                                        skip_create_weights=skip)
        self.linear_no_bias_cm = Linear(c_atom,
                                        c_atompair,
                                        bias=False,
                                        dtype=dtype,
                                        skip_create_weights=skip)
        if self.has_coords:
            # Diffusion conditioning (OSS create_offset=False -> scale-only LN;
            # s/z projections are zero-init, r is the noisy-coord projection).
            eps = config.norm_epsilon
            self.layernorm_s = nn.LayerNorm(config.c_s,
                                            bias=False,
                                            eps=eps,
                                            dtype=dtype)
            self.linear_no_bias_s = Linear(config.c_s,
                                           c_atom,
                                           bias=False,
                                           dtype=dtype,
                                           skip_create_weights=skip)
            self.layernorm_z = nn.LayerNorm(config.c_z,
                                            bias=False,
                                            eps=eps,
                                            dtype=dtype)
            self.linear_no_bias_z = Linear(config.c_z,
                                           c_atompair,
                                           bias=False,
                                           dtype=dtype,
                                           skip_create_weights=skip)
            self.linear_no_bias_r = Linear(3,
                                           c_atom,
                                           bias=False,
                                           dtype=dtype,
                                           skip_create_weights=skip)
        self.small_mlp = nn.Sequential(
            nn.ReLU(),
            Linear(c_atompair,
                   c_atompair,
                   bias=False,
                   dtype=dtype,
                   skip_create_weights=skip),
            nn.ReLU(),
            Linear(c_atompair,
                   c_atompair,
                   bias=False,
                   dtype=dtype,
                   skip_create_weights=skip),
            nn.ReLU(),
            Linear(c_atompair,
                   c_atompair,
                   bias=False,
                   dtype=dtype,
                   skip_create_weights=skip),
        )
        # ``model_copy`` skips re-validation, which would reject int fields
        # that are left as ``None`` in the source config.
        atc = config.atom_transformer_config.model_copy(
            update={
                "dtype": config.dtype,
                "skip_create_weights": skip
            })
        self.atom_transformer = ProtenixDiffusionTransformer(atc)
        self.linear_no_bias_q = Linear(c_atom,
                                       c_token,
                                       bias=False,
                                       dtype=dtype,
                                       skip_create_weights=skip)

    def prepare_cache(
        self,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_element: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
        d_lm: torch.Tensor,
        v_lm: torch.Tensor,
        pad_info: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_shape = ref_pos.shape[:-2]
        n_atom = ref_pos.shape[-2]
        ref_features = torch.cat(
            [
                ref_pos,
                torch.arcsinh(ref_charge).reshape(*batch_shape, n_atom, 1),
                ref_mask.reshape(*batch_shape, n_atom, 1),
                ref_element.reshape(*batch_shape, n_atom, 128),
                ref_atom_name_chars.reshape(*batch_shape, n_atom, 4 * 64),
            ],
            dim=-1,
        ).to(self.dtype)
        c_l = self.linear_no_bias_ref(ref_features)
        c_l = c_l * ref_mask.reshape(*batch_shape, n_atom, 1)

        v_lm_f = v_lm.to(dtype=c_l.dtype)
        masked_v = v_lm_f * pad_info["mask_trunked"].unsqueeze(-1)
        pair_features = torch.cat([
            d_lm * masked_v,
            v_lm_f / (1 + (d_lm**2).sum(dim=-1, keepdim=True)),
            v_lm_f,
        ],
                                  dim=-1)
        p_lm = self.linear_no_bias_pair(pair_features)
        return p_lm, c_l

    def forward(
        self,
        atom_to_token_idx: torch.Tensor,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        ref_element: torch.Tensor,
        d_lm: torch.Tensor,
        v_lm: torch.Tensor,
        pad_info: dict,
        r_l: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed reference-conformer atom features → ``(a, q_l, c_l, p_lm)``.

        Args:
            atom_to_token_idx: ``[B, N_atom]``
            ref_pos: ``[B, N_atom, 3]``
            ref_charge / ref_mask: ``[B, N_atom]``
            ref_atom_name_chars: ``[B, N_atom, 4, 64]`` (or flat ``4*64``)
            ref_element: ``[B, N_atom, 128]``
            d_lm: ``[B, K, W, H, 3]`` windowed ref displacements
            v_lm: ``[B, K, W, H, 1]`` valid-pair mask in windows
            pad_info: dict with ``mask_trunked`` ``[B, K, W, H]``
            r_l: ``[B, S, N_atom, 3]`` noisy coords when ``has_coords``
            s: ``[B, S, N_token, c_s]`` trunk single when ``has_coords``
            z: ``[B, S, N_token, N_token, c_z]`` pair when ``has_coords``

        Returns:
            Without coords: ``a`` ``[B, N_token, c_token]``,
            ``q_l`` / ``c_l`` ``[B, N_atom, c_atom]``,
            ``p_lm`` ``[B, K, W, H, c_atompair]``.
            With ``has_coords``: same ranks with leading ``[B, S, ...]``.
        """
        n_atom = ref_pos.shape[-2]

        d = self.dtype
        ref_pos, ref_charge, ref_mask = ref_pos.to(d), ref_charge.to(
            d), ref_mask.to(d)
        ref_element = ref_element.to(d)
        ref_atom_name_chars = ref_atom_name_chars.to(d)
        d_lm = d_lm.to(d)
        if r_l is not None:
            r_l = r_l.to(d)
        if s is not None:
            s = s.to(d)
        if z is not None:
            z = z.to(d)

        p_lm, c_l = self.prepare_cache(ref_pos, ref_charge, ref_mask,
                                       ref_element, ref_atom_name_chars,
                                       atom_to_token_idx, d_lm, v_lm, pad_info)

        K = p_lm.shape[1]
        if attn_metadata is None:
            attn_metadata = self.atom_transformer.build_attn_metadata(
                K, self.n_queries, self.n_keys, c_l.device)

        if self.has_coords:
            return self._forward_coords(atom_to_token_idx, p_lm, c_l, r_l, s,
                                        z, K, attn_metadata)

        q_l = c_l
        n_token = int(atom_to_token_idx.max().item()) + 1
        c_l_q = to_blocks(c_l, K, self.n_queries)
        c_l_k = attn_metadata.query_to_keys(c_l_q)
        p_lm = p_lm + self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
        p_lm = p_lm + self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
        p_lm = p_lm + self.small_mlp(p_lm)

        atom_mask = ref_mask.new_ones(*ref_mask.shape[:-1], n_atom)
        q_l = self.atom_transformer(q_l, c_l, p_lm, atom_mask, self.n_queries,
                                    self.n_keys, attn_metadata)

        a = self._aggregate_atom_to_token(F.relu(self.linear_no_bias_q(q_l)),
                                          atom_to_token_idx, n_token)
        return a, q_l, c_l, p_lm

    def _prepare_coords(
        self,
        atom_to_token_idx: torch.Tensor,
        p_lm: torch.Tensor,
        c_l: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        K: int,
        attn_metadata: AttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step-invariant diffusion conditioning of the reference-conformer base.

        Broadcasts trunk ``s`` / ``z`` over ``S`` (folded into batch). Cacheable
        across the sampling loop (independent of noisy coords ``r_l``).
        """
        W = self.n_queries
        B, S = s.shape[0], s.shape[1]
        a2t = _expand_bs(atom_to_token_idx, B, S)

        z_proj = _fold_bs(self.linear_no_bias_z(self.layernorm_z(z)), B, S)
        z_block = _broadcast_token_pair_to_blocks(z_proj, a2t, K, W,
                                                  attn_metadata)
        p_lm = _expand_bs(p_lm, B, S) + z_block

        s_proj = _fold_bs(self.linear_no_bias_s(self.layernorm_s(s)), B, S)
        c_l = _expand_bs(c_l, B, S)
        c_l = c_l + _broadcast_token_to_atom(s_proj, a2t)

        c_l_q = to_blocks(c_l, K, W)
        c_l_k = attn_metadata.query_to_keys(c_l_q)
        p_lm = p_lm + self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
        p_lm = p_lm + self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
        p_lm = p_lm + self.small_mlp(p_lm)
        return c_l, p_lm, a2t

    def _run_coords(
        self,
        a2t: torch.Tensor,
        c_l: torch.Tensor,
        p_lm: torch.Tensor,
        r_l: torch.Tensor,
        n_token: int,
        attn_metadata: AttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Noisy-coord-dependent path on pre-conditioned ``c_l`` / ``p_lm``."""
        W, H = self.n_queries, self.n_keys
        BS = c_l.shape[0]
        q_l = c_l + self.linear_no_bias_r(r_l.reshape(BS, *r_l.shape[2:]))
        atom_mask = c_l.new_ones(BS, c_l.shape[-2])
        q_l = self.atom_transformer(q_l, c_l, p_lm, atom_mask, W, H,
                                    attn_metadata)
        a = self._aggregate_atom_to_token(F.relu(self.linear_no_bias_q(q_l)),
                                          a2t, n_token)
        return a, q_l

    def _forward_coords(
        self,
        atom_to_token_idx: torch.Tensor,
        p_lm: torch.Tensor,
        c_l: torch.Tensor,
        r_l: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        K: int,
        attn_metadata: AttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Coordinate-conditioned (``has_coords``) encoder path -> ``[B, S, ...]``."""
        B, S, n_token = s.shape[0], s.shape[1], s.shape[-2]
        c_l, p_lm, a2t = self._prepare_coords(atom_to_token_idx, p_lm, c_l, s,
                                              z, K, attn_metadata)
        a, q_l = self._run_coords(a2t, c_l, p_lm, r_l, n_token, attn_metadata)
        return (a.reshape(B, S, n_token,
                          -1), q_l.reshape(B, S, q_l.shape[-2], -1),
                c_l.reshape(B, S, c_l.shape[-2],
                            -1), p_lm.reshape(B, S, *p_lm.shape[1:]))

    def prepare_coords_cache(
        self,
        atom_to_token_idx: torch.Tensor,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        ref_element: torch.Tensor,
        d_lm: torch.Tensor,
        v_lm: torch.Tensor,
        pad_info: dict,
        s: torch.Tensor,
        z: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, AttentionMetadata]:
        """Sample-independent ``c_l`` / ``p_lm`` for the cached diffusion path.

        Folds with ``S=1`` so returned tensors hold one copy; ``run_coords_cached``
        broadcasts over samples.

        Args:
            s: ``[B, N_token, c_s]`` (unsqueezed to ``S=1`` internally)
            z: ``[B, N_token, N_token, c_z]``
            Other ref / window args: same shapes as :meth:`forward`

        Returns:
            ``c_l`` ``[B, N_atom, c_atom]``,
            ``p_lm`` ``[B, K, W, H, c_atompair]``,
            ``attn_metadata``
        """
        d = self.dtype
        ref_pos, ref_charge, ref_mask = ref_pos.to(d), ref_charge.to(
            d), ref_mask.to(d)
        ref_element = ref_element.to(d)
        ref_atom_name_chars = ref_atom_name_chars.to(d)
        d_lm = d_lm.to(d)
        s, z = s.to(d).unsqueeze(1), z.to(d).unsqueeze(1)
        p_lm, c_l = self.prepare_cache(ref_pos, ref_charge, ref_mask,
                                       ref_element, ref_atom_name_chars,
                                       atom_to_token_idx, d_lm, v_lm, pad_info)
        K = p_lm.shape[1]
        if attn_metadata is None:
            attn_metadata = self.atom_transformer.build_attn_metadata(
                K, self.n_queries, self.n_keys, c_l.device)
        c_l, p_lm, _ = self._prepare_coords(atom_to_token_idx, p_lm, c_l, s, z,
                                            K, attn_metadata)
        return c_l, p_lm, attn_metadata

    def run_coords_cached(
        self,
        atom_to_token_idx: torch.Tensor,
        c_l: torch.Tensor,
        p_lm: torch.Tensor,
        r_l: torch.Tensor,
        n_token: int,
        attn_metadata: AttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-step coord path from cached sample-independent ``c_l`` / ``p_lm``.

        Args:
            atom_to_token_idx: ``[B, N_atom]``
            c_l: ``[B, N_atom, c_atom]`` from :meth:`prepare_coords_cache`
            p_lm: ``[B, K, W, H, c_atompair]``
            r_l: ``[B, S, N_atom, 3]``
            n_token: token count

        Returns:
            ``a`` ``[B, S, N_token, c_token]``,
            ``q_l`` / ``c_l`` ``[B, S, N_atom, c_atom]``,
            ``p_lm`` ``[B, S, K, W, H, c_atompair]``
        """
        B, S = r_l.shape[0], r_l.shape[1]
        r_l = r_l.to(self.dtype)
        c_l = _expand_bs(c_l, B, S)
        p_lm = _expand_bs(p_lm, B, S)
        a2t = _expand_bs(atom_to_token_idx, B, S)
        a, q_l = self._run_coords(a2t, c_l, p_lm, r_l, n_token, attn_metadata)
        return (a.reshape(B, S, n_token,
                          -1), q_l.reshape(B, S, q_l.shape[-2], -1),
                c_l.reshape(B, S, c_l.shape[-2],
                            -1), p_lm.reshape(B, S, *p_lm.shape[1:]))

    @staticmethod
    def _aggregate_atom_to_token(x_atom: torch.Tensor,
                                 atom_to_token_idx: torch.Tensor,
                                 n_token: int) -> torch.Tensor:
        idx = atom_to_token_idx.unsqueeze(-1).expand_as(x_atom)
        out = x_atom.new_zeros(*x_atom.shape[:-2], n_token, x_atom.shape[-1])
        out.scatter_add_(-2, idx, x_atom)
        counts = x_atom.new_zeros(*x_atom.shape[:-2], n_token, 1)
        ones = x_atom.new_ones(*x_atom.shape[:-1], 1)
        counts.scatter_add_(-2, atom_to_token_idx.unsqueeze(-1), ones)
        return out / counts.clamp(min=1)


class ProtenixAtomAttentionDecoder(nn.Module):
    """Protenix AF3 Algorithm 6 atom attention decoder.

    Broadcasts token single ``a`` back to atoms, adds encoder skip ``q_skip``,
    runs the local-windowed atom DiT, projects to a per-atom coordinate update.
    Caller folds ``N_sample`` into ``B``.
    """

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        dtype = config.torch_dtype
        skip = config.skip_create_weights
        c_atom = config.c_atom
        self.dtype = dtype
        self.c_atom = c_atom
        self.n_queries = config.n_queries
        self.n_keys = config.n_keys

        self.linear_no_bias_a = Linear(config.c_token,
                                       c_atom,
                                       bias=False,
                                       dtype=dtype,
                                       skip_create_weights=skip)
        atc = config.atom_transformer_config.model_copy(
            update={
                "dtype": config.dtype,
                "skip_create_weights": skip
            })
        self.atom_transformer = ProtenixDiffusionTransformer(atc)
        # OSS layernorm_q is create_offset=False -> scale-only.
        self.layernorm_q = nn.LayerNorm(c_atom,
                                        bias=False,
                                        eps=config.norm_epsilon,
                                        dtype=dtype)
        self.linear_no_bias_out = Linear(c_atom,
                                         3,
                                         bias=False,
                                         dtype=dtype,
                                         skip_create_weights=skip)

    def forward(
        self,
        atom_to_token_idx: torch.Tensor,
        a: torch.Tensor,
        q_skip: torch.Tensor,
        c_skip: torch.Tensor,
        p_skip: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        """Decode per-token features to a per-atom coordinate update.

        Args:
            atom_to_token_idx: ``[B, N_atom]`` (``N_sample`` already folded into ``B``)
            a: ``[B, N_token, c_token]`` token single after DiT
            q_skip: ``[B, N_atom, c_atom]`` encoder skip
            c_skip: ``[B, N_atom, c_atom]``
            p_skip: ``[B, K, W, H, c_atompair]``

        Returns:
            ``[B, N_atom, 3]`` per-atom coordinate update
        """
        d = self.dtype
        a, q_skip, c_skip = a.to(d), q_skip.to(d), c_skip.to(d)
        p_skip = p_skip.to(d)
        q = _broadcast_token_to_atom(self.linear_no_bias_a(a),
                                     atom_to_token_idx) + q_skip

        K = p_skip.shape[1]
        if attn_metadata is None:
            attn_metadata = self.atom_transformer.build_attn_metadata(
                K, self.n_queries, self.n_keys, q.device)
        atom_mask = q.new_ones(*q.shape[:-1])
        q = self.atom_transformer(q, c_skip, p_skip, atom_mask, self.n_queries,
                                  self.n_keys, attn_metadata)

        return self.linear_no_bias_out(self.layernorm_q(q))
