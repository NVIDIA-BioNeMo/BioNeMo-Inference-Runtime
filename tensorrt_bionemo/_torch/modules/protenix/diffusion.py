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
"""Protenix diffusion conditioning (AF3 Algorithm 21) and EDM sampler."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.auto_chunk import (CHUNK_REGISTRY,
                                                DIFFUSION_PAIR_TRANSITION)
from tensorrt_bionemo._torch.layers.linear import Linear
from tensorrt_bionemo._torch.layers.noise_scheduler import (
    SampleDiffusion, create_noise_schedule)
from tensorrt_bionemo._torch.layers.position_encoders import (
    FourierEmbedding, RelativePositionEncoder)
from tensorrt_bionemo._torch.layers.transformers.diffusion_transformer import \
    ProtenixDiffusionTransformer
from tensorrt_bionemo._torch.layers.transition import Transition
from tensorrt_bionemo._torch.modules.protenix._common import (
    DIFFUSION_CONSUMED_FEATURES, atom_encoder_kwargs)
from tensorrt_bionemo._torch.modules.protenix.atom_attention import (
    ProtenixAtomAttentionDecoder, ProtenixAtomAttentionEncoder)
from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.utils import str_dtype_to_torch


class ProtenixDiffusionConditioning(nn.Module):
    """AF3 Algorithm 21 diffusion conditioning (Protenix variant).

    Pair path fuses ``z_trunk`` with a fresh RPE (own ``relpe`` weights) via
    fp32 projection + two SwiGLU transitions, then casts to ``z_pair_dtype``.
    Single path fuses ``s_trunk`` / ``s_inputs`` with a Fourier noise embedding.
    """

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        dtype = config.torch_dtype
        skip = config.skip_create_weights
        mapping = config.mapping
        eps = config.norm_epsilon
        c_s = config.c_s
        c_z = config.c_z
        c_s_inputs = config.c_s_inputs
        c_noise = config.c_noise_embedding
        self.sigma_data = config.sigma_data
        self.z_pair_dtype = str_dtype_to_torch(config.z_pair_dtype)

        # Pair path: shared RPE (relpe) has its own weights, distinct from the
        # trunk's top-level RPE. Protenix options: fix_sym_check / no cyclic.
        rc = config.relpe_config
        self.relpe = RelativePositionEncoder(token_z=rc.c_z,
                                             r_max=rc.r_max,
                                             s_max=rc.s_max,
                                             fix_sym_check=rc.fix_sym_check,
                                             cyclic_pos_enc=rc.cyclic_pos_enc,
                                             dtype=dtype,
                                             mapping=mapping,
                                             skip_create_weights=skip)
        self.layernorm_z = nn.LayerNorm(2 * c_z,
                                        bias=False,
                                        eps=eps,
                                        dtype=dtype)
        self.linear_no_bias_z = Linear(2 * c_z,
                                       c_z,
                                       bias=False,
                                       dtype=dtype,
                                       skip_create_weights=skip)
        self.transition_z = nn.ModuleList([
            Transition(dim=c_z,
                       hidden=2 * c_z,
                       eps=eps,
                       dtype=self.z_pair_dtype,
                       mapping=mapping,
                       skip_create_weights=skip,
                       auto_chunk_policy=CHUNK_REGISTRY.get(
                           DIFFUSION_PAIR_TRANSITION)) for _ in range(2)
        ])

        # Single path.
        self.layernorm_s = nn.LayerNorm(c_s + c_s_inputs,
                                        bias=False,
                                        eps=eps,
                                        dtype=dtype)
        self.linear_no_bias_s = Linear(c_s + c_s_inputs,
                                       c_s,
                                       bias=False,
                                       dtype=dtype,
                                       skip_create_weights=skip)
        self.fourier_embedding = FourierEmbedding(c_noise,
                                                  dtype=dtype,
                                                  mapping=mapping)
        self.layernorm_n = nn.LayerNorm(c_noise,
                                        bias=False,
                                        eps=eps,
                                        dtype=dtype)
        self.linear_no_bias_n = Linear(c_noise,
                                       c_s,
                                       bias=False,
                                       dtype=dtype,
                                       skip_create_weights=skip)
        self.transition_s = nn.ModuleList([
            Transition(dim=c_s,
                       hidden=2 * c_s,
                       eps=eps,
                       dtype=dtype,
                       mapping=mapping,
                       skip_create_weights=skip) for _ in range(2)
        ])

    def forward(
        self,
        t_hat_noise_level: torch.Tensor,
        relp: torch.Tensor,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Condition single/pair trunk embeddings on noise (AF3 Alg. 21).

        Args:
            t_hat_noise_level: ``[B, S]`` per-sample noise levels
            relp: relative-position features for ``relpe``
            s_inputs: ``[B, N_token, c_s_inputs]``
            s_trunk: ``[B, N_token, c_s]``
            z_trunk: ``[B, N_token, N_token, c_z]``

        Returns:
            ``s`` ``[B, S, N_token, c_s]``, ``z`` ``[B, N_token, N_token, c_z]``
        """
        pair_z = self.prepare_pair(relp, z_trunk)
        single_s = self.forward_single(t_hat_noise_level, s_inputs, s_trunk)
        return single_s, pair_z

    def _joint_layernorm_linear_z(self, z, relpe):
        """Fused LN+Linear over concat(z, relpe) without materializing the concat."""
        if z.shape != relpe.shape:
            raise ValueError(f"shape mismatch: {z.shape} vs {relpe.shape}")
        var_z, mean_z = torch.var_mean(z, dim=-1, correction=0, keepdim=True)
        var_r, mean_r = torch.var_mean(relpe,
                                       dim=-1,
                                       correction=0,
                                       keepdim=True)

        delta = mean_z - mean_r
        mean = (mean_z + mean_r) * 0.5
        rstd = var_z.add_(var_r).mul_(0.5)
        rstd.addcmul_(delta, delta, value=0.25)
        rstd.add_(self.layernorm_z.eps).rsqrt_()

        linear = self.linear_no_bias_z
        weight = linear.weight * self.layernorm_z.weight.unsqueeze(0)
        weight_z, weight_r = weight.chunk(2, dim=-1)

        output = linear.apply_linear(z, weight_z, None)
        output.add_(linear.apply_linear(relpe, weight_r, None))
        output.addcmul_(mean, weight.sum(-1), value=-1.0)
        output.mul_(rstd)
        return output

    def prepare_pair(self, relp: torch.Tensor,
                     z_trunk: torch.Tensor) -> torch.Tensor:
        """Pair conditioning (noise-independent; cacheable across the EDM loop).

        Args:
            relp: relative-position features for ``relpe``
            z_trunk: ``[B, N_token, N_token, c_z]``

        Returns:
            ``[B, N_token, N_token, c_z]`` conditioned pair (``z_pair_dtype``)
        """
        relpe_z = self.relpe(relp=relp)
        pair_z = self._joint_layernorm_linear_z(z_trunk,
                                                relpe_z).to(self.z_pair_dtype)
        for layer in self.transition_z:
            pair_z = pair_z + layer(pair_z)
        return pair_z

    def forward_single(self, t_hat_noise_level: torch.Tensor,
                       s_inputs: torch.Tensor,
                       s_trunk: torch.Tensor) -> torch.Tensor:
        """Single conditioning (noise-dependent; recomputed each diffusion step).

        Args:
            t_hat_noise_level: ``[B, S]``
            s_inputs: ``[B, N_token, c_s_inputs]``
            s_trunk: ``[B, N_token, c_s]``

        Returns:
            ``[B, S, N_token, c_s]``
        """
        single_s = torch.cat([s_trunk, s_inputs], dim=-1)
        single_s = self.linear_no_bias_s(self.layernorm_s(single_s))
        noise_n = self.fourier_embedding(
            torch.log(t_hat_noise_level / self.sigma_data) / 4).to(
                single_s.dtype)
        single_s = single_s.unsqueeze(-3) + self.linear_no_bias_n(
            self.layernorm_n(noise_n)).unsqueeze(-2)
        for layer in self.transition_s:
            single_s = single_s + layer(single_s)
        return single_s


class ProtenixDiffusionModule(nn.Module):
    """AF3 Algorithm 20 diffusion module (Protenix): one EDM denoise step.

    Atom decoder stays fp32 for e2e accuracy; token transformer may run bf16.
    ``N_sample`` is folded into batch for the token transformer / decoder.
    """

    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        dtype = config.torch_dtype
        skip = config.skip_create_weights
        eps = config.norm_epsilon
        c_s, c_token = config.c_s, config.c_token
        self.dtype = dtype
        self.sigma_data = config.sigma_data
        self.n_queries = config.n_queries
        self.n_keys = config.n_keys

        self.diffusion_conditioning = ProtenixDiffusionConditioning(
            config.diffusion_conditioning_config)
        self.atom_attention_encoder = ProtenixAtomAttentionEncoder(
            config.atom_encoder_config)
        # Alg. 20 line 4: LayerNorm(c_s, create_offset=False) -> scale-only.
        self.layernorm_s = nn.LayerNorm(c_s, bias=False, eps=eps, dtype=dtype)
        self.linear_no_bias_s = Linear(c_s,
                                       c_token,
                                       bias=False,
                                       dtype=dtype,
                                       skip_create_weights=skip)
        # Token transformer keeps its own dtype (bf16 + auto pairwise backend
        # while the rest stays fp32); only mapping / skip are propagated.
        ttc = config.token_transformer_config.model_copy(
            update={
                "mapping": config.mapping,
                "skip_create_weights": skip
            })
        self.diffusion_transformer = ProtenixDiffusionTransformer(ttc)
        self._token_dtype = ttc.torch_dtype
        self.layernorm_a = nn.LayerNorm(c_token,
                                        bias=False,
                                        eps=eps,
                                        dtype=dtype)
        self.atom_attention_decoder = ProtenixAtomAttentionDecoder(
            config.atom_decoder_config)

    def prepare_cache(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> dict[str, Any]:
        """Precompute step-invariant shared vars (pair_z + atom encoder base).

        Args:
            s_inputs: ``[B, N_token, c_s_inputs]``
            s_trunk: ``[B, N_token, c_s]``
            z_trunk: ``[B, N_token, N_token, c_z]``

        Returns:
            Cache dict with ``pair_z`` ``[B, N_token, N_token, c_z]``,
            ``atom_c_l`` ``[B, N_atom, c_atom]``,
            ``atom_p_lm`` ``[B, K, W, H, c_atompair]``,
            ``attn_metadata``, ``n_token``.
        """
        pair_z = self.diffusion_conditioning.prepare_pair(
            input_feature_dict["relp"], z_trunk)
        atom_c_l, atom_p_lm, attn_metadata = \
            self.atom_attention_encoder.prepare_coords_cache(
                **atom_encoder_kwargs(input_feature_dict),
                s=s_trunk,
                z=pair_z,
                attn_metadata=attn_metadata)
        return {
            "pair_z": pair_z,
            "atom_c_l": atom_c_l,
            "atom_p_lm": atom_p_lm,
            "attn_metadata": attn_metadata,
            "n_token": s_trunk.shape[-2],
        }

    def f_forward(
        self,
        r_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
        cache: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Raw denoiser ``F_theta``: scaled noisy coords → coordinate update.

        Args:
            r_noisy: ``[B, S, N_atom, 3]`` EDM-scaled noisy coordinates
            t_hat_noise_level: ``[B, S]``
            s_inputs: ``[B, N_token, c_s_inputs]``
            s_trunk: ``[B, N_token, c_s]``
            z_trunk: ``[B, N_token, N_token, c_z]`` (unused when ``cache`` set)
            cache: optional :meth:`prepare_cache` dict

        Returns:
            ``[B, S, N_atom, 3]`` coordinate update ``r_update``
        """
        B, S = r_noisy.shape[0], r_noisy.shape[1]
        n_atom = r_noisy.shape[-2]

        if cache is not None:
            attn_metadata = cache["attn_metadata"]
            n_token = cache["n_token"]
            z_pair = cache["pair_z"]
            s_single = self.diffusion_conditioning.forward_single(
                t_hat_noise_level, s_inputs, s_trunk)
            a_token, q_skip, c_skip, p_skip = \
                self.atom_attention_encoder.run_coords_cached(
                    input_feature_dict["atom_to_token_idx"], cache["atom_c_l"],
                    cache["atom_p_lm"], r_noisy, n_token, attn_metadata)
        else:
            if attn_metadata is None:
                K = (n_atom + self.n_queries - 1) // self.n_queries
                attn_metadata = self.atom_attention_encoder.atom_transformer.\
                    build_attn_metadata(K, self.n_queries, self.n_keys,
                                        r_noisy.device)

            s_single, z_pair = self.diffusion_conditioning(
                t_hat_noise_level, input_feature_dict["relp"], s_inputs,
                s_trunk, z_trunk)

            s_trunk_s = s_trunk.unsqueeze(1).expand(B, S, *s_trunk.shape[1:])
            z_pair_s = z_pair.unsqueeze(1).expand(B, S, *z_pair.shape[1:])
            a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(
                **atom_encoder_kwargs(input_feature_dict),
                r_l=r_noisy,
                s=s_trunk_s,
                z=z_pair_s,
                attn_metadata=attn_metadata)
            n_token = a_token.shape[-2]

        # Alg. 20 line 4: add the conditioned single, then the token transformer.
        a_token = a_token.to(self.dtype)
        a_token = a_token + self.linear_no_bias_s(self.layernorm_s(s_single))

        # Flatten B*S for the token path (numerically important) but keep the
        # sample-independent pair at [B, N, N, c_z] — avoids S copies of z_pair
        # and per-layer pair-bias projection. Default inference has B=1 so the
        # [1, H, N, N] bias broadcasts; CuTeDSL infers mult=S from Q's batch.
        BS = B * S
        a_bs = a_token.reshape(BS, n_token, -1).to(self._token_dtype)
        s_bs = s_single.reshape(BS, n_token, -1).to(self._token_dtype)
        z_token = z_pair.to(self._token_dtype)
        token_mask = a_bs.new_ones(B, n_token)
        a_bs = self.diffusion_transformer(a_bs, s_bs, z_token, token_mask)
        a_bs = self.layernorm_a(a_bs.to(self.dtype))

        a2t_bs = input_feature_dict["atom_to_token_idx"].unsqueeze(1).expand(
            B, S, -1).reshape(BS, -1)
        r_update = self.atom_attention_decoder(
            a2t_bs,
            a_bs,
            q_skip.reshape(BS, q_skip.shape[-2], -1),
            c_skip.reshape(BS, c_skip.shape[-2], -1),
            p_skip.reshape(BS, *p_skip.shape[2:]),
            attn_metadata=attn_metadata)
        return r_update.reshape(B, S, n_atom, 3)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
        cache: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """One EDM denoise step: ``x_noisy``, noise level → ``x_denoised``.

        Args:
            x_noisy: ``[B, S, N_atom, 3]``
            t_hat_noise_level: ``[B, S]``
            s_inputs: ``[B, N_token, c_s_inputs]``
            s_trunk: ``[B, N_token, c_s]``
            z_trunk: ``[B, N_token, N_token, c_z]``
            cache: optional :meth:`prepare_cache` dict

        Returns:
            ``[B, S, N_atom, 3]`` denoised coordinates
        """
        r_noisy = x_noisy / torch.sqrt(self.sigma_data**2 +
                                       t_hat_noise_level**2)[..., None, None]
        r_update = self.f_forward(r_noisy, t_hat_noise_level,
                                  input_feature_dict, s_inputs, s_trunk,
                                  z_trunk, attn_metadata, cache)
        s_ratio = (t_hat_noise_level / self.sigma_data)[..., None, None].to(
            r_update.dtype)
        x_denoised = (1 / (1 + s_ratio**2) * x_noisy +
                      t_hat_noise_level[..., None, None] /
                      torch.sqrt(1 + s_ratio**2) * r_update).to(r_update.dtype)
        return x_denoised


class ProtenixSampleDiffusion(SampleDiffusion):
    """Protenix EDM diffusion sampler (AF3 Algorithm 18)."""

    def __init__(self,
                 diffusion_module: nn.Module,
                 *,
                 gamma0: float = 0.8,
                 gamma_min: float = 1.0,
                 noise_scale: float = 1.003,
                 step_scale: float = 1.5,
                 s_max: float = 160.0,
                 s_min: float = 4e-4,
                 rho: float = 7.0,
                 n_step: int = 200,
                 use_cache: bool = False) -> None:
        super().__init__(gamma0=gamma0,
                         gamma_min=gamma_min,
                         noise_scale=noise_scale,
                         step_scale=step_scale)
        self.diffusion_module = diffusion_module
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho
        self.n_step = n_step
        # OSS enable_diffusion_shared_vars_cache: precompute step-invariant
        # conditioning / reference base once per rollout.
        self.use_cache = use_cache

    def denoise(self, x_noisy: torch.Tensor, sigma_hat: torch.Tensor,
                ctx: dict[str, Any]) -> torch.Tensor:
        batch_shape, n_sample = ctx["batch_shape"], ctx["n_sample"]
        t_hat = sigma_hat.reshape(
            (1, ) * (len(batch_shape) + 1)).expand(*batch_shape,
                                                   n_sample).to(x_noisy.dtype)
        return self.diffusion_module(
            x_noisy=x_noisy,
            t_hat_noise_level=t_hat,
            input_feature_dict=ctx["input_feature_dict"],
            s_inputs=ctx["s_inputs"],
            s_trunk=ctx["s_trunk"],
            z_trunk=ctx["z_trunk"],
            attn_metadata=ctx["attn_metadata"],
            cache=ctx.get("cache"))

    def noise_schedule(self,
                       n_step: Optional[int] = None,
                       device: torch.device = torch.device("cpu"),
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """AF3 inference noise schedule (``[n_step + 1]``, final step -> 0)."""
        return create_noise_schedule(
            num_points=(n_step or self.n_step) + 1,
            sigma_data=self.diffusion_module.sigma_data,
            s_max=self.s_max,
            s_min=self.s_min,
            rho=self.rho,
            device=device,
            dtype=dtype,
            final="zero")

    def sample_coords(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        *,
        num_sampling_steps: Optional[int] = None,
        N_sample: int = 1,
        attn_metadata: Optional[AttentionMetadata] = None,
        atom_mask: Optional[torch.Tensor] = None,
        drop_consumed_relp: bool = False,
        drop_consumed_features: bool = False,
    ) -> torch.Tensor:
        """AF3 Alg. 18: build noise schedule and roll out ``SampleDiffusion.sample``.

        ``drop_consumed_relp`` / ``drop_consumed_features`` free ~9 GB ``relp``
        (and ref/window tensors) once pair conditioning / prepare_cache no
        longer need them — only safe when the caller owns the feature dict.

        Args:
            s_inputs: ``[*batch, N_token, c_s_inputs]``
            s_trunk: ``[*batch, N_token, c_s]``
            z_trunk: ``[*batch, N_token, N_token, c_z]``
            N_sample: diffusion sample count ``S``

        Returns:
            ``[*batch, N_sample, N_atom, 3]`` predicted coordinates
        """
        N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
        batch_shape = s_inputs.shape[:-2]
        device, dtype = s_inputs.device, s_inputs.dtype
        if atom_mask is None:
            atom_mask = input_feature_dict.get("ref_mask")
        mask = None if atom_mask is None else atom_mask.unsqueeze(-2).to(dtype)
        schedule = self.noise_schedule(num_sampling_steps, device,
                                       torch.float32)
        cache = None
        if self.use_cache:
            cache = self.diffusion_module.prepare_cache(
                input_feature_dict, s_inputs, s_trunk, z_trunk, attn_metadata)
            attn_metadata = cache["attn_metadata"]
            # relp is dead once prepare_cache has consumed it; drop before the
            # loop so its memory frees across every denoise step.
            if drop_consumed_features:
                for name in DIFFUSION_CONSUMED_FEATURES:
                    input_feature_dict.pop(name, None)
            elif drop_consumed_relp:
                input_feature_dict.pop("relp", None)
        coords = self.sample(schedule, (*batch_shape, N_sample, N_atom, 3),
                             device,
                             dtype,
                             atom_mask=mask,
                             batch_shape=batch_shape,
                             n_sample=N_sample,
                             input_feature_dict=input_feature_dict,
                             s_inputs=s_inputs,
                             s_trunk=s_trunk,
                             z_trunk=z_trunk,
                             attn_metadata=attn_metadata,
                             cache=cache)
        # Uncached rollout reads relp every step — free only after the loop.
        if not self.use_cache:
            if drop_consumed_features:
                for name in DIFFUSION_CONSUMED_FEATURES:
                    input_feature_dict.pop(name, None)
            elif drop_consumed_relp:
                input_feature_dict.pop("relp", None)
        return coords
