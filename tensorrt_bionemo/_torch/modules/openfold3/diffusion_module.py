# Copyright 2025 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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
"""
Diffusion module. Implements the algorithms in section 3.7 of the
Supplementary Information.
"""
import math
import torch
import torch.nn as nn

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.graph_optimization.config import (
    GraphOptimizationMode, InputAcceptanceDimSpec, InputKeyMethod)
from tensorrt_bionemo._torch.graph_optimization.decorator import (
    NamedDimTies, support_graph_optimization)
from tensorrt_bionemo._torch.layers.conditioning import DiffusionConditioning
from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo._torch.layers.noise_scheduler import \
    SampleDiffusion as _SampleDiffusion
from tensorrt_bionemo._torch.layers.noise_scheduler import \
    create_noise_schedule as _create_noise_schedule
from tensorrt_bionemo._torch.layers.random_augmentation import (
    quaternion_to_matrix, random_quaternions)
from tensorrt_bionemo._torch.layers.transformers.diffusion_transformer import \
    OpenFold3DiffusionTransformer as DiffusionTransformer
from tensorrt_bionemo._torch.modules.openfold3.sequence_local_atom_attention import (
    AtomAttentionDecoder, AtomAttentionEncoder)
from tensorrt_bionemo._torch.modules.openfold3.utils.atomize_utils import \
    compute_atom_broadcast_index
from tensorrt_bionemo._torch.utils import (recursive_calling_load_weights,
                                           safe_generator)
from tensorrt_bionemo.configs import BaseConfig


def broadcast_atom_mask(positions: torch.Tensor,
                        atom_mask: torch.Tensor) -> torch.Tensor:
    """Reshape ``atom_mask`` to broadcast against an atom-position tensor.

    ``positions`` is ``[*, ..., N_atom, 3]`` and ``atom_mask`` is
    ``[*, N_atom]``, where ``positions`` may carry extra batch dims (e.g. a
    diffusion-samples axis ``S``) between the mask's batch dims and the atom
    axis — as in ``[B, S, N_atom, 3]`` vs ``[B, N_atom]``. Returns the mask
    reshaped to ``[*mask_batch, 1, ..., 1, N_atom, 1]`` (cast to ``positions``'s
    dtype) so it broadcasts over those extra dims and the coordinate axis. A bare
    ``atom_mask[..., None]`` would instead right-align the mask's batch dim with
    ``positions``'s samples dim and fail / misbroadcast.
    """
    extra_batch_dims = positions.ndim - atom_mask.ndim - 1
    return atom_mask.reshape(
        *atom_mask.shape[:-1],
        *((1,) * extra_batch_dims),
        atom_mask.shape[-1],
        1,
    ).to(positions.dtype)


def sample_rotations(
        shape,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator = None) -> torch.Tensor:
    """Sample random rotation matrices via random unit quaternions."""

    n = math.prod(shape)
    q = random_quaternions(n, dtype=dtype, device=device, generator=generator)
    return quaternion_to_matrix(q).reshape(*shape, 3, 3)


def centre_random_augmentation(xl: torch.Tensor,
                               atom_mask: torch.Tensor,
                               scale_trans: float = 1.0,
                               generator: torch.Generator = None
                               ) -> torch.Tensor:
    """
    Implements AF3 Algorithm 19.

    Args:
        xl:
            [*, N_atom, 3] Atom positions
        atom_mask:
            [*, N_atom] Atom mask
        scale_trans:
            Translation scaling factor
        generator:
            Optional private RNG (see ``safe_generator``). Supplied
            by ``SampleDiffusion.forward`` so the diffusion rollout's RNG stays
            off the default CUDA generator that ``torch.cuda.graph`` capture
            registers — otherwise these eager draws raise "Offset increment
            outside graph capture" once the diffusion module is graph-wrapped.
    Returns:
        Updated atom position with random global rotation and translation
    """
    rots = sample_rotations(shape=xl.shape[:-2],
                            dtype=xl.dtype,
                            device=xl.device,
                            generator=generator)

    trans = scale_trans * torch.randn((*xl.shape[:-2], 3),
                                      dtype=xl.dtype,
                                      device=xl.device,
                                      generator=generator)

    atom_mask_broadcast = broadcast_atom_mask(xl, atom_mask)
    mean_xl = torch.sum(
        xl * atom_mask_broadcast,
        dim=-2,
        keepdim=True,
    ) / torch.sum(atom_mask_broadcast, dim=-2, keepdim=True).clamp(min=1e-7)

    # center coordinates
    pos_centered = xl - mean_xl
    pos_out = pos_centered @ rots.transpose(-1, -2) + trans[..., None, :]
    pos_out = pos_out * atom_mask_broadcast

    return pos_out


# Move this somewhere else?
def create_noise_schedule(
    no_rollout_steps: float,
    sigma_data: float,
    s_max: float,
    s_min: float,
    p: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """AF3 noise schedule (Page 24) — thin wrapper over the shared
    :func:`create_noise_schedule`. OpenFold3 uses ``no_rollout_steps + 1``
    points and keeps the final (``s_min``) level (``final="keep"``).

    Args:
        no_rollout_steps: number of diffusion rollout steps.
        sigma_data / s_max / s_min / p: schedule parameters.
        dtype / device: output dtype / device.

    Returns:
        Noise schedule ``[no_rollout_steps + 1]``.
    """
    return _create_noise_schedule(num_points=int(no_rollout_steps) + 1,
                                  sigma_data=sigma_data,
                                  s_max=s_max,
                                  s_min=s_min,
                                  rho=p,
                                  device=device,
                                  dtype=dtype,
                                  final="keep")


@support_graph_optimization(
    # si_input/si_trunk (-2), zij_trunk (-2 and -3), and token_mask (-1) carry
    # ``num_tokens``. The output is denoised atom coordinates ([*, N_atom, 3]),
    # whose axes are atoms/coords (no token axis), so there is no output tie.
    named_dims=(
        NamedDimTies(
            name="num_tokens",
            input_dims=(
                ("si_input", (-2,)),
                ("si_trunk", (-2,)),
                ("zij_trunk", (-2, -3)),
                ("token_mask", (-1,)),
            )
        ),
    ),
    graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
    verify_capture=False,
    input_key_method=InputKeyMethod.EXACT,
    input_acceptance_dim_spec=InputAcceptanceDimSpec(
        name="num_tokens", dim_len_max=1024,
    ),
)
class DiffusionModule(nn.Module):
    """
    Implements AF3 Algorithm 20.
    """

    def __init__(self, config: BaseConfig):
        """
        Args:
            config:
                Configuration dictionary for diffusion module
        """
        super().__init__()
        self.c_s = config.c_s
        self.c_token = config.c_token
        self.sigma_data = config.sigma_data
        self.dtype = config.torch_dtype
        self.mapping = config.mapping
        self.skip_create_weights = config.skip_create_weights
        self.sq_sigma_data = config.sigma_data**2

        self.diffusion_conditioning = DiffusionConditioning(
            c_s_input=config.c_s_input,
            c_s=config.c_s,
            c_z=config.c_z,
            c_fourier_emb=config.diffusion_conditioning_config.c_fourier_emb,
            max_relative_idx=config.diffusion_conditioning_config.
            max_relative_idx,
            max_relative_chain=config.diffusion_conditioning_config.
            max_relative_chain,
            sigma_data=config.sigma_data,
            eps=config.eps,
            dtype=config.torch_dtype,
            mapping=config.mapping,
            skip_create_weights=config.skip_create_weights)

        self.atom_attn_enc = AtomAttentionEncoder(
            c_atom_ref_element=config.c_atom_ref_element,
            c_atom_ref_name_chars=config.c_atom_ref_name_chars,
            c_atom=config.c_atom,
            c_atom_pair=config.c_atom_pair,
            c_token=config.c_token,
            atom_transformer_config=config.atom_transformer_encoder_config,
            n_query=config.n_query,
            n_key=config.n_key,
            c_s=config.c_s,
            c_z=config.c_z,
            inf=config.inf,
            eps=config.eps,
            add_noisy_pos=config.add_noisy_pos,
            dtype=config.torch_dtype,
            mapping=config.mapping,
            skip_create_weights=config.skip_create_weights)

        self.layer_norm_s = nn.LayerNorm(self.c_s,
                                         bias=False,
                                         dtype=self.dtype,
                                         eps=config.eps)
        self.linear_s = Linear(self.c_s,
                               self.c_token,
                               bias=False,
                               dtype=self.dtype,
                               mapping=self.mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=self.skip_create_weights)

        self.diffusion_transformer = DiffusionTransformer(
            config=config.diffusion_transformer_config.token_transformer)

        self.layer_norm_a = nn.LayerNorm(self.c_token,
                                         bias=False,
                                         dtype=self.dtype,
                                         eps=config.eps)

        self.atom_attn_dec = AtomAttentionDecoder(
            c_atom=config.c_atom,
            c_atom_pair=config.c_atom_pair,
            c_token=config.c_token,
            c_hidden=config.c_hidden,
            n_query=config.n_query,
            n_key=config.n_key,
            atom_attn_decoder_config=config.atom_transformer_decoder_config,
            inf=config.inf,
            eps=config.eps,
            dtype=config.torch_dtype,
            mapping=config.mapping,
            skip_create_weights=config.skip_create_weights)

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # verify whether all the weights are loaded
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weights}")

    def forward(
        self,
        batch: dict,
        xl_noisy: torch.Tensor,
        token_mask: torch.Tensor,
        atom_mask: torch.Tensor,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        attn_metadata: AttentionMetadata,
        use_conditioning: bool = True,
    ) -> torch.Tensor:
        """
        Note:
            ``SampleDiffusion.forward`` calls this with a diffusion-samples axis
            ``S`` at dim 1 — ``xl_noisy=[B, S, N_atom, 3]`` — which the atom
            encoder carries onto ``ai=[B, S, N_token, c]``. The token-level
            conditioning (``si``/``zij``/``token_mask``) carries a *size-1* sample
            axis at dim 1 (``[B, 1, ...]``) and so broadcasts over ``S`` for any
            batch size ``B``. Batch sizes > 1 are supported end-to-end and stay
            CUDA-graph-capturable: the ``atom_broadcast_index`` fast path in
            ``broadcast_token_feat_to_atoms`` (a static ``index_select`` in place
            of a data-dependent ``repeat_interleave``) handles any batch size.

        Args:
            batch:
                Feature dictionary
            xl_noisy:
                [*, N_atom, 3] Noisy atom positions
            token_mask:
                [*, N_token] Token mask
            atom_mask:
                [*, N_atom] Atom mask
            t:
                [*] Noise level at a diffusion step
            si_input:
                [*, N_token, c_s_input] Input embedding
            si_trunk:
                [*, N_token, c_s] Single representation
            zij_trunk:
                [*, N_token, N_token, c_z] Pair representation
            use_conditioning:
                Whether to condition with the trunk representations
        Returns:
            [*, N_atom, 3] Denoised atom positions
        """
        si, zij = self.diffusion_conditioning(
            batch=batch,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            use_conditioning=use_conditioning)

        xl_noisy = xl_noisy * broadcast_atom_mask(xl_noisy, atom_mask)

        rl_noisy = xl_noisy / torch.sqrt(t[..., None, None] ** 2 +
                                         self.sigma_data**2)

        # Note: These modules are not memory-intensive compared to other parts of the
        # model (i.e. TemplateStack) so chunking is unnecessary for now.
        
        # Input (dim 0 = batch B, dim 1 = the diffusion-samples axis S; the
        # token-level tensors carry a size-1 S placeholder that broadcasts):
        #   atom_mask: [B, 1, N_atom]
        #   rl_noisy:  [B, S, N_atom, 3]
        #   si_trunk:  [B, 1, N_token, c_s=384]
        #   zij:       [B, 1, N_token, N_token, c_z]
        #
        # Output
        #   ai:  [B, S, N_token, c_atom=768]
        #   ql:  [B, 1, N_atom, c_atom]
        #   cl:  [B, 1, N_atom, c_atom]
        #   plm: [B, S, N_blocks, N_query, N_key, c_atom_pair]
        ai, ql, cl, plm = self.atom_attn_enc(
            batch=batch,
            atom_mask=atom_mask,
            rl=rl_noisy,
            si_trunk=si_trunk,
            zij_trunk=zij,
            attn_metadata=attn_metadata,
        )

        # Input
        #   si: [B, 1, N_token, c_token=384]
        #   ai: [B, S, N_token, c_atom=768]
        #   
        # The output of the linear_s has shape [B, 1, N_token, c_atom],
        # so broadcasts with ai for multiplicity S>1
        ai = ai + self.linear_s(self.layer_norm_s(si))

        token_dtype = self.diffusion_transformer.dtype
        ai = self.diffusion_transformer(a=ai.to(dtype=token_dtype),
                                        s=si.to(dtype=token_dtype),
                                        z=zij.to(dtype=token_dtype),
                                        mask=token_mask.to(dtype=token_dtype))
        if token_dtype != torch.float32:
            ai = ai.float()

        ai = self.layer_norm_a(ai)
        rl_update = self.atom_attn_dec(batch=batch,
                                       atom_mask=atom_mask,
                                       ai=ai,
                                       ql=ql,
                                       cl=cl,
                                       plm=plm,
                                       attn_metadata=attn_metadata)
        sq_t = t[..., None, None]**2
        xl_out = (self.sq_sigma_data / (self.sq_sigma_data + sq_t) * xl_noisy +
                  self.sigma_data * t[..., None, None] /
                  torch.sqrt(self.sq_sigma_data + sq_t) * rl_update)

        xl_out = xl_out * broadcast_atom_mask(xl_out, atom_mask)

        return xl_out


class OpenFold3SampleDiffusion(_SampleDiffusion):
    """OpenFold3 EDM sampler (AF3 Algorithm 18).

    Thin subclass of the shared :class:`SampleDiffusion`: reuses the base
    :meth:`sample` loop and implements only the :meth:`denoise` hook (the OF3
    diffusion-module call, which takes a scalar noise level ``t``).
    """

    def __init__(
        self,
        config: BaseConfig,
        diffusion_module: DiffusionModule,
    ):
        """
        Args:
            config:
                Diffusion sampling configuration. This initializer reads
                `gamma_0`, `gamma_min`, `noise_scale`, `step_scale`, and
                `use_conditioning` from this config.
            diffusion_module:
                Instantiated denoising diffusion module used at each sampling
                step.
        """
        super().__init__(gamma0=config.gamma_0,
                         gamma_min=config.gamma_min,
                         noise_scale=config.noise_scale,
                         step_scale=config.step_scale)
        self.diffusion_module = diffusion_module
        self.use_conditioning = config.use_conditioning

    def denoise(self, x_noisy: torch.Tensor, sigma_hat: torch.Tensor,
                ctx: dict) -> torch.Tensor:
        return self.diffusion_module(
            batch=ctx["batch"],
            xl_noisy=x_noisy,
            token_mask=ctx["batch"]["token_mask"],
            atom_mask=ctx["atom_mask"],
            t=sigma_hat.to(x_noisy.device),
            si_input=ctx["si_input"],
            si_trunk=ctx["si_trunk"],
            zij_trunk=ctx["zij_trunk"],
            attn_metadata=ctx["attn_metadata"],
            use_conditioning=ctx["use_conditioning"],
        )

    def forward(
        self,
        batch: dict,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        noise_schedule: torch.Tensor,
        no_rollout_samples: int,
        attn_metadata: AttentionMetadata,
        use_conditioning: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            batch:
                Feature dictionary
            si_input:
                [*, N_token, c_s_input] Input embedding
            si_trunk:
                [*, N_token, c_s] Single representation
            zij_trunk:
                [*, N_token, N_token, c_z] Pair representation
            noise_schedule:
                [no_rollout_steps] Noise schedule
            no_rollout_samples:
                [no_rollout_samples] Number of samples to generate for rollout
            attn_metadata:
                Attention metadata
            use_conditioning:
                Whether to condition with the trunk representations
        Returns:
            [*, N_atom, 3] Sampled atom positions
        """
        atom_mask = batch["atom_mask"]
        batch_dim, num_atoms = atom_mask.shape[0], atom_mask.shape[-1]
        device = atom_mask.device

        # Precompute the token->atom expansion index ONCE here — eagerly and
        # outside the CUDA-graph-captured ``self.diffusion_module`` — so the
        # atom-attention broadcasts inside it run as a static index_select
        # instead of a data-dependent torch.repeat_interleave (which forces a
        # device->host sync and is not graph-capturable). It is constant across
        # all rollout steps, so it also removes a per-step sync in eager mode.
        batch["atom_broadcast_index"] = compute_atom_broadcast_index(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch["num_atoms_per_token"],
        )

        # Draw the rollout's stochasticity from a private generator so these
        # eager torch.randn calls stay off the default CUDA generator that
        # torch.cuda.graph capture of ``self.diffusion_module`` registers (see
        # safe_generator). Cloning the default state keeps numerics identical to
        # the un-wrapped run; the default generator is advanced to match on exit.
        with safe_generator(device) as generator:
            xl = noise_schedule[0] * torch.randn(
                (batch_dim, no_rollout_samples, num_atoms, 3),
                device=device,
                dtype=self.diffusion_module.dtype,
                generator=generator,
            )

            for tau, c_tau in enumerate(noise_schedule[1:]):
                xl = centre_random_augmentation(xl=xl,
                                                atom_mask=atom_mask,
                                                generator=generator)

                gamma = self.gamma0 if c_tau > self.gamma_min else 0

                t = noise_schedule[tau] * (gamma + 1)

                noise = (self.noise_scale *
                         torch.sqrt(t**2 - noise_schedule[tau]**2) *
                         torch.randn(xl.shape,
                                     dtype=xl.dtype,
                                     device=xl.device,
                                     generator=generator))

                xl_noisy = xl + noise

                xl_denoised = self.diffusion_module(
                    batch=batch,
                    xl_noisy=xl_noisy,
                    token_mask=batch["token_mask"],
                    atom_mask=atom_mask,
                    t=t.to(xl_noisy.device),
                    si_input=si_input,
                    si_trunk=si_trunk,
                    zij_trunk=zij_trunk,
                    attn_metadata=attn_metadata,
                    use_conditioning=use_conditioning,
                )

                delta = (xl_noisy - xl_denoised) / t
                dt = c_tau - t
                xl = xl_noisy + self.step_scale * dt * delta

        return xl
