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
from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from bionemo_ir._torch.graph_optimization.config import (
    GraphOptimizationMode,
    InputAcceptanceDimSpec,
    InputKeyMethod,
)
from bionemo_ir._torch.graph_optimization.decorator import NamedDimTies, support_graph_optimization
from bionemo_ir._torch.layers.attention import AttentionMetadata
from bionemo_ir._torch.layers.conditioning import PairwiseConditioning, SingleConditioning
from bionemo_ir._torch.layers.linear import Linear
from bionemo_ir._torch.layers.position_encoders import FourierEmbedding
from bionemo_ir._torch.layers.random_augmentation import random_rotations
from bionemo_ir._torch.layers.transformers.atom import AtomAttentionDecoder, AtomAttentionEncoder
from bionemo_ir._torch.layers.transformers.diffusion_transformer import BoltzDiffusionTransformer
from bionemo_ir._torch.layers.transition import ConditionedTransitionBlock
from bionemo_ir._torch.modules.boltz.embedders import AtomEmbedding
from bionemo_ir._torch.modules.boltz.loss.diffusion import weighted_rigid_align
from bionemo_ir._torch.modules.boltz.physical.potentials import get_potentials
from bionemo_ir._torch.modules.boltz.physical.steering import BoltzSteeringParams
from bionemo_ir._torch.sampling import edm as edm_sampling
from bionemo_ir._torch.sampling.contracts import SamplingContext
from bionemo_ir._torch.sampling.denoise_integrator import DenoiseIntegratorTemplate
from bionemo_ir._torch.sampling.hooks import DenoiseHookPipeline
from bionemo_ir._torch.sampling.runner import GenerativeRunner
from bionemo_ir._torch.utils import recursive_calling_load_weights
from bionemo_ir.configs import BaseConfig
from bionemo_ir.pipeline.models.boltz2.const import num_pocket_contact_info, num_tokens
from bionemo_ir.runtime.buffers import PreallocatedBuffers


class DiffusionConditioning(nn.Module):
    def __init__(
        self,
        token_s: int,
        token_z: int,
        atom_s: int,
        atom_z: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        atom_feature_dim: int = 128,
        conditioning_transition_layers: int = 2,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
        eps: float = 1e-5,
        version: str = "v1",
        dtype: torch.dtype = torch.float32,
        pairwise_conditioner_dtype: torch.dtype | None = None,
        token_trans_bias_dtype: torch.dtype | None = None,
        skip_create_weights: bool = False,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        # The token-transformer bias [N, N, depth*heads] is the large one (depth=24, heads=16 ->
        # 384); build it directly in ``token_trans_bias_dtype`` (bf16) since the token transformer
        # consumes it at that precision anyway. ``None`` keeps it at ``dtype``.
        self.token_trans_bias_dtype = token_trans_bias_dtype or dtype

        # The pairwise conditioner's [N, N, 2*hidden] FFN dominates memory here; run it at a possibly
        # reduced precision (``pairwise_conditioner_dtype``, e.g. bf16) and cast the result back to
        # ``self.dtype`` in ``forward`` so the rest of conditioning (heads etc.) is unchanged.
        # ``None`` keeps it at ``dtype``.
        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=conditioning_transition_layers,
            eps=eps,
            dtype=pairwise_conditioner_dtype or dtype,
            skip_create_weights=skip_create_weights,
        )

        self.atom_embedding = AtomEmbedding(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            structure_prediction=True,
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
            version=version,
            dtype=dtype,
            skip_create_weights=skip_create_weights,
        )

        self.atom_enc_proj_z = nn.ModuleList()
        for _ in range(atom_encoder_depth):
            self.atom_enc_proj_z.append(
                nn.Sequential(
                    nn.LayerNorm(atom_z),
                    nn.Linear(atom_z, atom_encoder_heads, bias=False),
                )
            )

        self.atom_dec_proj_z = nn.ModuleList()
        for _ in range(atom_decoder_depth):
            self.atom_dec_proj_z.append(
                nn.Sequential(
                    nn.LayerNorm(atom_z),
                    nn.Linear(atom_z, atom_decoder_heads, bias=False),
                )
            )

        self.token_trans_proj_z = nn.ModuleList()
        for _ in range(token_transformer_depth):
            self.token_trans_proj_z.append(
                nn.Sequential(
                    nn.LayerNorm(token_z, dtype=self.token_trans_bias_dtype),
                    nn.Linear(token_z, token_transformer_heads, bias=False, dtype=self.token_trans_bias_dtype),
                )
            )

    def get_module_feed_dict(self, feed_dict: dict[str, torch.Tensor], module_name: str) -> dict[str, Any]:
        keys = []
        if module_name == "atom_embedding":
            keys = [
                "atom_to_token",
                "ref_pos",
                "atom_pad_mask",
                "ref_space_uid",
                "ref_charge",
                "ref_element",
                "ref_atom_name_chars",
                "atom_backbone_feat",
                "res_type",
                "modified",
                "mol_type",
            ]
        else:
            raise ValueError(f"Module name {module_name} not supported")
        return {key: feed_dict.get(key, None) for key in keys}

    def load_weights(self, weights: dict):
        loaded_weight = recursive_calling_load_weights(self, weights)
        # Every entry of ``weights`` must have been consumed.
        not_loaded_weights = set(weights.keys()) - loaded_weight
        if not_loaded_weights:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weights}")

    def forward(
        self,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        relative_position_encoding: torch.Tensor,
        feature_dict: dict[str, torch.Tensor],
        query_to_keys: Callable,
    ):
        """
        Args:
            s_trunk: (B, N, token_s)
                Trunk sequence embeddings
            z_trunk: (B, N, N, token_z)
                Trunk pairwise embeddings
            relative_position_encoding: (B, N, N, token_z)
                Relative position encoding for sequence local attention
            feature_dict: dict[str, torch.Tensor]
                Feature dict from DataLoader
            query_to_keys: Callable
                Query to keys function for sequence local attention
        Returns:
            q: (B, N, token_z)
                Query embeddings
            c: (B, N, token_s)
                Context embeddings
            atom_enc_bias: (B, N, atom_encoder_heads)
            atom_dec_bias: (B, N, atom_decoder_heads)
            token_trans_bias: (B, N, token_transformer_heads)
        """
        z = self.pairwise_conditioner(
            z_trunk,
            relative_position_encoding,
        ).to(self.dtype)

        q, c, p = self.atom_embedding(
            **self.get_module_feed_dict(feature_dict, "atom_embedding"),
            query_to_keys=query_to_keys,
            s_trunk=s_trunk,
            z=z,
        )

        atom_enc_bias = []
        for layer in self.atom_enc_proj_z:
            atom_enc_bias.append(layer(p))
        atom_enc_bias = torch.cat(atom_enc_bias, dim=-1)

        atom_dec_bias = []
        for layer in self.atom_dec_proj_z:
            atom_dec_bias.append(layer(p))
        atom_dec_bias = torch.cat(atom_dec_bias, dim=-1)

        # Cast z once and build the large [N, N, depth*heads] token-transformer bias directly in the
        # (bf16) token_trans_bias dtype -- it feeds the token transformer at that precision anyway.
        token_trans_bias = []
        z_ttb = z.to(self.token_trans_bias_dtype)
        for layer in self.token_trans_proj_z:
            token_trans_bias.append(layer(z_ttb))
        token_trans_bias = torch.cat(token_trans_bias, dim=-1)

        return q, c, atom_enc_bias, atom_dec_bias, token_trans_bias


@support_graph_optimization(
    # s_inputs/s_trunk (-2) and token_pad_mask (-1) carry ``num_tokens``. The
    # pair rep is nested inside diffusion_conditioning_kwargs (not tied) and the
    # output is atom coordinates (no token axis, so no output tie).
    named_dims=(
        NamedDimTies(
            name="num_tokens",
            input_dims=(
                ("s_inputs", (-2,)),
                ("s_trunk", (-2,)),
                ("token_pad_mask", (-1,)),
            ),
        ),
    ),
    graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
    input_key_method=InputKeyMethod.EXACT,
    # Default input management: accept up to 1024 tokens before falling back to
    # eager.
    input_acceptance_dim_spec=InputAcceptanceDimSpec(name="num_tokens", dim_len_max=1024),
)
class DiffusionModule(nn.Module):
    """Diffusion module"""

    def __init__(
        self,
        config: BaseConfig = None,
    ) -> None:
        super().__init__()
        # Set the dtype for the token transformer
        self.dtype = config.torch_dtype

        # Ensure the dtype is consistent for all the modules
        assert self.dtype == config.atom_encoder.torch_dtype, (
            f"DiffusionModule dtype: {self.dtype}, atom_encoder dtype: {config.atom_encoder.torch_dtype}"
        )
        assert self.dtype == config.atom_decoder.torch_dtype, (
            f"DiffusionModule dtype: {self.dtype}, atom_decoder dtype: {config.atom_decoder.torch_dtype}"
        )
        assert self.dtype == config.token_transformer.torch_dtype, (
            f"DiffusionModule dtype: {self.dtype}, token_transformer dtype: {config.token_transformer.torch_dtype}"
        )

        skip_create_weights = config.skip_create_weights
        eps = config.norm_epsilon

        self.atoms_per_window_queries = config.atoms_per_window_queries
        self.atoms_per_window_keys = config.atoms_per_window_keys

        additional_input_dim = 0
        if config.version == "v1":
            # NOTE: v1 uses additional input_dim for the single conditioning
            additional_input_dim = 2 * num_tokens + 1 + num_pocket_contact_info
        # conditioning
        self.single_conditioner = SingleConditioning(
            token_s=config.token_s,
            dim_fourier=config.dim_fourier,
            num_transitions=config.conditioning_transition_layers,
            additional_input_dim=additional_input_dim,
            eps=eps,
            dtype=self.dtype,
            skip_create_weights=skip_create_weights,
        )
        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=config.atom_s,
            token_s=config.token_s,
            atoms_per_window_queries=config.atoms_per_window_queries,
            atoms_per_window_keys=config.atoms_per_window_keys,
            structure_prediction=True,
            diffusion_transformer_config=config.atom_encoder,
            diffusion_transformer_cls=BoltzDiffusionTransformer,
            version=config.version,
            dtype=config.atom_encoder.torch_dtype,
            skip_create_weights=skip_create_weights,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * config.token_s, dtype=self.dtype, eps=eps),
            Linear(
                2 * config.token_s,
                2 * config.token_s,
                bias=False,
                dtype=self.dtype,
                skip_create_weights=config.skip_create_weights,
            ),
        )

        self.token_transformer = BoltzDiffusionTransformer(config=config.token_transformer)

        self.a_norm = nn.LayerNorm(2 * config.token_s, dtype=self.dtype, eps=eps)

        self.atom_attention_decoder = AtomAttentionDecoder(
            token_s=config.token_s,
            atom_s=config.atom_s,
            atoms_per_window_queries=config.atoms_per_window_queries,
            atoms_per_window_keys=config.atoms_per_window_keys,
            diffusion_transformer_config=config.atom_decoder,
            diffusion_transformer_cls=BoltzDiffusionTransformer,
            dtype=config.atom_decoder.torch_dtype,
            skip_create_weights=skip_create_weights,
        )

    def forward(
        self,
        atom_to_token: torch.Tensor,
        atom_pad_mask: torch.Tensor,
        token_pad_mask: torch.Tensor,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        r_noisy: torch.Tensor,
        times: torch.Tensor,
        diffusion_conditioning_kwargs: dict[str, torch.Tensor],
        attn_metadata: AttentionMetadata | None = None,
    ):
        """
        Note:
        The ``r_noisy`` layout differs from the upstream Boltz implementation:
        - Upstream Boltz: (B*multiplicity, N_atoms, 3)
        - Here: (B, multiplicity, N_atoms, 3)
        Keeping the two dimensions separate makes the batch and
        multiplicity axes explicit.
        This allows for better broadcasting when the atom coordinates are broadcasted to the atom attention decoder.

        Args:
            atom_to_token: (B, N_atoms, N_res)
                The atom to token mapping.
            atom_pad_mask: (B, N_atoms)
                The atom pad mask.
            token_pad_mask: (B, N_res)
                The token pad mask.
            s_inputs: (B, N, token_s)
                The input sequence embeddings.
            s_trunk: (B, N, token_s)
                The trunk sequence embeddings.
            r_noisy: (B, multiplicity, N_atoms, 3)
                The noisy atom coordinates.
            times: (B, 1)
                The time steps.
            diffusion_conditioning_kwargs: dict[str, torch.Tensor]
                - q: (B, N, token_z)
                - c: (B, N, token_s)
                - atom_enc_bias: (B, N, atom_encoder_heads)
                - token_trans_bias: (B, N, token_transformer_heads)
                - atom_dec_bias: (B, N, atom_decoder_heads)
            attn_metadata: AttentionMetadata | None
                The attention metadata.
        Returns:
            r_update: (B, multiplicity, N_atoms, 3)
                The updated atom coordinates.
        """
        assert r_noisy.ndim == 4, "r_noisy must be 4D, shape: (B, multiplicity, N_atoms, 3)"
        assert attn_metadata is not None, "Attention metadata is required for DiffusionTransformers"
        assert attn_metadata.query_to_keys is not None, "Query to keys is required for DiffusionTransformers"

        q = diffusion_conditioning_kwargs["q"]
        c = diffusion_conditioning_kwargs["c"]
        atom_enc_bias = diffusion_conditioning_kwargs["atom_enc_bias"]
        token_trans_bias = diffusion_conditioning_kwargs["token_trans_bias"]
        atom_dec_bias = diffusion_conditioning_kwargs["atom_dec_bias"]

        s_trunk = s_trunk.to(self.dtype)
        s_inputs = s_inputs.to(self.dtype)

        s, normed_fourier = self.single_conditioner(
            times,
            s_trunk,
            s_inputs,
        )
        # s: [B, multiplicity, N, 2*token_s]

        buffers: PreallocatedBuffers | None = None
        if self.token_transformer.pairwise_attention_backend == "CuTeDSL":
            buffers = {}

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        a, q_skip, c_skip = self.atom_attention_encoder(
            atom_to_token=atom_to_token,
            atom_pad_mask=atom_pad_mask,
            q=q,
            c=c,
            bias=atom_enc_bias,
            r=r_noisy.to(self.dtype),
            attn_metadata=attn_metadata,
            buffers=buffers,
        )
        # a: [B, multiplicity, N_res, 2 * token_s]
        # q_skip: [B, multiplicity, N_atoms, atom_s]
        # c_skip: [B, 1, N_atoms, D]

        # Full self-attention on token level, expand dims for broadcasting
        mask = token_pad_mask.unsqueeze(1)
        token_trans_bias = token_trans_bias.unsqueeze(1)
        a = a + self.s_to_a_linear(s.to(self.dtype))

        # Token transformer doesn't need query to keys, it's self-attention on token level.
        token_transformer_attn_metadata = AttentionMetadata()
        a = self.token_transformer(
            a=a,
            s=s,
            z=token_trans_bias,
            mask=mask,
            attn_metadata=token_transformer_attn_metadata,
            buffers=buffers,
        )
        a = self.a_norm(a)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update = self.atom_attention_decoder(
            atom_to_token=atom_to_token,
            atom_pad_mask=atom_pad_mask,
            a=a,
            q=q_skip,
            c=c_skip,
            bias=atom_dec_bias,
            attn_metadata=attn_metadata,
            buffers=buffers,
        )

        return r_update, a


class OutTokenFeatUpdate(nn.Module):
    """Output token feature update"""

    def __init__(
        self,
        sigma_data: float,
        token_s=384,
        dim_fourier=256,
        dtype: torch.dtype = torch.float32,
        skip_create_weights: bool = False,
    ):
        """Initialize the Output token feature update for confidence model.

        Parameters
        ----------
        sigma_data : float
            The standard deviation of the data distribution.
        token_s : int, optional
            The token dimension, by default 384.
        dim_fourier : int, optional
            The dimension of the fourier embedding, by default 256.

        """

        super().__init__()
        self.sigma_data = sigma_data
        self.dtype = dtype
        self.norm_next = nn.LayerNorm(2 * token_s, dtype=dtype)
        self.fourier_embed = FourierEmbedding(dim_fourier, dtype=dtype)
        self.norm_fourier = nn.LayerNorm(dim_fourier, dtype=dtype)
        self.transition_block = ConditionedTransitionBlock(
            2 * token_s, 2 * token_s + dim_fourier, dtype=dtype, skip_create_weights=skip_create_weights
        )

    def forward(
        self,
        times: torch.Tensor,
        acc_a: torch.Tensor,
        next_a: torch.Tensor,
    ):
        """
        Args:
            times: [B, multiplicity]
            acc_a: [B, multiplicity, N, 2 * token_s]
            next_a: [B, multiplicity, N, 2 * token_s]
        Returns:
            acc_a: [B, multiplicity, N, 2 * token_s]
        """
        next_a = next_a.to(self.dtype)
        acc_a = acc_a.to(self.dtype)

        # [B, multiplicity, N, 2 * token_s]
        next_a = self.norm_next(next_a)
        # [B, multiplicity, dim_fourier]
        fourier_embed = self.fourier_embed(times)
        normed_fourier = self.norm_fourier(fourier_embed).unsqueeze(2).expand(-1, -1, next_a.shape[2], -1)
        cond_a = torch.cat((acc_a, normed_fourier), dim=-1)

        acc_a = acc_a + self.transition_block(next_a, cond_a)

        return acc_a


@dataclass
class BoltzEDMSamplingState:
    """Persistent tensors carried between Boltz EDM steps."""

    atom_coords: torch.Tensor
    atom_mask: torch.Tensor
    atom_coords_denoised: torch.Tensor | None = None
    token_repr: torch.Tensor | None = None
    token_a: torch.Tensor | None = None


@dataclass(frozen=True)
class BoltzDenoisePrediction:
    """Chunk-assembled outputs from one Boltz denoiser invocation."""

    atom_coords_denoised: torch.Tensor
    token_a: torch.Tensor | None


@dataclass
class BoltzEDMDenoiseStep:
    """Mutable data passed through hooks around one Boltz denoiser call."""

    step_index: int
    state: BoltzEDMSamplingState
    sigma_last: float
    sigma_next: float
    gamma: float
    steering_time: float
    random_rotation: torch.Tensor | None = None
    noise: torch.Tensor | None = None
    noise_variance: float = 0.0
    noisy_atom_coords: torch.Tensor | None = None

    @property
    def sigma_hat(self) -> float:
        return self.sigma_last * (1 + self.gamma)


class BoltzRandomRigidAugmentationHook:
    """Apply Boltz's per-step rigid transform before denoising."""

    def __init__(self, batch_size: int, multiplicity: int) -> None:
        self.batch_size = batch_size
        self.multiplicity = multiplicity

    def before_denoise(
        self,
        step: BoltzEDMDenoiseStep,
        context: SamplingContext,
    ) -> BoltzEDMDenoiseStep:
        state = step.state
        random_rotation = random_rotations(
            self.batch_size * self.multiplicity,
            device=state.atom_coords.device,
            dtype=state.atom_coords.dtype,
            generator=context.generator,
        ).view(self.batch_size, self.multiplicity, 3, 3)
        random_translation = torch.randn(
            (self.batch_size, self.multiplicity, 1, 3),
            device=state.atom_coords.device,
            dtype=state.atom_coords.dtype,
            generator=context.generator,
        )

        state.atom_coords = state.atom_coords - state.atom_coords.mean(dim=-2, keepdims=True)
        state.atom_coords = torch.einsum("bmnd,bmds->bmns", state.atom_coords, random_rotation) + random_translation

        if state.atom_coords_denoised is not None:
            state.atom_coords_denoised = state.atom_coords_denoised - state.atom_coords_denoised.mean(
                dim=-2, keepdims=True
            )
            state.atom_coords_denoised = (
                torch.einsum("bmnd,bmds->bmns", state.atom_coords_denoised, random_rotation) + random_translation
            )

        step.random_rotation = random_rotation
        return step


class BoltzEDMChurnHook:
    """Add the EDM churn perturbation immediately before denoising."""

    def __init__(self, noise_scale: float) -> None:
        self.noise_scale = noise_scale

    def before_denoise(
        self,
        step: BoltzEDMDenoiseStep,
        context: SamplingContext,
    ) -> BoltzEDMDenoiseStep:
        step.noise_variance = self.noise_scale**2 * (step.sigma_hat**2 - step.sigma_last**2)
        step.noise = sqrt(step.noise_variance) * torch.randn(
            step.state.atom_coords.shape,
            device=step.state.atom_coords.device,
            dtype=step.state.atom_coords.dtype,
            generator=context.generator,
        )
        step.noisy_atom_coords = step.state.atom_coords + step.noise
        return step


class PotentialGuidanceHook:
    def __init__(
        self,
        steering_args: BoltzSteeringParams = None,
        atom_mask: torch.Tensor = None,
        multiplicity: int = 1,
        num_sampling_steps: int = 50,
        step_scale: float = 1.0,
        boltz2: bool = False,
        device: torch.device = None,
        feature_dict: dict[str, torch.Tensor] = None,
    ) -> None:
        """Initialize Boltz potential guidance as a two-sided denoise hook.

        Args:
            steering_args: BoltzSteeringParams
                The steering arguments.
            atom_mask: torch.Tensor
                The atom mask. Shape (B, N_atoms)
            multiplicity: int
                The multiplicity.
            num_sampling_steps: int
                The number of sampling steps.
            step_scale: float
        """
        self.steering_args = steering_args
        self.boltz2 = boltz2
        self.multiplicity = multiplicity
        self.energy_traj = None
        self.resample_weights = None
        self.scaled_guidance_update = None
        self.potentials = []
        self.num_sampling_steps = num_sampling_steps
        self.step_scale = step_scale
        self.device = device
        self.feature_dict = feature_dict
        self.batch_size = atom_mask.shape[0]
        self.step_idx = 0

        self.need_guidance_update = False

        if self.steering_args is None:
            return

        if not boltz2:
            # Boltz1 does not support contact guidance update
            self.steering_args.contact_guidance_update = False

        self.need_guidance_update = (
            self.steering_args.physical_guidance_update or self.steering_args.contact_guidance_update
        )
        self.potentials = get_potentials(steering_args, boltz2=boltz2)
        if self.steering_args.fk_steering:
            self.multiplicity = multiplicity * self.steering_args.num_particles
            # [B, multiplicity*num_particles, 0]
            self.energy_traj = torch.empty((self.batch_size, self.multiplicity, 0), device=device)
            # [B, multiplicity, num_particles]
            self.resample_weights = torch.ones(
                self.batch_size, multiplicity, self.steering_args.num_particles, device=device
            )
        if self.need_guidance_update:
            # [B, multiplicity*num_particles, N_atoms, 3]
            self.scaled_guidance_update = torch.zeros(
                (self.batch_size, self.multiplicity, atom_mask.shape[1], 3),
                dtype=torch.float32,
                device=device,
            )

    def before_denoise(
        self,
        step: BoltzEDMDenoiseStep,
        context: SamplingContext,
    ) -> BoltzEDMDenoiseStep:
        del context
        self.set_diffusion_reverse_step(step.step_index)
        if step.random_rotation is not None:
            self.apply_random_rotation(step.random_rotation)
        return step

    def after_denoise(
        self,
        step: BoltzEDMDenoiseStep,
        context: SamplingContext,
    ) -> BoltzEDMDenoiseStep:
        state = step.state
        if state.atom_coords_denoised is None or step.noise is None or step.noisy_atom_coords is None:
            raise RuntimeError("potential guidance requires churn and denoising outputs")

        self.update_resampling_weights(
            state.atom_coords_denoised,
            step.noise,
            step.steering_time,
            step.noise_variance,
            self.feature_dict,
        )
        state.atom_coords_denoised = self.apply_guidance_update(
            state.atom_coords_denoised,
            step.steering_time,
            step.sigma_next,
            step.sigma_hat,
            self.feature_dict,
        )
        (
            state.atom_coords,
            step.noisy_atom_coords,
            state.atom_mask,
            state.atom_coords_denoised,
            state.token_repr,
            state.token_a,
        ) = self.fk_resampling(
            state.atom_coords,
            step.noisy_atom_coords,
            state.atom_mask,
            step.noise_variance,
            state.atom_coords_denoised,
            state.token_repr,
            state.token_a,
            generator=context.generator,
        )
        return step

    def set_diffusion_reverse_step(self, step_idx: int) -> None:
        self.step_idx = step_idx

    def need_fk_resampling(self, noise_var: float) -> bool:
        if self.steering_args is None:
            return False
        if not self.steering_args.fk_steering:
            return False
        if self.step_idx % self.steering_args.fk_resampling_interval == 0 and noise_var > 0:
            return True
        if self.step_idx == self.num_sampling_steps - 1:
            return True
        return False

    def need_guidance_update_by_step(self) -> bool:
        if self.steering_args is None:
            return False
        if not self.need_guidance_update:
            return False
        if self.step_idx >= self.num_sampling_steps - 1:
            return False
        return True

    def apply_random_rotation(self, random_R: torch.Tensor) -> None:
        """
        Args:
            random_R(torch.Tensor):
                The random rotation matrix. Shape (B, mult, 3, 3)
        """
        if self.steering_args is None:
            return
        if self.need_guidance_update:
            self.scaled_guidance_update = torch.einsum("bmij,bmjk->bmik", self.scaled_guidance_update, random_R)

    def update_resampling_weights(
        self,
        atom_coords_denoised: torch.Tensor,
        eps: torch.Tensor,
        steering_t: float,
        noise_var: float,
        feed_dict: dict[str, torch.Tensor],
    ) -> None:

        if not self.need_fk_resampling(noise_var):
            return

        energy = torch.zeros(self.batch_size, self.multiplicity, device=self.device)
        for potential in self.potentials:
            parameters = potential.compute_parameters(steering_t)
            if parameters["resampling_weight"] > 0:
                component_energy = potential.compute(
                    atom_coords_denoised,
                    feed_dict,
                    parameters,
                )
                energy += parameters["resampling_weight"] * component_energy
        self.energy_traj = torch.cat((self.energy_traj, energy.unsqueeze(-1)), dim=-1)
        # Compute log G values
        if self.energy_traj.shape[-1] == 1:
            log_G = -1 * energy
        else:
            log_G = self.energy_traj[..., -2] - self.energy_traj[..., -1]

        # Compute ll difference between guided and unguided transition distribution
        if (
            self.steering_args.physical_guidance_update or self.steering_args.contact_guidance_update
        ) and noise_var > 0:
            ll_difference = (eps**2 - (eps + self.scaled_guidance_update) ** 2).sum(dim=(-1, -2)) / (2 * noise_var)
        else:
            ll_difference = torch.zeros_like(energy)

        # Compute resampling weights
        # [B, multiplicity, num_particles]
        self.resample_weights = F.softmax(
            (ll_difference + self.steering_args.fk_lambda * log_G).reshape(
                self.batch_size, -1, self.steering_args.num_particles
            ),
            dim=-1,
        )

    def apply_guidance_update(
        self,
        atom_coords_denoised: torch.Tensor,
        steering_t: float,
        sigma_t: float,
        t_hat: float,
        feed_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not self.need_guidance_update_by_step():
            return atom_coords_denoised

        guidance_update = torch.zeros_like(atom_coords_denoised)
        for guidance_step in range(self.steering_args.num_gd_steps):
            energy_gradient = torch.zeros_like(atom_coords_denoised)
            for potential in self.potentials:
                parameters = potential.compute_parameters(steering_t)
                if parameters["guidance_weight"] > 0 and (guidance_step) % parameters["guidance_interval"] == 0:
                    energy_gradient += parameters["guidance_weight"] * potential.compute_gradient(
                        atom_coords_denoised + guidance_update,
                        feed_dict,
                        parameters,
                    )
            guidance_update -= energy_gradient
        atom_coords_denoised += guidance_update
        self.scaled_guidance_update = guidance_update * -1 * self.step_scale * (sigma_t - t_hat) / t_hat
        return atom_coords_denoised

    def fk_resampling(
        self,
        atom_coords: torch.Tensor,
        atom_coords_noisy: torch.Tensor,
        atom_mask: torch.Tensor,
        noise_var: float,
        atom_coords_denoised: torch.Tensor | None = None,
        token_repr: torch.Tensor | None = None,
        token_a: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if not self.need_fk_resampling(noise_var):
            return atom_coords, atom_coords_noisy, atom_mask, atom_coords_denoised, token_repr, token_a

        # [B, multiplicity, num_particles] when in diffusion loop
        # [B, multiplicity, 1] when exiting diffusion loop

        sample_size = self.steering_args.num_particles if self.step_idx < self.num_sampling_steps - 1 else 1
        resample_indices = torch.multinomial(
            self.resample_weights.view(-1, self.steering_args.num_particles),
            sample_size,
            replacement=True,
            generator=generator,
        ).view(self.batch_size, -1, sample_size)
        resample_indices = (
            resample_indices
            + self.steering_args.num_particles
            * torch.arange(self.resample_weights.shape[1], device=self.resample_weights.device)[None, :, None]
        )
        resample_indices = resample_indices.view(self.batch_size, -1)  # [B, multiplicity*sample_size]

        batch_indices = torch.arange(self.batch_size, device=self.device).unsqueeze(1)

        atom_coords = atom_coords[batch_indices, resample_indices]
        atom_coords_noisy = atom_coords_noisy[batch_indices, resample_indices]
        atom_mask = atom_mask[batch_indices, resample_indices]
        if atom_coords_denoised is not None:
            atom_coords_denoised = atom_coords_denoised[batch_indices, resample_indices]
        self.energy_traj = self.energy_traj[batch_indices, resample_indices]
        if self.need_guidance_update:
            self.scaled_guidance_update = self.scaled_guidance_update[batch_indices, resample_indices]
        if token_repr is not None:
            token_repr = token_repr[batch_indices, resample_indices]
        if token_a is not None:
            token_a = token_a[batch_indices, resample_indices]
        return atom_coords, atom_coords_noisy, atom_mask, atom_coords_denoised, token_repr, token_a


class BoltzTokenRepresentationHook:
    """Accumulate Boltz1 token representations after each denoiser call."""

    def __init__(self, update: nn.Module, noise_conditioning: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.update = update
        self.noise_conditioning = noise_conditioning

    def after_denoise(
        self,
        step: BoltzEDMDenoiseStep,
        context: SamplingContext,
    ) -> BoltzEDMDenoiseStep:
        del context
        state = step.state
        if state.atom_coords_denoised is None or state.token_a is None:
            raise RuntimeError("token accumulation requires denoised coordinates and token_a")
        if state.token_repr is None:
            state.token_repr = torch.zeros_like(state.token_a)

        sigma = torch.full(
            (state.atom_coords_denoised.shape[0], state.atom_coords_denoised.shape[1]),
            step.sigma_hat,
            device=state.atom_coords_denoised.device,
        )
        state.token_repr = self.update(
            times=self.noise_conditioning(sigma),
            acc_a=state.token_repr,
            next_a=state.token_a,
        )
        return step


class BoltzRigidAlignmentHook:
    """Align noisy coordinates to the denoised prediction before Euler."""

    def after_denoise(
        self,
        step: BoltzEDMDenoiseStep,
        context: SamplingContext,
    ) -> BoltzEDMDenoiseStep:
        del context
        state = step.state
        if step.noisy_atom_coords is None or state.atom_coords_denoised is None:
            raise RuntimeError("rigid alignment requires noisy and denoised coordinates")
        with torch.autocast("cuda", enabled=False):
            step.noisy_atom_coords = weighted_rigid_align(
                step.noisy_atom_coords.float(),
                state.atom_coords_denoised.float(),
                state.atom_mask.float(),
                state.atom_mask.float(),
            )
        step.noisy_atom_coords = step.noisy_atom_coords.to(state.atom_coords_denoised)
        return step


class BoltzEDMIntegrator(
    DenoiseIntegratorTemplate[
        edm_sampling.EDMRolloutPlan,
        BoltzEDMSamplingState,
        BoltzEDMDenoiseStep,
        torch.Tensor,
        float,
        BoltzDenoisePrediction,
    ]
):
    """Run Boltz EDM state through the shared template-method lifecycle."""

    def __init__(
        self,
        config: edm_sampling.EDMIntegratorConfig,
        hook_pipeline: DenoiseHookPipeline[BoltzEDMDenoiseStep],
    ) -> None:
        super().__init__(hook_pipeline)
        self.config = config

    def num_steps(self, plan: edm_sampling.EDMRolloutPlan) -> int:
        return plan.num_steps

    def initialize(
        self,
        plan: edm_sampling.EDMRolloutPlan,
        context: SamplingContext,
    ) -> BoltzEDMSamplingState:
        if plan.atom_mask is None:
            raise ValueError("Boltz EDM rollout requires atom_mask")
        noise = torch.randn(
            plan.coords_shape,
            device=plan.device,
            dtype=plan.dtype,
            generator=context.generator,
        )
        return BoltzEDMSamplingState(atom_coords=plan.schedule[0] * noise, atom_mask=plan.atom_mask)

    def prepare_denoise_step(
        self,
        step_index: int,
        state: BoltzEDMSamplingState,
        plan: edm_sampling.EDMRolloutPlan,
        context: SamplingContext,
    ) -> BoltzEDMDenoiseStep:
        del context
        sigma_last = plan.schedule[step_index]
        sigma_next = plan.schedule[step_index + 1]
        gamma = torch.where(
            sigma_next > self.config.gamma_min, sigma_next.new_tensor(self.config.gamma0), sigma_next.new_zeros(())
        )
        return BoltzEDMDenoiseStep(
            step_index=step_index,
            state=state,
            sigma_last=sigma_last.item(),
            sigma_next=sigma_next.item(),
            gamma=gamma.item(),
            steering_time=1.0 - (step_index / plan.num_steps),
        )

    def denoiser_inputs(self, step: BoltzEDMDenoiseStep) -> tuple[torch.Tensor, float]:
        if step.noisy_atom_coords is None:
            raise RuntimeError("Boltz pre-denoise hooks did not produce noisy coordinates")
        return step.noisy_atom_coords, step.sigma_hat

    def attach_prediction(
        self,
        step: BoltzEDMDenoiseStep,
        prediction: BoltzDenoisePrediction,
    ) -> BoltzEDMDenoiseStep:
        step.state.atom_coords_denoised = prediction.atom_coords_denoised
        step.state.token_a = prediction.token_a
        return step

    def complete_step(
        self,
        step: BoltzEDMDenoiseStep,
    ) -> BoltzEDMSamplingState:
        state = step.state
        if step.noisy_atom_coords is None or state.atom_coords_denoised is None:
            raise RuntimeError("Boltz post-denoise hooks removed required Euler state")
        state.atom_coords = edm_sampling.edm_euler_update(
            step.noisy_atom_coords,
            state.atom_coords_denoised,
            step.sigma_hat,
            step.sigma_next,
            self.config.step_scale,
        )
        return state


class BoltzDiffusionSampler(nn.Module):
    """Boltz denoiser adapter composed with the shared EDM runtime."""

    def __init__(
        self,
        config: BaseConfig,
        diffusion_module: DiffusionModule,
    ):
        """
        Args:
            config:
                Structure sampling configuration.
            diffusion_module:
                Instantiated Boltz denoising network used by ``predict``.
        """
        atom_diffusion_config = config.atom_diffusion
        super().__init__()
        self.gamma0 = atom_diffusion_config.gamma_0
        self.gamma_min = atom_diffusion_config.gamma_min
        self.noise_scale = atom_diffusion_config.noise_scale
        self.step_scale = atom_diffusion_config.step_scale
        score_model_config = config.score_model
        self.diffusion_module = diffusion_module
        self.runner = GenerativeRunner()
        self.edm_integrator_config = edm_sampling.EDMIntegratorConfig(
            gamma0=self.gamma0,
            gamma_min=self.gamma_min,
            noise_scale=self.noise_scale,
            step_scale=self.step_scale,
        )

        # parameters
        self.sigma_min = atom_diffusion_config.sigma_min
        self.sigma_max = atom_diffusion_config.sigma_max
        self.sigma_data = atom_diffusion_config.sigma_data
        self.rho = atom_diffusion_config.rho
        self.P_mean = atom_diffusion_config.P_mean
        self.P_std = atom_diffusion_config.P_std
        self.num_sampling_steps = atom_diffusion_config.num_sampling_steps
        self.coordinate_augmentation = atom_diffusion_config.coordinate_augmentation
        self.version = atom_diffusion_config.version
        self.alignment_reverse_diff = atom_diffusion_config.alignment_reverse_diff
        self.synchronize_sigmas = atom_diffusion_config.synchronize_sigmas

        self.accumulate_token_repr = atom_diffusion_config.accumulate_token_repr
        self.out_token_feat_update = None

        self.dim_fourier = score_model_config.dim_fourier
        self.token_s = score_model_config.token_s

        if self.accumulate_token_repr and self.version == "v1":
            self.out_token_feat_update = OutTokenFeatUpdate(
                sigma_data=self.sigma_data,
                token_s=self.token_s,
                dim_fourier=self.dim_fourier,
                dtype=config.torch_dtype,
                skip_create_weights=config.skip_create_weights,
            )

    @property
    def device(self) -> torch.device:
        parameter = next(self.diffusion_module.parameters(), None)
        if parameter is not None:
            return parameter.device
        return torch.device("cuda", torch.cuda.current_device())

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        t = (sigma / self.sigma_data).clamp(min=1e-20)
        return torch.log(t) * 0.25

    def load_weights(self, weights: dict):
        # Checkpoints name the denoiser ``score_model``; it is injected here as
        # ``diffusion_module``, so rename that prefix before loading.
        runtime_weights = {
            (
                "diffusion_module." + name.removeprefix("score_model.") if name.startswith("score_model.") else name
            ): value
            for name, value in weights.items()
        }
        loaded_weight = recursive_calling_load_weights(self, runtime_weights)
        # Every entry of ``weights`` must have been consumed.
        not_loaded_weights = set(runtime_weights) - loaded_weight
        if not_loaded_weights:
            raise ValueError(f"The following weights are not loaded: {not_loaded_weights}")

    def denoise(
        self,
        s_trunk: torch.Tensor,
        s_inputs: torch.Tensor,
        noised_atom_coords: torch.Tensor,
        sigma: float,
        feature_dict: dict[str, torch.Tensor],
        network_condition_kwargs: dict[str, torch.Tensor],
        multiplicity: int = 1,
        attn_metadata: AttentionMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch, multiplicity), sigma, device=device)

        padded_sigma = rearrange(sigma, "b m -> b m 1 1")
        r_noisy = self.c_in(padded_sigma) * noised_atom_coords

        # [B, mult, N_atoms, 3]
        r_update, token_a = self.diffusion_module(
            s_trunk=s_trunk,
            s_inputs=s_inputs,
            r_noisy=r_noisy,
            times=self.c_noise(sigma),
            atom_to_token=feature_dict["atom_to_token"],
            atom_pad_mask=feature_dict["atom_pad_mask"],
            token_pad_mask=feature_dict["token_pad_mask"],
            diffusion_conditioning_kwargs=network_condition_kwargs,
            attn_metadata=attn_metadata,
        )
        # token_a: [B, mult, N_tokens, dim]
        if self.version == "v2":
            # Boltz2 does not use token_a
            token_a = None

        denoised_coords = self.c_skip(padded_sigma) * noised_atom_coords + self.c_out(padded_sigma) * r_update
        return denoised_coords, token_a

    def sample_schedule(self, num_sampling_steps=None):
        # Shared AF3 schedule: Boltz uses ``num_sampling_steps`` points and
        # appends a trailing 0 (``final="append_zero"``).
        return edm_sampling.EDMScheduleConfig(
            sigma_data=self.sigma_data, s_max=self.sigma_max, s_min=self.sigma_min, rho=self.rho, final="append_zero"
        ).build(num_steps=num_sampling_steps, device=self.device, dtype=torch.float32)

    def build_denoise_hook_pipeline(
        self,
        *,
        steering_args: BoltzSteeringParams | None,
        atom_mask: torch.Tensor,
        multiplicity: int,
        num_sampling_steps: int,
        feature_dict: dict[str, torch.Tensor],
    ) -> tuple[DenoiseHookPipeline[BoltzEDMDenoiseStep], int]:
        """Construct ordered, rollout-local hooks and resolved multiplicity."""
        potential_guidance = PotentialGuidanceHook(
            steering_args=steering_args,
            atom_mask=atom_mask,
            multiplicity=multiplicity,
            step_scale=self.edm_integrator_config.step_scale,
            boltz2=self.version == "v2",
            device=self.device,
            num_sampling_steps=num_sampling_steps,
            feature_dict=feature_dict,
        )
        resolved_multiplicity = potential_guidance.multiplicity

        before_denoise_hooks = [
            potential_guidance,
            BoltzEDMChurnHook(self.edm_integrator_config.noise_scale),
        ]
        if self.coordinate_augmentation:
            before_denoise_hooks.insert(0, BoltzRandomRigidAugmentationHook(atom_mask.shape[0], resolved_multiplicity))
        after_denoise_hooks = [potential_guidance]
        if self.out_token_feat_update is not None:
            after_denoise_hooks.append(BoltzTokenRepresentationHook(self.out_token_feat_update, self.c_noise))
        if self.alignment_reverse_diff:
            after_denoise_hooks.append(BoltzRigidAlignmentHook())

        return DenoiseHookPipeline(before_denoise_hooks, after_denoise_hooks), resolved_multiplicity

    def build_edm_integrator(
        self,
        hook_pipeline: DenoiseHookPipeline[BoltzEDMDenoiseStep],
    ) -> BoltzEDMIntegrator:
        """Build the Boltz implementation of the shared EDM lifecycle."""
        return BoltzEDMIntegrator(
            config=self.edm_integrator_config,
            hook_pipeline=hook_pipeline,
        )

    @torch.no_grad()
    def forward(
        self,
        s_trunk: torch.Tensor,
        s_inputs: torch.Tensor,
        num_sampling_steps: int = None,
        multiplicity: int = 1,
        max_parallel_samples: int = None,
        steering_args: BoltzSteeringParams = None,
        network_condition_kwargs: dict[str, torch.Tensor] = None,
        feature_dict: dict[str, torch.Tensor] = None,
        attn_metadata: AttentionMetadata | None = None,
        sampling_seed: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Sample the structure from the diffusion model.
        Args:
            s_trunk(torch.Tensor):
                The trunk sequence embeddings. Shape (B, N, token_s)
            s_inputs(torch.Tensor):
                The input sequence embeddings. Shape (B, N, token_s)
            num_sampling_steps(int):
                The number of sampling steps.
            multiplicity(int):
                The multiplicity of the sampling.
            max_parallel_samples(int):
                The maximum number of parallel samples.
            steering_args(BoltzSteeringParams):
                The steering arguments.
            network_condition_kwargs(dict[str, torch.Tensor]):
                The diffusion condition tensors.
            feature_dict(dict[str, torch.Tensor]):
                The feature dictionary from the DataLoader.
        Returns:
            dict[str, torch.Tensor]:
                The sampled structure.
        """
        # Sanity check
        atom_mask = feature_dict["atom_pad_mask"]
        B, _ = atom_mask.shape

        num_sampling_steps = num_sampling_steps if num_sampling_steps is not None else self.num_sampling_steps

        hook_pipeline, multiplicity = self.build_denoise_hook_pipeline(
            steering_args=steering_args,
            atom_mask=atom_mask,
            multiplicity=multiplicity,
            num_sampling_steps=num_sampling_steps,
            feature_dict=feature_dict,
        )
        if max_parallel_samples is None:
            max_parallel_samples = multiplicity

        # [B, N_atoms] -> [B, multiplicity*num_particles, N_atoms]
        atom_mask = atom_mask.unsqueeze(1).repeat_interleave(multiplicity, 1)

        # [B, multiplicity*num_particles, N_atoms, 3]
        coords_shape = (*atom_mask.shape, 3)

        sigmas = self.sample_schedule(num_sampling_steps)
        plan = edm_sampling.EDMRolloutPlan(
            schedule=sigmas,
            coords_shape=coords_shape,
            device=self.device,
            dtype=torch.float32,
            atom_mask=atom_mask,
            augment_coordinates=False,
        )
        integrator = self.build_edm_integrator(hook_pipeline)

        # Casting dtype for score model before run the loop, this ensure for both with and without autocast modes.
        network_condition_kwargs["q"] = network_condition_kwargs["q"].to(self.diffusion_module.dtype)
        network_condition_kwargs["c"] = network_condition_kwargs["c"].to(self.diffusion_module.dtype)
        network_condition_kwargs["atom_enc_bias"] = network_condition_kwargs["atom_enc_bias"].to(
            self.diffusion_module.dtype
        )
        network_condition_kwargs["token_trans_bias"] = network_condition_kwargs["token_trans_bias"].to(
            self.diffusion_module.dtype
        )
        network_condition_kwargs["atom_dec_bias"] = network_condition_kwargs["atom_dec_bias"].to(
            self.diffusion_module.dtype
        )

        def predict(atom_coords_noisy: torch.Tensor, sigma_hat: float) -> BoltzDenoisePrediction:
            atom_coords_denoised = torch.zeros_like(atom_coords_noisy)
            token_a = None
            sample_ids = torch.arange(multiplicity, device=atom_coords_noisy.device)
            # This fixes chunking for diffusion samples in the original code.
            n_chunks = (multiplicity + max_parallel_samples - 1) // max_parallel_samples
            for sample_ids_chunk in sample_ids.chunk(n_chunks):
                denoised_chunk, token_a_chunk = self.denoise(
                    s_trunk=s_trunk,
                    s_inputs=s_inputs,
                    noised_atom_coords=atom_coords_noisy[:, sample_ids_chunk],
                    sigma=sigma_hat,
                    feature_dict=feature_dict,
                    network_condition_kwargs=network_condition_kwargs,
                    multiplicity=sample_ids_chunk.numel(),
                    attn_metadata=attn_metadata,
                )
                atom_coords_denoised[:, sample_ids_chunk] = denoised_chunk
                if token_a_chunk is not None:
                    if token_a is None:
                        token_a = torch.zeros(
                            B,
                            multiplicity,
                            *token_a_chunk.shape[2:],
                            device=self.device,
                            dtype=token_a_chunk.dtype,
                        )
                    token_a[:, sample_ids_chunk] = token_a_chunk
            return BoltzDenoisePrediction(atom_coords_denoised, token_a)

        with SamplingContext.graph_safe(self.device, sampling_seed) as sampling_context:
            state = self.runner.run(integrator, plan, predict, sampling_context).final_state
        return {"sample_atom_coords": state.atom_coords, "diff_token_repr": state.token_repr}
