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
"""ProtenixV2 model.

Wires input embedder, RPE, constraint embedder, recycling trunk, diffusion +
EDM sampling, distogram / confidence heads, and confidence summary. Building
OSS-facing ``FoldingOutput`` is deferred to the Protenix data pipeline.
"""

from functools import partial
from typing import Any

import torch
import torch.nn as nn

# isort: off
from bionemo_ir._torch.attention_backend import (
    AttentionMetadata,
    auto_select_pairwise_attention_backend,
    auto_select_triangle_attention_backend,
)
from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import CUDAGraphOptimizationTracker
from bionemo_ir._torch.layers.linear import Linear
from bionemo_ir._torch.layers.normalization import replace_with_fused_layernorm
from bionemo_ir._torch.layers.position_encoders import RelativePositionEncoder
from bionemo_ir._torch.layers.sequence_local_atom import create_gather_indices, query_to_keys_optimized
from bionemo_ir._torch.modules.protenix import (
    ProtenixConfidenceHead,
    ProtenixConfidenceSummary,
    ProtenixConstraintEmbedder,
    ProtenixDiffusionModule,
    ProtenixDistogramHead,
    ProtenixDiffusionSampler,
    ProtenixInputFeatureEmbedder,
    ProtenixTrunk,
)
from bionemo_ir.configs import BaseConfig
from bionemo_ir.hubs import FoldingSupportMatrix as SupMat
from bionemo_ir.hubs import load_weights as load_weights_from_hubs

from ..optimize_module_setter import AcceleratedConfig, DiscoveredModuleRegistry, OptimizedModuleSetterMixin
from .config import PRETRAINED_CONFIG_REGISTRY
from .convert import (
    convert_confidence_head_torch,
    convert_constraint_embedder_torch,
    convert_diffusion_module_torch,
    convert_distogram_head_torch,
    convert_hf_input_embedder_torch,
    convert_input_projections_torch,
    convert_relative_position_encoding_torch,
    convert_trunk_torch,
)
# isort: on


def _unbatch(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Drop leading batch dim when present (OSS unbatched layout, B=1)."""
    return x[0] if x.dim() > ndim else x


def _configure_inference_precision(config: BaseConfig) -> BaseConfig:
    """Set trunk / diffusion / confidence backends and dtypes for inference."""
    tri_backend = auto_select_triangle_attention_backend(torch.bfloat16)
    pair_backend = auto_select_pairwise_attention_backend(torch.bfloat16)
    config.trunk_config.set_triangle_attention_backend(tri_backend)
    config.trunk_config.set_pairwise_attention_backend(pair_backend)

    dm = config.diffusion_module_config
    dm.token_transformer_config.set_dtype("bfloat16")
    dm.token_transformer_config.set_pairwise_attention_backend(pair_backend)
    # Atom encoder/decoder keep SDPA (windowed layout incompatible with CuTeDSL).
    dm.atom_encoder_config.set_dtype("float32")
    dm.atom_encoder_config.set_pairwise_attention_backend("SDPA")
    dm.atom_decoder_config.set_dtype("float32")
    dm.atom_decoder_config.set_pairwise_attention_backend("SDPA")

    ch_pf = config.confidence_head_config.pairformer_config
    ch_pf.set_dtype("bfloat16")
    ch_pf.set_triangle_attention_backend(tri_backend)
    ch_pf.set_pairwise_attention_backend(pair_backend)
    return config


class Protenix(nn.Module, OptimizedModuleSetterMixin):
    """ProtenixV2 model — input embedder + trunk + diffusion + heads."""

    # Whitelist gating modules discover with ``@support_graph_optimization``
    #
    # WARNING: do NOT graph-optimize the recycling-trunk pairformer
    # (``trunk.pairformer_stack``) — its CUDA-graph replay produces NaN. It is a
    # ``PairformerModule``, so ``@support_graph_optimization`` marks it and
    # generic discovery *sees* it as a candidate by qualified path; it is left
    # out of the aliases below on purpose and must never be given an
    # ``accelerated_configs`` entry (by role or by the ``trunk.pairformer_stack``
    # path). The same applies to ``confidence_head.pairformer_stack``.
    GRAPH_OPT_ENABLED_MODULES = {
        "token_transformer": "diffusion_sampler.diffusion_module.diffusion_transformer",
        "diffusion_module": "diffusion_sampler.diffusion_module",
    }

    def get_optimized_modules(self, accelerated_configs: dict[str, AcceleratedConfig]) -> DiscoveredModuleRegistry:
        return DiscoveredModuleRegistry(
            self,
            accelerated_configs,
            role_aliases=self.GRAPH_OPT_ENABLED_MODULES,
            graph_optimization_cls=CUDAGraphOptimizationTracker,
        )

    def __init__(
        self, config: BaseConfig = None, model_name: str | None = None, include_load_weights: bool = False
    ) -> None:
        super().__init__()
        self.model_name = model_name or SupMat.ProtenixV2
        self.config = config or self.get_pretrained_config(self.model_name)
        self.dtype = self.config.torch_dtype
        self.n_queries = self.config.n_queries
        self.n_keys = self.config.n_keys

        self.input_embedder = ProtenixInputFeatureEmbedder(self.config.input_embedder_config)
        rpe_config = self.config.relative_position_encoding_config
        self.relative_position_encoding = RelativePositionEncoder(
            token_z=rpe_config.c_z,
            r_max=rpe_config.r_max,
            s_max=rpe_config.s_max,
            fix_sym_check=rpe_config.fix_sym_check,
            cyclic_pos_enc=rpe_config.cyclic_pos_enc,
            dtype=self.dtype,
            skip_create_weights=self.config.skip_create_weights,
        )
        self.constraint_embedder = ProtenixConstraintEmbedder(self.config.constraint_embedder_config)

        # Single / pair initialization projections (AF3 Alg. 1 lines 2-6).
        c_s, c_z, c_s_inputs = (self.config.c_s, self.config.c_z, self.config.c_s_inputs)
        self.linear_no_bias_sinit = self._proj(c_s_inputs, c_s)
        self.linear_no_bias_zinit1 = self._proj(c_s, c_z)
        self.linear_no_bias_zinit2 = self._proj(c_s, c_z)
        self.linear_no_bias_token_bond = self._proj(1, c_z)

        self.trunk = ProtenixTrunk(self.config.trunk_config)

        sd = self.config.edm_sampling_config
        self.diffusion_sampler = ProtenixDiffusionSampler(
            ProtenixDiffusionModule(self.config.diffusion_module_config),
            gamma0=sd.gamma0,
            gamma_min=sd.gamma_min,
            noise_scale=sd.noise_scale_lambda,
            step_scale=sd.step_scale_eta,
            s_max=sd.s_max,
            s_min=sd.s_min,
            rho=sd.rho,
            n_step=sd.n_step,
            use_cache=sd.enable_diffusion_shared_vars_cache,
        )

        # Distogram head runs in fp32 (OSS disables autocast).
        dh = self.config.distogram_head_config
        self.distogram_head = ProtenixDistogramHead(
            c_z=dh.c_z, no_bins=dh.no_bins, dtype=dh.torch_dtype, skip_create_weights=self.config.skip_create_weights
        )

        self.confidence_head = ProtenixConfidenceHead(self.config.confidence_head_config)
        self.confidence_summary = ProtenixConfidenceSummary(self.config.confidence_summary_config)

        if include_load_weights:
            self.load_weights()
        replace_with_fused_layernorm(self)

    def _proj(self, c_in: int, c_out: int):
        return Linear(c_in, c_out, bias=False, dtype=self.dtype, skip_create_weights=self.config.skip_create_weights)

    def get_pretrained_config(self, model_name: str = SupMat.ProtenixV2) -> BaseConfig:
        config_class = PRETRAINED_CONFIG_REGISTRY.get(model_name)
        if config_class is None:
            raise ValueError(f"Protenix pretrained config not found for model name: {model_name}")
        return _configure_inference_precision(config_class())

    def generate_attn_metadata(self, batch: dict[str, torch.Tensor]) -> AttentionMetadata:
        """Build atom-attention metadata (query→keys gather).

        Windows atoms into ``K = ceil(N_atom / n_queries)`` blocks of size
        ``W=n_queries`` with key span ``H=n_keys`` — matching ``p_lm`` /
        ``d_lm`` (unlike OpenFold3, which appends a trailing block when
        ``N_atom`` is an exact multiple of ``n_queries``).
        """
        num_atoms = batch["ref_pos"].shape[-2]
        W, H = self.n_queries, self.n_keys
        K = (num_atoms + W - 1) // W
        device = batch["ref_pos"].device
        gather_indices, _ = create_gather_indices(K, W, H, device)
        query_to_keys_func = partial(query_to_keys_optimized, gather_indices=gather_indices, W=W, H=H)
        return AttentionMetadata(query_to_keys=query_to_keys_func, bias_cache={})

    def load_weights(self, weights: dict = None) -> None:
        """Load ported-module weights from a protenix-v2 checkpoint (or hub)."""
        if weights is None:
            weights = load_weights_from_hubs(name=self.model_name)

        def _strict(module: nn.Module, state: dict) -> None:
            module.load_state_dict(state, strict=True)

        _strict(
            self.input_embedder,
            convert_hf_input_embedder_torch(self.config.input_embedder_config, weights, prefix="input_embedder"),
        )
        _strict(
            self.relative_position_encoding,
            convert_relative_position_encoding_torch(
                self.config.relative_position_encoding_config, weights, prefix="relative_position_encoding"
            ),
        )
        _strict(
            self.constraint_embedder,
            convert_constraint_embedder_torch(
                self.config.constraint_embedder_config, weights, prefix="constraint_embedder"
            ),
        )
        for name, weight in convert_input_projections_torch(self.config, weights).items():
            getattr(self, name).load_state_dict({"weight": weight})
        _strict(self.trunk, convert_trunk_torch(self.config.trunk_config, weights))
        _strict(
            self.diffusion_sampler.diffusion_module,
            convert_diffusion_module_torch(self.config.diffusion_module_config, weights, prefix="diffusion_module"),
        )
        _strict(
            self.distogram_head,
            convert_distogram_head_torch(self.config.distogram_head_config, weights, prefix="distogram_head"),
        )
        _strict(
            self.confidence_head,
            convert_confidence_head_torch(self.config.confidence_head_config, weights, prefix="confidence_head"),
        )

    def _relative_position_encoding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Project precomputed or generated ``relp`` into pair channels."""
        relp = batch.get("relp")
        if relp is None:
            relp = self.relative_position_encoding.generate_relp(
                asym_id=batch["asym_id"],
                residue_index=batch["residue_index"],
                entity_id=batch["entity_id"],
                token_index=batch["token_index"],
                sym_id=batch["sym_id"],
            )
        return self.relative_position_encoding(relp=relp)

    def _ensure_relp(self, batch: dict[str, torch.Tensor], drop_features) -> None:
        """Generate ``relp`` if missing; drop residue/entity/token/sym indices."""
        if "relp" not in batch:
            batch["relp"] = self.relative_position_encoding.generate_relp(
                asym_id=batch["asym_id"],
                residue_index=batch["residue_index"],
                entity_id=batch["entity_id"],
                token_index=batch["token_index"],
                sym_id=batch["sym_id"],
            )
        # relp now owns all downstream relative-position information.
        drop_features("residue_index", "entity_id", "token_index", "sym_id")

    def _init_pair_state(
        self, s_inputs: torch.Tensor, batch: dict[str, torch.Tensor], drop_features
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """AF3 Alg. 1 single/pair init with in-place RPE/token_bond/constraint."""
        s_init = self.linear_no_bias_sinit(s_inputs)
        pair_state_dtype = self.trunk.pair_state_dtype
        z_init = self.linear_no_bias_zinit1(s_init).unsqueeze(-2) + self.linear_no_bias_zinit2(s_init).unsqueeze(-3)
        # Freshly owned z_init: accumulate pair contributions in place in fp32
        # to avoid retaining old + contribution + new pair-sized tensors.
        z_init.add_(self._relative_position_encoding(batch))
        token_bonds = batch.get("token_bonds")
        if token_bonds is not None:
            z_init.add_(self.linear_no_bias_token_bond(token_bonds.unsqueeze(-1).to(self.dtype)))
        del token_bonds
        if "constraint_feature" in batch:
            z_constraint = self.constraint_embedder(batch["constraint_feature"])
            if z_constraint is not None:
                z_init.add_(z_constraint)
            del z_constraint
        drop_features("token_bonds", "constraint_feature")
        z_init = z_init.to(pair_state_dtype)
        return s_init, z_init

    def _streaming_confidence(
        self,
        batch: dict[str, torch.Tensor],
        out: dict[str, Any],
        held: dict[str, torch.Tensor],
        coordinate: torch.Tensor,
        num_cycles: int,
        compact_output: bool,
        return_full_data: bool,
    ) -> None:
        """RAII confidence: one sample at a time so N_sample logit stacks never all resident.

        ``held`` owns ``s_inputs`` / ``s`` / ``z`` / ``distogram_logits``. When
        ``compact_output``, each entry is popped as soon as its last consumer
        has run, so the tensor is freed at the earliest possible point.
        """
        head, summ = self.confidence_head, self.confidence_summary
        # Reduce and release the raw distogram before constructing the
        # confidence pair state.
        contact_probs = summ.contact_probs(_unbatch(held["distogram_logits"], 3))
        if compact_output:
            del held["distogram_logits"]

        asym_id = _unbatch(batch["asym_id"], 1).long()
        atom_to_token_idx = _unbatch(batch["atom_to_token_idx"], 1).long()
        is_polymer = 1 - _unbatch(batch["is_ligand"], 1)
        has_frame = _unbatch(batch["has_frame"], 1)
        token_is_ligand = summ.token_is_ligand(asym_id, atom_to_token_idx, is_polymer)
        ctx = head.prepare(batch, held["s_inputs"], held["s"], held["z"], pair_mask=batch.get("pair_mask"))
        if compact_output:
            held.pop("s_inputs", None)
            held.pop("s", None)
            held.pop("z", None)

        # Remaining confidence inputs held by locals / ctx. Clearing the
        # working feature shell releases one-shot caller tensors when
        # consume_input_features is enabled.
        batch.clear()
        coordinate_ub = _unbatch(coordinate, 3)  # [N_sample, N_atom, 3]

        summary_list = []
        full_list = [] if return_full_data else None
        for i in range(coordinate_ub.shape[0]):
            plddt_i, pae_i, pde_i, _resolved_i = head.per_sample_logits(ctx, coordinate[..., i, :, :])
            summary_i, full_i = summ.summary_one_sample(
                contact_probs,
                _unbatch(plddt_i, 2),
                _unbatch(pae_i, 3),
                _unbatch(pde_i, 3),
                coordinate_ub[i],
                asym_id,
                has_frame,
                atom_to_token_idx,
                is_polymer,
                token_is_ligand,
                num_cycles,
                return_full_data=return_full_data,
            )
            summary_list.append(summary_i)
            if full_list is not None:
                full_list.append(full_i)
            # Do not carry one sample's [N,N,b] logits into the next invocation.
            del plddt_i, pae_i, pde_i, _resolved_i, full_i
        del ctx

        out["coordinate"] = coordinate_ub
        out["summary_confidence"] = summary_list
        if not compact_output:
            out["contact_probs"] = contact_probs
        if full_list is not None:
            out["full_data"] = full_list

    def sample_diffusion(
        self,
        batch: dict[str, torch.Tensor],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
        num_sampling_steps: int | None = None,
        diffusion_samples: int = 1,
        consume_input_features: bool = False,
        sampling_seed: int | None = None,
    ) -> torch.Tensor:
        """Run EDM sampling (AF3 Alg. 18); trunk embeddings cast to fp32.

        Args:
            s_inputs: ``[B, N_token, c_s_inputs]``
            s_trunk: ``[B, N_token, c_s]``
            z_trunk: ``[B, N_token, N_token, c_z]``
            diffusion_samples: sample count ``N_sample`` / ``S``

        Returns:
            ``[B, diffusion_samples, N_atom, 3]`` predicted coordinates
        """
        return self.diffusion_sampler.sample_coords(
            batch,
            s_inputs.float(),
            s_trunk.float(),
            z_trunk.float(),
            num_sampling_steps=num_sampling_steps,
            N_sample=diffusion_samples,
            attn_metadata=attn_metadata,
            drop_consumed_relp=True,
            drop_consumed_features=consume_input_features,
            seed=sampling_seed,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        recycling_steps: int = 3,
        num_sampling_steps: int | None = 200,
        diffusion_samples: int = 1,
        compact_output: bool = False,
        return_full_data: bool = True,
        consume_input_features: bool = False,
        sampling_seed: int | None = None,
    ) -> dict[str, Any]:
        """Trunk + diffusion + optional confidence (Boltz-style runtime args).

        ``recycling_steps`` → ``num_cycles = recycling_steps + 1``. Destructive
        ``consume_input_features`` pops tensors after final use and clears
        ``batch`` before return; default preserves the caller's dict via a
        shallow shell copy.

        Args:
            batch: feature dict. Typical token/atom keys include
                ``restype`` / ``profile`` / ``deletion_mean``
                ``[..., N_token, *]``, ``msa`` ``[..., N_msa, N_token]``,
                ``ref_pos`` ``[..., N_atom, 3]``, ``atom_to_token_idx``
                ``[..., N_atom]``, ``d_lm`` / ``v_lm`` / ``pad_info``
                (windowed ``K×W×H``), ``relp`` or RPE index fields, and
                optional template / confidence masks.
            recycling_steps: trunk recycle count (cycles = this + 1)
            num_sampling_steps: EDM steps (``None`` → config default)
            diffusion_samples: ``N_sample`` / ``S``
            compact_output: drop trunk tensors from the return dict
            return_full_data: include per-sample ``full_data`` when summarizing

        Returns:
            Full mode (``compact_output=False``)::

                s_inputs ``[B, N_token, c_s_inputs]``,
                s ``[B, N_token, c_s]``,
                z ``[B, N_token, N_token, c_z]``,
                coordinate ``[B, N_sample, N_atom, 3]``,
                distogram_logits ``[B, N_token, N_token, no_bins]``

            Compact / summary path: ``coordinate`` is unbatched to
            ``[N_sample, N_atom, 3]``; adds ``summary_confidence`` (list of
            per-sample dicts) and optionally ``full_data`` / ``contact_probs``.
            Confidence-only (no summary masks): also ``plddt_logits`` /
            ``pae_logits`` / ``pde_logits`` / ``resolved_logits``.
        """
        num_cycles = recycling_steps + 1

        # Precompute relp once (shared by trunk RPE and diffusion conditioning).
        # Destructive mode takes ownership; otherwise shallow-copy the shell.
        batch = batch if consume_input_features else {**batch}
        attn_metadata = self.generate_attn_metadata(batch)

        def _drop_features(*names: str) -> None:
            for name in names:
                batch.pop(name, None)

        self._ensure_relp(batch, _drop_features)

        s_inputs = self.input_embedder(batch, attn_metadata=attn_metadata)
        _drop_features("restype", "profile", "deletion_mean", "esm_token_embedding")

        s_init, z_init = self._init_pair_state(s_inputs, batch, _drop_features)

        s, z = self.trunk(batch, s_inputs, s_init, z_init, num_cycles=num_cycles)
        del s_init, z_init
        _drop_features(
            "msa",
            "has_deletion",
            "deletion_value",
            "template_distogram",
            "template_pseudo_beta_mask",
            "template_aatype",
            "template_unit_vector",
            "template_backbone_frame_mask",
        )

        distogram_logits = self.distogram_head(z)

        coordinate = self.sample_diffusion(
            batch,
            s_inputs,
            s,
            z,
            attn_metadata=attn_metadata,
            num_sampling_steps=num_sampling_steps,
            diffusion_samples=diffusion_samples,
            consume_input_features=consume_input_features,
            sampling_seed=sampling_seed,
        )
        del attn_metadata
        _drop_features(
            "relp",
            "ref_pos",
            "ref_charge",
            "ref_mask",
            "ref_atom_name_chars",
            "ref_element",
            "d_lm",
            "v_lm",
            "pad_info",
        )

        if compact_output:
            out = {"coordinate": coordinate}
        else:
            out = {
                "s_inputs": s_inputs,
                "s": s,
                "z": z,
                "coordinate": coordinate,
                "distogram_logits": distogram_logits,
            }

        have_conf = "distogram_rep_atom_mask" in batch and "atom_to_tokatom_idx" in batch
        have_summary = have_conf and "has_frame" in batch and "is_ligand" in batch
        if have_summary:
            # Transfer trunk/distogram refs into held so compact mode can drop
            # forward locals at the same pop boundaries as the original dels.
            held = {
                "s_inputs": s_inputs,
                "s": s,
                "z": z,
                "distogram_logits": distogram_logits,
            }
            if compact_output:
                del s_inputs, s, z, distogram_logits
            self._streaming_confidence(batch, out, held, coordinate, num_cycles, compact_output, return_full_data)
        elif have_conf and not compact_output:
            out.update(self.confidence_head(batch, s_inputs, s, z, coordinate, pair_mask=batch.get("pair_mask")))
        batch.clear()
        return out
