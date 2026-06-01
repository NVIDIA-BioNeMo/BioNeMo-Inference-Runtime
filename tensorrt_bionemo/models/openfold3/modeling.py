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

from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from tensorrt_bionemo._torch.attention_backend import (
    AttentionMetadata, auto_select_pairwise_attention_backend,
    auto_select_triangle_attention_backend)
from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo._torch.layers.sequence_local_atom import (
    create_gather_indices, create_indexing_matrix, query_to_keys_optimized)
from tensorrt_bionemo._torch.layers.transformers.pairformer import \
    PairformerModule
from tensorrt_bionemo._torch.modules.openfold3.confidence import \
    AuxiliaryHeadsAllAtom
from tensorrt_bionemo._torch.modules.openfold3.diffusion_module import (
    DiffusionModule, SampleDiffusion, create_noise_schedule)
from tensorrt_bionemo._torch.modules.openfold3.embedders import (
    InputEmbedderAllAtom, MSAModuleEmbedder, TemplateEmbedderAllAtom)
from tensorrt_bionemo._torch.modules.openfold3.trunk import MSAModuleStack
from tensorrt_bionemo._torch.tensor_utils import tensor_tree_map
from tensorrt_bionemo._trt.module_wrappers import (PairformerTRT,
                                                   TokenTransformerTRT)
from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.hubs import load_weights as load_weights_from_hubs
from tensorrt_bionemo.models.openfold3.config import PRETRAINED_CONFIG_REGISTRY
from tensorrt_bionemo.models.openfold3.convert import \
    convert_hf_openfold3_torch
from tensorrt_bionemo.registry import SupMat

from ..helper import (AcceleratedConfig, ModuleRegistry, ModuleSpec,
                      OptimizedModuleSetterMixin)


class OpenFold3ModuleRegistry(ModuleRegistry):

    def get_accelerated_modules(self) -> dict[str, ModuleSpec]:
        return {
            "pairformer":
            ModuleSpec(
                getter=lambda mod: mod.pairformer_stack,
                setter=lambda mod, opt: setattr(mod, "pairformer_stack", opt),
                trt_cls=PairformerTRT,
                compiled_cls=None,
            ),
            "token_transformer":
            ModuleSpec(
                getter=lambda mod:
                (mod.sample_diffusion.diffusion_module.diffusion_transformer),
                setter=lambda mod, opt: setattr(
                    mod.sample_diffusion.diffusion_module,
                    "diffusion_transformer", opt),
                trt_cls=TokenTransformerTRT,
                compiled_cls=None,
            ),
        }


class OpenFold3(nn.Module, OptimizedModuleSetterMixin):

    def get_optimized_modules(
        self, accelerated_configs: dict[str, AcceleratedConfig]
    ) -> OpenFold3ModuleRegistry:
        return OpenFold3ModuleRegistry(accelerated_configs)

    def __init__(self,
                 config: BaseConfig = None,
                 include_load_weights: bool = True,
                 model_name: Optional[str] = None,
                 diffusion_samples: Optional[int] = None):
        super().__init__()
        self.model_name = model_name or SupMat.OpenFold3
        self.config = config or self.get_pretrained_config(self.model_name)
        self.dtype = self.config.torch_dtype
        self.mapping = self.config.mapping
        self.skip_create_weights = self.config.skip_create_weights
        self.num_recycles = self.config.num_recycles
        self.num_cycles = self.num_recycles + 1
        self.n_query = self.config.n_query
        self.n_key = self.config.n_key
        self.no_rollout_steps = self.config.no_rollout_steps
        self.no_rollout_samples = self.config.no_rollout_samples if diffusion_samples is None else diffusion_samples

        self.noise_schedule = self.config.noise_schedule_config

        self.input_embedder = InputEmbedderAllAtom(
            config=self.config.input_embedder_config)
        self.layer_norm_z = nn.LayerNorm(self.config.c_z,
                                         dtype=self.config.torch_dtype,
                                         eps=self.config.norm_epsilon)
        self.linear_z = Linear(
            self.config.c_z,
            self.config.c_z,
            bias=False,
            dtype=self.config.torch_dtype,
            mapping=self.config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.config.skip_create_weights)

        self.template_embedder = TemplateEmbedderAllAtom(
            config=self.config.template_embedder_config)
        self.msa_module_embedder = MSAModuleEmbedder(
            config=self.config.msa_module_embedder_config)
        self.msa_module = MSAModuleStack(
            config=self.config.msa_stack_module_config)
        self.layer_norm_s = nn.LayerNorm(self.config.c_s,
                                         dtype=self.config.torch_dtype,
                                         eps=self.config.norm_epsilon)
        self.linear_s = Linear(
            self.config.c_s,
            self.config.c_s,
            bias=False,
            dtype=self.config.torch_dtype,
            mapping=self.config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=self.config.skip_create_weights)

        self.pairformer_stack = PairformerModule(
            config=self.config.trunk.pairformer)
        self.diffusion_module = DiffusionModule(
            config=self.config.diffusion_module_config)
        self.sample_diffusion = SampleDiffusion(
            config=self.config.sample_diffusion_config,
            diffusion_module=self.diffusion_module)
        self.aux_heads = AuxiliaryHeadsAllAtom(
            config=self.config.auxiliary_heads_config)

        if include_load_weights:
            self.load_weights()

    def generate_attn_metadata(self, batch: dict[str, torch.Tensor]):
        num_atoms = batch["atom_mask"].shape[-1]
        K = (num_atoms + (self.n_query -
                          (num_atoms % self.n_query))) // self.n_query
        W = self.n_query
        H = self.n_key
        device = batch["atom_mask"].device
        # keys_indexing_matrix is retained for backward compatibility with
        # code paths that may consult it; the actual query-to-keys op uses
        # the bit-exact gather path below.
        self.keys_indexing_matrix = create_indexing_matrix(K, W, H, device)
        gather_indices, _ = create_gather_indices(K, W, H, device)

        # Single OSS-equivalent zero-pad query→keys callable, pre-bound to
        # ``gather_indices``, ``W``, ``H``. Consumed by both atom attention
        # and ``convert_pair_atom_to_blocks``. Mirrors the Boltz-1/2 pattern.
        query_to_keys_func = partial(query_to_keys_optimized,
                                     gather_indices=gather_indices,
                                     W=W,
                                     H=H)
        return AttentionMetadata(query_to_keys=query_to_keys_func,
                                 bias_cache={})

    def get_pretrained_config(self,
                              model_name: str = SupMat.OpenFold3
                              ) -> BaseConfig:
        config_class = PRETRAINED_CONFIG_REGISTRY.get(model_name)
        if config_class is None:
            raise ValueError(
                f"OpenFold3 pretrained config not found for model name: {model_name}"
            )
        config = config_class()
        tri_backend = auto_select_triangle_attention_backend(torch.bfloat16)
        pair_backend = auto_select_pairwise_attention_backend(torch.bfloat16)
        config.msa_stack_module_config.set_triangle_attention_backend(
            tri_backend)
        config.template_embedder_config.set_triangle_attention_backend(
            tri_backend)

        config.trunk.pairformer.set_dtype("bfloat16")
        config.trunk.pairformer.s_path_dtype = torch.bfloat16
        config.trunk.set_triangle_attention_backend(tri_backend)
        config.trunk.set_pairwise_attention_backend(pair_backend)
        config.diffusion_module_config.diffusion_transformer_config.set_dtype(
            "bfloat16")
        config.diffusion_module_config.diffusion_transformer_config.token_transformer.set_pairwise_attention_backend(
            pair_backend)

        config.msa_stack_module_config.set_dtype("bfloat16")
        config.template_embedder_config.template_pair_stack.set_dtype(
            "bfloat16")

        # config.diffusion_module_config.set_dtype("bfloat16")
        # config.diffusion_module_config.diffusion_transformer_config.token_transformer.set_dtype("bfloat16")

        config.auxiliary_heads_config.pairformer.set_triangle_attention_backend(
            tri_backend)
        config.auxiliary_heads_config.pairformer.set_dtype("bfloat16")
        config.auxiliary_heads_config.pairformer.s_path_dtype = torch.bfloat16
        config.auxiliary_heads_config.pairformer.set_pairwise_attention_backend(
            pair_backend)
        config.auxiliary_heads_config.set_dtype("bfloat16")

        return config

    def load_weights(self, weights: dict = None):
        if weights is None:
            weights = load_weights_from_hubs(name=self.model_name)

        openfold3_weights = convert_hf_openfold3_torch(
            config=self.config, weights=weights, model_name=self.model_name)

        self.layer_norm_z.load_state_dict(openfold3_weights["layer_norm_z"])
        self.linear_z.load_weights(openfold3_weights["linear_z"])
        self.layer_norm_s.load_state_dict(openfold3_weights["layer_norm_s"])
        self.linear_s.load_weights(openfold3_weights["linear_s"])

        self.input_embedder.load_weights(openfold3_weights["input_embedder"])
        self.template_embedder.load_weights(
            openfold3_weights["template_embedder"])
        self.msa_module.load_weights(openfold3_weights["msa_stack"])
        self.msa_module_embedder.load_weights(
            openfold3_weights["msa_module_embedder"])
        self.diffusion_module.load_weights(
            openfold3_weights["diffusion_module"])
        self.pairformer_stack.load_weights(
            openfold3_weights["pairformer_stack"])
        self.aux_heads.load_weights(openfold3_weights["auxiliary_heads"])

    def feature_extraction(
        self,
        batch: dict,
        num_cycles: int,
        attn_metadata: dict = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Implements Algorithm 1 lines 1-14.

        Args:
            batch:
                Input feature dictionary
            num_cycles:
                Number of cycles to run

        Returns:
            s_input:
                [*, N_token, C_s_input] Single (input) representation
            s:
                [*, N_token, C_s] Single representation
            z:
                [*, N_token, N_token, C_z] Pair representation
        """

        s_input, s_init, z_init = self.input_embedder(
            batch=batch, attn_metadata=attn_metadata)
        # s: [*, N_token, C_s]
        # z: [*, N_token, N_token, C_z]
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        # token_mask: [*, N_token]
        # pair_mask: [*, N_token, N_token]
        token_mask = batch["token_mask"]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        for cycle_no in range(num_cycles):
            is_final_iter = cycle_no == (num_cycles - 1)

            # [*, N_token, N_token, C_z]
            z = z_init + self.linear_z(self.layer_norm_z(z))

            z = z + self.template_embedder(
                batch=batch, z=z, pair_mask=pair_mask)

            m, msa_mask = self.msa_module_embedder(batch=batch,
                                                   s_input=s_input)

            # Run MSA + pair embeddings through the MsaModule
            # m: [*, N_seq, N_token, C_m]
            # z: [*, N_token, N_token, C_z]

            z = self.msa_module(m,
                                z,
                                msa_mask=msa_mask.to(dtype=m.dtype),
                                pair_mask=pair_mask.to(dtype=z.dtype))

            s = s_init + self.linear_s(self.layer_norm_s(s))

            pairformer_dtype = self.config.trunk.pairformer.torch_dtype
            s, z = self.pairformer_stack(
                s=s.to(dtype=pairformer_dtype),
                z=z.to(dtype=pairformer_dtype),
                mask=token_mask.to(dtype=pairformer_dtype),
                pair_mask=pair_mask.to(dtype=pairformer_dtype))

            if pairformer_dtype != torch.float32:
                s = s.float()
                z = z.float()

        return s_input, s, z

    def prediction(
        self,
        batch: dict,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        attn_metadata: dict = None,
        no_rollout_steps: Optional[int] = None,
        no_rollout_samples: Optional[int] = None,
    ) -> dict:
        """
        Mini diffusion rollout described in section 4.1.
        Implements Algorithm 1 lines 15-18.

        Args:
            batch:
                Input feature dictionary
            si_input:
                [*, N_token, C_s_input] Single (input) representation
            si_trunk:
                [*, N_token, C_s] Single representation output from model trunk
            zij_trunk:
                [*, N_token, N_token, C_z] Pair representation output from model trunk

        Returns:
            Output dictionary containing the predicted trunk embeddings,
            all-atom positions, and confidence/distogram head logits
        """
        # Check this again for accuracy
        # Compute atom positions
        # with (
        #         torch.no_grad(),
        #         torch.amp.autocast(device_type="cuda", dtype=torch.float32),
        # ):
        no_rollout_steps_eff = (no_rollout_steps if no_rollout_steps
                                is not None else self.no_rollout_steps)
        no_rollout_samples_eff = (no_rollout_samples if no_rollout_samples
                                  is not None else self.no_rollout_samples)

        noise_schedule = create_noise_schedule(
            no_rollout_steps=no_rollout_steps_eff,
            p=self.noise_schedule.p,
            sigma_data=self.noise_schedule.sigma_data,
            s_max=self.noise_schedule.s_max,
            s_min=self.noise_schedule.s_min,
            dtype=si_input.dtype,
            device=si_input.device,
        )

        atom_positions_predicted = self.sample_diffusion(
            batch=batch,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            noise_schedule=noise_schedule,
            no_rollout_samples=no_rollout_samples_eff,
            attn_metadata=attn_metadata,
        )

        output = {
            "si_trunk": si_trunk,
            "zij_trunk": zij_trunk,
            "atom_positions_predicted": atom_positions_predicted,
        }

        aux_heads_output = self.aux_heads(batch=batch,
                                          si_input=si_input,
                                          output=output)

        output.update(aux_heads_output)

        return output

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        recycling_steps: int = 3,
        num_sampling_steps: Optional[int] = 200,
        diffusion_samples: int = 1,
    ):
        # Boltz-style runtime args (mirrors ``Boltz2.forward`` signature so the
        # generic ``FoldingEngine`` can pass the same ``runtime_args`` dict to
        # either model). Mapping to OpenFold3 internals:
        #   recycling_steps     → num_cycles = recycling_steps + 1
        #   num_sampling_steps  → no_rollout_steps   (diffusion rollout length)
        #   diffusion_samples   → no_rollout_samples (parallel rollout samples)
        num_cycles = recycling_steps + 1

        attn_metadata = self.generate_attn_metadata(batch)

        si_input, si_trunk, zij_trunk = self.feature_extraction(
            batch=batch, num_cycles=num_cycles, attn_metadata=attn_metadata)

        # Expand sampling dimension for rollout and diffusion
        si_input = si_input.unsqueeze(1)
        si_trunk = si_trunk.unsqueeze(1)
        zij_trunk = zij_trunk.unsqueeze(1)

        batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)

        output = self.prediction(
            batch=batch,
            si_input=si_input.to(dtype=self.diffusion_module.dtype),
            si_trunk=si_trunk.to(dtype=self.diffusion_module.dtype),
            zij_trunk=zij_trunk.to(dtype=self.diffusion_module.dtype),
            attn_metadata=attn_metadata,
            no_rollout_steps=num_sampling_steps,
            no_rollout_samples=diffusion_samples,
        )

        return output
