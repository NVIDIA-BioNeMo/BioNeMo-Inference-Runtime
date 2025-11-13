# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, Optional

import torch
import torch.nn as nn
from tensorrt_llm.functional import AllReduceParams
from tensorrt_llm.logger import logger

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.layers.distogram import DistogramModule
from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo._torch.layers.position_encoders import \
    RelativePositionEncoder
from tensorrt_bionemo._torch.layers.sequence_local_atom import (
    create_indexing_matrix, query_to_keys)
from tensorrt_bionemo._torch.modules.boltz.embedders import Boltz1InputEmbedder
from tensorrt_bionemo._torch.modules.boltz.physical.steering import \
    BoltzSteeringParams
from tensorrt_bionemo._torch.modules.boltz.structure import (
    AtomDiffusion, DiffusionConditioning)
from tensorrt_bionemo._torch.modules.boltz.trunk import Trunk
from tensorrt_bionemo.hubs import load_weights as load_weights_from_hubs
from tensorrt_bionemo.runtime import BaseContextMemoryManager

from ..helper import AcceleratedModules, build_optimized_module
from .configs import Boltz1Config
from .const import NUM_POCKET_CONTACT_INFO, NUM_TOKENS
from .convert import (convert_hf_diffusion_conditioning_torch,
                      convert_hf_diffusion_transformer_torch,
                      convert_hf_input_embedder_torch,
                      convert_hf_msa_module_torch, convert_hf_pairformer_torch,
                      convert_hf_structure_module_torch)
from .modules import (MSAModuleBackendBuilder, PairformerBackendBuilder,
                      TokenTransformerBackendBuilder)


class Boltz1AcceleratedModules(AcceleratedModules):

    def get_supported_module_names(self):
        return [
            "structure_pairformer", "confidence_pairformer",
            "token_transformer", "msa_module"
        ]


class Boltz1(nn.Module):

    def __init__(self, config: Boltz1Config = None):
        super().__init__()
        self.model_name = "boltz-1"
        self.config = config or Boltz1Config.from_pretrained()
        self.global_config = self.config.global_config

        # Setup for input embedder
        self.input_embedder_dtype = self.config.input_embedder_config.torch_dtype
        self.input_embedder_mapping = self.config.input_embedder_config.mapping
        self.input_embedder_config = self.config.input_embedder_config

        # Setup for trunk
        self.trunk_mapping = self.config.trunk_config.mapping
        self.trunk_dtype = self.config.trunk_config.torch_dtype
        self.trunk_config = self.config.trunk_config

        # Setup for atom diffusion
        self.structure_module_dtype = self.config.structure_module_config.torch_dtype
        self.structure_module_mapping = self.config.structure_module_config.mapping
        self.structure_module_config = self.config.structure_module_config

        # Setup steering params:
        self.steering_args = BoltzSteeringParams(contact_guidance_update=False)

        #### Build up modules ####

        ### Input embedder ###
        self.input_embedder = Boltz1InputEmbedder(self.input_embedder_config)

        ### Input projections ###
        s_input_dim = (self.global_config.token_s + 2 * NUM_TOKENS + 1 +
                       NUM_POCKET_CONTACT_INFO)
        self.s_init = Linear(s_input_dim,
                             self.global_config.token_s,
                             bias=False,
                             dtype=self.input_embedder_dtype,
                             mapping=self.input_embedder_mapping,
                             tensor_parallel_mode=TensorParallelMode.COLUMN,
                             gather_output=True,
                             skip_create_weights=False)
        self.z_init_1 = Linear(s_input_dim,
                               self.global_config.token_z,
                               bias=False,
                               dtype=self.input_embedder_dtype,
                               mapping=self.input_embedder_mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=False)
        self.z_init_2 = Linear(s_input_dim,
                               self.global_config.token_z,
                               bias=False,
                               dtype=self.input_embedder_dtype,
                               mapping=self.input_embedder_mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=False)
        self.rel_pos = RelativePositionEncoder(
            token_z=self.global_config.token_z,
            fix_sym_check=False,
            cyclic_pos_enc=True,
            period_broadcast=True,
            dtype=self.input_embedder_dtype,
            mapping=self.input_embedder_mapping,
            skip_create_weights=False)
        self.token_bonds = Linear(
            1,
            self.global_config.token_z,
            bias=False,
            dtype=self.input_embedder_dtype,
            mapping=self.input_embedder_mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=False)

        ### Trunk ###
        self.trunk = Trunk(config=self.trunk_config)

        ### Distogram ###
        self.distogram_module = DistogramModule(
            token_z=self.global_config.token_z,
            num_bins=self.global_config.num_bins,
            version="v1",
            dtype=self.structure_module_config.torch_dtype,
            mapping=self.structure_module_config.mapping,
            skip_create_weights=False)

        ### Atom diffusion ###
        score_model_config = self.structure_module_config.score_model_config
        atom_encoder_config = score_model_config.atom_encoder_config
        token_transformer_config = score_model_config.token_transformer_config
        atom_decoder_config = score_model_config.atom_decoder_config
        self.diffusion_conditioning = DiffusionConditioning(
            token_s=self.global_config.token_s,
            token_z=self.global_config.token_z,
            atom_s=self.global_config.atom_s,
            atom_z=self.global_config.atom_z,
            atoms_per_window_queries=self.input_embedder_config.
            atoms_per_window_queries,
            atoms_per_window_keys=self.input_embedder_config.
            atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_config.num_blocks,
            atom_encoder_heads=atom_encoder_config.num_heads,
            token_transformer_depth=token_transformer_config.num_blocks,
            token_transformer_heads=token_transformer_config.num_heads,
            atom_decoder_depth=atom_decoder_config.num_blocks,
            atom_decoder_heads=atom_decoder_config.num_heads,
            atom_feature_dim=self.input_embedder_config.atom_feature_dim,
            conditioning_transition_layers=score_model_config.
            conditioning_transition_layers,
            use_no_atom_char=False,
            use_atom_backbone_feat=False,
            use_residue_feats_atoms=False,
            dtype=self.structure_module_dtype,
            mapping=self.structure_module_mapping,
            skip_create_weights=False,
        )
        self.structure_module = AtomDiffusion(self.structure_module_config)

        #### End of building up modules ####

    def load_weights(self, weights: dict = None):
        if weights is None:
            logger.info(f"Input weights is None, try to load weights from hubs")
            weights = load_weights_from_hubs(name=self.model_name)
        # Load weights for input embedder
        input_embedder_weights = convert_hf_input_embedder_torch(
            config=self.input_embedder_config,
            weights=weights,
            model_name=self.model_name)
        self.input_embedder.load_weights(input_embedder_weights)

        # Load weights for input projections
        s_init_weights = [{
            "weight": weights["s_init.weight"],
            "bias": weights.get("s_init.bias", None)
        }]
        z_init_1_weights = [{
            "weight": weights["z_init_1.weight"],
            "bias": weights.get("z_init_1.bias", None)
        }]
        z_init_2_weights = [{
            "weight": weights["z_init_2.weight"],
            "bias": weights.get("z_init_2.bias", None)
        }]
        self.s_init.load_weights(s_init_weights)
        self.z_init_1.load_weights(z_init_1_weights)
        self.z_init_2.load_weights(z_init_2_weights)

        self.rel_pos.linear.load_weights([{
            "weight":
            weights["rel_pos.linear_layer.weight"],
            "bias":
            weights.get("rel_pos.linear_layer.bias", None)
        }])
        self.token_bonds.load_weights([{
            "weight":
            weights["token_bonds.weight"],
            "bias":
            weights.get("token_bonds.bias", None)
        }])

        # Load weights for trunk
        msa_module_config = self.trunk_config.msa_module_config
        pairformer_config = self.trunk_config.pairformer_config
        trunk_weights = {}
        trunk_weights["msa_module"] = convert_hf_msa_module_torch(
            config=msa_module_config,
            weights=weights,
            model_name=self.model_name)
        trunk_weights["pairformer_module"] = convert_hf_pairformer_torch(
            config=pairformer_config,
            weights=weights,
            model_name=self.model_name)
        # construct the remaining weights for the trunk module
        for subname in ["s_norm", "z_norm", "s_recycle", "z_recycle"]:
            if subname not in trunk_weights:
                trunk_weights[subname] = [{
                    "weight":
                    weights[subname + ".weight"],
                    "bias":
                    weights.get(subname + ".bias", None)
                }]
        self.trunk.load_weights(trunk_weights)

        # Load weights for distogram
        self.distogram_module.distogram.load_weights([{
            "weight":
            weights["distogram_module.distogram.weight"],
            "bias":
            weights.get("distogram_module.distogram.bias", None)
        }])

        # Load weights for atom diffusion
        diffusion_conditioning_weights = convert_hf_diffusion_conditioning_torch(
            config=self.structure_module_config.score_model_config,
            weights=weights,
            model_name=self.model_name)
        self.diffusion_conditioning.load_weights(diffusion_conditioning_weights)

        structure_module_weights = convert_hf_structure_module_torch(
            config=self.structure_module_config,
            weights=weights,
            model_name=self.model_name)
        self.structure_module.load_weights(structure_module_weights)

    def get_module_feed_dict(self, feed_dict: dict[str, torch.Tensor],
                             module_name: str) -> dict[str, Any]:
        keys = []
        if module_name == "input_embedder":
            keys = [
                "atom_to_token", "ref_pos", "atom_pad_mask", "ref_space_uid",
                "ref_charge", "ref_element", "ref_atom_name_chars", "res_type",
                "profile", "deletion_mean", "pocket_feature"
            ]
        elif module_name == "relative_position_encoding":
            keys = [
                "asym_id", "residue_index", "entity_id", "cyclic_period",
                "token_index", "sym_id"
            ]
        elif module_name == "trunk":
            keys = [
                "msa", "has_deletion", "deletion_value", "msa_paired",
                "msa_mask", "token_pad_mask"
            ]
        else:
            raise ValueError(f"Module name {module_name} not supported")
        return {key: feed_dict.get(key, None) for key in keys}

    def forward(
        self,
        feed_dict: dict[str, torch.Tensor],
        recycling_steps: int = 0,
        num_sampling_steps: Optional[int] = 200,
        diffusion_samples: int = 1,
        max_parallel_samples: Optional[int] = None,
        steering_args: BoltzSteeringParams = None,
        all_reduce_params: Optional[AllReduceParams] = None
    ) -> dict[str, torch.Tensor]:

        if steering_args is None:
            steering_args = self.steering_args

        # Setup query to keys function for sequence local attention
        B, N_atoms, N_tokens = feed_dict["atom_to_token"].shape
        W = self.input_embedder_config.atoms_per_window_queries
        H = self.input_embedder_config.atoms_per_window_keys
        K = N_atoms // W
        keys_indexing_matrix = create_indexing_matrix(
            K, W, H, device=torch.device("cuda"))
        query_to_keys_func = partial(query_to_keys,
                                     keys_indexing_matrix=keys_indexing_matrix,
                                     W=W,
                                     H=H)
        attn_metadata = AttentionMetadata(query_to_keys=query_to_keys_func)

        # Run input embedder step
        s_inputs = self.input_embedder(
            **self.get_module_feed_dict(feed_dict, "input_embedder"),
            attn_metadata=attn_metadata,
            all_reduce_params=all_reduce_params,
        )
        # Initialize the sequence and pairwise embeddings
        s_init = self.s_init(s_inputs)
        z_init = (self.z_init_1(s_inputs)[:, :, None] +
                  self.z_init_2(s_inputs)[:, None, :])
        relative_position_encoding = self.rel_pos(
            **self.get_module_feed_dict(feed_dict,
                                        "relative_position_encoding"), )
        z_init = z_init + relative_position_encoding
        z_init = z_init + self.token_bonds(feed_dict["token_bonds"].float())

        # Run trunk module
        s, z = self.trunk(**self.get_module_feed_dict(feed_dict, "trunk"),
                          s_init=s_init,
                          z_init=z_init,
                          s_inputs=s_inputs,
                          recycling_steps=recycling_steps,
                          all_reduce_params=all_reduce_params)
        # Run distogram module
        pair_distogram = self.distogram_module(z)

        # Run diffusion conditioning module
        q, c, atom_enc_bias, atom_dec_bias, token_trans_bias = self.diffusion_conditioning(
            s_trunk=s,
            z_trunk=z,
            relative_position_encoding=relative_position_encoding,
            feature_dict=feed_dict,
            query_to_keys=query_to_keys_func,
        )

        network_condition_kwargs = {
            "q": q,
            "c": c,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_trans_bias": token_trans_bias,
        }

        # Run structure module
        struct_module_output = self.structure_module.sample(
            s_trunk=s.float(),
            s_inputs=s_inputs.float(),
            feature_dict=feed_dict,
            num_sampling_steps=num_sampling_steps,
            multiplicity=diffusion_samples,
            max_parallel_samples=max_parallel_samples,
            network_condition_kwargs=network_condition_kwargs,
            attn_metadata=attn_metadata,
            all_reduce_params=all_reduce_params,
            steering_args=steering_args,
        )

        ret = {
            "pdistogram": pair_distogram,
            "s": s,
            "z": z,
        }
        ret.update(struct_module_output)

        remain = {
            "s_inputs": s_inputs,
            "relative_position_encoding": relative_position_encoding,
        }
        # TODO: to be continued
        return ret, remain

    @staticmethod
    def optimize(
        model: nn.Module,
        accelerated_modules: Boltz1AcceleratedModules,
        context_memory_allocator: Optional[BaseContextMemoryManager] = None
    ) -> nn.Module:
        """
        This function is used to build the optimized version of Boltz1 model from the original.
        Args:
            model: The original model to be optimized.
            accelerated_modules: A dictionary of modules to be accelerated.
            context_memory_allocator: The context memory allocator to be used for each module.
        Returns:
            The Boltz1 optimized model.
        """
        if accelerated_modules is None:
            return model

        module_names = accelerated_modules.get_module_names()
        device = next(model.parameters()).device
        state_dict = model.state_dict()

        opt_m = {}

        if "structure_pairformer" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "structure_pairformer")
            backend = accelerated_modules.get_module_backend(
                "structure_pairformer")
            default_config = accelerated_modules.get_default_module_config(
                "structure_pairformer")
            structure_pairformer = build_optimized_module(
                state_dict=state_dict,
                module_name="structure_pairformer",
                backend_builder=PairformerBackendBuilder,
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                compile=False,  # TODO: Whether to compile the module
                device=device,
                context_memory_allocator=context_memory_allocator,
                default_config=default_config,
                convert_weights_func=convert_hf_pairformer_torch,
                convert_weights_func_kwargs={
                    "config": default_config,
                    "pairformer_type": "structure",
                    "weights": state_dict
                },
            )
            opt_m["structure_pairformer"] = structure_pairformer
            setattr(model, "pairformer_module", structure_pairformer)

        if "confidence_pairformer" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "confidence_pairformer")
            backend = accelerated_modules.get_module_backend(
                "confidence_pairformer")
            default_config = accelerated_modules.get_default_module_config(
                "confidence_pairformer")
            confidence_pairformer = build_optimized_module(
                state_dict=state_dict,
                module_name="confidence_pairformer",
                backend_builder=PairformerBackendBuilder,
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                compile=False,  # TODO: Whether to compile the module
                device=device,
                context_memory_allocator=context_memory_allocator,
                default_config=default_config,
                convert_weights_func=convert_hf_pairformer_torch,
                convert_weights_func_kwargs={
                    "config": default_config,
                    "pairformer_type": "confidence",
                    "weights": state_dict
                },
            )
            opt_m["confidence_pairformer"] = confidence_pairformer
            setattr(model.confidence_module, "pairformer_module",
                    confidence_pairformer)

        if "token_transformer" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "token_transformer")
            backend = accelerated_modules.get_module_backend(
                "token_transformer")
            default_config = accelerated_modules.get_default_module_config(
                "token_transformer")
            token_transformer = build_optimized_module(
                state_dict=state_dict,
                module_name="token_transformer",
                backend_builder=TokenTransformerBackendBuilder,
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                compile=False,  # TODO: Whether to compile the module
                device=device,
                context_memory_allocator=context_memory_allocator,
                default_config=default_config,
                convert_weights_func=convert_hf_diffusion_transformer_torch,
                convert_weights_func_kwargs={
                    "config": default_config,
                    "weights": state_dict
                },
            )
            opt_m["token_transformer"] = token_transformer
            setattr(model.structure_module.score_model, "token_transformer",
                    token_transformer)

        if "msa_module" in module_names:
            checkpoint_dir = accelerated_modules.get_module_checkpoint(
                "msa_module")
            backend = accelerated_modules.get_module_backend("msa_module")
            default_config = accelerated_modules.get_default_module_config(
                "msa_module")
            msa_module = build_optimized_module(
                state_dict=state_dict,
                module_name="msa_module",
                backend_builder=MSAModuleBackendBuilder,
                checkpoint_dir=checkpoint_dir,
                backend=backend,
                compile=False,  # TODO: Whether to compile the module
                device=device,
                context_memory_allocator=context_memory_allocator,
                default_config=default_config,
                convert_weights_func=convert_hf_msa_module_torch,
                convert_weights_func_kwargs={
                    "config": default_config,
                    "weights": state_dict
                },
            )
            opt_m["msa_module"] = msa_module
            setattr(model, "msa_module", msa_module)
        return model, opt_m
