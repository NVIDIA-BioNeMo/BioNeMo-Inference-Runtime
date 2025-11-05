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
from typing import Optional

import torch
import torch.nn as nn
from tensorrt_llm.functional import AllReduceParams
from tensorrt_llm.logger import logger

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.layers.embedders.boltz import Boltz1InputEmbedder
from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo._torch.layers.position_encoders import \
    RelativePositionEncoder
from tensorrt_bionemo._torch.layers.recycling.boltz import Recycling
from tensorrt_bionemo._torch.layers.sequence_local_atom import (
    create_indexing_matrix, query_to_keys)
from tensorrt_bionemo.hubs import load_weights as load_weights_from_hubs
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.runtime import BaseContextMemoryManager

from ..helper import AcceleratedModules, build_optimized_module
from .configs import Boltz1Config
from .const import NUM_POCKET_CONTACT_INFO, NUM_TOKENS
from .convert import (convert_hf_diffusion_transformer_torch,
                      convert_hf_input_embedder_torch,
                      convert_hf_msa_module_torch, convert_hf_pairformer_torch)
from .modules import (MSAModuleBackendBuilder, PairformerBackendBuilder,
                      TokenTransformerBackendBuilder)


class Boltz1AcceleratedModules(AcceleratedModules):

    def get_supported_module_names(self):
        return [
            "structure_pairformer", "confidence_pairformer",
            "token_transformer", "msa_module"
        ]


class Boltz1(nn.Module):

    def __init__(self,
                 config: Boltz1Config = None,
                 input_embedder_dtype: torch.dtype = torch.float32,
                 input_embedder_mapping: Optional[Mapping] = None,
                 recycling_dtype: torch.dtype = torch.float32,
                 recycling_mapping: Optional[Mapping] = None):
        super().__init__()
        self.model_name = "boltz-1"
        self.config = config or Boltz1Config.from_pretrained()

        # Setup for input embedder
        self.input_embedder_dtype = input_embedder_dtype
        self.input_embedder_mapping = input_embedder_mapping or Mapping()
        self.input_embedder_config = self.config.input_embedder_config
        self.input_embedder_config.dtype = self.input_embedder_dtype
        self.input_embedder_config.mapping = self.input_embedder_mapping

        # Setup for recycling
        self.recycling_mapping = recycling_mapping or Mapping()
        self.recycling_dtype = recycling_dtype
        self.structure_pairformer_config = self.config.structure_pairformer_config
        self.msa_module_config = self.config.msa_module_config

        self.structure_pairformer_config.mapping = self.recycling_mapping
        self.msa_module_config.mapping = self.recycling_mapping
        self.msa_module_config.set_dtype(self.recycling_dtype)
        self.structure_pairformer_config.set_dtype(self.recycling_dtype)

        #### Build up modules ####

        ### Input embedder ###
        self.input_embedder = Boltz1InputEmbedder(self.input_embedder_config)

        ### Input projections ###
        s_input_dim = (self.config.token_s + 2 * NUM_TOKENS + 1 +
                       NUM_POCKET_CONTACT_INFO)
        self.s_init = Linear(s_input_dim,
                             self.config.token_s,
                             bias=False,
                             dtype=self.input_embedder_dtype,
                             mapping=self.input_embedder_mapping,
                             tensor_parallel_mode=TensorParallelMode.COLUMN,
                             gather_output=True,
                             skip_create_weights=False)
        self.z_init_1 = Linear(s_input_dim,
                               self.config.token_z,
                               bias=False,
                               dtype=self.input_embedder_dtype,
                               mapping=self.input_embedder_mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=False)
        self.z_init_2 = Linear(s_input_dim,
                               self.config.token_z,
                               bias=False,
                               dtype=self.input_embedder_dtype,
                               mapping=self.input_embedder_mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=False)
        self.rel_pos = RelativePositionEncoder(
            token_z=self.config.token_z,
            fix_sym_check=False,
            cyclic_pos_enc=True,
            period_broadcast=True,
            dtype=self.input_embedder_dtype,
            mapping=self.input_embedder_mapping,
            skip_create_weights=False)
        self.token_bonds = Linear(
            1,
            self.config.token_z,
            bias=False,
            dtype=self.input_embedder_dtype,
            mapping=self.input_embedder_mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=False)

        ### Recycling ###
        self.recycling = Recycling(
            msa_module_config=self.msa_module_config,
            pairformer_module_config=self.structure_pairformer_config,
            mapping=self.recycling_mapping)

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

        # Load weights for recycling
        recycling_weights = {}
        recycling_weights["msa_module"] = convert_hf_msa_module_torch(
            config=self.msa_module_config,
            weights=weights,
            model_name=self.model_name)
        recycling_weights["pairformer_module"] = convert_hf_pairformer_torch(
            config=self.structure_pairformer_config,
            weights=weights,
            model_name=self.model_name)
        # construct the remaining weights for the recycling module
        for subname in ["s_norm", "z_norm", "s_recycle", "z_recycle"]:
            if subname not in recycling_weights:
                recycling_weights[subname] = [{
                    "weight":
                    weights[subname + ".weight"],
                    "bias":
                    weights.get(subname + ".bias", None)
                }]
        self.recycling.load_weights(recycling_weights)

    def forward(
        self,
        feed_dict: dict[str, torch.Tensor],
        recycling_steps: int = 0,
        num_sampling_steps: Optional[int] = 200,
        diffusion_samples: int = 1,
        all_reduce_params: Optional[AllReduceParams] = None
    ) -> dict[str, torch.Tensor]:

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
        attn_metadata = AttentionMetadata(
            query_to_keys=query_to_keys_func,
            bias_cache=None  # Dont need bias cache for input embedder attention
        )

        # Run input embedder step
        s_inputs = self.input_embedder(
            atom_to_token=feed_dict["atom_to_token"],
            ref_pos=feed_dict["ref_pos"],
            atom_pad_mask=feed_dict["atom_pad_mask"],
            ref_space_uid=feed_dict["ref_space_uid"],
            ref_charge=feed_dict["ref_charge"],
            ref_element=feed_dict["ref_element"],
            ref_atom_name_chars=feed_dict["ref_atom_name_chars"],
            res_type=feed_dict["res_type"],
            profile=feed_dict["profile"],
            deletion_mean=feed_dict["deletion_mean"],
            pocket_feature=feed_dict["pocket_feature"],
            attn_metadata=attn_metadata,
            all_reduce_params=all_reduce_params,
        )
        # Initialize the sequence and pairwise embeddings
        s_init = self.s_init(s_inputs)
        z_init = (self.z_init_1(s_inputs)[:, :, None] +
                  self.z_init_2(s_inputs)[:, None, :])
        relative_position_encoding = self.rel_pos(
            asym_id=feed_dict["asym_id"],
            residue_index=feed_dict["residue_index"],
            entity_id=feed_dict["entity_id"],
            cyclic_period=feed_dict["cyclic_period"],
            token_index=feed_dict["token_index"],
            sym_id=feed_dict["sym_id"],
        )
        z_init = z_init + relative_position_encoding
        z_init = z_init + self.token_bonds(feed_dict["token_bonds"].float())

        # Do recycling
        s, z = self.recycling(s_init=s_init,
                              z_init=z_init,
                              s_inputs=s_inputs,
                              msa=feed_dict["msa"],
                              has_deletion=feed_dict["has_deletion"],
                              deletion_value=feed_dict["deletion_value"],
                              msa_paired=feed_dict["msa_paired"],
                              msa_mask=feed_dict["msa_mask"],
                              token_pad_mask=feed_dict["token_pad_mask"],
                              recycling_steps=recycling_steps,
                              all_reduce_params=all_reduce_params)

        # TODO: to be continued
        return {
            "s": s,
            "z": z,
            "s_inputs": s_inputs,
            "relative_position_encoding": relative_position_encoding
        }

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
