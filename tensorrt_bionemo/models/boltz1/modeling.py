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

from tensorrt_bionemo._torch.attention_backend import (
    AttentionMetadata, auto_select_pairwise_attention_backend,
    auto_select_triangle_attention_backend)
from tensorrt_bionemo._torch.distributed import AllReduceParams
from tensorrt_bionemo._torch.graph_optimization.cuda_graph.runtime import \
    CUDAGraphOptimizationTracker
from tensorrt_bionemo._torch.layers.distogram import DistogramModule
from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo._torch.layers.position_encoders import \
    RelativePositionEncoder
from tensorrt_bionemo._torch.layers.sequence_local_atom import (
    create_gather_indices, query_to_keys_optimized)
from tensorrt_bionemo._torch.modules.boltz.confidence import \
    Boltz1ConfidenceModule
from tensorrt_bionemo._torch.modules.boltz.embedders import Boltz1InputEmbedder
from tensorrt_bionemo._torch.modules.boltz.physical.steering import \
    BoltzSteeringParams
from tensorrt_bionemo._torch.modules.boltz.structure import (
    AtomDiffusion, DiffusionConditioning)
from tensorrt_bionemo._torch.modules.boltz.trunk import Trunk
from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.hubs import FoldingSupportMatrix as SupMat
from tensorrt_bionemo.hubs import load_weights as load_weights_from_hubs
from tensorrt_bionemo.logger import logger
from tensorrt_bionemo.pipeline.models.boltz2.const import (
    num_pocket_contact_info, num_tokens)
from tensorrt_bionemo.utils import str_dtype_to_torch

from ..optimize_module_setter import (AcceleratedConfig,
                                      DiscoveredModuleRegistry,
                                      OptimizedModuleSetterMixin)
from .config import PRETRAINED_CONFIG_REGISTRY
from .convert import (convert_hf_confidence_torch,
                      convert_hf_diffusion_conditioning_torch,
                      convert_hf_input_embedder_torch,
                      convert_hf_msa_module_torch, convert_hf_pairformer_torch,
                      convert_hf_structure_module_torch)


class Boltz1(nn.Module, OptimizedModuleSetterMixin):
    # Whitelists gating modules are discovered generically 
    # via ``@support_graph_optimization``.
    GRAPH_OPT_ENABLED_MODULES = {
        "token_transformer": "structure_module.score_model.token_transformer",
        "diffusion_module": "structure_module.score_model",
    }
    
    def get_optimized_modules(
        self, accelerated_configs: dict[str, AcceleratedConfig]
    ) -> DiscoveredModuleRegistry:
        return DiscoveredModuleRegistry(
            self, accelerated_configs,
            role_aliases=self.GRAPH_OPT_ENABLED_MODULES,
            graph_optimization_cls=CUDAGraphOptimizationTracker)

    def __init__(self,
                 config: BaseConfig = None,
                 include_load_weights: bool = True,
                 model_name: Optional[str] = None):
        super().__init__()
        self.model_name = model_name or SupMat.Boltz1
        self.config = config or self.get_pretrained_config(self.model_name)

        # Setup for input embedder
        self.input_embedder_dtype = self.config.input_embedder.torch_dtype
        self.input_embedder_mapping = self.config.input_embedder.mapping
        self.input_embedder_config = self.config.input_embedder

        # Setup for trunk
        self.trunk_mapping = self.config.trunk.mapping
        self.trunk_dtype = self.config.trunk.torch_dtype
        self.trunk_config = self.config.trunk
        self.recompute_rel_pos = getattr(self.config, "recompute_rel_pos",
                                         False)

        # Setup steering params:
        self.steering_args = BoltzSteeringParams(contact_guidance_update=False)

        # Setup for atom diffusion
        self.structure_module_dtype = self.config.structure_module.torch_dtype
        self.structure_module_mapping = self.config.structure_module.mapping
        self.structure_module_config = self.config.structure_module

        # Setup for confidence module
        self.confidence_module_config = self.config.confidence_module
        self.confidence_module_dtype = self.config.confidence_module.torch_dtype
        self.confidence_module_mapping = self.config.confidence_module.mapping
        self.confidence_module_config = self.config.confidence_module

        #### Build up modules ####

        ### Input embedder ###
        self.input_embedder = Boltz1InputEmbedder(self.input_embedder_config)

        ### Input projections ###
        s_input_dim = (self.config.token_s + 2 * num_tokens + 1 +
                       num_pocket_contact_info)
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

        ### Trunk ###
        self.trunk = Trunk(config=self.trunk_config)

        ### Distogram ###
        self.distogram_module = DistogramModule(
            token_z=self.config.token_z,
            num_bins=self.config.num_bins,
            version="v1",
            dtype=self.structure_module_config.torch_dtype,
            mapping=self.structure_module_config.mapping,
            skip_create_weights=False)

        ### Atom diffusion ###
        score_model_config = self.structure_module_config.score_model
        atom_encoder_config = score_model_config.atom_encoder
        token_transformer_config = score_model_config.token_transformer
        atom_decoder_config = score_model_config.atom_decoder
        self.diffusion_conditioning = DiffusionConditioning(
            token_s=self.config.token_s,
            token_z=self.config.token_z,
            atom_s=self.config.atom_s,
            atom_z=self.config.atom_z,
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
            pairwise_conditioner_dtype=str_dtype_to_torch(
                score_model_config.pairwise_conditioning_dtype),
            token_trans_bias_dtype=str_dtype_to_torch(
                score_model_config.token_trans_bias_dtype),
            mapping=self.structure_module_mapping,
            skip_create_weights=False,
        )
        self.structure_module = AtomDiffusion(self.structure_module_config)

        ### Confidence module ###
        self.confidence_module = Boltz1ConfidenceModule(
            self.confidence_module_config)
        #### End of building up modules ####

        if include_load_weights:
            self.load_weights()

        self.eval()

    def load_weights(self, weights: dict = None):
        if weights is None:
            logger.info(
                f"Input weights is None, try to load weights from hubs")
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
        msa_module_config = self.trunk_config.msa_module
        pairformer_config = self.trunk_config.pairformer
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
            config=self.structure_module_config.score_model,
            weights=weights,
            model_name=self.model_name)
        self.diffusion_conditioning.load_weights(
            diffusion_conditioning_weights)

        structure_module_weights = convert_hf_structure_module_torch(
            config=self.structure_module_config,
            weights=weights,
            model_name=self.model_name)
        self.structure_module.load_weights(structure_module_weights)

        # Load weights for confidence module
        confidence_module_weights = convert_hf_confidence_torch(
            config=self.confidence_module_config,
            weights=weights,
            model_name=self.model_name)
        self.confidence_module.load_weights(confidence_module_weights)

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

    @staticmethod
    def get_pretrained_config(model_name: str = SupMat.Boltz1) -> BaseConfig:
        # Set default optimization configs
        config_class = PRETRAINED_CONFIG_REGISTRY.get(model_name)
        if config_class is None:
            raise ValueError(
                f"Boltz1 pretrained config not found for model name: {model_name}"
            )
        config = config_class()
        config.trunk.set_dtype(torch.bfloat16)
        tri_backend = auto_select_triangle_attention_backend(torch.bfloat16)
        pair_backend = auto_select_pairwise_attention_backend(torch.bfloat16)

        config.trunk.set_triangle_attention_backend(tri_backend)
        config.trunk.set_pairwise_attention_backend(pair_backend)
        config.trunk.pairformer.s_path_dtype = torch.bfloat16
        config.structure_module.score_model.set_dtype(torch.bfloat16)
        config.structure_module.score_model.set_pairwise_attention_backend(
            pair_backend)

        config.confidence_module.set_triangle_attention_backend(tri_backend)
        config.confidence_module.msa_module.set_dtype(torch.bfloat16)
        config.confidence_module.pairformer.set_dtype(torch.bfloat16)
        config.confidence_module.pairformer.s_path_dtype = torch.bfloat16
        config.confidence_module.pairformer.set_pairwise_attention_backend(
            pair_backend)

        return config

    def create_attn_metadata(self, n_atoms: int) -> AttentionMetadata:
        W = self.input_embedder_config.atoms_per_window_queries
        H = self.input_embedder_config.atoms_per_window_keys
        K = n_atoms // W
        gather_indices, _ = create_gather_indices(K,
                                                  W,
                                                  H,
                                                  device=torch.device("cuda"))
        query_to_keys_func = partial(query_to_keys_optimized,
                                     gather_indices=gather_indices,
                                     W=W,
                                     H=H)
        return AttentionMetadata(query_to_keys=query_to_keys_func,
                                 bias_cache=None)

    def forward(
        self,
        feed_dict: dict[str, torch.Tensor],
        recycling_steps: int = 3,
        num_sampling_steps: Optional[int] = 200,
        diffusion_samples: int = 1,
        max_parallel_samples: Optional[int] = None,
        steering_args: BoltzSteeringParams = None,
        all_reduce_params: Optional[AllReduceParams] = None
    ) -> dict[str, torch.Tensor]:

        # Training-only feats: never read in inference (forward + postprocessor). Drop them up front
        # so they don't sit on the GPU -- disto_target is ~7.9 GB and r_set_to_rep_atom ~1 GB at
        # N~4000. Handles feed_dicts from the OSS data pipeline, which still emits these. (Boltz1 has
        # no token_to_center_atom.)
        for _train_only_key in ("disto_target", "r_set_to_rep_atom"):
            feed_dict.pop(_train_only_key, None)

        if steering_args is None:
            steering_args = self.steering_args

        # Setup query to keys function for sequence local attention
        B, N_atoms, N_tokens = feed_dict["atom_to_token"].shape
        attn_metadata = self.create_attn_metadata(N_atoms)

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
        rel_pos_feats = self.get_module_feed_dict(
            feed_dict, "relative_position_encoding")
        relative_position_encoding = self.rel_pos(**rel_pos_feats)
        # In-place accumulation: z_init is a freshly-owned [B,N,N,c_z] (from the broadcast add
        # above), so fold each term into it rather than allocating a new z_init per '+' (each of
        # which is ~8 GB fp32 at N~4000). Inference-only.
        z_init += relative_position_encoding
        if self.recompute_rel_pos:
            # Consumed into z_init above; drop the [N,N,token_z] encoding rather than hold it
            # (~15 GB fp32 at N~5k) across the trunk -- recomputed from rel_pos_feats (tiny [B,N]
            # index tensors) just before diffusion_conditioning below.
            relative_position_encoding = None
        z_init += self.token_bonds(feed_dict["token_bonds"].float())

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
        if self.recompute_rel_pos:
            relative_position_encoding = self.rel_pos(**rel_pos_feats)
        q, c, atom_enc_bias, atom_dec_bias, token_trans_bias = self.diffusion_conditioning(
            s_trunk=s,
            z_trunk=z,
            relative_position_encoding=relative_position_encoding,
            feature_dict=feed_dict,
            query_to_keys=attn_metadata.query_to_keys,
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

        # Free diffusion-stage condition tensors before the confidence module (~12 GB at large N):
        # they are consumed by the sampler above and not read by the confidence stage.
        del q, c, atom_enc_bias, atom_dec_bias, token_trans_bias
        del network_condition_kwargs

        # Keep only the sampler outputs the confidence stage / return dict need, then drop the rest
        # of the sampler output dict.
        x_pred = struct_module_output["sample_atom_coords"]
        s_diffusion = (struct_module_output["diff_token_repr"] if
                       self.confidence_module_config.use_s_diffusion else None)
        del struct_module_output

        confidence_module_output = self.confidence_module(
            s=s,
            z=z,
            s_diffusion=s_diffusion,
            x_pred=x_pred,
            feature_dict=feed_dict,
            pred_distogram_logits=pair_distogram.float(),
            multiplicity=diffusion_samples,
            attn_metadata=attn_metadata,
            all_reduce_params=all_reduce_params,
        )

        iptm_score = confidence_module_output["iptm"]
        if torch.allclose(iptm_score, torch.zeros_like(iptm_score)):
            iptm_score = confidence_module_output["ptm"]

        ret = {
            "confidence_score":
            (4 * confidence_module_output["complex_plddt"] + iptm_score) / 5,
            "masks":
            feed_dict["atom_pad_mask"],
            "token_masks":
            feed_dict["token_pad_mask"],
            "coords":
            x_pred,
            "complex_plddt":
            confidence_module_output["complex_plddt"],
            "complex_iplddt":
            confidence_module_output["complex_iplddt"],
            "complex_pde":
            confidence_module_output["complex_pde"],
            "complex_ipde":
            confidence_module_output["complex_ipde"],
            "plddt":
            confidence_module_output["plddt"],
            "ptm":
            confidence_module_output["ptm"],
            "iptm":
            confidence_module_output["iptm"],
            "ligand_iptm":
            confidence_module_output["ligand_iptm"],
            "protein_iptm":
            confidence_module_output["protein_iptm"],
            "pair_chains_iptm":
            confidence_module_output["pair_chains_iptm"],
        }
        return ret
