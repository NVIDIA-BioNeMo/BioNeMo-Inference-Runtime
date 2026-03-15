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
import torch.nn.functional as F
from tensorrt_llm_lite.logger import logger

from tensorrt_bionemo._torch.attention_backend import AttentionMetadata
from tensorrt_bionemo._torch.distributed import AllReduceParams
from tensorrt_bionemo._torch.layers.conditioning import ContactConditioning
from tensorrt_bionemo._torch.layers.distogram import DistogramModule
from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo._torch.layers.position_encoders import \
    RelativePositionEncoder
from tensorrt_bionemo._torch.layers.sequence_local_atom import (
    create_indexing_matrix, query_to_keys)
from tensorrt_bionemo._torch.modules.boltz.affinity import (
    AffinityModule, compute_distogram, create_cross_pair_mask, get_best_coords)
from tensorrt_bionemo._torch.modules.boltz.confidence import \
    Boltz2ConfidenceModule
from tensorrt_bionemo._torch.modules.boltz.embedders import Boltz2InputEmbedder
from tensorrt_bionemo._torch.modules.boltz.physical.steering import \
    BoltzSteeringParams
from tensorrt_bionemo._torch.modules.boltz.structure import (
    AtomDiffusion, DiffusionConditioning)
from tensorrt_bionemo._torch.modules.boltz.trunk import Trunk
from tensorrt_bionemo._trt.module_wrappers import (PairformerTRT,
                                                   TokenTransformerTRT)
from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.hubs import FoldingSupportMatrix as SupMat
from tensorrt_bionemo.hubs import load_weights as load_weights_from_hubs
from tensorrt_bionemo.pipeline.models.boltz.const import (
    CONTACT_CONDITIONING_INFO, NUM_BOND_TYPES)

from ..helper import (AcceleratedConfig, ModuleRegistry, ModuleSpec,
                      OptimizedModuleSetterMixin)
from .config import PRETRAINED_CONFIG_REGISTRY, Boltz2AffinityConfig
from .convert import (convert_hf_affinity_module_torch,
                      convert_hf_confidence_module_torch,
                      convert_hf_diffusion_conditioning_torch,
                      convert_hf_input_embedder_torch,
                      convert_hf_msa_module_torch, convert_hf_pairformer_torch,
                      convert_hf_structure_module_torch)


class Boltz2ModuleRegistry(ModuleRegistry):

    def get_accelerated_modules(self) -> dict[str, ModuleSpec]:
        return {
            "structure_pairformer":
            ModuleSpec(
                getter=lambda mod: mod.trunk.pairformer_module,
                setter=lambda mod, opt: setattr(mod.trunk, "pairformer_module",
                                                opt),
                trt_cls=PairformerTRT,
                compiled_cls=None,
            ),
            "confidence_pairformer":
            ModuleSpec(
                getter=lambda mod: mod.confidence_module.pairformer_stack,
                setter=lambda mod, opt: setattr(mod.confidence_module,
                                                "pairformer_stack", opt),
                trt_cls=PairformerTRT,
                compiled_cls=None,
            ),
            "token_transformer":
            ModuleSpec(
                getter=lambda mod:
                (mod.structure_module.score_model.token_transformer),
                setter=lambda mod, opt: setattr(
                    mod.structure_module.score_model, "token_transformer", opt
                ),
                trt_cls=TokenTransformerTRT,
                compiled_cls=None,
            ),
            "structure_msa":
            ModuleSpec(
                getter=lambda mod: mod.trunk.msa_module,
                setter=lambda mod, opt: setattr(mod.trunk, "msa_module", opt),
                compiled_cls=None,
            ),
        }


class Boltz2AffinityModuleRegistry(Boltz2ModuleRegistry):
    pass


class Boltz2(nn.Module, OptimizedModuleSetterMixin):

    def __init__(self,
                 config: BaseConfig = None,
                 model_name: Optional[str] = None,
                 include_load_weights: bool = True):
        super().__init__()

        self.model_name = model_name or SupMat.Boltz2
        self.config = config or self.get_pretrained_config(self.model_name)

        self.confidence_prediction = self.config.confidence_prediction
        self.skip_run_structure = self.config.skip_run_structure
        # Setup for input embedder
        self.input_embedder_dtype = self.config.input_embedder.torch_dtype
        self.input_embedder_mapping = self.config.input_embedder.mapping
        self.input_embedder_config = self.config.input_embedder

        # Setup for trunk
        self.trunk_mapping = self.config.trunk.mapping
        self.trunk_dtype = self.config.trunk.torch_dtype
        self.trunk_config = self.config.trunk

        # Setup for atom diffusion
        self.structure_module_dtype = self.config.structure_module.torch_dtype
        self.structure_module_mapping = self.config.structure_module.mapping
        self.structure_module_config = self.config.structure_module

        #Setup for confidence module
        self.confidence_module_dtype = self.config.confidence_module.torch_dtype
        self.confidence_module_mapping = self.config.confidence_module.mapping

        # Build up modules
        self.input_embedder = Boltz2InputEmbedder(self.input_embedder_config)

        ### Input projections ###
        self.s_init = Linear(self.config.token_s,
                             self.config.token_s,
                             bias=False,
                             dtype=self.input_embedder_dtype,
                             mapping=self.input_embedder_mapping,
                             tensor_parallel_mode=TensorParallelMode.COLUMN,
                             gather_output=True,
                             skip_create_weights=False)
        self.z_init_1 = Linear(self.config.token_s,
                               self.config.token_z,
                               bias=False,
                               dtype=self.input_embedder_dtype,
                               mapping=self.input_embedder_mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=False)
        self.z_init_2 = Linear(self.config.token_s,
                               self.config.token_z,
                               bias=False,
                               dtype=self.input_embedder_dtype,
                               mapping=self.input_embedder_mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=False)
        self.rel_pos = RelativePositionEncoder(
            token_z=self.config.token_z,
            fix_sym_check=self.config.fix_sym_check,
            cyclic_pos_enc=self.config.cyclic_pos_enc,
            period_broadcast=False,
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
        if self.config.bond_type_feature:
            self.token_bonds_type = nn.Embedding(NUM_BOND_TYPES + 1,
                                                 self.config.token_z)
        self.contact_conditioning = ContactConditioning(
            token_z=self.config.token_z,
            cutoff_min=self.config.conditioning_cutoff_min,
            cutoff_max=self.config.conditioning_cutoff_max,
            contact_conditioning_info=CONTACT_CONDITIONING_INFO)

        ### Trunk ###
        self.trunk = Trunk(self.trunk_config)

        ### Distogram ###
        self.distogram_module = DistogramModule(
            token_z=self.config.token_z,
            num_bins=self.config.num_bins,
            num_distograms=self.config.num_distograms,
            version="v2",
            dtype=self.structure_module_dtype,
            mapping=self.structure_module_mapping,
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
            use_no_atom_char=self.input_embedder_config.use_no_atom_char,
            use_atom_backbone_feat=self.input_embedder_config.
            use_atom_backbone_feat,
            use_residue_feats_atoms=self.input_embedder_config.
            use_residue_feats_atoms,
            version="v2",
            dtype=self.structure_module_dtype,
            mapping=self.structure_module_mapping,
            skip_create_weights=False,
        )
        self.structure_module = AtomDiffusion(self.structure_module_config)

        self.confidence_module = Boltz2ConfidenceModule(
            self.config.confidence_module,
            dtype=self.confidence_module_dtype,
            mapping=self.confidence_module_mapping,
            skip_create_weights=False)

        #### End of building up modules ####

        if include_load_weights:
            self.load_weights()

        self.eval()

    def get_optimized_modules(
        self,
        accelerated_configs: dict[str,
                                  AcceleratedConfig]) -> Boltz2ModuleRegistry:
        return Boltz2ModuleRegistry(accelerated_configs)

    @staticmethod
    def get_pretrained_config(model_name: str = SupMat.Boltz2) -> BaseConfig:
        # Set default optimization configs
        config_class = PRETRAINED_CONFIG_REGISTRY.get(model_name)
        if config_class is None:
            raise ValueError(
                f"Boltz1 pretrained config not found for model name: {model_name}"
            )
        config = config_class()
        config.input_embedder.diffusion_transformer.set_dtype(torch.bfloat16)
        config.trunk.set_dtype(torch.bfloat16)
        config.trunk.pairformer.s_path_dtype = torch.bfloat16
        config.trunk.set_triangle_attention_backend("CUEQUIV")
        config.structure_module.score_model.set_dtype(torch.bfloat16)

        config.confidence_module.set_triangle_attention_backend("CUEQUIV")
        config.confidence_module.pairformer.set_dtype(torch.bfloat16)
        return config

    def load_weights(self, weights: dict = None) -> None:
        """
        Args:
            weights: The weights of the model. State dict of the original model.
        """
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
        if self.config.bond_type_feature:
            self.token_bonds_type.weight.data.copy_(
                weights["token_bonds_type.weight"])

        self.contact_conditioning.encoding_unspecified.data.copy_(
            weights["contact_conditioning.encoding_unspecified"])
        self.contact_conditioning.encoding_unselected.data.copy_(
            weights["contact_conditioning.encoding_unselected"])
        self.contact_conditioning.encoder.load_weights([{
            "weight":
            weights["contact_conditioning.encoder.weight"],
            "bias":
            weights.get("contact_conditioning.encoder.bias", None)
        }])
        self.contact_conditioning.fourier_embedding.proj.load_weights([{
            "weight":
            weights["contact_conditioning.fourier_embedding.proj.weight"],
            "bias":
            weights.get("contact_conditioning.fourier_embedding.proj.bias",
                        None)
        }])

        assert weights is not None, "Input weights is None"
        trunk_weights = {}
        # load the weights for the msa_module and pairformer_module
        trunk_weights["msa_module"] = convert_hf_msa_module_torch(
            config=self.trunk_config.msa_module,
            weights=weights,
            model_name=self.model_name)
        trunk_weights["pairformer_module"] = convert_hf_pairformer_torch(
            config=self.trunk_config.pairformer,
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

        confidence_module_weights = convert_hf_confidence_module_torch(
            config=self.config.confidence_module,
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
                "profile", "deletion_mean", "pocket_feature",
                "atom_backbone_feat", "method_feature", "modified",
                "cyclic_period", "mol_type"
            ]
        elif module_name == "input_embedder_affinity":
            keys = [
                "atom_to_token", "ref_pos", "atom_pad_mask", "ref_space_uid",
                "ref_charge", "ref_element", "ref_atom_name_chars", "res_type",
                "profile_affinity", "deletion_mean_affinity", "pocket_feature",
                "atom_backbone_feat", "method_feature", "modified",
                "cyclic_period", "mol_type"
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
        dict_ret = {key: feed_dict.get(key, None) for key in keys}

        if module_name == "input_embedder_affinity":
            dict_ret['profile'] = dict_ret.pop('profile_affinity')
            dict_ret['deletion_mean'] = dict_ret.pop('deletion_mean_affinity')

        return dict_ret

    def create_attn_metadata(self, n_atoms: int) -> AttentionMetadata:
        W = self.input_embedder_config.atoms_per_window_queries
        H = self.input_embedder_config.atoms_per_window_keys
        K = n_atoms // W
        keys_indexing_matrix = create_indexing_matrix(
            K, W, H, device=torch.device("cuda"))
        query_to_keys_func = partial(query_to_keys,
                                     keys_indexing_matrix=keys_indexing_matrix,
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
        affinity: bool = False,
        all_reduce_params: Optional[AllReduceParams] = None
    ) -> dict[str, torch.Tensor]:

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
        # Initialize the sequence embeddings
        s_init = self.s_init(s_inputs)

        # Initialize pairwise embeddings
        z_init = (self.z_init_1(s_inputs)[:, :, None] +
                  self.z_init_2(s_inputs)[:, None, :])
        relative_position_encoding = self.rel_pos(
            **self.get_module_feed_dict(feed_dict,
                                        "relative_position_encoding"), )
        z_init = z_init + relative_position_encoding

        z_init = z_init + self.token_bonds(feed_dict["token_bonds"].float())

        if self.config.bond_type_feature:
            z_init = z_init + self.token_bonds_type(
                feed_dict["type_bonds"].long())
        z_init = z_init + self.contact_conditioning(
            feed_dict["contact_conditioning"], feed_dict["contact_threshold"])

        # Do trunk
        s, z = self.trunk(s_init=s_init,
                          z_init=z_init,
                          s_inputs=s_inputs,
                          **self.get_module_feed_dict(feed_dict, "trunk"),
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
            s_trunk=s,
            s_inputs=s_inputs,
            feature_dict=feed_dict,
            num_sampling_steps=num_sampling_steps,
            multiplicity=diffusion_samples,
            max_parallel_samples=max_parallel_samples,
            network_condition_kwargs=network_condition_kwargs,
            attn_metadata=attn_metadata,
            all_reduce_params=all_reduce_params,
            steering_args=steering_args,
        )
        x_pred = struct_module_output["sample_atom_coords"]

        feed_dict["frames_idx"] = feed_dict["frames_idx"].squeeze(1)

        confidence_module_output = self.confidence_module(
            s_inputs=s_inputs,
            s=s,
            z=z,
            x_pred=x_pred,
            feats=feed_dict,
            pred_distogram_logits=pair_distogram[:, :, :, 0],
            multiplicity=diffusion_samples,
            run_sequentially=True,
            max_parallel_samples=max_parallel_samples
            if max_parallel_samples is not None else 1,
            all_reduce_params=all_reduce_params,
        )

        boltz2_output_dictionary = {
            "confidence_score":
            (4 * confidence_module_output["complex_plddt"] +
             confidence_module_output["iptm"]) / 5,
            "masks":
            feed_dict["atom_pad_mask"],
            "token_masks":
            feed_dict["token_pad_mask"],
            "coords":
            x_pred,
            "pde":
            confidence_module_output["pde"],
            "pae":
            confidence_module_output["pae"],
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
        if affinity:
            affinity_output_dictionary = {
                'z': z,
                'attention_metadata': attn_metadata
            }
            return boltz2_output_dictionary, affinity_output_dictionary

        return boltz2_output_dictionary


class Boltz2Affinity(Boltz2, OptimizedModuleSetterMixin):

    def __init__(self,
                 config: Boltz2AffinityConfig = None,
                 model_name: Optional[str] = None,
                 include_load_weights: bool = True):

        if config is None:
            config = Boltz2.get_pretrained_config()

        super().__init__(config=config, include_load_weights=False)

        self.model_name = model_name or SupMat.Boltz2Affinity
        self.config = config
        self.affinity_module1 = AffinityModule(self.config.affinity.module1)
        self.affinity_module2 = AffinityModule(self.config.affinity.module2)

        boundaries = torch.linspace(
            2, self.config.affinity.module1.max_dist,
            self.config.affinity.module1.num_dist_bins - 1)
        self.register_buffer("boundaries_1", boundaries)
        boundaries = torch.linspace(
            2, self.config.affinity.module2.max_dist,
            self.config.affinity.module2.num_dist_bins - 1)
        self.register_buffer("boundaries_2", boundaries)

        if include_load_weights:
            self.load_weights()

    @staticmethod
    def get_pretrained_config(
            model_name: str = SupMat.Boltz2Affinity) -> Boltz2AffinityConfig:
        config = Boltz2.get_pretrained_config(model_name)
        config.input_embedder.diffusion_transformer.set_dtype(torch.bfloat16)
        config.trunk.set_dtype(torch.bfloat16)
        config.trunk.set_triangle_attention_backend("CUEQUIV")
        config.structure_module.score_model.set_dtype(torch.bfloat16)

        config.confidence_module.set_triangle_attention_backend("CUEQUIV")
        config.confidence_module.pairformer.set_dtype(torch.bfloat16)

        config.affinity.module1.set_dtype(torch.bfloat16)
        config.affinity.module2.set_dtype(torch.bfloat16)
        return config

    def load_weights(self, weights: dict = None) -> None:
        if weights is None:
            weights = load_weights_from_hubs(name=self.model_name)
        super().load_weights(weights)
        self.load_affinity_weights(weights)

    def load_affinity_weights(self, weights: dict = None) -> None:
        if weights is None:
            logger.info(
                f"Input weights is None, try to load weights from hubs")
            weights = load_weights_from_hubs(name=self.model_name)
        affinity_weights_1 = convert_hf_affinity_module_torch(
            config=self.config.affinity.module1,
            weights=weights,
            model_name=self.model_name,
            affinity_module_name="affinity_module1")
        affinity_weights_2 = convert_hf_affinity_module_torch(
            config=self.config.affinity.module2,
            weights=weights,
            model_name=self.model_name,
            affinity_module_name="affinity_module2")
        self.affinity_module1.load_weights(affinity_weights_1)
        self.affinity_module2.load_weights(affinity_weights_2)

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

        if steering_args is not None:
            steering_args.physical_guidance_update = False
            steering_args.contact_guidance_update = False

        boltz2_output_dictionary, affinity_output_dictionary = super().forward(
            feed_dict,
            recycling_steps,
            num_sampling_steps,
            diffusion_samples,
            max_parallel_samples,
            steering_args,
            affinity=True,
            all_reduce_params=all_reduce_params)

        s_inputs_affinity = self.input_embedder(
            **self.get_module_feed_dict(feed_dict, "input_embedder_affinity"),
            attn_metadata=affinity_output_dictionary['attention_metadata'],
            all_reduce_params=all_reduce_params,
        )

        # Get the best coordinates to get the affinity prediction.
        best_coords = get_best_coords(boltz2_output_dictionary['coords'],
                                      boltz2_output_dictionary['iptm'])
        distogram = compute_distogram(best_coords, self.boundaries_1,
                                      feed_dict["token_to_rep_atom"])

        cross_pair_mask_0, cross_pair_mask_1 = create_cross_pair_mask(
            feed_dict["token_pad_mask"],
            feed_dict["mol_type"],
            feed_dict["affinity_token_mask"],
            include_mask_for_head=True)

        z_affinity = affinity_output_dictionary[
            'z'] * cross_pair_mask_0.unsqueeze(-1)

        affinity_probabilities = []
        affinity_pred_values = []
        for boundaries, affinity_module in zip(
            [self.boundaries_1, self.boundaries_2],
            [self.affinity_module1, self.affinity_module2]):
            distogram = compute_distogram(best_coords, boundaries,
                                          feed_dict["token_to_rep_atom"])

            affinity_module_output = affinity_module(
                s=s_inputs_affinity,
                z=z_affinity,
                distogram=distogram,
                cross_pair_mask_0=cross_pair_mask_0,
                cross_pair_mask_1=cross_pair_mask_1,
                attn_metadatas={},
                all_reduce_params=all_reduce_params,
            )

            affinity_pred_values.append(affinity_module_output[0])
            affinity_probabilities.append(F.sigmoid(affinity_module_output[1]))

        boltz2_output_dictionary.update({
            "affinity_pred_value":
            torch.concat(affinity_pred_values).mean(),
            "affinity_probability_binary":
            torch.concat(affinity_probabilities).mean()
        })

        return boltz2_output_dictionary
