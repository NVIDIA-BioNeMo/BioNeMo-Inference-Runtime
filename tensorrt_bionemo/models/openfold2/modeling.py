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
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from tensorrt_llm.functional import AllReduceParams
from tensorrt_llm.logger import logger

import tensorrt_bionemo.pipeline.openfold2.const as rc
from tensorrt_bionemo._torch.attention_backend import get_attention_backend
from tensorrt_bionemo._torch.modules.openfold2.embedders import (
    ExtraMSAEmbedder, InputEmbedder, InputEmbedderMultimer, RecyclingEmbedder,
    TemplateEmbedder, TemplateEmbedderMultimer)
from tensorrt_bionemo._torch.modules.openfold2.trunk import (EvoformerStack,
                                                             ExtraMSAStack)
from tensorrt_bionemo._torch.modules.openfold2.utils.feats import (
    build_extra_msa_feat, build_extra_msa_feat_multimer, pseudo_beta_fn)
from tensorrt_bionemo._torch.tensor_utils import tensor_tree_map
from tensorrt_bionemo._trt.module_wrappers import EvoformerStackTRT
from tensorrt_bionemo.configs import BaseConfig
from tensorrt_bionemo.hubs import FoldingSupportMatrix as SupMat
from tensorrt_bionemo.hubs import load_weights as load_weights_from_hubs

from ..helper import AcceleratedModules, OptimizedModuleSetterMixin
from .config import PRETRAINED_CONFIG_REGISTRY
from .convert import (convert_hf_evoformer_torch,
                      convert_hf_extra_msa_embedder_torch,
                      convert_hf_extra_msa_stack_torch,
                      convert_hf_input_embedder_torch,
                      convert_hf_recycling_embedder_torch,
                      convert_hf_template_embedder_multimer_torch,
                      convert_hf_template_embedder_torch)


class OpenFold2AcceleratedModules(AcceleratedModules):

    def get_supported_modules(self) -> dict[str, tuple[nn.Module, Callable]]:

        def evoformer_setter(mod: nn.Module, optimized: nn.Module) -> nn.Module:
            org = mod.evoformer
            setattr(mod, "evoformer", optimized)
            return org

        return {
            "evoformer": (EvoformerStackTRT, evoformer_setter),
        }


class OpenFold2(nn.Module, OptimizedModuleSetterMixin):
    # FIXME: Implement this
    def __init__(self,
                 config: BaseConfig = None,
                 include_load_weights: bool = True,
                 model_name: Optional[str] = None):
        super().__init__()
        self.model_name = model_name or SupMat.OpenFold2_PTM1
        self.config = config or self.get_pretrained_config(self.model_name)
        print(self.config)
        self.is_multimer = self.config.is_multimer

        if self.is_multimer:
            self.input_embedder = InputEmbedderMultimer(
                self.config.input_embedder)
        else:
            self.input_embedder = InputEmbedder(self.config.input_embedder)

        self.recycling_embedder = RecyclingEmbedder(
            self.config.recycling_embedder)

        self.extra_msa_embedder = None
        self.extra_msa_stack = None
        if self.config.enable_extra_msa:
            self.extra_msa_fn = build_extra_msa_feat
            if self.is_multimer:
                self.extra_msa_fn = build_extra_msa_feat_multimer
            self.extra_msa_embedder = ExtraMSAEmbedder(
                self.config.extra_msa_embedder)
            self.extra_msa_stack = ExtraMSAStack(
                self.config.trunk.extra_msa_stack)
        self.template_embedder = None
        if self.config.enable_template:
            if self.is_multimer:
                self.template_embedder = TemplateEmbedderMultimer(
                    self.config.template_embedder)
            else:
                self.template_embedder = TemplateEmbedder(
                    self.config.template_embedder)
        self.evoformer = EvoformerStack(self.config.trunk.evoformer_stack)

    def load_weights(self, weights: dict = None):
        if weights is None:
            logger.info(f"Input weights is None, try to load weights from hubs")
            weights = load_weights_from_hubs(name=self.model_name)

        input_embedder_weights = convert_hf_input_embedder_torch(
            config=self.config.input_embedder,
            weights=weights,
            model_name=self.model_name)
        self.input_embedder.load_weights(input_embedder_weights)

        recycling_embedder_weights = convert_hf_recycling_embedder_torch(
            config=self.config.recycling_embedder,
            weights=weights,
            model_name=self.model_name)
        self.recycling_embedder.load_weights(recycling_embedder_weights)

        if self.config.enable_extra_msa:
            extra_msa_embedder_weights = convert_hf_extra_msa_embedder_torch(
                config=self.config.extra_msa_embedder,
                weights=weights,
                model_name=self.model_name)
            self.extra_msa_embedder.load_weights(extra_msa_embedder_weights)

            extra_msa_stack_weights = convert_hf_extra_msa_stack_torch(
                config=self.config.trunk.extra_msa_stack,
                weights=weights,
                model_name=self.model_name)
            self.extra_msa_stack.load_weights(extra_msa_stack_weights)

        if self.config.enable_template:
            if self.is_multimer:
                template_embedder_weights = convert_hf_template_embedder_multimer_torch(
                    config=self.config.template_embedder,
                    weights=weights,
                    model_name=self.model_name)
            else:
                template_embedder_weights = convert_hf_template_embedder_torch(
                    config=self.config.template_embedder,
                    weights=weights,
                    model_name=self.model_name)
            self.template_embedder.load_weights(template_embedder_weights)

        evoformer_weights = convert_hf_evoformer_torch(
            config=self.config.trunk.evoformer_stack,
            weights=weights,
            model_name=self.model_name)
        self.evoformer.load_weights(evoformer_weights)

    @staticmethod
    def get_pretrained_config(model_name: str) -> BaseConfig:
        config_class = PRETRAINED_CONFIG_REGISTRY.get(model_name)
        if config_class is None:
            raise ValueError(
                f"OpenFold2 pretrained config not found for model name: {model_name}"
            )
        return config_class()

    def get_current_batch(self,
                          feed_dict: dict[str, torch.Tensor],
                          cycle_no: int = 0) -> dict[str, torch.Tensor]:
        fetch_current_batch = lambda t: t[..., cycle_no]
        ret = tensor_tree_map(fetch_current_batch, feed_dict)
        return ret

    def get_module_feed_dict(self, feed_dict: dict[str, torch.Tensor],
                             module_name: str) -> dict[str, Any]:
        if module_name == "input_embedder":
            ret = {
                "target_feat": feed_dict["target_feat"],
                "residue_index": feed_dict["residue_index"],
                "msa_feat": feed_dict["msa_feat"],
            }
            if self.is_multimer:
                ret.update({
                    "asym_id": feed_dict["asym_id"],
                    "entity_id": feed_dict["entity_id"],
                    "sym_id": feed_dict["sym_id"],
                })
            return ret
        elif module_name == "extra_msa_feat":
            ret = {
                "extra_msa": feed_dict["extra_msa"],
            }
            if not self.is_multimer:
                ret.update({
                    "extra_has_deletion":
                    feed_dict["extra_has_deletion"],
                    "extra_deletion_value":
                    feed_dict["extra_deletion_value"],
                })
            else:
                ret.update({
                    "extra_deletion_matrix":
                    feed_dict["extra_deletion_matrix"],
                })
            return ret
        else:
            raise ValueError(f"Module name {module_name} not supported")

    def embed_templates(
        self,
        feats: dict[str, torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        templ_dim: int,
        all_reduce_params: Optional[AllReduceParams] = None
    ) -> dict[str, torch.Tensor]:

        template_feats = {
            k: v
            for k, v in feats.items() if k.startswith("template_")
        }
        if self.is_multimer:
            asym_id = feats["asym_id"]
            multichain_mask_2d = (asym_id[..., None] == asym_id[..., None, :])
            template_embeds = self.template_embedder(
                template_feats,
                z,
                pair_mask,
                templ_dim,
                multichain_mask_2d=multichain_mask_2d,
                all_reduce_params=all_reduce_params,
            )
            feats["template_torsion_angles_mask"] = (
                template_embeds["template_mask"])
        else:
            template_embeds = self.template_embedder(
                template_feats,
                z,
                pair_mask,
                templ_dim,
                all_reduce_params=all_reduce_params,
            )
        return template_embeds

    def iteration(
        self,
        feats: dict[str, torch.Tensor],
        prevs: list[torch.Tensor],
        all_reduce_params: Optional[AllReduceParams] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: dict[str, torch.Tensor]
            prevs: list[torch.Tensor]
            all_reduce_params: Optional[AllReduceParams] = None
        Returns:
            tuple[torch.Tensor, torch.Tensor]
        """
        # Prep some features
        batch_dims = feats["target_feat"].shape[:-2]
        no_batch_dims = len(batch_dims)
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]

        m_1_prev, z_prev, x_prev = reversed([prevs.pop() for _ in range(3)])
        m, z = self.input_embedder(**self.get_module_feed_dict(
            feats, "input_embedder"),
                                   all_reduce_params=all_reduce_params)

        pseudo_beta_x_prev = pseudo_beta_fn(feats["aatype"], x_prev, None).to(z)

        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev, z_prev, pseudo_beta_x_prev)

        # [*, S_c, N, C_m]
        m[..., 0, :, :] += m_1_prev_emb

        # [*, N, N, C_z]
        z = z + z_prev_emb

        if self.config.enable_template:
            template_embeds = self.embed_templates(
                feats,
                z,
                pair_mask.to(z),
                no_batch_dims,
                all_reduce_params=all_reduce_params,
            )
            z = z + template_embeds.pop("template_pair_embedding")

            if "template_single_embedding" in template_embeds:
                # [*, S = S_c + S_t, N, C_m]
                m = torch.cat([m, template_embeds["template_single_embedding"]],
                              dim=-3)
                if not self.config.is_multimer:
                    torsion_angles_mask = feats["template_torsion_angles_mask"]
                    msa_mask = torch.cat(
                        [feats["msa_mask"], torsion_angles_mask[..., 2]],
                        dim=-2)
                else:
                    msa_mask = torch.cat(
                        [feats["msa_mask"], template_embeds["template_mask"]],
                        dim=-2,
                    )
        a = None
        if self.config.enable_extra_msa:
            extra_msa_feat = self.extra_msa_fn(
                **self.get_module_feed_dict(feats, "extra_msa_feat"))
            a = self.extra_msa_embedder(extra_msa_feat)

            triangle_metadata_cls = get_attention_backend(
                self.config.trunk.extra_msa_stack.triangle_attention_backend
            ).Metadata

            z = self.extra_msa_stack(
                a,
                z,
                msa_mask=feats["extra_msa_mask"].to(m),
                pair_mask=pair_mask.to(m),
                all_reduce_params=all_reduce_params,
                attn_metadata=triangle_metadata_cls(),
            )

        triangle_metadata_cls = get_attention_backend(
            self.config.trunk.evoformer_stack.triangle_attention_backend
        ).Metadata
        m, z, s = self.evoformer(
            m,
            z,
            msa_mask=msa_mask.to(m),
            pair_mask=pair_mask.to(m),
            all_reduce_params=all_reduce_params,
            attn_metadata=triangle_metadata_cls(),
        )
        # FIXME: return the correct outputs, currently it returns for debugging purposes
        return m_1_prev, z_prev, x_prev, m, z, s, a, m_1_prev_emb, z_prev_emb, msa_mask

    def forward(
        self,
        feed_dict: dict[str, torch.Tensor],
        recycling_steps: int = None,
        all_reduce_params: Optional[AllReduceParams] = None
    ) -> dict[str, torch.Tensor]:
        """ Forward pass for the OpenFold2 model """
        device = next(self.parameters()).device
        batch_dims = feed_dict["target_feat"].shape[:-3]
        n_res = feed_dict["target_feat"].shape[-3]
        feed_dict["msa_feat"].shape[-4]
        num_iters = feed_dict["aatype"].shape[-1]
        if recycling_steps is not None:
            num_iters = min(num_iters, recycling_steps)

        # [*, N_res, C_m]
        m_1_prev = torch.zeros(
            (*batch_dims, n_res, self.config.input_embedder.c_m),
            device=device,
            dtype=self.config.input_embedder.torch_dtype)

        # [*, N_res, N_res, C_z]
        z_prev = torch.zeros(
            (*batch_dims, n_res, n_res, self.config.input_embedder.c_z),
            device=device,
            dtype=self.config.input_embedder.torch_dtype)

        # [*, N_res, 3]
        x_prev = torch.zeros((*batch_dims, n_res, rc.atom_type_num, 3),
                             device=device,
                             dtype=self.config.input_embedder.torch_dtype)

        prevs = [m_1_prev, z_prev, x_prev]

        for cycle_no in range(num_iters):
            batch = self.get_current_batch(feed_dict, cycle_no)
            outputs = self.iteration(batch, prevs, all_reduce_params)

        return outputs
