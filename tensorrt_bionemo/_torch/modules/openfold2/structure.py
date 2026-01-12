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

from typing import Optional, Tuple

import torch
import torch.nn as nn

from tensorrt_bionemo._torch.layers.linear import Linear, TensorParallelMode
from tensorrt_bionemo._torch.modules.openfold2.point_attention import \
    InvariantPointAttention
from tensorrt_bionemo._torch.modules.openfold2.utils.feats import (
    frames_and_literature_positions_to_atom14_pos, torsion_angles_to_frames)
from tensorrt_bionemo._torch.modules.openfold2.utils.geometry.quat_rigid import \
    QuatRigid
from tensorrt_bionemo._torch.modules.openfold2.utils.geometry.rigid_matrix_vector import \
    Rigid3Array
from tensorrt_bionemo._torch.modules.openfold2.utils.rigid_utils import (
    Rigid, Rotation)
from tensorrt_bionemo._torch.tensor_utils import dict_multimap
from tensorrt_bionemo._torch.utils import recursive_calling_load_weights
from tensorrt_bionemo.configs.base import BaseConfig
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.pipeline.models.openfold2.const import (
    restype_atom14_mask, restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group, restype_rigid_group_default_frame)


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self,
                 c_s: int,
                 dtype: torch.dtype = torch.float32,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s,
                             6,
                             bias=True,
                             dtype=dtype,
                             mapping=mapping,
                             skip_create_weights=skip_create_weights)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        update = self.linear(s)

        return update


class AngleResnetBlock(nn.Module):

    def __init__(self,
                 c_hidden,
                 dtype: torch.dtype = torch.float32,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(AngleResnetBlock, self).__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(self.c_hidden,
                               self.c_hidden,
                               bias=True,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=skip_create_weights)
        self.linear_2 = Linear(self.c_hidden,
                               self.c_hidden,
                               bias=True,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=skip_create_weights)

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:

        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial


class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self,
                 c_in,
                 c_hidden,
                 no_blocks,
                 no_angles,
                 epsilon,
                 dtype: torch.dtype = torch.float32,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super(AngleResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = Linear(self.c_in,
                                self.c_hidden,
                                bias=True,
                                dtype=dtype,
                                mapping=mapping,
                                tensor_parallel_mode=TensorParallelMode.COLUMN,
                                gather_output=True,
                                skip_create_weights=skip_create_weights)

        self.linear_initial = Linear(
            self.c_in,
            self.c_hidden,
            bias=True,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden,
                                     dtype=dtype,
                                     mapping=mapping,
                                     skip_create_weights=skip_create_weights)
            self.layers.append(layer)

        self.linear_out = Linear(
            self.c_hidden,
            self.no_angles * 2,
            bias=True,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            skip_create_weights=skip_create_weights)

        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor,
                s_initial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """

        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        s = self.linear_out(s)
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s**2, dim=-1, keepdim=True),
                min=self.eps,
            ))
        s = s / norm_denom

        return unnormalized_s, s


class StructureModuleTransitionLayer(nn.Module):

    def __init__(self,
                 c,
                 dtype: torch.dtype = torch.float32,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False):
        super(StructureModuleTransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c,
                               self.c,
                               bias=True,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=skip_create_weights)

        self.linear_2 = Linear(self.c,
                               self.c,
                               bias=True,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=skip_create_weights)

        self.linear_3 = Linear(self.c,
                               self.c,
                               bias=True,
                               dtype=dtype,
                               mapping=mapping,
                               tensor_parallel_mode=TensorParallelMode.COLUMN,
                               gather_output=True,
                               skip_create_weights=skip_create_weights)

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class StructureModuleTransition(nn.Module):

    def __init__(self,
                 c: int,
                 num_layers: int,
                 dtype: torch.dtype = torch.float32,
                 mapping: Optional[Mapping] = None,
                 skip_create_weights: bool = False,
                 eps: float = 1e-5):
        super(StructureModuleTransition, self).__init__()

        self.c = c
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            l = StructureModuleTransitionLayer(
                self.c,
                dtype=dtype,
                mapping=mapping,
                skip_create_weights=skip_create_weights)
            self.layers.append(l)

        self.layer_norm = nn.LayerNorm(self.c, dtype=dtype, eps=eps)

    def forward(self, s):

        for layer in self.layers:
            s = layer(s)

        s = self.layer_norm(s)

        return s


class StructureModule(nn.Module):

    def __init__(self, config: BaseConfig):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
        """
        super(StructureModule, self).__init__()
        self.config = config

        self.c_s = config.c_s
        self.c_z = config.c_z
        self.c_ipa = config.c_ipa
        self.c_resnet = config.c_resnet
        self.no_heads_ipa = config.no_heads_ipa
        self.no_qk_points = config.no_qk_points
        self.no_v_points = config.no_v_points
        self.no_blocks = config.no_blocks
        self.no_transition_layers = config.no_transition_layers
        self.no_resnet_blocks = config.no_resnet_blocks
        self.no_angles = config.no_angles
        self.trans_scale_factor = config.trans_scale_factor
        self.epsilon = config.epsilon
        self.inf = config.inf
        self.is_multimer = config.is_multimer

        self.layer_norm_s = nn.LayerNorm(self.c_s,
                                         dtype=config.torch_dtype,
                                         eps=self.epsilon)
        self.layer_norm_z = nn.LayerNorm(self.c_z,
                                         dtype=config.torch_dtype,
                                         eps=self.epsilon)

        self.linear_in = Linear(self.c_s,
                                self.c_s,
                                bias=True,
                                dtype=config.torch_dtype,
                                mapping=config.mapping,
                                tensor_parallel_mode=TensorParallelMode.COLUMN,
                                gather_output=True,
                                skip_create_weights=config.skip_create_weights)

        self.ipa = InvariantPointAttention(
            c_s=self.c_s,
            c_z=self.c_z,
            c_hidden=self.c_ipa,
            no_heads=self.no_heads_ipa,
            no_qk_points=self.no_qk_points,
            no_v_points=self.no_v_points,
            inf=self.inf,
            eps=self.epsilon,
            is_multimer=self.is_multimer,
            dtype=config.torch_dtype,
            mapping=config.mapping,
            skip_create_weights=config.skip_create_weights,
        )

        self.layer_norm_ipa = nn.LayerNorm(self.c_s,
                                           dtype=config.torch_dtype,
                                           eps=self.epsilon)

        self.transition = StructureModuleTransition(
            c=self.c_s,
            num_layers=self.no_transition_layers,
            dtype=config.torch_dtype,
            mapping=config.mapping,
            skip_create_weights=config.skip_create_weights,
        )

        if self.is_multimer:
            self.bb_update = QuatRigid(
                c_hidden=self.c_s,
                dtype=config.torch_dtype,
                mapping=config.mapping,
                skip_create_weights=config.skip_create_weights)
        else:
            self.bb_update = BackboneUpdate(
                c_s=self.c_s,
                dtype=config.torch_dtype,
                mapping=config.mapping,
                skip_create_weights=config.skip_create_weights)

        self.angle_resnet = AngleResnet(
            c_in=self.c_s,
            c_hidden=self.c_resnet,
            no_blocks=self.no_resnet_blocks,
            no_angles=self.no_angles,
            epsilon=self.epsilon,
            dtype=config.torch_dtype,
            mapping=config.mapping,
            skip_create_weights=config.skip_create_weights,
        )

        self._init_residue_constants(config.torch_dtype)

    def load_weights(self, weights: dict):

        self.ipa.head_weights.data.copy_(weights["ipa.head_weights"])
        weights.pop("ipa.head_weights")
        filter_func = lambda name, _: name == "ipa"
        loaded_weight = recursive_calling_load_weights(self, weights,
                                                       filter_func)
        not_loaded_weight = set(weights.keys()) - loaded_weight
        if not_loaded_weight:
            raise ValueError(
                f"The following weights are not loaded: {not_loaded_weight}")

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        aatype: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        # cast the tensors to the correct dtype
        s = s.to(dtype=self.config.torch_dtype)
        z = z.to(dtype=self.config.torch_dtype)
        if mask is not None:
            mask = mask.to(dtype=self.config.torch_dtype)

        if (self.is_multimer):
            outputs = self._forward_multimer(s, z, aatype, mask)
        else:
            outputs = self._forward_monomer(s, z, aatype, mask)

        return outputs

    def _forward_monomer(self, s, z, aatype, mask=None):
        """
        Args:
            evoformer_output_dict:
                Dictionary containing:
                    "single":
                        [*, N_res, C_s] single representation
                    "pair":
                        [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """

        if mask is None:
            mask = s.new_ones(s.shape[:-1])

        s = self.layer_norm_s(s)

        z = self.layer_norm_z(z)

        s_initial = s
        s = self.linear_in(s)

        rigids = Rigid.identity(
            s.shape[:-1],
            s.dtype,
            s.device,
            self.training,
            fmt="quat",
        )
        outputs = []
        for i in range(self.no_blocks):
            s = s + self.ipa(s, z, rigids, mask)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            rigids = rigids.compose_q_update_vec(self.bb_update(s))

            # To hew as closely as possible to AlphaFold, we convert our
            # quaternion-based transformations to rotation-matrix ones
            # here
            backb_to_global = Rigid(
                Rotation(rot_mats=rigids.get_rots().get_rot_mats(),
                         quats=None),
                rigids.get_trans(),
            )

            backb_to_global = backb_to_global.scale_translation(
                self.trans_scale_factor)

            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            all_frames_to_global = torsion_angles_to_frames(
                backb_to_global, angles, aatype, self.default_frames)

            pred_xyz = frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global, aatype, self.default_frames,
                self.group_idx, self.atom_mask, self.lit_positions)

            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)

            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
                "states": s,
            }

            outputs.append(preds)

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs

    def _forward_multimer(self, s, z, aatype, mask=None):

        if mask is None:
            mask = s.new_ones(s.shape[:-1])

        s = self.layer_norm_s(s)

        z = self.layer_norm_z(z)

        s_initial = s
        s = self.linear_in(s)

        rigids = Rigid3Array.identity(
            s.shape[:-1],
            s.device,
        )
        outputs = []
        for i in range(self.no_blocks):
            s = s + self.ipa(s, z, rigids, mask)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            self.bb_update(s)
            rigids = rigids @ self.bb_update(s)

            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            all_frames_to_global = torsion_angles_to_frames(
                rigids.scale_translation(self.trans_scale_factor), angles,
                aatype, self.default_frames)

            pred_xyz = frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global, aatype, self.default_frames,
                self.group_idx, self.atom_mask, self.lit_positions)

            preds = {
                "frames":
                rigids.scale_translation(self.trans_scale_factor).to_tensor(),
                "sidechain_frames":
                all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles":
                unnormalized_angles,
                "angles":
                angles,
                "positions":
                pred_xyz
            }

            preds = {k: v.to(dtype=s.dtype) for k, v in preds.items()}

            outputs.append(preds)

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs

    def _init_residue_constants(self, dtype: torch.dtype):

        self.register_buffer(
            "default_frames",
            torch.tensor(
                restype_rigid_group_default_frame,
                dtype=dtype,
                requires_grad=False,
            ))

        self.register_buffer(
            "group_idx",
            torch.tensor(
                restype_atom14_to_rigid_group,
                requires_grad=False,
            ))
        self.register_buffer(
            "atom_mask",
            torch.tensor(
                restype_atom14_mask,
                dtype=dtype,
                requires_grad=False,
            ))
        self.register_buffer(
            "lit_positions",
            torch.tensor(
                restype_atom14_rigid_group_positions,
                dtype=dtype,
                requires_grad=False,
            ))
