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

import os
from dataclasses import dataclass

import pytest
import torch
from tensorrt_llm_lite._utils import str_dtype_to_torch
from test_utils.boltz.create_and_load_weights import (
    create_atom_embedding_weights, load_atom_embedding_weights_torch)
from test_utils.boltz.ref_layers import RefAtomEmbedding

from tensorrt_bionemo._torch.modules.boltz.embedders import AtomEmbedding


@dataclass(kw_only=True, frozen=True)
class Scenario:
    seq_len: int = 32
    dtype: str = "float32"
    atom_window_queries: int = 32
    atom_window_keys: int = 128
    n_atoms: int = 1024
    n_res: int = 128


@pytest.mark.parametrize("sc", [Scenario()])
def test_atom_embedding(sc: Scenario):
    torch.manual_seed(42)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    str_dtype_to_torch(sc.dtype)
    device = torch.device('cuda')

    ref_module = RefAtomEmbedding.load_weights().to(device)
    ref_module.eval()

    weights_and_biases = create_atom_embedding_weights(from_ref=ref_module)

    module = AtomEmbedding(
        atom_s=ref_module.atom_s,
        atom_z=ref_module.atom_z,
        token_s=ref_module.token_s,
        token_z=ref_module.token_z,
        atoms_per_window_queries=sc.atom_window_queries,
        atoms_per_window_keys=sc.atom_window_keys,
        atom_feature_dim=ref_module.atom_feature_dim,
        structure_prediction=ref_module.structure_prediction,
        use_no_atom_char=ref_module.use_no_atom_char,
        use_atom_backbone_feat=ref_module.use_atom_backbone_feat,
        use_residue_feats_atoms=ref_module.use_residue_feats_atoms,
        version=ref_module.version).to(device).eval()

    load_atom_embedding_weights_torch(module, weights_and_biases)

    ref_pos = torch.randn(bs,
                          sc.n_atoms,
                          3,
                          dtype=torch.float32,
                          device="cuda")
    atom_pad_mask = torch.ones(bs,
                               sc.n_atoms,
                               dtype=torch.float32,
                               device="cuda")
    ref_space_uid = torch.randint(0,
                                  10, (bs, sc.n_atoms),
                                  dtype=torch.long,
                                  device="cuda")
    ref_charge = torch.randn(bs,
                             sc.n_atoms,
                             dtype=torch.float32,
                             device="cuda")
    ref_element = torch.randint(0,
                                10, (bs, sc.n_atoms, 128),
                                dtype=torch.long,
                                device="cuda")
    ref_atom_name_chars = torch.randint(0,
                                        10, (bs, sc.n_atoms, 4, 64),
                                        dtype=torch.long,
                                        device="cuda")
    atom_backbone_feat = torch.randint(0,
                                       10, (bs, sc.n_atoms, 17),
                                       dtype=torch.long,
                                       device="cuda")
    res_type = torch.randint(0,
                             33, (bs, sc.n_res, 3),
                             dtype=torch.long,
                             device="cuda")
    modified = torch.randint(0,
                             2, (bs, sc.n_res),
                             dtype=torch.long,
                             device="cuda")
    mol_type = torch.randint(0,
                             24, (bs, sc.n_res),
                             dtype=torch.long,
                             device="cuda")
    atom_to_token = torch.randint(0,
                                  10, (bs, sc.n_atoms, sc.n_res),
                                  dtype=torch.long,
                                  device="cuda")

    with torch.no_grad():
        ref_q, ref_c, ref_p, to_keys = ref_module(
            atom_to_token=atom_to_token,
            ref_pos=ref_pos,
            atom_pad_mask=atom_pad_mask,
            ref_space_uid=ref_space_uid,
            ref_charge=ref_charge,
            ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            atom_backbone_feat=atom_backbone_feat,
            res_type=res_type,
            modified=modified,
            mol_type=mol_type)

        q, c, p = module(atom_to_token=atom_to_token,
                         ref_pos=ref_pos,
                         atom_pad_mask=atom_pad_mask,
                         ref_space_uid=ref_space_uid,
                         ref_charge=ref_charge,
                         ref_element=ref_element,
                         ref_atom_name_chars=ref_atom_name_chars,
                         atom_backbone_feat=atom_backbone_feat,
                         res_type=res_type,
                         modified=modified,
                         mol_type=mol_type,
                         query_to_keys=to_keys)

        torch.testing.assert_close(ref_q, q)
        torch.testing.assert_close(ref_c, c)
        torch.testing.assert_close(ref_p, p)
