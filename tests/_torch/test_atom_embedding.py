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

import os
from dataclasses import dataclass

import pytest
import torch
from test_utils.boltz.create_and_load_weights import create_atom_embedding_weights, load_atom_embedding_weights_torch
from test_utils.boltz.ref_layers import RefAtomEmbedding

from bionemo_ir._torch.layers.sequence_local_atom import build_local_attn_metadata
from bionemo_ir._torch.modules.boltz.embedders import AtomEmbedding
from bionemo_ir.utils import str_dtype_to_torch


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
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    bs = 1
    str_dtype_to_torch(sc.dtype)
    device = torch.device("cuda")

    ref_module = RefAtomEmbedding.load_weights().to(device)
    ref_module.eval()

    weights_and_biases = create_atom_embedding_weights(from_ref=ref_module)

    module = (
        AtomEmbedding(
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
            version=ref_module.version,
        )
        .to(device)
        .eval()
    )

    load_atom_embedding_weights_torch(module, weights_and_biases)

    ref_pos = torch.randn(bs, sc.n_atoms, 3, dtype=torch.float32, device="cuda")
    atom_pad_mask = torch.ones(bs, sc.n_atoms, dtype=torch.float32, device="cuda")
    ref_space_uid = torch.randint(0, 10, (bs, sc.n_atoms), dtype=torch.long, device="cuda")
    ref_charge = torch.randn(bs, sc.n_atoms, dtype=torch.float32, device="cuda")
    ref_element = torch.randint(0, 10, (bs, sc.n_atoms, 128), dtype=torch.long, device="cuda")
    ref_atom_name_chars = torch.randint(0, 10, (bs, sc.n_atoms, 4, 64), dtype=torch.long, device="cuda")
    atom_backbone_feat = torch.randint(0, 10, (bs, sc.n_atoms, 17), dtype=torch.long, device="cuda")
    res_type = torch.randint(0, 33, (bs, sc.n_res, 3), dtype=torch.long, device="cuda")
    modified = torch.randint(0, 2, (bs, sc.n_res), dtype=torch.long, device="cuda")
    mol_type = torch.randint(0, 24, (bs, sc.n_res), dtype=torch.long, device="cuda")
    atom_to_token = torch.randint(0, 10, (bs, sc.n_atoms, sc.n_res), dtype=torch.long, device="cuda")

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
            mol_type=mol_type,
        )

        q, c, p = module(
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
            mol_type=mol_type,
            query_to_keys=to_keys,
        )

        torch.testing.assert_close(ref_q, q)
        torch.testing.assert_close(ref_c, c)
        torch.testing.assert_close(ref_p, p)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("num_windows", [1, 2])
def test_structure_prediction_preserves_single_atom_window(version: str, num_windows: int) -> None:
    """Keep the atom-window axis when structure prediction has one window."""
    batch_size, window_size, key_size = 1, 32, 128
    n_atoms = num_windows * window_size
    n_tokens = num_windows
    atom_s, atom_z, token_s, token_z = 8, 4, 8, 8
    atom_feature_dim = 389 if version == "v1" else 388
    device = torch.device("cuda")

    module = AtomEmbedding(
        atom_s=atom_s,
        atom_z=atom_z,
        token_s=token_s,
        token_z=token_z,
        atoms_per_window_queries=window_size,
        atoms_per_window_keys=key_size,
        atom_feature_dim=atom_feature_dim,
        structure_prediction=True,
        version=version,
    ).to(device)
    for parameter in module.parameters():
        torch.nn.init.normal_(parameter, std=0.02)

    atom_to_token = torch.eye(n_tokens, device=device).repeat_interleave(window_size, dim=0).unsqueeze(0)
    atom_pad_mask = torch.ones(batch_size, n_atoms, device=device)
    if num_windows == 1:
        atom_pad_mask[:, 8:] = 0
        atom_to_token[:, 8:, :] = 0

    with torch.no_grad():
        q, c, p = module(
            atom_to_token=atom_to_token,
            ref_pos=torch.randn(batch_size, n_atoms, 3, device=device),
            atom_pad_mask=atom_pad_mask,
            ref_space_uid=torch.zeros(batch_size, n_atoms, dtype=torch.long, device=device),
            ref_charge=torch.zeros(batch_size, n_atoms, device=device),
            ref_element=torch.zeros(batch_size, n_atoms, 128, device=device),
            ref_atom_name_chars=torch.zeros(batch_size, n_atoms, 4, 64, device=device),
            query_to_keys=build_local_attn_metadata(window_size, key_size).query_to_keys,
            s_trunk=torch.randn(batch_size, n_tokens, token_s, device=device),
            z=torch.randn(batch_size, n_tokens, n_tokens, token_z, device=device),
        )

    assert q.shape == c.shape == (batch_size, n_atoms, atom_s)
    assert p.shape == (batch_size, num_windows, window_size, key_size, atom_z)
    assert torch.isfinite(p).all()
