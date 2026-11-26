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
from functools import partial

import pytest
import torch
from test_utils.boltz.create_and_load_weights import create_atom_embedding_weights, load_atom_embedding_weights_torch
from test_utils.boltz.ref_layers import RefAtomEmbedding

from bionemo_ir._torch.layers.sequence_local_atom import (
    build_local_attn_metadata,
    create_gather_indices,
    create_indexing_matrix,
    query_to_keys,
    query_to_keys_optimized,
)
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
@pytest.mark.parametrize("q2k_layout", ["metadata_4d", "legacy_5d"])
def test_structure_prediction_preserves_single_atom_window(
    version: str,
    num_windows: int,
    q2k_layout: str,
) -> None:
    """Keep the atom-window axis when structure prediction has one window."""
    torch.manual_seed(0)
    batch_size, window_size, key_size = 1, 32, 128
    n_atoms = num_windows * window_size
    n_tokens = max(2, num_windows * 2)
    atom_s, atom_z, token_s, token_z = 8, 4, 8, 8
    atom_feature_dim = 389 if version == "v1" else 388
    device = torch.device("cuda")

    module = (
        AtomEmbedding(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=window_size,
            atoms_per_window_keys=key_size,
            atom_feature_dim=atom_feature_dim,
            structure_prediction=True,
            version=version,
        )
        .to(device)
        .eval()
    )
    with torch.no_grad():
        for parameter in module.parameters():
            torch.nn.init.normal_(parameter, std=0.05)

    atom_to_token = torch.zeros(batch_size, n_atoms, n_tokens, device=device)
    atom_pad_mask = torch.ones(batch_size, n_atoms, device=device)
    valid_atoms = 8 if num_windows == 1 else n_atoms
    for atom_idx in range(valid_atoms):
        atom_to_token[:, atom_idx, atom_idx % n_tokens] = 1.0
    if valid_atoms < n_atoms:
        atom_pad_mask[:, valid_atoms:] = 0.0

    ref_pos = torch.randn(batch_size, n_atoms, 3, device=device)
    ref_space_uid = torch.zeros(batch_size, n_atoms, dtype=torch.long, device=device)
    ref_charge = torch.randn(batch_size, n_atoms, device=device)
    ref_element = torch.zeros(batch_size, n_atoms, 128, device=device)
    ref_element[:, :, 6] = 1.0
    ref_atom_name_chars = torch.zeros(batch_size, n_atoms, 4, 64, device=device)
    s_trunk = torch.randn(batch_size, n_tokens, token_s, device=device)
    z = torch.randn(batch_size, n_tokens, n_tokens, token_z, device=device)

    if q2k_layout == "metadata_4d":
        q2k = build_local_attn_metadata(window_size, key_size).query_to_keys
    else:
        idx_mat = create_indexing_matrix(num_windows, window_size, key_size, device)
        q2k = partial(query_to_keys, keys_indexing_matrix=idx_mat, W=window_size, H=key_size)

    with torch.no_grad():
        q, c, p = module(
            atom_to_token=atom_to_token,
            ref_pos=ref_pos,
            atom_pad_mask=atom_pad_mask,
            ref_space_uid=ref_space_uid,
            ref_charge=ref_charge,
            ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            query_to_keys=q2k,
            s_trunk=s_trunk,
            z=z,
        )

        # Reference check with explicit 4D query_to_keys_optimized gather for z_to_p.
        gather_indices, _ = create_gather_indices(num_windows, window_size, key_size, device)
        ref_q2k = partial(query_to_keys_optimized, gather_indices=gather_indices, W=window_size, H=key_size)
        q_ref, c_ref, p_ref = module(
            atom_to_token=atom_to_token,
            ref_pos=ref_pos,
            atom_pad_mask=atom_pad_mask,
            ref_space_uid=ref_space_uid,
            ref_charge=ref_charge,
            ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            query_to_keys=ref_q2k,
            s_trunk=s_trunk,
            z=z,
        )
        # Verify z actually contributes to p through the atom_to_token_keys einsum.
        _, _, p_zero_z = module(
            atom_to_token=atom_to_token,
            ref_pos=ref_pos,
            atom_pad_mask=atom_pad_mask,
            ref_space_uid=ref_space_uid,
            ref_charge=ref_charge,
            ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            query_to_keys=q2k,
            s_trunk=s_trunk,
            z=torch.zeros_like(z),
        )

    assert q.shape == c.shape == (batch_size, n_atoms, atom_s)
    assert p.shape == (batch_size, num_windows, window_size, key_size, atom_z)
    assert torch.isfinite(p).all()
    torch.testing.assert_close(q, q_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(c, c_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(p, p_ref, atol=1e-5, rtol=1e-5)
    assert (p - p_zero_z).abs().max().item() > 0.0
