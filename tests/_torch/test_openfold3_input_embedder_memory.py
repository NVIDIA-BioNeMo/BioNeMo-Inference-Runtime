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

import pytest
import torch

import bionemo_ir._torch.modules.openfold3.embedders as embedder_module
from bionemo_ir._torch.modules.openfold3.embedders import InputEmbedderAllAtom
from bionemo_ir._torch.modules.openfold3.utils.relpos import relpos_complex
from bionemo_ir._torch.utils import CHUNK_REGISTRY, INPUT_EMBEDDER_PAIR_BUILD


def _pair_builder(device: torch.device, c_z: int = 8) -> InputEmbedderAllAtom:
    module = InputEmbedderAllAtom.__new__(InputEmbedderAllAtom)
    torch.nn.Module.__init__(module)
    module.max_relative_idx = 2
    module.max_relative_chain = 1
    num_relpos_dims = 2 * (2 * module.max_relative_idx + 2) + 2 * module.max_relative_chain + 3
    module.linear_relpos = torch.nn.Linear(num_relpos_dims, c_z, bias=False, device=device)
    module.linear_token_bonds = torch.nn.Linear(1, c_z, bias=False, device=device)
    module.pair_build_chunk_policy = CHUNK_REGISTRY.get(INPUT_EMBEDDER_PAIR_BUILD)
    return module


def _pair_inputs(
    device: torch.device, tokens: int, c_z: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(42)
    embedded = torch.randn(1, tokens, 2 * c_z, device=device)
    embedded_i, embedded_j = embedded.chunk(2, dim=-1)
    batch = {
        name: torch.randint(0, 4, (1, tokens), device=device)
        for name in ("residue_index", "asym_id", "entity_id", "token_index", "sym_id")
    }
    batch["token_bonds"] = torch.randint(0, 2, (1, tokens, tokens), device=device)
    return embedded_i, embedded_j, batch


@pytest.mark.parametrize("pair_output_dtype", [None, torch.bfloat16])
def test_openfold3_input_pair_build_chunks_rows_exactly(
    monkeypatch: pytest.MonkeyPatch,
    pair_output_dtype: torch.dtype | None,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    tokens, c_z = 16, 8
    module = _pair_builder(device, c_z)
    embedded_i, embedded_j, batch = _pair_inputs(device, tokens, c_z)
    policy = CHUNK_REGISTRY.get(INPUT_EMBEDDER_PAIR_BUILD)
    assert policy is not None

    with torch.inference_mode():
        expected = module._build_pair_embeddings_dense(embedded_i, embedded_j, batch)
        if pair_output_dtype is not None:
            expected = expected.to(dtype=pair_output_dtype)

    encoded_rows: list[int] = []

    def record_relative_positions(
        batch: dict[str, torch.Tensor],
        max_relative_idx: int,
        max_relative_chain: int,
        *,
        row_slice: slice | None = None,
    ) -> torch.Tensor:
        result = relpos_complex(batch, max_relative_idx, max_relative_chain, row_slice=row_slice)
        encoded_rows.append(result.shape[-3])
        return result

    monkeypatch.setattr(embedder_module, "relpos_complex", record_relative_positions)
    relpos_projection_rows: list[int] = []
    bond_projection_rows: list[int] = []
    relpos_handle = module.linear_relpos.register_forward_pre_hook(
        lambda _module, inputs: relpos_projection_rows.append(inputs[0].shape[-3])
    )
    bond_handle = module.linear_token_bonds.register_forward_pre_hook(
        lambda _module, inputs: bond_projection_rows.append(inputs[0].shape[-3])
    )
    module.pair_build_chunk_policy = policy.replace(chunk_size=7, min_size=1)
    try:
        with torch.inference_mode():
            actual = module._build_pair_embeddings(
                embedded_i,
                embedded_j,
                batch,
                pair_output_dtype=pair_output_dtype,
            )
    finally:
        relpos_handle.remove()
        bond_handle.remove()

    assert encoded_rows == [7, 7, 2]
    assert relpos_projection_rows == [7, 7, 2]
    assert bond_projection_rows == [7, 7, 2]
    assert actual.dtype == (embedded_i.dtype if pair_output_dtype is None else pair_output_dtype)
    assert torch.equal(actual, expected)
