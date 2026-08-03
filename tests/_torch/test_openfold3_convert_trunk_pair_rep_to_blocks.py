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
"""Equivalence tests for ``convert_pair_atom_to_blocks``.

The TRT-BNM implementation is built to be bit-equivalent to the OSS reference
``convert_trunk_pair_rep_to_blocks`` (in
``test_utils.openfold3.atom_attention_block_utils``). Both:

* pad ``atom_to_token_index`` and ``atom_mask`` with zeros at the left/right
  so that the K-side window for each block sits at flat positions
  ``[k*n_query, k*n_query + n_key)`` of the padded buffer (zeros outside
  ``[0, N_atom)``);
* gather ``zij_trunk`` with those indices (OOB → token 0, then masked out);
* multiply by the atom pair mask.

TRT-BNM uses ``query_to_keys_optimized`` to produce the K-side window —
its sentinel zero row at flat index ``K*n_query`` makes OOB columns gather
zeros, exactly matching the OSS ``F.pad(value=0)`` + ``unfold`` approach.

The two implementations agree bit-exactly when ``N_atom % n_query != 0`` and
the OSS-defined number of blocks ``ceil(N_atom / n_query)`` equals the
production ``K`` (``pad_to_multiple_and_divide`` always rounds up to a full
extra block when divisible, which the OSS ref does not — but this only adds
an all-zero trailing block, semantically a no-op). We test only the
non-divisible case here.
"""

from dataclasses import dataclass
from functools import partial

import pytest
import torch
from test_utils.openfold3.atom_attention_block_utils import \
    convert_trunk_pair_rep_to_blocks

from tensorrt_bionemo._torch.attention_backend.interface import \
    AttentionMetadata
from tensorrt_bionemo._torch.layers.sequence_local_atom import (
    create_gather_indices, pad_to_multiple_and_divide, query_to_keys_optimized)
from tensorrt_bionemo._torch.modules.openfold3.sequence_local_atom_attention import \
    convert_pair_atom_to_blocks


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_tokens: int
    n_atoms: int
    n_queries: int = 32
    n_keys: int = 128
    n_dims: int = 16
    torch_dtype: torch.dtype = torch.float32
    has_sample_dim: bool = False
    full_mask: bool = False


def _build_attn_metadata(atom_mask_2d: torch.Tensor, n_query: int, n_key: int,
                         device: torch.device) -> AttentionMetadata:
    """Match ``OpenFold3.generate_attn_metadata`` so we exercise the production
    code path."""
    mask_blocked, _ = pad_to_multiple_and_divide(atom_mask_2d,
                                                 multiple=n_query,
                                                 dim=1)
    K = mask_blocked.shape[1]
    gather_indices, _ = create_gather_indices(K, n_query, n_key, device)
    query_to_keys_func = partial(query_to_keys_optimized,
                                 gather_indices=gather_indices,
                                 W=n_query,
                                 H=n_key)
    return AttentionMetadata(query_to_keys=query_to_keys_func, bias_cache={})


def _make_inputs(sc: Scenario, device: torch.device):
    """Synthetic data: ~``n_atoms/n_tokens`` atoms per token via
    ``repeat_interleave``, optional sample dim, optional all-ones mask."""
    torch.manual_seed(0)

    atoms_per_tok = max(1, sc.n_atoms // sc.n_tokens)
    a2t = torch.repeat_interleave(torch.arange(sc.n_tokens), atoms_per_tok)
    if a2t.numel() < sc.n_atoms:
        tail = torch.full((sc.n_atoms - a2t.numel(), ),
                          sc.n_tokens - 1,
                          dtype=a2t.dtype)
        a2t = torch.cat([a2t, tail])
    a2t = a2t[:sc.n_atoms].to(device).unsqueeze(0)

    if sc.full_mask:
        atom_mask = torch.ones(1,
                               sc.n_atoms,
                               device=device,
                               dtype=torch.float32)
    else:
        atom_mask = torch.randint(0,
                                  2, (1, sc.n_atoms),
                                  dtype=torch.float32,
                                  device=device)

    zij_trunk = torch.randn(1,
                            sc.n_tokens,
                            sc.n_tokens,
                            sc.n_dims,
                            dtype=sc.torch_dtype,
                            device=device)

    if sc.has_sample_dim:
        a2t = a2t.unsqueeze(1)
        atom_mask = atom_mask.unsqueeze(1)
        zij_trunk = zij_trunk.unsqueeze(1)

    return zij_trunk, a2t, atom_mask


# We deliberately pick (n_atoms, n_query) pairs that are NOT divisible so the
# OSS reference ``num_blocks = ceil(n_atom/n_query)`` matches production K.
_SCENARIOS = [
    Scenario(n_tokens=76, n_atoms=601),
    Scenario(n_tokens=76, n_atoms=601, torch_dtype=torch.bfloat16),
    Scenario(n_tokens=76, n_atoms=601, has_sample_dim=True),
    Scenario(n_tokens=400, n_atoms=3201),
    # The previously-broken regression: n_tokens in TF32 unit-resolution range.
    Scenario(n_tokens=2552, n_atoms=19955, full_mask=True),
    Scenario(n_tokens=2552, n_atoms=19955, has_sample_dim=True,
             full_mask=True),
]
_SCENARIO_IDS = [
    "small-float32",
    "small-bfloat16",
    "small-with-sample-dim",
    "mid-size",
    "large-tf32-regression",
    "large-tf32-regression-sample-dim",
]


@pytest.mark.parametrize("sc", _SCENARIOS, ids=_SCENARIO_IDS)
def test_matches_oss_reference(sc: Scenario):
    """Production must match the OSS reference bit-exactly (up to dtype
    rounding for bfloat16)."""
    device = torch.device("cuda")
    zij_trunk, a2t, atom_mask = _make_inputs(sc, device)

    flat_mask = atom_mask.reshape(-1, sc.n_atoms)
    attn_metadata = _build_attn_metadata(flat_mask,
                                         sc.n_queries,
                                         sc.n_keys,
                                         device=device)

    batch = {"atom_to_token_index": a2t, "atom_mask": atom_mask}
    output = convert_pair_atom_to_blocks(batch=batch,
                                         zij_trunk=zij_trunk,
                                         n_query=sc.n_queries,
                                         n_key=sc.n_keys,
                                         attn_metadata=attn_metadata)
    torch.cuda.synchronize()

    # Reference path expects no sample dim; squeeze and re-insert for compare.
    if sc.has_sample_dim:
        ref_output = convert_trunk_pair_rep_to_blocks(
            batch={
                "atom_to_token_index": a2t.squeeze(1),
                "atom_mask": atom_mask.squeeze(1),
            },
            zij_trunk=zij_trunk.squeeze(1),
            n_query=sc.n_queries,
            n_key=sc.n_keys,
        ).unsqueeze(1)
    else:
        ref_output = convert_trunk_pair_rep_to_blocks(batch=batch,
                                                      zij_trunk=zij_trunk,
                                                      n_query=sc.n_queries,
                                                      n_key=sc.n_keys)

    assert output.shape == ref_output.shape, (output.shape, ref_output.shape)
    assert torch.isfinite(output).all()
    assert torch.isfinite(ref_output).all()

    # bfloat16 path: the OSS reference auto-promotes to float32 via the
    # ``bfloat16 * float_mask`` multiply, while production explicitly casts
    # back to plm.dtype. Compare in a common (bfloat16) dtype with tolerance.
    if sc.torch_dtype == torch.bfloat16:
        torch.testing.assert_close(output,
                                   ref_output.to(torch.bfloat16),
                                   atol=1e-2,
                                   rtol=1e-2)
    else:
        torch.testing.assert_close(output, ref_output, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("seq_len", [128, 200, 400, 1024, 2048, 2552, 3000])
def test_no_oob_sweep(seq_len: int):
    """Sweep over sequence lengths spanning the TF32 unit-resolution range
    (2048-4096), which used to trigger CUDA device-side asserts in the einsum
    path. The gather-based ``query_to_keys_optimized`` must run cleanly."""
    device = torch.device("cuda")
    sc = Scenario(
        n_tokens=seq_len,
        n_atoms=seq_len * 8 - 5,  # not divisible by n_query
        full_mask=True,
        n_dims=8)
    zij_trunk, a2t, atom_mask = _make_inputs(sc, device)

    flat_mask = atom_mask.reshape(-1, sc.n_atoms)
    attn_metadata = _build_attn_metadata(flat_mask,
                                         sc.n_queries,
                                         sc.n_keys,
                                         device=device)
    batch = {"atom_to_token_index": a2t, "atom_mask": atom_mask}
    output = convert_pair_atom_to_blocks(batch=batch,
                                         zij_trunk=zij_trunk,
                                         n_query=sc.n_queries,
                                         n_key=sc.n_keys,
                                         attn_metadata=attn_metadata)
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
