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

from tensorrt_bionemo._torch.modules.boltz.confidence import Boltz1ConfidenceHeads, Boltz2ConfidenceModule
from tensorrt_bionemo._torch.modules.boltz.confidence_utils import compute_contact_prob, repeat_with_multiplicity
from tensorrt_bionemo.models.boltz1.config import ConfidenceHeadsConfig as Boltz1ConfidenceHeadsConfig
from tensorrt_bionemo.models.boltz2.config import ConfidenceModuleConfig
from tensorrt_bionemo.pipeline.models.boltz2.const import contact_conditioning_info

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="confidence module builds weights on CUDA")

N_TOKENS = 6
N_ATOMS = 12
TOKEN_S = 32
TOKEN_Z = 16
NUM_DIST_BINS = 64


def _initialize_unloaded_weights(module: torch.nn.Module) -> torch.nn.Module:
    """Fill checkpoint-backed parameters with finite, deterministic test values."""
    torch.manual_seed(0)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if name.endswith("weight"):
                parameter.normal_(mean=0.0, std=0.02)
            else:
                parameter.zero_()
        for submodule in module.modules():
            if isinstance(submodule, torch.nn.LayerNorm):
                submodule.reset_parameters()
    return module


def _tiny_module(device):
    config = ConfidenceModuleConfig().copy_and_validate(token_s=TOKEN_S, token_z=TOKEN_Z, num_dist_bins=NUM_DIST_BINS)
    config.pairformer = config.pairformer.copy_and_validate(
        token_s=TOKEN_S,
        token_z=TOKEN_Z,
        num_blocks=1,
        num_heads=2,
        pairwise_num_heads=2,
        pairwise_head_width=8,
    )
    config.confidence_heads = config.confidence_heads.copy_and_validate(token_s=TOKEN_S, token_z=TOKEN_Z)
    # Linear builds its weights on CUDA already; the norms need moving explicitly.
    return _initialize_unloaded_weights(Boltz2ConfidenceModule(config, dtype=torch.float32).eval().to(device))


def _tiny_feats(device):
    token_to_rep_atom = torch.zeros(1, N_TOKENS, N_ATOMS, device=device)
    token_to_rep_atom[0, torch.arange(N_TOKENS), torch.arange(N_TOKENS)] = 1.0
    idx = torch.arange(N_TOKENS, device=device).unsqueeze(0)
    return {
        "token_pad_mask": torch.ones(1, N_TOKENS, device=device),
        "token_to_rep_atom": token_to_rep_atom,
        # Two chains, so the interface / different-chain branches are exercised.
        "asym_id": (idx >= N_TOKENS // 2).long(),
        "residue_index": idx,
        "entity_id": torch.zeros(1, N_TOKENS, dtype=torch.long, device=device),
        "cyclic_period": torch.zeros(1, N_TOKENS, device=device),
        "token_index": idx,
        "sym_id": torch.zeros(1, N_TOKENS, dtype=torch.long, device=device),
        "token_bonds": torch.zeros(1, N_TOKENS, N_TOKENS, 1, device=device),
        "type_bonds": torch.zeros(1, N_TOKENS, N_TOKENS, dtype=torch.long, device=device),
        "contact_conditioning": torch.zeros(1, N_TOKENS, N_TOKENS, len(contact_conditioning_info), device=device),
        "contact_threshold": torch.zeros(1, N_TOKENS, N_TOKENS, device=device),
        "mol_type": torch.zeros(1, N_TOKENS, dtype=torch.long, device=device),
        "frames_idx": torch.zeros(1, N_TOKENS, 3, dtype=torch.long, device=device),
        "atom_to_token": token_to_rep_atom.transpose(1, 2).contiguous(),
        "atom_pad_mask": torch.ones(1, N_ATOMS, device=device),
    }


def _run(module, prob_contact, device, multiplicity=1, run_sequentially=True, max_parallel_samples=1):
    torch.manual_seed(0)
    return module(
        s_inputs=torch.randn(1, N_TOKENS, TOKEN_S, device=device),
        s=torch.randn(1, N_TOKENS, TOKEN_S, device=device),
        z=torch.randn(1, N_TOKENS, N_TOKENS, TOKEN_Z, device=device),
        x_pred=torch.randn(1, multiplicity, N_ATOMS, 3, device=device),
        feats=_tiny_feats(device),
        prob_contact=prob_contact,
        multiplicity=multiplicity,
        run_sequentially=run_sequentially,
        max_parallel_samples=max_parallel_samples,
    )


@pytest.mark.parametrize("multiplicity", [1, 2])
def test_forward_consumes_reduced_prob_contact(multiplicity):
    """The module takes the [B, N, N] contact probability, not the [B, N, N, num_bins] logits."""
    device = torch.device("cuda")
    module = _tiny_module(device)

    torch.manual_seed(1)
    logits = torch.randn(1, N_TOKENS, N_TOKENS, NUM_DIST_BINS, device=device)
    prob_contact = compute_contact_prob(logits)
    assert prob_contact.shape == (1, N_TOKENS, N_TOKENS)

    out = _run(module, prob_contact, device, multiplicity=multiplicity)

    for key in ("pde", "plddt", "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"):
        value = out[key]
        assert value.shape[:2] == (1, multiplicity), key
        assert torch.isfinite(value).all(), key


@pytest.mark.parametrize(
    ("run_sequentially", "max_parallel_samples", "expected_iterations"),
    [(True, 1, 1), (True, 3, 1), (False, 1, 2)],
)
def test_run_sequentially_controls_iteration_count(run_sequentially, max_parallel_samples, expected_iterations):
    device = torch.device("cuda")
    module = _tiny_module(device)
    iterations = []
    hook = module.pairformer_stack.register_forward_pre_hook(lambda *_args: iterations.append(None))

    try:
        _run(
            module,
            torch.rand(1, N_TOKENS, N_TOKENS, device=device),
            device,
            multiplicity=2,
            run_sequentially=run_sequentially,
            max_parallel_samples=max_parallel_samples,
        )
    finally:
        hook.remove()

    assert len(iterations) == expected_iterations


def test_prob_contact_weights_the_pde_aggregate():
    """``prob_contact`` must land as the pair weight of the gPDE average, off-diagonal only."""
    device = torch.device("cuda")
    multiplicity = 2
    module = _tiny_module(device)

    torch.manual_seed(1)
    prob_contact = torch.rand(1, N_TOKENS, N_TOKENS, device=device)
    out = _run(module, prob_contact, device, multiplicity=multiplicity)

    off_diagonal = 1 - torch.eye(N_TOKENS, device=device)
    token_pair_mask = off_diagonal * repeat_with_multiplicity(prob_contact, multiplicity)
    expected = (out["pde"] * token_pair_mask).sum(dim=(2, 3)) / token_pair_mask.sum(dim=(2, 3))

    torch.testing.assert_close(out["complex_pde"], expected)


def test_boltz1_heads_consume_reduced_prob_contact():
    """Boltz1's heads take the same reduced input, and the reduction is exact against the old form."""
    device = torch.device("cuda")
    multiplicity = 2
    config = Boltz1ConfidenceHeadsConfig().copy_and_validate(token_s=TOKEN_S, token_z=TOKEN_Z, compute_pae=False)
    heads = _initialize_unloaded_weights(Boltz1ConfidenceHeads(config).eval().to(device))

    torch.manual_seed(0)
    idx = torch.arange(N_TOKENS, device=device).unsqueeze(0)
    feature_dict = {
        "asym_id": (idx >= N_TOKENS // 2).long(),
        "token_pad_mask": torch.ones(1, N_TOKENS, device=device),
        "mol_type": torch.zeros(1, N_TOKENS, dtype=torch.long, device=device),
    }
    logits = torch.randn(1, N_TOKENS, N_TOKENS, NUM_DIST_BINS, device=device)

    # What the heads used to compute internally: repeat the logits, softmax, mask, sum.
    contacts = torch.zeros(1, 1, 1, 1, NUM_DIST_BINS, device=device)
    contacts[..., :20] = 1.0
    legacy = (repeat_with_multiplicity(torch.softmax(logits.float(), dim=-1), multiplicity) * contacts).sum(-1)
    torch.testing.assert_close(
        repeat_with_multiplicity(compute_contact_prob(logits), multiplicity), legacy, rtol=0, atol=0
    )

    out = heads(
        s=torch.randn(1, multiplicity, N_TOKENS, TOKEN_S, device=device),
        z=torch.randn(1, multiplicity, N_TOKENS, N_TOKENS, TOKEN_Z, device=device),
        x_pred=torch.randn(1, multiplicity, N_TOKENS, 3, device=device),
        d=torch.rand(1, multiplicity, N_TOKENS, N_TOKENS, device=device) * 16,
        prob_contact=compute_contact_prob(logits),
        feature_dict=feature_dict,
    )
    for key in ("pde", "plddt", "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"):
        assert torch.isfinite(out[key]).all(), key
