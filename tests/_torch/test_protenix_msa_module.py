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
"""OSS equivalence tests for the Protenix MSA module (AF3 Algorithm 8)."""

import os
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from bionemo_ir._torch.modules.protenix import ProtenixMSAModule
from bionemo_ir.models.protenix.config import ProtenixMSAModuleConfig
from bionemo_ir.models.protenix.convert import convert_msa_module_torch
from bionemo_ir.utils import str_dtype_to_torch
from tests._torch import skip_if_cutedsl
from tests.common.test_utils.protenix.ref_layers_from_oss import RefProtenixMSAModuleFromOSS


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_token: int = 16
    n_msa: int = 8
    c_m: int = 128
    # Four pair heads satisfy CuTeDSL alignment.
    c_z: int = 128
    c_s_inputs: int = 449
    n_blocks: int = 2
    batch_size: int = 1
    dtype: str = "float32"
    tri_backend: str = "VANILLA"


def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (torch.sqrt(torch.mean((a - b) ** 2)) / (torch.sqrt(torch.mean(b**2)) + 1e-8)).item()


def _config(sc: Scenario) -> ProtenixMSAModuleConfig:
    return ProtenixMSAModuleConfig(
        # The OSS reference runs with row sampling disabled.
        sample_rows=False,
        c_m=sc.c_m,
        c_z=sc.c_z,
        c_hidden_mul=sc.c_z,
        c_hidden_pair_att=32,
        no_heads_pair=sc.c_z // 32,
        no_blocks=sc.n_blocks,
        c_s_inputs=sc.c_s_inputs,
        dtype=sc.dtype,
        triangle_attention_backend=sc.tri_backend,
    )


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(dtype="float32"),
        Scenario(dtype="bfloat16"),
        Scenario(dtype="bfloat16", tri_backend="CuTeDSL"),
    ],
    ids=["fp32", "bf16", "cutedsl"],
)
def test_protenix_msa_module(sc: Scenario):
    skip_if_cutedsl(sc.tri_backend)
    torch.manual_seed(42)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    device = torch.device("cuda")
    torch_dtype = str_dtype_to_torch(sc.dtype)

    ref = (
        RefProtenixMSAModuleFromOSS.build(n_blocks=sc.n_blocks, c_m=sc.c_m, c_z=sc.c_z, c_s_inputs=sc.c_s_inputs)
        .to(device=device, dtype=torch.float32)
        .eval()
    )

    config = _config(sc)
    model = ProtenixMSAModule(config).to(device).eval()
    converted = convert_msa_module_torch(config, ref.state_dict(), prefix="")
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    bs, n, s_msa = sc.batch_size, sc.n_token, sc.n_msa
    feat = {
        "msa": torch.randint(0, 32, (bs, s_msa, n), device=device),
        "has_deletion": (torch.randn(bs, s_msa, n, device=device) > 0).float(),
        "deletion_value": torch.rand(bs, s_msa, n, device=device),
    }
    z = torch.randn(bs, n, n, sc.c_z, device=device)
    s_inputs = torch.randn(bs, n, sc.c_s_inputs, device=device)

    with torch.inference_mode():
        ref_z = ref(feat, z, s_inputs, pair_mask=None)
        act_z = model(
            {k: v.to(torch_dtype) if v.is_floating_point() else v for k, v in feat.items()},
            z.to(torch_dtype),
            s_inputs.to(torch_dtype),
        )

    r = _rmse_ratio(act_z, ref_z)
    tol = 1e-2 if torch_dtype == torch.float32 else 8e-2
    assert r < tol, f"rmse_ratio={r:.3e} exceeds {tol:.0e} ({sc.dtype})"


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_protenix_msa_embedding_gather_matches_dense(dtype_name: str):
    """Embedding + scalar columns matches the original one-hot concat Linear."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = str_dtype_to_torch(dtype_name)
    sc = Scenario(dtype=dtype_name, n_token=12, n_msa=5)
    model = ProtenixMSAModule(_config(sc)).to(device).eval()
    with torch.no_grad():
        model.linear_no_bias_m.weight.normal_(mean=0.0, std=0.02)
        model.linear_no_bias_s.weight.normal_(mean=0.0, std=0.02)

    feat = {
        "msa": torch.randint(0, 32, (sc.batch_size, sc.n_msa, sc.n_token), device=device),
        "has_deletion": (torch.randn(sc.batch_size, sc.n_msa, sc.n_token, device=device) > 0).to(dtype),
        "deletion_value": torch.rand(sc.batch_size, sc.n_msa, sc.n_token, dtype=dtype, device=device),
    }
    s_inputs = torch.randn(sc.batch_size, sc.n_token, sc.c_s_inputs, dtype=dtype, device=device)

    with torch.inference_mode():
        dense_input = torch.cat(
            [
                F.one_hot(feat["msa"], num_classes=32).to(dtype),
                feat["has_deletion"].unsqueeze(-1),
                feat["deletion_value"].unsqueeze(-1),
            ],
            dim=-1,
        )
        expected = model.linear_no_bias_m(dense_input)
        expected = expected + model.linear_no_bias_s(s_inputs).unsqueeze(1)
        actual = model._embed_msa(feat, s_inputs)

    tol = {"atol": 1e-5, "rtol": 1e-5} if dtype == torch.float32 else {"atol": 3e-3, "rtol": 3e-2}
    torch.testing.assert_close(actual, expected, **tol)


def test_protenix_msa_module_precomputed_masks():
    """Precomputed and inline pair masks are bit-identical."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    sc = Scenario(dtype="float32")
    model = ProtenixMSAModule(_config(sc)).to(device).eval()
    # Linear.create_weights() allocates with torch.empty (uninitialized) and no
    # checkpoint is loaded here, so seed every parameter with finite values.
    # Otherwise stray NaNs in uninitialized memory poison both outputs and the
    # bit-identity check compares nan==nan (flaky across CI hosts).
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(mean=0.0, std=0.02)

    bs, n, s_msa = sc.batch_size, sc.n_token, sc.n_msa
    feat = {
        "msa": torch.randint(0, 32, (bs, s_msa, n), device=device),
        "has_deletion": (torch.randn(bs, s_msa, n, device=device) > 0).float(),
        "deletion_value": torch.rand(bs, s_msa, n, device=device),
    }
    z = torch.randn(bs, n, n, sc.c_z, device=device)
    s_inputs = torch.randn(bs, n, sc.c_s_inputs, device=device)
    pair_mask = torch.ones(bs, n, n, device=device)

    with torch.inference_mode():
        out_default = model(feat, z, s_inputs, pair_mask=pair_mask)
        precomputed = model.build_pair_masks(pair_mask)
        out_precomputed = model(feat, z, s_inputs, pair_mask=pair_mask, precomputed_masks=precomputed)
    assert torch.equal(out_default, out_precomputed)


@pytest.mark.parametrize("cutoff", [3, 16384])
def test_protenix_msa_rows_sampled_per_call_from_the_seeded_generator(cutoff):
    """Each call keeps a seeded random subset of rows: 1..N_msa, capped at the cutoff, at a fixed depth."""
    config = _config(Scenario(dtype="float32")).model_copy(update={"sample_rows": True, "sample_cutoff": cutoff})
    module = ProtenixMSAModule(config)
    rows = torch.arange(10).repeat_interleave(4).reshape(1, 10, 4)
    feat = {"msa": rows, "has_deletion": rows.float(), "deletion_value": rows.float()}

    def draws(seed):
        generator = torch.Generator().manual_seed(seed)
        out = []
        for _ in range(8):
            sampled = module._sample_rows(feat, generator)
            assert sampled["msa"].shape == (1, min(10, cutoff), 4)
            out.append(sampled["msa"][0, sampled["msa_row_mask"], 0].tolist())
        return out

    first = draws(7)
    assert first == draws(7)
    assert len({tuple(d) for d in first}) > 1
    for kept in first:
        assert 1 <= len(kept) <= min(10, cutoff)
        assert len(set(kept)) == len(kept) and set(kept) <= set(range(10))


@pytest.mark.parametrize("cutoff", [5, 16384])
@pytest.mark.parametrize("explicit_mask", [False, True])
def test_protenix_msa_masked_rows_match_the_sampled_subset(cutoff, explicit_mask):
    """Masked rows of a fixed-depth draw leave z as the sampled rows alone do."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    sc = Scenario(dtype="float32", n_msa=12)
    config = _config(sc).model_copy(update={"sample_rows": True, "sample_cutoff": cutoff})
    model = ProtenixMSAModule(config).to(device).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(mean=0.0, std=0.02)

    bs, n, s_msa = sc.batch_size, sc.n_token, sc.n_msa
    feat = {
        "msa": torch.randint(0, 32, (bs, s_msa, n), device=device),
        "has_deletion": (torch.randn(bs, s_msa, n, device=device) > 0).float(),
        "deletion_value": torch.rand(bs, s_msa, n, device=device),
    }
    z = torch.randn(bs, n, n, sc.c_z, device=device)
    s_inputs = torch.randn(bs, n, sc.c_s_inputs, device=device)
    mask = torch.randint(0, 2, (bs, s_msa, n), device=device).float() if explicit_mask else None

    partial = 0
    for seed in range(6):
        sampled = model._sample_rows(
            {**feat, **({"msa_mask": mask} if explicit_mask else {})},
            torch.Generator().manual_seed(seed),
            torch.Generator(device=device).manual_seed(seed),
        )
        keep = sampled["msa_row_mask"]
        partial += int(keep.sum()) < keep.numel()
        subset = {name: sampled[name][:, keep] for name in ("msa", "has_deletion", "deletion_value")}
        with torch.inference_mode():
            masked = model(
                feat,
                z,
                s_inputs,
                generator=torch.Generator().manual_seed(seed),
                row_generator=torch.Generator(device=device).manual_seed(seed),
                msa_mask=mask,
            )
            expected = model(
                subset, z, s_inputs, sample_rows=False, msa_mask=sampled["msa_mask"][:, keep] if explicit_mask else None
            )
        torch.testing.assert_close(masked, expected, rtol=1e-5, atol=1e-5)
    assert partial  # At least one draw masks rows.


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("depth", [0, 1, 13])
def test_msa_sampling_matches_upstream(device: str, depth: int) -> None:
    from protenix.model.utils import sample_msa_feature_dict_random_without_replacement

    module = ProtenixMSAModule(_config(Scenario(dtype="float32")))
    rows = torch.arange(depth, device=device).reshape(1, depth, 1).expand(1, depth, 4)
    features = {"msa": rows, "has_deletion": rows.float(), "deletion_value": rows.float()}
    dims = dict.fromkeys(module.input_feature, -2)
    torch.manual_seed(101)
    expected = [sample_msa_feature_dict_random_without_replacement(features, dims, cutoff=5) for _ in range(6)]
    torch.manual_seed(101)
    module.config.sample_cutoff = 5
    for reference in expected:
        sampled = module._sample_rows(features, None)
        for name in dims:
            assert torch.equal(sampled[name][:, sampled["msa_row_mask"]], reference[name])
