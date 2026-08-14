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
"""OSS equivalence and integration tests for Protenix diffusion (AF3 Algorithm 20).

Real-checkpoint tests skip when the checkpoint is unavailable.
"""

import functools
import os
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from bionemo_ir._torch.graph_optimization.config import CUDAGraphOptimizationConfig, GraphOptimizationMode
from bionemo_ir._torch.graph_optimization.cuda_graph.runtime import (
    CUDAGraphOptimizationTracker,
    CUDAGraphPreparationState,
)
from bionemo_ir._torch.layers.transformers.diffusion_transformer import ProtenixDiffusionTransformer
from bionemo_ir._torch.modules.protenix import ProtenixDiffusionModule, ProtenixSampleDiffusion
from bionemo_ir.configs import DiffusionTransformerConfig
from bionemo_ir.hubs import FoldingSupportMatrix as SupMat
from bionemo_ir.hubs import load_weights as load_weights_from_hubs
from bionemo_ir.models.protenix.config import (
    AtomAttentionDecoderConfig,
    DiffusionAtomAttentionEncoderConfig,
    DiffusionConditioningConfig,
    DiffusionModuleConfig,
    RelativePositionEncodingConfig,
)
from bionemo_ir.models.protenix.convert import convert_diffusion_module_torch
from bionemo_ir.utils import str_dtype_to_torch
from tests._torch import skip_if_cutedsl
from tests.common.test_utils.protenix.ref_layers_from_oss import (
    RefProtenixDiffusionModuleFromOSS,
    update_input_feature_dict,
)


@dataclass(kw_only=True, frozen=True)
class Scenario:
    n_atom: int = 40
    n_token: int = 10
    n_sample: int = 2
    c_token: int = 32
    c_atom: int = 128
    c_atompair: int = 16
    c_s: int = 32
    c_z: int = 16
    c_s_inputs: int = 16
    c_noise: int = 256  # OSS DiffusionModule hardcodes c_noise_embedding=256
    atom_n_blocks: int = 2
    atom_n_heads: int = 4
    token_n_blocks: int = 2
    token_n_heads: int = 4
    n_queries: int = 32
    n_keys: int = 128
    sigma_data: float = 16.0
    dtype: str = "float32"
    # Score-model components may override the outer dtype.
    token_dtype: str = None
    enc_dtype: str = None
    dec_dtype: str = None
    z_pair_dtype: str = "bfloat16"


def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return (torch.sqrt(torch.mean((a - b) ** 2)) / (torch.sqrt(torch.mean(b**2)) + 1e-8)).item()


def _atom_tc(sc: Scenario, dtype: str) -> DiffusionTransformerConfig:
    return DiffusionTransformerConfig(
        num_blocks=sc.atom_n_blocks,
        num_heads=sc.atom_n_heads,
        dim=sc.c_atom,
        dim_single_cond=sc.c_atom,
        dim_pairwise=sc.c_atompair,
        bias_proj=True,
        pair_norm=True,
        initial_norm=False,
        attention_initial_norm=False,
        use_ada_layer_norm=True,
        use_separate_layer_norm=True,
        chain_kv_norm=True,
        attn_output_gate=True,
        conditioned_transition_using_silu=True,
        transition_expansion_factor=2,
        pairwise_attention_backend="SDPA",
        dtype=dtype,
    )


def _token_tc(
    sc: Scenario, dtype: str, backend: str = "SDPA", precompute_bias: bool = True
) -> DiffusionTransformerConfig:
    return DiffusionTransformerConfig(
        num_blocks=sc.token_n_blocks,
        num_heads=sc.token_n_heads,
        dim=sc.c_token,
        dim_single_cond=sc.c_s,
        dim_pairwise=sc.c_z,
        bias_proj=True,
        pair_norm=True,
        initial_norm=True,
        attention_initial_norm=False,
        use_ada_layer_norm=True,
        use_separate_layer_norm=False,
        chain_kv_norm=False,
        attn_output_gate=True,
        conditioned_transition_using_silu=True,
        transition_expansion_factor=2,
        precompute_bias=precompute_bias,
        pairwise_attention_backend=backend,
        dtype=dtype,
    )


def _build_config(sc: Scenario) -> DiffusionModuleConfig:
    d = sc.dtype
    td = sc.token_dtype or sc.dtype
    ed = sc.enc_dtype or sc.dtype
    dd = sc.dec_dtype or sc.dtype
    return DiffusionModuleConfig(
        c_s=sc.c_s,
        c_z=sc.c_z,
        c_token=sc.c_token,
        n_queries=sc.n_queries,
        n_keys=sc.n_keys,
        sigma_data=sc.sigma_data,
        diffusion_conditioning_config=DiffusionConditioningConfig(
            c_s=sc.c_s,
            c_z=sc.c_z,
            c_s_inputs=sc.c_s_inputs,
            c_noise_embedding=sc.c_noise,
            relpe_config=RelativePositionEncodingConfig(c_z=sc.c_z),
            z_pair_dtype=sc.z_pair_dtype,
            dtype=d,
        ),
        atom_encoder_config=DiffusionAtomAttentionEncoderConfig(
            c_token=sc.c_token,
            c_atom=sc.c_atom,
            c_atompair=sc.c_atompair,
            c_s=sc.c_s,
            c_z=sc.c_z,
            n_queries=sc.n_queries,
            n_keys=sc.n_keys,
            atom_transformer_config=_atom_tc(sc, ed),
            dtype=ed,
        ),
        token_transformer_config=_token_tc(sc, td),
        atom_decoder_config=AtomAttentionDecoderConfig(
            c_token=sc.c_token,
            c_atom=sc.c_atom,
            c_atompair=sc.c_atompair,
            n_queries=sc.n_queries,
            n_keys=sc.n_keys,
            atom_transformer_config=_atom_tc(sc, dd),
            dtype=dd,
        ),
        dtype=d,
    )


def _make_features(device: torch.device, ref, sc: Scenario) -> dict:
    torch.manual_seed(11)
    n_atom, n_token = sc.n_atom, sc.n_token
    atom_to_token_idx = (torch.arange(n_atom, device=device) // (n_atom // n_token)).long()
    element_idx = torch.randint(0, 128, (n_atom,), device=device)
    name_chars_idx = torch.randint(0, 64, (n_atom, 4), device=device)
    feats = {
        "atom_to_token_idx": atom_to_token_idx,
        "ref_pos": torch.randn(n_atom, 3, device=device),
        "ref_charge": torch.zeros(n_atom, device=device),
        "ref_mask": torch.ones(n_atom, device=device),
        "ref_element": F.one_hot(element_idx, 128).float(),
        "ref_atom_name_chars": F.one_hot(name_chars_idx, 64).float(),
        "ref_space_uid": atom_to_token_idx.clone(),
    }
    feats = update_input_feature_dict(feats)
    tok = torch.arange(n_token, device=device)
    zero = torch.zeros(n_token, dtype=torch.long, device=device)
    feats["relp"] = ref.diffusion_conditioning.relpe.generate_relp(
        {
            "asym_id": zero,
            "residue_index": tok.long(),
            "entity_id": zero,
            "token_index": tok.long(),
            "sym_id": zero,
        }
    )["relp"].float()
    return feats


@pytest.mark.parametrize("backend", ["SDPA", "CuTeDSL"])
@pytest.mark.parametrize("precompute_bias", [False, True], ids=["per_layer_bias", "mega_bias"])
def test_token_transformer_broadcasts_sample_independent_pair(backend: str, precompute_bias: bool):
    """Match sample-independent pair broadcast to explicit ``B*S`` expansion."""
    skip_if_cutedsl(backend)
    torch.manual_seed(123)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    sc = Scenario(c_token=128, c_s=128, c_z=16, token_n_blocks=2, token_n_heads=4, dtype="bfloat16")
    model = (
        ProtenixDiffusionTransformer(_token_tc(sc, "bfloat16", backend=backend, precompute_bias=precompute_bias))
        .to(device)
        .eval()
    )

    # Deterministic non-zero weights; preserve LayerNorm scales near one.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("weight") and "norm" in name:
                param.fill_(1)
            elif param.ndim > 1:
                param.normal_(mean=0.0, std=0.02)
            else:
                param.zero_()

    B, S, N = 1, 3, 40
    a = torch.randn(B, S, N, sc.c_token, device=device, dtype=dtype) * 0.1
    s = torch.randn(B, S, N, sc.c_s, device=device, dtype=dtype) * 0.1
    z = torch.randn(B, N, N, sc.c_z, device=device, dtype=dtype) * 0.1
    mask = torch.ones(B, N, device=device, dtype=dtype)

    with torch.inference_mode():
        broadcast = model(a.reshape(B * S, N, sc.c_token), s.reshape(B * S, N, sc.c_s), z, mask).reshape(
            B, S, N, sc.c_token
        )

        # Explicit baseline replicates pair inputs over samples.
        explicit = model(
            a.reshape(B * S, N, sc.c_token),
            s.reshape(B * S, N, sc.c_s),
            z.unsqueeze(1).expand(B, S, N, N, sc.c_z).reshape(B * S, N, N, sc.c_z).contiguous(),
            mask.unsqueeze(1).expand(B, S, N).reshape(B * S, N).contiguous(),
        ).reshape(B, S, N, sc.c_token)

    assert broadcast.shape == (B, S, N, sc.c_token)
    assert torch.isfinite(broadcast).all()
    r = _rmse_ratio(broadcast, explicit)
    tol = 1e-6 if precompute_bias else 2e-2
    assert r < tol, (
        f"{backend} pair-bias broadcast diverged from explicit expansion: "
        f"rmse={r:.3e} exceeds {tol:.0e}, "
        f"precompute_bias={precompute_bias}"
    )


# Full protenix-v2 diffusion dimensions.
_FULL = {
    "sigma_data": 16.0,
    "c_atom": 128,
    "c_atompair": 16,
    "c_token": 768,
    "c_s": 384,
    "c_z": 256,
    "c_s_inputs": 449,
    "atom_n_blocks": 3,
    "atom_n_heads": 4,
    "token_n_blocks": 24,
    "token_n_heads": 16,
}


@functools.lru_cache(maxsize=1)
def _diffusion_weights() -> dict:
    """Load protenix-v2 weights from the configured hub."""
    return load_weights_from_hubs(SupMat.ProtenixV2, local_files_only=False)


def _full_config(sc: Scenario) -> DiffusionModuleConfig:
    """Full protenix-v2 ``DiffusionModuleConfig`` with per-sub-module dtypes."""
    config = DiffusionModuleConfig(dtype=sc.dtype)
    config.diffusion_conditioning_config.set_dtype(sc.dtype)
    config.diffusion_conditioning_config.z_pair_dtype = sc.z_pair_dtype
    config.atom_encoder_config.set_dtype(sc.enc_dtype or sc.dtype)
    config.token_transformer_config.set_dtype(sc.token_dtype or sc.dtype)
    config.atom_decoder_config.set_dtype(sc.dec_dtype or sc.dtype)
    return config


@pytest.fixture(scope="module")
def real_case():
    """Build the full-size OSS reference case once."""
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    torch.manual_seed(42)
    try:
        weights = _diffusion_weights()
    except Exception as exc:  # offline / hub unreachable
        pytest.skip(f"protenix-v2 checkpoint unavailable: {exc}")
    device = torch.device("cuda")

    ref = RefProtenixDiffusionModuleFromOSS.build(**_FULL).to(device=device, dtype=torch.float32).eval()
    oss_w = {k[len("diffusion_module.") :]: v for k, v in weights.items() if k.startswith("diffusion_module.")}
    ref.load_state_dict(oss_w, strict=True)

    sc = Scenario()  # n_atom=40, n_token=10, n_sample=2
    feats = _make_features(device, ref, sc)
    x_noisy = torch.randn(sc.n_sample, sc.n_atom, 3, device=device)
    t = torch.rand(sc.n_sample, device=device) * 20.0 + 0.5
    s_inputs = torch.randn(sc.n_token, _FULL["c_s_inputs"], device=device)
    s_trunk = torch.randn(sc.n_token, _FULL["c_s"], device=device)
    z_trunk = torch.randn(sc.n_token, sc.n_token, _FULL["c_z"], device=device)
    with torch.inference_mode():
        exp = ref(
            x_noisy, t, feats, s_inputs, s_trunk, z_trunk, None, None, None
        )  # OSS: pair_z / p_lm / c_l = None -> computed fresh
    return {
        "weights": weights,
        "exp": exp,
        "feats": feats,
        "x_noisy": x_noisy,
        "t": t,
        "s_inputs": s_inputs,
        "s_trunk": s_trunk,
        "z_trunk": z_trunk,
        "device": device,
    }


def _real_bioir_module(sc: Scenario, real_case):
    """Build the ported module and batched inputs from ``real_case``."""
    device = real_case["device"]
    torch_dtype = str_dtype_to_torch(sc.dtype)
    config = _full_config(sc)
    model = ProtenixDiffusionModule(config).to(device).eval()
    converted = convert_diffusion_module_torch(config, real_case["weights"], prefix="diffusion_module")
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert not missing and not unexpected, (list(missing)[:5], list(unexpected)[:5])

    feats = real_case["feats"]

    def _bt(x):  # add batch dim, cast floats to the module dtype
        x = x.unsqueeze(0)
        return x.to(torch_dtype) if x.is_floating_point() else x

    batch = {
        k: _bt(feats[k])
        for k in (
            "atom_to_token_idx",
            "ref_pos",
            "ref_charge",
            "ref_mask",
            "ref_atom_name_chars",
            "ref_element",
            "d_lm",
            "v_lm",
            "relp",
        )
    }
    batch["pad_info"] = feats["pad_info"]  # unbatched (broadcasts)
    s_inputs = real_case["s_inputs"].unsqueeze(0).to(torch_dtype)
    s_trunk = real_case["s_trunk"].unsqueeze(0).to(torch_dtype)
    z_trunk = real_case["z_trunk"].unsqueeze(0).to(torch_dtype)
    return model, batch, s_inputs, s_trunk, z_trunk, torch_dtype


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(dtype="float32", z_pair_dtype="float32"),
        Scenario(dtype="bfloat16"),
        Scenario(dtype="float32", token_dtype="bfloat16", enc_dtype="bfloat16", dec_dtype="bfloat16"),
    ],
    ids=["fp32", "bf16", "fp32_bf16_score"],
)
def test_protenix_diffusion_module(sc: Scenario, real_case):
    model, batch, s_inputs, s_trunk, z_trunk, torch_dtype = _real_bioir_module(sc, real_case)
    x_noisy = real_case["x_noisy"].unsqueeze(0).to(torch_dtype)
    t = real_case["t"].unsqueeze(0).to(torch_dtype)

    token_shapes = []

    def _capture_token_inputs(_module, args):
        token_shapes.append(tuple(tuple(x.shape) for x in args[:4]))

    handle = model.diffusion_transformer.register_forward_pre_hook(_capture_token_inputs)
    with torch.inference_mode():
        act = model(x_noisy, t, batch, s_inputs, s_trunk, z_trunk).squeeze(0)
    handle.remove()

    # Token inputs fold samples; pair inputs remain sample-independent.
    assert token_shapes == [
        (
            (Scenario().n_sample, Scenario().n_token, _FULL["c_token"]),
            (Scenario().n_sample, Scenario().n_token, _FULL["c_s"]),
            (1, Scenario().n_token, Scenario().n_token, _FULL["c_z"]),
            (1, Scenario().n_token),
        )
    ]

    assert torch.isfinite(act).all(), f"non-finite output ({sc.dtype})"
    r = _rmse_ratio(act, real_case["exp"])
    any_bf16 = "bfloat16" in (sc.dtype, sc.token_dtype, sc.enc_dtype, sc.dec_dtype, sc.z_pair_dtype)
    tol = 1.5e-1 if any_bf16 else 5e-3
    assert r < tol, f"x_denoised rmse_ratio={r:.3e} exceeds {tol:.0e} ({sc.dtype})"

    # Cached step-invariant work must reproduce the uncached step.
    with torch.inference_mode():
        cache = model.prepare_cache(batch, s_inputs, s_trunk, z_trunk)
        assert cache["pair_z"].dtype == str_dtype_to_torch(sc.z_pair_dtype)
        act_cached = model(x_noisy, t, batch, s_inputs, s_trunk, z_trunk, cache=cache).squeeze(0)
    assert torch.isfinite(act_cached).all(), f"non-finite cached ({sc.dtype})"
    rc = _rmse_ratio(act_cached, act)
    tol_c = 2e-2 if any_bf16 else 1e-4
    assert rc < tol_c, f"shared-vars cache diverged from uncached: rmse={rc:.3e} exceeds {tol_c:.0e} ({sc.dtype})"


def _sampler_inputs(device: torch.device, sc: Scenario, module):
    """Random reference features + trunk embeddings for the sampler rollout."""
    B, n_atom, n_token = 1, sc.n_atom, sc.n_token
    a2t = (torch.arange(n_atom, device=device) // (n_atom // n_token)).long()
    feats = {
        "atom_to_token_idx": a2t,
        "ref_pos": torch.randn(n_atom, 3, device=device),
        "ref_charge": torch.zeros(n_atom, device=device),
        "ref_mask": torch.ones(n_atom, device=device),
        "ref_element": F.one_hot(torch.randint(0, 128, (n_atom,), device=device), 128).float(),
        "ref_atom_name_chars": F.one_hot(torch.randint(0, 64, (n_atom, 4), device=device), 64).float(),
        "ref_space_uid": a2t.clone(),
    }
    feats = update_input_feature_dict(feats)
    batch = {
        k: feats[k].unsqueeze(0)
        for k in (
            "atom_to_token_idx",
            "ref_pos",
            "ref_charge",
            "ref_mask",
            "ref_atom_name_chars",
            "ref_element",
            "d_lm",
            "v_lm",
        )
    }
    batch["pad_info"] = feats["pad_info"]
    tok = torch.arange(n_token, device=device).unsqueeze(0)
    zero = torch.zeros(1, n_token, dtype=torch.long, device=device)
    batch["relp"] = module.diffusion_conditioning.relpe.generate_relp(
        asym_id=zero, residue_index=tok, entity_id=zero, token_index=tok, sym_id=zero
    )
    s_inputs = torch.randn(B, n_token, sc.c_s_inputs, device=device)
    s_trunk = torch.randn(B, n_token, sc.c_s, device=device)
    z_trunk = torch.randn(B, n_token, n_token, sc.c_z, device=device)
    return batch, s_inputs, s_trunk, z_trunk


def test_sample_diffusion_smoke():
    """Check EDM rollout shape and finiteness."""
    # Keep the random-weight smoke in full fp32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Seed CPU + CUDA: sample()/randn are on GPU. Default Linear init is too
    # large for multi-step EDM (/sigma_hat) under xdist's varying CUDA RNG;
    # use the same small-init recipe as
    # test_token_transformer_broadcasts_sample_independent_pair. Real-checkpoint
    # tests cover numerical fidelity.
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda")
    sc = Scenario(dtype="float32", z_pair_dtype="float32")
    B, S, n_atom = 1, sc.n_sample, sc.n_atom

    module = ProtenixDiffusionModule(_build_config(sc)).to(device).eval()
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith("weight") and "norm" in name:
                param.fill_(1)
            elif param.ndim > 1:
                param.normal_(mean=0.0, std=0.02)
            else:
                param.zero_()
    sampler = ProtenixSampleDiffusion(module).to(device).eval()
    batch, s_inputs, s_trunk, z_trunk = _sampler_inputs(device, sc, module)
    s_inputs = s_inputs * 0.1
    s_trunk = s_trunk * 0.1
    z_trunk = z_trunk * 0.1

    with torch.inference_mode():
        x = sampler.sample_coords(batch, s_inputs, s_trunk, z_trunk, num_sampling_steps=4, N_sample=S)

    assert x.shape == (B, S, n_atom, 3)
    assert torch.isfinite(x).all()


@pytest.mark.parametrize(
    "sc",
    [
        Scenario(dtype="float32", z_pair_dtype="float32"),
        Scenario(dtype="float32", token_dtype="bfloat16", enc_dtype="bfloat16", dec_dtype="bfloat16"),
    ],
    ids=["fp32", "fp32_bf16_score"],
)
def test_sample_diffusion_shared_vars_cache(sc: Scenario, real_case):
    """Match cached and uncached EDM rollouts under identical noise."""
    torch.backends.cuda.matmul.allow_tf32 = False
    model, batch, s_inputs, s_trunk, z_trunk, _ = _real_bioir_module(sc, real_case)
    n_sample = Scenario().n_sample
    uncached = ProtenixSampleDiffusion(model, use_cache=False).eval()
    cached = ProtenixSampleDiffusion(model, use_cache=True).eval()
    assert not uncached.use_cache and cached.use_cache

    def _roll(sampler):
        torch.manual_seed(1234)
        with torch.inference_mode():
            return sampler.sample_coords(batch, s_inputs, s_trunk, z_trunk, num_sampling_steps=8, N_sample=n_sample)

    x_ref = _roll(uncached)
    x_cached = _roll(cached)
    assert x_cached.shape == (1, n_sample, real_case["x_noisy"].shape[-2], 3)
    assert torch.isfinite(x_cached).all()
    any_bf16 = "bfloat16" in (sc.dtype, sc.token_dtype, sc.enc_dtype, sc.dec_dtype, sc.z_pair_dtype)
    r = _rmse_ratio(x_cached, x_ref)
    tol = 5e-2 if any_bf16 else 1e-3
    assert r < tol, (
        f"cache changed the rollout ({sc.dtype}): rmse={r:.3e} max|Δ|={(x_cached - x_ref).abs().max().item():.3e}"
    )


def test_sample_coords_drop_consumed_features(real_case):
    """Drop cache-only inputs without changing the rollout."""
    torch.backends.cuda.matmul.allow_tf32 = False
    sc = Scenario(dtype="float32", z_pair_dtype="float32")
    model, batch, s_inputs, s_trunk, z_trunk, _ = _real_bioir_module(sc, real_case)
    n_sample = Scenario().n_sample
    sampler = ProtenixSampleDiffusion(model, use_cache=True).eval()
    consumed = {
        "relp",
        "ref_pos",
        "ref_charge",
        "ref_mask",
        "ref_atom_name_chars",
        "ref_element",
        "d_lm",
        "v_lm",
        "pad_info",
    }

    def _roll(drop):
        torch.manual_seed(1234)
        with torch.inference_mode():
            return sampler.sample_coords(
                batch, s_inputs, s_trunk, z_trunk, num_sampling_steps=8, N_sample=n_sample, drop_consumed_features=drop
            )

    # Keep inputs for the following destructive run.
    assert consumed <= batch.keys()
    x_keep = _roll(False)
    assert consumed <= batch.keys()
    # Destructive mode pops cache-only inputs before denoising.
    x_drop = _roll(True)
    assert consumed.isdisjoint(batch)
    r = _rmse_ratio(x_drop, x_keep)
    assert r < 1e-4, (
        f"dropping relp changed the cached rollout: rmse={r:.3e} max|Δ|={(x_drop - x_keep).abs().max().item():.3e}"
    )


def test_protenix_module_registry_wiring():
    """Check token-transformer CUDA-graph discovery wiring.

    Discovery is generic rather than per-model: modules opt in via
    ``@support_graph_optimization``, so the ``token_transformer`` role alias
    resolves to the decorated ``diffusion_transformer`` by qualified path,
    and the getter/setter use ``get_submodule`` / ``set_submodule``.
    """
    import torch.nn as nn

    from bionemo_ir.models.protenix import Protenix

    model = Protenix(config=Protenix.get_pretrained_config("protenix-v2"), include_load_weights=False)
    reg = model.get_optimized_modules({})
    spec = reg.get_accelerated_modules()["token_transformer"]
    assert spec.graph_optimization_cls is CUDAGraphOptimizationTracker

    path = "diffusion_sampler.diffusion_module.diffusion_transformer"
    assert reg._name_to_path["token_transformer"] == path
    assert spec.getter(model) is model.get_submodule(path)

    replacement = nn.Identity()
    spec.setter(model, replacement)
    assert model.get_submodule(path) is replacement


def test_enabled_modules_act_as_cudagraph_whitelist():
    """``GRAPH_OPT_ENABLED_MODULES`` is a whitelist: a module may be cuda-graphed
    only if its path is a value in the alias map. The recycling-trunk and
    confidence-head pairformers are decorated (``@support_graph_optimization``)
    and so are discoverable by qualified path, but their CUDA-graph replay
    produces NaN and they are deliberately not aliased — configuring them must
    be refused rather than silently graphed."""
    from bionemo_ir.configs import AcceleratedConfig, BackendType
    from bionemo_ir.models.protenix import Protenix

    model = Protenix(config=Protenix.get_pretrained_config("protenix-v2"), include_load_weights=False)

    def cfg():
        return AcceleratedConfig(backend=BackendType.TORCH)

    # A decorated-but-non-whitelisted path is rejected...
    for bad in ("trunk.pairformer_stack", "confidence_head.pairformer_stack"):
        with pytest.raises(ValueError, match="non-whitelisted"):
            model.get_optimized_modules({bad: cfg()})

    # ...while both the alias name and its resolved value-path are accepted.
    resolved = "diffusion_sampler.diffusion_module.diffusion_transformer"
    for good in ("token_transformer", resolved):
        reg = model.get_optimized_modules({good: cfg()})
        assert reg.get_module_names() == [good]


def test_token_transformer_cudagraph_parity(real_case):
    """Check CUDA-graph capture, replay, and rollout parity."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sc = Scenario(dtype="float32", token_dtype="bfloat16", enc_dtype="bfloat16", dec_dtype="bfloat16")
    model, batch, s_inputs, s_trunk, z_trunk, _ = _real_bioir_module(sc, real_case)
    n_sample = Scenario().n_sample

    def _roll(m, steps=12):
        sampler = ProtenixSampleDiffusion(m, use_cache=True).eval()
        torch.manual_seed(1234)
        with torch.inference_mode():
            return sampler.sample_coords(batch, s_inputs, s_trunk, z_trunk, num_sampling_steps=steps, N_sample=n_sample)

    x_eager = _roll(model)

    eager_tt = model.diffusion_transformer
    tracker = CUDAGraphOptimizationTracker(
        config=CUDAGraphOptimizationConfig(
            graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
            verify_capture=True,
            num_graphs_max_for_this_module=2,
        ),
        inner_module=eager_tt,
    )
    tracker.set_fallback_module(eager_tt)
    model.diffusion_transformer = tracker

    x_cg = _roll(model)

    # The fixed per-step shape must capture without eager fallback.
    states = [
        (s.preparation_state, tracker.fallback_to_eager_by_key.get(k, False))
        for k, s in tracker.graph_state_by_key.items()
    ]
    assert states and all(ps == CUDAGraphPreparationState.GRAPH_VERIFIED and not fb for ps, fb in states), (
        f"expected every captured key GRAPH_VERIFIED with no eager fallback, got {[(ps.name, fb) for ps, fb in states]}"
    )
    assert torch.isfinite(x_cg).all()
    # Tiny replay deltas amplify over the chaotic bf16 rollout; guard only
    # against gross divergence because verify_capture checks exact replay.
    r = _rmse_ratio(x_cg, x_eager)
    assert r < 1e-1, (
        f"cuda-graph token transformer diverged from eager: "
        f"rmse={r:.3e} "
        f"max|Δ|={(x_cg - x_eager).abs().max().item():.3e}"
    )
