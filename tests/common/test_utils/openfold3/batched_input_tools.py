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
"""Assemble a real batched (B>1) input for the OpenFold3 ``DiffusionModule``.

The folding pipeline runs one structure per forward (``engine_stage`` asserts
``len(rows) == 1``), so a B>1 batch has to be assembled at the module-input
level. :func:`make_batched_diffusion_inputs` runs the eager pipeline over the
requested samples, captures each one's real ``DiffusionModule.forward`` kwargs,
zero-pads them to a common token/atom count, and stacks them into a single
batch-size-``len(sample_ids)`` input — with ``attn_metadata`` and the precomputed
``atom_broadcast_index`` rebuilt for the padded shape.
"""

import tempfile
from functools import partial
from pathlib import Path

import pytest
import torch

from bionemo_ir._torch.attention_backend.interface import AttentionMetadata
from bionemo_ir._torch.layers.sequence_local_atom import create_gather_indices, query_to_keys_optimized
from bionemo_ir._torch.modules.openfold3.diffusion_module import DiffusionModule
from bionemo_ir._torch.modules.openfold3.utils.atomize_utils import compute_atom_broadcast_index
from bionemo_ir.pipeline.processor.engine_proc import EngineProcessorConfig
from bionemo_ir.pipeline.stages.configs import WriterStageConfig
from tests.common.test_utils.model_forwards import (
    _AVAILABILITY_EXC,
    DIFFUSION_SAMPLES,
    NUM_SAMPLING_STEPS,
    RECYCLING_STEPS,
    _default_model_config,
    _find_sample_json,
    _load_request,
    _run_pipeline,
)

# The pipeline availability exception, re-exported so callers can gate on it.
AVAILABILITY_EXC = _AVAILABILITY_EXC


def harness_skip_reason(sample_ids=()) -> str | None:
    """Return a pytest-skip reason when a required sample's input json is
    missing; ``None`` when all are present."""
    for sid in sample_ids:
        if _find_sample_json(sid) is None:
            return f"sample {sid} not available"
    return None


# OpenFold3 sequence-local atom-attention window sizes (config defaults).
_N_QUERY = 32
_N_KEY = 128


def clone_tree(x):
    """Deep-clone tensors (detaching inference-mode) so captured inputs are
    normal tensors reusable across calls; pass non-tensors through."""
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, dict):
        return {k: clone_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(clone_tree(v) for v in x)
    return x


def _pad_axis(t: torch.Tensor, axis: int, target: int) -> torch.Tensor:
    """Zero-pad ``t`` along ``axis`` up to length ``target``."""
    if t.shape[axis] >= target:
        return t
    fill = list(t.shape)
    fill[axis] = target - t.shape[axis]
    return torch.cat([t, t.new_zeros(fill)], dim=axis)


def _pad_tree(x, n_atom: int, n_token: int, N_atom: int, N_token: int):
    """Pad every tensor's atom axes (== ``n_atom``) to ``N_atom`` and token axes
    (== ``n_token``) to ``N_token``. Non-tensors pass through.

    Relies on the atom count, token count, and every channel width being
    mutually distinct — true for real OpenFold3 diffusion features (atoms in the
    thousands, tokens in the hundreds, channels <= 449)."""
    if torch.is_tensor(x):
        for axis in range(x.ndim):
            length = x.shape[axis]
            if length == n_atom:
                x = _pad_axis(x, axis, N_atom)
            elif length == n_token:
                x = _pad_axis(x, axis, N_token)
        return x
    if isinstance(x, dict):
        return {k: _pad_tree(v, n_atom, n_token, N_atom, N_token) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_pad_tree(v, n_atom, n_token, N_atom, N_token) for v in x)
    return x


def _rebuild_attn_metadata(n_atom: int, device: torch.device) -> AttentionMetadata:
    """Sequence-local ``query_to_keys`` gather for a padded atom count (mirrors
    ``OpenFold3.generate_attn_metadata``)."""
    K = (n_atom + (_N_QUERY - (n_atom % _N_QUERY))) // _N_QUERY
    gather_indices, _ = create_gather_indices(K, _N_QUERY, _N_KEY, device)
    query_to_keys = partial(query_to_keys_optimized, gather_indices=gather_indices, W=_N_QUERY, H=_N_KEY)
    return AttentionMetadata(query_to_keys=query_to_keys, bias_cache={})


def _build_eager_config(output_dir: Path) -> EngineProcessorConfig:
    """A plain (no-acceleration) serial-backend OpenFold3 processor config.

    The eager run is used only to capture real ``DiffusionModule`` inputs, so it
    passes no ``accelerated_configs`` — the model runs eager end to end.
    ``_default_model_config`` sets each atom transformer's ``shared_pair_norm``
    to match the checkpoint the engine will load.
    """
    engine_kwargs: dict = {"profile_inference": False}
    default_cfg = _default_model_config("openfold3")
    if default_cfg is not None:
        engine_kwargs["config"] = default_cfg
    return EngineProcessorConfig(
        model_source="openfold3",
        executor_backend=None,  # in-process SerialProcessor
        engine_kwargs=engine_kwargs,
        runtime_args={
            "recycling_steps": RECYCLING_STEPS,
            "num_sampling_steps": NUM_SAMPLING_STEPS,
            "diffusion_samples": DIFFUSION_SAMPLES,
        },
        writer_stage=WriterStageConfig(output_path=str(output_dir), format="cif"),
    )


def _capture_per_sample_inputs(sample_ids):
    """Run the eager OF3 pipeline over ``sample_ids``, capturing the first
    ``DiffusionModule.forward`` kwargs per distinct token count.

    Returns ``(module, kwargs_per_sample_ordered_by_token_count)``. Raises the
    weights/metadata availability exception when the checkpoint is missing.
    """
    module_box: dict = {}
    by_token_count: dict[int, dict] = {}
    orig_forward = DiffusionModule.forward

    def capturing_forward(self, **kwargs):
        out = orig_forward(self, **kwargs)
        n_token = kwargs["si_trunk"].shape[-2]
        module_box.setdefault("module", self)
        by_token_count.setdefault(n_token, clone_tree(kwargs))
        return out

    requests = [_load_request(sid) for sid in sample_ids]
    with tempfile.TemporaryDirectory() as out_dir:
        config = _build_eager_config(Path(out_dir))
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(DiffusionModule, "forward", capturing_forward)
            _run_pipeline(config, requests, tuple(sample_ids), Path(out_dir))

    kwargs_list = [by_token_count[nt] for nt in sorted(by_token_count)]
    assert len(kwargs_list) == len(sample_ids), (
        f"expected {len(sample_ids)} distinct-size captures, got {len(kwargs_list)}"
    )
    return module_box["module"], kwargs_list


def _assemble_batched_kwargs(per_sample):
    """Pad per-sample (B=1) captured kwargs to a common shape and stack them into
    one ``B == len(per_sample)`` kwargs dict (see :func:`make_batched_diffusion_inputs`)."""
    atoms = [k["xl_noisy"].shape[-2] for k in per_sample]
    tokens = [k["si_trunk"].shape[-2] for k in per_sample]
    N_atom, N_token = max(atoms), max(tokens)
    device = per_sample[0]["xl_noisy"].device

    # ``atom_broadcast_index`` is size/content-specific and ``attn_metadata`` is
    # a closure; drop both before padding and rebuild them for the padded batch.
    padded = []
    for kwargs, n_atom, n_token in zip(per_sample, atoms, tokens, strict=False):
        kw = dict(kwargs)
        kw.pop("attn_metadata", None)
        batch = dict(kw["batch"])
        batch.pop("atom_broadcast_index", None)
        kw["batch"] = batch
        padded.append(_pad_tree(kw, n_atom, n_token, N_atom, N_token))

    # ``t`` (scalar noise level) and ``use_conditioning`` are size-independent
    # and identical across the first denoise step, so sample 0's carry over.
    batched = dict(padded[0])
    for key in ("xl_noisy", "token_mask", "atom_mask", "si_input", "si_trunk", "zij_trunk"):
        batched[key] = torch.cat([p[key] for p in padded], dim=0)
    # Stack every ``batch`` field whose padded shape matches across samples
    # (the atom/token-indexed features the diffusion module consumes). Fields
    # that stay ragged after padding (e.g. MSA/template tensors with a
    # sample-varying depth axis) are not read by the diffusion module and are
    # dropped rather than force-stacked.
    batched_batch = {}
    for k, v0 in padded[0]["batch"].items():
        if not torch.is_tensor(v0):
            batched_batch[k] = v0
            continue
        vs = [p["batch"][k] for p in padded]
        if all(v.shape == v0.shape for v in vs):
            batched_batch[k] = torch.cat(vs, dim=0)
    batched["batch"] = batched_batch
    batched["batch"]["atom_broadcast_index"] = compute_atom_broadcast_index(
        batched["batch"]["token_mask"], batched["batch"]["num_atoms_per_token"]
    )
    batched["attn_metadata"] = _rebuild_attn_metadata(N_atom, device)
    return batched


def capture_and_assemble(sample_ids=("T1038", "T1047s1")):
    """Capture the samples' real B=1 diffusion inputs once, and return both the
    per-sample inputs and the assembled batch built from the *same* capture.

    Returns ``(module, per_sample_kwargs, batched_kwargs)``: ``per_sample_kwargs``
    is a list of valid B=1 kwargs (one per sample, ordered by token count) and
    ``batched_kwargs`` is the ``B == len(sample_ids)`` stack of those same inputs.
    Sharing one capture keeps the random ``xl_noisy`` identical between the two,
    so a per-sample B=1 run is directly comparable to its slice of the batch
    (running the samples through separate pipeline invocations would draw
    different noise, since RNG advances across samples within one run).
    """
    module, per_sample = _capture_per_sample_inputs(sample_ids)
    return module, per_sample, _assemble_batched_kwargs(per_sample)


def make_batched_diffusion_inputs(sample_ids=("T1038", "T1047s1")):
    """Assemble one batched ``DiffusionModule.forward`` input from ``sample_ids``.

    Captures each sample's real (B=1) OpenFold3 diffusion inputs, zero-pads them
    to the common max token/atom count, and stacks them along a new batch dim
    (size ``len(sample_ids)``). ``attn_metadata`` and ``atom_broadcast_index``
    are rebuilt for the padded shape.

    Returns:
        ``(module, batched_kwargs)`` — the real-weight ``DiffusionModule`` and a
        kwargs dict callable as ``module(**batched_kwargs)`` at
        ``B == len(sample_ids)``.
    """
    module, per_sample = _capture_per_sample_inputs(sample_ids)
    return module, _assemble_batched_kwargs(per_sample)
