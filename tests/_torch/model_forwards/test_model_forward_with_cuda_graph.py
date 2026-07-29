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
"""CUDA-graph parity test for the diffusion (token) transformer.

Verifies that wrapping the diffusion (token) transformer in the CUDA-graph
optimizer produces the *same* predicted structures as the unoptimized
("original") model — for both **OpenFold3** and **boltz-2**.

Both runs go through the public ``build_processor`` API (the same entry point
used by ``docs/release_artifacts/scripts/run_pipeline.py``) on the in-process
**serial** backend, over three bundled sample targets from
``examples/data/samples`` (the three smallest CASP14 monomers). The only
difference between the two runs is the ``accelerated_configs`` passed to the
engine:

  * **original** — no acceleration; the diffusion transformer runs eager.
  * **cuda-graph** — ``token_transformer`` is wrapped in a
    ``CUDAGraphOptimizationTracker`` (warmup -> capture -> verify -> replay,
    with permanent eager fallback on any failure).

The diffusion sampling loop calls the token transformer with a fixed input
shape across all rollout steps, so a single graph is captured per target and
replayed for the remaining steps (capture-once / replay-many). RNG is seeded
identically per request, so a correct graph yields structures that match the
eager run.

The test asserts, per model:
  1. the graph engaged and self-verified for every captured key
     (``GRAPH_VERIFIED`` with no eager fallback), and
  2. the cuda-graph prediction is as accurate as the eager one: per target,
     its CA lDDT to the experimental ground truth is within
     ``LDDT_GT_PARITY_TOL`` of the eager prediction's CA lDDT to ground truth.
"""

import gc
import os
import tempfile
from pathlib import Path

import pytest
import torch

import tests
from tensorrt_bionemo._torch.graph_optimization.config import (
    CUDAGraphOptimizationConfig, GraphOptimizationMode, InputKeyMethod,
    InputRoutingConfigFactory)
from tensorrt_bionemo._torch.graph_optimization.cuda_graph.runtime import (
    CUDAGraphOptimizationTracker, CUDAGraphPreparationState)
from tensorrt_bionemo.configs import AcceleratedConfig, BackendType, BaseConfig
from tensorrt_bionemo.data.schemas import InputRequest
from tensorrt_bionemo.pipeline.processor.engine_proc import (
    EngineProcessorConfig, build_processor)
from tensorrt_bionemo.pipeline.stages.configs import WriterStageConfig
from tensorrt_bionemo.pipeline.stages.engine_stage import \
    FoldingPredictionError
from tests.common.test_utils.basic import path_for_package_in_repo
from tests.common.test_utils.model_forwards import (
    _AVAILABILITY_EXC, _CKPT_ENV, _HF_CKPT, _lddt_to_reference, _load_request,
    _model_weights_available)
from tests.common.test_utils.seeding import seed_everything

# --- Test configuration ----------------------------------------------------
# Models whose diffusion/token transformer is graph-optimizable. Each is run
# through the full pipeline twice (eager vs cuda-graph) and compared..
MODEL_SOURCES = ("openfold3", )
SEED = 42

# Bundled sample data (no Git LFS — real files shipped in the repo).
REPO_ROOT = path_for_package_in_repo(tests).parent
SAMPLES_DIR = REPO_ROOT / "examples" / "data" / "samples"
# Bundled experimental ground-truth structures (``{sample_id}.pdb``).
GT_DIR = SAMPLES_DIR / "gt"
# Four smallest CASP14 monomers (≈95 / 100 / 199 / 232 residues)
#       id      num_tokens   num_token_bucket
#       T1031   95              (1, 128)
#       T1033   100             (1, 128)
#       T1038   199             (129, 256)
#       T1047s1 232             (129, 256)
#SAMPLE_ID_TUPLE_B = ("T1047s1", )
SAMPLE_ID_TUPLE_C = ("T1038", )
SAMPLE_ID_TUPLE_D = ("T1038", "T1047s1")
SAMPLE_ID_TUPLE_E = ("T1047s1", "T1038")

# Every distinct sample id referenced by any tuple above; used only for the
# module-level data-availability check below. Each test receives its own
# ``sample_ids`` tuple via the ``sample_id_tuple`` parametrization.
_ALL_SAMPLE_IDS = tuple(sorted({
    sid
    for tup in (SAMPLE_ID_TUPLE_C,
                SAMPLE_ID_TUPLE_D, SAMPLE_ID_TUPLE_E)
    for sid in tup
}))

# Diffusion runtime args. num_sampling_steps need only exceed the warmup
# threshold (3 calls) so each target's graph captures and then replays.
RECYCLING_STEPS = 3
NUM_SAMPLING_STEPS = 200
DIFFUSION_SAMPLES = 5

# Max allowed gap between the cuda-graph and eager predictions' CA lDDT to
# ground truth. Padding changes the order of operations in GEMMs (float
# non-associativity), so the two need only agree in accuracy, not bit-for-bit.
LDDT_GT_PARITY_TOL = 0.07

def _find_sample_json(sample_id: str) -> Path | None:
    """Return the first ``<sample_id>.json`` found anywhere under SAMPLES_DIR.

    Samples are grouped into per-category subdirectories (``monomers/``,
    ``homopolymers/``, ``rna_dna_ligand/``, ...), so a target's input json is
    located by a recursive search rather than assuming a fixed subdirectory.
    Returns ``None`` when no such file exists.
    """
    return next(SAMPLES_DIR.rglob(f"{sample_id}.json"), None)


_SAMPLES_AVAILABLE = SAMPLES_DIR.is_dir() and all(
    _find_sample_json(sid) is not None for sid in _ALL_SAMPLE_IDS)


_CKPT_ENV = {"openfold3": "OPENFOLD3_CKPT", "boltz-2": "boltz-2_CKPT"}
_HF_CKPT = {
    "openfold3": ("OpenFold/OpenFold3", "checkpoints/of3-p2-155k.pt"),
    "boltz-2": ("boltz-community/boltz-2", "boltz-2_conf.ckpt"),
}

@pytest.fixture(autouse=True)
def _free_captured_cuda_graphs():
    """Drop every CUDA graph captured during the test before the next one runs.

    ``torch.cuda.graph(...)`` capture (in ``CUDAGraphOptimizationTracker``)
    registers the *process-global* default CUDA generator with the captured
    graph, and that registration lives as long as the graph does. The captured
    graphs are held by the pipeline's in-process model, which is only reachable
    through GC cycles (nn.Module parent<->child refs), so plain refcounting does
    not free them deterministically at test end. If a graph outlives the test,
    the default generator stays graph-registered and a later test in the same
    pytest worker that draws eager RNG from it (e.g. a bare
    ``torch.randn(..., device="cuda")`` in ``tests/ops/test_gated_sigmoid.py``)
    can raise "Offset increment outside graph capture encountered unexpectedly".

    Forcing a collection here frees the model and its graphs, which un-registers
    the default generator, isolating this module's graph state from the rest of
    the suite. Cheap relative to a graph-capture test and safe as a no-op when
    CUDA is unavailable.
    """
    yield
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _default_model_config(model_source: str):
    """Pretrained model config with any checkpoint-compatibility overrides.

    Returns ``None`` to use the engine's default pretrained config.

    OpenFold3 ships checkpoints whose atom-transformer pair LayerNorms use
    different layouts, so each config's ``shared_pair_norm`` flag must match
    whichever checkpoint the engine will load (``OPENFOLD3_CKPT`` if set, else
    the HF default ``of3-p2-155k.pt``):

    * **shared** layout: a single ``atom_transformer.layer_norm_z`` per atom
      transformer (e.g. ``of3-p2-155k.pt``). The converter looks for this key
      only when ``shared_pair_norm=True`` (the pretrained-config default).
    * **per-block** layout: a LayerNorm per block
      (``blocks.N.attention_pair_bias.layer_norm_z``), e.g.
      ``v19_78k_ft3_converted.pt`` or ``of3_ft3_v1.pt``. The converter looks
      for these only when ``shared_pair_norm=False``.

    Matching the wrong layout makes the weight converter look for keys the
    checkpoint does not contain (``KeyError`` at load). Rather than guess the
    layout from the checkpoint filename, inspect the checkpoint's keys and set
    each transformer's flag from what is actually present.
    """
    if model_source != "openfold3":
        return None
    from tensorrt_bionemo.registry import get_model_class
    cfg = get_model_class(model_source).get_pretrained_config(model_source)
    # Only a locally-present ``OPENFOLD3_CKPT`` can be inspected cheaply; when
    # it is unset the engine downloads the HF default (shared layout), which
    # already matches the pretrained-config default, so leave the flags alone.
    ckpt = os.environ.get("OPENFOLD3_CKPT")
    if not ckpt or not Path(ckpt).is_file():
        return cfg
    state_dict = torch.load(
        ckpt, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # (checkpoint prefix, config attr holding that transformer's config)
    transformers = [
        ("input_embedder.atom_attn_enc",
         cfg.input_embedder_config.atom_transformer_config),
        ("diffusion_module.atom_attn_enc",
         cfg.diffusion_module_config.atom_transformer_encoder_config),
        ("diffusion_module.atom_attn_dec",
         cfg.diffusion_module_config.atom_transformer_decoder_config),
    ]
    for prefix, tf_cfg in transformers:
        shared_key = f"{prefix}.atom_transformer.layer_norm_z.weight"
        tf_cfg.shared_pair_norm = shared_key in state_dict
    return cfg


# ===========================================================================
# Pipeline construction + execution (serial backend, via build_processor)
# ===========================================================================
def _openfold3_graph_optimization_config(
    module_name: str,
    input_key_method: InputKeyMethod,
    sample_ids: tuple[str, ...],
    verify_capture : bool = True,
    ) -> CUDAGraphOptimizationConfig:
    """Build the CUDA-graph optimization config for an OpenFold3 ``module_name``.

    Sets up an ``InputRoutingConfigFactory`` with the module's input-acceptance
    dims (and, for ``BUCKETED_SHAPES``, its padded/bucketed dims plus output
    ties). ``num_graphs_max_for_this_module`` is sized to ``len(sample_ids)``
    (at most one captured graph per target). Raises for an unknown module or
    unsupported ``input_key_method``.
    """
    if module_name == "token_transformer":
        
        # input acceptance
        input_routing = InputRoutingConfigFactory()
        input_routing.set_input_acceptance_dim(
            dim_name="num_tokens",
            dim_len_max=1024,
        )
        # Inputs (kwargs by name); z carries the token axis on dims -2 and -3.
        input_routing.input_dim_is_acceptance("a", -2, "num_tokens")
        input_routing.input_dim_is_acceptance("s", -2, "num_tokens")
        input_routing.input_dim_is_acceptance("z", -2, "num_tokens")
        input_routing.input_dim_is_acceptance("z", -3, "num_tokens")
        input_routing.input_dim_is_acceptance("mask", -1, "num_tokens")
        
        if input_key_method==InputKeyMethod.EXACT: # prod settting
            return CUDAGraphOptimizationConfig(
                graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
                input_key_method=input_key_method,
                input_routing_config=input_routing.export_config(),
                verify_capture=verify_capture,
                num_graphs_max_for_this_module=len(sample_ids),
            )
        
        elif input_key_method==InputKeyMethod.BUCKETED_SHAPES: # experimental setting
            input_routing.set_padded_dim(
                dim_name="num_tokens",
                dim_len_min=1,      # the model's min_seq_len floor
                dim_len_max=1024,    # comfortably above the largest target (~199)
                num_intervals=8,
                spacing_method="linear",
            )
            # Inputs (kwargs by name); z carries the token axis on dims -2 and -3.
            input_routing.input_dim_is_padded("a", -2, "num_tokens")
            input_routing.input_dim_is_padded("s", -2, "num_tokens")
            input_routing.input_dim_is_padded("z", -2, "num_tokens")
            input_routing.input_dim_is_padded("z", -3, "num_tokens")
            input_routing.input_dim_is_padded("mask", -1, "num_tokens")
            # Output tensor 0 (the updated token representation) token axis.
            input_routing.output_dim_is_padded(0, -2, "num_tokens")
            
            return CUDAGraphOptimizationConfig(
                graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
                input_key_method=input_key_method,
                input_routing_config=input_routing.export_config(),
                verify_capture=verify_capture,
                num_graphs_max_for_this_module=len(sample_ids))
        
        else:
            raise NotImplementedError(f"unsupported input_key_method {input_key_method!r} for module {module_name!r}")

        
    elif module_name=="diffusion_module":

        # input acceptance
        input_routing = InputRoutingConfigFactory()
        input_routing.set_input_acceptance_dim(
            dim_name="num_tokens",
            dim_len_max=1024,
        )
        # DiffusionModule.forward args (kwargs by name): the token axis rides
        # token_mask (-1), si_input/si_trunk (-2), and zij_trunk (-2 and -3).
        # xl_noisy/atom_mask carry the atom axis, so they are left untied.
        input_routing.input_dim_is_acceptance("si_input", -2, "num_tokens")
        input_routing.input_dim_is_acceptance("si_trunk", -2, "num_tokens")
        input_routing.input_dim_is_acceptance("zij_trunk", -2, "num_tokens")
        input_routing.input_dim_is_acceptance("zij_trunk", -3, "num_tokens")
        input_routing.input_dim_is_acceptance("token_mask", -1, "num_tokens")
        
        if input_key_method==InputKeyMethod.EXACT: # prod setting
            return CUDAGraphOptimizationConfig(
                graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
                input_key_method=input_key_method,
                input_routing_config=input_routing.export_config(),
                verify_capture=verify_capture,
                num_graphs_max_for_this_module=len(sample_ids),
            )

        elif input_key_method==InputKeyMethod.BUCKETED_SHAPES: # experimental setting
            input_routing.set_padded_dim(
                dim_name="num_tokens",
                dim_len_min=1,      # the model's min_seq_len floor
                dim_len_max=1024,    # comfortably above the largest target (~199)
                num_intervals=8,
                spacing_method="linear",
            )
            # DiffusionModule.forward args (kwargs by name): the token axis rides
            # token_mask (-1), si_input/si_trunk (-2), and zij_trunk (-2 and -3).
            # xl_noisy/atom_mask carry the atom axis, so they are left untied.
            input_routing.input_dim_is_padded("si_input", -2, "num_tokens")
            input_routing.input_dim_is_padded("si_trunk", -2, "num_tokens")
            input_routing.input_dim_is_padded("zij_trunk", -2, "num_tokens")
            input_routing.input_dim_is_padded("zij_trunk", -3, "num_tokens")
            input_routing.input_dim_is_padded("token_mask", -1, "num_tokens")
            # Output tensor 0 (the updated token representation) token axis.
            input_routing.output_dim_is_padded(0, -2, "num_tokens")

            return CUDAGraphOptimizationConfig(
                graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
                input_key_method=input_key_method,
                input_routing_config=input_routing.export_config(),
                verify_capture=verify_capture,
                num_graphs_max_for_this_module=len(sample_ids))
        else:
            raise NotImplementedError(f"unsupported input_key_method {input_key_method!r} for module {module_name!r}")

    elif module_name=="structure_pairformer":
        
        input_routing = InputRoutingConfigFactory()
        input_routing.set_input_acceptance_dim(
            dim_name="num_tokens",
            dim_len_max=1024,
        )
        # Inputs (kwargs by name); z carries the token axis on dims -2 and -3.
        input_routing.input_dim_is_acceptance("s", -2, "num_tokens")
        input_routing.input_dim_is_acceptance("z", -2, "num_tokens")
        input_routing.input_dim_is_acceptance("z", -3, "num_tokens")
        input_routing.input_dim_is_acceptance("mask", -1, "num_tokens")
        input_routing.input_dim_is_acceptance("pair_mask", -1, "num_tokens")
        input_routing.input_dim_is_acceptance("pair_mask", -2, "num_tokens")
        
        if input_key_method == InputKeyMethod.EXACT: # experimental setting
            return CUDAGraphOptimizationConfig(
                graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
                input_key_method=input_key_method,
                input_routing_config=input_routing.export_config(),
                verify_capture=verify_capture,
                num_graphs_max_for_this_module=len(sample_ids),
            )
            
        elif input_key_method == InputKeyMethod.BUCKETED_SHAPES: # prod setting
            input_routing.set_padded_dim(
                dim_name="num_tokens",
                dim_len_min=1,      # the model's min_seq_len floor
                dim_len_max=1024,    # comfortably above the largest target (~199)
                num_intervals=8,
                multiple_of=128,
                spacing_method="linear",
            )
            # Inputs (kwargs by name); z carries the token axis on dims -2 and -3.
            input_routing.input_dim_is_padded("s", -2, "num_tokens")
            input_routing.input_dim_is_padded("z", -2, "num_tokens")
            input_routing.input_dim_is_padded("z", -3, "num_tokens")
            input_routing.input_dim_is_padded("mask", -1, "num_tokens")
            input_routing.input_dim_is_padded("pair_mask", -1, "num_tokens")
            input_routing.input_dim_is_padded("pair_mask", -2, "num_tokens")
            # Output tensor 0 (the updated token representation) token axis.
            input_routing.output_dim_is_padded(0, -2, "num_tokens")
            input_routing.output_dim_is_padded(1, -2, "num_tokens")
            input_routing.output_dim_is_padded(1, -3, "num_tokens")
            
            return CUDAGraphOptimizationConfig(
                graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
                input_key_method=input_key_method,
                input_routing_config=input_routing.export_config(),
                verify_capture=verify_capture,
                num_graphs_max_for_this_module=len(sample_ids))
        else:
            raise NotImplementedError

    else:
        raise ValueError(f"unsupported input_key_method {input_key_method!r} for module {module_name!r}")


def _boltz2_graph_optimization_config(
    module_name: str,
    input_key_method: InputKeyMethod,
    sample_ids: tuple[str, ...],
    verify_capture: bool = True) -> CUDAGraphOptimizationConfig:
    """Build the CUDA-graph optimization config for a boltz-2 ``module_name``.

    The token axis (``num_tokens``) is tied per module:

    * ``token_transformer`` — called with keyword args ``a``/``s``/``z``/
      ``mask``; ``z`` (the pair rep) carries the token axis on dims -2 and -3.
    * ``structure_pairformer`` — boltz-2 calls ``pairformer_module(s, z,
      mask=..., pair_mask=...)`` with the single/pair reps passed
      *positionally*, so the tracker names them ``arg0``/``arg1`` (not
      ``s``/``z`` as in OpenFold3); ``mask``/``pair_mask`` are keyword and keep
      their names.
    * ``diffusion_module`` — boltz-2's ``DiffusionModule.forward`` carries the
      token axis on ``token_pad_mask`` (-1) and ``s_inputs``/``s_trunk`` (-2);
      its pair representation is nested inside ``diffusion_conditioning_kwargs``
      (so it is not tied here) and its output is atom coordinates (no output
      tie).

    Raises for an unknown module or unsupported ``input_key_method``.
    """
    if module_name == "token_transformer":

        # input acceptance
        input_routing = InputRoutingConfigFactory()
        input_routing.set_input_acceptance_dim(
            dim_name="num_tokens",
            dim_len_max=1024,
        )
        # Inputs (kwargs by name); z carries the token axis on dims -2 and -3.
        input_routing.input_dim_is_acceptance("a", -2, "num_tokens")
        input_routing.input_dim_is_acceptance("s", -2, "num_tokens")
        input_routing.input_dim_is_acceptance("z", -2, "num_tokens")
        input_routing.input_dim_is_acceptance("z", -3, "num_tokens")
        input_routing.input_dim_is_acceptance("mask", -1, "num_tokens")

        if input_key_method == InputKeyMethod.EXACT: # prod setting
            return CUDAGraphOptimizationConfig(
                graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
                input_key_method=input_key_method,
                input_routing_config=input_routing.export_config(),
                verify_capture=verify_capture,
                num_graphs_max_for_this_module=len(sample_ids),
            )

        elif input_key_method == InputKeyMethod.BUCKETED_SHAPES: # experimental setting
            input_routing.set_padded_dim(
                dim_name="num_tokens",
                dim_len_min=1,      # the model's min_seq_len floor
                dim_len_max=1024,    # comfortably above the largest target (~199)
                num_intervals=8,
                spacing_method="linear",
            )
            # Inputs (kwargs by name); z carries the token axis on dims -2 and -3.
            input_routing.input_dim_is_padded("a", -2, "num_tokens")
            input_routing.input_dim_is_padded("s", -2, "num_tokens")
            input_routing.input_dim_is_padded("z", -2, "num_tokens")
            input_routing.input_dim_is_padded("z", -3, "num_tokens")
            input_routing.input_dim_is_padded("mask", -1, "num_tokens")
            # Output tensor 0 (the updated token representation) token axis.
            input_routing.output_dim_is_padded(0, -2, "num_tokens")

            return CUDAGraphOptimizationConfig(
                graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
                input_key_method=input_key_method,
                input_routing_config=input_routing.export_config(),
                verify_capture=verify_capture,
                num_graphs_max_for_this_module=len(sample_ids))
        else:
            raise NotImplementedError(f"unsupported input_key_method {input_key_method!r} for module {module_name!r}")

    elif module_name == "diffusion_module":

        # input acceptance
        input_routing = InputRoutingConfigFactory()
        input_routing.set_input_acceptance_dim(
            dim_name="num_tokens",
            dim_len_max=1024,
        )
        # boltz-2 DiffusionModule.forward args (kwargs by name): the token axis rides
        # token_pad_mask (-1) and s_inputs/s_trunk (-2). The pair rep is nested inside
        # diffusion_conditioning_kwargs, and r_noisy/atom_pad_mask carry the atom
        # axis, so they are left untied.
        input_routing.input_dim_is_acceptance("s_inputs", -2, "num_tokens")
        input_routing.input_dim_is_acceptance("s_trunk", -2, "num_tokens")
        input_routing.input_dim_is_acceptance("token_pad_mask", -1, "num_tokens")

        if input_key_method == InputKeyMethod.EXACT: # prod setting
            return CUDAGraphOptimizationConfig(
                graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
                input_key_method=input_key_method,
                input_routing_config=input_routing.export_config(),
                verify_capture=verify_capture,
                num_graphs_max_for_this_module=len(sample_ids),
            )

        elif input_key_method == InputKeyMethod.BUCKETED_SHAPES: # experimental setting
            input_routing.set_padded_dim(
                dim_name="num_tokens",
                dim_len_min=1,      # the model's min_seq_len floor
                dim_len_max=1024,    # comfortably above the largest target (~199)
                num_intervals=8,
                spacing_method="linear",
            )
            input_routing.input_dim_is_padded("s_inputs", -2, "num_tokens")
            input_routing.input_dim_is_padded("s_trunk", -2, "num_tokens")
            input_routing.input_dim_is_padded("token_pad_mask", -1, "num_tokens")

            return CUDAGraphOptimizationConfig(
                graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
                input_key_method=input_key_method,
                input_routing_config=input_routing.export_config(),
                verify_capture=verify_capture,
                num_graphs_max_for_this_module=len(sample_ids))
        else:
            raise NotImplementedError(f"unsupported input_key_method {input_key_method!r} for module {module_name!r}")
        
    elif module_name == "structure_pairformer":  # not activated in production

        input_routing = InputRoutingConfigFactory()
        input_routing.set_input_acceptance_dim(
            dim_name="num_tokens",
            dim_len_max=1024,
        )
        # boltz-2 passes s, z positionally -> arg0, arg1; mask/pair_mask by
        # keyword. arg1 (the pair rep) carries the token axis on dims -2 and -3.
        input_routing.input_dim_is_acceptance("arg0", -2, "num_tokens")
        input_routing.input_dim_is_acceptance("arg1", -2, "num_tokens")
        input_routing.input_dim_is_acceptance("arg1", -3, "num_tokens")
        input_routing.input_dim_is_acceptance("mask", -1, "num_tokens")
        input_routing.input_dim_is_acceptance("pair_mask", -1, "num_tokens")
        input_routing.input_dim_is_acceptance("pair_mask", -2, "num_tokens")

        if input_key_method == InputKeyMethod.EXACT: # experimental setting
            return CUDAGraphOptimizationConfig(
                graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
                input_key_method=input_key_method,
                input_routing_config=input_routing.export_config(),
                verify_capture=verify_capture,
                num_graphs_max_for_this_module=len(sample_ids),
            )

        elif input_key_method == InputKeyMethod.BUCKETED_SHAPES: # prod setting
            input_routing.set_padded_dim(
                dim_name="num_tokens",
                dim_len_min=1,      # the model's min_seq_len floor
                dim_len_max=1024,    # comfortably above the largest target (~199)
                num_intervals=8,
                spacing_method="linear",
            )
            input_routing.input_dim_is_padded("arg0", -2, "num_tokens")
            input_routing.input_dim_is_padded("arg1", -2, "num_tokens")
            input_routing.input_dim_is_padded("arg1", -3, "num_tokens")
            input_routing.input_dim_is_padded("mask", -1, "num_tokens")
            input_routing.input_dim_is_padded("pair_mask", -1, "num_tokens")
            input_routing.input_dim_is_padded("pair_mask", -2, "num_tokens")
            # Outputs (s, z) are returned as a tuple -> positional, same as OpenFold3.
            input_routing.output_dim_is_padded(0, -2, "num_tokens")
            input_routing.output_dim_is_padded(1, -2, "num_tokens")
            input_routing.output_dim_is_padded(1, -3, "num_tokens")

            return CUDAGraphOptimizationConfig(
                graph_optimization_mode=GraphOptimizationMode.CUDA_GRAPH_VIA_TORCH,
                input_key_method=input_key_method,
                input_routing_config=input_routing.export_config(),
                verify_capture=verify_capture,
                num_graphs_max_for_this_module=len(sample_ids))
        else:
            raise NotImplementedError(f"unsupported input_key_method {input_key_method!r} for module {module_name!r}")
    else:
        raise ValueError(f"unsupported module {module_name!r} for boltz-2")


def _build_graph_optimization_config(
    model_source: str,
    module_name: str,
    input_key_method: InputKeyMethod,
    sample_ids: tuple[str, ...],
    use_cuda_graph: bool) -> CUDAGraphOptimizationConfig | None:
    """Build the CUDA-graph optimization config for ``module_name``.

    Returns ``None`` when ``use_cuda_graph`` is False (eager, no acceleration).
    Otherwise sets up an ``InputRoutingConfigFactory`` with the module's
    input-acceptance dims (and, for ``BUCKETED_SHAPES``, its padded/bucketed
    dims plus output ties), then returns the ``CUDAGraphOptimizationConfig``
    (with ``num_graphs_max_for_this_module`` sized to ``len(sample_ids)``).
    Raises for an unknown module or unsupported ``input_key_method``.
    """
    if not use_cuda_graph:
        return None
    if model_source == "openfold3":
        return _openfold3_graph_optimization_config(module_name, input_key_method, sample_ids)
    elif model_source == "boltz-2":
        return _boltz2_graph_optimization_config(module_name, input_key_method, sample_ids)
    else:
        raise ValueError(f"unsupported model_source {model_source!r}")


def _build_processor_config(model_source: str,
                            module_name: str,
                            output_dir: Path,
                            use_cuda_graph: bool,
                            sample_ids: tuple[str, ...],
                            input_key_method: InputKeyMethod = InputKeyMethod.EXACT
                            ) -> EngineProcessorConfig:
    """Build a serial-backend ``EngineProcessorConfig`` for ``model_source``.

    When ``use_cuda_graph`` is set, the engine is given an ``accelerated_configs``
    entry that swaps the eager ``token_transformer`` for a
    ``CUDAGraphOptimizationTracker`` (TORCH backend + CUDA-graph framework). The
    engine applies this via ``model.optimize(...)`` at construction time. The
    ``module_name`` module key is shared by OpenFold3 and boltz-2.

    ``input_key_method`` selects how each call is reduced to a graph-cache key:
    ``EXACT`` captures one graph per distinct input-shape signature, while
    ``BUCKETED_SHAPES`` pads inputs into shape buckets so targets in the
    same bucket share one captured graph.
    """
    
    engine_kwargs: dict = {"profile_inference": True}
    default_model_cfg = _default_model_config(model_source)
    if default_model_cfg is not None:
        engine_kwargs["config"] = default_model_cfg
    
    # overwrite default graph_optimization_config with a CUDAGraphOptimizationConfig if use_cuda_graph is True
    graph_optimization_config = _build_graph_optimization_config(
        model_source=model_source,
        module_name=module_name,
        input_key_method=input_key_method,
        sample_ids=sample_ids,
        use_cuda_graph=use_cuda_graph)

    engine_kwargs["accelerated_configs"] = {
        module_name:
        AcceleratedConfig(
            backend=BackendType.TORCH,
            default=BaseConfig(
                graph_optimization_config=graph_optimization_config),
        ),
    }
    # insert engine_kwargs, which include the shape configuration
    engine_processor_config = EngineProcessorConfig(
        model_source=model_source,
        executor_backend=None,  # None -> in-process SerialProcessor
        engine_kwargs=engine_kwargs,
        runtime_args={
            "recycling_steps": RECYCLING_STEPS,
            "num_sampling_steps": NUM_SAMPLING_STEPS,
            "diffusion_samples": DIFFUSION_SAMPLES,
        },
        writer_stage=WriterStageConfig(output_path=str(output_dir),
                                       format="cif"))
    return engine_processor_config


def _run_pipeline(model_source: str,
                  module_name: str,
                  requests: list[InputRequest],
                  output_dir: Path,
                  use_cuda_graph: bool,
                  input_key_method: InputKeyMethod,
                  sample_ids: tuple[str, ...],
                  ) -> tuple[dict[str, Path], object]:
    """Run the serial pipeline over ``requests`` and return written CIF paths.

    Returns ``(paths_by_id, processor)``. The processor is returned so the
    caller can introspect the in-process model (serial backend) to confirm the
    graph engaged. ``should_continue_on_error`` defaults to False, so any
    per-request failure raises rather than silently producing an empty output.
    """
    config = _build_processor_config(
        model_source=model_source,
        module_name=module_name,
        output_dir=output_dir,
        use_cuda_graph=use_cuda_graph,
        sample_ids=sample_ids,
        input_key_method=input_key_method)
    processor = build_processor(config)

    records = [{
        "record": req,
        "__record_id": req["input_id"],
        "random_seed": SEED,
    } for req in requests]

    # No outer inference_mode: the folding engine's execute() already applies
    # @torch.inference_mode() for the model forward (this is what disables grad
    # so the tracker captures/replays), while the upstream CPU stages run in
    # normal mode — matching run_pipeline.py's serial path.
    seed_everything(SEED)
    try:
        processor(records)
    except FoldingPredictionError as exc:
        # The engine wraps the real per-request error in a batch-level
        # FoldingPredictionError; surface the underlying model/postprocess
        # exception (with its own traceback) so failures are diagnosable, and
        # so a weights/metadata availability error still reaches the skip
        # handler in the test body.
        raise (exc.__cause__ or exc) from None

    paths: dict[str, Path] = {}
    for sid in sample_ids:
        cif_path = output_dir / f"{sid}.cif"
        assert cif_path.exists(), (
            f"pipeline did not write expected output {cif_path} "
            f"(model={model_source}, use_cuda_graph={use_cuda_graph})")
        paths[sid] = cif_path
    return paths, processor


def _graph_states(processor) -> list[tuple]:
    """Return the per-key ``(preparation_state, fallback_to_eager)`` tuples.

    Reaches into the serial processor's in-process folding engine and searches
    the model for the ``CUDAGraphOptimizationTracker`` that replaced the token
    transformer. Searching ``model.modules()`` keeps this model-agnostic — the
    tracker lives at a different path in OpenFold3 vs boltz-2.
    """
    tracker = None
    for udf in processor._udf_instances.values():
        folding = getattr(udf, "folding", None)
        if folding is None:
            continue
        for module in folding.engine.model.modules():
            if isinstance(module, CUDAGraphOptimizationTracker):
                tracker = module
                break
        break
    assert isinstance(tracker, CUDAGraphOptimizationTracker), (
        "expected the token transformer to be wrapped in a "
        f"CUDAGraphOptimizationTracker, found {type(tracker).__name__}")
    return [(s.preparation_state,
             tracker.fallback_to_eager_by_key.get(k, False))
            for k, s in tracker.graph_state_by_key.items()]


def _assert_graph_verified_and_no_eager_fallback(states: list[tuple]) -> None:
    """Assert every captured key ended ``GRAPH_VERIFIED`` with no eager fallback.

    Args:
        states: per-key ``(preparation_state, fallback_to_eager)`` tuples as
            returned by :func:`_graph_states`.
    """
    assert states and all(
        ps == CUDAGraphPreparationState.GRAPH_VERIFIED and not fb
        for ps, fb in states), (
            "expected every captured key to end GRAPH_VERIFIED with no "
            f"eager fallback, got {[(ps.name, fb) for ps, fb in states]}")


# ===========================================================================
# Structure comparison
# ===========================================================================
def _assert_cudagraph_lddt_to_gt_matches_eager(
    model_source: str,
    original_paths: dict,
    cudagraph_paths: dict,
    input_key_method: InputKeyMethod,
    sample_ids: tuple[str, ...]) -> None:
    """Assert the cuda-graph prediction is as accurate as the eager one.

    For each target, computes the CA lDDT of the eager prediction and of the
    cuda-graph prediction, each against the bundled experimental ground-truth
    structure, and asserts the two lDDTs agree within ``LDDT_GT_PARITY_TOL``. A
    correct graph shares weights, inputs, and RNG, so it should be neither
    better nor worse against ground truth than eager — up to float
    non-associativity from padded GEMMs.

    Args:
        model_source: model key, used for diagnostic messages.
        original_paths: ``{sample_id: cif_path}`` from the eager run.
        cudagraph_paths: ``{sample_id: cif_path}`` from the cuda-graph run.
        input_key_method: keying method, labels the diagnostic output.
        sample_ids: the targets to compare (this test's ``sample_id_tuple``).
    """
    for sid in sample_ids:
        gt = GT_DIR / f"{sid}.pdb"
        lddt_eager = _lddt_to_reference(original_paths[sid], gt)
        lddt_cudagraph = _lddt_to_reference(cudagraph_paths[sid], gt)
        delta = (lddt_cudagraph - lddt_eager)
        print(f"[parity] {model_source} {input_key_method.name} {sid}: "
              f"cudagraph-vs-GT CA lDDT={lddt_cudagraph:.4f} Δ={delta:.4f}, "
              f"eager-vs-GT CA lDDT={lddt_eager:.4f} ")
              
        assert lddt_cudagraph >= lddt_eager - LDDT_GT_PARITY_TOL, (
            f"{model_source} {sid}: cudagraph-vs-GT lDDT-CA {lddt_cudagraph:.4f}"
            f" from eager-vs-GT {lddt_eager:.4f} by {delta:.4f}"
            f" is worse than tol={LDDT_GT_PARITY_TOL} — the graph changed the prediction's accuracy")


# ===========================================================================
# Test
# ===========================================================================
# Per-model parametrization shared by every module's parity test: each model is
# skipped individually when its checkpoint can't be obtained.
_MODEL_PARAMS = [
    pytest.param(
        m,
        marks=pytest.mark.skipif(
            not _model_weights_available(m, _ckpt_env=_CKPT_ENV, _hf_ckpt=_HF_CKPT),
            reason=f"{m} checkpoint unavailable (set {_CKPT_ENV[m]} or HF "
            f"auth for {_HF_CKPT[m][0]})"),
    ) for m in MODEL_SOURCES
]

# The token transformer is exercised with both graph-cache keying strategies:
# EXACT (one graph per distinct shape) and BUCKETED_SHAPES (shape
# bucketing that lets targets share a captured graph).
_INPUT_KEY_METHOD_PARAMS = [
    pytest.param(InputKeyMethod.EXACT, id="exact"),
    pytest.param(InputKeyMethod.BUCKETED_SHAPES, id="bucketed_shapes"),
]

# Subset of the above restricted to the parameter sets that include
# ``InputKeyMethod.EXACT`` — for modules with no BUCKETED_SHAPES config path.
_INPUT_KEY_METHOD_PARAMS_EXACT_ONLY = [
    p for p in _INPUT_KEY_METHOD_PARAMS if InputKeyMethod.EXACT in p.values
]

# ``sample_id_tuple`` parametrizations. Each tuple is one test invocation whose
# eager and cuda-graph runs fold exactly those targets; the test id joins the
# member sample ids with '-'.
_SAMPLE_ID_TUPLE_PARAMS_ALL = [
    pytest.param(t, id="-".join(t)) for t in (
        SAMPLE_ID_TUPLE_C, SAMPLE_ID_TUPLE_D, SAMPLE_ID_TUPLE_E)
]

_SAMPLE_ID_TUPLE_PARAMS_D = [
    pytest.param(t, id="-".join(t)) for t in (SAMPLE_ID_TUPLE_D,)
]

# Multi-target tuples only, for the modules exercised with EXACT keying that
# run over the multi-sample tuples (D, E).
_SAMPLE_ID_TUPLE_PARAMS_MULTI = [
    pytest.param(t, id="-".join(t)) for t in (
        SAMPLE_ID_TUPLE_D, SAMPLE_ID_TUPLE_E)
]

# Shared skip/parametrize stack for every per-module parity test: CUDA + sample
# data required, run each (model_source, input_key_method) combination. Listed
# top-to-bottom exactly as the decorators would stack.
_CUDA_GRAPH_PARITY_MARKS = (
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="pipeline CUDA-graph parity test requires CUDA"),
    pytest.mark.skipif(not _SAMPLES_AVAILABLE,
                       reason=f"sample data not found under {SAMPLES_DIR}"),
    pytest.mark.parametrize("input_key_method", _INPUT_KEY_METHOD_PARAMS),
    pytest.mark.parametrize("model_source", _MODEL_PARAMS),
)

def _cuda_graph_parity_marks(func):
    """Apply the shared ``_CUDA_GRAPH_PARITY_MARKS`` stack to a parity test.

    Applied innermost-first (bottom of the tuple up) so the result is identical
    to writing the marks as stacked decorators — same skips, same test IDs.
    """
    for mark in reversed(_CUDA_GRAPH_PARITY_MARKS):
        func = mark(func)
    return func


def _cuda_graph_parity_marks_exact_only(func):
    """Like ``_cuda_graph_parity_marks`` but with the ``input_key_method``
    parametrization restricted to the parameter sets that include
    ``InputKeyMethod.EXACT``: for modules exercised with EXACT keying only
    (e.g. ``diffusion_module`` / ``sample_diffusion``, which have no
    BUCKETED_SHAPES config path). The test still takes ``input_key_method``,
    but it only ever resolves to ``InputKeyMethod.EXACT``."""
    exact_only = pytest.mark.parametrize(
        "input_key_method", _INPUT_KEY_METHOD_PARAMS_EXACT_ONLY)
    # Swap the full input_key_method parametrize for the EXACT-only subset,
    # leaving the other marks (skips, model_source) untouched.
    marks = tuple(
        exact_only if m is _CUDA_GRAPH_PARITY_MARKS[2] else m
        for m in _CUDA_GRAPH_PARITY_MARKS)
    for mark in reversed(marks):
        func = mark(func)
    return func


def _assert_cuda_graph_parity(model_source: str, module_name: str,
                             sample_ids: tuple[str, ...],
                             input_key_method: InputKeyMethod = InputKeyMethod.EXACT
                             ) -> None:
    """Run ``model_source`` eager vs cuda-graph on ``module_name`` and assert
    the graph captured/verified for every target and left the prediction
    unchanged. Shared body of the per-module parity tests below.

    ``sample_ids`` is the target tuple to fold (this test's ``sample_id_tuple``);
    ``input_key_method`` selects the graph-cache keying used for the cuda-graph
    run (see :func:`_build_processor_config`)."""
    requests = [_load_request(sid) for sid in sample_ids]

    with tempfile.TemporaryDirectory() as original_dir, \
            tempfile.TemporaryDirectory() as cudagraph_dir:
        original_dir = Path(original_dir)
        cudagraph_dir = Path(cudagraph_dir)

        try:
            # --- Run 1: model with the CUDA-graph-wrapped module -----------
            cudagraph_paths, cudagraph_proc = _run_pipeline(
                model_source=model_source,
                requests=requests,
                output_dir=cudagraph_dir,
                use_cuda_graph=True,
                module_name=module_name,
                input_key_method=input_key_method,
                sample_ids=sample_ids)
            torch.cuda.empty_cache()
            
            # --- Run 2: original (eager) model -----------------------------
            original_paths, _ = _run_pipeline(
                model_source=model_source,
                requests=requests,
                output_dir=original_dir,
                use_cuda_graph=False,
                module_name=module_name,
                input_key_method=input_key_method,
                sample_ids=sample_ids)
            torch.cuda.empty_cache()


        except _AVAILABILITY_EXC as exc:
            pytest.skip(f"{model_source}: weights/metadata unavailable "
                        f"({type(exc).__name__}: {exc})")

        # The token transformer must have captured AND verified a graph for
        # every distinct target shape, with no eager fallback.
        states = _graph_states(cudagraph_proc)
        _assert_graph_verified_and_no_eager_fallback(states)
        
        if input_key_method == InputKeyMethod.EXACT:
            # Each target is a distinct token-transformer input shape, so EXACT
            # keying captures exactly one graph per target.
            assert len(states) == len(sample_ids), (
                f"expected {len(sample_ids)} captured graphs (one per target), "
                f"got {len(states)}")
            
            # --- Parity: cuda-graph predictions must match the original --------
            _assert_cudagraph_lddt_to_gt_matches_eager(
                model_source=model_source,
                original_paths=original_paths,
                cudagraph_paths=cudagraph_paths,
                input_key_method=input_key_method,
                sample_ids=sample_ids)

        elif input_key_method == InputKeyMethod.BUCKETED_SHAPES:
            # Bucketed keying pads targets into shared shape buckets, so several
            # targets can replay one captured graph: expect between one graph
            # (all targets in one bucket) and one-per-target.
            assert 1 <= len(states) <= len(sample_ids), (
                f"expected 1..{len(sample_ids)} captured graphs for "
                f"{input_key_method.name}, got {len(states)}")
            _assert_cudagraph_lddt_to_gt_matches_eager(
                model_source=model_source,
                original_paths=original_paths,
                cudagraph_paths=cudagraph_paths,
                input_key_method=input_key_method,
                sample_ids=sample_ids)
        else:
            raise Exception("not implemented")


@_cuda_graph_parity_marks
@pytest.mark.parametrize("sample_id_tuple", _SAMPLE_ID_TUPLE_PARAMS_ALL)
def test_cuda_graph_pairformer_parity(model_source, input_key_method,
                                      sample_id_tuple):
    _assert_cuda_graph_parity(model_source, "structure_pairformer",
                             sample_ids=sample_id_tuple,
                             input_key_method=input_key_method)


@_cuda_graph_parity_marks_exact_only
@pytest.mark.parametrize("sample_id_tuple", _SAMPLE_ID_TUPLE_PARAMS_MULTI)
def test_cuda_graph_token_transformer_parity(model_source, input_key_method,
                                             sample_id_tuple):
    _assert_cuda_graph_parity(model_source, "token_transformer",
                             sample_ids=sample_id_tuple,
                             input_key_method=input_key_method)


@_cuda_graph_parity_marks_exact_only
@pytest.mark.parametrize("sample_id_tuple", _SAMPLE_ID_TUPLE_PARAMS_MULTI)
def test_cuda_graph_diffusion_module_parity(model_source, input_key_method,
                                            sample_id_tuple):
    _assert_cuda_graph_parity(model_source, "diffusion_module",
                             sample_ids=sample_id_tuple,
                             input_key_method=input_key_method)



