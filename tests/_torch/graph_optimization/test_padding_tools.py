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
"""Shape-bucket padding tools.

``GraphOptimizationTracker.pad_input`` grows the flagged input dims up to their
bucket length so one captured graph can serve a range of live shapes;
``unpad_output`` truncates the tied output dims back to the live length after the
(bucketed) forward. These tests exercise each directly: that ``pad_input`` pads
exactly the flagged dims — leaving feature/batch dims and non-tensor leaves
untouched and preserving the live values — and that ``unpad_output`` recovers the
live output from a bucket-padded one.

The input container here mirrors the call signature of
``OpenFold3DiffusionTransformer.forward(a, s, z, mask, ...)`` — the token
representation ``a``, the single conditioning ``s``, the pair representation
``z``, and the token ``mask`` — with the token (seq-len) axis of each tensor
tied to one named padded dimension. ``a`` and ``s`` are passed positionally and
``z``/``mask`` (plus a non-tensor ``buffers``) as keyword args, exercising both
the ``arg{i}`` and keyword walk paths. These are pure shape ops, so the test
needs no CUDA and does not instantiate the module.
"""

import pytest
import torch

from tensorrt_bionemo._torch.graph_optimization.config import (
    CUDAGraphOptimizationConfig,
    InputKeyMethod,
    InputRoutingConfigFactory,
    NamedDimTies,
)
from tensorrt_bionemo._torch.graph_optimization.cuda_graph.runtime import CUDAGraphOptimizationTracker

# Small stand-in feature dims for OpenFold3DiffusionTransformer inputs.
BS = 1
DIM = 8  # a: [BS, n_tokens, DIM]
DIM_SINGLE_COND = 6  # s: [BS, n_tokens, DIM_SINGLE_COND]
DIM_PAIRWISE = 4  # z: [BS, n_tokens, n_tokens, DIM_PAIRWISE]

# Bucket boundary lengths configured by _make_bucketer's padded dim.
_BUCKET_LENGTHS = (4, 7, 10, 13, 16)


def _bucket_len(n_tokens: int) -> int:
    """Lowest configured bucket length >= ``n_tokens``."""
    return min(b for b in _BUCKET_LENGTHS if b >= n_tokens)


def _make_bucketer() -> InputRoutingConfigFactory:
    """One named padded dim (lengths 4, 7, 10, 13, 16) tied to the token axis of
    every ``OpenFold3DiffusionTransformer.forward`` tensor input and of its
    output.

    ``a``/``s`` are passed positionally (walk paths ``arg0``/``arg1``) and
    ``z``/``mask`` by keyword (walk paths ``z``/``mask``); ``z`` carries the
    token dim on both axis 1 and axis 2. The forward returns the updated token
    representation ``a`` (output 0), whose token axis (1) is tied to the same
    padded dim so ``pad_output``/``unpad_output`` know which output dim rides it.
    """
    bucketer = InputRoutingConfigFactory()
    # a/s passed positionally (arg0/arg1), z/mask by keyword; z carries the token
    # dim on axes 1 and 2; output 0 (updated a) on axis 1.
    bucketer.set_named_dim_ties(
        [
            NamedDimTies(
                name="n_tokens",
                input_dims=(("arg0", (1,)), ("arg1", (1,)), ("z", (1, 2)), ("mask", (1,))),
                output_dims=((0, (1,)),),
            ),
        ]
    )
    # multiple_of=1 disables the default 128-alignment snap so the small bucket
    # lengths (4, 7, 10, 13, 16) this round-trip test relies on are preserved.
    bucketer.set_padded_dim("n_tokens", dim_len_min=4, dim_len_max=16, num_intervals=4, multiple_of=1)
    return bucketer


def _make_tracker(input_routing_config=None) -> CUDAGraphOptimizationTracker:
    # The config only accepts an omitted (defaulted) input_routing_config, not an
    # explicit None, so build the kwargs conditionally.
    cfg_kwargs = {"input_key_method": InputKeyMethod.BUCKETED_SHAPES}
    if input_routing_config is not None:
        cfg_kwargs["input_routing_config"] = input_routing_config
    cfg = CUDAGraphOptimizationConfig(**cfg_kwargs)
    return CUDAGraphOptimizationTracker(cfg, inner_module=None)


def _diffusion_transformer_inputs(n_tokens: int):
    """``(args, kwargs)`` for ``OpenFold3DiffusionTransformer.forward`` at the
    given live token count: ``a``, ``s`` positionally; ``z``, ``mask`` and a
    non-tensor ``buffers`` by keyword."""
    torch.manual_seed(n_tokens)
    a = torch.randn(BS, n_tokens, DIM)
    s = torch.randn(BS, n_tokens, DIM_SINGLE_COND)
    z = torch.randn(BS, n_tokens, n_tokens, DIM_PAIRWISE)
    mask = torch.ones(BS, n_tokens)
    args = (a, s)
    kwargs = {"z": z, "mask": mask, "buffers": None}
    return args, kwargs


def _diffusion_transformer_output(n_tokens: int):
    """The forward's return value: the updated token representation ``a`` of
    shape ``[BS, n_tokens, DIM]`` (output tensor 0)."""
    torch.manual_seed(1000 + n_tokens)
    return torch.randn(BS, n_tokens, DIM)


def _assert_container_identical(actual, expected, path="") -> None:
    """Recursively assert two containers are structurally and value-identical."""
    assert type(actual) is type(expected), f"type mismatch at {path!r}: {type(actual)} != {type(expected)}"
    if isinstance(expected, torch.Tensor):
        assert actual.shape == expected.shape, (
            f"shape mismatch at {path!r}: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )

        # assert bit-level equality
        assert torch.equal(actual, expected), f"value mismatch at {path!r}"
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"keys mismatch at {path!r}"
        for k in expected:
            _assert_container_identical(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), f"length mismatch at {path!r}"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_container_identical(a, e, f"{path}.{i}")
    else:
        assert actual == expected, f"leaf mismatch at {path!r}"


def test_pad_input_pads_flagged_token_axis():
    """pad_input grows every tensor's flagged token axis to the live length's
    bucket, zero-fills the padded region, and leaves feature/batch dims and the
    non-tensor leaf untouched."""
    tracker = _make_tracker(_make_bucketer().export_config())
    # Live token count 5 sits below the bucket boundary 7, so real padding happens.
    args, kwargs = _diffusion_transformer_inputs(n_tokens=5)
    shapes = tracker._extract_tensor_container_shapes(args, kwargs)

    padded_args, padded_kwargs = tracker.pad_input(args, kwargs, input_tensor_shapes=shapes)

    # The token axis of every flagged tensor is padded 5 -> 7; feature dims and
    # the batch dim are untouched.
    assert padded_args[0].shape == (BS, 7, DIM)  # a
    assert padded_args[1].shape == (BS, 7, DIM_SINGLE_COND)  # s
    assert padded_kwargs["z"].shape == (BS, 7, 7, DIM_PAIRWISE)
    assert padded_kwargs["mask"].shape == (BS, 7)
    # The non-tensor kwarg is passed through untouched.
    assert padded_kwargs["buffers"] is None
    # The live region preserves the input values and the padded region is zeros.
    assert torch.equal(padded_args[0][:, :5, :], args[0])
    assert torch.count_nonzero(padded_args[0][:, 5:, :]) == 0


@pytest.mark.parametrize("n_tokens", [4, 5, 8, 11, 16])
def test_pad_input_across_live_token_counts(n_tokens):
    """Each flagged token axis is padded to its bucket length — whether the live
    count lands exactly on a bucket boundary or between two — and the live region
    is preserved exactly (both token axes of ``z``)."""
    tracker = _make_tracker(_make_bucketer().export_config())
    args, kwargs = _diffusion_transformer_inputs(n_tokens)
    shapes = tracker._extract_tensor_container_shapes(args, kwargs)
    bucket = _bucket_len(n_tokens)

    padded_args, padded_kwargs = tracker.pad_input(args, kwargs, input_tensor_shapes=shapes)

    assert padded_args[0].shape == (BS, bucket, DIM)
    assert padded_args[1].shape == (BS, bucket, DIM_SINGLE_COND)
    assert padded_kwargs["z"].shape == (BS, bucket, bucket, DIM_PAIRWISE)
    assert padded_kwargs["mask"].shape == (BS, bucket)
    assert torch.equal(padded_args[0][:, :n_tokens, :], args[0])
    assert torch.equal(padded_kwargs["z"][:, :n_tokens, :n_tokens, :], kwargs["z"])


def test_pad_input_without_bucket_config_is_noop():
    """With no shape-bucket config pad_input returns the inputs untouched."""
    tracker = _make_tracker(input_routing_config=None)
    args, kwargs = _diffusion_transformer_inputs(n_tokens=5)
    shapes = tracker._extract_tensor_container_shapes(args, kwargs)

    padded_args, padded_kwargs = tracker.pad_input(args, kwargs, input_tensor_shapes=shapes)

    _assert_container_identical(padded_args, args, "padded_args")
    _assert_container_identical(padded_kwargs, kwargs, "padded_kwargs")


# ----------------------------------------------------------------------------
# Output side: ``unpad_output`` truncates each output tensor dim tied to a padded
# input dim back to the live input length (read from ``input_tensor_shapes``).
# The bucketer ties output 0's token axis (dim 1) to ``n_tokens`` — see
# _make_bucketer. A captured graph writes a fixed (bucketed) output buffer,
# mimicked here by zero-padding the live output up to its bucket length before
# unpadding.
# ----------------------------------------------------------------------------
def _pad_token_axis_to_bucket(output: torch.Tensor) -> torch.Tensor:
    """Zero-pad the token axis (dim 1) of ``output`` [BS, n_tokens, DIM] up to its
    bucket length — the fixed (bucketed) static-output-buffer shape that
    ``unpad_output`` must truncate back to the live length."""
    n_tokens = output.shape[1]
    return torch.nn.functional.pad(output, (0, 0, 0, _bucket_len(n_tokens) - n_tokens))


@pytest.mark.parametrize("n_tokens", [4, 5, 8, 11, 16])
def test_unpad_output_truncates_to_live_length(n_tokens):
    """A bucket-padded output is truncated back to the live token length,
    recovering the original tensor exactly. Covers exact-boundary counts (4, 16)
    and between-bucket counts (5, 8, 11)."""
    tracker = _make_tracker(_make_bucketer().export_config())
    args, kwargs = _diffusion_transformer_inputs(n_tokens)
    input_tensor_shapes = tracker._extract_tensor_container_shapes(args, kwargs)
    output = _diffusion_transformer_output(n_tokens)  # [BS, n_tokens, DIM]

    restored = tracker.unpad_output((_pad_token_axis_to_bucket(output),), input_tensor_shapes=input_tensor_shapes)

    assert restored.shape == (BS, n_tokens, DIM)
    _assert_container_identical(restored, output, "output")


def test_unpad_output_without_bucket_config_is_noop():
    """With no shape-bucket config ``unpad_output`` returns the output unchanged."""
    tracker = _make_tracker(input_routing_config=None)
    args, kwargs = _diffusion_transformer_inputs(n_tokens=5)
    input_tensor_shapes = tracker._extract_tensor_container_shapes(args, kwargs)
    output = _diffusion_transformer_output(n_tokens=5)

    restored = tracker.unpad_output((output,), input_tensor_shapes=input_tensor_shapes)

    _assert_container_identical(restored, output, "output")
