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

from __future__ import annotations

import importlib
import sys

import pytest
import torch

import tests as tests_package
from bionemo_ir._torch import attention_backend
from bionemo_ir._torch.attention_backend import pairwise_attention, triangle_attention
from bionemo_ir._torch.attention_backend import utils as attention_utils
from tests import require_public_cutedsl_library
from tests._torch import CUTEDSL_TEST_MODES_ENV, cutedsl_test_modes, make_left_aligned_pair_mask


@pytest.mark.parametrize(
    "module",
    [attention_backend, pairwise_attention, triangle_attention],
    ids=["root", "pairwise", "triangle"],
)
def test_attention_backend_exports_are_importable(module):
    for name in module.__all__:
        assert hasattr(module, name), f"{module.__name__} does not export {name}"


@pytest.mark.parametrize(
    "attention_type,backend_name,expected",
    [
        (attention_backend.AttentionType.TRIANGLE, "VANILLA", triangle_attention.VanillaTriangleAttention),
        (attention_backend.AttentionType.TRIANGLE, "SDPA", triangle_attention.SDPATriangleAttention),
        (attention_backend.AttentionType.TRIANGLE, "CUEQUIV", triangle_attention.CuEquivAttention),
        (attention_backend.AttentionType.TRIANGLE, "CuTeDSL", triangle_attention.TriangleAttentionCuTeLeftMask),
        (attention_backend.AttentionType.PAIRWISE, "VANILLA", pairwise_attention.VanillaPairwiseAttention),
        (attention_backend.AttentionType.PAIRWISE, "SDPA", pairwise_attention.SDPAPairwiseAttention),
        (attention_backend.AttentionType.PAIRWISE, "CuTeDSL", pairwise_attention.PairwiseAttentionCuTeLeftMask),
    ],
)
def test_attention_backend_dispatch(attention_type, backend_name, expected):
    assert attention_backend.get_attention_backend(backend_name, attention_type) is expected


def test_removed_trifast_backend_is_rejected():
    with pytest.raises(ValueError, match="TRIFAST"):
        attention_backend.get_attention_backend("TRIFAST", attention_backend.AttentionType.TRIANGLE)


def test_create_triangle_sdpa_backend():
    backend = attention_backend.create_attention(
        "SDPA",
        layer_idx=0,
        num_heads=4,
        head_dim=32,
        num_kv_heads=4,
        attention_type=attention_backend.AttentionType.TRIANGLE,
    )
    assert isinstance(backend, triangle_attention.SDPATriangleAttention)


def test_triangle_auto_select_falls_back_to_sdpa(monkeypatch):
    monkeypatch.setattr(attention_utils, "get_sm_version", lambda: 70)
    monkeypatch.setitem(sys.modules, "cuequivariance_ops_torch", None)

    assert attention_backend.auto_select_triangle_attention_backend(torch.float32) == "SDPA"


@pytest.mark.parametrize("backend_name", ["VANILLA", "SDPA", "CUEQUIV"])
def test_triangle_default_mask_precompute_registry(backend_name):
    pair_mask = make_left_aligned_pair_mask(1, 8, dtype=torch.float32, device="cpu")
    result = attention_backend.precompute_pair_masks(backend_name, pair_mask)

    assert result.mask_bias.shape == (1, 8, 1, 1, 8)
    assert result.mask_bias_transposed.shape == (1, 8, 1, 1, 8)


def test_cutedsl_test_modes_auto_detects_private_source(monkeypatch):
    monkeypatch.delenv(CUTEDSL_TEST_MODES_ENV, raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())

    assert cutedsl_test_modes("private.cutedsl.source") == ("source",)


def test_cutedsl_test_modes_auto_detects_public_build(monkeypatch):
    monkeypatch.delenv(CUTEDSL_TEST_MODES_ENV, raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    assert cutedsl_test_modes("private.cutedsl.source") == ("cubin",)


def test_public_source_accepts_built_cutedsl_library(monkeypatch, tmp_path):
    monkeypatch.setattr(tests_package, "_PRIVATE_CUTEDSL_SOURCE_DIR", tmp_path / "cute")
    monkeypatch.setattr(importlib, "import_module", lambda _name: object())

    require_public_cutedsl_library()


def test_public_source_with_only_package_init_requires_built_cutedsl_library(monkeypatch, tmp_path):
    source_dir = tmp_path / "cute"
    source_dir.mkdir()
    (source_dir / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(tests_package, "_PRIVATE_CUTEDSL_SOURCE_DIR", source_dir)

    def missing_library(name: str) -> None:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing_library)
    with pytest.warns(RuntimeWarning, match="Build the shared library first"):
        with pytest.raises(pytest.UsageError, match="_cutedsl_kernels"):
            require_public_cutedsl_library()


def test_private_source_does_not_require_built_cutedsl_library(monkeypatch, tmp_path):
    source_dir = tmp_path / "cute"
    source_dir.mkdir()
    (source_dir / "kernel.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(tests_package, "_PRIVATE_CUTEDSL_SOURCE_DIR", source_dir)

    def unexpected_import(name: str) -> None:
        raise AssertionError(f"private source unexpectedly imported {name}")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)
    require_public_cutedsl_library()


def test_cutedsl_test_modes_parses_explicit_modes(monkeypatch):
    monkeypatch.setenv(CUTEDSL_TEST_MODES_ENV, " CUBIN,source,cubin ")

    assert cutedsl_test_modes() == ("cubin", "source")


@pytest.mark.parametrize("value", ["", "unknown", "source,invalid"])
def test_cutedsl_test_modes_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv(CUTEDSL_TEST_MODES_ENV, value)

    with pytest.raises(pytest.UsageError, match=CUTEDSL_TEST_MODES_ENV):
        cutedsl_test_modes()
