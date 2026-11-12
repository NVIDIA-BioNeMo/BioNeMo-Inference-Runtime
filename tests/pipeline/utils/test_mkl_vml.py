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
"""MKL VML warm-up: featurization's first large CPU VML call must survive a fresh threaded process."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch

from bionemo_ir.pipeline.utils import _mkl_vml

_DELETION_VALUE = textwrap.dedent(
    """
    import sys
    import threading
    import torch
    from bionemo_ir.pipeline.models.openfold3.common import compute_deletion_value

    default = sys.argv[1]
    torch.set_num_threads(16)
    if default == "half":
        torch.set_default_dtype(torch.float16)
    deletion = torch.randint(0, 50, (4096, 1734), dtype=torch.int32, device="cpu")
    out = []

    def run():
        if default in ("meta", "cuda"):
            torch.set_default_device(default)
        out.append(compute_deletion_value(deletion))

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    assert out and out[0].device.type == "cpu" and torch.isfinite(out[0]).all()
    """
)

_FEATURE_STAGE = textwrap.dedent(
    """
    import asyncio
    import threading
    from types import SimpleNamespace

    import torch
    from bionemo_ir.pipeline.base import default_context_and_feature_merger
    from bionemo_ir.pipeline.models.openfold2.feature_collators import MultimerMakeMaskedMsa
    from bionemo_ir.pipeline.stages.feature_generator_stage import FeatureGeneratorUDF

    torch.set_num_threads(16)
    n_msa, n_res = 2048, 400
    row = {
        "msa": torch.randint(0, 21, (n_msa, n_res)),
        "msa_profile": torch.softmax(torch.rand(n_res, 22), -1),
        "msa_mask": torch.ones(n_msa, n_res),
    }
    udf = FeatureGeneratorUDF(
        compute_by_rows=True,
        drop_keys=None,
        expected_input_keys=[],
        update_row=False,
        feature_generators=[],
        features_merger_func=default_context_and_feature_merger,
        feature_collators=[MultimerMakeMaskedMsa(SimpleNamespace(resample_msa_in_recycling=True))],
    )
    out = []
    worker = threading.Thread(target=lambda: out.append(asyncio.run(udf.udf_for_item(row))))
    worker.start()
    worker.join()
    assert out and "bert_mask" in out[0]
    """
)


def _assert_survives(script: str, *args: str, attempts: int = 8) -> None:
    for attempt in range(attempts):
        result = subprocess.run(
            [sys.executable, "-c", script, *args], capture_output=True, text=True, timeout=300, check=False
        )
        assert result.returncode == 0, f"attempt {attempt}: rc={result.returncode}\n{result.stderr[-800:]}"


def test_ensure_vml_initialized_runs_once_per_thread_count(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real_atan = torch.atan
    monkeypatch.setattr(torch, "atan", lambda x: calls.append(x.numel()) or real_atan(x))
    monkeypatch.setattr(_mkl_vml, "_warmed_threads", None)
    original = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        _mkl_vml.ensure_vml_initialized()
        _mkl_vml.ensure_vml_initialized()
        assert calls == [1024]
        torch.set_num_threads(3)
        _mkl_vml.ensure_vml_initialized()
        assert calls == [1024, 1024]
    finally:
        torch.set_num_threads(original)


@pytest.mark.skipif(not torch.backends.mkl.is_available(), reason="MKL VML race needs an MKL build of torch")
@pytest.mark.parametrize(
    "default",
    [
        "none",
        "half",
        "meta",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")),
    ],
)
def test_deletion_value_in_fresh_threaded_process(default: str) -> None:
    """Without a CPU float32 warm-up, about half of fresh 16-thread processes died with SIGSEGV at this shape.

    The default dtype and device must not move the warm-up off MKL VML's CPU float32 path.
    """
    _assert_survives(_DELETION_VALUE, default)


@pytest.mark.skipif(not torch.backends.mkl.is_available(), reason="MKL VML race needs an MKL build of torch")
def test_feature_stage_warms_vml_before_collators() -> None:
    """OpenFold2's masked-MSA collator calls ``torch.log`` before any atan; the stage must warm VML first.

    Without the stage warm-up, about 30% of fresh 16-thread processes died with SIGSEGV at this shape.
    """
    _assert_survives(_FEATURE_STAGE, attempts=16)
