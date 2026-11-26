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

"""Importing one model must not import another model's code.

Each check runs in a fresh interpreter, so modules the test session already
imported do not hide a regression.
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import bionemo_ir

MODELS = ("boltz1", "boltz2", "openfold2", "openfold3", "protenix", "rfproteina")
_PACKAGE_PARENT = Path(bionemo_ir.__file__).resolve().parent.parent

# Boltz-1 reuses Boltz-2's MSA config, denoising graph policy and token vocabulary, and Boltz-2 converts its
# Pairformer and diffusion transformer with Boltz-1's checkpoint mapping. The pair is one model family.
_MODEL_ALLOWED = {
    "boltz1": {
        "bionemo_ir.models.boltz2",
        "bionemo_ir.models.boltz2.config",
        "bionemo_ir.pipeline.models.boltz2",
        "bionemo_ir.pipeline.models.boltz2.const",
    },
    "boltz2": {"bionemo_ir.models.boltz1", "bionemo_ir.models.boltz1.convert"},
}

# Pipelines another model's pipeline is built on. Boltz-1's data pipeline is Boltz-2's; Protenix shares Boltz-2's
# atom vocabulary and OpenFold3's residue vocabulary and template selection.
_PIPELINE_ALLOWED = {
    "boltz1": {
        f"bionemo_ir.pipeline.models.boltz2.{name}"
        for name in (
            "feature_collators",
            "feature_context",
            "feature_factory",
            "feature_generators",
            "featurizer",
            "postprocessor",
            "structure",
            "template_logic",
            "tokenizer_logic",
        )
    },
    "protenix": {
        "bionemo_ir.pipeline.models.boltz2",
        "bionemo_ir.pipeline.models.boltz2.const",
        "bionemo_ir.pipeline.models.openfold3",
        "bionemo_ir.pipeline.models.openfold3.common",
        "bionemo_ir.pipeline.models.openfold3.const",
        "bionemo_ir.pipeline.models.openfold3.template_logic",
    },
}

_PROBE = """
import json, re, sys
own, models = sys.argv[1], set(json.loads(sys.argv[2]))
exec(sys.argv[3])
pattern = re.compile(r"^bionemo_ir\\.(?:models|_torch\\.modules|pipeline\\.models)\\.(\\w+)")
# The shared Boltz network modules belong to both Boltz models.
owners = {"boltz": {"boltz1", "boltz2"}}
foreign = sorted(
    name for name in sys.modules
    if (match := pattern.match(name))
    and (owner := owners.get(match.group(1), {match.group(1)}) & models)
    and own not in owner
)
print(json.dumps(foreign))
"""


def _foreign_modules(own: str, code: str) -> set[str]:
    """Run ``code`` in a fresh CPU-only interpreter; return the other models' modules it imported."""
    # Resolve the same bionemo_ir this session imported, whether a source tree or an installed wheel.
    python_path = os.pathsep.join(filter(None, [str(_PACKAGE_PARENT), os.environ.get("PYTHONPATH")]))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": python_path}
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, own, json.dumps(MODELS), code],
        capture_output=True,
        check=True,
        env=env,
        text=True,
    )
    return set(json.loads(result.stdout.strip().splitlines()[-1]))


def _unexpected(model: str, foreign: set[str]) -> list[str]:
    return sorted(foreign - _MODEL_ALLOWED.get(model, set()))


@pytest.mark.parametrize(
    ("model", "attr"),
    [
        ("boltz1", "Boltz1"),
        ("boltz2", "Boltz2"),
        ("openfold2", "OpenFold2"),
        ("openfold3", "OpenFold3"),
        ("protenix", "Protenix"),
        ("rfproteina", "RFProteina"),
    ],
)
def test_model_class_imports_only_its_model(model: str, attr: str) -> None:
    """``from bionemo_ir.models.<model> import <Class>`` loads no other model's modules."""
    foreign = _foreign_modules(model, f"from bionemo_ir.models.{model} import {attr}")

    assert _unexpected(model, foreign) == []


def test_models_package_imports_no_model() -> None:
    """``import bionemo_ir.models`` defers every model until its class is accessed."""
    assert _foreign_modules("", "import bionemo_ir.models") == set()


@pytest.mark.parametrize(
    ("model", "factory"),
    [
        ("boltz1", "Boltz1Factory"),
        ("boltz2", "Boltz2Factory"),
        ("openfold2", "OpenFold2Factory"),
        ("openfold2", "OpenFold2MultimerFactory"),
        ("openfold3", "OpenFold3Factory"),
        ("protenix", "ProtenixFactory"),
    ],
)
def test_registry_factory_imports_only_its_model(model: str, factory: str) -> None:
    """The registry calls that assemble a pipeline load no other model's modules but shared pipeline helpers."""
    code = (
        f"from bionemo_ir.registry import {factory} as f\n"
        "f.get_model_class(); f.get_tokenizer(); f.get_feature_factory(); f.get_postprocessor()"
    )
    foreign = _foreign_modules(model, code) - _PIPELINE_ALLOWED.get(model, set())

    assert _unexpected(model, foreign) == []


@pytest.mark.parametrize("model", ["", *MODELS])
def test_probe_assigns_shared_boltz_modules_to_both_boltz_models(model: str) -> None:
    """``bionemo_ir._torch.modules.boltz`` is foreign to every model but Boltz-1 and Boltz-2."""
    name = "bionemo_ir._torch.modules.boltz.trunk"
    code = f"import sys, types; sys.modules[{name!r}] = types.ModuleType({name!r})"

    expected = [] if model in ("boltz1", "boltz2") else [name]
    assert _unexpected(model, _foreign_modules(model, code)) == expected


@pytest.mark.parametrize("package", ["bionemo_ir.models", *(f"bionemo_ir.models.{model}" for model in MODELS)])
def test_public_names_resolve(package: str) -> None:
    """Every name in a model package's ``__all__`` still resolves after the lazy split."""
    module = importlib.import_module(package)

    for name in module.__all__:
        assert getattr(module, name) is not None, name
