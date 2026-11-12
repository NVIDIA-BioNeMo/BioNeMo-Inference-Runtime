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

import hashlib
from pathlib import Path

import pytest
import torch

from bionemo_ir.hubs import hf, integrity, local


@pytest.mark.parametrize("zipped", [True, False])
def test_protenix_checkpoint_maps_a_zip_and_reads_a_legacy_file(
    zipped: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    path = tmp_path / "protenix.pt"
    torch.save({"model": {"module.layer.weight": weight}}, path, _use_new_zipfile_serialization=zipped)
    mapped = []
    load = torch.load

    def spy(*args: object, **kwargs: object) -> object:
        mapped.append(kwargs["mmap"])
        return load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    state_dict = local._load_protenix_state_dict(str(path))

    assert mapped == [zipped]
    assert list(state_dict) == ["layer.weight"]
    assert torch.equal(state_dict["layer.weight"], weight)


def test_hub_checkpoint_lands_on_the_cpu_mapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hub checkpoint without a model-specific loader, such as OpenFold2's, maps onto the CPU."""
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    path = tmp_path / "finetuning_2.pt"
    torch.save({"layer.weight": weight}, path)
    monkeypatch.setitem(integrity.CHECKPOINT_SHA256, path.name, hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(hf, "hf_hub_download", lambda **_kwargs: str(path))
    calls = []
    load = torch.load

    def spy(*args: object, **kwargs: object) -> object:
        calls.append((kwargs["map_location"], kwargs["mmap"]))
        return load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    state_dict = hf.load_state_dict_from_hf(repo_id="nz/OpenFold", filename=path.name)

    assert calls == [("cpu", True)]
    assert torch.equal(state_dict["layer.weight"], weight)
