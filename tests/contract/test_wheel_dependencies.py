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

"""Keep the release dependency exception scoped to upstream metadata."""

import importlib.metadata
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / ".gitlab/ci/scripts/check_wheel_dependencies.py"
if not _SCRIPT.is_file():
    pytest.skip("GitLab release tools are unavailable in this checkout.", allow_module_level=True)
_CHECK = runpy.run_path(str(_SCRIPT))
_KNOWN_ERROR = "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform\n"


@pytest.mark.parametrize(
    ("machine", "system", "version", "tags", "status", "stdout", "stderr", "expected"),
    [
        ("aarch64", "linux", "0.8.1", "sbsa", 0, "No broken requirements found.\n", "", 0),
        ("aarch64", "linux", "0.8.1.post1", "aarch64", 0, "No broken requirements found.\n", "", 0),
        ("aarch64", "linux", "missing", "", 0, "No broken requirements found.\n", "", 0),
        ("aarch64", "linux", "0.8.1", "sbsa", 1, _KNOWN_ERROR, "", 0),
        ("x86_64", "linux", "0.8.1", "sbsa", 1, _KNOWN_ERROR, "", 1),
        ("aarch64", "darwin", "0.8.1", "sbsa", 1, _KNOWN_ERROR, "", 1),
        ("aarch64", "linux", "0.8.1.post1", "sbsa", 1, _KNOWN_ERROR, "", 1),
        ("aarch64", "linux", "0.8.0", "sbsa", 1, _KNOWN_ERROR, "", 1),
        ("aarch64", "linux", "0.8.1", "aarch64", 1, _KNOWN_ERROR, "", 1),
        ("aarch64", "linux", "0.8.1", "", 1, _KNOWN_ERROR, "", 1),
        ("aarch64", "linux", "0.8.1", "sbsa\nTag: py3-none-any", 1, _KNOWN_ERROR, "", 1),
        ("aarch64", "linux", "missing", "sbsa", 1, _KNOWN_ERROR, "", 1),
        ("aarch64", "linux", "0.8.1", "sbsa", 1, _KNOWN_ERROR + "foo requires bar, which is not installed.\n", "", 1),
        ("aarch64", "linux", "0.8.1", "sbsa", 1, "foo requires bar, which is not installed.\n", "", 1),
        ("aarch64", "linux", "0.8.1", "sbsa", 1, _KNOWN_ERROR, "pip failed\n", 1),
        ("aarch64", "linux", "0.8.1", "sbsa", 2, _KNOWN_ERROR, "", 2),
    ],
)
def test_dependency_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    machine: str,
    system: str,
    version: str,
    tags: str,
    status: int,
    stdout: str,
    stderr: str,
    expected: int,
) -> None:
    metadata = tmp_path / "nvidia_cusparselt_cu13.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Name: nvidia-cusparselt-cu13\nVersion: {version}\n")
    if tags:
        (metadata / "WHEEL").write_text(f"Wheel-Version: 1.0\nTag: py3-none-manylinux2014_{tags}\n")
    before = {path.name: path.read_bytes() for path in metadata.iterdir()}

    def find_distribution(name: str) -> importlib.metadata.Distribution:
        assert name == "nvidia-cusparselt-cu13"
        if version == "missing":
            raise importlib.metadata.PackageNotFoundError(name)
        return importlib.metadata.PathDistribution(metadata)

    def run_check(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert args == [sys.executable, "-m", "pip", "check"]
        assert kwargs == {"capture_output": True, "text": True, "check": False}
        return subprocess.CompletedProcess(args, status, stdout, stderr)

    monkeypatch.setattr(_CHECK["sys"], "platform", system)
    monkeypatch.setattr(_CHECK["platform"], "machine", lambda: machine)
    monkeypatch.setattr(_CHECK["importlib"].metadata, "distribution", find_distribution)
    monkeypatch.setattr(_CHECK["subprocess"], "run", run_check)
    assert _CHECK["main"]() == expected
    output = capsys.readouterr()
    assert output.out.startswith(stdout)
    assert output.err == stderr
    assert ("WARNING:" in output.out) == (status == 1 and expected == 0)
    assert {path.name: path.read_bytes() for path in metadata.iterdir()} == before
