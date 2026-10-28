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

from pathlib import Path

import pytest

from tests.conftest import _is_launcher_free_only

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        (["tests/contract", "tests/cubin"], True),
        (
            [
                "tests/contract",
                "tests/cubin",
                ".gitlab/ci/scripts/test_publish_release_wheel.py",
                ".gitlab/ci/scripts/test_source_builds.py",
            ],
            True,
        ),
        ([".gitlab/ci/scripts/test_source_builds.py::test_example"], True),
        (["tests/contract", "tests/_torch"], False),
        (["tests"], False),
        ([], False),
    ],
)
def test_launcher_requirement(
    targets: list[str], expected: bool, pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(_ROOT)
    monkeypatch.setattr(pytestconfig, "args", targets)
    assert _is_launcher_free_only(pytestconfig) is expected
