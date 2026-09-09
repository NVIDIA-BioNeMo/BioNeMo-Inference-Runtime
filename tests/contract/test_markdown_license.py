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

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_HOOK_PATH = Path(__file__).resolve().parents[2] / "scripts" / "insert_markdown_license.py"


def _load_markdown_license() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bioir_test_insert_markdown_license", _HOOK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


markdown_license = _load_markdown_license()

_YAML_LICENSE = """---
# SPDX-License-Identifier: Apache-2.0
{}
---

# Existing document
"""
_HTML_LICENSE = """<!--
SPDX-License-Identifier: Apache-2.0
-->

# Existing template
"""
_FERN_FIELDS = """title: "BioNeMo Inference Runtime"
description:
  "GPU-accelerated PyTorch inference for biomolecular structure prediction
  models."
layout: overview
"""


@pytest.mark.parametrize("suffix", (".md", ".mdx"))
def test_new_markdown_gets_empty_mapping_frontmatter(tmp_path: Path, suffix: str) -> None:
    document = tmp_path / f"new{suffix}"
    document.write_text("# New document\n", encoding="utf-8")

    assert markdown_license.main([str(document)]) == 1

    rendered = document.read_text(encoding="utf-8")
    frontmatter, body = rendered.removeprefix("---\n").split("---\n", 1)
    lines = frontmatter.splitlines()
    assert "# SPDX-License-Identifier: Apache-2.0" in lines
    assert lines[-1] == "{}"
    assert all(line == "{}" or line.startswith("#") for line in lines)
    assert body == "\n# New document\n"
    assert markdown_license.main([str(document)]) == 0


@pytest.mark.parametrize("suffix", (".md", ".mdx"))
def test_comment_only_frontmatter_gets_empty_mapping(tmp_path: Path, suffix: str) -> None:
    document = tmp_path / f"comments_only{suffix}"
    document.write_text(
        "---\n# SPDX-License-Identifier: Apache-2.0\n---\n\n# Existing document\n",
        encoding="utf-8",
    )

    assert markdown_license.main([str(document)]) == 1
    assert document.read_text(encoding="utf-8") == (
        "---\n# SPDX-License-Identifier: Apache-2.0\n{}\n---\n\n# Existing document\n"
    )
    assert markdown_license.main([str(document)]) == 0


@pytest.mark.parametrize("suffix", (".md", ".mdx"))
def test_yaml_fields_without_license_keep_mapping(tmp_path: Path, suffix: str) -> None:
    fields = _FERN_FIELDS
    document = tmp_path / f"fields{suffix}"
    document.write_text(f"---\n{fields}---\n\n# Existing document\n", encoding="utf-8")

    assert markdown_license.main([str(document)]) == 1

    rendered = document.read_text(encoding="utf-8")
    frontmatter, body = rendered.removeprefix("---\n").split("---\n", 1)
    lines = frontmatter.splitlines()
    assert "# SPDX-License-Identifier: Apache-2.0" in lines
    assert "{}" not in lines
    for line in fields.splitlines():
        if line:
            assert line in lines
    assert lines.index("# SPDX-License-Identifier: Apache-2.0") < next(
        i for i, line in enumerate(lines) if line and not line.startswith("#")
    )
    assert body == "\n# Existing document\n"
    assert markdown_license.main([str(document)]) == 0


@pytest.mark.parametrize("suffix", (".md", ".mdx"))
def test_yaml_fields_with_license_are_unchanged(tmp_path: Path, suffix: str) -> None:
    fields = _FERN_FIELDS
    content = f"---\n# SPDX-License-Identifier: Apache-2.0\n{fields}---\n\n# Existing document\n"
    document = tmp_path / f"licensed_fields{suffix}"
    document.write_text(content, encoding="utf-8")

    assert markdown_license.main([str(document)]) == 0
    assert document.read_text(encoding="utf-8") == content
    assert "{}" not in content.split("---\n", 2)[1]


@pytest.mark.parametrize("suffix", (".md", ".mdx"))
@pytest.mark.parametrize("content", (_YAML_LICENSE, _HTML_LICENSE), ids=("yaml", "html"))
def test_existing_license_is_unchanged(tmp_path: Path, suffix: str, content: str) -> None:
    document = tmp_path / f"existing{suffix}"
    document.write_text(content, encoding="utf-8")

    assert markdown_license.main([str(document)]) == 0
    assert document.read_text(encoding="utf-8") == content


@pytest.mark.parametrize(
    ("opening", "error"),
    (("---", "Unclosed frontmatter"), ("<!--", "Unclosed comment")),
)
def test_unclosed_licensed_block_fails(tmp_path: Path, opening: str, error: str) -> None:
    document = tmp_path / "unclosed.md"
    document.write_text(f"{opening}\n# SPDX-License-Identifier: Apache-2.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        markdown_license.main([str(document)])


def test_new_license_uses_current_year(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    document = tmp_path / "new.md"
    document.write_text("# New document\n", encoding="utf-8")
    monkeypatch.setattr(markdown_license, "_current_year", lambda: 2030)

    assert markdown_license.main([str(document)]) == 1

    rendered = document.read_text(encoding="utf-8")
    assert "Copyright (c) 2030 NVIDIA CORPORATION" in rendered
    assert "Copyright (c) 2026 NVIDIA CORPORATION" not in rendered
