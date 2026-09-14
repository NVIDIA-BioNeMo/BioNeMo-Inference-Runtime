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

from check_math import _lint_math

PAGE = Path("docs/ref/page.md")


def test_math_fence_is_rejected() -> None:
    """Both backtick and tilde math fences are findings."""
    findings = _lint_math(
        PAGE,
        "before\n```math\n\\operatorname{predict}(x)\n```\n~~~math\nx_i\n~~~\n",
    )

    assert [f.message for f in findings] == [
        "math fence renders as a code block on the Fern site; use $$ ... $$ display math",
        "math fence renders as a code block on the Fern site; use $$ ... $$ display math",
    ]


def test_blocked_macros_are_rejected() -> None:
    """Macros GitHub refuses are findings, in source order."""
    findings = _lint_math(
        PAGE,
        "$$\n\\operatorname{predict}(x) + \\unicode{x41}\n$$\n",
    )

    assert [f.message for f in findings] == [
        r"macro is not allowed by GitHub's math renderer: \operatorname; use \mathrm{...} instead",
        r"macro is not allowed by GitHub's math renderer: \unicode",
    ]


def test_math_in_code_spans_and_fences_is_ignored() -> None:
    """Inline code and non-math fences never produce findings."""
    findings = _lint_math(
        PAGE,
        "Use `\\operatorname` only where GitHub allows it, e.g. `$$x$$` inline.\n"
        "```text\n"
        "\\operatorname{predict}(x)\n"
        "$$\n"
        "```\n",
    )

    assert findings == []


def test_display_math_is_checked() -> None:
    """A blank line inside and an unterminated $$ block are findings."""
    findings = _lint_math(
        PAGE,
        "$$\nx_i\n\n$$\n$$\ny_i\n",
    )

    assert [f.message for f in findings] == [
        "blank line inside the $$ block opened on line 1",
        "unclosed $$ display-math block",
    ]
