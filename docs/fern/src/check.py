#!/usr/bin/env python3
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

"""Validate canonical documentation and compose the generated Fern tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import check_doc_links
import check_fern_versions
import check_public_api
from common import DEFAULT_SITE_ROOT, REPO_ROOT, report


def _parser() -> argparse.ArgumentParser:
    """Build the unified documentation-check parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--site-root", type=Path, default=DEFAULT_SITE_ROOT)
    return parser


def main() -> int:
    """Run documentation checks and the requested composition command."""
    args = _parser().parse_args()
    source_root = args.source_root.expanduser().resolve()
    site_root = args.site_root.expanduser().resolve()
    try:
        findings = [
            *check_public_api.check(source_root),
            *check_doc_links.check(source_root / "docs", source_root / "docs" / "fern"),
        ]
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    status = report(findings)
    if status:
        return status

    try:
        check_fern_versions.sync_development(source_root, site_root)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
