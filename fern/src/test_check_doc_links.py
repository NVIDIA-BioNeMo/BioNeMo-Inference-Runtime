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

from check_doc_links import _navigation_routes


def test_navigation_routes_tracks_section_scope(tmp_path: Path) -> None:
    index_yml = tmp_path / "index.yml"
    index_yml.write_text(
        """navigation:
  - section: Learn
    contents:
      - section: Key Concepts
        skip-slug: true
        contents:
          - page: Embeddings
            path: embeddings.md
      - section: Generation
        contents:
          - page: Likelihood
            path: likelihood.md
      - page: Summary
        path: summary.md
  - page: Home
    path: home.md
""",
        encoding="utf-8",
    )

    routes = {path.name: route for path, route in _navigation_routes(index_yml).items()}

    assert routes == {
        "embeddings.md": "learn/embeddings",
        "home.md": "home",
        "likelihood.md": "learn/generation/likelihood",
        "summary.md": "learn/summary",
    }


def test_navigation_routes_restores_section_for_trailing_property(tmp_path: Path) -> None:
    index_yml = tmp_path / "index.yml"
    index_yml.write_text(
        """navigation:
  - section: Learn
    contents:
      - section: Concepts
        contents:
          - page: Embeddings
            path: embeddings.md
    slug: guides
""",
        encoding="utf-8",
    )

    routes = {path.name: route for path, route in _navigation_routes(index_yml).items()}

    assert routes == {"embeddings.md": "guides/concepts/embeddings"}
