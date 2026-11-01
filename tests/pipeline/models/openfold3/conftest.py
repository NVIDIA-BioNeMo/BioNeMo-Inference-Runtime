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

from collections.abc import Iterator

import pytest

from bionemo_ir.pipeline.models.openfold3 import feature_context as fc


@pytest.fixture(autouse=True)
def empty_conformer_cache() -> Iterator[None]:
    """Tests patch the embedder, so no test may see another test's conformers."""
    fc._CONFORMERS.clear()
    yield
    fc._CONFORMERS.clear()
