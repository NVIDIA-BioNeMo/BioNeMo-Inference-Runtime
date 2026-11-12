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
"""Initialize MKL's vector math library before a threaded CPU transcendental.

The first CPU VML call (``torch.atan``, ``log``, ``sqrt``, ``exp``, ``sin`` and
the like, in float32 or float64) on more than 99 elements, made while torch's
OpenMP team has more than one thread, reaches MKL's threaded VML path. That path
loads its function table lazily and without a lock: a team thread can see the
table half stored and call a NULL slot, which kills the process with SIGSEGV.
The table is shared by every VML function and precision, so one small call on
the calling thread loads it before any team runs, and the race cannot happen.
Elementwise VML results do not depend on the thread count, so outputs are
unchanged.
"""

import threading

import torch

_WARMUP_ELEMENTS = 1024  # above MKL's 99-element threshold for the threaded path
_lock = threading.Lock()
_warmed_threads: int | None = None


def ensure_vml_initialized() -> None:
    """Load MKL VML's function table on this thread, once per torch thread count.

    Call it before the first large CPU VML call. A call made while torch runs
    single-threaded stays on MKL's direct path and loads nothing, so a later
    ``torch.set_num_threads`` re-arms the warm-up. The warm-up tensor
    is CPU float32 whatever the default device and dtype, since a half, meta
    or CUDA tensor never reaches MKL VML.
    """
    global _warmed_threads
    threads = torch.get_num_threads()
    if _warmed_threads == threads:
        return
    with _lock:
        if _warmed_threads != threads:
            torch.atan(torch.zeros(_WARMUP_ELEMENTS, device="cpu", dtype=torch.float32))
            _warmed_threads = threads
