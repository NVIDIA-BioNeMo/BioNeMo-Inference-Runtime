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

import os

from mpi4py import MPI


def set_mpi_env():
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_world_size = comm.Get_size()
    # --- torch.distributed env ---
    os.environ["RANK"] = str(mpi_rank)
    os.environ["WORLD_SIZE"] = str(mpi_world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")

    return mpi_rank, mpi_world_size
