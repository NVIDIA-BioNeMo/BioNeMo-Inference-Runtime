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
