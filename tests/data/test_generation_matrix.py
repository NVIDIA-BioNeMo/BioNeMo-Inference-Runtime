import numpy as np
import torch

import tensorrt_bionemo  # noqa: F401


def test_generation_matrix():
    sequences = ["ATGC", "ATGC", "ATGC"]
    N = 4

    encoded = [s.encode("ascii") for s in sequences]
    lengths = np.array([len(s) for s in encoded], dtype=np.int32)

    offsets = np.zeros(len(lengths), dtype=np.int32)
    offsets[1:] = np.cumsum(lengths[:-1])
    flat_buffer = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()

    flat_buffer = torch.from_numpy(flat_buffer).cuda()
    offsets = torch.from_numpy(offsets).cuda()
    lengths = torch.from_numpy(lengths).cuda()

    torch.ops.trtbnm.generate_deletion_matrix(flat_buffer, offsets, lengths, N)
