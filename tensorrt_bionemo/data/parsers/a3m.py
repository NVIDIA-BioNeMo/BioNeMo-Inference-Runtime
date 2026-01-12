import string
from io import StringIO
from pathlib import Path
from typing import Optional, TextIO

import numpy as np
import torch

from tensorrt_bionemo.data.parsers.fasta import parse_fasta_content


def _generate_deletion_matrix_gpu(sequences: list[str]) -> torch.Tensor:
    """
    Generate a deletion matrix for a list of sequences on GPU.
    """
    # TODO: Fully implement this function
    N = 0
    for seq in sequences[0]:
        if not seq.islower():
            N += 1
    encoded = [s.encode("ascii") for s in sequences]
    lengths = np.array([len(s) for s in encoded], dtype=np.int32)
    offsets = np.zeros(len(lengths), dtype=np.int32)
    offsets[1:] = np.cumsum(lengths[:-1])
    flat_buffer = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()

    torch.from_numpy(flat_buffer).cuda()
    offsets = torch.from_numpy(offsets).cuda()
    lengths = torch.from_numpy(lengths).cuda()

    # torch.ops.trtbnm.generate_deletion_matrix(flat_buffer, offsets, lengths, N)
    raise NotImplementedError("Not implemented")


def _generate_deletion_matrix_cpu(sequences: list[str]) -> torch.Tensor:
    deletion_matrix = []
    for msa_sequence in sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)

    ret = np.array(deletion_matrix)
    return ret


def generate_deletion_matrix(sequences: list[str],
                             gpu_preferred: bool = False) -> torch.Tensor:
    if gpu_preferred and torch.cuda.is_available():
        return _generate_deletion_matrix_gpu(sequences)
    else:
        return _generate_deletion_matrix_cpu(sequences)


class A3MParsed(dict):

    def __init__(self,
                 sequences: list[str],
                 raw: list[str],
                 descriptions: Optional[list[str]] = None):
        super().__init__(sequences=sequences,
                         raw=raw,
                         descriptions=descriptions)


def parse_a3m_content(content: StringIO | TextIO) -> A3MParsed:
    """
    Read an a3m file from a string or text stream and return a list of sequences.
    """

    sequences, descriptions = parse_fasta_content(content, return_as_list=True)
    deletion_table = str.maketrans("", "", string.ascii_lowercase)
    aligned_sequences = [s.translate(deletion_table) for s in sequences]
    ret = A3MParsed(sequences=aligned_sequences,
                    raw=sequences,
                    descriptions=descriptions)
    return ret


def read_a3m(file_path: str | Path) -> A3MParsed:
    with open(file_path, "r") as f:
        return parse_a3m_content(f)
