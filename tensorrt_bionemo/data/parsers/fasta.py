from pathlib import Path

from Bio import SeqIO

from tensorrt_bionemo.data.schema.sequence import Sequence


def read_fasta_content(content: StringIO | TextIO) -> list[Sequence]:
    fasta_sequences = SeqIO.parse(content, "fasta")
    return [
        Sequence(sequence=str(fasta.seq), description=fasta.description)
        for fasta in fasta_sequences
    ]


def read_fasta(file_path: str | Path) -> list[Sequence]:
    with open(file_path) as source:
        return read_fasta_content(source)
